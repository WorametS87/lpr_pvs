import traceback, os, json, argparse
from pathlib import Path
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

from read_plate_main3 import read_plate_from_crop

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def iter_images(input_path: Path):
    if input_path.is_file():
        yield input_path
        return
    for p in sorted(input_path.rglob("*")):
        if p.suffix.lower() in IMG_EXTS:
            yield p
class PlateImage:
    def __init__(self, image):
        self.image = image
        self.prov_box_xyxy = None

    def __getattr__(self, name):
        return getattr(self.image, name)

    def __getitem__(self, key):
        return self.image[key]

    def __setitem__(self, key, value):
        self.image[key] = value

    def copy(self):
        return PlateImage(self.image.copy())

def pad_xyxy(x1, y1, x2, y2, W, H, pad=0.12):
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    dx = bw * pad
    dy = bh * pad
    x1p = int(max(0, x1 - dx))
    y1p = int(max(0, y1 - dy))
    x2p = int(min(W, x2 + dx))
    y2p = int(min(H, y2 + dy))
    return x1p, y1p, x2p, y2p

def accept_plate_box(x1, y1, x2, y2, W, H, conf: float):
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    ar = bw / bh
    area = bw * bh
    area_frac = area / float(W * H + 1e-9)
    min_w = max(55, int(W * 0.02))
    min_h = max(22, int(H * 0.015))
    return not (
        bw < min_w or bh < min_h or
        ar < 1.2 or ar > 8 or
        area_frac < 0.00010 or area_frac > 0.14
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="4thGen_out")
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--stage2", default=None)
    ap.add_argument("--run-stage2", action="store_true")
    ap.add_argument("--imgsz1", type=int, default=1280)
    ap.add_argument("--conf1", type=float, default=0.25)
    ap.add_argument("--iou1", type=float, default=0.50)
    ap.add_argument("--maxdet1", type=int, default=80)
    ap.add_argument("--pad", type=float, default=0.12)
    ap.add_argument("--debug-dir", default=None)
    args = ap.parse_args()

    if YOLO is None:
        raise RuntimeError("ultralytics not available. Install with: pip install ultralytics")

    input_path = Path(args.input)
    out_dir = Path(args.out)
    crops_dir = out_dir / "crops"
    overlays_dir = out_dir / "overlays"
    crops_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    stage1 = YOLO(args.stage1)
    stage2 = YOLO(args.stage2) if (args.run_stage2 and args.stage2) else None

    results = []
    
    for img_path in iter_images(input_path):
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        H, W = bgr.shape[:2]

        r = stage1.predict(
            source=bgr,
            imgsz=args.imgsz1,
            conf=args.conf1,
            iou=args.iou1,
            max_det=args.maxdet1,
            verbose=False,
            device=""
        )[0]

        boxes = getattr(r, "boxes", None)
        img_name = img_path.stem
        this_crop_dir = crops_dir / img_name
        this_crop_dir.mkdir(parents=True, exist_ok=True)

        overlay = bgr.copy()
        plates = []

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()

            for i in range(len(xyxy)):
                x1, y1, x2, y2 = map(float, xyxy[i])
                c = float(confs[i])
                xi1, yi1, xi2, yi2 = pad_xyxy(x1, y1, x2, y2, W, H, pad=args.pad)

                if not accept_plate_box(xi1, yi1, xi2, yi2, W, H, c):
                    continue

                cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(overlay, f"{c:.3f}", (int(x1), max(0, int(y1) - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                crop = bgr[yi1:yi2, xi1:xi2]
                crop_path = this_crop_dir / f"plate_{i+1:02d}_conf_{c:.3f}.jpg"
                cv2.imwrite(str(crop_path), crop)

                item = {
                    "plate_index": int(len(plates) + 1),
                    "plate_conf": c,
                    "plate_box_xyxy": [int(xi1), int(yi1), int(xi2), int(yi2)],
                    "plate_crop_path": str(crop_path),
                }

                try:
                    debug_mode = "save"
                    debug_dir = args.debug_dir or str(out_dir / "debug2")
                    image_id = crop_path.stem
                    debug_crop_dir = os.path.join(debug_dir, img_name, f"plate_{item['plate_index']:02d}")

                    plate_crop_bgr = PlateImage(crop)
                    prov_crop_bgr = None

                    if stage2:
                        r2 = stage2.predict(plate_crop_bgr.image, imgsz=256, conf=0.25, iou=0.5, max_det=10, verbose=False)[0]
                        boxes2 = getattr(r2, "boxes", None)

                        if boxes2 is not None and len(boxes2) > 0:
                            cls2 = boxes2.cls.cpu().numpy().astype(int)
                            xyxy2 = boxes2.xyxy.cpu().numpy()
                            conf2 = boxes2.conf.cpu().numpy()

                            # letters = cls==0, province = cls==1
                            prov_boxes = [xyxy2[i] for i in range(len(cls2)) if cls2[i] == 1]
                            if prov_boxes:
                                x1p, y1p, x2p, y2p = map(int, prov_boxes[0])
                                prov_crop_bgr = plate_crop_bgr[y1p:y2p, x1p:x2p]

                                # 👇️ patch the crop with province box as attribute
                                plate_crop_bgr = plate_crop_bgr.copy()
                                plate_crop_bgr.prov_box_xyxy = (x1p, y1p, x2p, y2p)

                    
                    
                    print("🔍 DEBUG", type(plate_crop_bgr), plate_crop_bgr.shape if plate_crop_bgr is not None else "None")
                    
                    prov_crop_bgr = prov_crop_bgr if 'prov_crop_bgr' in locals() else None


                    plate_info = read_plate_from_crop(
                        plate_crop_bgr=plate_crop_bgr,
                        prov_bgr=prov_crop_bgr,
                        debug_dir=debug_crop_dir, 
                        debug_mode="save"
                    )

                    item.update(plate_info or {})
                    item["stage2_ok"] = True
                    item["stage2_reason"] = "ok" if stage2 else "fallback_full_ocr"

                except Exception as e:
                    loc = traceback.extract_tb(e.__traceback__)[-1]
                    item.update({
                        "stage2_ok": False,
                        "stage2_reason": f"exception:{type(e).__name__} at {Path(loc.filename).name}:{loc.lineno} in {loc.name}",
                        "raw": "",
                        "cleaned": "",
                        "line1": "",
                        "line2": "",
                        "province": "",
                        "province_score": 0.0,
                    })

                plates.append(item)

        overlay_path = overlays_dir / f"{img_name}_overlay.jpg"
        cv2.imwrite(str(overlay_path), overlay)

        results.append({
            "image": str(img_path),
            "H": H,
            "W": W,
            "overlay": str(overlay_path),
            "plates": plates,
        })

    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
