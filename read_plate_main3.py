import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from util3 import (
    _ocr_with_conf,
    _try_warp_plate_border,
    ALLOW_THAI_FULL,
    clean_province_text,
    match_province,
    _basic_clean,
    read_plate_text,
)

# ------------------------------------------------------------
# Province scoring weights
# ------------------------------------------------------------
# Combined as: score = fuzzy*0.7 + conf*0.5
FUZZY_WEIGHT = 0.7
CONF_WEIGHT  = 0.5


# ------------------------------------------------------------
# Utility: Safe score for a province candidate
# ------------------------------------------------------------
def _province_candidate_score(fuzzy_score, conf):
    return FUZZY_WEIGHT * float(fuzzy_score) + CONF_WEIGHT * float(conf)


# ------------------------------------------------------------
# Main API
# ------------------------------------------------------------
def read_plate_from_crop(
    plate_crop_bgr,
    prov_bgr=None,
    debug_dir=None,
    debug_mode="save",
    prov_box_xyxy=None,
):
    """
    The main Stage2 logic:
    - Extract plate text using read_plate_text() from util3.py
    - Build multiple province OCR candidates:
        1) YOLO province crop
        2) Warped-bottom strip
        3) Raw-bottom strip
        4) Full-plate bottom 25%
        5) EasyOCR fallback
        6) Tail-text fallback (only if score improves)
    - Select best province based on combined score:
          score = 0.7*fuzzy + 0.5*ocr_conf
    """
    # --------------------------------------------------------
    # Setup base result package
    # --------------------------------------------------------
    result = {
        "stage2_ok": False,
        "stage2_reason": "",
        "raw": "",
        "cleaned": "",
        "line1": "",
        "line2": "",
        "province": "",
        "province_score": 0.0,
        "prov_candidate_conf": 0.0,
        "prov_candidate_tag": "",
        "prov_candidate_text": "",
    }

    # --------------------------------------------------------
    # Fail-fast for missing crop
    # --------------------------------------------------------
    if plate_crop_bgr is None or getattr(plate_crop_bgr, "image", None) is None:
        result["stage2_reason"] = "empty_crop"
        return result

    img = plate_crop_bgr.image

    # --------------------------------------------------------
    # Run plate OCR (letters + digits)
    # --------------------------------------------------------
    try:
        plate_info = read_plate_text(
            img,
            debug_dir=debug_dir,
            debug_id="plate_ocr",
            debug_mode=debug_mode,
        )
        result.update(plate_info)
    except Exception as e:
        result["stage2_reason"] = f"plate_ocr_exception:{type(e).__name__}"
        return result

    base_province = result.get("province", "") or ""
    base_score    = float(result.get("province_score", 0.0) or 0.0)

    # Collect province candidates here
    candidates = []

    # --------------------------------------------------------
    # Helper: Add a candidate entry
    # --------------------------------------------------------
    def add_candidate(tag, text, fuzzy_score, conf):
        clean_txt = clean_province_text(_basic_clean(text))
        province, fscore = match_province(clean_txt)

        combined = _province_candidate_score(fscore, conf)
        candidates.append({
            "tag": tag,
            "province": province,
            "prov_text": clean_txt,
            "fuzzy": float(fscore),
            "conf": float(conf),
            "combined": float(combined),
        })


    # --------------------------------------------------------
    # Candidate #1 — YOLO province crop (most reliable)
    # --------------------------------------------------------
    if prov_bgr is not None and getattr(prov_bgr, "size", 0) != 0:
        try:
            # Resize for OCR
            resized = cv2.resize(prov_bgr, (160, 32), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

            # multiple preproc variants for YOLO crop
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
            adapt = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                15, 10
            )

            variants = [
                ("yolo_gray",  gray),
                ("yolo_clahe", clahe),
                ("yolo_adapt", adapt),
            ]

            for tag, imgvar in variants:
                txt, conf = _ocr_with_conf(imgvar, ALLOW_THAI_FULL, beam=True)
                add_candidate(tag, txt, fuzzy_score=0.0, conf=conf)
        except Exception:
            pass


    # --------------------------------------------------------
    # Candidate #2 — YOLO province box assignment (if provided from infer_v4)
    # --------------------------------------------------------
    if hasattr(plate_crop_bgr, "prov_box_xyxy") and plate_crop_bgr.prov_box_xyxy:
        try:
            x1, y1, x2, y2 = plate_crop_bgr.prov_box_xyxy
            crop = img[y1:y2, x1:x2]
            if crop.size > 0:
                resized = cv2.resize(crop, (160, 32), interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

                txt, conf = _ocr_with_conf(gray, ALLOW_THAI_FULL, beam=True)
                add_candidate("yolo_box_crop", txt, fuzzy_score=0.0, conf=conf)
        except Exception:
            pass


    # --------------------------------------------------------
    # Candidate #3 — Warped bottom strip
    # --------------------------------------------------------
    try:
        warped, _, _ = _try_warp_plate_border(img, debug_dir=debug_dir)
        if warped is not None:
            h, w = warped.shape[:2]
            strip = warped[int(h * 0.75):, :]
            if strip.shape[0] >= 20 and strip.shape[1] >= 60:
                resized = cv2.resize(strip, (160, 32), interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

                txt, conf = _ocr_with_conf(gray, ALLOW_THAI_FULL, beam=True)
                add_candidate("warp_bottom", txt, fuzzy_score=0.0, conf=conf)
    except Exception:
        pass


    # --------------------------------------------------------
    # Candidate #4 — Raw plate bottom strip
    # --------------------------------------------------------
    try:
        H, W = img.shape[:2]
        raw_strip = img[int(H * 0.75):, :]
        if raw_strip.shape[0] >= 20 and raw_strip.shape[1] >= 60:
            resized = cv2.resize(raw_strip, (160, 32), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

            txt, conf = _ocr_with_conf(gray, ALLOW_THAI_FULL, beam=True)
            add_candidate("raw_bottom", txt, fuzzy_score=0.0, conf=conf)
    except Exception:
        pass


    # --------------------------------------------------------
    # Candidate #5 — Full-plate bottom 25% (helps with blur)
    # --------------------------------------------------------
    try:
        H, W = img.shape[:2]
        full_strip = img[int(H * 0.70):, :]
        if full_strip.shape[0] >= 20 and full_strip.shape[1] >= 80:
            resized = cv2.resize(full_strip, (220, 40), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

            txt, conf = _ocr_with_conf(gray, ALLOW_THAI_FULL, beam=True)
            add_candidate("full_bottom", txt, fuzzy_score=0.0, conf=conf)
    except Exception:
        pass
    # --------------------------------------------------------
    # Candidate #6 — EasyOCR direct fallback (on warped or resized strip)
    # --------------------------------------------------------
    try:
        # Prefer warped region for fallback
        if warped is not None:
            H2, W2 = warped.shape[:2]
            fb = warped[int(H2 * 0.75):, :]
        else:
            fb = raw_strip

        if fb is not None and fb.size > 0:
            resized = cv2.resize(fb, (200, 42), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

            # EasyOCR raw fallback
            fallback = READER.readtext(gray, detail=1, paragraph=False)
            if fallback:
                txt = fallback[0][1]
                conf = float(fallback[0][2])
                add_candidate("easy_fallback", txt, fuzzy_score=0.0, conf=conf)
    except Exception:
        pass


    # --------------------------------------------------------
    # Candidate #7 — Tail text fallback (from plate OCR)
    # Only used if it improves score (never overwrite good province)
    # --------------------------------------------------------
    try:
        tail = result.get("raw", "")[-30:]
        cleaned_tail = clean_province_text(_basic_clean(tail))
        province_tail, fscore_tail = match_province(cleaned_tail)

        if province_tail:
            add_candidate("tail", cleaned_tail, fuzzy_score=fscore_tail, conf=0.0)
    except Exception:
        pass


    # --------------------------------------------------------
    # SELECT BEST CANDIDATE
    # --------------------------------------------------------
    # Base plate OCR may already have a province detected
    if base_province:
        base_combined = _province_candidate_score(base_score, 0.0)
        candidates.append({
            "tag": "base_plate_ocr",
            "province": base_province,
            "prov_text": base_province,
            "fuzzy": float(base_score),
            "conf": 0.0,
            "combined": float(base_combined),
        })

    # If no candidates at all → province empty
    if not candidates:
        result["province"] = base_province
        result["province_score"] = base_score
        result["stage2_ok"] = True
        result["stage2_reason"] = "ok"
        return result

    # Pick candidate with highest combined score
    best = max(candidates, key=lambda x: x["combined"])

    # Accept only if score exceeds 0.25 (safe noise floor)
    if best["combined"] < 0.25:
        # Keep original plate's fallback province if exists
        result["province"] = base_province
        result["province_score"] = base_score
        result["prov_candidate_tag"] = "low_conf_reject"
        result["prov_candidate_text"] = ""
        result["prov_candidate_conf"] = 0.0
        result["stage2_ok"] = True
        result["stage2_reason"] = "ok"
        return result

    # Strong enough → accept best candidate
    result["province"] = best["province"]
    result["province_score"] = float(best["fuzzy"])
    result["prov_candidate_text"] = best["prov_text"]
    result["prov_candidate_conf"] = float(best["conf"])
    result["prov_candidate_tag"] = best["tag"]

        # --------------------------------------------------------
    # SPECIAL OVERRIDE: กรุงเทพมหานคร (Bangkok)
    # --------------------------------------------------------
    raw_all = (result.get("raw", "") or "")
    low_prov = result["province"]
    low_score = result["province_score"]

    # Condition A — fuzzy/conf score low OR province not กรุงเทพมหานคร
    uncertain = (
        low_prov != "กรุงเทพมหานคร" and
        low_score < 0.80 and
        result["prov_candidate_conf"] < 0.20
    )

    # Condition B — strong textual hints inside plate OCR
    hints = any([
        "กทม" in raw_all,
        "เทพ" in raw_all,
        "มหานคร" in raw_all,
        "รงเทพ" in raw_all,    # common OCR glitch
        "ทนทพน" in raw_all,    # warped-bottom glitch
        "กนพน" in raw_all,
        "หนกร" in raw_all,
    ])

    if uncertain and hints:
        result["province"] = "กรุงเทพมหานคร"
        result["province_score"] = 0.90
        result["prov_candidate_tag"] = "BKK_override"
        result["prov_candidate_text"] = "AUTO_BKK"
        result["prov_candidate_conf"] = 0.0

    # --------------------------------------------------------
    # DEBUG OVERLAY for province (text drawn on resized strip)
    # --------------------------------------------------------
    if debug_dir and best["province"]:
        try:
            overlay = np.zeros((36, 200, 3), dtype=np.uint8)
            overlay[:] = (40, 40, 40)

            pil = Image.fromarray(overlay)
            draw = ImageDraw.Draw(pil)

            try:
                font = ImageFont.truetype("Arial Unicode.ttf", 20)
            except:
                font = ImageFont.load_default()

            draw.text((5, 5), f"{best['province']}  ({best['combined']:.2f})",
                      fill=(255, 0, 0), font=font)

            pil.save(os.path.join(debug_dir, "province_overlay.jpg"))
        except Exception:
            pass

    # --------------------------------------------------------
    # FINALIZE
    # --------------------------------------------------------
    result["stage2_ok"] = True
    result["stage2_reason"] = "ok"
    return result
