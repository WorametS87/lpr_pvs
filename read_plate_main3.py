def read_plate_from_crop(
    plate_crop_bgr,
    prov_bgr=None,
    debug_dir=None,
    debug_mode='save',
    prov_box_xyxy=None,
    conf_parts: float = 0.25,
    iou_parts: float = 0.5,
):
    from util3 import (
        _ocr_with_conf,
        _try_warp_plate_border,
        ALLOW_PLATE,
        ALLOW_PROVINCE,
        _fuzzy_match_province,
        read_plate_text,
        THAI_DIACRITICS_RE,
        ALLOW_THAI_FULL,
        clean_province_text,
        match_province,
        _basic_clean,
    )
    import os
    import cv2
    import numpy as np
    import supervision as sv
    from PIL import Image, ImageDraw, ImageFont

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

    if plate_crop_bgr is None or getattr(plate_crop_bgr, "image", None) is None:
        result["stage2_reason"] = "empty_crop"
        return result

    try:
        img = plate_crop_bgr.image

        letter_info = read_plate_text(
            img,
            debug_dir=debug_dir,
            debug_id="stage2",
            debug_mode=debug_mode
        )
        result.update(letter_info)

        if prov_bgr is None and parts_model is not None:
            result["prov_candidate_tag"] = "stage2_yolo_detect"
            try:
                detections = parts_model(prov_bgr or img, verbose=False)[0]
                for det in detections.boxes.data.cpu().numpy():
                    x1, y1, x2, y2, conf, cls_id = det
                    if int(cls_id) == 1:
                        province_crop = img[int(y1):int(y2), int(x1):int(x2)]
                        if province_crop.shape[0] >= 20 and province_crop.shape[1] >= 80:
                            if debug_dir:
                                cv2.imwrite(os.path.join(debug_dir, "20_prov_crop_detected.jpg"), province_crop)

                            gray = cv2.cvtColor(province_crop, cv2.COLOR_BGR2GRAY)
                            text, conf = _ocr_with_conf(gray, allowlist=ALLOW_THAI_FULL, beam=True)
                            cleaned = clean_province_text(_basic_clean(text))
                            matched, score = _fuzzy_match_province(cleaned)

                            result["province"] = matched
                            result["province_score"] = score
                            result["prov_candidate_text"] = cleaned
                            result["prov_candidate_conf"] = conf
                            result["prov_candidate_tag"] = "yolo_detected"

                            if score >= 0.6 or conf >= 0.3:
                                return result

            except Exception as e:
                print(" YOLOv8 province detection failed:", e)

        if result.get("province_score", 0.0) < 0.75:
            raw_from_plate = result.get("raw", "")
            cleaned = clean_province_text(_basic_clean(raw_from_plate))
            fallback_prov, fallback_score = match_province(cleaned, cutoff=0.7, margin=0.03)
            if fallback_prov and fallback_score > result.get("province_score", 0.0):
                result["province"] = fallback_prov
                result["province_score"] = fallback_score
                result["prov_candidate_tag"] = "from_raw_fallback"
                result["prov_candidate_text"] = cleaned
    except Exception as e:
        result["stage2_reason"] = f"letter_error:{type(e).__name__}"
        return result

    try:
        prov_crop = None
        warped, _, _ = _try_warp_plate_border(img, debug_dir=debug_dir)
        result["debug_info"] = {}

        if prov_bgr is not None:
            prov_crop = prov_bgr
            result["prov_candidate_tag"] = "prov_from_input"
            print("✅ Using provided prov_bgr for OCR")

        elif hasattr(plate_crop_bgr, "prov_box_xyxy") and plate_crop_bgr.prov_box_xyxy:
            x1, y1, x2, y2 = plate_crop_bgr.prov_box_xyxy
            crop_from = warped if warped is not None else img
            prov_crop = crop_from[y1:y2, x1:x2]
            result["prov_candidate_tag"] = "prov_from_box"
            print(f"✅ Using YOLO box crop: {(x1,y1,x2,y2)} from {'warped' if warped is not None else 'raw'}")

        elif warped is not None:
            h, w = warped.shape[:2]
            top = int(h * 0.75)
            bot = h
            crop_from_warped = warped[top:bot, :]
            if crop_from_warped.shape[0] >= 20 and crop_from_warped.shape[1] >= 80:
                prov_crop = crop_from_warped
                result["prov_candidate_tag"] = "fallback_from_warped_bottom"
                print("🔁 Using fallback from raw bottom crop:", crop_from_warped.shape)
            else:
                print("⚠️ Raw plate fallback also too small:", crop_from_warped.shape)

        if prov_crop is None:
            result["stage2_reason"] = "prov_crop_failed"
            return result

        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, "00_plate_crop_warped.jpg"), warped)
            cv2.imwrite(os.path.join(debug_dir, "20_prov_input.jpg"), prov_crop)

        resized = cv2.resize(prov_crop, (160, 32), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 10)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        clahe_img = clahe.apply(gray)

        cand_imgs = [gray, clahe_img, adapt]
        cands = []
        for tag, img in zip(["gray", "clahe", "adapt"], cand_imgs):
            text, conf = _ocr_with_conf(img, allowlist=ALLOW_THAI_FULL, beam=True)
            text_clean = clean_province_text(_basic_clean(text))
            matched, score = _fuzzy_match_province(text_clean)
            combined = score + 0.5 * conf
            cands.append((matched, combined, tag, text_clean, conf))

        best_p, best_score, best_tag, best_prov_text, best_conf = max(cands, key=lambda x: x[1])

        if debug_dir:
            bin_img = cv2.adaptiveThreshold(
                clahe_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10
            )
            kernel = np.ones((2, 2), np.uint8)
            bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=1)
            cv2.imwrite(os.path.join(debug_dir, "21_prov_resize.jpg"), resized)
            cv2.imwrite(os.path.join(debug_dir, "22_prov_clahe.jpg"), clahe_img)
            cv2.imwrite(os.path.join(debug_dir, "23_prov_bin.jpg"), bin_img)

        result["prov_candidate_text"] = best_prov_text
        result["prov_candidate_conf"] = best_conf
        result["province"] = best_p
        result["province_score"] = best_score

        if best_score < 0.58 and best_conf < 0.15:
            print(" Rejecting weak province match:", best_p, f"(score={best_score:.2f}, conf={best_conf:.2f})")
            result["province"] = ""
            result["province_score"] = 0.0
            result["prov_candidate_tag"] = "rejected_low_conf"

        result["prov_candidate_tag"] = best_tag

        if debug_dir and best_p and resized is not None:
            overlay = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB) if len(resized.shape) == 2 or resized.shape[2] == 1 else resized.copy()
            pil = Image.fromarray(overlay)
            draw = ImageDraw.Draw(pil)
            try:
                font = ImageFont.truetype("Arial Unicode.ttf", 20)
            except:
                font = ImageFont.load_default()
            draw.text((5, 5), best_p, font=font, fill=(255, 0, 0))
            pil.save(os.path.join(debug_dir, "24_prov_text_overlay.jpg"))

    except Exception as e:
        print(" Province OCR Error:", e)
        result["prov_candidate_tag"] = f"prov_ocr_error:{type(e).__name__}"

        if result["province"] == "" and 'resized' in locals():
            try:
                extra_texts = reader.readtext(resized, detail=1, paragraph=False)
                if extra_texts:
                    fallback_text = extra_texts[0][1]
                    fallback_conf = extra_texts[0][2]
                    cleaned_text = clean_province_text(_basic_clean(fallback_text))
                    fallback_prov, fallback_score = match_province(cleaned_text)
                    print(f"🔁 EasyOCR fallback raw: {fallback_text} → match: {fallback_prov}, score={fallback_score:.2f}, conf={fallback_conf:.2f}")
                    if fallback_score > result["province_score"]:
                        result["province"] = fallback_prov
                        result["province_score"] = fallback_score
                        result["prov_candidate_text"] = cleaned_text
                        result["prov_candidate_conf"] = fallback_conf
                        result["prov_candidate_tag"] = "easyocr_direct"
            except Exception as e:
                print(" EasyOCR fallback failed:", e)

    if (not result["province"] or result["province_score"] < 0.70):
        fallback_text = result.get("raw", "")
        if fallback_text:
            last_words = fallback_text.strip().split()[-2:]
            joined = "".join(last_words)
            cleaned = clean_province_text(joined)
            if len(cleaned) >= 3:
                fallback_prov, fallback_score = match_province(cleaned)
                if fallback_score > result["province_score"]:
                    result["province"] = fallback_prov
                    result["province_score"] = fallback_score
                    result["prov_candidate_text"] = cleaned
                    result["prov_candidate_tag"] = "fallback_from_plate_ocr"
                    print(f" Fallback province used: {fallback_prov} (score={fallback_score:.2f})")

    result["stage2_ok"] = True
    result["stage2_reason"] = "ok"
    return result
