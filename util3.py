import os
import re
import cv2
import difflib
import numpy as np
import easyocr

# -------------------------
# OCR Reader (init once)
# -------------------------
READER = easyocr.Reader(["th"], gpu=False)

# -------------------------
# Allowlists
# -------------------------
ALLOW_PLATE = (
    "กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ"
    "0123456789 "
)
ALLOW_THAI_FULL = "".join(chr(c) for c in range(0x0E00, 0x0E80)) + "0123456789 "

# Province list
THAI_PROVINCES = [
    "กรุงเทพมหานคร","กระบี่","กำแพงเพชร","กาญจนบุรี","กาฬสินธุ์","ขอนแก่น","จันทบุรี","ฉะเชิงเทรา",
    "ชลบุรี","ชัยนาท","ชัยภูมิ","ชุมพร","เชียงใหม่","เชียงราย","ตรัง","ตราด","ตาก","นครนายก",
    "นครปฐม","นครพนม","นครราชสีมา","นครศรีธรรมราช","นครสวรรค์","นนทบุรี","นราธิวาส","น่าน",
    "บึงกาฬ","บุรีรัมย์","เบตง","ปทุมธานี","ประจวบคีรีขันธ์","ปราจีนบุรี","ปัตตานี","พระนครศรีอยุธยา","พะเยา",
    "พังงา","พัทลุง","พิจิตร","พิษณุโลก","เพชรบุรี","เพชรบูรณ์","แพร่","ภูเก็ต","มหาสารคาม","มุกดาหาร",
    "แม่ฮ่องสอน","ยโสธร","ยะลา","ร้อยเอ็ด","ระนอง","ระยอง","ราชบุรี","ลพบุรี","ลำปาง","ลำพูน",
    "เลย","ศรีสะเกษ","สกลนคร","สงขลา","สตูล","สมุทรปราการ","สมุทรสงคราม","สมุทรสาคร","สระแก้ว",
    "สระบุรี","สิงห์บุรี","สุโขทัย","สุพรรณบุรี","สุราษฎร์ธานี","สุรินทร์","หนองคาย","หนองบัวลำภู","อ่างทอง",
    "อำนาจเจริญ","อุดรธานี","อุตรดิตถ์","อุทัยธานี","อุบลราชธานี"
]

THAI_DIACRITICS_RE = re.compile(r"[\u0E31-\u0E4E]")

PLATE_CONF_WEIGHT = 30.0  # weight to favor higher OCR confidence for plate choice
PROV_CONF_WEIGHT = 0.5    # weight to favor higher OCR confidence for province choice
ALLOW_PROVINCE = "กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรฤลฦวศษสหฬอฮ"

def _fuzzy_match_province(text):
    import difflib
    
    match = difflib.get_close_matches(text, THAI_PROVINCES, n=1, cutoff=0.35)
    if match:
        score = difflib.SequenceMatcher(None, text, match[0]).ratio()
        return match[0], score
    return "", 0.0
 
# -------------------------
# Debug helper
# -------------------------
def _dbg_save(*args):
    """
    supports:
      _dbg_save(debug_dir, debug_id, suffix, img)
      _dbg_save(debug_mode, debug_dir, debug_id, suffix, img)
    """
    if len(args) == 4:
        debug_dir, debug_id, suffix, img = args
    elif len(args) == 5:
        debug_mode, debug_dir, debug_id, suffix, img = args
        if debug_mode != "save":
            return
    else:
        raise TypeError(f"_dbg_save expected 4 or 5 args, got {len(args)}")

    if not debug_dir or not debug_id or img is None:
        return
    os.makedirs(debug_dir, exist_ok=True)
    cv2.imwrite(os.path.join(debug_dir, f"{debug_id}_{suffix}.png"), img)


# -------------------------
# OCR helper
# -------------------------

def _try_warp_plate_border(plate_bgr, debug_dir=None):

    h, w = plate_bgr.shape[:2]
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            break
    else:
        # Fallback if no quadrilateral is found
        pts = np.array([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1]
        ])

    # Order points: top-left, top-right, bottom-right, bottom-left
    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute max width and height
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(plate_bgr, M, (maxWidth, maxHeight))

    if debug_dir:
        # Save full warped image
        cv2.imwrite(os.path.join(debug_dir, "00_plate_crop_warped.jpg"), warped)

        # Create and save bottom crop fallback
        h, w = warped.shape[:2]
        crop_from_warped = warped[int(h * 0.75):h, :]
        cv2.imwrite(os.path.join(debug_dir, "zz_warped.jpg"), crop_from_warped)


    return warped, M, rect


def segment_chars_with_opencv(plate_bgr, debug_dir=None, debug_id="opencv_seg"):
    """
    Segments characters from the license plate using OpenCV like the PDF paper:
    1. Grayscale → GaussianBlur → AdaptiveThreshold → Canny → Morph
    2. FindContours for character regions
    """
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive thresholding
    bin_img = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        19, 9
    )

    # Canny edge detection
    edges = cv2.Canny(bin_img, 100, 200)

    # Morphological closing
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Find contours as character candidates
    contours, _ = cv2.findContours(morph.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter by size and sort left-to-right
    candidates = []
    H, W = morph.shape
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h > 0.4 * H and h < 0.95 * H and w > 5 and w < W * 0.25:
            candidates.append((x, y, x + w, y + h))

    candidates = sorted(candidates, key=lambda b: b[0])

    # Debug drawing
    if debug_dir:
        vis = cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)
        for x1, y1, x2, y2 in candidates:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.imwrite(f"{debug_dir}/{debug_id}_seg_vis.png", vis)

    # Return cropped char regions
    char_crops = [gray[y1:y2, x1:x2] for x1, y1, x2, y2 in candidates]
    return char_crops

def _ocr_text(img, allowlist, mag_ratio=2.0, beam=False):
    if img is None:
        return ""
    kwargs = dict(detail=0, paragraph=False, allowlist=allowlist, mag_ratio=mag_ratio)
    if beam:
        kwargs["decoder"] = "beamsearch"
        kwargs["beamWidth"] = 5
    out = READER.readtext(img, **kwargs)
    return " ".join(t for t in out if t).strip()


def _ocr_with_conf(img, allowlist, mag_ratio=2.0, beam=False):
    """Return (text, avg_conf) using detail=1 from EasyOCR."""
    if img is None:
        return "", 0.0
    kwargs = {
        "detail": 1,
        "paragraph": False,
        "allowlist": allowlist,
        "mag_ratio": mag_ratio
    }
    if beam:
        kwargs["decoder"] = "beamsearch"
        kwargs["beamWidth"] = 5
    try:
        out = READER.readtext(img, **kwargs)
        texts = [t[1] for t in out if t and t[1]]
        confs = [float(t[2]) for t in out if t and len(t) > 2]
        if not texts:
            return "", 0.0
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        return " ".join(texts).strip(), avg_conf
    except Exception as e:
        print(f"❌ EasyOCR failed: {e}")
        return "", 0.0

# -------------------------
# Cleaning
# -------------------------
def _basic_clean(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[^0-9ก-๙ ]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def clean_plate_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[^0-9ก-ฮ ]", "", s)
    s = THAI_DIACRITICS_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def clean_province_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\n", " ")
    s = re.sub(r"[^ก-๙ ]", " ", s)
    s = re.sub(r"\d+", " ", s)
    s = THAI_DIACRITICS_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _norm_thai(s: str) -> str:
    s = (s or "").strip()
    s = THAI_DIACRITICS_RE.sub("", s)
    s = re.sub(r"\s+", "", s)
    return s


# -------------------------
# Province matching (safe)
# -------------------------
def match_province(text: str, cutoff=0.7, margin=0.02):
    text = (text or "").strip()
    if not text:
        return "", 0.0

    tokens = [t for t in text.split() if len(_norm_thai(t)) >= 3]
    if not tokens:
        return "", 0.0

    cands = set()
    n = len(tokens)
    for i in range(n):
        for j in range(i + 1, min(n, i + 3) + 1):
            cands.add("".join(tokens[i:j]))
    cands.add("".join(tokens))

    best_p, best_s = "", 0.0
    second = 0.0

    for cand in cands:
        nc = _norm_thai(cand)
        if len(nc) < 3:
            continue
        for p in THAI_PROVINCES:
            sp = difflib.SequenceMatcher(None, nc, _norm_thai(p)).ratio()
            if sp > best_s:
                second = best_s
                best_s = sp
                best_p = p
            elif sp > second:
                second = sp

    if best_p and best_s >= cutoff and (best_s - second) >= margin:
        return best_p, float(best_s)
    return "", float(best_s)


# -------------------------
# Parse plate (top line + digits)
# -------------------------
def parse_plate_text(raw: str):
    raw = (raw or "").strip()
    cleaned = clean_plate_text(raw)

    pattern = re.compile(r"([0-9]?[ก-ฮ]{1,3})\s*([0-9]{1,4})")
    matches = list(pattern.finditer(cleaned))
    if not matches:
        return {"raw": raw, "cleaned": cleaned, "line1": "", "line2": ""}

    best_m = None
    best_score = -1e9
    for m in matches:
        l1, l2 = m.group(1), m.group(2)
        digits_len = len(l2)
        letters_part = l1[1:] if (l1 and l1[0].isdigit()) else l1
        letters_len = len(letters_part)

        score = 0.0
        score += digits_len * 25.0
        score += letters_len * 8.0
        if digits_len >= 3:
            score += 60.0
        if digits_len == 4:
            score += 30.0
        if digits_len == 1:
            score -= 80.0
        score -= m.start() * 0.05

        if score > best_score:
            best_score = score
            best_m = m

    return {
        "raw": raw,
        "cleaned": cleaned,
        "line1": best_m.group(1) if best_m else "",
        "line2": best_m.group(2) if best_m else "",
    }


# =========================================================
# WHERE YOU REDUCE NOISE (single place to tune)
# =========================================================
def preprocess_plate_for_ocr(bgr_img, target_height=128):
    import cv2
    import numpy as np

    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    # Resize with scale
    scale = target_height / gray.shape[0]
    new_w = max(1, int(gray.shape[1] * scale * 1.6))  # Stretch width
    gray = cv2.resize(gray, (new_w, target_height), interpolation=cv2.INTER_CUBIC)

    # Gaussian Blur (less edge destruction than median)
    blurred = cv2.GaussianBlur(gray, (5, 5), sigmaX=0)

    # Adaptive Thresholding (handles uneven lighting)
    bin_img = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        19, 9
    )

    # Edge detection (Canny)
    edges = cv2.Canny(bin_img, 100, 200)

    # Morph operations
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    return gray, bin_img, morph


# -------------------------
# Main API
# -------------------------
def read_plate_text(
    plate_bgr,
    do_border_crop=True,
    prov_bgr=None,
    debug_id=None,
    debug_dir=None,
    debug_mode="save",
):
    if debug_mode != "save":
        debug_dir = None

    # Updated preprocessing
    pre_gray, pre_bin, pre_morph = preprocess_plate_for_ocr(plate_bgr, target_height=192)
    if pre_gray is None:
        return None

    _dbg_save(debug_dir, debug_id, "10_ocr_plate_gray", pre_gray)
    _dbg_save(debug_dir, debug_id, "11_ocr_plate_bin", pre_bin)
    _dbg_save(debug_dir, debug_id, "12_ocr_plate_morph", pre_morph)

    # OCR plate (multiple candidates)
    plate_raw_g, conf_g = _ocr_with_conf(pre_gray, ALLOW_PLATE, mag_ratio=2.0, beam=False)
    plate_raw_b, conf_b = _ocr_with_conf(pre_bin, ALLOW_PLATE, mag_ratio=2.0, beam=False)
    plate_raw_m, conf_m = _ocr_with_conf(pre_morph, ALLOW_PLATE, mag_ratio=2.5, beam=True)
    plate_raw_hi, conf_hi = _ocr_with_conf(pre_gray, ALLOW_PLATE, mag_ratio=3.2, beam=True)

    plate_raw_g = plate_raw_g or ""

    info_g = parse_plate_text(plate_raw_g);   info_g["_conf"] = conf_g
    info_b = parse_plate_text(plate_raw_b);   info_b["_conf"] = conf_b
    info_m = parse_plate_text(plate_raw_m);   info_m["_conf"] = conf_m
    info_hi = parse_plate_text(plate_raw_hi); info_hi["_conf"] = conf_hi

    if debug_dir and debug_id:
        try:
            os.makedirs(debug_dir, exist_ok=True)
            with open(os.path.join(debug_dir, f"{debug_id}_ocr_candidates.txt"), "w", encoding="utf-8") as fh:
                fh.write(f"gray:   '{plate_raw_g}'  conf={conf_g}\n")
                fh.write(f"bin:    '{plate_raw_b}'  conf={conf_b}\n")
                fh.write(f"morph:  '{plate_raw_m}'  conf={conf_m}\n")
                fh.write(f"hi:     '{plate_raw_hi}'  conf={conf_hi}\n")
        except Exception:
            pass

    def _plate_score(info):
        l1 = info.get("line1") or ""
        l2 = info.get("line2") or ""
        if not l1 or not l2:
            return -1e9
        digits_len = len(l2)
        letters_part = l1[1:] if (l1 and l1[0].isdigit()) else l1
        letters_len = len(letters_part)
        s = digits_len * 25 + letters_len * 8
        if digits_len >= 3: s += 60
        if digits_len == 4: s += 30
        s += PLATE_CONF_WEIGHT * float(info.get("_conf", 0.0))
        return s

    best_plate = max([info_g, info_b, info_m, info_hi], key=_plate_score)
    best_prov, best_score, best_tag, best_prov_text = "", 0.0, "", ""

    if not best_prov or best_score < 0.80:
        tail_text = best_plate.get("raw", "")[-25:]  # take last few chars
        cleaned_tail = clean_province_text(_basic_clean(tail_text))
        tail_match, tail_score = match_province(cleaned_tail, cutoff=0.80, margin=0.03)
        if tail_match and tail_score > best_score:
            best_prov, best_score, best_tag, best_prov_text = tail_match, tail_score, "tail_fallback", cleaned_tail


    best_prov, best_score, best_tag, best_prov_text = "", 0.0, "", ""
    prov_candidate_conf = 0.0

    if prov_bgr is not None and getattr(prov_bgr, "size", 0) != 0:
        prov_g0 = cv2.cvtColor(prov_bgr, cv2.COLOR_BGR2GRAY) if len(prov_bgr.shape) == 3 else prov_bgr.copy()
        _dbg_save(debug_dir, debug_id, "20_prov_crop_gray0", prov_g0)

        ph, pw = prov_g0.shape[:2]
        if ph > 0:
            target_h = 128
            scale = float(target_h) / float(ph)
            new_w = max(1, int(pw * scale * 1.6))
            prov_g0 = cv2.resize(prov_g0, (new_w, target_h), interpolation=cv2.INTER_CUBIC)

        _dbg_save(debug_dir, debug_id, "21_prov_crop_resized", prov_g0)

        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        prov_g0 = clahe.apply(prov_g0)
        _dbg_save(debug_dir, debug_id, "22_prov_crop_clahe", prov_g0)

        prov_raw_a, prov_conf_a = _ocr_with_conf(prov_g0, ALLOW_THAI_FULL, mag_ratio=3.2, beam=True)
        prov_txt_a = clean_province_text(_basic_clean(prov_raw_a))
        prov_m_a, sc_a = match_province(prov_txt_a, cutoff=0.60, margin=0.02)

        _, prov_bin = cv2.threshold(prov_g0, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _dbg_save(debug_dir, debug_id, "23_prov_crop_bin", prov_bin)
        prov_raw_b, prov_conf_b = _ocr_with_conf(prov_bin, ALLOW_THAI_FULL, mag_ratio=3.2, beam=True)
        prov_txt_b = clean_province_text(_basic_clean(prov_raw_b))
        prov_m_b, sc_b = match_province(prov_txt_b, cutoff=0.80, margin=0.04)

        prov_inv = 255 - prov_bin
        prov_inv = cv2.normalize(prov_inv, None, 0, 255, cv2.NORM_MINMAX)
        _dbg_save(debug_dir, debug_id, "24_prov_crop_inv", prov_inv)
        prov_raw_c, prov_conf_c = _ocr_with_conf(prov_inv, ALLOW_THAI_FULL, mag_ratio=3.2, beam=True)
        prov_txt_c = clean_province_text(_basic_clean(prov_raw_c))
        prov_m_c, sc_c = match_province(prov_txt_c, cutoff=0.80, margin=0.04)

        cands = [
            (prov_m_a, sc_a + PROV_CONF_WEIGHT * float(prov_conf_a), "prov_crop_gray", prov_txt_a, float(prov_conf_a)),
            (prov_m_b, sc_b + PROV_CONF_WEIGHT * float(prov_conf_b), "prov_crop_bin",  prov_txt_b, float(prov_conf_b)),
            (prov_m_c, sc_c + PROV_CONF_WEIGHT * float(prov_conf_c), "prov_crop_inv",  prov_txt_c, float(prov_conf_c)),
        ]
        best_p, best_comb, best_tag, best_text, best_conf = max(cands, key=lambda x: x[1])
        best_prov, best_score, best_tag, best_prov_text = best_p, float(best_comb), best_tag, best_text
        prov_candidate_conf = float(best_conf)

    combined_raw = (best_plate.get("raw", "") + " " + (best_prov_text or "")).strip()

    return {
        "raw": combined_raw,
        "cleaned": _basic_clean(combined_raw),
        "line1": best_plate.get("line1", ""),
        "line2": best_plate.get("line2", ""),
        "province": best_prov,
        "province_score": float(best_score),
        "prov_candidate_conf": float(prov_candidate_conf),
        "prov_candidate_tag": best_tag,
        "prov_candidate_text": best_prov_text,
    }