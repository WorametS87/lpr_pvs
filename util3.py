import os
import re
import cv2
import difflib
import numpy as np
import easyocr

# ---------------------------------------------------------
# OCR ENGINE (global singleton)
# ---------------------------------------------------------
# EasyOCR Thai reader (CPU, stable)
READER = easyocr.Reader(["th"], gpu=False)

# ---------------------------------------------------------
# ALLOWLISTS
# ---------------------------------------------------------
ALLOW_PLATE = (
    "กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ"
    "0123456789 "
)

# Thai unicode block + digits (used for province OCR)
ALLOW_THAI_FULL = "".join(chr(c) for c in range(0x0E00, 0x0E80)) + "0123456789 "

# Province letters (no digits)
ALLOW_PROVINCE = (
    "กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรฤลฦวศษสหฬอฮ"
)

# ---------------------------------------------------------
# THAI PROVINCE LIST (77)
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# REGEX HELPERS
# ---------------------------------------------------------
THAI_DIACRITICS_RE = re.compile(r"[\u0E31-\u0E4E]")

# Weighting system
PLATE_CONF_WEIGHT = 30.0   # favor strong OCR confidence for plate
PROV_CONF_WEIGHT  = 0.5    # confidence helps province but fuzzy dominates

# ---------------------------------------------------------
# DEBUG SAVE WRAPPER
# ---------------------------------------------------------
def _dbg_save(*args):
    """
    Save debug images safely.
    Supports either:
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
    path = os.path.join(debug_dir, f"{debug_id}_{suffix}.png")
    cv2.imwrite(path, img)

# ---------------------------------------------------------
# BASIC CLEANING
# ---------------------------------------------------------
def _basic_clean(s: str) -> str:
    """Remove noise, keep Thai & digits."""
    if not s:
        return ""
    s = s.replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[^0-9ก-๙ ]", "", s)
    return re.sub(r"\s+", " ", s).strip()

def clean_plate_text(s: str) -> str:
    """Clean plate text: keep Thai + digits, remove diacritics."""
    if not s:
        return ""
    s = s.replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[^0-9ก-ฮ ]", "", s)
    s = THAI_DIACRITICS_RE.sub("", s)
    return re.sub(r"\s+", " ", s).strip()

def clean_province_text(s: str) -> str:
    """Clean province text: remove digits, diacritics."""
    if not s:
        return ""
    s = s.replace("\n", " ")
    s = re.sub(r"[^ก-๙ ]", " ", s)
    s = re.sub(r"\d+", " ", s)
    s = THAI_DIACRITICS_RE.sub("", s)
    return re.sub(r"\s+", " ", s).strip()

def _norm_thai(s: str) -> str:
    """Normalize Thai text for fuzzy matching."""
    s = (s or "").strip()
    s = THAI_DIACRITICS_RE.sub("", s)
    return re.sub(r"\s+", "", s)

# ---------------------------------------------------------
# FUZZY PROVINCE MATCHING (CCTV optimized)
# ---------------------------------------------------------
def match_province(text: str, cutoff=0.52, margin=0.02):
    """
    Try to match cleaned OCR text to a valid Thai province name.
    CCTV-optimized threshold: more tolerant but uses margin rule.
    """
    text = (text or "").strip()
    if not text:
        return "", 0.0

    # Split into meaningful tokens
    tokens = [t for t in text.split() if len(_norm_thai(t)) >= 3]
    if not tokens:
        return "", 0.0

    # Generate candidate combinations of tokens
    cands = set()
    n = len(tokens)
    for i in range(n):
        for j in range(i + 1, min(n, i + 3) + 1):
            cands.add("".join(tokens[i:j]))
    cands.add("".join(tokens))

    best_p = ""
    best_s = 0.0
    second_best = 0.0

    for cand in cands:
        nc = _norm_thai(cand)
        if len(nc) < 3:
            continue
        for p in THAI_PROVINCES:
            score = difflib.SequenceMatcher(None, nc, _norm_thai(p)).ratio()
            if score > best_s:
                second_best = best_s
                best_s = score
                best_p = p
            elif score > second_best:
                second_best = score

    # Margin rule

    if best_p and best_s >= cutoff and (best_s - second_best) >= margin:
        return best_p, float(best_s)

    return "", float(best_s)

# ---------------------------------------------------------
# PARSE PLATE TEXT (letters + digits)
# ---------------------------------------------------------
def parse_plate_text(raw: str):
    """
    Extract Thai plate format:
        [digit?][letters 1–3] [digits 1–4]
    Then score candidates for best structure.
    """
    raw = (raw or "").strip()
    cleaned = clean_plate_text(raw)

    pattern = re.compile(r"([0-9]?[ก-ฮ]{1,3})\s*([0-9]{1,4})")
    matches = list(pattern.finditer(cleaned))

    if not matches:
        return {
            "raw": raw,
            "cleaned": cleaned,
            "line1": "",
            "line2": "",
        }

    best = None
    best_score = -1e9

    for m in matches:
        l1, l2 = m.group(1), m.group(2)

        digits_len = len(l2)
        letters = l1[1:] if l1 and l1[0].isdigit() else l1
        letters_len = len(letters)

        score = (
            digits_len * 25 +
            letters_len * 8 +
            (60 if digits_len >= 3 else 0) +
            (30 if digits_len == 4 else 0) -
            (80 if digits_len == 1 else 0) -
            m.start() * 0.05
        )

        if score > best_score:
            best_score = score
            best = m

    return {
        "raw": raw,
        "cleaned": cleaned,
        "line1": best.group(1) if best else "",
        "line2": best.group(2) if best else "",
    }
# ---------------------------------------------------------
# OCR TEXT HELPERS
# ---------------------------------------------------------
def _ocr_text(img, allowlist, mag_ratio=2.0, beam=False):
    if img is None:
        return ""
    kwargs = dict(
        detail=0,
        paragraph=False,
        allowlist=allowlist,
        mag_ratio=mag_ratio,
    )
    if beam:
        kwargs["decoder"] = "beamsearch"
        kwargs["beamWidth"] = 5

    out = READER.readtext(img, **kwargs)
    return " ".join(t for t in out if t).strip()


def _ocr_with_conf(img, allowlist, mag_ratio=2.0, beam=False):
    """
    OCR returning (text, avg_confidence).
    Uses EasyOCR detail mode for confidence extraction.
    """
    if img is None:
        return "", 0.0

    kwargs = dict(
        detail=1,
        paragraph=False,
        allowlist=allowlist,
        mag_ratio=mag_ratio,
    )
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

    except Exception:
        return "", 0.0


# ---------------------------------------------------------
# SEGMENTATION (Optional, used for character-level debugging)
# ---------------------------------------------------------
def segment_chars_with_opencv(plate_bgr, debug_dir=None, debug_id="seg"):
    """
    Basic segmentation like the academic paper approach.
    """
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    bin_img = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        19, 9
    )

    edges = cv2.Canny(bin_img, 100, 200)

    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        morph.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    H, W = morph.shape
    chars = []
    bboxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h > 0.4 * H and h < 0.95 * H and w > 5 and w < W * 0.25:
            chars.append(gray[y:y+h, x:x+w])
            bboxes.append((x, y, x+w, y+h))

    # Debug visualization
    if debug_dir:
        vis = cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)
        for x1, y1, x2, y2 in bboxes:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.imwrite(os.path.join(debug_dir, f"{debug_id}_seg_vis.png"), vis)

    return chars


# ---------------------------------------------------------
# PLATE BORDER WARP
# ---------------------------------------------------------
def _try_warp_plate_border(plate_bgr, debug_dir=None):
    """
    Warps a license plate based on detected quadrilateral edges.
    Used for improving province OCR.
    """
    h, w = plate_bgr.shape[:2]

    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    quad = None

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2)
            break

    if quad is None:
        quad = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])

    # Order points
    s = quad.sum(axis=1)
    diff = np.diff(quad, axis=1)

    rect = np.zeros((4, 2), dtype="float32")
    rect[0] = quad[np.argmin(s)]
    rect[2] = quad[np.argmax(s)]
    rect[1] = quad[np.argmin(diff)]
    rect[3] = quad[np.argmax(diff)]

    tl, tr, br, bl = rect

    # Warp dimensions
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = max(int(heightA), int(heightB))

    dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(plate_bgr, M, (maxW, maxH))

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "warp_plate.jpg"), warped)

    return warped, M, rect


# ---------------------------------------------------------
# PREPROCESSING FOR OCR (Plate)
# ---------------------------------------------------------
def preprocess_plate_for_ocr(bgr_img, target_height=192):
    """
    Plate preprocessing pipeline:
    - Resize (height → target)
    - Gaussian blur
    - Adaptive threshold
    - Edge morphology
    """
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    scale = target_height / gray.shape[0]
    new_w = max(1, int(gray.shape[1] * scale * 1.6))
    gray = cv2.resize(gray, (new_w, target_height), interpolation=cv2.INTER_CUBIC)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    bin_img = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        19, 9
    )

    edges = cv2.Canny(bin_img, 100, 200)

    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    return gray, bin_img, morph


# ---------------------------------------------------------
# MAIN PLATE OCR (Stage2 internal)
# ---------------------------------------------------------
def read_plate_text(
    plate_bgr,
    debug_id=None,
    debug_dir=None,
    debug_mode="save",
):
    """
    Returns:
        {
            "raw": full raw text,
            "cleaned": cleaned raw,
            "line1": letters,
            "line2": digits,
            "province": "",
            "province_score": float,
            ...
        }
    """
    if debug_mode != "save":
        debug_dir = None

    # Preprocess plate
    gray, bin_img, morph = preprocess_plate_for_ocr(plate_bgr)

    _dbg_save(debug_dir, debug_id, "gray", gray)
    _dbg_save(debug_dir, debug_id, "bin", bin_img)
    _dbg_save(debug_dir, debug_id, "morph", morph)

    # OCR candidates
    raw_g, conf_g   = _ocr_with_conf(gray,     ALLOW_PLATE, mag_ratio=2.0, beam=False)
    raw_b, conf_b   = _ocr_with_conf(bin_img,  ALLOW_PLATE, mag_ratio=2.0, beam=False)
    raw_m, conf_m   = _ocr_with_conf(morph,    ALLOW_PLATE, mag_ratio=2.5, beam=True)
    raw_hi, conf_hi = _ocr_with_conf(gray,     ALLOW_PLATE, mag_ratio=3.2, beam=True)

    raw_g = raw_g or ""

    # Parse & annotate
    info_g  = parse_plate_text(raw_g);  info_g["_conf"]  = conf_g
    info_b  = parse_plate_text(raw_b);  info_b["_conf"]  = conf_b
    info_m  = parse_plate_text(raw_m);  info_m["_conf"]  = conf_m
    info_hi = parse_plate_text(raw_hi); info_hi["_conf"] = conf_hi

    # Choose best plate reading
    def plate_score(info):
        l1 = info.get("line1") or ""
        l2 = info.get("line2") or ""
        if not l1 or not l2:
            return -1e9
        digits_len = len(l2)
        letters = l1[1:] if l1 and l1[0].isdigit() else l1
        letters_len = len(letters)

        return (
            digits_len * 25 +
            letters_len * 8 +
            (60 if digits_len >= 3 else 0) +
            (30 if digits_len == 4 else 0) +
            PLATE_CONF_WEIGHT * float(info.get("_conf", 0.0))
        )

    best_plate = max([info_g, info_b, info_m, info_hi], key=plate_score)

    # Tail text fallback for province
    tail = best_plate.get("raw", "")[-25:]
    cleaned_tail = clean_province_text(_basic_clean(tail))
    tail_prov, tail_score = match_province(cleaned_tail)

    province = tail_prov
    province_score = tail_score
    prov_tag = "tail_fallback"
    prov_text = cleaned_tail
    prov_conf = 0.0

    # Assemble final payload
    combined_raw = (best_plate.get("raw", "") + " " + (prov_text or "")).strip()

    return {
        "raw": combined_raw,
        "cleaned": _basic_clean(combined_raw),
        "line1": best_plate.get("line1", ""),
        "line2": best_plate.get("line2", ""),
        "province": province,
        "province_score": float(province_score),
        "prov_candidate_conf": float(prov_conf),
        "prov_candidate_tag": prov_tag,
        "prov_candidate_text": prov_text,
    }
