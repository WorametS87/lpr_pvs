# util4.py
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import cv2
import re

EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


THAI_RE = re.compile(r"[\u0E00-\u0E7F]")
DIGIT_RE = re.compile(r"\d")

def normalize_plate_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", "", s)
    return s

def accept_ocr(text: str, conf: float, min_conf: float = 0.45) -> bool:
    t = normalize_plate_text(text)

    if conf is None or conf < min_conf:
        return False

    if len(DIGIT_RE.findall(t)) < 3:
        return False

    if not THAI_RE.search(t):
        return False

    return True

def pad_bbox(x1, y1, x2, y2, img_w, img_h, pad=0.15):
    w = x2 - x1
    h = y2 - y1
    px = int(w * pad)
    py = int(h * pad)
    return (
        max(0, x1 - px), max(0, y1 - py),
        min(img_w, x2 + px), min(img_h, y2 + py)
    )


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def collect_images(input_path: str) -> List[Path]:
    p = Path(input_path)
    if p.is_file():
        return [p]

    imgs: List[Path] = []
    for ext in EXTS:
        imgs.extend(p.rglob(f"*{ext}"))
    return sorted(imgs)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def pad_xyxy(
    x1: float, y1: float, x2: float, y2: float,
    pad_frac: float, W: int, H: int
) -> Tuple[int, int, int, int]:
    """
    Pads an xyxy box by pad_frac of its width/height, clamped to image bounds.
    """
    bw = x2 - x1
    bh = y2 - y1
    dx = bw * pad_frac
    dy = bh * pad_frac

    x1p = int(clamp(x1 - dx, 0, W - 1))
    y1p = int(clamp(y1 - dy, 0, H - 1))
    x2p = int(clamp(x2 + dx, 0, W - 1))
    y2p = int(clamp(y2 + dy, 0, H - 1))
    return x1p, y1p, x2p, y2p


def crop(img_bgr: np.ndarray, xyxy: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = map(int, xyxy)
    if x2 <= x1 or y2 <= y1:
        return None
    c = img_bgr[y1:y2, x1:x2]
    if c.size == 0:
        return None
    return c.copy()


def extract_yolo_boxes(result) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Extract xyxy/conf/cls from a single Ultralytics result (one image).
    Returns None triplet if no boxes.
    """
    if result.boxes is None or len(result.boxes) == 0:
        return None, None, None
    xyxy = result.boxes.xyxy.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy().astype(int)
    return xyxy, conf, cls


def sort_left_to_right(
    xyxy: np.ndarray,
    conf: Optional[np.ndarray] = None,
    cls: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Stable ordering for multiple plates: left->right based on x1.
    """
    order = np.argsort(xyxy[:, 0])
    xyxy2 = xyxy[order]
    conf2 = conf[order] if conf is not None else None
    cls2 = cls[order] if cls is not None else None
    return xyxy2, conf2, cls2


def draw_boxes(
    img_bgr: np.ndarray,
    boxes_xyxy: np.ndarray,
    labels: Optional[List[str]] = None,
    thickness: int = 2
) -> np.ndarray:
    """
    Draws boxes in blue. Labels optional.
    """
    out = img_bgr.copy()
    for i, b in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = map(int, b)
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 0), thickness)
        if labels is not None and i < len(labels):
            cv2.putText(
                out,
                labels[i],
                (x1, max(0, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
                cv2.LINE_AA
            )
    return out


def save_json(path: Path, obj: Any) -> None:
    import json
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
