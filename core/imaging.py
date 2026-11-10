"""Small image helpers shared by several modules."""
from __future__ import annotations

import cv2
import numpy as np


def decode_image(data: bytes) -> np.ndarray | None:
    """Decode JPEG/PNG/WebP bytes into a BGR image. Returns None if not a valid image."""
    if not data:
        return None
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def limit_size(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def encode_jpeg(img: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return buf.tobytes()


def crop_with_margin(img: np.ndarray, bbox: tuple[int, int, int, int], margin: float = 0.08) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    mx, my = int((x2 - x1) * margin), int((y2 - y1) * margin)
    x1, y1 = max(0, x1 - mx), max(0, y1 - my)
    x2, y2 = min(w, x2 + mx), min(h, y2 + my)
    return img[y1:y2, x1:x2].copy()


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def hue_in_ranges(hue: np.ndarray, ranges: list[list[float]]) -> np.ndarray:
    """OpenCV hue (0..180). Returns boolean array: hue inside any [lo, hi] range."""
    out = np.zeros(hue.shape, dtype=bool)
    for lo, hi in ranges:
        out |= (hue >= lo) & (hue <= hi)
    return out
