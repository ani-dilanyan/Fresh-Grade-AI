"""Generate synthetic sample images for demos and automated tests.

Creates data/samples/*.jpg: healthy, bruised and rotten apples, an orange, a lemon, a banana
and a multi-fruit scene with a calibration card (ArUco marker at a known scale of 4 px/mm).

These are NOT training data — they exist so the pipeline can be tested without a camera
or internet. For the demo use real fruit or real photos.

Usage:  python tools/make_samples.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "samples"
PX_PER_MM = 4.0
rng = np.random.default_rng(7)


def background(h=900, w=1200) -> np.ndarray:
    y = np.linspace(0, 1, h)[:, None]
    base = (238 - 14 * y) * np.ones((h, w))
    img = np.dstack([base - 4, base, base + 2]).astype(np.float32)
    img += rng.normal(0, 2.0, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def shadow(img: np.ndarray, center, axes, angle=0):
    sh = np.zeros(img.shape[:2], np.float32)
    cv2.ellipse(sh, (center[0] + int(axes[0] * 0.12), center[1] + int(axes[1] * 0.18)), axes, angle, 0, 360, 1.0, -1)
    sh = cv2.GaussianBlur(sh, (0, 0), axes[0] * 0.15)
    img[:] = np.clip(img.astype(np.float32) * (1 - 0.28 * sh[..., None]), 0, 255).astype(np.uint8)


def fruit(img: np.ndarray, center, axes, color_bgr, angle=0, texture=6.0, highlight=True) -> np.ndarray:
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # radial shading for a 3-D look (light from top-left)
    dx = (xx - (center[0] - axes[0] * 0.3)) / axes[0]
    dy = (yy - (center[1] - axes[1] * 0.3)) / axes[1]
    r = np.sqrt(dx ** 2 + dy ** 2)
    shade = np.clip(1.15 - 0.38 * r, 0.55, 1.15)
    col = np.array(color_bgr, np.float32)[None, None, :] * shade[..., None]
    col += rng.normal(0, texture, (h, w, 1))
    if highlight:
        hl = np.exp(-(((xx - (center[0] - axes[0] * 0.35)) ** 2) / (2 * (axes[0] * 0.12) ** 2)
                      + ((yy - (center[1] - axes[1] * 0.4)) ** 2) / (2 * (axes[1] * 0.09) ** 2)))
        col += 70 * hl[..., None]
    col = np.clip(col, 0, 255).astype(np.uint8)
    m3 = cv2.GaussianBlur(mask, (5, 5), 0).astype(np.float32)[..., None] / 255.0
    img[:] = (img * (1 - m3) + col * m3).astype(np.uint8)
    return mask


def blemish(img: np.ndarray, mask: np.ndarray, center, radius, color_bgr=(40, 70, 100), irregular=True):
    h, w = img.shape[:2]
    spot = np.zeros((h, w), np.uint8)
    if irregular:
        pts = []
        for k in range(14):
            a = 2 * np.pi * k / 14
            rr = radius * rng.uniform(0.7, 1.25)
            pts.append([center[0] + rr * np.cos(a), center[1] + rr * np.sin(a)])
        cv2.fillPoly(spot, [np.array(pts, np.int32)], 255)
    else:
        cv2.circle(spot, center, radius, 255, -1)
    spot = cv2.bitwise_and(spot, mask)
    alpha = cv2.GaussianBlur(spot, (0, 0), max(1.5, radius * 0.15)).astype(np.float32)[..., None] / 255.0
    col = np.array(color_bgr, np.float32)[None, None, :] + rng.normal(0, 6, (h, w, 1))
    img[:] = np.clip(img * (1 - alpha * 0.92) + col * alpha * 0.92, 0, 255).astype(np.uint8)


def aruco_card(img: np.ndarray, top_left, size_mm=50.0):
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    px = int(size_mm * PX_PER_MM)
    marker = cv2.aruco.generateImageMarker(d, 0, px) if hasattr(cv2.aruco, "generateImageMarker") else cv2.aruco.drawMarker(d, 0, px)
    x, y = top_left
    border = int(8 * PX_PER_MM)
    cv2.rectangle(img, (x - border, y - border), (x + px + border, y + px + border), (250, 250, 250), -1)
    img[y:y + px, x:x + px] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)


def save(name: str, img: np.ndarray):
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("wrote", OUT / name)


RED = (40, 40, 185)
ORANGE = (20, 130, 235)
YELLOW = (40, 205, 235)
LEMON = (50, 215, 225)


def make_all():
    d = int(76 * PX_PER_MM / 2)  # 76 mm apple

    img = background(); c = (600, 450)
    shadow(img, c, (d, d)); fruit(img, c, (d, d), RED)
    save("apple_healthy.jpg", img)

    img = background(); c = (600, 450)
    shadow(img, c, (d, d)); m = fruit(img, c, (d, d), RED)
    blemish(img, m, (650, 500), 34, (30, 50, 80)); blemish(img, m, (540, 410), 16, (30, 50, 80))
    save("apple_bruised.jpg", img)

    img = background(); c = (600, 450)
    shadow(img, c, (d, d)); m = fruit(img, c, (d, d), RED)
    blemish(img, m, (630, 480), 95, (18, 32, 48)); blemish(img, m, (520, 380), 45, (20, 38, 55))
    save("apple_rotten.jpg", img)

    img = background(); c = (600, 450); r = int(80 * PX_PER_MM / 2)
    shadow(img, c, (r, r)); fruit(img, c, (r, r), ORANGE, texture=9)
    save("orange_healthy.jpg", img)

    img = background(); c = (600, 450)
    shadow(img, c, (150, 105)); fruit(img, c, (150, 105), LEMON, angle=10)
    save("lemon.jpg", img)

    img = background(); c = (600, 450)
    shadow(img, c, (330, 70), 12); m = fruit(img, c, (330, 70), YELLOW, angle=12)
    blemish(img, m, (450, 420), 14, (30, 60, 90)); blemish(img, m, (720, 470), 10, (30, 60, 90))
    save("banana_spots.jpg", img)

    # multi-fruit scene with calibration card
    img = background(1000, 1500)
    aruco_card(img, (110, 380))
    c1, c2, c3 = (650, 480), (1000, 330), (1180, 720)
    shadow(img, c1, (d, d)); m1 = fruit(img, c1, (d, d), RED)
    r = int(80 * PX_PER_MM / 2)
    shadow(img, c2, (r - 20, r - 20)); fruit(img, c2, (r - 20, r - 20), ORANGE, texture=9)
    shadow(img, c3, (d - 10, d - 10)); m3 = fruit(img, c3, (d - 10, d - 10), RED)
    blemish(img, m3, (1220, 760), 40, (30, 50, 80))
    save("mixed_with_card.jpg", img)


if __name__ == "__main__":
    if not hasattr(cv2, "aruco"):
        sys.exit("cv2.aruco missing: pip install opencv-contrib-python")
    make_all()
