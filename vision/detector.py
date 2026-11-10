"""Member 1 — Fruit detection & instance segmentation.

Three implementations of the FruitDetector interface:

* YoloFruitDetector      – Ultralytics YOLO11-seg. Uses Member 1's fine-tuned weights
                           (artifacts/fruit_seg.pt) when present, otherwise the pretrained
                           COCO model, which already knows apple, orange and banana.
* ClassicalFruitDetector – OpenCV colour segmentation baseline. Needs no training and works
                           well on a plain, light background (the demo set-up).
* AutoFruitDetector      – YOLO first, classical fallback when YOLO is unavailable or finds
                           nothing (e.g. tomatoes with the COCO model).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from contracts.schemas import DetectedFruit, FruitType
from core.imaging import mask_bbox

log = logging.getLogger("freshgrade.detector")

# COCO class ids in the pretrained YOLO model that correspond to our fruit types
COCO_FRUIT_IDS = {46: FruitType.banana, 47: FruitType.apple, 49: FruitType.orange}


def _name_to_fruit(name: str) -> Optional[FruitType]:
    n = name.strip().lower()
    for ft in FruitType:
        if ft.value in n:  # "apple", "Apple", "fresh_apple", "rotten apple" ...
            return ft
    return None


# --------------------------------------------------------------------------------------
# Classical OpenCV detector (baseline)
# --------------------------------------------------------------------------------------
class ClassicalFruitDetector:
    """Background subtraction in Lab space + contour analysis + hue-based fruit typing."""

    name = "classical"

    def __init__(self, min_area_frac: float = 0.004, min_saturation: float = 55):
        self.min_area_frac = min_area_frac
        self.min_saturation = min_saturation

    @staticmethod
    def _border(lab: np.ndarray) -> np.ndarray:
        h, w = lab.shape[:2]
        b = max(2, int(min(h, w) * 0.04))
        return np.concatenate(
            [lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3), lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)]
        ).astype(np.float32)

    @classmethod
    def _background_lab(cls, lab: np.ndarray) -> np.ndarray:
        return np.median(cls._border(lab), axis=0)

    @classmethod
    def plain_background(cls, image_bgr: np.ndarray) -> bool:
        """True when the image border is a fairly uniform colour (table, paper, light box)."""
        small = cv2.resize(image_bgr, (320, int(320 * image_bgr.shape[0] / image_bgr.shape[1])))
        lab = cv2.cvtColor(cv2.GaussianBlur(small, (5, 5), 0), cv2.COLOR_BGR2LAB)
        border = cls._border(lab)
        med = np.median(border, axis=0)
        frac_close = float(np.mean(np.linalg.norm(border - med, axis=1) < 22))
        return frac_close > 0.85

    @staticmethod
    def classify_by_colour(hsv: np.ndarray, mask: np.ndarray, elongation: float) -> tuple[FruitType, float]:
        """Very rough fruit typing from dominant hue and shape (baseline only)."""
        hue = hsv[..., 0][mask > 0].astype(np.float32)
        sat = hsv[..., 1][mask > 0].astype(np.float32)
        hue = hue[sat > 60] if np.count_nonzero(sat > 60) > 50 else hue
        # circular-aware: map reds near 180 to negative so they cluster with 0
        hue = np.where(hue > 150, hue - 180, hue)
        h = float(np.median(hue))
        if h < 8:
            return FruitType.apple, 0.55  # red: apple (tomato needs the trained model)
        if h < 20:
            return FruitType.orange, 0.55
        if h < 36:
            return (FruitType.banana, 0.55) if elongation > 1.8 else (FruitType.lemon, 0.5)
        return FruitType.apple, 0.45  # green fruit: most likely a green apple

    def detect(self, image_bgr: np.ndarray) -> list[DetectedFruit]:
        h, w = image_bgr.shape[:2]
        blur = cv2.GaussianBlur(image_bgr, (5, 5), 0)
        lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB).astype(np.float32)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        bg = self._background_lab(lab)
        dist = np.linalg.norm(lab - bg, axis=2)
        # foreground = clearly different from background AND reasonably saturated colour
        fg = ((dist > 28) & (hsv[..., 1] > 40)).astype(np.uint8) * 255
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k, iterations=2)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=3)

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = self.min_area_frac * h * w
        fruits: list[DetectedFruit] = []
        for c in sorted(contours, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            mask = np.zeros((h, w), np.uint8)
            cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)
            if float(np.median(hsv[..., 1][mask > 0])) < self.min_saturation:
                continue  # grey / black / white objects (e.g. the calibration marker) are not fruit
            rect = cv2.minAreaRect(c)
            (rw, rh) = rect[1]
            elong = max(rw, rh) / max(1.0, min(rw, rh))
            ftype, conf = self.classify_by_colour(hsv, mask, elong)
            fruits.append(
                DetectedFruit(
                    fruit_id=len(fruits),
                    fruit_type=ftype,
                    confidence=conf,
                    bbox=mask_bbox(mask),
                    mask_area_px=int(np.count_nonzero(mask)),
                    detector=self.name,
                    mask=mask,
                )
            )
            if len(fruits) >= 12:
                break
        return fruits


# --------------------------------------------------------------------------------------
# YOLO detector (deep learning)
# --------------------------------------------------------------------------------------
class YoloFruitDetector:
    name = "yolo"

    def __init__(self, custom_weights: Optional[Path], pretrained_weights: str, min_confidence: float, imgsz: int):
        from ultralytics import YOLO  # imported lazily: optional dependency

        self.custom = bool(custom_weights and custom_weights.exists())
        weights = str(custom_weights) if self.custom else pretrained_weights
        log.info("Loading YOLO weights: %s", weights)
        self.model = YOLO(weights)
        self.min_conf = min_confidence
        self.imgsz = imgsz
        self.name = "yolo-custom" if self.custom else "yolo-coco"
        # Build class-id -> FruitType map
        self.class_map: dict[int, FruitType] = {}
        names = self.model.names
        if self.custom:
            for cid, cname in names.items():
                ft = _name_to_fruit(cname)
                if ft:
                    self.class_map[int(cid)] = ft
        else:
            self.class_map = dict(COCO_FRUIT_IDS)
        # warm-up so the first live frame is fast
        self.model.predict(np.zeros((imgsz, imgsz, 3), np.uint8), imgsz=imgsz, verbose=False)

    @staticmethod
    def _is_red(image_bgr: np.ndarray, mask: np.ndarray) -> bool:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        sel = (mask > 0) & (hsv[..., 1] > 70)
        if np.count_nonzero(sel) < 50:
            return False
        hue = hsv[..., 0][sel].astype(np.float32)
        red = (hue <= 7) | (hue >= 165)
        return float(np.mean(red)) > 0.5

    @staticmethod
    def _is_lemon_like(image_bgr: np.ndarray, mask: np.ndarray) -> bool:
        """Lemons are oval with pointed ends and/or yellow; oranges are round and orange."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return False
        c = max(contours, key=cv2.contourArea)
        elong = 1.0
        if len(c) >= 5:
            (_, _), (d1, d2), _ = cv2.fitEllipse(c)
            elong = max(d1, d2) / max(1.0, min(d1, d2))
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        sel = (mask > 0) & (hsv[..., 1] > 70)
        yellow = float(np.mean((hsv[..., 0][sel] >= 22) & (hsv[..., 0][sel] <= 34))) if np.count_nonzero(sel) > 50 else 0.0
        return elong > 1.22 or yellow > 0.5

    def detect(self, image_bgr: np.ndarray) -> list[DetectedFruit]:
        h, w = image_bgr.shape[:2]
        res = self.model.predict(
            image_bgr, imgsz=self.imgsz, conf=self.min_conf, classes=list(self.class_map), verbose=False, retina_masks=False
        )[0]
        fruits: list[DetectedFruit] = []
        if res.boxes is None or len(res.boxes) == 0 or res.masks is None:
            return fruits
        polys = res.masks.xy  # polygons in original image coordinates
        for i in range(len(res.boxes)):
            cid = int(res.boxes.cls[i])
            ft = self.class_map.get(cid)
            if ft is None or i >= len(polys) or len(polys[i]) < 3:
                continue
            mask = np.zeros((h, w), np.uint8)
            cv2.fillPoly(mask, [polys[i].astype(np.int32)], 255)
            area = int(np.count_nonzero(mask))
            if area < 0.002 * h * w:
                continue
            if not self.custom and ft == FruitType.orange:
                # the generic COCO model has no "lemon" and often calls red apples "orange"
                if self._is_red(image_bgr, mask):
                    ft = FruitType.apple
                elif self._is_lemon_like(image_bgr, mask):
                    ft = FruitType.lemon
            fruits.append(
                DetectedFruit(
                    fruit_id=len(fruits),
                    fruit_type=ft,
                    confidence=float(res.boxes.conf[i]),
                    bbox=mask_bbox(mask),
                    mask_area_px=area,
                    detector=self.name,
                    mask=mask,
                )
            )
        return fruits


class AutoFruitDetector:
    """Combines both detectors.

    * YOLO runs first (robust on cluttered internet photos).
    * If YOLO finds nothing, the classical detector is used.
    * On a plain background (the demo set-up) classical detections that do not overlap a YOLO
      fruit are added too — e.g. a tomato next to an apple when only the COCO model is available.
    """

    def __init__(self, yolo: Optional[YoloFruitDetector], classical: ClassicalFruitDetector):
        self.yolo = yolo
        self.classical = classical
        self.name = f"auto({yolo.name if yolo else 'no-yolo'}+classical)"

    @staticmethod
    def _overlap(a: np.ndarray, b: np.ndarray) -> float:
        inter = np.count_nonzero(a & b)
        return inter / max(1, min(np.count_nonzero(a), np.count_nonzero(b)))

    def detect(self, image_bgr: np.ndarray) -> list[DetectedFruit]:
        fruits: list[DetectedFruit] = []
        if self.yolo is not None:
            try:
                fruits = self.yolo.detect(image_bgr)
            except Exception:  # pragma: no cover - defensive
                log.exception("YOLO detection failed, using classical fallback")
        if not fruits:
            return self.classical.detect(image_bgr)
        if self.classical.plain_background(image_bgr):
            for cf in self.classical.detect(image_bgr):
                if all(self._overlap(cf.mask, yf.mask) < 0.2 for yf in fruits):
                    cf.fruit_id = len(fruits)
                    fruits.append(cf)
        return fruits


def build_detector(cfg: dict, resolve) -> object:
    """Factory used by the application (see config/settings.yaml → detector.type)."""
    kind = cfg.get("type", "auto")
    classical = ClassicalFruitDetector()
    if kind == "classical":
        return classical
    if kind == "mock":
        from mocks.mock_modules import MockDetector

        return MockDetector()
    yolo = None
    try:
        yolo = YoloFruitDetector(
            resolve(cfg.get("custom_weights", "artifacts/fruit_seg.pt")),
            cfg.get("pretrained_weights", "yolo11n-seg.pt"),
            float(cfg.get("min_confidence", 0.35)),
            int(cfg.get("imgsz", 640)),
        )
    except ImportError:
        log.warning("ultralytics not installed → using classical detector. Install requirements-ml.txt for YOLO.")
    except Exception:
        log.exception("Could not load YOLO (offline? missing weights?) → using classical detector")
    if kind == "yolo" and yolo is not None:
        return yolo
    return AutoFruitDetector(yolo, classical)
