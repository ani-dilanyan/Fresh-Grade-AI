"""Member 1 — Calibration (ArUco card) and size / volume / weight estimation."""
from __future__ import annotations

import logging
import math
from typing import Optional

import cv2
import numpy as np

from contracts.schemas import Calibration, DetectedFruit, Geometry

log = logging.getLogger("freshgrade.geometry")


class ArucoGeometryEstimator:
    """Implements GeometryEstimator.

    calibrate(): finds the printed ArUco marker (default 50 mm, DICT_4X4_50, id 0)
                 and returns millimetres per pixel.
    measure():   fits an ellipse to the fruit mask and estimates
                 volume as an ellipsoid of revolution V = 4/3·π·a·b² and weight = V·density.
    """

    def __init__(self, ontology_provider, dictionary: str = "DICT_4X4_50", marker_id: int = 0, marker_size_mm: float = 50.0):
        self.ontology_provider = ontology_provider  # callable returning the current ontology dict
        self.marker_id = marker_id
        self.marker_size_mm = marker_size_mm
        self.aruco_ok = hasattr(cv2, "aruco")
        if self.aruco_ok:
            d = getattr(cv2.aruco, dictionary)
            self.dictionary = cv2.aruco.getPredefinedDictionary(d)
            if hasattr(cv2.aruco, "ArucoDetector"):
                self._detector = cv2.aruco.ArucoDetector(self.dictionary, cv2.aruco.DetectorParameters())
            else:  # OpenCV < 4.7
                self._detector = None
                self._params = cv2.aruco.DetectorParameters_create()
        else:
            log.warning("cv2.aruco not available → calibration disabled (install opencv-contrib-python)")

    # ---------------------------------------------------------------- calibration
    def calibrate(self, image_bgr: np.ndarray) -> Optional[Calibration]:
        if not self.aruco_ok:
            return None
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        if self._detector is not None:
            corners, ids, _ = self._detector.detectMarkers(gray)
        else:  # pragma: no cover
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self._params)
        if ids is None:
            return None
        for c, mid in zip(corners, ids.flatten()):
            if int(mid) != self.marker_id:
                continue
            pts = c.reshape(4, 2)
            sides = [np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]
            side_px = float(np.mean(sides))
            if side_px < 10:
                continue
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)
            return Calibration(
                mm_per_px=self.marker_size_mm / side_px,
                method="aruco",
                marker_bbox=(int(x1), int(y1), int(x2), int(y2)),
            )
        return None

    # ---------------------------------------------------------------- measurement
    @staticmethod
    def axes_px(mask: np.ndarray) -> tuple[float, float]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return 0.0, 0.0
        c = max(contours, key=cv2.contourArea)
        if len(c) >= 5:
            (_, _), (d1, d2), _ = cv2.fitEllipse(c)
        else:
            _, (d1, d2), _ = cv2.minAreaRect(c)
        return float(max(d1, d2)), float(min(d1, d2))

    def measure(self, fruit: DetectedFruit, calib: Optional[Calibration]) -> Geometry:
        profile = self.ontology_provider()["fruit_profiles"][fruit.fruit_type.value]
        major, minor = self.axes_px(fruit.mask) if fruit.mask is not None else (0.0, 0.0)
        typical_w = float(profile["typical_weight_g"])
        if calib is None or major == 0:
            return Geometry(
                calibrated=False,
                major_axis_px=round(major, 1),
                minor_axis_px=round(minor, 1),
                est_weight_g=typical_w,
                size_class="unmeasured",
            )
        major_mm, minor_mm = major * calib.mm_per_px, minor * calib.mm_per_px
        a, b = major_mm / 2, minor_mm / 2
        volume_cm3 = (4.0 / 3.0) * math.pi * a * b * b / 1000.0
        weight = volume_cm3 * float(profile["density_g_per_cm3"])
        # size measured along the relevant axis: diameter for round fruit, length for banana
        size_ref = major_mm if fruit.fruit_type.value == "banana" else minor_mm
        s = profile["size_mm"]
        size_class = "small" if size_ref < s["small_below"] else "large" if size_ref > s["large_above"] else "medium"
        return Geometry(
            calibrated=True,
            major_axis_px=round(major, 1),
            minor_axis_px=round(minor, 1),
            diameter_mm=round(minor_mm, 1),
            length_mm=round(major_mm, 1),
            volume_cm3=round(volume_cm3, 1),
            est_weight_g=round(weight, 1),
            size_class=size_class,
        )
