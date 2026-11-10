"""Member 4 — Inspection pipeline orchestrator.

image → calibrate → detect → per fruit (measure, analyse condition, LLM opinion, grade)
      → overlay & heat-maps → persist → publish InspectionCompleted (inventory, reports)
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.db import Database
from app.services.event_bus import INSPECTION_COMPLETED, EventBus
from contracts.schemas import (
    FruitInspection,
    InspectionResult,
    LiveDetection,
    LiveFrameResult,
    QualityClass,
)
from core.imaging import crop_with_margin, encode_jpeg

log = logging.getLogger("freshgrade.inspection")

CLASS_COLORS_BGR = {
    QualityClass.extra: (60, 160, 30),
    QualityClass.class_i: (150, 160, 20),
    QualityClass.class_ii: (0, 165, 245),
    QualityClass.reject: (40, 40, 220),
}
CLASS_SHORT = {QualityClass.extra: "EXTRA", QualityClass.class_i: "CLASS I", QualityClass.class_ii: "CLASS II", QualityClass.reject: "REJECT"}


class NoFruitFound(Exception):
    pass


class InspectionService:
    def __init__(self, detector, geometry, condition, llm, grading, db: Database, bus: EventBus, media_root: Path):
        self.detector = detector
        self.geometry = geometry
        self.condition = condition
        self.llm = llm
        self.grading = grading
        self.db = db
        self.bus = bus
        self.media_root = media_root
        self._cv_lock = threading.Lock()  # models are not guaranteed thread-safe

    # ------------------------------------------------------------------ live preview
    def live_frame(self, image_bgr: np.ndarray) -> LiveFrameResult:
        t0 = time.perf_counter()
        h, w = image_bgr.shape[:2]
        with self._cv_lock:
            fruits = self.detector.detect(image_bgr)
            calib = self.geometry.calibrate(image_bgr)
        dets = [
            LiveDetection(
                fruit_type=f.fruit_type,
                confidence=round(f.confidence, 2),
                box=(f.bbox[0] / w, f.bbox[1] / h, f.bbox[2] / w, f.bbox[3] / h),
            )
            for f in fruits
        ]
        return LiveFrameResult(detections=dets, calibrated=calib is not None, ms=int((time.perf_counter() - t0) * 1000))

    # ------------------------------------------------------------------ full inspection
    def _analyse_local(self, image: np.ndarray):
        with self._cv_lock:
            calib = self.geometry.calibrate(image)
            fruits = self.detector.detect(image)
            analysed = []
            for f in fruits:
                geo = self.geometry.measure(f, calib)
                cond = self.condition.analyze(image, f)
                analysed.append((f, geo, cond))
        return calib, analysed

    async def inspect(self, image: np.ndarray, source: str = "upload", use_llm: bool = True) -> InspectionResult:
        t0 = time.perf_counter()
        inspection_id = uuid.uuid4().hex[:12]
        folder = self.media_root / inspection_id
        folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder / "original.jpg"), image)

        calib, analysed = await asyncio.to_thread(self._analyse_local, image)
        if not analysed:
            raise NoFruitFound("No fruit found. Use a plain light background, good lighting, and keep the whole fruit in view.")

        warnings: list[str] = []
        opinions: list = [None] * len(analysed)
        if use_llm and getattr(self.llm, "enabled", False):
            crops = [encode_jpeg(crop_with_margin(image, f.bbox, 0.08), 88) for f, _, _ in analysed]
            tasks = [self.llm.inspect(c, f.fruit_type.value) for c, (f, _, _) in zip(crops, analysed)]
            try:
                opinions = await asyncio.wait_for(asyncio.gather(*tasks), timeout=getattr(self.llm, "timeout_s", 12) + 3)
            except asyncio.TimeoutError:
                warnings.append("AI inspector timed out; grade is based on local models only.")
            if all(o is None for o in opinions) and not warnings:
                err = getattr(self.llm, "last_error", None)
                warnings.append("AI inspector unavailable" + (f" ({err})" if err else "") + "; grade is based on local models only.")

        fruits_out: list[FruitInspection] = []
        for i, ((f, geo, cond), llm_op) in enumerate(zip(analysed, opinions)):
            grade = self.grading.grade(f, geo, cond, llm_op)
            if cond.heatmap is not None:
                rel = f"{inspection_id}/heatmap_{i}.jpg"
                cv2.imwrite(str(self.media_root / rel), cond.heatmap)
                cond.heatmap_path = rel
            fruits_out.append(FruitInspection(fruit=f, geometry=geo, condition=cond, llm=llm_op, grade=grade))

        if calib is None:
            warnings.append("No calibration card detected: sizes are not measured and weights use typical values.")

        overlay = self.render_overlay(image, fruits_out, calib)
        cv2.imwrite(str(folder / "overlay.jpg"), overlay)

        result = InspectionResult(
            inspection_id=inspection_id,
            created_at=datetime.now(timezone.utc),
            source=source,
            image_path=f"{inspection_id}/original.jpg",
            overlay_path=f"{inspection_id}/overlay.jpg",
            calibrated=calib is not None,
            fruits=fruits_out,
            processing_ms=int((time.perf_counter() - t0) * 1000),
            warnings=warnings,
        )
        payload = result.model_dump(mode="json")
        self.db.save_inspection(payload)
        errors = self.bus.publish(INSPECTION_COMPLETED, payload)
        if errors:
            log.error("Automation errors: %s", errors)
        return result

    # ------------------------------------------------------------------ rendering
    @staticmethod
    def render_overlay(image: np.ndarray, fruits: list[FruitInspection], calib) -> np.ndarray:
        out = image.copy()
        h, w = out.shape[:2]
        thick = max(2, int(round(min(h, w) / 300)))
        font_scale = max(0.5, min(h, w) / 900)
        for i, fi in enumerate(fruits, 1):
            color = CLASS_COLORS_BGR[fi.grade.quality_class]
            mask = fi.fruit.mask
            if mask is not None:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(out, contours, -1, color, thick + 1)
            dm = fi.condition.defect_mask
            if dm is not None and np.count_nonzero(dm):
                dc, _ = cv2.findContours(dm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(out, dc, -1, (255, 0, 255), thick)
            x1, y1, x2, y2 = fi.fruit.bbox
            label = f"#{i} {fi.fruit.fruit_type.value.upper()} {CLASS_SHORT[fi.grade.quality_class]} {fi.grade.score}"
            (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
            ty = y1 - 8 if y1 - th - 14 > 0 else y2 + th + 12
            tx = min(max(0, x1), max(0, w - tw - 10))
            cv2.rectangle(out, (tx, ty - th - 8), (tx + tw + 10, ty + base), color, -1)
            cv2.putText(out, label, (tx + 5, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thick, cv2.LINE_AA)
        if calib is not None and calib.marker_bbox:
            x1, y1, x2, y2 = calib.marker_bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 200, 0), thick)
            cv2.putText(out, "SCALE", (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, (255, 200, 0), thick, cv2.LINE_AA)
        return out
