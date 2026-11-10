"""Mock implementations of every interface (Member 4).

They let the whole application run end-to-end before any model is trained, and are used
in tests. Select them in config/settings.yaml (type: mock).
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from contracts.schemas import ConditionResult, DetectedFruit, FruitType, LLMOpinion, QualityClass


class MockDetector:
    name = "mock"

    def detect(self, image_bgr: np.ndarray) -> list[DetectedFruit]:
        h, w = image_bgr.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        r = int(min(h, w) * 0.25)
        cv2.circle(mask, (w // 2, h // 2), r, 255, -1)
        return [
            DetectedFruit(
                fruit_id=0,
                fruit_type=FruitType.apple,
                confidence=0.9,
                bbox=(w // 2 - r, h // 2 - r, w // 2 + r, h // 2 + r),
                mask_area_px=int(np.count_nonzero(mask)),
                detector=self.name,
                mask=mask,
            )
        ]


class MockConditionAnalyzer:
    name = "mock"

    def analyze(self, image_bgr: np.ndarray, fruit: DetectedFruit) -> ConditionResult:
        return ConditionResult(
            defect_pct=2.0, defects=[], anomaly_score=0.28, rotten_prob=0.05, color_uniformity=0.8, target_color_pct=90.0, analyzer=self.name
        )


class MockLLMInspector:
    name = "mock"
    enabled = True

    async def inspect(self, crop_jpeg: bytes, fruit_type: str) -> Optional[LLMOpinion]:
        return LLMOpinion(defect_types=["scab_spot"], suggested_class=QualityClass.class_i, reasoning="Mock opinion.", provider="mock")
