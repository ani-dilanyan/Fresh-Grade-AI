"""Shared data contract for FreshGrade AI.

Every module exchanges data ONLY through these pydantic models.
Changes to this file require approval from all four team members (via PR).
Owner: Member 4 (platform). Consumers: everyone.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class FruitType(str, Enum):
    apple = "apple"
    tomato = "tomato"
    orange = "orange"
    banana = "banana"
    lemon = "lemon"


class QualityClass(str, Enum):
    extra = "Extra"
    class_i = "I"
    class_ii = "II"
    reject = "Reject"


# Ordering used for comparisons (lower index = better quality)
CLASS_ORDER = [QualityClass.extra, QualityClass.class_i, QualityClass.class_ii, QualityClass.reject]


def class_rank(c: QualityClass) -> int:
    return CLASS_ORDER.index(c)


BBox = tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels


class DetectedFruit(BaseModel):
    """Output of Member 1's detector. `mask` is an in-memory binary mask (not serialised)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    fruit_id: int
    fruit_type: FruitType
    confidence: float = Field(ge=0, le=1)
    bbox: BBox
    mask_area_px: int = 0
    detector: str = "unknown"  # which implementation produced it (yolo / classical / mock)
    mask: Optional[np.ndarray] = Field(default=None, exclude=True, repr=False)


class Calibration(BaseModel):
    mm_per_px: float = Field(gt=0)
    method: str = "aruco"
    marker_bbox: Optional[BBox] = None


class Geometry(BaseModel):
    calibrated: bool
    major_axis_px: float
    minor_axis_px: float
    diameter_mm: Optional[float] = None
    length_mm: Optional[float] = None
    volume_cm3: Optional[float] = None
    est_weight_g: float
    size_class: str  # small | medium | large | unmeasured


class DefectRegion(BaseModel):
    area_px: int
    area_pct: float
    bbox: BBox


class ConditionResult(BaseModel):
    """Output of Member 2's condition analyser."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    defect_pct: float = Field(ge=0, le=100)
    defects: list[DefectRegion] = []
    anomaly_score: float = Field(ge=0, le=1)
    rotten_prob: float = Field(ge=0, le=1)
    color_uniformity: float = Field(ge=0, le=1)
    target_color_pct: float = Field(ge=0, le=100)
    # trained condition model (optional): probability of visible damage/ageing (incl. rot) and its verdict
    damage_prob: float = Field(default=0.0, ge=0, le=1)
    condition_label: Optional[str] = None
    condition_probs: dict[str, float] = {}
    analyzer: str = "unknown"
    heatmap_path: Optional[str] = None
    # in-memory artefacts (not serialised)
    defect_mask: Optional[np.ndarray] = Field(default=None, exclude=True, repr=False)
    heatmap: Optional[np.ndarray] = Field(default=None, exclude=True, repr=False)


class LLMOpinion(BaseModel):
    """Output of Member 3's vision-LLM inspector (optional second opinion)."""

    defect_types: list[str] = []
    suggested_class: QualityClass
    fruit_type_seen: Optional[str] = None
    reasoning: str = ""
    latency_ms: int = 0
    provider: str = "unknown"


class GradeResult(BaseModel):
    """Output of Member 3's grading engine."""

    quality_class: QualityClass
    score: int = Field(ge=0, le=100)
    confidence: float = Field(ge=0, le=1)
    reasons: list[str]
    needs_manual_review: bool = False
    price_per_kg: float
    item_value: float
    currency: str = "EUR"
    bin_code: str = ""


class FruitInspection(BaseModel):
    fruit: DetectedFruit
    geometry: Geometry
    condition: ConditionResult
    llm: Optional[LLMOpinion] = None
    grade: GradeResult


class InspectionResult(BaseModel):
    inspection_id: str
    created_at: datetime
    source: str  # live | upload | api
    image_path: str
    overlay_path: str
    calibrated: bool = False
    fruits: list[FruitInspection]
    processing_ms: int
    warnings: list[str] = []


class LiveDetection(BaseModel):
    """Lightweight message for the live WebSocket preview."""

    fruit_type: FruitType
    confidence: float
    box: tuple[float, float, float, float]  # normalised x1, y1, x2, y2 (0..1)


class LiveFrameResult(BaseModel):
    detections: list[LiveDetection]
    calibrated: bool
    ms: int
