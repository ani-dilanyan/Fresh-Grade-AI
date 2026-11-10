"""Module interfaces (ports). Each team member implements one of these.

The application only ever talks to these Protocols, so any implementation
(real model, classical baseline or mock) can be swapped via config/settings.yaml.
"""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import numpy as np

from contracts.schemas import (
    Calibration,
    ConditionResult,
    DetectedFruit,
    Geometry,
    GradeResult,
    LLMOpinion,
)


@runtime_checkable
class FruitDetector(Protocol):  # Member 1
    name: str

    def detect(self, image_bgr: np.ndarray) -> list[DetectedFruit]:
        """Return every fruit found in the image, each with a binary mask."""
        ...


@runtime_checkable
class GeometryEstimator(Protocol):  # Member 1
    def calibrate(self, image_bgr: np.ndarray) -> Optional[Calibration]:
        """Find the calibration card (ArUco marker) and return the mm-per-pixel scale."""
        ...

    def measure(self, fruit: DetectedFruit, calib: Optional[Calibration]) -> Geometry:
        ...


@runtime_checkable
class ConditionAnalyzer(Protocol):  # Member 2
    name: str

    def analyze(self, image_bgr: np.ndarray, fruit: DetectedFruit) -> ConditionResult:
        ...


@runtime_checkable
class LLMInspector(Protocol):  # Member 3
    name: str
    enabled: bool

    async def inspect(self, crop_jpeg: bytes, fruit_type: str) -> Optional[LLMOpinion]:
        """Return a structured second opinion, or None when unavailable / timed out."""
        ...


@runtime_checkable
class GradingEngine(Protocol):  # Member 3
    def grade(
        self,
        fruit: DetectedFruit,
        geometry: Geometry,
        condition: ConditionResult,
        llm: Optional[LLMOpinion],
    ) -> GradeResult:
        ...
