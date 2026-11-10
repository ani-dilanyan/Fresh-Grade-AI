"""Dependency wiring (Member 4): builds every module from config/settings.yaml."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.db import Database
from app.services.event_bus import INSPECTION_COMPLETED, EventBus
from app.services.inspection_service import InspectionService
from automation.inventory_service import InventoryService
from automation.pricing_job import PricingJob
from core import config
from defects.analyzer import ClassicalConditionAnalyzer
from defects.classifier import try_load_classifier
from grading.llm_inspector import build_llm_inspector
from grading.pricing import PricingEngine
from grading.rules import RuleBasedGradingEngine
from vision.detector import build_detector
from vision.geometry import ArucoGeometryEstimator

log = logging.getLogger("freshgrade.container")


class Container:
    def __init__(self, settings: dict[str, Any] | None = None):
        config.load_dotenv()
        self.settings = settings or config.load_settings()
        s = self.settings
        self.data_dir: Path = config.data_dir(s)
        self.media_root = self.data_dir / "images"
        self.media_root.mkdir(parents=True, exist_ok=True)

        # config providers (re-read on demand so edits on the settings page apply immediately)
        self._ontology = config.load_ontology()
        self._pricing = config.load_pricing()

        self.db = Database(self.data_dir / "freshgrade.db")
        self.bus = EventBus()

        # Member 1
        self.detector = build_detector(s["detector"], config.resolve)
        cal = s["calibration"]
        self.geometry = ArucoGeometryEstimator(self.ontology, cal["aruco_dictionary"], int(cal["marker_id"]), float(cal["marker_size_mm"]))
        # Member 2
        ccfg = s["condition"]
        if ccfg.get("type") == "mock":
            from mocks.mock_modules import MockConditionAnalyzer

            self.condition = MockConditionAnalyzer()
        else:
            clf = (
                try_load_classifier(
                    config.resolve(ccfg["classifier_onnx"]),
                    config.resolve(ccfg["classifier_labels"]),
                    config.resolve(ccfg.get("classifier_meta", "artifacts/condition_classifier_meta.json")),
                )
                if ccfg.get("type") == "onnx"
                else None
            )
            self.condition = ClassicalConditionAnalyzer(self.ontology, clf)
        # Member 3
        self.pricing = PricingEngine(self.pricing_cfg)
        self.grading = RuleBasedGradingEngine(self.ontology, self.pricing)
        self.llm = build_llm_inspector(s["llm"])
        # Member 4
        self.inventory = InventoryService(self.db)
        self.pricing_job = PricingJob(self.db, self.pricing_cfg)
        self.bus.subscribe(INSPECTION_COMPLETED, self.inventory.on_inspection_completed)
        self.inspection = InspectionService(
            self.detector, self.geometry, self.condition, self.llm, self.grading, self.db, self.bus, self.media_root
        )
        log.info(
            "FreshGrade ready: detector=%s condition=%s llm=%s data=%s",
            getattr(self.detector, "name", "?"), getattr(self.condition, "name", "?"), getattr(self.llm, "name", "?"), self.data_dir,
        )

    def ontology(self) -> dict[str, Any]:
        return self._ontology

    def pricing_cfg(self) -> dict[str, Any]:
        return self._pricing

    def reload_ontology(self) -> None:
        self._ontology = config.load_ontology()

    def update_ontology(self, data: dict[str, Any]) -> None:
        config.save_ontology(data)
        self._ontology = data

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "detector": getattr(self.detector, "name", "?"),
            "condition_analyzer": getattr(self.condition, "name", "?"),
            "llm_inspector": getattr(self.llm, "name", "?"),
            "llm_enabled": bool(getattr(self.llm, "enabled", False)),
            "calibration_available": bool(getattr(self.geometry, "aruco_ok", False)),
            "database": str(self.db.path),
            "fruit_types": list(self._ontology["fruit_profiles"].keys()),
        }
