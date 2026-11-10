"""Member 2 — Fruit condition classifier (ONNX runtime).

Trained on real photos (training/train_condition_classifier.py or training/train_condition_yolo.py)
and exported to artifacts/condition_classifier.onnx, with:
    artifacts/condition_classifier_labels.json   e.g. ["damaged", "fresh", "rotten"]
    artifacts/condition_classifier_meta.json     {"size": 192, "normalize": "none" | "imagenet", "resize": "crop" | "stretch"}

Label meaning (by name):
    contains "rot"/"spoil"/"mould"            → rotten (counts as rot and as damage)
    contains "damag"/"bruise"/"spot"/"stale"/
             "normal"/"defect"/"aged"/"bad"   → damaged (visible damage or ageing)
    anything else (e.g. "fresh")              → fresh
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger("freshgrade.classifier")

ROT_WORDS = ("rot", "spoil", "mould", "mold")
DAMAGE_WORDS = ("damag", "bruise", "spot", "stale", "normal", "defect", "aged", "bad", "scab")
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def label_kind(label: str) -> str:
    l = label.lower()
    if any(w in l for w in ROT_WORDS):
        return "rotten"
    if any(w in l for w in DAMAGE_WORDS):
        return "damaged"
    return "fresh"


class OnnxConditionClassifier:
    def __init__(self, model_path: Path, labels_path: Path, meta_path: Optional[Path] = None):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        meta = json.loads(meta_path.read_text()) if meta_path and meta_path.exists() else {}
        shape = self.session.get_inputs()[0].shape
        self.size = int(meta.get("size") or (shape[-1] if isinstance(shape[-1], int) else 224))
        self.normalize = meta.get("normalize", "imagenet")
        self.resize = meta.get("resize", "stretch")
        self.labels: list[str] = json.loads(labels_path.read_text()) if labels_path.exists() else ["fresh", "rotten"]
        self.kinds = [label_kind(l) for l in self.labels]
        log.info("Loaded condition classifier %s labels=%s meta=%s", model_path.name, self.labels, meta)

    def _prep(self, crop_bgr: np.ndarray) -> np.ndarray:
        s = self.size
        if self.resize == "crop":  # shortest side → s, then centre crop (Ultralytics classify transform)
            h, w = crop_bgr.shape[:2]
            k = s / min(h, w)
            img = cv2.resize(crop_bgr, (max(s, round(w * k)), max(s, round(h * k))), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]
            y0, x0 = (h - s) // 2, (w - s) // 2
            img = img[y0:y0 + s, x0:x0 + s]
        else:
            img = cv2.resize(crop_bgr, (s, s), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if self.normalize == "imagenet":
            img = (img - MEAN) / STD
        return np.transpose(img, (2, 0, 1))[None].astype(np.float32)

    def predict(self, crop_bgr: np.ndarray) -> dict[str, float]:
        out = self.session.run(None, {self.input_name: self._prep(crop_bgr)})[0][0].astype(np.float64)
        if out.min() < 0 or abs(out.sum() - 1.0) > 1e-3:  # logits → softmax
            e = np.exp(out - out.max())
            out = e / e.sum()
        return {l: float(p) for l, p in zip(self.labels, out)}

    def condition(self, crop_bgr: np.ndarray) -> tuple[float, float, str, dict[str, float]]:
        """Return (rotten_prob, damage_prob incl. rot, top label, all probabilities)."""
        probs = self.predict(crop_bgr)
        rot = sum(p for l, p in probs.items() if label_kind(l) == "rotten")
        dmg = sum(p for l, p in probs.items() if label_kind(l) in ("damaged", "rotten"))
        top = max(probs, key=probs.get)
        return rot, dmg, top, probs

    # backwards compatible helper
    def rotten_probability(self, crop_bgr: np.ndarray) -> float:
        return self.condition(crop_bgr)[0]


def try_load_classifier(model_path: Path, labels_path: Path, meta_path: Optional[Path] = None) -> Optional[OnnxConditionClassifier]:
    if not model_path.exists():
        log.info("No condition classifier at %s → classical condition estimate only", model_path)
        return None
    try:
        return OnnxConditionClassifier(model_path, labels_path, meta_path)
    except Exception:
        log.exception("Failed to load condition classifier; continuing without it")
        return None
