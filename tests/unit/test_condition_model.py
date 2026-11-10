"""Member 2 — the trained condition model (artifacts/condition_classifier.onnx) on held-out REAL photos."""
from pathlib import Path

import cv2
import pytest

from core import config
from defects.classifier import try_load_classifier

ART = config.ROOT / "artifacts"
FIX = Path(__file__).resolve().parent.parent / "fixtures" / "real"

clf = try_load_classifier(ART / "condition_classifier.onnx", ART / "condition_classifier_labels.json", ART / "condition_classifier_meta.json")
pytestmark = pytest.mark.skipif(clf is None, reason="no trained condition model in artifacts/")


def _scores(label):
    out = []
    for f in sorted((FIX / label).glob("*.jpg")):
        rot, dmg, top, _ = clf.condition(cv2.imread(str(f)))
        out.append((f.name, rot, dmg, top))
    return out


def test_rotten_fruit_is_flagged():
    res = _scores("rotten")
    assert res
    assert sum(1 for _, rot, _, _ in res if rot >= 0.5) >= len(res) - 1, res


def test_fresh_fruit_is_not_called_rotten():
    res = _scores("fresh")
    assert res
    assert sum(1 for _, rot, _, _ in res if rot < 0.5) >= len(res) - 1, res
