"""Members 1 & 2 — detection, calibration, geometry and defect analysis on synthetic images."""
import cv2
import numpy as np
import pytest

from contracts.schemas import FruitType
from core import config
from defects.analyzer import ClassicalConditionAnalyzer
from vision.detector import ClassicalFruitDetector
from vision.geometry import ArucoGeometryEstimator

ONTO = config.load_ontology()


@pytest.fixture(scope="module")
def det():
    return ClassicalFruitDetector()


@pytest.fixture(scope="module")
def analyzer():
    return ClassicalConditionAnalyzer(lambda: ONTO)


def load(samples, name):
    img = cv2.imread(str(samples / name))
    assert img is not None
    return img


@pytest.mark.parametrize("name,expected", [
    ("apple_healthy.jpg", FruitType.apple), ("orange_healthy.jpg", FruitType.orange),
    ("banana_spots.jpg", FruitType.banana), ("lemon.jpg", FruitType.lemon),
])
def test_classical_detector_finds_one_fruit(samples, det, name, expected):
    fruits = det.detect(load(samples, name))
    assert len(fruits) == 1
    assert fruits[0].fruit_type == expected
    assert fruits[0].mask is not None and fruits[0].mask_area_px > 10000


def test_detector_ignores_calibration_marker_and_finds_three(samples, det):
    assert len(det.detect(load(samples, "mixed_with_card.jpg"))) == 3


def test_empty_background_has_no_fruit(det):
    img = np.full((480, 640, 3), 235, np.uint8)
    assert det.detect(img) == []


def test_calibration_and_size(samples, det):
    img = load(samples, "mixed_with_card.jpg")
    geo = ArucoGeometryEstimator(lambda: ONTO)
    cal = geo.calibrate(img)
    assert cal is not None
    assert cal.mm_per_px == pytest.approx(0.25, rel=0.03)  # samples are drawn at 4 px/mm
    biggest = max(det.detect(img), key=lambda f: f.mask_area_px)
    g = geo.measure(biggest, cal)
    assert g.calibrated and g.diameter_mm == pytest.approx(76, abs=4)  # drawn as a 76 mm apple
    assert 150 < g.est_weight_g < 260


def test_no_card_means_uncalibrated(samples, det):
    img = load(samples, "apple_healthy.jpg")
    geo = ArucoGeometryEstimator(lambda: ONTO)
    assert geo.calibrate(img) is None
    g = geo.measure(det.detect(img)[0], None)
    assert not g.calibrated and g.est_weight_g == ONTO["fruit_profiles"]["apple"]["typical_weight_g"]


def test_defect_percentage_orders_healthy_bruised_rotten(samples, det, analyzer):
    pct = {}
    for name in ("apple_healthy.jpg", "apple_bruised.jpg", "apple_rotten.jpg"):
        img = load(samples, name)
        res = analyzer.analyze(img, det.detect(img)[0])
        pct[name] = res.defect_pct
        assert res.heatmap is not None
    assert pct["apple_healthy.jpg"] < 1.0
    assert pct["apple_healthy.jpg"] < pct["apple_bruised.jpg"] < pct["apple_rotten.jpg"]
    assert pct["apple_rotten.jpg"] > 10


def test_healthy_fruit_has_uniform_colour(samples, det, analyzer):
    img = load(samples, "orange_healthy.jpg")
    res = analyzer.analyze(img, det.detect(img)[0])
    assert res.color_uniformity > 0.8 and res.target_color_pct > 90
