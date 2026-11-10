"""Member 3 — grading engine unit tests (good, borderline, damaged and rejected produce)."""
import pytest

from contracts.schemas import ConditionResult, DetectedFruit, FruitType, Geometry, LLMOpinion, QualityClass
from core import config
from grading.pricing import PricingEngine
from grading.rules import RuleBasedGradingEngine


@pytest.fixture()
def engine():
    onto = config.load_ontology()
    pr = config.load_pricing()
    return RuleBasedGradingEngine(lambda: onto, PricingEngine(lambda: pr))


def fruit(conf=0.9, ftype=FruitType.apple):
    return DetectedFruit(fruit_id=0, fruit_type=ftype, confidence=conf, bbox=(0, 0, 10, 10))


def geo(calibrated=True, d=75.0):
    if not calibrated:
        return Geometry(calibrated=False, major_axis_px=100, minor_axis_px=100, est_weight_g=180, size_class="unmeasured")
    return Geometry(calibrated=True, major_axis_px=100, minor_axis_px=100, diameter_mm=d, length_mm=d, volume_cm3=220, est_weight_g=176, size_class="medium")


def cond(defect=0.0, unif=0.9, rot=0.01, damage=0.0, label=None, target=95):
    return ConditionResult(defect_pct=defect, anomaly_score=0.1, rotten_prob=rot, color_uniformity=unif, target_color_pct=target,
                           damage_prob=damage, condition_label=label)


def test_perfect_apple_is_extra(engine):
    g = engine.grade(fruit(), geo(), cond(), None)
    assert g.quality_class == QualityClass.extra
    assert g.score >= 90
    assert g.price_per_kg == 2.40
    assert g.item_value == pytest.approx(2.40 * 0.176, abs=0.001)
    assert g.bin_code == "APL-EXTRA"


def test_borderline_defect_goes_to_class_i(engine):
    g = engine.grade(fruit(), geo(), cond(defect=2.0), None)
    assert g.quality_class == QualityClass.class_i
    assert any("Not Extra Class" in r for r in g.reasons)


def test_damaged_apple_is_class_ii(engine):
    assert engine.grade(fruit(), geo(), cond(defect=6.0), None).quality_class == QualityClass.class_ii


def test_heavily_damaged_is_reject(engine):
    assert engine.grade(fruit(), geo(), cond(defect=12.0), None).quality_class == QualityClass.reject


def test_rot_hard_rule(engine):
    g = engine.grade(fruit(), geo(), cond(defect=0.5, rot=0.95), None)
    assert g.quality_class == QualityClass.reject
    assert g.price_per_kg == 0.0 and g.bin_code == "REJ"


def test_small_diameter_drops_class(engine):
    g = engine.grade(fruit(), geo(d=58.0), cond(), None)  # below Extra (65) and I (60) minimum
    assert g.quality_class == QualityClass.class_ii


def test_uncalibrated_ignores_size(engine):
    g = engine.grade(fruit(), geo(calibrated=False), cond(), None)
    assert g.quality_class == QualityClass.extra
    assert any("not measured" in r for r in g.reasons)


def test_llm_agreement_raises_confidence(engine):
    base = engine.grade(fruit(), geo(), cond(), None).confidence
    agree = engine.grade(fruit(), geo(), cond(), LLMOpinion(suggested_class=QualityClass.extra)).confidence
    assert agree > base


def test_llm_strong_disagreement_triggers_review(engine):
    g = engine.grade(fruit(), geo(), cond(), LLMOpinion(suggested_class=QualityClass.reject, defect_types=["rot"]))
    assert g.needs_manual_review
    assert g.quality_class == QualityClass.extra  # local rules still decide the class


def test_low_detection_confidence_triggers_review(engine):
    assert engine.grade(fruit(conf=0.3), geo(), cond(), None).needs_manual_review


def test_every_fruit_profile_grades(engine):
    for ft in FruitType:
        g = engine.grade(fruit(ftype=ft), geo(calibrated=False), cond(defect=1.0), None)
        assert g.quality_class in (QualityClass.extra, QualityClass.class_i)


def test_condition_model_damage_caps_class(engine):
    # few visible defects, but the trained model sees ageing/damage → at most Class II
    g = engine.grade(fruit(), geo(), cond(defect=0.5, damage=0.7, label="damaged"), None)
    assert g.quality_class == QualityClass.class_ii
    assert any("Condition model" in r for r in g.reasons)


def test_condition_model_rot_rejects(engine):
    g = engine.grade(fruit(), geo(), cond(defect=1.0, rot=0.9, damage=0.95, label="rotten"), None)
    assert g.quality_class == QualityClass.reject


def test_damage_ignored_without_model(engine):
    assert engine.grade(fruit(), geo(), cond(damage=0.9, label=None), None).quality_class == QualityClass.extra


def test_overripe_banana_rejected_by_colour(engine):
    # skin mostly brown: almost none of it has the expected yellow colour
    g = engine.grade(fruit(ftype=FruitType.banana), geo(calibrated=False), cond(defect=0.6, target=1), None)
    assert g.quality_class == QualityClass.reject
    assert any("expected colour" in r for r in g.reasons)
