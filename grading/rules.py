"""Member 3 — Ontology-driven grading engine with explainable reasons and LLM fusion.

Decision procedure (all thresholds come from config/ontology.json):
  1. Hard rules  : rot / very large damage → Reject.
  2. Class rules : best class (Extra → I → II) whose defect, uniformity and size limits are all met.
  3. Score 0-100 : weighted defect, condition, colour and size components.
  4. Fusion      : vision-LLM opinion raises or lowers confidence; a large disagreement
                   or LLM-reported rot sends the fruit to manual review.
  5. Price       : €/kg for the class × estimated weight (config/pricing.yaml).
"""
from __future__ import annotations

from typing import Callable, Optional

from contracts.schemas import (
    CLASS_ORDER,
    ConditionResult,
    DetectedFruit,
    Geometry,
    GradeResult,
    LLMOpinion,
    QualityClass,
    class_rank,
)
from grading.pricing import PricingEngine

CLASS_LABEL = {"Extra": "Extra Class", "I": "Class I", "II": "Class II", "Reject": "Reject"}


class RuleBasedGradingEngine:
    def __init__(self, ontology_provider: Callable[[], dict], pricing: PricingEngine):
        self.ontology_provider = ontology_provider
        self.pricing = pricing

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _class_failures(cls_rules: dict, geometry: Geometry, condition: ConditionResult, label: str) -> list[str]:
        fails = []
        if condition.defect_pct > cls_rules["max_defect_pct"]:
            fails.append(
                f"Visible defects cover {condition.defect_pct:.1f}% of the surface ({label} allows up to {cls_rules['max_defect_pct']:.1f}%)."
            )
        if condition.color_uniformity < cls_rules["min_uniformity"]:
            fails.append(
                f"Colour uniformity {condition.color_uniformity:.2f} is below the {label} minimum of {cls_rules['min_uniformity']:.2f}."
            )
        max_dmg = cls_rules.get("max_damage_prob", 1.0)
        if condition.condition_label is not None and condition.damage_prob > max_dmg:
            fails.append(
                f"Condition model sees visible damage or ageing ({condition.damage_prob:.0%}; {label} allows up to {max_dmg:.0%})."
            )
        min_tc = cls_rules.get("min_target_color_pct", 0) or 0
        if min_tc and condition.target_color_pct < min_tc:
            fails.append(
                f"Only {condition.target_color_pct:.0f}% of the skin has the expected colour ({label} needs at least {min_tc:.0f}%)."
            )
        min_d = cls_rules.get("min_diameter_mm", 0) or 0
        if geometry.calibrated and geometry.diameter_mm is not None and min_d and geometry.diameter_mm < min_d:
            fails.append(f"Diameter {geometry.diameter_mm:.0f} mm is below the {label} minimum of {min_d:.0f} mm.")
        return fails

    @staticmethod
    def _score(geometry: Geometry, condition: ConditionResult, weights: dict, ii_max: float) -> int:
        defect_c = max(0.0, 1.0 - condition.defect_pct / max(0.1, ii_max * 1.5))
        cond_c = 1.0 - max(condition.rotten_prob, 0.6 * condition.damage_prob)
        color_c = 0.7 * condition.color_uniformity + 0.3 * (condition.target_color_pct / 100.0)
        size_c = {"large": 1.0, "medium": 1.0, "small": 0.7}.get(geometry.size_class, 0.85)
        s = (
            weights["defect"] * defect_c
            + weights["condition"] * cond_c
            + weights["color"] * color_c
            + weights["size"] * size_c
        )
        return int(round(100 * max(0.0, min(1.0, s))))

    # ------------------------------------------------------------------ interface
    def grade(
        self,
        fruit: DetectedFruit,
        geometry: Geometry,
        condition: ConditionResult,
        llm: Optional[LLMOpinion],
    ) -> GradeResult:
        onto = self.ontology_provider()
        profile = onto["fruit_profiles"][fruit.fruit_type.value]
        hard = onto["hard_rules"]
        reasons: list[str] = []
        review = False

        # 1) hard rules
        quality: Optional[QualityClass] = None
        if condition.rotten_prob >= hard["reject_if_rotten_prob_at_least"]:
            quality = QualityClass.reject
            reasons.append(f"Signs of rot/decay detected (probability {condition.rotten_prob:.0%}). Product is out of grade.")
        elif condition.defect_pct > hard["reject_if_defect_pct_above"]:
            quality = QualityClass.reject
            reasons.append(f"Damage covers {condition.defect_pct:.1f}% of the visible surface, beyond any sellable class.")

        # The trained condition model is far more reliable than the pixel-based defect map. When it sees
        # healthy skin, natural marks counted by the defect map weigh less (and vice versa).
        if condition.condition_label is not None:
            w = hard.get("defect_weight_when_model_sees_fresh", 0.3)
            eff = condition.defect_pct * (w + (1 - w) * condition.damage_prob)
            if eff < condition.defect_pct - 0.05:
                reasons.append(
                    f"Defect map marks {condition.defect_pct:.1f}% of the surface; the condition model sees healthy skin "
                    f"({1 - condition.damage_prob:.0%}), so {eff:.1f}% is used for grading."
                )
            condition = condition.model_copy(update={"defect_pct": round(eff, 2)})

        # 2) class rules
        if quality is None:
            for cls in (QualityClass.extra, QualityClass.class_i, QualityClass.class_ii):
                rules = profile["classes"][cls.value]
                fails = self._class_failures(rules, geometry, condition, CLASS_LABEL[cls.value])
                if not fails:
                    quality = cls
                    break
                if cls in (QualityClass.extra, QualityClass.class_i):
                    reasons.extend(f"Not {CLASS_LABEL[cls.value]}: {f}" for f in fails[:2])
                else:
                    reasons.extend(fails)
            if quality is None:
                quality = QualityClass.reject
                reasons.append("Does not meet the minimum requirements of Class II.")
            elif quality == QualityClass.extra:
                reasons.append(
                    f"Practically free from defects ({condition.defect_pct:.1f}%) with even colour (uniformity {condition.color_uniformity:.2f})."
                )
            else:
                reasons.append(f"Meets all {CLASS_LABEL[quality.value]} requirements.")

        if not geometry.calibrated:
            reasons.append("Size not measured (no calibration card in view); weight uses the typical value for this fruit.")
        else:
            reasons.append(f"Measured diameter {geometry.diameter_mm:.0f} mm, est. weight {geometry.est_weight_g:.0f} g ({geometry.size_class}).")

        # 3) score + base confidence
        score = self._score(geometry, condition, onto["score_weights"], profile["classes"]["II"]["max_defect_pct"])
        confidence = 0.45 + 0.5 * fruit.confidence
        if fruit.confidence < hard["manual_review_if_detection_conf_below"]:
            review = True
            reasons.append(f"Low detection confidence ({fruit.confidence:.0%}); please verify manually.")

        # 4) LLM fusion
        if llm is not None:
            gap = abs(class_rank(llm.suggested_class) - class_rank(quality))
            if gap == 0:
                confidence += 0.12
                reasons.append(f"AI inspector agrees ({CLASS_LABEL[llm.suggested_class.value]}).")
            elif gap >= hard["manual_review_if_llm_class_gap_at_least"]:
                confidence -= 0.25
                review = True
                reasons.append(
                    f"AI inspector suggests {CLASS_LABEL[llm.suggested_class.value]} — strong disagreement, sent to manual review."
                )
            else:
                confidence -= 0.08
                reasons.append(f"AI inspector suggests {CLASS_LABEL[llm.suggested_class.value]} (one class apart).")
            if hard.get("llm_reported_rot_forces_review") and "rot" in llm.defect_types and quality != QualityClass.reject:
                review = True
                reasons.append("AI inspector reports possible rot — manual check required.")

        # 5) price
        price_kg, value = self.pricing.price(fruit.fruit_type.value, quality.value, geometry.est_weight_g)
        return GradeResult(
            quality_class=quality,
            score=score if quality != QualityClass.reject else min(score, 35),
            confidence=round(max(0.05, min(0.99, confidence)), 2),
            reasons=reasons,
            needs_manual_review=review,
            price_per_kg=price_kg,
            item_value=value,
            currency=self.pricing.currency,
            bin_code=self.pricing.bin_code(fruit.fruit_type.value, quality.value),
        )


__all__ = ["RuleBasedGradingEngine", "CLASS_ORDER"]
