"""Member 2 — Surface-defect analysis (ConditionAnalyzer implementation).

Pipeline per fruit:
 1. Erode the fruit mask (removes dark rim/shadow at the edge).
 2. Reference colour = median Lab of the fruit surface.
 3. Per-pixel anomaly score from darkening (ΔL) and chromatic shift (Δab) vs. the reference,
    ignoring specular highlights.
 4. Threshold + clean-up → defect mask, defect % of visible surface, defect regions.
 5. Colour uniformity + target-colour % on the healthy surface.
 6. Rotten probability: heuristic from the defect map, fused with the trained ONNX classifier
    when available (artifacts/condition_classifier.onnx).

The anomaly map is also rendered as a heat-map image for the UI and the reports.
Member 2 can replace step 3/4 by a trained PatchCore model (training/train_patchcore.py)
without changing the output contract.
"""
from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

from contracts.schemas import ConditionResult, DefectRegion, DetectedFruit
from defects.classifier import OnnxConditionClassifier
from defects.color import color_metrics
from core.imaging import crop_with_margin


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class ClassicalConditionAnalyzer:
    name = "classical"

    # Tunable parameters (selected with tools/tune_defects.py on labelled internet images)
    PARAMS = {
        "local_sigma_div": 8.0,    # neighbourhood size for local darkening = fruit span / this
        "local_lo": 0.04,          # relative darkening where the local cue starts
        "gate_local_by_shadow": True,
        "score_threshold": 0.26,   # anomaly score above which a pixel is a defect
        "dull_margin": 4.0,        # saturation drop that marks a pixel as "duller" than healthy skin
        "shadow_s_margin": 0.0,    # same-hue pixels within this saturation margin are treated as shadow
        "hue_lo": 7.0,             # hue shift (degrees) where the hue cue starts
        "brown_dark_ratio": 0.92,  # a "browned" pixel must be darker than this share of the skin brightness
        "global_lo": 0.22,         # relative darkening where the large-dark-area cue starts
    }

    def __init__(self, ontology_provider, classifier: Optional[OnnxConditionClassifier] = None):
        self.ontology_provider = ontology_provider
        self.classifier = classifier
        if classifier is not None:
            self.name = "classical+onnx-classifier"

    # ------------------------------------------------------------------ core maps
    @classmethod
    def anomaly_map(cls, image_bgr: np.ndarray, mask: np.ndarray, params: Optional[dict] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (score_map 0..1, surface_mask, lab, hsv)."""
        P = {**cls.PARAMS, **(params or {})}
        blur = cv2.GaussianBlur(image_bgr, (3, 3), 0)
        lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB).astype(np.float32)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        ys, xs = np.nonzero(mask)
        span = max(1, min(xs.max() - xs.min(), ys.max() - ys.min())) if len(xs) else 1
        # drop background pixels that a coarse mask may include (white / grey, unsaturated, bright)
        bg_like = (hsv[..., 1] < 35) & (hsv[..., 2] > 170)
        refined = mask.copy()
        refined[bg_like] = 0
        if np.count_nonzero(refined) > 0.6 * np.count_nonzero(mask):
            mask = refined
        r = max(2, int(span * 0.05))
        surface = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        if np.count_nonzero(surface) < 100:
            surface = mask.copy()
        sel = surface > 0
        # rim weight: the outer ring of a round fruit is naturally darker (curvature), so it needs
        # stronger evidence before it counts as a defect
        dist = cv2.distanceTransform(surface, cv2.DIST_L2, 5)
        rim_w = np.clip(dist / max(1.0, 0.18 * span), 0.2, 1.0)

        L = lab[..., 0]
        S = hsv[..., 1].astype(np.float32)
        H = hsv[..., 0].astype(np.float32) * 2.0  # degrees 0..360

        ref_L = float(np.median(L[sel]))
        ref_S = float(np.median(S[sel]))
        hs = H[sel & (S > 40)]
        if len(hs) < 30:
            hs = H[sel]
        # dominant surface hue = mode of a smoothed circular histogram (robust to large defects)
        hist, _ = np.histogram(hs, bins=72, range=(0, 360))
        hist = np.convolve(np.concatenate([hist[-1:], hist, hist[:1]]), np.ones(3) / 3, mode="same")[1:-1]
        peak = float(np.argmax(hist) * 5 + 2.5)
        near = hs[np.minimum(np.abs(hs - peak), 360 - np.abs(hs - peak)) <= 12]
        ang = np.deg2rad(near if len(near) else hs)
        ref_H = float(np.rad2deg(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) % 360)

        # specular highlights: brighter and whiter than the surface → never a defect
        specular = (L > ref_L + 12) & (S < ref_S * 0.75)

        # 1) LOCAL darkening: compare each pixel with the smoothed brightness of its neighbourhood
        #    (normalised convolution inside the mask) → removes the natural shading of a round fruit
        sigma = max(3.0, span / P["local_sigma_div"])
        m = sel.astype(np.float32)
        num = cv2.GaussianBlur(L * m, (0, 0), sigma)
        den = cv2.GaussianBlur(m, (0, 0), sigma) + 1e-6
        local_L = num / den
        # relative darkening (works for dark-red and light-yellow fruit alike)
        d_local = (local_L - L) / (local_L + 1.0)
        # 2) saturation drop (dull / brownish areas); shading barely changes HSV saturation
        d_sat = np.where(L <= ref_L + 10, ref_S - S, 0.0)
        # 3) hue shift vs. the surface colour (e.g. red apple → brown bruise). Only counts where the
        #    pixel is also darker or duller, so natural lighter streaks (bi-colour apples) are ignored.
        dh = np.abs(H - ref_H)
        dh = np.minimum(dh, 360 - dh)
        dh_raw = dh.copy()
        d_global = (ref_L - L) / (ref_L + 1.0)
        # Decay and bruises are DULLER (less saturated) than healthy skin, whereas natural colour
        # variation (deep red vs. yellow side of a bi-colour apple) keeps or raises saturation.
        duller = d_sat > P["dull_margin"]
        # Decay can also stay saturated but turn BROWN: hue moves into the orange-brown band and the
        # skin gets darker (e.g. a shrivelled rotten patch on a red apple). Yellow streaks are lighter.
        brown_band = (H >= 14) & (H <= 50) & (L < ref_L * P["brown_dark_ratio"])
        browned = brown_band & (dh_raw > P["hue_lo"] + 4)
        browner = (duller | browned) & (dh > P["hue_lo"] + 4)
        dh = np.where(duller | browned, dh, 0.0)
        # large dark areas only count when they are also brown-shifted (decay), not plain shadow
        d_global = np.where(browner, d_global, 0.0)
        # shadow-like pixels: same hue and NOT duller than healthy skin → shading, never a defect
        shadow_like = (dh_raw < P["hue_lo"]) & (S >= ref_S - P["shadow_s_margin"])

        local_term = np.clip((d_local - P["local_lo"]) / 0.25, 0, 1)
        if P["gate_local_by_shadow"]:
            local_term = np.where(shadow_like, 0.0, local_term)
        other = (
            0.20 * np.clip((d_sat - 30.0) / 70.0, 0, 1)
            + 0.25 * np.clip((dh - P["hue_lo"]) / 20.0, 0, 1)
            + 0.20 * np.clip((d_global - P["global_lo"]) / 0.30, 0, 1)
        )
        other = np.where(shadow_like, 0.0, other)
        score = 0.35 * local_term + other
        # two or more independent cues at once → strong defect evidence
        cues = (
            (local_term > 0.15).astype(np.uint8)
            + (d_sat > 45).astype(np.uint8)
            + (dh > 14).astype(np.uint8)
            + (d_global > P["global_lo"] + 0.06).astype(np.uint8)
        )
        score = np.where(cues >= 2, np.clip(score * 1.6 + 0.12, 0, 1), score)
        score = score * rim_w
        score[~sel | specular] = 0.0
        return score.astype(np.float32), surface, lab, hsv

    @staticmethod
    def roughness(lab: np.ndarray, surface: np.ndarray) -> float:
        """Skin roughness 0..1 from fine-scale Laplacian energy (scab, wrinkles, cracks raise it)."""
        L = cv2.GaussianBlur(lab[..., 0], (3, 3), 0)
        lap = np.abs(cv2.Laplacian(L, cv2.CV_32F))
        sel = surface > 0
        if np.count_nonzero(sel) < 50:
            return 0.0
        tex = float(lap[sel].mean()) / (float(L[sel].mean()) + 1.0) * 100.0
        return float(np.clip((tex - 0.8) / 1.6, 0.0, 1.0))

    @staticmethod
    def heatmap_image(image_bgr: np.ndarray, score: np.ndarray, defect_mask: np.ndarray, bbox) -> np.ndarray:
        heat = cv2.applyColorMap((score * 255).astype(np.uint8), cv2.COLORMAP_JET)
        blend = cv2.addWeighted(image_bgr, 0.55, heat, 0.45, 0)
        contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blend, contours, -1, (255, 255, 255), 2)
        return crop_with_margin(blend, bbox, 0.1)

    # ------------------------------------------------------------------ interface
    def analyze(self, image_bgr: np.ndarray, fruit: DetectedFruit) -> ConditionResult:
        mask = fruit.mask
        profile = self.ontology_provider()["fruit_profiles"][fruit.fruit_type.value]
        score, surface, lab, hsv = self.anomaly_map(image_bgr, mask)
        n_surface = max(1, int(np.count_nonzero(surface)))

        defect = (score > self.PARAMS["score_threshold"]).astype(np.uint8) * 255
        defect = cv2.morphologyEx(defect, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        defect = cv2.morphologyEx(defect, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        defect[surface == 0] = 0

        num, labels, stats, _ = cv2.connectedComponentsWithStats(defect, connectivity=8)
        min_region = max(12, int(0.0015 * n_surface))
        regions: list[DefectRegion] = []
        clean = np.zeros_like(defect)
        for i in range(1, num):
            x, y, w, h, area = stats[i]
            if area < min_region:
                continue
            clean[labels == i] = 255
            regions.append(
                DefectRegion(area_px=int(area), area_pct=round(100.0 * area / n_surface, 2), bbox=(int(x), int(y), int(x + w), int(y + h)))
            )
        regions.sort(key=lambda r: r.area_px, reverse=True)
        defect_pct = 100.0 * float(np.count_nonzero(clean)) / n_surface

        healthy = (surface > 0) & (clean == 0)
        uniformity, target_pct = color_metrics(lab, hsv, healthy.astype(np.uint8), profile["target_hue_ranges"])

        mean_score = float(score[clean > 0].mean()) if np.count_nonzero(clean) else 0.0
        rough = self.roughness(lab, surface)
        anomaly_score = 1.0 - math.exp(-defect_pct / 6.0) * (1.0 - 0.5 * rough)
        heuristic_rot = 0.7 * _sigmoid((defect_pct - 14.0) / 3.5) * (0.6 + 0.4 * mean_score) + 0.3 * rough * min(1.0, defect_pct / 5.0)
        rotten_prob = heuristic_rot
        damage_prob, label, probs = 0.0, None, {}
        if self.classifier is not None:
            # the trained model sees the whole fruit (as in training: box + 6 % margin) and is trusted most
            crop = crop_with_margin(image_bgr, fruit.bbox, 0.06)
            clf_rot, damage_prob, label, probs = self.classifier.condition(crop)
            rotten_prob = max(0.9 * clf_rot + 0.1 * heuristic_rot, heuristic_rot)

        return ConditionResult(
            damage_prob=round(float(np.clip(damage_prob, 0, 1)), 3),
            condition_label=label,
            condition_probs={k: round(v, 3) for k, v in probs.items()},
            defect_pct=round(min(100.0, defect_pct), 2),
            defects=regions[:20],
            anomaly_score=round(float(np.clip(anomaly_score, 0, 1)), 3),
            rotten_prob=round(float(np.clip(rotten_prob, 0, 1)), 3),
            color_uniformity=uniformity,
            target_color_pct=target_pct,
            analyzer=self.name,
            defect_mask=clean,
            heatmap=self.heatmap_image(image_bgr, score, clean, fruit.bbox),
        )
