"""Member 2 — Colour uniformity and target-colour (ripeness) analysis."""
from __future__ import annotations

import numpy as np

from core.imaging import hue_in_ranges


def color_metrics(
    lab: np.ndarray, hsv: np.ndarray, region: np.ndarray, target_hue_ranges: list[list[float]]
) -> tuple[float, float]:
    """Return (uniformity 0..1, target_colour_pct 0..100) computed on `region` pixels.

    Uniformity is based on the circular spread of the Lab hue angle h = atan2(b*, a*), which is
    largely independent of shading (a darker side of the same apple keeps the same hue):
        uniformity = exp(-circular_std(h) / 25°)
    An even colour gives ~1.0; blotchy or two-tone fruit gives lower values.
    """
    sel = region > 0
    sat = hsv[..., 1]
    sel_col = sel & (sat > 45)
    if np.count_nonzero(sel_col) < 30:
        sel_col = sel
    n = int(np.count_nonzero(sel_col))
    if n < 30:
        return 0.0, 0.0
    a = lab[..., 1][sel_col].astype(np.float32) - 128.0
    b = lab[..., 2][sel_col].astype(np.float32) - 128.0
    ang = np.arctan2(b, a)
    R = float(np.hypot(np.cos(ang).mean(), np.sin(ang).mean()))
    circ_std_deg = float(np.rad2deg(np.sqrt(max(0.0, -2.0 * np.log(max(R, 1e-6))))))
    uniformity = float(np.exp(-circ_std_deg / 25.0))

    hue = hsv[..., 0][sel_col].astype(np.float32)
    in_target = hue_in_ranges(hue, target_hue_ranges)
    target_pct = 100.0 * float(np.count_nonzero(in_target)) / n
    return round(uniformity, 3), round(target_pct, 1)
