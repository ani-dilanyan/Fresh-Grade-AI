"""Member 2 — Tune / evaluate the classical defect analyser on labelled images.

Folder layout (any internet dataset sorted into folders works):
    <data>/fresh/*.jpg            healthy fruit   (name given with --good)
    <data>/<any other folder>/    defective fruit (bruise, rot, scab, ...)

The script detects the fruit (same detector as the app), measures the defect % for every
parameter combination in a small grid and reports the balanced accuracy of the rule
"defective if defect % > --limit" (default 3 %, the Class I limit). The best parameters can be
copied into ClassicalConditionAnalyzer.PARAMS. Results also make a good table for the report.

Usage:
    python tools/tune_defects.py --data path/to/apple_images --good fresh --max-per-class 40
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import config  # noqa: E402
from defects.analyzer import ClassicalConditionAnalyzer  # noqa: E402
from vision.detector import build_detector  # noqa: E402

GRID = {
    "local_sigma_div": [8.0, 12.0],
    "local_lo": [0.04, 0.08],
    "gate_local_by_shadow": [True],
    "score_threshold": [0.26, 0.34],
    "dull_margin": [4.0, 10.0],
    "shadow_s_margin": [0.0, 8.0],
    "hue_lo": [7.0, 11.0],
}


def load_items(data: Path, good: str, max_per_class: int, detector) -> list[tuple[np.ndarray, np.ndarray, int, str]]:
    items = []
    for folder in sorted(p for p in data.iterdir() if p.is_dir()):
        label = 0 if folder.name == good else 1
        files = sorted([*folder.glob("*.jpg"), *folder.glob("*.jpeg"), *folder.glob("*.png")])[:max_per_class]
        for f in files:
            img = cv2.imread(str(f))
            if img is None:
                continue
            img = cv2.resize(img, (512, int(512 * img.shape[0] / img.shape[1])))
            fruits = detector.detect(img)
            if not fruits:
                continue
            fr = max(fruits, key=lambda x: x.mask_area_px)
            items.append((img, fr.mask, label, folder.name))
    return items


def defect_pct(img, mask, params) -> float:
    score, surface, _, _ = ClassicalConditionAnalyzer.anomaly_map(img, mask, params)
    d = (score > params["score_threshold"]).astype(np.uint8)
    d = cv2.morphologyEx(d, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return 100.0 * float(np.count_nonzero(d & (surface > 0))) / max(1, int(np.count_nonzero(surface)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--good", default="fresh")
    ap.add_argument("--limit", type=float, default=2.0)
    ap.add_argument("--max-per-class", type=int, default=40)
    args = ap.parse_args()

    settings = config.load_settings()
    detector = build_detector(settings["detector"], config.resolve)
    items = load_items(args.data, args.good, args.max_per_class, detector)
    n_good = sum(1 for i in items if i[2] == 0)
    print(f"{len(items)} images with a detected fruit ({n_good} good, {len(items) - n_good} defective)")
    if n_good == 0 or n_good == len(items):
        sys.exit("Need both good and defective images.")

    results = []
    keys = list(GRID)
    for values in itertools.product(*GRID.values()):
        params = dict(zip(keys, values))
        pcts = np.array([defect_pct(img, m, params) for img, m, _, _ in items])
        labels = np.array([lab for _, _, lab, _ in items])
        pred = pcts > args.limit
        tnr = float(np.mean(~pred[labels == 0]))
        tpr = float(np.mean(pred[labels == 1]))
        results.append(((tnr + tpr) / 2, tnr, tpr, params))
    results.sort(key=lambda r: r[0], reverse=True)
    print("\nTop parameter sets (balanced accuracy, specificity on good fruit, sensitivity on defects):")
    for bal, tnr, tpr, p in results[:8]:
        print(f"  bal={bal:.2f}  good-ok={tnr:.2f}  defect-found={tpr:.2f}  {p}")
    print("\nCurrent PARAMS:", ClassicalConditionAnalyzer.PARAMS)


if __name__ == "__main__":
    main()
