"""Evaluate the WHOLE grading pipeline on labelled real photos.

Folder layout (as produced for classifier training):
    <data>/fresh/*.jpg    <data>/damaged/*.jpg    <data>/rotten/*.jpg
File names may start with a group prefix such as "ffd_banana_..." or "hdf_apple_..." to get
per-fruit results.

For every image the app's detector + condition analyser + grading engine run exactly as in the
web app. The report shows, per true condition, how the fruits were graded, plus three headline
numbers for the report:
    fresh accepted  = fresh fruit graded Extra or Class I
    damaged caught  = damaged fruit graded Class II or Reject
    rotten rejected = rotten fruit graded Reject

Usage:
    python tools/evaluate_grading.py --data path/to/val --max-per-class 300
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.container import Container  # noqa: E402


def group_of(name: str) -> str:
    parts = name.split("_")
    return f"{parts[0]}_{parts[1]}" if len(parts) > 2 else "all"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--max-per-class", type=int, default=300)
    ap.add_argument("--json", type=Path, help="optional path to save the results")
    args = ap.parse_args()

    c = Container()
    det, geo, cond, grader = c.detector, c.geometry, c.condition, c.grading
    print(f"detector={getattr(det, 'name', '?')} condition={getattr(cond, 'name', '?')}")

    table: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    per_group: dict[tuple[str, str], collections.Counter] = collections.defaultdict(collections.Counter)
    for label_dir in sorted(p for p in args.data.iterdir() if p.is_dir()):
        truth = label_dir.name
        files = sorted(label_dir.glob("*.jpg"))
        random.Random(0).shuffle(files)
        files = files[: args.max_per_class]
        for f in files:
            img = cv2.imread(str(f))
            if img is None:
                continue
            fruits = det.detect(img)
            if not fruits:
                table[truth]["no fruit"] += 1
                per_group[(group_of(f.name), truth)]["no fruit"] += 1
                continue
            fr = max(fruits, key=lambda x: x.mask_area_px)
            g = grader.grade(fr, geo.measure(fr, None), cond.analyze(img, fr), None)
            table[truth][g.quality_class.value] += 1
            per_group[(group_of(f.name), truth)][g.quality_class.value] += 1

    def pct(cnt: collections.Counter, keys) -> float:
        n = sum(cnt.values())
        return 100.0 * sum(cnt[k] for k in keys) / n if n else 0.0

    order = ["Extra", "I", "II", "Reject", "no fruit"]
    print("\nTrue condition → graded as (counts)")
    for truth, cnt in table.items():
        print(f"  {truth:8s} n={sum(cnt.values()):4d}  " + "  ".join(f"{k}:{cnt[k]}" for k in order))
    summary = {
        "fresh_accepted_pct": round(pct(table["fresh"], ["Extra", "I"]), 1),
        "damaged_caught_pct": round(pct(table["damaged"], ["II", "Reject"]), 1),
        "rotten_rejected_pct": round(pct(table["rotten"], ["Reject"]), 1),
        "rotten_not_sold_as_good_pct": round(pct(table["rotten"], ["II", "Reject"]), 1),
    }
    print("\nPer fruit:")
    for (grp, truth), cnt in sorted(per_group.items()):
        print(f"  {grp:14s} {truth:8s} n={sum(cnt.values()):4d}  " + "  ".join(f"{k}:{cnt[k]}" for k in order))
    print("\nSummary:", json.dumps(summary))
    if args.json:
        args.json.write_text(json.dumps({"summary": summary, "table": {k: dict(v) for k, v in table.items()}}, indent=2))


if __name__ == "__main__":
    main()
