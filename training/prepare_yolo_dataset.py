"""Member 1 — Merge several internet segmentation datasets into ONE YOLO-seg dataset.

Typical sources: Roboflow Universe exports in "YOLOv8 / YOLO11 segmentation" format, e.g. the
"fruits segmentations" dataset (Apple, Orange, Banana, Tomato, Lemon). Each export folder has
data.yaml + train/valid/test folders with images/ and labels/.

Class names are mapped to our five classes (apple, tomato, orange, banana, lemon) by name, so
"Apple", "red_apple", "rotten apple" all become `apple`. Unknown classes are dropped.

Usage:
    python training/prepare_yolo_dataset.py --sources exports/fruits-seg exports/tomato-seg --out datasets/freshgrade_seg
Then train with training/train_yolo_seg.py.
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import yaml

TARGET = ["apple", "tomato", "orange", "banana", "lemon"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def map_name(name: str) -> int | None:
    n = name.lower()
    for i, t in enumerate(TARGET):
        if t in n:
            return i
    return None


def find_split_dirs(src: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for split in ("train", "valid", "val", "test"):
        img_dir, lbl_dir = src / split / "images", src / split / "labels"
        if img_dir.exists() and lbl_dir.exists():
            pairs.append((img_dir, lbl_dir))
    if not pairs and (src / "images").exists():
        pairs.append((src / "images", src / "labels"))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    samples: list[tuple[Path, list[str]]] = []
    for src in args.sources:
        names = yaml.safe_load((src / "data.yaml").read_text())["names"]
        names = names if isinstance(names, list) else [names[k] for k in sorted(names)]
        remap = {i: map_name(n) for i, n in enumerate(names)}
        print(f"{src}: classes {names} → {[TARGET[v] if v is not None else 'drop' for v in remap.values()]}")
        for img_dir, lbl_dir in find_split_dirs(src):
            for img in img_dir.iterdir():
                if img.suffix.lower() not in IMG_EXT:
                    continue
                lbl = lbl_dir / (img.stem + ".txt")
                if not lbl.exists():
                    continue
                lines = []
                for line in lbl.read_text().splitlines():
                    parts = line.split()
                    if len(parts) < 7:  # segmentation needs class + ≥3 points
                        continue
                    new = remap.get(int(parts[0]))
                    if new is not None:
                        lines.append(" ".join([str(new)] + parts[1:]))
                if lines:
                    samples.append((img, lines))

    random.Random(args.seed).shuffle(samples)
    n = len(samples)
    n_test, n_val = int(n * args.test_frac), int(n * args.val_frac)
    splits = {"test": samples[:n_test], "val": samples[n_test:n_test + n_val], "train": samples[n_test + n_val:]}
    for split, items in splits.items():
        (args.out / split / "images").mkdir(parents=True, exist_ok=True)
        (args.out / split / "labels").mkdir(parents=True, exist_ok=True)
        for k, (img, lines) in enumerate(items):
            stem = f"{split}_{k:06d}"
            shutil.copy2(img, args.out / split / "images" / (stem + img.suffix.lower()))
            (args.out / split / "labels" / (stem + ".txt")).write_text("\n".join(lines) + "\n")
        print(f"{split}: {len(items)} images")
    data = {"path": str(args.out.resolve()), "train": "train/images", "val": "val/images", "test": "test/images",
            "names": {i: t for i, t in enumerate(TARGET)}}
    (args.out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    print("wrote", args.out / "data.yaml")


if __name__ == "__main__":
    main()
