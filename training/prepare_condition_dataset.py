"""Member 2 — Build the fresh / damaged / rotten training set from public real-photo datasets.

Sources (both free to download, no account needed):
  1. Healthy-Defective-Fruits (GitHub: luischuquim/Healthy-Defective-Fruits) — real apple and mango
     photos: fresh, bruise, scab / black spot, rot.
       git clone --depth 1 --filter=blob:none --sparse https://github.com/luischuquim/healthy-defective-fruits hdf
       cd hdf && git sparse-checkout set apple_images/apple_real_images mango_images/mango_real_images
  2. Fruit-freshness-detection-dataset (GitHub: zijianchen98, Apache-2.0) — 3,202 real photos of apples,
     bananas and oranges with boxes labelled fresh / normal (ageing) / rotten.
       git clone --depth 1 --filter=blob:none --sparse https://github.com/zijianchen98/Fruit-freshness-detection-dataset ffd
       cd ffd && git sparse-checkout set Annotations JPEGImages

Label mapping:
  fresh                                  → fresh
  bruise, scab, black spot, "normal"     → damaged   (visible damage or ageing)
  rot, rotten                            → rotten

Usage:
  python training/prepare_condition_dataset.py --hdf path/to/hdf --ffd path/to/ffd --out datasets/condition
Output: <out>/train/<label>/*.jpg and <out>/val/<label>/*.jpg (1 in 7 images held out), file names
prefixed with their source and fruit (e.g. ffd_banana_123.jpg) for per-fruit evaluation.
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2

HDF_MAP = {"fresh": "fresh", "bruise_defect": "damaged", "scab_defect": "damaged", "black_spot_defect": "damaged", "rot_defect": "rotten"}
FFD_MAP = {"fresh": "fresh", "normal": "damaged", "rotten": "rotten"}


def collect(hdf: Path | None, ffd: Path | None) -> list[tuple[str, list[int] | None, str, str]]:
    items = []
    if hdf:
        for fruit in ("apple", "mango"):
            for cat, lab in HDF_MAP.items():
                for f in glob.glob(str(hdf / f"{fruit}_images/{fruit}_real_images/{cat}/*")):
                    items.append((f, None, lab, f"hdf_{fruit}"))
    if ffd:
        for x in glob.glob(str(ffd / "Annotations/*.xml")):
            root = ET.parse(x).getroot()
            img = ffd / "JPEGImages" / (Path(x).stem + ".jpg")
            if not img.exists():
                continue
            for o in root.findall("object"):
                level, fruit = o.find("name").text.split("_")
                b = o.find("bndbox")
                box = [int(float(b.find(k).text)) for k in ("xmin", "ymin", "xmax", "ymax")]
                items.append((str(img), box, FFD_MAP[level], f"ffd_{fruit}"))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf", type=Path)
    ap.add_argument("--ffd", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    items = collect(args.hdf, args.ffd)
    random.Random(args.seed).shuffle(items)
    counts: dict[tuple[str, str], int] = {}
    for i, (path, box, lab, grp) in enumerate(items):
        img = cv2.imread(path)
        if img is None:
            continue
        if box:  # crop the annotated fruit with a 6 % margin (the app crops the same way)
            h, w = img.shape[:2]
            x1, y1, x2, y2 = box
            mx, my = int((x2 - x1) * 0.06), int((y2 - y1) * 0.06)
            img = img[max(0, y1 - my):min(h, y2 + my), max(0, x1 - mx):min(w, x2 + mx)]
            if img.size == 0 or min(img.shape[:2]) < 24:
                continue
        s = 256 / max(img.shape[:2])
        if s < 1:
            img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
        split = "val" if i % 7 == 0 else "train"
        os.makedirs(args.out / split / lab, exist_ok=True)
        cv2.imwrite(str(args.out / split / lab / f"{grp}_{i}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        counts[(split, lab)] = counts.get((split, lab), 0) + 1
    for k in sorted(counts):
        print(k, counts[k])


if __name__ == "__main__":
    main()
