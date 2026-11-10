"""Member 1 — Fine-tune YOLO11-seg on the merged fruit dataset.

Run on Google Colab / Kaggle (free GPU) or locally:
    pip install ultralytics
    python training/train_yolo_seg.py --data datasets/freshgrade_seg/data.yaml --epochs 60 --model yolo11n-seg.pt

The best weights are copied to artifacts/fruit_seg.pt, which the app picks up automatically
(config/settings.yaml → detector.custom_weights). Validation metrics (mask mAP50-95 etc.) are
printed and saved by Ultralytics under runs/segment/<name>/ — use them in the report.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolo11n-seg.pt", help="pretrained starting point (n = fastest on CPU)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="0 for GPU, cpu for CPU (auto if omitted)")
    ap.add_argument("--name", default="freshgrade_seg")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        patience=15,
        # augmentation to reduce the internet-photo → webcam domain gap
        hsv_h=0.02, hsv_s=0.6, hsv_v=0.5, degrees=15, translate=0.1, scale=0.5, fliplr=0.5, mosaic=1.0,
    )
    metrics = model.val(data=args.data, split="test")
    print("Test mask mAP50-95:", getattr(metrics.seg, "map", None), " mAP50:", getattr(metrics.seg, "map50", None))

    best = Path(model.trainer.best)
    out = ROOT / "artifacts" / "fruit_seg.pt"
    out.parent.mkdir(exist_ok=True)
    shutil.copy2(best, out)
    print("Copied best weights to", out)


if __name__ == "__main__":
    main()
