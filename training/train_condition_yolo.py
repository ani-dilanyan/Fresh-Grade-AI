"""Member 2 — Train the fresh / damaged / rotten condition model (Ultralytics YOLO11-cls, PyTorch)
and export it to ONNX for the app.

Uses ImageNet-pretrained YOLO11n-cls weights (downloaded automatically from the Ultralytics GitHub
releases), so it learns quickly even on a laptop CPU (about 5 min/epoch for ~15k images at 192 px).

Usage:
    python training/prepare_condition_dataset.py --hdf hdf --ffd ffd --out datasets/condition
    python training/train_condition_yolo.py --data datasets/condition --epochs 10

Outputs (picked up automatically by the app on restart, config/settings.yaml → condition.type: onnx):
    artifacts/condition_classifier.onnx
    artifacts/condition_classifier_labels.json
    artifacts/condition_classifier_meta.json
    artifacts/condition_classifier_metrics.json   (top-1 accuracy, per-class recall, confusion matrix)
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--model", default="yolo11n-cls.pt")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--imgsz", type=int, default=192)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(data=str(args.data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device,
                project=str(ROOT / "runs"), name="condition", exist_ok=True, patience=4, plots=False,
                hsv_h=0.015, hsv_s=0.5, hsv_v=0.4, degrees=15, fliplr=0.5, flipud=0.2, erasing=0.2)
    best = YOLO(str(Path(model.trainer.save_dir) / "weights" / "best.pt"))
    export_artifacts(best, args.data, args.imgsz)


def export_artifacts(best, data: Path, imgsz: int) -> None:
    import numpy as np

    out = ROOT / "artifacts"
    out.mkdir(exist_ok=True)
    onnx_path = Path(best.export(format="onnx", imgsz=imgsz, simplify=False, dynamic=False))
    shutil.copy2(onnx_path, out / "condition_classifier.onnx")
    labels = [best.names[i] for i in range(len(best.names))]
    (out / "condition_classifier_labels.json").write_text(json.dumps(labels))
    (out / "condition_classifier_meta.json").write_text(json.dumps({"size": imgsz, "normalize": "none", "resize": "crop", "arch": "yolo11n-cls"}))

    # validation metrics for the report
    cm = np.zeros((len(labels), len(labels)), int)
    for ti, lab in enumerate(labels):
        files = sorted((data / "val" / lab).glob("*.jpg"))
        for i in range(0, len(files), 64):
            for r in best.predict([str(f) for f in files[i:i + 64]], imgsz=imgsz, verbose=False):
                cm[ti, int(r.probs.top1)] += 1
    recall = {lab: round(float(cm[i, i] / max(1, cm[i].sum())), 4) for i, lab in enumerate(labels)}
    metrics = {"top1_accuracy": round(float(np.trace(cm) / max(1, cm.sum())), 4), "recall": recall,
               "confusion_matrix": cm.tolist(), "labels": labels, "imgsz": imgsz}
    (out / "condition_classifier_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
