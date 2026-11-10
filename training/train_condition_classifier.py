"""Member 2 — Train the fresh / rotten condition classifier (PyTorch) and export it to ONNX.

Dataset layout (ImageFolder): one folder per class, e.g. built from the Kaggle
"Fruits fresh and rotten for classification" dataset or Healthy-Defective-Fruits:
    data/condition/train/fresh/*.jpg     data/condition/train/rotten/*.jpg
    data/condition/val/fresh/*.jpg       data/condition/val/rotten/*.jpg
(Any folder name containing "rotten", "stale", "defect" or "bad" counts as bad condition.)
If there is no val/ folder, 15 % of train/ is held out automatically.

Run (Colab/Kaggle GPU recommended, CPU works for small data):
    pip install torch torchvision onnx
    python training/train_condition_classifier.py --data data/condition --epochs 8

Outputs (picked up automatically by the app, config/settings.yaml → condition.type: onnx):
    artifacts/condition_classifier.onnx
    artifacts/condition_classifier_labels.json
    artifacts/condition_classifier_metrics.json   (accuracy, per-class F1, confusion matrix)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

ROOT = Path(__file__).resolve().parent.parent
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def build_model(name: str, n_classes: int, pretrained: bool) -> nn.Module:
    if name == "mobilenet_v3":
        m = models.mobilenet_v3_small(weights="DEFAULT" if pretrained else None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_classes)
    else:  # efficientnet_b0
        m = models.efficientnet_b0(weights="DEFAULT" if pretrained else None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_classes)
    return m


def evaluate(model, loader, device, n_classes):
    model.eval()
    cm = torch.zeros(n_classes, n_classes, dtype=torch.long)
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(1).cpu()
            for t, p in zip(y, pred):
                cm[t, p] += 1
    acc = cm.diag().sum().item() / max(1, cm.sum().item())
    f1 = []
    for c in range(n_classes):
        tp = cm[c, c].item(); fp = cm[:, c].sum().item() - tp; fn = cm[c, :].sum().item() - tp
        p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
        f1.append(2 * p * r / max(1e-9, p + r))
    return acc, f1, cm.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--arch", default="efficientnet_b0", choices=["efficientnet_b0", "mobilenet_v3"])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--no-pretrained", action="store_true", help="train from scratch (only for smoke tests)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(args.size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(), transforms.RandomRotation(20),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.03),  # lighting robustness for webcam use
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([transforms.Resize((args.size, args.size)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    train_dir = args.data / "train" if (args.data / "train").exists() else args.data
    full = datasets.ImageFolder(train_dir, transform=train_tf)
    classes = full.classes
    if (args.data / "val").exists():
        train_ds, val_ds = full, datasets.ImageFolder(args.data / "val", transform=eval_tf)
    else:
        n_val = max(1, int(0.15 * len(full)))
        train_ds, val_ds = random_split(full, [len(full) - n_val, n_val], generator=torch.Generator().manual_seed(0))
        val_ds.dataset = datasets.ImageFolder(train_dir, transform=eval_tf)  # no augmentation for validation
    print(f"classes={classes} train={len(train_ds)} val={len(val_ds)} device={device}")

    tl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2)
    vl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2)
    model = build_model(args.arch, len(classes), not args.no_pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_acc, best_state = -1.0, None
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); total = 0.0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = loss_fn(model(x), y); loss.backward(); opt.step()
            total += loss.item() * len(x)
        sched.step()
        acc, f1, _ = evaluate(model, vl, device, len(classes))
        print(f"epoch {ep + 1}/{args.epochs} loss={total / len(train_ds):.4f} val_acc={acc:.3f} f1={[round(v, 3) for v in f1]} ({time.time() - t0:.0f}s)")
        if acc > best_acc:
            best_acc, best_state = acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval().cpu()
    acc, f1, cm = evaluate(model, vl, "cpu", len(classes))
    out = ROOT / "artifacts"
    out.mkdir(exist_ok=True)
    dummy = torch.randn(1, 3, args.size, args.size)
    torch.onnx.export(model, dummy, str(out / "condition_classifier.onnx"), input_names=["input"], output_names=["logits"],
                      opset_version=17, dynamo=False)
    (out / "condition_classifier_labels.json").write_text(json.dumps(classes))
    (out / "condition_classifier_metrics.json").write_text(json.dumps(
        {"arch": args.arch, "val_accuracy": acc, "f1_per_class": dict(zip(classes, f1)), "confusion_matrix": cm, "classes": classes}, indent=2))
    print(f"Saved artifacts/condition_classifier.onnx (val acc {acc:.3f}). Restart the app to use it.")


if __name__ == "__main__":
    main()
