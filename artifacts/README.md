# Trained model artifacts

Drop trained models here; the app detects them on start-up (check `/api/health`).

| File | Produced by | Used by |
|---|---|---|
| `fruit_seg.pt` | `training/train_yolo_seg.py` (Member 1) | `vision/detector.py` (YOLO custom weights) |
| `condition_classifier.onnx` + `_labels.json` + `_meta.json` | `training/train_condition_yolo.py` (Member 2; ships pre-trained: fresh / damaged / rotten, trained on ~14,600 real photos of apples, mangoes, bananas and oranges) | `defects/classifier.py` |

Without these files the app still works: it uses the pretrained YOLO11 COCO model (apple, orange, banana)
plus the classical OpenCV detector and defect analyser.
