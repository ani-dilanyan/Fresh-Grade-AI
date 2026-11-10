# Team guide — who owns what, and how to plug in your module

The application already runs end-to-end. Each member **improves their own module behind its
interface** without touching the others. The contract is `contracts/schemas.py` (data) and
`contracts/interfaces.py` (methods). Changes to `contracts/` need a PR approved by all four.

| Member | Module | Files you own | Interface you implement | Swap-in mechanism |
|---|---|---|---|---|
| 1 | Detection & geometry | `vision/`, `training/prepare_yolo_dataset.py`, `training/train_yolo_seg.py` | `FruitDetector`, `GeometryEstimator` | Put trained weights at `artifacts/fruit_seg.pt` — picked up automatically |
| 2 | Condition analysis | `defects/`, `training/train_condition_classifier.py`, `tools/tune_defects.py` | `ConditionAnalyzer` | Put `artifacts/condition_classifier.onnx` + `_labels.json` — picked up automatically |
| 3 | Knowledge & decision | `grading/`, `config/ontology.json`, `config/pricing.yaml` | `GradingEngine`, `LLMInspector` | Edit the ontology (or the **Grading rules** page); add `ANTHROPIC_API_KEY` to `.env` |
| 4 | Platform & automation | `app/`, `automation/`, `reports/`, `mocks/`, `contracts/`, `tests/e2e`, CI, Docker, docs | wiring + services | `config/settings.yaml` |

## Member 1 — steps
1. Download segmentation datasets in **YOLOv8/YOLO11 segmentation** format (e.g. Roboflow Universe
   "fruits segmentations": Apple, Orange, Banana, Tomato, Lemon). Record source + licence in `data/README.md`.
2. `python training/prepare_yolo_dataset.py --sources <export1> <export2> --out datasets/freshgrade_seg`
3. On Colab/Kaggle GPU: `pip install ultralytics` then
   `python training/train_yolo_seg.py --data datasets/freshgrade_seg/data.yaml --epochs 60`
4. Copy the resulting `artifacts/fruit_seg.pt` into the repo, restart the app → `/api/health` shows `yolo-custom`.
5. Report metrics: mask mAP50-95 (printed at the end, also in `runs/segment/...`), size error with the calibration card.

## Member 2 — steps
1. A trained fresh / damaged / rotten model already ships in `artifacts/` (99.6% held-out accuracy).
   To improve it, add more real photos, especially tomatoes and lemons, then rebuild:
   `python training/prepare_condition_dataset.py --hdf <hdf> --ffd <ffd> --out datasets/condition`
   `python training/train_condition_yolo.py --data datasets/condition --epochs 10`
2. Measure the whole pipeline: `python tools/evaluate_grading.py --data datasets/condition/val`.
3. Restart the app → `/api/health` shows `classical+onnx-classifier`.
4. Tune the classical defect map: sort some images into `fresh/`, `bruise/`, `rot/` … and run
   `python tools/tune_defects.py --data <folder> --good fresh`; copy the best values into `ClassicalConditionAnalyzer.PARAMS`.
5. Report: classifier accuracy/F1/confusion matrix (`artifacts/condition_classifier_metrics.json`), tuning table.

## Member 3 — steps
1. Refine `config/ontology.json` from the UNECE standards (classes, tolerances, sizes) and cite them.
2. Add your API key to `.env` (`ANTHROPIC_API_KEY=...`) to enable the AI inspector; tune the prompt in `grading/llm_inspector.py`.
3. Create the team-labelled test set (~100 images graded by two people) and compare with the system
   (`/api/inspect` JSON output or the History page) → confusion matrix for the report.
4. Unit tests: `tests/unit/test_grading.py` — add cases for every rule you change.

## Member 4 — steps
1. Run the app (README), run tests (`python -m pytest -v`), keep CI green.
2. Accessibility audit with `docs/ada_checklist.md`.
3. Integrate teammates' artifacts, measure latency (`processing_ms` in results), prepare the live demo.

## Git workflow
- `main` is always runnable. Work on `feature/<member>-<topic>` branches, open a PR, one reviewer.
- Commit messages: `vision: add ArUco calibration (#12)` — the report requires detailed commit history.
- Never commit `.env` (API keys) or `data/runtime/`.
