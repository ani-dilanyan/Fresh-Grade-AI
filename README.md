# FreshGrade AI — Computer-Vision Grading System for Local Produce Quality

UFAR · Faculty of Computer Science and Applied Mathematics · L2 Project S3 (2025-2026)

FreshGrade AI grades apples, tomatoes, oranges, bananas and lemons from a **live camera** or an
**uploaded photo**:

1. It finds each fruit and segments it.
2. It measures size, volume and weight using a printed calibration card.
3. It detects surface defects, measures defect % and colour uniformity, and estimates the probability of rot.
4. It assigns a UNECE-style quality class (Extra / I / II / Reject) and explains why.
5. It estimates a market price.
6. It updates the inventory, reprices stock, and produces warehouse sorting sheets and PDF quality reports.

An optional AI second opinion (Claude vision API) can cross-check each grade.

---

## 1. Run it (about 10 minutes the first time)

### Requirements
* **Python 3.10 – 3.13** ([python.org](https://www.python.org/downloads/); on Windows tick *“Add Python to PATH”*).
* Chrome, Edge or Firefox. You also need a webcam for the live mode; upload works without one.

### Step 1 — Open a terminal in the project folder
* **Windows:** open the `freshgrade` folder in File Explorer, click the address bar, type `cmd`, press Enter.
* **macOS:** right-click the `freshgrade` folder → *New Terminal at Folder*.

### Step 2 — Create a virtual environment and install
**Windows**
```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-ml.txt
```
**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-ml.txt
```
`requirements-ml.txt` installs everything, including YOLO and PyTorch. It is a large download of about 1–2 GB.
If the internet is slow, install `requirements.txt` instead. The app then runs with the
classical OpenCV detector only, which works well on a plain background, and you can add the ML part later.

### Step 3 — Start the app
```bash
python run.py
```
The browser opens **http://localhost:8000**. Alternatively run `uvicorn app.main:app` and open that URL yourself.
On the first start, the pretrained YOLO model (6 MB) is downloaded automatically.
Stop the app with **CTRL + C**.

### Step 4 — Try it
1. **Live inspect:** click *Start camera*, allow camera access, and hold a fruit in front of a plain light background.
   Live boxes appear. Press **Space** (or *Inspect now*) to get the full grade.
2. **Upload:** run `python tools/make_samples.py` to create test images in `data/samples/`, or use
   any fruit photo from the internet.
3. **Calibration (for real size and weight):**
   * Open *Reports → Calibration card PDF* and print it at **100 % / Actual size**.
   * Check that the black square measures 50 mm.
   * Place the card next to the fruit.
4. Open **Inventory**, **History**, **Reports** and **Grading rules** to see the automation, the reports and the live threshold editing.

### Step 5 (optional) — AI second opinion
Copy `.env.example` to `.env` and paste your Anthropic API key:
```
ANTHROPIC_API_KEY=sk-ant-...
```
Restart the app. The *AI second opinion* checkbox becomes active. Without a key, or offline, the app
simply grades with the local models.

### Run the tests
```bash
python -m pytest -v
```
The suite has 40 tests: unit, integration and end-to-end (API, WebSocket, HTML pages and PDFs). They need no camera, internet or GPU.

---

## 2. What is inside

```
contracts/    schemas.py (pydantic data contract), interfaces.py (module Protocols)      ← all members
vision/       detector.py (YOLO11-seg + OpenCV baseline), geometry.py (ArUco, volume)    ← Member 1
defects/      analyzer.py (defect map, heat-map), color.py, classifier.py (ONNX)          ← Member 2
grading/      rules.py (ontology rules + LLM fusion), pricing.py, llm_inspector.py       ← Member 3
config/       ontology.json (defect ontology & class limits), pricing.yaml, settings.yaml
app/          FastAPI app, live WebSocket, pages, DB, inspection orchestrator           ← Member 4
automation/   inventory_service.py (event-driven stock update), pricing_job.py (markdowns)
reports/      quality_report.py, sorting_sheet.py, calibration_card.py (PDF/CSV)
mocks/        mock implementations of every interface (parallel development, tests)
training/     prepare_yolo_dataset.py, train_yolo_seg.py, train_condition_classifier.py
tools/        make_samples.py (synthetic test images), tune_defects.py (parameter search)
tests/        unit/, integration/, e2e/
docs/         team_guide.md (who does what), ada_checklist.md (accessibility evidence)
```

### How a grade is produced
```
image → ArUco calibration → fruit detection & masks → per fruit:
        size/volume/weight · defect map (defect %, regions) · colour uniformity · rot probability
        · optional AI opinion → ontology rules → class + score + reasons + price
      → overlay & heat-maps → SQLite → InspectionCompleted event → inventory + bins → reports
```

### Switching modules (config/settings.yaml)
| Setting | Options |
|---|---|
| `detector.type` | `auto` (YOLO + classical fallback), `yolo`, `classical`, `mock` |
| `condition.type` | `onnx` (classical defect map + trained classifier if present), `classical`, `mock` |
| `llm.type` | `anthropic`, `disabled`, `mock` |

Trained models are loaded automatically when you place them in `artifacts/`; see `docs/team_guide.md`.
Check what is active at **http://localhost:8000/api/health**. Interactive API docs are at **/docs**.

---

## 3. Accuracy (measured on real photos)

The app ships with a trained **condition model** (`artifacts/condition_classifier.onnx`, YOLO11n-cls, PyTorch → ONNX).
It labels each fruit fresh / damaged / rotten and was trained on about 14,600 real photos of apples, mangoes,
bananas and oranges. The data comes from two public GitHub datasets:
- Healthy-Defective-Fruits;
- Fruit-freshness-detection-dataset (Apache-2.0).

See `training/prepare_condition_dataset.py` and `training/train_condition_yolo.py` to rebuild it.

* **Condition model on 2,441 held-out photos:** 99.6% accuracy.
  * Recall: fresh 99.4%, damaged 99.7%, rotten 99.8%.
* **Whole grading pipeline** on 900 held-out real photos (`python tools/evaluate_grading.py --data <val folder>`):

| | Before (colour rules only) | Now (with trained model) |
|---|---|---|
| Rotten fruit never sold as good (graded Class II or Reject) | 66% | **98%** |
| Rotten fruit rejected | 41% | **89%** |
| Damaged / ageing fruit downgraded (Class II or Reject) | 45% | **97%** |
| Fresh fruit accepted (Extra or Class I) | 82% | **94%** |

**Limitations:**
* The held-out photos come from the same datasets as the training photos; accuracy on very different photos, such as a webcam in poor light, can be lower.
* Tomatoes and lemons are graded by the same condition model, but it saw no tomato or lemon photos in training.
* The generic COCO detector has no lemon or tomato class. Colour and shape rules relabel round orange-yellow fruit; Member 1's fine-tuned `fruit_seg.pt` fixes this properly.
* Only the visible side of the fruit is graded.

## 4. Troubleshooting
| Problem | Fix |
|---|---|
| `python` not found (Windows) | Reinstall Python with *Add to PATH*, or use `py -m venv .venv` |
| Camera does not start | Use **http://localhost:8000** (not an IP address), allow camera access in the browser, close other apps using the camera, or pick another camera in the dropdown |
| “No fruit found” | Plain light background, more light, whole fruit in the frame, fruit fills a larger part of the image |
| Health shows `no-yolo` | `pip install -r requirements-ml.txt`, and be online on the first start so the YOLO weights can download |
| Port 8000 busy | `python run.py --port 8080` |
| Reset all data | Stop the app and delete the `data/runtime/` folder |
| Docker | `docker compose up --build`, then http://localhost:8000 |
