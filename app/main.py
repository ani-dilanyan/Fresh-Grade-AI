"""FreshGrade AI — FastAPI application (Member 4).

Run:  uvicorn app.main:app --reload      →  http://localhost:8000
"""
from __future__ import annotations

import asyncio
import copy
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.container import Container
from app.services.inspection_service import NoFruitFound
from core import config
from core.imaging import decode_image, limit_size
from reports.calibration_card import calibration_card_pdf
from reports.quality_report import build_quality_report
from reports.sorting_sheet import inventory_csv, sorting_sheet_csv, sorting_sheet_pdf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("freshgrade")

APP_DIR = Path(__file__).resolve().parent
CLASS_LABEL = {"Extra": "Extra Class", "I": "Class I", "II": "Class II", "Reject": "Reject"}
CLASS_CSS = {"Extra": "extra", "I": "c1", "II": "c2", "Reject": "rej"}
CLASS_ICON = {"Extra": "★", "I": "✓", "II": "◐", "Reject": "✕"}


def create_app(settings: Optional[dict[str, Any]] = None) -> FastAPI:
    container = Container(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        interval = float(container.settings["automation"].get("pricing_job_interval_min", 60)) * 60

        async def pricing_loop():
            while True:
                await asyncio.sleep(interval)
                try:
                    res = await asyncio.to_thread(container.pricing_job.run)
                    log.info("Scheduled repricing: %s", {k: v for k, v in res.items() if k != "changes"})
                except Exception:
                    log.exception("Scheduled repricing failed")

        task = asyncio.create_task(pricing_loop())
        yield
        task.cancel()

    app = FastAPI(title="FreshGrade AI", version="1.0", lifespan=lifespan,
                  description="Computer-vision grading system for local produce quality")
    app.state.c = container
    app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
    app.mount("/media", StaticFiles(directory=container.media_root), name="media")
    templates = Jinja2Templates(directory=APP_DIR / "templates")
    templates.env.globals.update(CLASS_LABEL=CLASS_LABEL, CLASS_CSS=CLASS_CSS, CLASS_ICON=CLASS_ICON, app_title=container.settings["app"]["title"])

    max_bytes = int(container.settings["app"]["max_upload_mb"]) * 1024 * 1024
    max_side = int(container.settings["app"]["max_image_side_px"])

    def page(request: Request, name: str, **ctx) -> HTMLResponse:
        ctx.setdefault("active", name.split(".")[0])
        return templates.TemplateResponse(request, name, ctx)

    async def run_inspection(data: bytes, source: str, use_llm: bool) -> dict[str, Any]:
        if len(data) > max_bytes:
            raise HTTPException(413, f"Image is larger than {max_bytes // (1024 * 1024)} MB.")
        img = decode_image(data)
        if img is None:
            raise HTTPException(400, "The file is not a readable image (use JPG, PNG or WebP).")
        img = limit_size(img, max_side)
        try:
            result = await container.inspection.inspect(img, source=source, use_llm=use_llm)
        except NoFruitFound as e:
            raise HTTPException(422, str(e))
        return result.model_dump(mode="json")

    # ================================================================== API
    @app.get("/api/health")
    def health():
        return container.health()

    @app.post("/api/inspect")
    async def api_inspect(file: UploadFile = File(...), source: str = Form("upload"), use_llm: bool = Form(True)):
        data = await file.read()
        return await run_inspection(data, source if source in ("live", "upload", "api") else "upload", use_llm)

    @app.get("/api/inspections")
    def api_inspections(limit: int = 100, fruit: Optional[str] = None, quality: Optional[str] = None):
        return container.db.list_inspections(limit, fruit, quality)

    @app.get("/api/inspections/{inspection_id}")
    def api_inspection(inspection_id: str):
        r = container.db.get_inspection(inspection_id)
        if not r:
            raise HTTPException(404, "Inspection not found")
        return r

    @app.get("/api/inventory")
    def api_inventory():
        return {"summary": container.db.inventory_summary(), "items": container.db.items(), "stats": container.db.stats()}

    @app.post("/api/inventory/reprice")
    def api_reprice(simulate_days: float = 0.0):
        return container.pricing_job.run(simulate_extra_days=max(0.0, min(simulate_days, 60.0)))

    @app.post("/api/inventory/{item_id}/status")
    def api_item_status(item_id: int, status: str):
        if status not in ("sold", "discarded"):
            raise HTTPException(400, "status must be 'sold' or 'discarded'")
        if not any(i["id"] == item_id for i in container.db.items(status="in_stock")):
            raise HTTPException(404, "Item not in stock")
        container.db.set_status(item_id, status, status)
        return {"ok": True}

    @app.get("/api/reports/sorting-sheet")
    def api_sorting_sheet(format: str = "pdf", inspection_id: Optional[str] = None):
        items = container.db.items(status="in_stock", inspection_id=inspection_id)
        if format == "csv":
            return Response(sorting_sheet_csv(items), media_type="text/csv",
                            headers={"Content-Disposition": 'attachment; filename="sorting_sheet.csv"'})
        pdf = sorting_sheet_pdf(items, container.settings["app"]["title"])
        return Response(pdf, media_type="application/pdf", headers={"Content-Disposition": 'inline; filename="sorting_sheet.pdf"'})

    @app.get("/api/reports/quality/{inspection_id}")
    def api_quality_report(inspection_id: str):
        r = container.db.get_inspection(inspection_id)
        if not r:
            raise HTTPException(404, "Inspection not found")
        pdf = build_quality_report(r, container.media_root, container.settings["app"]["title"])
        return Response(pdf, media_type="application/pdf",
                        headers={"Content-Disposition": f'inline; filename="quality_report_{inspection_id}.pdf"'})

    @app.get("/api/reports/inspection/{inspection_id}.json")
    def api_inspection_json(inspection_id: str):
        r = container.db.get_inspection(inspection_id)
        if not r:
            raise HTTPException(404, "Inspection not found")
        return JSONResponse(r, headers={"Content-Disposition": f'attachment; filename="inspection_{inspection_id}.json"'})

    @app.get("/api/reports/inventory.csv")
    def api_inventory_csv():
        return Response(inventory_csv(container.db.items()), media_type="text/csv",
                        headers={"Content-Disposition": 'attachment; filename="inventory.csv"'})

    @app.get("/api/ontology")
    def api_get_ontology():
        return container.ontology()

    @app.put("/api/ontology")
    async def api_put_ontology(request: Request):
        data = await request.json()
        _validate_ontology(data)
        container.update_ontology(data)
        return {"ok": True}

    @app.get("/calibration-card.pdf")
    def calibration_card():
        cal = container.settings["calibration"]
        pdf = calibration_card_pdf(cal["aruco_dictionary"], int(cal["marker_id"]), float(cal["marker_size_mm"]))
        return Response(pdf, media_type="application/pdf", headers={"Content-Disposition": 'inline; filename="calibration_card.pdf"'})

    # ================================================================== live WebSocket
    @app.websocket("/ws/live")
    async def ws_live(ws: WebSocket):
        await ws.accept()
        try:
            while True:
                data = await ws.receive_bytes()
                img = decode_image(data)
                if img is None:
                    await ws.send_json({"error": "bad frame"})
                    continue
                img = limit_size(img, 960)
                res = await asyncio.to_thread(container.inspection.live_frame, img)
                await ws.send_text(res.model_dump_json())
        except WebSocketDisconnect:
            pass
        except Exception:
            log.exception("Live stream error")
            try:
                await ws.close()
            except Exception:
                pass

    # ================================================================== HTML pages
    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        return page(request, "live.html", health=container.health())

    @app.get("/upload", response_class=HTMLResponse)
    def upload_page(request: Request, error: Optional[str] = None):
        return page(request, "upload.html", error=error, health=container.health())

    @app.post("/upload")
    async def upload_form(file: UploadFile = File(...), use_llm: Optional[str] = Form(None)):
        """No-JavaScript fallback: classic form post, then redirect to the results page."""
        try:
            r = await run_inspection(await file.read(), "upload", use_llm is not None)
        except HTTPException as e:
            from urllib.parse import quote

            return RedirectResponse(f"/upload?error={quote(e.detail)}", status_code=303)
        return RedirectResponse(f"/results/{r['inspection_id']}", status_code=303)

    @app.get("/results/{inspection_id}", response_class=HTMLResponse)
    def results_page(request: Request, inspection_id: str):
        r = container.db.get_inspection(inspection_id)
        if not r:
            raise HTTPException(404, "Inspection not found")
        return page(request, "results.html", r=r, active="history")

    @app.get("/inventory", response_class=HTMLResponse)
    def inventory_page(request: Request, msg: Optional[str] = None):
        return page(request, "inventory.html", summary=container.db.inventory_summary(), items=container.db.items(status="in_stock"),
                    ledger=container.db.ledger(40), stats=container.db.stats(), msg=msg)

    @app.post("/inventory/reprice")
    def inventory_reprice(simulate_days: float = Form(0.0)):
        res = container.pricing_job.run(simulate_extra_days=max(0.0, min(simulate_days, 60.0)))
        return RedirectResponse(f"/inventory?msg=Repricing done: {res['repriced']} of {res['checked']} items changed", status_code=303)

    @app.post("/inventory/{item_id}/status")
    def inventory_status(item_id: int, status: str = Form(...)):
        if status in ("sold", "discarded"):
            container.db.set_status(item_id, status, status)
        return RedirectResponse(f"/inventory?msg=Item {item_id} marked {status}", status_code=303)

    @app.get("/history", response_class=HTMLResponse)
    def history_page(request: Request, fruit: Optional[str] = None, quality: Optional[str] = None):
        return page(request, "history.html", rows=container.db.list_inspections(200, fruit or None, quality or None),
                    fruit=fruit or "", quality=quality or "", fruits=list(container.ontology()["fruit_profiles"].keys()))

    @app.get("/reports", response_class=HTMLResponse)
    def reports_page(request: Request):
        return page(request, "reports.html", stats=container.db.stats(), recent=container.db.list_inspections(15),
                    prices=container.db.price_history(15))

    @app.get("/settings", response_class=HTMLResponse)
    def settings_page(request: Request, fruit: str = "apple", msg: Optional[str] = None):
        onto = container.ontology()
        if fruit not in onto["fruit_profiles"]:
            fruit = "apple"
        return page(request, "settings.html", onto=onto, fruit=fruit, msg=msg, health=container.health())

    @app.post("/settings")
    async def settings_save(request: Request):
        form = await request.form()
        fruit = str(form.get("fruit", "apple"))
        onto = copy.deepcopy(container.ontology())
        if form.get("action") == "restore":
            default = json.loads((config.CONFIG_DIR / "ontology.default.json").read_text(encoding="utf-8"))
            container.update_ontology(default)
            return RedirectResponse(f"/settings?fruit={fruit}&msg=Default thresholds restored", status_code=303)
        try:
            prof = onto["fruit_profiles"][fruit]
            for cls in ("Extra", "I", "II"):
                prof["classes"][cls]["max_defect_pct"] = float(form[f"{cls}_max_defect_pct"])
                prof["classes"][cls]["min_uniformity"] = float(form[f"{cls}_min_uniformity"])
                prof["classes"][cls]["min_diameter_mm"] = float(form[f"{cls}_min_diameter_mm"])
            onto["hard_rules"]["reject_if_rotten_prob_at_least"] = float(form["reject_if_rotten_prob_at_least"])
            onto["hard_rules"]["reject_if_defect_pct_above"] = float(form["reject_if_defect_pct_above"])
            _validate_ontology(onto)
        except (KeyError, ValueError, HTTPException) as e:
            detail = getattr(e, "detail", str(e))
            return RedirectResponse(f"/settings?fruit={fruit}&msg=Not saved: {detail}", status_code=303)
        container.update_ontology(onto)
        return RedirectResponse(f"/settings?fruit={fruit}&msg=Thresholds saved — new inspections use them immediately", status_code=303)

    @app.exception_handler(404)
    async def not_found(request: Request, exc):
        if request.url.path.startswith("/api") or request.url.path.startswith("/media"):
            return JSONResponse({"detail": getattr(exc, "detail", "Not found")}, status_code=404)
        return templates.TemplateResponse(request, "error.html", {"message": getattr(exc, "detail", "Page not found"), "active": ""}, status_code=404)

    return app


def _validate_ontology(data: dict[str, Any]) -> None:
    try:
        for name, prof in data["fruit_profiles"].items():
            prev_def = -1.0
            for cls in ("Extra", "I", "II"):
                r = prof["classes"][cls]
                d, u = float(r["max_defect_pct"]), float(r["min_uniformity"])
                if not (0 <= d <= 100) or not (0 <= u <= 1):
                    raise ValueError(f"{name}/{cls}: defect % must be 0-100 and uniformity 0-1")
                if d < prev_def:
                    raise ValueError(f"{name}: defect limits must increase from Extra to II")
                prev_def = d
        hr = data["hard_rules"]
        if not (0 < float(hr["reject_if_rotten_prob_at_least"]) <= 1):
            raise ValueError("rot probability threshold must be in (0, 1]")
        float(hr["reject_if_defect_pct_above"])
    except (KeyError, TypeError) as e:
        raise HTTPException(400, f"Invalid ontology: missing {e}")
    except ValueError as e:
        raise HTTPException(400, f"Invalid ontology: {e}")


app = create_app()
