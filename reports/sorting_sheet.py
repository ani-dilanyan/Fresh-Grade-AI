"""Member 4 — Warehouse sorting sheets (printable PDF and CSV) and inventory export."""
from __future__ import annotations

import csv
import io
from datetime import datetime
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from reports.quality_report import CLASS_LABEL, safe

COLUMNS = ["bin_code", "item_id", "fruit_type", "quality_class", "score", "est_weight_g", "price_per_kg", "item_value",
           "needs_manual_review", "inspection_id"]


def _rows(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        {
            "bin_code": it["bin_code"],
            "item_id": it["id"],
            "fruit_type": it["fruit_type"],
            "quality_class": it["quality_class"],
            "score": it["score"],
            "est_weight_g": round(it["est_weight_g"]),
            "price_per_kg": f"{it['price_per_kg']:.2f}",
            "item_value": f"{it['item_value']:.2f}",
            "needs_manual_review": "YES" if it["needs_manual_review"] else "",
            "inspection_id": it["inspection_id"][:8],
        }
        for it in items
    ]
    rows.sort(key=lambda r: (r["bin_code"], r["item_id"]))
    return rows


def sorting_sheet_csv(items: list[dict[str, Any]]) -> str:
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=COLUMNS)
    w.writeheader()
    w.writerows(_rows(items))
    return out.getvalue()


def sorting_sheet_pdf(items: list[dict[str, Any]], title: str = "FreshGrade AI") -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=12 * mm, rightMargin=12 * mm, topMargin=12 * mm, bottomMargin=12 * mm,
                            title="Warehouse sorting sheet", author=title)
    ss = getSampleStyleSheet()
    story: list[Any] = [
        Paragraph(safe(f"{title} — Warehouse Sorting Sheet"), ss["Title"]),
        Paragraph(safe(f"Generated {datetime.now():%Y-%m-%d %H:%M} · {len(items)} item(s). Tick each item when placed in its bin."), ss["Normal"]),
        Spacer(1, 8),
    ]
    rows = _rows(items)
    if not rows:
        story.append(Paragraph("No items in stock.", ss["Normal"]))
    bins: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        bins.setdefault(r["bin_code"], []).append(r)
    for bin_code, rs in bins.items():
        cls = rs[0]["quality_class"]
        kg = sum(r["est_weight_g"] for r in rs) / 1000
        story.append(Paragraph(safe(f"Bin {bin_code} — {CLASS_LABEL.get(cls, cls)} — {len(rs)} item(s), approx. {kg:.2f} kg"), ss["Heading3"]))
        data = [["Done", "Item", "Fruit", "Class", "Score", "Weight g", "EUR/kg", "Value", "Review", "Inspection"]]
        data += [["[  ]", r["item_id"], r["fruit_type"], r["quality_class"], r["score"], r["est_weight_g"], r["price_per_kg"],
                  r["item_value"], r["needs_manual_review"], r["inspection_id"]] for r in rs]
        t = Table(data, repeatRows=1)
        t.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#7f8c86")),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4d3a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f6f3")]),
        ]))
        story += [t, Spacer(1, 8)]
    doc.build(story)
    return buf.getvalue()


def inventory_csv(items: list[dict[str, Any]]) -> str:
    out = io.StringIO()
    if not items:
        return "no items\n"
    w = csv.DictWriter(out, fieldnames=list(items[0].keys()))
    w.writeheader()
    w.writerows(items)
    return out.getvalue()
