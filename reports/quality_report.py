"""Member 4 — Per-inspection quality report (PDF, real text for accessibility)."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

CLASS_LABEL = {"Extra": "Extra Class", "I": "Class I", "II": "Class II", "Reject": "Reject"}


def safe(text: Any) -> str:
    """ReportLab's built-in fonts use cp1252; replace unsupported characters and escape markup."""
    s = str(text).replace("→", "->").replace("≥", ">=").replace("≤", "<=")
    s = s.encode("cp1252", errors="replace").decode("cp1252")
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _img(path: Path, max_w: float, max_h: float) -> Image | None:
    if not path.exists():
        return None
    img = Image(str(path))
    ratio = min(max_w / img.imageWidth, max_h / img.imageHeight)
    img.drawWidth, img.drawHeight = img.imageWidth * ratio, img.imageHeight * ratio
    return img


def build_quality_report(result: dict[str, Any], media_root: Path, title: str = "FreshGrade AI") -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=14 * mm, bottomMargin=14 * mm,
        title=f"Quality report {result['inspection_id']}", author=title, subject="Produce quality inspection report",
    )
    ss = getSampleStyleSheet()
    small = ParagraphStyle("small", parent=ss["Normal"], fontSize=9, leading=12)
    story: list[Any] = [
        Paragraph(safe(f"{title} — Produce Quality Report"), ss["Title"]),
        Paragraph(
            safe(f"Inspection {result['inspection_id']} · {str(result['created_at'])[:19].replace('T', ' ')} UTC · "
                 f"source: {result['source']} · {len(result['fruits'])} fruit(s) · {result['processing_ms']} ms"),
            small,
        ),
        Spacer(1, 6),
    ]
    overlay = _img(media_root / result["overlay_path"], 178 * mm, 95 * mm)
    if overlay:
        story += [overlay, Paragraph("Figure: analysed image with detected fruit, grade labels and defect outlines.", small), Spacer(1, 8)]

    for i, f in enumerate(result["fruits"], 1):
        g, c, geo, fr = f["grade"], f["condition"], f["geometry"], f["fruit"]
        story.append(Paragraph(safe(f"Fruit {i}: {fr['fruit_type'].title()} — {CLASS_LABEL[g['quality_class']]}"), ss["Heading2"]))
        rows = [
            ["Quality class", CLASS_LABEL[g["quality_class"]], "Quality score", f"{g['score']} / 100"],
            ["Confidence", f"{g['confidence']:.0%}", "Manual review", "Yes" if g["needs_manual_review"] else "No"],
            ["Defect area", f"{c['defect_pct']:.1f} % of visible surface", "Defect regions", str(len(c["defects"]))],
            ["Colour uniformity", f"{c['color_uniformity']:.2f}", "Target colour", f"{c['target_color_pct']:.0f} %"],
            ["Rot probability", f"{c['rotten_prob']:.0%}", "Detector conf.", f"{fr['confidence']:.0%} ({fr['detector']})"],
            ["Size", (f"{geo['diameter_mm']:.0f} mm, {geo['volume_cm3']:.0f} cm3" if geo["calibrated"] else "not calibrated"),
             "Est. weight", f"{geo['est_weight_g']:.0f} g"],
            ["Price", f"{g['price_per_kg']:.2f} {g['currency']}/kg", "Item value", f"{g['item_value']:.2f} {g['currency']}"],
            ["Sorting bin", g["bin_code"], "AI inspector", (f["llm"]["suggested_class"] if f.get("llm") else "not used")],
            ["Condition model", (c.get("condition_label") or "not used"), "Damage / ageing", f"{c.get('damage_prob', 0):.0%}"],
        ]
        t = Table([[Paragraph(safe(x), small) for x in r] for r in rows], colWidths=[32 * mm, 57 * mm, 32 * mm, 57 * mm])
        t.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9aa5a0")),
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef3ef")),
            ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#eef3ef")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story += [t, Spacer(1, 4), Paragraph("<b>Reasons</b>", small)]
        story += [Paragraph(safe("• " + r), small) for r in g["reasons"]]
        if f.get("llm") and f["llm"].get("reasoning"):
            story.append(Paragraph(safe("AI inspector: " + f["llm"]["reasoning"]), small))
        if c.get("heatmap_path"):
            hm = _img(media_root / c["heatmap_path"], 60 * mm, 60 * mm)
            if hm:
                story += [Spacer(1, 4), hm, Paragraph("Defect heat-map (red = most anomalous surface).", small)]
        story.append(Spacer(1, 10))
    if result.get("warnings"):
        story.append(Paragraph("<b>Warnings</b>", small))
        story += [Paragraph(safe("• " + w), small) for w in result["warnings"]]
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Grades follow the FreshGrade ontology (UNECE-style classes). Measurements refer to the visible surface only.", small))
    doc.build(story)
    return buf.getvalue()
