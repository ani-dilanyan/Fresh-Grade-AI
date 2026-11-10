"""Printable ArUco calibration card (PDF). Print at 100 % scale ("actual size")."""
from __future__ import annotations

import io

import cv2
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


def marker_png(dictionary: str = "DICT_4X4_50", marker_id: int = 0, px: int = 600) -> bytes:
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    if hasattr(cv2.aruco, "generateImageMarker"):
        img = cv2.aruco.generateImageMarker(d, marker_id, px)
    else:  # OpenCV < 4.7
        img = np.zeros((px, px), np.uint8)
        img = cv2.aruco.drawMarker(d, marker_id, px, img, 1)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def calibration_card_pdf(dictionary: str = "DICT_4X4_50", marker_id: int = 0, size_mm: float = 50.0) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    c.setTitle("FreshGrade calibration card")
    w, h = A4
    c.setFont("Helvetica-Bold", 18)
    c.drawString(20 * mm, h - 25 * mm, "FreshGrade AI - Calibration card")
    c.setFont("Helvetica", 10.5)
    lines = [
        f"Marker: {dictionary}, id {marker_id}. The black square must measure exactly {size_mm:.0f} mm.",
        "Print at 100% / 'Actual size' (no 'fit to page'). Check the square with a ruler before use.",
        "Place the card flat on the table NEXT to the fruit, at the same distance from the camera.",
        "Keep the white border around the marker visible.",
    ]
    for i, t in enumerate(lines):
        c.drawString(20 * mm, h - (35 + i * 6) * mm, t)
    img = ImageReader(io.BytesIO(marker_png(dictionary, marker_id)))
    x, y = (w - size_mm * mm) / 2, h / 2 - size_mm * mm / 2
    c.drawImage(img, x, y, width=size_mm * mm, height=size_mm * mm)
    # ruler below the marker for verification
    c.setLineWidth(0.8)
    c.line(x, y - 10 * mm, x + size_mm * mm, y - 10 * mm)
    for k in range(int(size_mm) + 1):
        tick = 3 * mm if k % 10 == 0 else 1.5 * mm
        c.line(x + k * mm, y - 10 * mm, x + k * mm, y - 10 * mm + tick)
    c.setFont("Helvetica", 9)
    c.drawCentredString(w / 2, y - 16 * mm, f"{size_mm:.0f} mm reference ruler")
    c.showPage()
    c.save()
    return buf.getvalue()
