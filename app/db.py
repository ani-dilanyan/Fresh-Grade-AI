"""SQLite persistence (Member 4). Repository pattern over the standard-library sqlite3 module."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS inspections (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    source TEXT NOT NULL,
    image_path TEXT NOT NULL,
    overlay_path TEXT NOT NULL,
    fruit_count INTEGER NOT NULL,
    processing_ms INTEGER NOT NULL,
    result_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    inspection_id TEXT NOT NULL REFERENCES inspections(id),
    fruit_index INTEGER NOT NULL,
    fruit_type TEXT NOT NULL,
    quality_class TEXT NOT NULL,
    score INTEGER NOT NULL,
    confidence REAL NOT NULL,
    defect_pct REAL NOT NULL,
    color_uniformity REAL NOT NULL,
    est_weight_g REAL NOT NULL,
    base_price_per_kg REAL NOT NULL,
    price_per_kg REAL NOT NULL,
    item_value REAL NOT NULL,
    needs_manual_review INTEGER NOT NULL,
    bin_code TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'in_stock',
    created_at TEXT NOT NULL,
    stocked_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS inventory_ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id INTEGER NOT NULL REFERENCES items(id),
    event TEXT NOT NULL,
    quantity_delta INTEGER NOT NULL,
    note TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS price_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id INTEGER NOT NULL REFERENCES items(id),
    old_price REAL NOT NULL,
    new_price REAL NOT NULL,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_items_status ON items(status);
CREATE INDEX IF NOT EXISTS idx_items_inspection ON items(inspection_id);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as con:
            con.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.path, timeout=10)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        try:
            yield con
            con.commit()
        finally:
            con.close()

    # ------------------------------------------------------------------ inspections
    def save_inspection(self, result: dict[str, Any]) -> None:
        with self.connect() as con:
            con.execute(
                "INSERT INTO inspections VALUES (?,?,?,?,?,?,?,?)",
                (
                    result["inspection_id"],
                    result["created_at"],
                    result["source"],
                    result["image_path"],
                    result["overlay_path"],
                    len(result["fruits"]),
                    result["processing_ms"],
                    json.dumps(result, default=str),
                ),
            )

    def get_inspection(self, inspection_id: str) -> Optional[dict[str, Any]]:
        with self.connect() as con:
            row = con.execute("SELECT result_json FROM inspections WHERE id=?", (inspection_id,)).fetchone()
        return json.loads(row["result_json"]) if row else None

    def list_inspections(self, limit: int = 100, fruit: Optional[str] = None, quality: Optional[str] = None) -> list[dict[str, Any]]:
        q = "SELECT i.* FROM inspections i"
        args: list[Any] = []
        where = []
        if fruit or quality:
            q += " WHERE i.id IN (SELECT inspection_id FROM items WHERE 1=1"
            if fruit:
                q += " AND fruit_type=?"
                args.append(fruit)
            if quality:
                q += " AND quality_class=?"
                args.append(quality)
            q += ")"
        q += " ORDER BY i.created_at DESC LIMIT ?"
        args.append(limit)
        with self.connect() as con:
            rows = con.execute(q, args).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            res = json.loads(d.pop("result_json"))
            d["summary"] = [
                {"fruit_type": f["fruit"]["fruit_type"], "quality_class": f["grade"]["quality_class"], "score": f["grade"]["score"]}
                for f in res["fruits"]
            ]
            out.append(d)
        return out

    # ------------------------------------------------------------------ items / inventory
    def add_item(self, inspection_id: str, idx: int, fruit: dict[str, Any]) -> int:
        g, c = fruit["grade"], fruit["condition"]
        ts = now_iso()
        with self.connect() as con:
            cur = con.execute(
                """INSERT INTO items (inspection_id, fruit_index, fruit_type, quality_class, score, confidence, defect_pct,
                   color_uniformity, est_weight_g, base_price_per_kg, price_per_kg, item_value, needs_manual_review, bin_code,
                   status, created_at, stocked_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    inspection_id,
                    idx,
                    fruit["fruit"]["fruit_type"],
                    g["quality_class"],
                    g["score"],
                    g["confidence"],
                    c["defect_pct"],
                    c["color_uniformity"],
                    fruit["geometry"]["est_weight_g"],
                    g["price_per_kg"],
                    g["price_per_kg"],
                    g["item_value"],
                    int(g["needs_manual_review"]),
                    g["bin_code"],
                    "discarded" if g["quality_class"] == "Reject" else "in_stock",
                    ts,
                    ts,
                ),
            )
            item_id = int(cur.lastrowid)
            event = "rejected" if g["quality_class"] == "Reject" else "added"
            con.execute(
                "INSERT INTO inventory_ledger (item_id, event, quantity_delta, note, created_at) VALUES (?,?,?,?,?)",
                (item_id, event, 0 if event == "rejected" else 1, f"bin {g['bin_code']}", ts),
            )
        return item_id

    def items(self, status: Optional[str] = None, inspection_id: Optional[str] = None) -> list[dict[str, Any]]:
        q, args = "SELECT * FROM items WHERE 1=1", []
        if status:
            q += " AND status=?"
            args.append(status)
        if inspection_id:
            q += " AND inspection_id=?"
            args.append(inspection_id)
        q += " ORDER BY id DESC"
        with self.connect() as con:
            return [dict(r) for r in con.execute(q, args).fetchall()]

    def inventory_summary(self) -> list[dict[str, Any]]:
        with self.connect() as con:
            rows = con.execute(
                """SELECT fruit_type, quality_class, bin_code, COUNT(*) AS qty, ROUND(SUM(est_weight_g)/1000.0, 3) AS kg,
                          ROUND(SUM(item_value), 2) AS value, ROUND(AVG(price_per_kg), 2) AS avg_price_per_kg
                   FROM items WHERE status='in_stock' GROUP BY fruit_type, quality_class, bin_code
                   ORDER BY fruit_type, quality_class"""
            ).fetchall()
        return [dict(r) for r in rows]

    def ledger(self, limit: int = 200) -> list[dict[str, Any]]:
        with self.connect() as con:
            rows = con.execute(
                """SELECT l.*, i.fruit_type, i.quality_class FROM inventory_ledger l JOIN items i ON i.id=l.item_id
                   ORDER BY l.id DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def update_price(self, item_id: int, new_price: float, new_value: float, reason: str) -> None:
        ts = now_iso()
        with self.connect() as con:
            old = con.execute("SELECT price_per_kg FROM items WHERE id=?", (item_id,)).fetchone()["price_per_kg"]
            con.execute("UPDATE items SET price_per_kg=?, item_value=? WHERE id=?", (new_price, new_value, item_id))
            con.execute(
                "INSERT INTO price_history (item_id, old_price, new_price, reason, created_at) VALUES (?,?,?,?,?)",
                (item_id, old, new_price, reason, ts),
            )
            con.execute(
                "INSERT INTO inventory_ledger (item_id, event, quantity_delta, note, created_at) VALUES (?,?,?,?,?)",
                (item_id, "repriced", 0, f"{old:.2f} → {new_price:.2f} €/kg ({reason})", ts),
            )

    def set_status(self, item_id: int, status: str, note: str = "") -> None:
        ts = now_iso()
        with self.connect() as con:
            con.execute("UPDATE items SET status=? WHERE id=?", (status, item_id))
            con.execute(
                "INSERT INTO inventory_ledger (item_id, event, quantity_delta, note, created_at) VALUES (?,?,?,?,?)",
                (item_id, status, -1, note, ts),
            )

    def price_history(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.connect() as con:
            return [dict(r) for r in con.execute("SELECT * FROM price_history ORDER BY id DESC LIMIT ?", (limit,)).fetchall()]

    def stats(self) -> dict[str, Any]:
        with self.connect() as con:
            n_insp = con.execute("SELECT COUNT(*) FROM inspections").fetchone()[0]
            by_class = {
                r["quality_class"]: r["n"] for r in con.execute("SELECT quality_class, COUNT(*) n FROM items GROUP BY quality_class")
            }
            in_stock = con.execute("SELECT COUNT(*), COALESCE(SUM(item_value),0) FROM items WHERE status='in_stock'").fetchone()
            review = con.execute("SELECT COUNT(*) FROM items WHERE needs_manual_review=1 AND status='in_stock'").fetchone()[0]
        return {
            "inspections": n_insp,
            "items_by_class": by_class,
            "in_stock": in_stock[0],
            "stock_value": round(in_stock[1], 2),
            "manual_review": review,
        }
