"""Member 4 — Automatic inventory update after every inspection."""
from __future__ import annotations

import logging
from typing import Any

from app.db import Database

log = logging.getLogger("freshgrade.inventory")


class InventoryService:
    def __init__(self, db: Database):
        self.db = db

    def on_inspection_completed(self, result: dict[str, Any]) -> None:
        """Event handler: each graded fruit becomes an item in the inventory ledger.

        Sellable classes go to stock in their sorting bin; rejects are recorded as discarded.
        """
        for idx, fruit in enumerate(result["fruits"]):
            item_id = self.db.add_item(result["inspection_id"], idx, fruit)
            log.info("Inventory: item %s %s %s → %s", item_id, fruit["fruit"]["fruit_type"], fruit["grade"]["quality_class"], fruit["grade"]["bin_code"])

    def mark_sold(self, item_id: int) -> None:
        self.db.set_status(item_id, "sold", "sold")

    def mark_discarded(self, item_id: int) -> None:
        self.db.set_status(item_id, "discarded", "discarded manually")
