"""Member 4 — Automatic stock price adjustment.

Policy (config/pricing.yaml → markdown_policy):
  * age markdown   : after `grace_days`, reduce by `daily_markdown_pct` per day, never below `floor_pct`
  * grade-mix rule : if the share of Class II items in stock exceeds `class_ii_share_threshold`,
                     Class II items get an extra `class_ii_extra_markdown_pct` markdown (clear the bin)

`simulate_extra_days` lets you demonstrate ageing during the presentation without waiting.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

from app.db import Database


class PricingJob:
    def __init__(self, db: Database, pricing_provider: Callable[[], dict]):
        self.db = db
        self.pricing_provider = pricing_provider

    def run(self, simulate_extra_days: float = 0.0) -> dict[str, Any]:
        policy = self.pricing_provider()["markdown_policy"]
        items = self.db.items(status="in_stock")
        if not items:
            return {"repriced": 0, "checked": 0, "class_ii_share": 0.0, "changes": []}
        n_ii = sum(1 for i in items if i["quality_class"] == "II")
        share_ii = n_ii / len(items)
        now = datetime.now(timezone.utc)
        changes = []
        for it in items:
            stocked = datetime.fromisoformat(it["stocked_at"])
            days = (now - stocked).total_seconds() / 86400.0 + simulate_extra_days
            markdown = max(0.0, days - policy["grace_days"]) * policy["daily_markdown_pct"]
            reasons = []
            if markdown > 0:
                reasons.append(f"age {days:.1f} d")
            if it["quality_class"] == "II" and share_ii > policy["class_ii_share_threshold"]:
                markdown += policy["class_ii_extra_markdown_pct"]
                reasons.append(f"Class II share {share_ii:.0%}")
            markdown = min(markdown, 100.0 - policy["floor_pct"])
            new_price = round(it["base_price_per_kg"] * (1 - markdown / 100.0), 2)
            if abs(new_price - it["price_per_kg"]) >= 0.01:
                new_value = round(new_price * it["est_weight_g"] / 1000.0, 3)
                self.db.update_price(it["id"], new_price, new_value, ", ".join(reasons) or "price restored")
                changes.append({"item_id": it["id"], "old": it["price_per_kg"], "new": new_price, "reason": ", ".join(reasons)})
        return {"repriced": len(changes), "checked": len(items), "class_ii_share": round(share_ii, 2), "changes": changes}
