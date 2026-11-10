"""Member 3 — Market value calculation from quality class and estimated weight."""
from __future__ import annotations

from typing import Callable


class PricingEngine:
    def __init__(self, pricing_provider: Callable[[], dict]):
        self.pricing_provider = pricing_provider

    @property
    def currency(self) -> str:
        return self.pricing_provider().get("currency", "EUR")

    def price_per_kg(self, fruit: str, quality: str) -> float:
        table = self.pricing_provider()["price_per_kg"]
        return float(table.get(fruit, {}).get(quality, 0.0))

    def price(self, fruit: str, quality: str, weight_g: float) -> tuple[float, float]:
        per_kg = self.price_per_kg(fruit, quality)
        return round(per_kg, 2), round(per_kg * weight_g / 1000.0, 3)

    def bin_code(self, fruit: str, quality: str) -> str:
        if quality == "Reject":
            return "REJ"
        prefix = self.pricing_provider().get("bin_prefix", {}).get(fruit, fruit[:3].upper())
        return f"{prefix}-{quality.upper()}"
