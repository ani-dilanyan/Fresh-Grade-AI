"""Tiny in-process publish/subscribe bus (Member 4).

The inspection pipeline publishes `InspectionCompleted`; automation services subscribe.
This keeps grading code unaware of inventory, pricing or reporting (loose coupling).
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Callable

log = logging.getLogger("freshgrade.events")

INSPECTION_COMPLETED = "InspectionCompleted"


class EventBus:
    def __init__(self) -> None:
        self._subs: dict[str, list[Callable[[dict[str, Any]], None]]] = defaultdict(list)

    def subscribe(self, event: str, handler: Callable[[dict[str, Any]], None]) -> None:
        self._subs[event].append(handler)

    def publish(self, event: str, payload: dict[str, Any]) -> list[str]:
        errors = []
        for h in self._subs[event]:
            try:
                h(payload)
            except Exception as e:  # one failing subscriber must not break the others
                log.exception("Event handler %s failed", getattr(h, "__name__", h))
                errors.append(f"{getattr(h, '__qualname__', h)}: {e}")
        return errors
