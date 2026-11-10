"""Member 3 — Vision-LLM inspector (optional expert second opinion).

Sends the fruit crop to the Claude Messages API and asks for a strict JSON verdict using the
project ontology. Returns None on any problem (no key, offline, timeout, bad JSON), so the
system always works with local models alone.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
from typing import Optional

import httpx

from contracts.schemas import LLMOpinion, QualityClass

log = logging.getLogger("freshgrade.llm")

API_URL = "https://api.anthropic.com/v1/messages"

PROMPT = """You are an expert fresh-produce quality inspector working to UNECE-style marketing standards.
The image shows one {fruit} (the detector's guess). Inspect only the visible surface.

Quality classes:
- "Extra": superior quality, practically free from defects, even colour.
- "I": good quality, slight skin defects (roughly up to 3% of the surface), slight colour defects.
- "II": minimum requirements met, skin defects up to roughly 8% of the surface.
- "Reject": rot, mould, decay, large bruises, or damage beyond Class II.

Allowed defect type ids: bruise, rot, scab_spot, crack, sunburn_discoloration, insect_damage.

Reply with ONLY a JSON object, no other text:
{{"fruit_type_seen": "<fruit name>", "defect_types": ["<ids>"], "suggested_class": "Extra|I|II|Reject", "reasoning": "<one or two sentences>"}}"""


class NoLLMInspector:
    name = "disabled"
    enabled = False

    async def inspect(self, crop_jpeg: bytes, fruit_type: str) -> Optional[LLMOpinion]:
        return None


class AnthropicLLMInspector:
    name = "anthropic"
    enabled = True

    def __init__(self, model: str, timeout_s: float = 12.0, max_tokens: int = 500, api_key: Optional[str] = None):
        self.model = model
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.last_error: Optional[str] = None

    @staticmethod
    def parse(text: str) -> Optional[dict]:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None

    async def inspect(self, crop_jpeg: bytes, fruit_type: str) -> Optional[LLMOpinion]:
        t0 = time.perf_counter()
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(crop_jpeg).decode()},
                        },
                        {"type": "text", "text": PROMPT.format(fruit=fruit_type)},
                    ],
                }
            ],
        }
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                r = await client.post(API_URL, json=payload, headers=headers)
            if r.status_code != 200:
                self.last_error = f"HTTP {r.status_code}: {r.text[:200]}"
                log.warning("LLM inspector error %s", self.last_error)
                return None
            text = "".join(b.get("text", "") for b in r.json().get("content", []) if b.get("type") == "text")
            data = self.parse(text)
            if not data:
                self.last_error = "Could not parse JSON from model reply"
                return None
            cls = str(data.get("suggested_class", "")).strip()
            cls = {"EXTRA": "Extra", "1": "I", "2": "II", "CLASS I": "I", "CLASS II": "II", "REJECT": "Reject"}.get(cls.upper(), cls)
            if cls not in {q.value for q in QualityClass}:
                self.last_error = f"Unknown class '{cls}'"
                return None
            self.last_error = None
            return LLMOpinion(
                defect_types=[str(d) for d in data.get("defect_types", [])][:6],
                suggested_class=QualityClass(cls),
                fruit_type_seen=str(data.get("fruit_type_seen", ""))[:40] or None,
                reasoning=str(data.get("reasoning", ""))[:400],
                latency_ms=int((time.perf_counter() - t0) * 1000),
                provider=f"anthropic:{self.model}",
            )
        except (httpx.HTTPError, ValueError) as e:
            self.last_error = f"{type(e).__name__}: {e}"
            log.warning("LLM inspector unavailable: %s", self.last_error)
            return None


def build_llm_inspector(cfg: dict):
    if cfg.get("type") == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicLLMInspector(cfg.get("model", "claude-sonnet-5-5"), float(cfg.get("timeout_s", 12)), int(cfg.get("max_tokens", 500)))
    if cfg.get("type") == "mock":
        from mocks.mock_modules import MockLLMInspector

        return MockLLMInspector()
    return NoLLMInspector()
