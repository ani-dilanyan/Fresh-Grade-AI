"""Configuration loading (settings.yaml, ontology.json, pricing.yaml, .env)."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = Path(os.environ.get("FRESHGRADE_CONFIG_DIR", ROOT / "config"))

_lock = threading.Lock()


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Minimal .env loader (KEY=VALUE lines). Existing environment variables win."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else ROOT / p


def load_settings() -> dict[str, Any]:
    with open(CONFIG_DIR / "settings.yaml", encoding="utf-8") as f:
        s = yaml.safe_load(f)
    # environment overrides, e.g. FRESHGRADE_DETECTOR=classical, FRESHGRADE_LLM=disabled
    for env, section in (("FRESHGRADE_DETECTOR", "detector"), ("FRESHGRADE_CONDITION", "condition"), ("FRESHGRADE_LLM", "llm")):
        if os.environ.get(env):
            s[section]["type"] = os.environ[env]
    return s


def load_pricing() -> dict[str, Any]:
    with open(CONFIG_DIR / "pricing.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ontology_path() -> Path:
    return CONFIG_DIR / "ontology.json"


def load_ontology() -> dict[str, Any]:
    with open(ontology_path(), encoding="utf-8") as f:
        return json.load(f)


def save_ontology(data: dict[str, Any]) -> None:
    with _lock:
        tmp = ontology_path().with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(ontology_path())


def data_dir(settings: dict[str, Any]) -> Path:
    d = os.environ.get("FRESHGRADE_DATA_DIR") or settings["paths"]["data_dir"]
    p = resolve(d)
    p.mkdir(parents=True, exist_ok=True)
    return p
