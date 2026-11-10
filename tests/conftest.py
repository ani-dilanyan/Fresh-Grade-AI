"""Test configuration: isolated data + config directories, classical detector, no LLM, no network."""
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_TMP = Path(tempfile.mkdtemp(prefix="freshgrade_test_"))
shutil.copytree(ROOT / "config", _TMP / "config")
os.environ["FRESHGRADE_CONFIG_DIR"] = str(_TMP / "config")
os.environ["FRESHGRADE_DATA_DIR"] = str(_TMP / "data")
os.environ["FRESHGRADE_DETECTOR"] = "classical"
os.environ["FRESHGRADE_LLM"] = "disabled"
os.environ["FRESHGRADE_CONDITION"] = "classical"  # deterministic; the trained model has its own tests
os.environ.pop("ANTHROPIC_API_KEY", None)

SAMPLES = ROOT / "data" / "samples"


def _ensure_samples():
    if not (SAMPLES / "mixed_with_card.jpg").exists():
        sys.path.insert(0, str(ROOT / "tools"))
        import make_samples

        make_samples.make_all()


_ensure_samples()


@pytest.fixture(scope="session")
def samples() -> Path:
    return SAMPLES


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c
