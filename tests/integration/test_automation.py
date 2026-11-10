"""Member 4 — inspection service + database + event bus + automation (inventory, pricing job)."""
import asyncio
import tempfile
from pathlib import Path

import cv2
import pytest

from app.container import Container
from contracts.schemas import InspectionResult


@pytest.fixture(scope="module")
def container():
    import os

    os.environ["FRESHGRADE_DATA_DIR"] = tempfile.mkdtemp(prefix="fg_int_")
    return Container()


def test_pipeline_updates_inventory(container, samples):
    img = cv2.imread(str(samples / "mixed_with_card.jpg"))
    res = asyncio.run(container.inspection.inspect(img, source="upload"))
    assert isinstance(res, InspectionResult) and len(res.fruits) == 3 and res.calibrated
    assert (container.media_root / res.overlay_path).exists()
    items = container.db.items(inspection_id=res.inspection_id)
    assert len(items) == 3
    assert container.db.get_inspection(res.inspection_id)["inspection_id"] == res.inspection_id
    # every fruit has a ledger event
    assert len([l for l in container.db.ledger() if l["item_id"] in {i["id"] for i in items}]) == 3


def test_reject_is_recorded_as_discarded(container, samples):
    img = cv2.imread(str(samples / "apple_rotten.jpg"))
    res = asyncio.run(container.inspection.inspect(img))
    assert res.fruits[0].grade.quality_class.value == "Reject"
    item = container.db.items(inspection_id=res.inspection_id)[0]
    assert item["status"] == "discarded" and item["bin_code"] == "REJ"


def test_pricing_job_marks_down_aged_stock(container):
    before = {i["id"]: i["price_per_kg"] for i in container.db.items(status="in_stock")}
    assert before
    out = container.pricing_job.run(simulate_extra_days=6)  # 6 days → (6-2)*5% = 20% markdown
    assert out["repriced"] >= 1
    after = {i["id"]: i for i in container.db.items(status="in_stock")}
    for iid, old in before.items():
        if old > 0:
            assert after[iid]["price_per_kg"] < old
            assert after[iid]["price_per_kg"] >= round(after[iid]["base_price_per_kg"] * 0.5, 2) - 0.01
    assert container.db.price_history()


def test_price_floor(container):
    container.pricing_job.run(simulate_extra_days=60)
    for i in container.db.items(status="in_stock"):
        assert i["price_per_kg"] == pytest.approx(i["base_price_per_kg"] * 0.5, abs=0.011)


def test_no_fruit_raises(container):
    import numpy as np

    from app.services.inspection_service import NoFruitFound

    with pytest.raises(NoFruitFound):
        asyncio.run(container.inspection.inspect(np.full((400, 400, 3), 230, np.uint8)))
