"""Member 4 — end-to-end tests through the HTTP API, WebSocket and HTML pages."""
import json


def post_image(client, path, source="upload"):
    with open(path, "rb") as f:
        return client.post("/api/inspect", files={"file": (path.name, f, "image/jpeg")}, data={"source": source, "use_llm": "false"})


def test_health(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["detector"] == "classical" and not h["llm_enabled"]


def test_full_flow_inspect_inventory_reports(client, samples):
    r = post_image(client, samples / "mixed_with_card.jpg")
    assert r.status_code == 200, r.text
    data = r.json()
    iid = data["inspection_id"]
    assert len(data["fruits"]) == 3 and data["calibrated"]
    for f in data["fruits"]:
        assert f["grade"]["quality_class"] in ("Extra", "I", "II", "Reject")
        assert f["grade"]["reasons"]
        assert "mask" not in f["fruit"]  # in-memory masks are never serialised

    # media files are served
    assert client.get(f"/media/{data['overlay_path']}").status_code == 200
    # history + detail
    assert any(row["id"] == iid for row in client.get("/api/inspections").json())
    assert client.get(f"/api/inspections/{iid}").json()["inspection_id"] == iid
    # inventory updated automatically
    inv = client.get("/api/inventory").json()
    assert inv["stats"]["inspections"] >= 1 and len(inv["items"]) >= 3
    # reports
    pdf = client.get(f"/api/reports/quality/{iid}")
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"
    sheet = client.get("/api/reports/sorting-sheet?format=pdf")
    assert sheet.status_code == 200 and sheet.content[:4] == b"%PDF"
    csv = client.get("/api/reports/sorting-sheet?format=csv").text
    assert csv.startswith("bin_code,item_id")
    assert client.get("/api/reports/inventory.csv").status_code == 200
    assert json.loads(client.get(f"/api/reports/inspection/{iid}.json").content)["inspection_id"] == iid
    # repricing endpoint
    assert client.post("/api/inventory/reprice?simulate_days=5").json()["checked"] >= 1


def test_html_pages_render(client, samples):
    iid = post_image(client, samples / "apple_bruised.jpg").json()["inspection_id"]
    for url in ["/", "/upload", "/inventory", "/history", "/reports", "/settings", "/settings?fruit=banana", f"/results/{iid}"]:
        r = client.get(url)
        assert r.status_code == 200, url
        assert '<main id="main"' in r.text  # landmark for screen readers
    assert client.get("/results/doesnotexist").status_code == 404


def test_no_js_upload_form_redirects(client, samples):
    with open(samples / "apple_healthy.jpg", "rb") as f:
        r = client.post("/upload", files={"file": ("a.jpg", f, "image/jpeg")}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].startswith("/results/")


def test_invalid_upload_and_no_fruit(client):
    r = client.post("/api/inspect", files={"file": ("x.jpg", b"not an image", "image/jpeg")})
    assert r.status_code == 400
    import cv2
    import numpy as np

    blank = cv2.imencode(".jpg", np.full((300, 300, 3), 235, np.uint8))[1].tobytes()
    r = client.post("/api/inspect", files={"file": ("b.jpg", blank, "image/jpeg")})
    assert r.status_code == 422 and "No fruit" in r.json()["detail"]


def test_live_websocket(client, samples):
    data = (samples / "mixed_with_card.jpg").read_bytes()
    with client.websocket_connect("/ws/live") as ws:
        ws.send_bytes(data)
        msg = json.loads(ws.receive_text())
        assert len(msg["detections"]) == 3 and msg["calibrated"] is True
        for d in msg["detections"]:
            assert all(0 <= v <= 1 for v in d["box"])
        ws.send_bytes(b"garbage")
        assert "error" in json.loads(ws.receive_text())


def test_ontology_edit_changes_grade(client, samples):
    onto = client.get("/api/ontology").json()
    # make Extra/I/II impossible for apples → bruised apple must become Reject
    for cls in ("Extra", "I", "II"):
        onto["fruit_profiles"]["apple"]["classes"][cls]["max_defect_pct"] = 0.0
    assert client.put("/api/ontology", json=onto).status_code == 200
    g = post_image(client, samples / "apple_bruised.jpg").json()["fruits"][0]["grade"]
    assert g["quality_class"] == "Reject"
    # invalid ontology is refused
    onto["fruit_profiles"]["apple"]["classes"]["I"]["min_uniformity"] = 5
    assert client.put("/api/ontology", json=onto).status_code == 400
    # restore defaults through the settings form
    r = client.post("/settings", data={"fruit": "apple", "action": "restore"}, follow_redirects=False)
    assert r.status_code == 303
    assert client.get("/api/ontology").json()["fruit_profiles"]["apple"]["classes"]["I"]["max_defect_pct"] == 3.0


def test_calibration_card_pdf(client):
    r = client.get("/calibration-card.pdf")
    assert r.status_code == 200 and r.content[:4] == b"%PDF"
