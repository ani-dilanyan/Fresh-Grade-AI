// FreshGrade AI — live camera preview (WebSocket) and snapshot inspection.
(() => {
  const video = document.getElementById("video");
  const overlay = document.getElementById("overlay");
  const ctx = overlay.getContext("2d");
  const startBtn = document.getElementById("startBtn");
  const inspectBtn = document.getElementById("inspectBtn");
  const select = document.getElementById("cameraSelect");
  const statusEl = document.getElementById("status");
  const detList = document.getElementById("detList");
  const calibState = document.getElementById("calibState");
  const placeholder = document.getElementById("placeholder");
  const liveChip = document.getElementById("liveChip");
  const fpsEl = document.getElementById("fps");
  const useLlm = document.getElementById("useLlm");

  const grab = document.createElement("canvas");
  let stream = null, ws = null, waiting = false, running = false, busy = false;
  let frames = 0, lastFpsT = performance.now(), lastDets = [];
  const COLORS = { apple: "#e74c3c", tomato: "#ff6f3c", orange: "#f39c12", banana: "#f1c40f", lemon: "#d4e157" };

  const say = (msg) => { statusEl.textContent = msg; };

  async function listCameras() {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const cams = devices.filter((d) => d.kind === "videoinput");
      const current = select.value;
      select.innerHTML = '<option value="">Default camera</option>';
      cams.forEach((c, i) => {
        const o = document.createElement("option");
        o.value = c.deviceId; o.textContent = c.label || `Camera ${i + 1}`;
        select.appendChild(o);
      });
      select.value = current;
    } catch (e) { /* ignore */ }
  }

  async function startCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      say("This browser cannot access the camera here. Open the app at http://localhost:8000 in Chrome, Edge or Firefox, or use Upload.");
      return;
    }
    stopCamera();
    const constraints = { audio: false, video: select.value ? { deviceId: { exact: select.value }, width: { ideal: 1280 }, height: { ideal: 720 } }
                                                         : { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "environment" } };
    try {
      stream = await navigator.mediaDevices.getUserMedia(constraints);
    } catch (e) {
      say("Camera access was blocked or no camera found (" + e.name + "). Allow camera access, or use Upload.");
      return;
    }
    video.srcObject = stream;
    await video.play();
    placeholder.hidden = true;
    liveChip.hidden = false;
    inspectBtn.disabled = false;
    startBtn.textContent = "Restart camera";
    await listCameras();
    connect();
    inspectBtn.focus();  // keyboard users can press Space/Enter right away
    say("Camera on. Press Space or Inspect now to grade the fruit.");
  }

  function stopCamera() {
    running = false;
    if (ws) { try { ws.close(); } catch (e) {} ws = null; }
    if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
  }

  function connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws/live`);
    ws.binaryType = "arraybuffer";
    ws.onopen = () => { running = true; waiting = false; sendFrame(); };
    ws.onmessage = (ev) => {
      waiting = false;
      try {
        const res = JSON.parse(ev.data);
        if (!res.error) { lastDets = res.detections; updatePanel(res); }
      } catch (e) {}
      frames++;
      const now = performance.now();
      if (now - lastFpsT > 1000) { fpsEl.textContent = `${frames} fps`; frames = 0; lastFpsT = now; }
      setTimeout(sendFrame, 60);
    };
    ws.onclose = () => { if (running) { running = false; setTimeout(() => stream && connect(), 1000); } };
    ws.onerror = () => say("Live preview connection problem; retrying…");
    requestAnimationFrame(draw);
  }

  function sendFrame() {
    if (!running || waiting || !ws || ws.readyState !== 1 || !video.videoWidth) { if (running) setTimeout(sendFrame, 200); return; }
    const w = 640, h = Math.round(640 * video.videoHeight / video.videoWidth);
    grab.width = w; grab.height = h;
    grab.getContext("2d").drawImage(video, 0, 0, w, h);
    waiting = true;
    grab.toBlob((b) => { if (b && ws && ws.readyState === 1) b.arrayBuffer().then((buf) => ws.send(buf)); else waiting = false; }, "image/jpeg", 0.7);
  }

  function draw() {
    if (!stream) return;
    const vw = video.videoWidth, vh = video.videoHeight;
    if (vw) {
      const rect = overlay.getBoundingClientRect();
      overlay.width = rect.width; overlay.height = rect.height;
      const scale = Math.min(rect.width / vw, rect.height / vh);
      const dw = vw * scale, dh = vh * scale, ox = (rect.width - dw) / 2, oy = (rect.height - dh) / 2;
      ctx.clearRect(0, 0, overlay.width, overlay.height);
      ctx.lineWidth = 3; ctx.font = "600 15px system-ui, sans-serif";
      for (const d of lastDets) {
        const [x1, y1, x2, y2] = d.box;
        const x = ox + x1 * dw, y = oy + y1 * dh, w = (x2 - x1) * dw, h = (y2 - y1) * dh;
        const c = COLORS[d.fruit_type] || "#2ecc71";
        ctx.strokeStyle = c; ctx.strokeRect(x, y, w, h);
        const label = `${d.fruit_type} ${(d.confidence * 100).toFixed(0)}%`;
        const tw = ctx.measureText(label).width + 12;
        ctx.fillStyle = "rgba(0,0,0,.72)"; ctx.fillRect(x, Math.max(0, y - 24), tw, 22);
        ctx.fillStyle = "#fff"; ctx.fillText(label, x + 6, Math.max(16, y - 8));
      }
    }
    requestAnimationFrame(draw);
  }

  function updatePanel(res) {
    detList.innerHTML = "";
    if (!res.detections.length) { detList.innerHTML = '<li class="muted">No fruit in view yet.</li>'; }
    res.detections.forEach((d, i) => {
      const li = document.createElement("li");
      li.textContent = `#${i + 1} ${d.fruit_type} (${(d.confidence * 100).toFixed(0)}% confidence)`;
      detList.appendChild(li);
    });
    calibState.textContent = res.calibrated ? "Calibration card: detected ✓ (real size will be measured)" : "Calibration card: not detected";
  }

  async function inspect() {
    if (busy || !stream || !video.videoWidth) return;
    busy = true; inspectBtn.disabled = true;
    say("Analysing the fruit… please hold still.");
    grab.width = video.videoWidth; grab.height = video.videoHeight;
    grab.getContext("2d").drawImage(video, 0, 0);
    const blob = await new Promise((r) => grab.toBlob(r, "image/jpeg", 0.92));
    const fd = new FormData();
    fd.append("file", blob, "snapshot.jpg");
    fd.append("source", "live");
    fd.append("use_llm", useLlm && useLlm.checked ? "true" : "false");
    try {
      const r = await fetch("/api/inspect", { method: "POST", body: fd });
      const data = await r.json();
      if (!r.ok) { say(data.detail || "Inspection failed."); busy = false; inspectBtn.disabled = false; return; }
      const f = data.fruits.map((x) => `${x.fruit.fruit_type} ${x.grade.quality_class}`).join(", ");
      say(`Done: ${f}. Opening results…`);
      window.location.href = `/results/${data.inspection_id}`;
    } catch (e) {
      say("Could not reach the server. Is the app still running?");
      busy = false; inspectBtn.disabled = false;
    }
  }

  startBtn.addEventListener("click", startCamera);
  select.addEventListener("change", () => { if (stream) startCamera(); });
  inspectBtn.addEventListener("click", inspect);
  document.addEventListener("keydown", (e) => {
    const tag = (e.target.tagName || "").toLowerCase();
    if (e.code === "Space" && !["input", "select", "textarea", "button", "a"].includes(tag)) { e.preventDefault(); inspect(); }
  });
  window.addEventListener("beforeunload", stopCamera);
})();
