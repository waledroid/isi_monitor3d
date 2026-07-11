// Compressed-video passthrough player (WebCodecs) — window.__passthrough.
//
// The camera already compressed the video: isistream taps the ORIGINAL
// H.264/H.265 bitstream and the server relays it over the shared /ws/video
// socket as `camh264:<camera_id>` (see api/routes_ws_video.py). This module
// hardware-decodes the access units in the browser (VideoDecoder), paints
// them on the cam view's <canvas>, and draws the detection overlays
// CLIENT-SIDE from the /ws/overlays observation feed — replacing the
// server-side overlay drawing + JPEG encode for the big CAM views.
//
// Classic deferred script (like video_ws.js) so big_panel.js (classic) can
// call it; loaded right after video_ws.js in base.html.
//
// API (all per camera; element ids by convention `${camId}-img/-video/-det`):
//   __passthrough.activate(camId)   — try passthrough for that cam view; falls
//                                     back to the JPEG stream BY ITSELF when
//                                     unavailable (attaches `cam:<id>`)
//   __passthrough.deactivate(camId) — tear down (view hidden / warp toggled);
//                                     never touches the `cam:<id>` JPEG sub
//   __passthrough.frameSize(camId)  — [w, h] of the decoded frame while active
//                                     (live_overlay.js uses it in place of the
//                                     now-srcless <img>.naturalWidth)
//
// FALLBACK CHAIN (each step lands on the classic server-drawn JPEG stream):
//   ui-settings `video_passthrough` === false → JPEG (camh264 never subscribed)
//   no window.VideoDecoder (no WebCodecs)     → JPEG
//   INIT {"available": false} (socket absent, isistream stopped/frames mode,
//     server send backlog)                     → JPEG
//   no codec candidate passes isConfigSupported (e.g. HEVC w/o HW) → JPEG
//   decoder error / decode() throw / no frame within the watchdog → JPEG
// Fallback is sticky per session; a `config:saved` event resets and retries.
//
// VideoFrames are closed in a try/finally EVERY time — leaking them exhausts
// the decoder's frame pool and stalls decode permanently.
//
// Perf HUD: open the dashboard with ?perfhud=1 — per-stream decode queue
// depth, frames received/presented/dropped, avg decode ms, avg overlay-draw
// ms, and long-tasks/min (PerformanceObserver). This is the "does it slow the
// browser" instrument.

(function () {
  "use strict";

  // COCO-17 skeleton edges (matches the server-side pose overlay).
  const COCO_EDGES = [
    [0, 1], [0, 2], [1, 3], [2, 4],          // head
    [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], // arms
    [5, 11], [6, 12], [11, 12],              // torso
    [11, 13], [13, 15], [12, 14], [14, 16],  // legs
  ];
  const KEYPOINT_MIN_CONF = 0.35;

  // Annex-B in-band parameter sets (no `description` in the config = annexb
  // bitstream format per the WebCodecs registry). Try progressively higher
  // profiles/levels; the first supported wins.
  const CODEC_CANDIDATES = {
    h264: ["avc1.42E01F", "avc1.4D401F", "avc1.64001F", "avc1.640028"],
    h265: ["hvc1.1.6.L120.90", "hev1.1.6.L120.90", "hvc1.1.6.L153.B0", "hev1.1.6.L153.B0"],
  };

  const WATCHDOG_MS = 8000;      // sub → first decoded frame, else fall back
  const MAX_BUFFERED_AUS = 240;  // pre-decoder-ready buffer (~8 s at 30 fps)
  // Robustness: a decoder falling behind (queue piling up) renders visibly
  // smeared/ghosted frames until the next keyframe — the "giant blurry label"
  // artifact. If the queue exceeds this, flush and resync at the next keyframe
  // (sub-second) instead of painting smears.
  const MAX_DECODE_QUEUE = 6;
  // If no keyframe has been decoded in this long while active, the H.264
  // passthrough is unhealthy (lost keyframe / long GOP under loss) — hold the
  // last GOOD frame and stop painting deltas; fall back to the clean
  // server-JPEG path once it stays unhealthy past the grace.
  const KEYFRAME_STALE_MS = 4000;
  const KEYFRAME_GIVEUP_MS = 9000;
  const PREFS_REFRESH_MS = 5000; // show_* toggles auto-save server-side; poll

  const PERF_HUD = new URLSearchParams(location.search).get("perfhud") === "1";

  const sessions = new Map();    // camId -> session
  let prefs = { passthrough: true, showBoxes: true, showMasks: true, showNodes: true };
  let prefsLoaded = null;        // promise
  let prefsTimer = null;

  // ---- prefs (/api/ui-settings) -------------------------------------------

  function loadPrefs() {
    if (!prefsLoaded) {
      prefsLoaded = fetch("/api/ui-settings")
        .then((r) => (r.ok ? r.json() : {}))
        .then((ui) => {
          prefs = {
            passthrough: ui.video_passthrough !== false,   // default ON
            showBoxes: ui.show_boxes !== false,
            showMasks: ui.show_masks !== false,
            showNodes: ui.show_nodes !== false,
          };
          // The SERVER owns the class palette — same colours as the zone
          // panels and their twins (which are drawn server-side).
          if (ui.class_colors && typeof ui.class_colors === "object") {
            classColors = ui.class_colors;
          }
          if (typeof ui.class_color_default === "string") {
            classColorDefault = ui.class_color_default;
          }
        })
        .catch(() => { /* keep defaults */ });
    }
    return prefsLoaded;
  }

  function schedulePrefsRefresh() {
    if (prefsTimer !== null) return;
    prefsTimer = setInterval(() => {
      if (sessions.size === 0) { clearInterval(prefsTimer); prefsTimer = null; return; }
      prefsLoaded = null;
      loadPrefs();
    }, PREFS_REFRESH_MS);
  }

  // ---- perf HUD instrumentation --------------------------------------------

  let longTaskTimes = [];
  if (PERF_HUD && "PerformanceObserver" in window) {
    try {
      new PerformanceObserver((list) => {
        for (const _ of list.getEntries()) longTaskTimes.push(performance.now());
      }).observe({ type: "longtask", buffered: true });
    } catch { /* longtask not supported */ }
  }
  function longTasksPerMin() {
    const cut = performance.now() - 60000;
    longTaskTimes = longTaskTimes.filter((t) => t > cut);
    return longTaskTimes.length;
  }
  if (PERF_HUD) {
    setInterval(() => {                       // keep HUD numbers moving even
      for (const s of sessions.values()) {    // when observations are quiet
        if (s.active) drawOverlay(s);
      }
    }, 500);
  }

  // ---- session lifecycle ----------------------------------------------------

  function makeSession(camId) {
    const img = document.getElementById(`${camId}-img`);
    const video = document.getElementById(`${camId}-video`);
    const det = document.getElementById(`${camId}-det`);
    if (!img || !video || !det) return null;
    return {
      camId,
      els: { img, video, det },
      ctxV: video.getContext("2d"),
      ctxD: det.getContext("2d"),
      decoder: null,
      configuring: false,
      pendingAUs: [],
      needKeyframe: true,
      active: false,          // true once the first frame painted
      fellBack: false,
      stopped: false,
      watchdog: null,
      videoW: 0,
      videoH: 0,
      obs: null,              // latest /ws/overlays payload
      decodeT0: new Map(),    // chunk timestamp -> submit time (decode-ms)
      lastKeyMs: 0,           // performance.now() of the last decoded keyframe
      degraded: false,        // holding last-good frame, awaiting a keyframe
      healthTimer: null,
      stats: { au: 0, presented: 0, dropped: 0, resyncs: 0, decodeMs: 0, overlayMs: 0 },
    };
  }

  async function activate(camId) {
    if (sessions.has(camId)) return;          // attempting, active, or fallen back
    if (suspendedCams) {
      // Authoring (draw/calibrate) in progress — serve JPEG until it ends so
      // the <img> keeps real pixels + naturalWidth for the click-capture math.
      if (!suspendedCams.includes(camId)) suspendedCams.push(camId);
      const img = document.getElementById(`${camId}-img`);
      if (img && window.__videoWS) window.__videoWS.attach(img, `cam:${camId}`);
      return;
    }
    const s = makeSession(camId);
    if (!s) {                                  // markup missing — plain JPEG
      const img = document.getElementById(`${camId}-img`);
      if (img && window.__videoWS) window.__videoWS.attach(img, `cam:${camId}`);
      return;
    }
    sessions.set(camId, s);
    schedulePrefsRefresh();
    await loadPrefs();
    if (s.stopped) return;                    // deactivated while prefs loaded
    if (!prefs.passthrough) { fallBack(s, "video_passthrough disabled"); return; }
    if (typeof window.VideoDecoder === "undefined") {
      fallBack(s, "WebCodecs VideoDecoder not available");
      return;
    }
    if (!window.__videoWS || !window.__videoWS.attachRaw) {
      fallBack(s, "video transport missing attachRaw");
      return;
    }
    s.watchdog = setTimeout(() => {
      if (!s.active && !s.fellBack && !s.stopped) fallBack(s, "no decoded frame (watchdog)");
    }, WATCHDOG_MS);
    window.__videoWS.attachRaw(`camh264:${camId}`, (payload) => onPayload(s, payload));
  }

  function deactivate(camId) {
    if (suspendedCams) suspendedCams = suspendedCams.filter((c) => c !== camId);
    const s = sessions.get(camId);
    if (!s) return;
    s.stopped = true;
    teardownDecoding(s);
    sessions.delete(camId);
    syncOverlaySocket();
    // NOTE: the `cam:<id>` JPEG sub (ours from a fallback, or big_panel's) is
    // deliberately left alone — big_panel owns attach/detach for that kind.
  }

  function teardownDecoding(s) {
    if (s.watchdog !== null) { clearTimeout(s.watchdog); s.watchdog = null; }
    if (s.decoder) {
      try { s.decoder.close(); } catch { /* already closed */ }
      s.decoder = null;
    }
    if (s.healthTimer !== null) { clearInterval(s.healthTimer); s.healthTimer = null; }
    s.pendingAUs = [];
    s.decodeT0.clear();
    if (window.__videoWS) window.__videoWS.detach(`camh264:${s.camId}`);
    // Wipe the canvas — otherwise the last (possibly smeared) frame lingers
    // under a swap to JPEG or a re-decode.
    try { s.ctxV.clearRect(0, 0, s.els.video.width, s.els.video.height); } catch { /* noop */ }
    try { s.ctxD.clearRect(0, 0, s.els.det.width, s.els.det.height); } catch { /* noop */ }
    s.els.video.classList.add("hidden");
    s.els.det.classList.add("hidden");
    s.els.img.classList.remove("pt-hidden");
    s.active = false;
  }

  function fallBack(s, reason) {
    if (s.fellBack || s.stopped) return;
    s.fellBack = true;
    console.warn(`passthrough[${s.camId}]: falling back to JPEG — ${reason}`);
    teardownDecoding(s);
    syncOverlaySocket();
    if (window.__videoWS) window.__videoWS.attach(s.els.img, `cam:${s.camId}`);
  }

  // ---- bitstream handling ---------------------------------------------------

  function onPayload(s, payload) {
    if (s.stopped || s.fellBack || payload.length < 1) return;
    if (payload[0] === 0) {                    // INIT
      let init;
      try {
        init = JSON.parse(new TextDecoder().decode(payload.subarray(1)));
      } catch { fallBack(s, "unparseable INIT"); return; }
      if (!init.available) { fallBack(s, `stream unavailable: ${init.reason || "?"}`); return; }
      setupDecoder(s, init.codec);             // async; AUs buffer meanwhile
      return;
    }
    if (payload[0] !== 1 || payload.length < 10) return;   // unknown type
    const dv = new DataView(payload.buffer, payload.byteOffset, payload.byteLength);
    const au = {
      ts: dv.getFloat64(1, true),
      key: (dv.getUint8(9) & 1) === 1,
      data: payload.subarray(10),
    };
    s.stats.au++;
    if (!s.decoder || s.configuring) {
      // Decoder not ready yet (codec probe in flight): buffer, bounded. On
      // overflow drop everything and resync at the next keyframe.
      if (s.pendingAUs.length >= MAX_BUFFERED_AUS) {
        s.stats.dropped += s.pendingAUs.length;
        s.pendingAUs = [];
        s.needKeyframe = true;
      }
      s.pendingAUs.push(au);
      return;
    }
    decodeAU(s, au);
  }

  async function setupDecoder(s, codecName) {
    // A fresh INIT (reconnect / resubscribe) always restarts the decoder —
    // the stream restarts at a keyframe by contract.
    s.configuring = true;
    try {
      if (s.decoder) { try { s.decoder.close(); } catch { /* noop */ } s.decoder = null; }
      s.needKeyframe = true;
      const candidates = CODEC_CANDIDATES[codecName] || [];
      let chosen = null;
      for (const codec of candidates) {
        try {
          const res = await VideoDecoder.isConfigSupported({ codec, optimizeForLatency: true });
          if (res && res.supported) { chosen = codec; break; }
        } catch { /* malformed candidate for this UA — try the next */ }
      }
      if (s.stopped || s.fellBack) return;
      if (!chosen) { fallBack(s, `no supported decoder config for ${codecName}`); return; }
      const decoder = new VideoDecoder({
        output: (frame) => present(s, frame),
        error: (e) => fallBack(s, `decoder error: ${(e && e.message) || e}`),
      });
      decoder.configure({ codec: chosen, optimizeForLatency: true });
      s.decoder = decoder;
      const buffered = s.pendingAUs;
      s.pendingAUs = [];
      for (const au of buffered) decodeAU(s, au);
    } finally {
      s.configuring = false;
    }
  }

  function decodeAU(s, au) {
    if (!s.decoder || s.decoder.state !== "configured") return;
    // Decoder falling behind → drop to the next keyframe instead of queuing
    // deltas that will paint smeared. Costs one GOP of latency, kills the
    // ghosting.
    if (!s.needKeyframe && s.decoder.decodeQueueSize > MAX_DECODE_QUEUE) {
      s.needKeyframe = true;
      s.degraded = true;
      s.stats.resyncs++;
    }
    if (s.needKeyframe) {
      if (!au.key) { s.stats.dropped++; return; }
      s.needKeyframe = false;
    }
    if (au.key) { s.lastKeyMs = performance.now(); s.degraded = false; }
    const usTs = Math.round(au.ts * 1e6);      // capture_ts (s) → µs
    try {
      s.decodeT0.set(usTs, performance.now());
      if (s.decodeT0.size > 240) s.decodeT0.delete(s.decodeT0.keys().next().value);
      s.decoder.decode(new EncodedVideoChunk({
        type: au.key ? "key" : "delta",
        timestamp: usTs,
        data: au.data,
      }));
    } catch (e) {
      fallBack(s, `decode() threw: ${e.message || e}`);
    }
  }

  function present(s, frame) {
    const t0 = s.decodeT0.get(frame.timestamp);
    if (t0 !== undefined) {
      s.decodeT0.delete(frame.timestamp);
      const dt = performance.now() - t0;
      s.stats.decodeMs = s.stats.decodeMs ? s.stats.decodeMs * 0.9 + dt * 0.1 : dt;
    }
    try {
      if (s.stopped || s.fellBack) return;
      // Degraded (post-backlog, pre-keyframe): hold the last GOOD frame rather
      // than paint a smeared delta. present() still fires for in-flight deltas
      // decoded before the flush — skip them.
      if (s.degraded) return;
      const w = frame.displayWidth, h = frame.displayHeight;
      if (s.els.video.width !== w || s.els.video.height !== h) {
        s.els.video.width = w;
        s.els.video.height = h;
      }
      s.ctxV.drawImage(frame, 0, 0, w, h);
      s.videoW = w;
      s.videoH = h;
    } finally {
      frame.close();                           // MANDATORY — never leak VideoFrames
    }
    if (s.stopped || s.fellBack) return;
    s.stats.presented++;
    if (!s.active) markActive(s);
    // Overlay redraw rides the 15 Hz observation feed (plus the HUD interval
    // when enabled) — deliberately no per-video-frame overlay work.
  }

  function startHealthMonitor(s) {
    if (s.healthTimer !== null) return;
    s.lastKeyMs = performance.now();
    s.healthTimer = setInterval(() => {
      if (s.stopped || s.fellBack || !s.active) return;
      const age = performance.now() - s.lastKeyMs;
      if (age > KEYFRAME_STALE_MS && !s.degraded) {
        // No keyframe in a while → treat subsequent deltas as suspect: hold
        // the last good frame and wait for a keyframe.
        s.degraded = true;
        s.needKeyframe = true;
        s.stats.resyncs++;
      }
      if (age > KEYFRAME_GIVEUP_MS) {
        fallBack(s, `no keyframe for ${(age / 1000).toFixed(1)}s`);
      }
    }, 1000);
  }

  function markActive(s) {
    s.active = true;
    if (s.watchdog !== null) { clearTimeout(s.watchdog); s.watchdog = null; }
    startHealthMonitor(s);
    s.els.video.classList.remove("hidden");
    s.els.det.classList.remove("hidden");
    // visibility (not display): the <img> keeps its layout box so draw-mode /
    // overlay geometry that reads its client size stays valid.
    s.els.img.classList.add("pt-hidden");
    if (window.__videoWS) window.__videoWS.detach(`cam:${s.camId}`);  // stop any JPEG
    syncOverlaySocket();
  }

  // ---- observation feed (/ws/overlays) --------------------------------------

  let obsWs = null;
  let obsRetries = 0;
  let obsTimer = null;

  function activeOverlayCams() {
    const cams = [];
    for (const s of sessions.values()) {
      if (s.active && !s.fellBack && !s.stopped) cams.push(s.camId);
    }
    return cams;
  }

  function syncOverlaySocket() {
    const cams = activeOverlayCams();
    if (cams.length === 0) {
      if (obsWs) { try { obsWs.close(); } catch { /* noop */ } obsWs = null; }
      return;
    }
    if (obsWs && obsWs.readyState === WebSocket.OPEN) {
      obsWs.send(JSON.stringify({ cameras: cams }));
      return;
    }
    if (!obsWs || obsWs.readyState === WebSocket.CLOSED) connectOverlays();
    // CONNECTING/CLOSING: onopen (or the reconnect timer) sends the fresh set.
  }

  function connectOverlays() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    obsWs = new WebSocket(`${proto}://${location.host}/ws/overlays`);
    obsWs.onopen = () => {
      obsRetries = 0;
      const cams = activeOverlayCams();
      if (cams.length) obsWs.send(JSON.stringify({ cameras: cams }));
    };
    obsWs.onmessage = (ev) => {
      let msg;
      try { msg = JSON.parse(ev.data); } catch { return; }
      const s = sessions.get(msg.camera_id);
      if (!s || !s.active) return;
      s.obs = msg;
      drawOverlay(s);
    };
    obsWs.onclose = () => {
      obsWs = null;
      if (activeOverlayCams().length === 0 || obsTimer) return;
      const delay = Math.min(5000, 500 * 2 ** obsRetries++);
      obsTimer = setTimeout(() => { obsTimer = null; syncOverlaySocket(); }, delay);
    };
    obsWs.onerror = () => { try { obsWs.close(); } catch { /* noop */ } };
  }

  // ---- overlay drawing -------------------------------------------------------

  // MUST match monitor_web/overlay.py CLASS_COLORS_HEX — otherwise the same
  // pallet is green on the zone panels (server-drawn) and blue on the cam
  // views (client-drawn). The server sends the live palette on /api/ui-settings
  // (`class_colors`); this table is the fallback and is pinned equal by a test.
  const FALLBACK_CLASS_COLORS = {
    palette: "#50dc50",
    pallet: "#50dc50",
    carton: "#ff7878",
    polybag: "#78b4ff",
    person: "#ffd54f",
    forklift: "#ff7043",
  };
  const DEFAULT_CLASS_COLOR = "#ffffff";
  let classColors = FALLBACK_CLASS_COLORS;
  let classColorDefault = DEFAULT_CLASS_COLOR;

  function colorForClass(cls) {
    const key = String(cls || "").toLowerCase();
    return classColors[key] || classColorDefault;
  }

  function drawOverlay(s) {
    if (!s.active || !s.videoW) return;
    const t0 = performance.now();
    const det = s.els.det;
    // Same intrinsic size + object-fit as the video canvas → drawing in
    // decoded-frame pixel coordinates aligns exactly, no display math.
    if (det.width !== s.videoW || det.height !== s.videoH) {
      det.width = s.videoW;
      det.height = s.videoH;
    }
    const ctx = s.ctxD;
    ctx.clearRect(0, 0, det.width, det.height);
    const obs = s.obs;
    if (obs && Array.isArray(obs.dets)) {
      const sx = s.videoW / (obs.frame_wh[0] || s.videoW);
      const sy = s.videoH / (obs.frame_wh[1] || s.videoH);
      const lw = Math.max(2, Math.round(s.videoW / 640));
      ctx.font = `${Math.max(11, Math.round(s.videoW / 90))}px monospace`;
      for (const d of obs.dets) {
        const color = colorForClass(d.cls);
        if (prefs.showMasks && Array.isArray(d.mask_poly) && d.mask_poly.length >= 3) {
          ctx.beginPath();
          for (let i = 0; i < d.mask_poly.length; i++) {
            const [mx, my] = d.mask_poly[i];
            if (i === 0) ctx.moveTo(mx * sx, my * sy); else ctx.lineTo(mx * sx, my * sy);
          }
          ctx.closePath();
          ctx.fillStyle = color + "33";
          ctx.fill();
          ctx.strokeStyle = color;
          ctx.lineWidth = lw;
          ctx.stroke();
        }
        if (prefs.showBoxes && Array.isArray(d.bbox_xyxy)) {
          const [x1, y1, x2, y2] = d.bbox_xyxy;
          ctx.strokeStyle = color;
          ctx.lineWidth = lw;
          ctx.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);
          const label = `${d.cls} ${(d.confidence * 100).toFixed(0)}%`;
          const tw = ctx.measureText(label).width;
          const ly = Math.max(16, y1 * sy - 4);
          ctx.fillStyle = "rgba(0,0,0,0.6)";
          ctx.fillRect(x1 * sx, ly - 13, tw + 8, 16);
          ctx.fillStyle = color;
          ctx.fillText(label, x1 * sx + 4, ly);
        }
        if (Array.isArray(d.keypoints_uv) && d.keypoints_uv.length >= 17) {
          drawSkeleton(ctx, d.keypoints_uv, sx, sy, lw);
        }
        if (prefs.showNodes && Array.isArray(d.foot_uv)) {
          ctx.beginPath();
          ctx.arc(d.foot_uv[0] * sx, d.foot_uv[1] * sy, lw * 2.5, 0, Math.PI * 2);
          ctx.fillStyle = color;
          ctx.fill();
        }
      }
    }
    const dt = performance.now() - t0;
    s.stats.overlayMs = s.stats.overlayMs ? s.stats.overlayMs * 0.9 + dt * 0.1 : dt;
    if (PERF_HUD) drawHud(s, ctx);
  }

  function drawSkeleton(ctx, kps, sx, sy, lw) {
    ctx.strokeStyle = "#7fe0a0";
    ctx.lineWidth = lw;
    for (const [a, b] of COCO_EDGES) {
      const ka = kps[a], kb = kps[b];
      if (!ka || !kb || ka[2] < KEYPOINT_MIN_CONF || kb[2] < KEYPOINT_MIN_CONF) continue;
      ctx.beginPath();
      ctx.moveTo(ka[0] * sx, ka[1] * sy);
      ctx.lineTo(kb[0] * sx, kb[1] * sy);
      ctx.stroke();
    }
    ctx.fillStyle = "#d5ffe6";
    for (const k of kps) {
      if (k[2] < KEYPOINT_MIN_CONF) continue;
      ctx.beginPath();
      ctx.arc(k[0] * sx, k[1] * sy, lw * 1.5, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function drawHud(s, ctx) {
    const lines = [
      `passthrough ${s.camId}  ${s.videoW}x${s.videoH}`,
      `queue ${s.decoder ? s.decoder.decodeQueueSize : "-"}  au ${s.stats.au}`,
      `presented ${s.stats.presented}  dropped ${s.stats.dropped}  resyncs ${s.stats.resyncs}`,
      `keyframe age ${((performance.now() - s.lastKeyMs) / 1000).toFixed(1)}s${s.degraded ? "  DEGRADED(hold)" : ""}`,
      `decode ${s.stats.decodeMs.toFixed(1)} ms  overlay ${s.stats.overlayMs.toFixed(2)} ms`,
      `longtasks/min ${longTasksPerMin()}`,
    ];
    ctx.font = "12px monospace";
    ctx.textBaseline = "top";
    let y = 8;
    for (const line of lines) {
      const tw = ctx.measureText(line).width;
      ctx.fillStyle = "rgba(0,0,0,0.6)";
      ctx.fillRect(6, y - 2, tw + 8, 16);
      ctx.fillStyle = "#9ef01a";
      ctx.fillText(line, 10, y);
      y += 17;
    }
    ctx.textBaseline = "alphabetic";
  }

  // ---- authoring suspension (body.cam-drawing) -------------------------------
  //
  // Zone drawing / Mode-1 calibration clicks / patch authoring all run on the
  // JPEG <img> (pixel clicks + naturalWidth math in draw_mode.js and friends).
  // draw_mode toggles body.cam-drawing for the duration — while it's set, the
  // player yields the view back to the JPEG stream and resumes afterwards.

  let suspendedCams = null;   // non-null (list of camIds) while authoring

  function suspendAll() {
    if (suspendedCams) return;
    const cams = [...sessions.keys()];
    for (const camId of cams) {
      const img = sessions.get(camId).els.img;
      deactivate(camId);                       // suspendedCams still null here
      if (window.__videoWS) window.__videoWS.attach(img, `cam:${camId}`);
    }
    suspendedCams = cams;
  }

  function resumeAll() {
    if (!suspendedCams) return;
    const cams = suspendedCams;
    suspendedCams = null;
    // activate() retries passthrough; on success markActive() detaches the
    // temporary JPEG sub again.
    for (const camId of cams) activate(camId);
  }

  new MutationObserver(() => {
    if (document.body.classList.contains("cam-drawing")) suspendAll();
    else resumeAll();
  }).observe(document.body, { attributes: true, attributeFilter: ["class"] });

  // ---- global events ---------------------------------------------------------

  // A config save can flip video_passthrough, change a camera source, or
  // hot-restart isistream: reset every session (fallback stickiness included)
  // and retry — activate() re-reads prefs and re-walks the fallback chain.
  document.addEventListener("config:saved", () => {
    prefsLoaded = null;
    const cams = [...sessions.keys()];
    for (const camId of cams) deactivate(camId);
    for (const camId of cams) activate(camId);
  });

  window.__passthrough = {
    activate,
    deactivate,
    frameSize(camId) {
      const s = sessions.get(camId);
      return s && s.active && s.videoW ? [s.videoW, s.videoH] : null;
    },
  };
})();
