// Multiplexed video transport — ALL panel video over ONE WebSocket (/ws/video).
//
// Replaces per-panel MJPEG <img src="/stream/..."> connections, which each held
// one of the browser's ~6 HTTP/1.1 connections per host and could starve a
// settings POST / page reload into a frozen UI. One socket carries every panel.
//
// Classic deferred script (NOT a module) so big_panel.js (also classic) can use
// it; modules (zone_patch.js) reach it via the same window.__videoWS global.
//
// API:
//   __videoWS.attach(imgEl, streamId)  — render that server stream into <img>
//   __videoWS.detach(streamId)         — stop it (panel hidden / removed)
//   __videoWS.resubscribeAll()         — drop + re-establish every stream (after
//                                        a config save, so a new model/source
//                                        applies without a page reload)
//
// Frames arrive as binary: uint8 idLen | stream-id utf8 | JPEG bytes — rendered
// via blob URLs into the existing <img> elements, so all CSS (object-fit,
// expand overlay) keeps working unchanged. Auto-reconnects with backoff.

(function () {
  "use strict";

  const subs = new Map();   // streamId -> { img, lastUrl }
  let ws = null;
  let retries = 0;
  let retryTimer = null;
  const decoder = new TextDecoder();

  function wsUrl() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    return `${proto}://${location.host}/ws/video`;
  }

  function connect() {
    if (ws && (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN)) return;
    ws = new WebSocket(wsUrl());
    ws.binaryType = "arraybuffer";
    ws.onopen = () => {
      retries = 0;
      for (const id of subs.keys()) ws.send(JSON.stringify({ sub: id }));
    };
    ws.onmessage = (ev) => {
      if (typeof ev.data === "string") {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.error) console.warn(`videoWS[${msg.stream}]: ${msg.error}`);
        } catch { /* ignore */ }
        return;
      }
      const view = new Uint8Array(ev.data);
      const idLen = view[0];
      const id = decoder.decode(view.subarray(1, 1 + idLen));
      const entry = subs.get(id);
      if (!entry) return;
      const url = URL.createObjectURL(new Blob([view.subarray(1 + idLen)], { type: "image/jpeg" }));
      const prev = entry.lastUrl;
      entry.lastUrl = url;
      entry.img.src = url;
      // Revoke the PREVIOUS frame's URL once the new one has loaded (revoking
      // immediately can abort the in-flight decode on slow machines).
      if (prev) entry.img.addEventListener("load", () => URL.revokeObjectURL(prev), { once: true });
    };
    ws.onclose = () => scheduleReconnect();
    ws.onerror = () => { try { ws.close(); } catch { /* already closed */ } };
  }

  function scheduleReconnect() {
    if (retryTimer || subs.size === 0) return;
    const delay = Math.min(5000, 500 * 2 ** retries++);
    retryTimer = setTimeout(() => { retryTimer = null; connect(); }, delay);
  }

  function send(obj) {
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
  }

  function attach(img, streamId) {
    if (!img || !streamId) return;
    // One stream per <img>: drop any other stream already bound to this element.
    for (const [id, entry] of subs) {
      if (entry.img === img && id !== streamId) detach(id);
    }
    if (subs.has(streamId)) { subs.get(streamId).img = img; return; }
    subs.set(streamId, { img, lastUrl: null });
    connect();
    send({ sub: streamId });
  }

  function detach(streamId) {
    const entry = subs.get(streamId);
    if (!entry) return;
    subs.delete(streamId);
    if (entry.lastUrl) URL.revokeObjectURL(entry.lastUrl);
    send({ unsub: streamId });
  }

  function resubscribeAll() {
    for (const id of subs.keys()) {
      send({ unsub: id });
      send({ sub: id });
    }
  }

  window.__videoWS = { attach, detach, resubscribeAll };
})();
