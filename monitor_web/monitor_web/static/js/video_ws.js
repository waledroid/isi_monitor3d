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
// Protocol (client → server, text/JSON): {sub: id} | {unsub: id} | {ack: id}.
// Frames arrive as binary: uint8 idLen | stream-id utf8 | JPEG bytes.
//
// LATEST-FRAME-ONLY rendering: each binary message only fills a per-stream
// one-slot holder; a single requestAnimationFrame loop swaps the NEWEST slot
// into the <img> (older undisplayed frames are simply overwritten — never
// queued), revokes the previous blob URL unconditionally at swap, and sends
// {ack: id} so the server's credit gate paces sends to the client's real
// render rate. This is what stops the demo-observed frame accumulation: a
// slow/stalled browser now always shows the newest frame at its own pace
// instead of walking a growing FIFO of stale ones (and no longer leaks blob
// URLs whose load events never fired). Auto-reconnects with backoff.

(function () {
  "use strict";

  const subs = new Map();   // streamId -> { img, lastUrl, pending }
  let ws = null;
  let retries = 0;
  let retryTimer = null;
  let rafId = null;
  const decoder = new TextDecoder();

  function wsUrl() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    return `${proto}://${location.host}/ws/video`;
  }

  function scheduleRender() {
    if (rafId === null) rafId = requestAnimationFrame(render);
  }

  // One rAF pass for all streams: swap each pending (newest) frame into its
  // <img>, revoke the previous blob URL (aborting a stale in-flight decode is
  // exactly the point), and ack so the server refills our credit.
  function render() {
    rafId = null;
    for (const [id, entry] of subs) {
      if (entry.pending === null) continue;
      const buf = entry.pending;
      entry.pending = null;
      const url = URL.createObjectURL(new Blob([buf], { type: "image/jpeg" }));
      if (entry.lastUrl) URL.revokeObjectURL(entry.lastUrl);
      entry.lastUrl = url;
      entry.img.src = url;
      send({ ack: id });
    }
  }

  function connect() {
    if (ws && (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN)) return;
    ws = new WebSocket(wsUrl());
    ws.binaryType = "arraybuffer";
    ws.onopen = () => {
      retries = 0;
      for (const id of subs.keys()) {
        ws.send(JSON.stringify({ sub: id }));
        ws.send(JSON.stringify({ ack: id }));   // prime the credit window
      }
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
      // Latest-slot only — no blob/src work here (that happens once per
      // animation frame in render(), always with the newest payload).
      entry.pending = view.subarray(1 + idLen);
      scheduleRender();
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
    subs.set(streamId, { img, lastUrl: null, pending: null });
    connect();
    send({ sub: streamId });
    send({ ack: streamId });                   // prime the credit window
  }

  function detach(streamId) {
    const entry = subs.get(streamId);
    if (!entry) return;
    subs.delete(streamId);
    entry.pending = null;
    if (entry.lastUrl) URL.revokeObjectURL(entry.lastUrl);
    send({ unsub: streamId });
  }

  function resubscribeAll() {
    for (const id of subs.keys()) {
      send({ unsub: id });
      send({ sub: id });
      send({ ack: id });                       // re-prime credit after rebuild
    }
  }

  // rAF is suspended in hidden tabs — on return, drain whatever landed in the
  // slots and re-prime the server's credit so full rate resumes instantly.
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) scheduleRender();
  });

  window.__videoWS = { attach, detach, resubscribeAll };
})();
