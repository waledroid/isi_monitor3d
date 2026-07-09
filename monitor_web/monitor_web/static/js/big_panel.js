// Big-panel reactive state on Alpine.js (CDN, no build).
//
// Classic deferred script (NOT an ES module) loaded right before Alpine, so the
// `alpine:init` listener below is guaranteed to register before Alpine starts.
// Registers a global Alpine.store('bigPanel') that the markup binds to via
// x-show / :class / :src / @click:
//   - view switching (map / cam_a / cam_b / mp4)
//   - expand overlay (~1080p, not fullscreen)
//   - CAM 2 / MP4 tab visibility + per-view MJPEG :src + LIVE badge + START/STOP
//
// Persistence (view + mp4 selection across reloads, session-scoped) is plain
// sessionStorage via an Alpine.effect — no persist plugin dependency.
//
// Cross-module: mp4_unlock.js sets mp4Unlocked; zone_manager.js calls selectMp4
// + dispatches "config:saved". The 3D map (floor_map_3d.js) is independent.

(function () {
  "use strict";

  // "unified" (bird's-eye composite of both cameras) is retired — the rig uses
  // overlapping cameras + triangulation, not a merged BEV render. The stream code
  // stays dormant; the tab is hidden (dashboard.html) and it's not selectable here.
  const VIEWS = ["map", "cam_a", "cam_b", "mp4"];
  const BADGE_KEY = { map: "live", cam_a: "cam_1", cam_b: "cam_2", unified: "unified", mp4: "mp4" };
  const BADGE_FALLBACK = { map: "LIVE", cam_a: "CAM 1", cam_b: "CAM 2", unified: "UNIFIED", mp4: "MP4" };

  function strings() {
    return (window.__monitor_web && window.__monitor_web.strings) || {};
  }

  function initialView() {
    const v = sessionStorage.getItem("active_view");
    // Default is CAM 1 — the CARTE (3D map) is a diagnostic view gated behind
    // the same logo-double-click unlock as MP4, and only renders when chosen.
    const unlocked = sessionStorage.getItem("mp4_unlocked") === "1";
    if (v === "map" && !unlocked) return "cam_a";
    return VIEWS.includes(v) ? v : "cam_a";
  }

  function register() {
    if (!window.Alpine || window.Alpine.store("bigPanel")) return;
    window.Alpine.store("bigPanel", {
      view: initialView(),
      mp4Selected: sessionStorage.getItem("mp4_selected") || "",
      expanded: false,
      closing: false,
      hasCamB: false,
      // Per-camera live-feed flags, pushed by ws_tracks.js from /api/status. The
      // UNIFIED tab requires BOTH cameras actually STREAMING (Phase A: Mode 2 when
      // the feed truly arrives, not merely when cam_b is configured).
      camLive: {},
      mp4Unlocked: sessionStorage.getItem("mp4_unlocked") === "1",
      // Cameras with a calibration in the current mode. Refreshed from
      // /api/calibrate/status on the `calibration:changed` event + initial load.
      // Calibration does NOT auto-warp the feed — the live view always runs
      // detection on the RAW frame (the production path). The rectified bird's-eye
      // is a verification-only view shown on demand via the warp toggle below.
      calibratedCams: [],
      // Verification toggle: when true (and the active camera is calibrated) the
      // feed switches to the plain rectified floor view to eyeball the
      // calibration. Off by default and never persisted — detection must come
      // back live on reload. Out-of-domain for the detector, so no boxes here.
      showWarp: false,
      // Bumped on `config:saved` so the MP4 dev viewer (still MJPEG) reconnects.
      // Live cam / unified / zone video runs over the single /ws/video socket
      // (video_ws.js) — see the subscription effect in boot().
      streamNonce: 0,

      // ---- getters ----
      get badge() {
        if (this.isWarped) return strings().badge_rectified || "RECTIFIED";
        return strings()[BADGE_KEY[this.view] || "live"] || BADGE_FALLBACK[this.view] || "LIVE";
      },
      // The active view is a camera that has a calibration — i.e. the
      // verification warp toggle is meaningful and should be offered.
      get canVerify() {
        return (this.view === "cam_a" || this.view === "cam_b")
          && this.calibratedCams.includes(this.view);
      },
      // True when the active camera view is showing the rectified (verification)
      // feed. The warped frame is auto-fit to its own (non-panel) aspect ratio,
      // so it's shown with object-fit: contain to display the whole rectified
      // frame without cropping or stretching (raw feeds keep object-fit: cover).
      get isWarped() {
        return this.showWarp && this.canVerify;
      },
      // Flip between the live (raw + detection) feed and the rectified
      // verification view for the active camera. The /ws/video subscription
      // effect in boot() reacts to `isWarped` and swaps the stream.
      toggleWarp() {
        this.showWarp = !this.showWarp;
      },
      // Refresh which cameras are calibrated (gates the verification toggle's
      // availability) and resubscribe so a new calibration applies live.
      async refreshCalibration() {
        try {
          const res = await fetch("/api/calibrate/status");
          const data = res.ok ? await res.json() : {};
          this.calibratedCams = Array.isArray(data.calibrated_cameras)
            ? data.calibrated_cameras : [];
        } catch {
          // Transient fetch failure (loop congestion / reload race): KEEP the
          // last-known list — downgrading to [] made the "Flatten floor"
          // verify toggle vanish and read as "no calibration" when the
          // calibration was perfectly fine.
        }
        if (window.__videoWS) window.__videoWS.resubscribeAll();
      },
      get mp4Src() {
        if (this.view !== "mp4" || !this.mp4Selected) return null;
        const safe = this.mp4Selected.split("/").map(encodeURIComponent).join("/");
        return `/stream/mp4/${safe}?n=${this.streamNonce}`;
      },
      // The UNIFIED tab is offered only when BOTH cameras are actually live
      // (configured + streaming) — so it appears/disappears with the real feed.
      get unifiedAvailable() {
        return this.hasCamB && !!this.camLive.cam_a && !!this.camLive.cam_b;
      },

      // ---- view + expand ----
      select(v) {
        if (v === "cam_b" && !this.hasCamB) return;
        if (v === "unified" && !this.unifiedAvailable) return;   // needs 2 LIVE cameras
        // mp4Unlocked is the shared DEV unlock (logo double-click + password):
        // it gates both the MP4 viewer and the CARTE 3D map.
        if ((v === "mp4" || v === "map") && !this.mp4Unlocked) return;
        this.showWarp = false;   // every view starts live; warp is opt-in per view
        this.view = v;
      },
      toggleExpand() {
        if (this.expanded) { this.collapse(); return; }
        this.closing = false;
        this.expanded = true;
        window.dispatchEvent(new Event("resize"));   // nudge Pixi to recompute
      },
      collapse() {
        if (!this.expanded) return;
        this.expanded = false;   // backdrop fades out (x-transition leave)
        this.closing = true;     // panel stays fixed + plays the shrink, then snaps to flow
        setTimeout(() => { this.closing = false; window.dispatchEvent(new Event("resize")); }, 160);
      },
      // ---- in-column tall expansion (the <> toggle) ----
      // One left-column panel ('big' | 'zone1' | 'zone2') grows to the FULL column
      // height while its siblings hide; the sidebar is untouched. Distinct from
      // `expanded` ([] full-width overlay), which only the big panel offers.
      tallPanel: null,
      toggleTall(which) {
        this.tallPanel = this.tallPanel === which ? null : which;
        // Streams/canvas re-fit to the new box (3D map + MJPEG <img> sizing).
        requestAnimationFrame(() => window.dispatchEvent(new Event("resize")));
      },

      // ---- START / STOP ----
      // Surface the outcome: START that doesn't reach "running" (missing config,
      // boot crash) used to fail silently — now we flash the reason so the
      // operator isn't left wondering why nothing happened.
      async start() {
        try {
          const res = await fetch("/api/control/start", { method: "POST" });
          const data = await res.json();
          if (data.state !== "running") {
            const why = data.reason || `state: ${data.state}`;
            this.flashStatus(`Backbone did not start — ${why}`, true);
          } else {
            this.flashStatus("Backbone running", false);
          }
        } catch (e) {
          this.flashStatus("Start request failed", true);
        }
        if (window.htmx) htmx.trigger("#logs-content", "load");   // refresh logs now
      },
      stopping: false,
      async stop() {
        if (this.stopping) return;             // ignore double-clicks
        this.stopping = true;
        // Immediate feedback — the server-side teardown takes seconds
        // (SIGTERM grace + GPU session release); without this the button
        // felt dead and got hammered.
        this.flashStatus("Stopping backbone…", false);
        try {
          const res = await fetch("/api/control/stop", { method: "POST" });
          const data = await res.json();
          this.flashStatus(data.state === "stopped" ? "Backbone stopped" : `state: ${data.state}`, false);
        } catch (e) {
          this.flashStatus("Stop request failed", true);
        } finally {
          this.stopping = false;
        }
        if (window.htmx) htmx.trigger("#logs-content", "load");
      },
      // Lightweight transient banner in the STATUS sidebar pane.
      flashStatus(msg, isError) {
        const box = document.getElementById("status-content");
        if (!box) { (isError ? console.error : console.info)(msg); return; }
        const line = document.createElement("div");
        line.className = "status-flash" + (isError ? " is-error" : "");
        line.textContent = msg;
        box.prepend(line);
        setTimeout(() => line.remove(), 8000);
      },

      // ---- MP4 (called by the zone-manager picker) ----
      selectMp4(name) {
        if (!name) { this.mp4Selected = ""; return; }
        this.mp4Selected = name;
        this.view = "mp4";
        fetch("/api/ui-settings", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ mp4_selected: name }),
        }).catch(() => {});
      },

      // ---- data load ----
      async refreshCameras() {
        try {
          const res = await fetch("/api/config");
          if (!res.ok) return;
          const cams = (await res.json()).cameras || {};
          this.hasCamB = !!(cams.cam_b && (cams.cam_b.url || cams.cam_b.device));
          if (!this.hasCamB && this.view === "cam_b") this.view = "cam_a";
        } catch (e) { /* quiet */ }
      },
      async loadUi() {
        try {
          const res = await fetch("/api/ui-settings");
          if (!res.ok) return;
          const ui = await res.json();
          if (ui.mp4_selected && !this.mp4Selected) this.mp4Selected = ui.mp4_selected;
        } catch (e) { /* quiet */ }
      },

      boot() {
        // Persist view + mp4 selection (session-scoped) reactively.
        window.Alpine.effect(() => {
          sessionStorage.setItem("active_view", this.view);
          sessionStorage.setItem("mp4_selected", this.mp4Selected);
        });
        // Guard a stale persisted view that's not valid for this host/session.
        if (this.view === "cam_b" && !this.hasCamB) this.view = "cam_a";
        if ((this.view === "mp4" || this.view === "map") && !this.mp4Unlocked) this.view = "cam_a";
        // Leave the UNIFIED view if it stops being available (cam_b feed dropped).
        window.Alpine.effect(() => {
          if (this.view === "unified" && !this.unifiedAvailable) this.view = "cam_a";
        });
        // /ws/video subscriptions follow the ACTIVE view: only the visible
        // panel's stream is subscribed (hidden tabs hold no stream — this is
        // what freed the browser's per-host connection slots). Reacts to view
        // switches and the warp verification toggle.
        window.Alpine.effect(() => {
          const VWS = window.__videoWS;
          if (!VWS) return;
          ["cam_a", "cam_b"].forEach((cam) => {
            const img = document.getElementById(`${cam}-img`);
            if (this.view === cam && img) {
              const warped = this.showWarp && this.calibratedCams.includes(cam);
              VWS.attach(img, `cam:${cam}${warped ? ":warp" : ""}`);
            } else {
              VWS.detach(`cam:${cam}`);
              VWS.detach(`cam:${cam}:warp`);
            }
          });
          const uimg = document.getElementById("unified-img");
          if (this.view === "unified" && uimg) VWS.attach(uimg, "unified");
          else VWS.detach("unified");
        });
        this.loadUi();
        this.refreshCameras();
        this.refreshCalibration();
        document.addEventListener("config:saved", () => {
          this.refreshCameras();
          // A camera-count change can switch the mode (and thus which calibration
          // applies), so re-read calibration state too. refreshCalibration() also
          // resubscribes every /ws/video stream, so a changed detection model or
          // camera source applies live — no reload, no new tab.
          this.refreshCalibration();
          this.streamNonce++;   // MP4 dev viewer (MJPEG) reconnect
        });
        // Calibration saved/cleared → refresh which cameras can be verified.
        document.addEventListener("calibration:saved", () => this.refreshCalibration());
        document.addEventListener("calibration:cleared", () => this.refreshCalibration());
      },
    });

    // Reusable expand/collapse behaviour for the zone panels (each its own state).
    // Same two-way animation as the big panel: pop-in on expand, shrink on close.
    window.Alpine.data("expandable", () => ({
      expanded: false,
      closing: false,
      toggleExpand() {
        if (this.expanded) { this.collapse(); return; }
        this.closing = false;
        this.expanded = true;
      },
      collapse() {
        if (!this.expanded) return;
        this.expanded = false;
        this.closing = true;
        setTimeout(() => { this.closing = false; }, 160);
      },
    }));
  }

  // Register before Alpine starts (normal path), or immediately if Alpine is
  // somehow already present.
  if (window.Alpine) register();
  else document.addEventListener("alpine:init", register);
})();
