# src/shared/vision_engine.py
import cv2
import time
import logging
import supervision as sv
import numpy as np
from datetime import datetime
from src.utils.event_logger import EventLogger

logger = logging.getLogger(__name__)

class VisionEngine:
    """Unified orchestrator for detection, tracking, counting, and annotation.

    The single stateful object used by both the CLI scripts and the
    Flask web app. Wraps a model-agnostic inferencer and adds:

    - **ByteTrack** persistent object tracking (IDs survive occlusion).
    - **LineZone** crossing detection — counts each object exactly once
      per session.
    - **EventLogger** per-crossing CSV rows (auto-rotates at midnight,
      keeps 30 days of history by default).
    - Supervision annotators for masks, traces, bounding boxes, and labels.

    The engine is model-agnostic: pass any inferencer that implements
    :class:`~src.inference.base_inferencer.BaseInferencer`.

    Args:
        inferencer: An instance of ``YOLOInferencer``,
            ``RFDETRInferencer``, or ``OptimizedONNXInferencer``.
        config: The full ``train.yaml`` config dict, used for confidence
            threshold, tracker parameters, and logging settings.

    Example:
        ```python
        from src.inference.onnx_inferencer import OptimizedONNXInferencer
        from src.shared.vision_engine import VisionEngine

        engine = VisionEngine(
            inferencer=OptimizedONNXInferencer("best.onnx"),
            config=config
        )
        annotated, detections, event = engine.process_frame(frame, counts)
        if event:
            print(f"{event['class']} crossed the line (ID #{event['id']})")
        ```
    """

    def __init__(self, inferencer, config: dict):
        self.inferencer = inferencer
        self.config = config
        
        # 1. Pipeline Config
        inf_cfg = config.get('inference', {})
        conf_thresh = inf_cfg.get('conf_threshold', 0.3)
        track_cfg = inf_cfg.get('tracker', {})
        
        # 2. Tracking & Core Analytics
        self.tracker = sv.ByteTrack(
            track_activation_threshold=conf_thresh,
            lost_track_buffer=track_cfg.get('track_buffer', 60),
            minimum_matching_threshold=track_cfg.get('match_thresh', 0.9)
        )
        
        # 3. Line Counting Logic (Stateful)
        self.line_zone = None 
        self.counted_ids = set()
        
        # 4. Logging & Telemetry — one CSV row per line crossing.
        # The events subdir keeps them separate from legacy snapshot logs
        # and is the dir the /api/chart endpoint reads from.
        log_cfg = inf_cfg.get('logging', {})
        self.class_names_list = list(self.inferencer.class_names.values())
        base_log_dir = log_cfg.get('log_dir', 'logs')
        retention_days = int(log_cfg.get('retention_days', 30))
        self.event_logger = EventLogger(
            log_dir=f"{base_log_dir.rstrip('/')}/events",
            retention_days=retention_days,
        )

        # 5. Visual Annotators
        self.mask_annotator = sv.MaskAnnotator(opacity=0.3)
        self.trace_annotator = sv.TraceAnnotator(thickness=1, trace_length=30)
        self.box_annotator = sv.BoxAnnotator(thickness=1)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.3, text_thickness=1, text_padding=2)
        # display_in_count / display_out_count = False removes the small badge
        # rectangle supervision renders at the line's midpoint. Counts are
        # already visible in the UI footer and /api/stats.
        self.line_annotator = sv.LineZoneAnnotator(
            thickness=1,
            text_thickness=0,
            text_scale=0,
            display_in_count=False,
            display_out_count=False,
        )

        # 6. Line configuration (can be overridden before first frame)
        self.line_orientation = 'vertical'
        self.line_position = 0.5
        self.belt_direction = 'left_to_right'
        self._frame_w = 0
        self._frame_h = 0

    # Belt direction → leading-edge anchor map. The leading edge is the side
    # of the bbox that enters the line zone FIRST given the belt's motion.
    # Using the leading edge maximises the sorter's reaction window.
    _ANCHOR_MAP = {
        ('vertical',   'left_to_right'): sv.Position.CENTER_RIGHT,
        ('vertical',   'right_to_left'): sv.Position.CENTER_LEFT,
        ('horizontal', 'top_to_bottom'): sv.Position.BOTTOM_CENTER,
        ('horizontal', 'bottom_to_top'): sv.Position.TOP_CENTER,
    }

    def swap_inferencer(self, new_inferencer):
        """Replace the model without losing counts, tracker IDs, or line state.

        Rebuilds palette-dependent annotators against the new inferencer's
        class_names so per-class colours reflect the new model's convention
        (YOLO 0-indexed vs RF-DETR 1-indexed). Tracker, counted_ids,
        line_zone, and event_logger are preserved so hot-swap keeps
        counts and per-event history intact.
        """
        self.inferencer = new_inferencer
        self.class_names_list = list(new_inferencer.class_names.values())

        # Rebuild only the annotators whose palette is indexed by class_id.
        # Trace and line annotators don't depend on class_id — leave them.
        self.mask_annotator = sv.MaskAnnotator(opacity=0.3)
        self.box_annotator = sv.BoxAnnotator(thickness=1)
        self.label_annotator = sv.LabelAnnotator(
            text_scale=0.3, text_thickness=1, text_padding=2,
        )

    def init_line(self, width, height, position=0.5, orientation='vertical',
                   belt_direction=None):
        """Initializes the counting line based on frame dimensions.

        Args:
            width: Frame width in pixels.
            height: Frame height in pixels.
            position: Line position as a fraction (0.1–0.9). For vertical
                lines this is the x-offset; for horizontal, the y-offset.
            orientation: ``'vertical'`` (default) or ``'horizontal'``.
            belt_direction: One of ``'left_to_right'``, ``'right_to_left'``,
                ``'top_to_bottom'``, ``'bottom_to_top'``. Determines which
                side of the bbox is the leading edge and therefore the
                trigger anchor. If ``None``, uses ``self.belt_direction``.
        """
        self.line_position = position
        self.line_orientation = orientation
        if belt_direction is not None:
            self.belt_direction = belt_direction
        self._frame_w = width
        self._frame_h = height

        if orientation == 'horizontal':
            line_y = int(height * position)
            start = sv.Point(0, line_y)
            end = sv.Point(width, line_y)
        else:
            line_x = int(width * position)
            start = sv.Point(line_x, 0)
            end = sv.Point(line_x, height)

        # Pick the leading-edge anchor based on orientation + belt direction.
        anchor = self._ANCHOR_MAP.get(
            (orientation, self.belt_direction),
            sv.Position.BOTTOM_CENTER,   # safe fallback
        )

        self.line_zone = sv.LineZone(
            start=start, end=end,
            triggering_anchors=[anchor],
        )

    def process_frame(self, frame: np.ndarray, class_totals: dict):
        """Run detection, tracking, and line-crossing counting on one frame.

        Called once per frame from the inference thread. Thread-safe
        provided it is called from a single thread (the inference loop
        owns it exclusively).

        Args:
            frame: Raw BGR frame as a NumPy array (from OpenCV).
            class_totals: Dict updated **in-place** with crossing counts
                (e.g. ``{'carton': 12, 'polybag': 5}``). Owned by
                ``StreamHandler``; do not pass the same dict from
                multiple threads simultaneously.

        Returns:
            A 3-tuple ``(annotated, detections, event)``:

            - ``annotated``: BGR frame with masks, traces, boxes, labels,
              and the counting line drawn.
            - ``detections``: ``sv.Detections`` object for the current
              frame (tracker IDs assigned).
            - ``events``: list of ``{"class": str, "id": int}`` dicts —
              one entry per new line crossing this frame. Empty list if
              no crossings. Multiple entries if several objects crossed
              in the same frame (close-together on the belt).

        Note:
            ``counted_ids`` is automatically pruned when it exceeds
            50,000 entries (prevents unbounded memory growth in
            sessions longer than 8–12 hours).
        """
        if self.line_zone is None:
            h, w = frame.shape[:2]
            self.init_line(w, h, self.line_position, self.line_orientation)

        # 1. Core AI Logic (timed for performance dashboard)
        _t0 = time.perf_counter()
        frame_input = np.ascontiguousarray(frame)
        detections = self.inferencer.predict_frame(frame_input)
        _t1 = time.perf_counter()
        detections = self.tracker.update_with_detections(detections)
        _t2 = time.perf_counter()
        self.last_timing = {
            'forward_ms': (_t1 - _t0) * 1000,
            'tracker_ms': (_t2 - _t1) * 1000,
        }
        
        # 2. Crossing/Trigger Logic
        in_cross, out_cross = self.line_zone.trigger(detections=detections)
        all_crossings = in_cross | out_cross
        
        # Collect EVERY new crossing this frame — the caller publishes one
        # UDP datagram per event so the sorter never misses a trigger when
        # two close-together objects cross in the same frame.
        new_events = []
        for i, crossed in enumerate(all_crossings):
            if crossed and detections.tracker_id is not None:
                t_id = int(detections.tracker_id[i])
                if t_id not in self.counted_ids:
                    class_id = int(detections.class_id[i])
                    name = self.inferencer.class_names.get(class_id, "object")
                    class_totals[name] = class_totals.get(name, 0) + 1
                    self.counted_ids.add(t_id)
                    new_events.append({"class": name, "id": t_id})
                    self.event_logger.log(name, t_id)

        # Prune counted_ids to prevent unbounded growth over long shifts.
        # ByteTrack IDs are monotonically increasing — old IDs are never reassigned,
        # so it is safe to discard the lower half once the set gets large.
        if len(self.counted_ids) > 50_000:
            sorted_ids = sorted(self.counted_ids)
            self.counted_ids = set(sorted_ids[len(sorted_ids) // 2:])
            logger.info(f"♻️ counted_ids pruned to {len(self.counted_ids)} entries")

        # 3. Visual Composition
        annotated = frame.copy()
        if detections.mask is not None:
            annotated = self.mask_annotator.annotate(scene=annotated, detections=detections)
        
        annotated = self.trace_annotator.annotate(scene=annotated, detections=detections)
        annotated = self.box_annotator.annotate(scene=annotated, detections=detections)
        
        if detections.tracker_id is not None:
            labels = [f"#{t_id} {self.inferencer.class_names.get(c_id, 'obj')}" 
                      for c_id, t_id in zip(detections.class_id, detections.tracker_id)]
            annotated = self.label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
        
        annotated = self.line_annotator.annotate(frame=annotated, line_counter=self.line_zone)
        
        return annotated, detections, new_events

    def cleanup(self, class_totals):
        """Safe shutdown. EventLogger writes on every event, so there is
        nothing to flush here; method kept for API stability."""
        return
