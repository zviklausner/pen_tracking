"""Video processing pipeline."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass

import cv2

from pen_tracking.config import TrackerConfig
from pen_tracking.frame_renderer import render_frame
from pen_tracking.hand_tracker import HandDetectionResult, HandTracker
from pen_tracking.pen_tracker_cv import PenTracker
from pen_tracking.tracking_result import FrameTrackingResult
from pen_tracking.video_io import create_video_writer, supports_gui

logger = logging.getLogger(__name__)

_EMPTY_HAND = HandDetectionResult(hand_landmarks=[], handedness=[])


@dataclass
class ProcessingStats:
    frames_processed: int = 0
    frames_with_hands: int = 0
    frames_with_pen: int = 0


ProgressCallback = Callable[[int, int], None]


def process_video(
    config: TrackerConfig,
    *,
    progress: ProgressCallback | None = None,
    stop_event: threading.Event | None = None,
) -> ProcessingStats:
    """Run hand and/or pen tracking over a video file.

    ``progress`` is called with ``(frames_read, total_frames)``; ``total_frames``
    may be 0 when the container does not report a frame count.
    """
    config.validate()
    stats = ProcessingStats()

    cap = cv2.VideoCapture(str(config.video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {config.video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0)
    output_fps = fps / config.frame_stride
    writer = None
    window_name = "AInav Core Engine"
    hand_context = (
        HandTracker(
            str(config.model_path),
            max_num_hands=config.max_num_hands,
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )
        if config.track_hand
        else nullcontext()
    )

    try:
        with hand_context as hand_tracker:
            pen_tracker = (
                PenTracker(
                    max_value=config.pen_max_value,
                    axis_distance=config.pen_axis_distance,
                    smoothing=config.pen_smoothing,
                )
                if config.track_pen
                else None
            )

            frame_index = 0

            while cap.isOpened():
                if stop_event is not None and stop_event.is_set():
                    logger.info("Stopped by request after %s frames.", frame_index)
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                if progress is not None:
                    progress(frame_index + 1, total_frames)

                if frame_index % config.frame_stride != 0:
                    frame_index += 1
                    continue

                timestamp_ms = int((frame_index / fps) * 1000)
                hand_detection = _EMPTY_HAND
                if hand_tracker is not None:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    hand_detection = hand_tracker.detect(rgb_frame, timestamp_ms)

                pen_detection = None
                if pen_tracker is not None:
                    if hand_detection.hand_landmarks:
                        pen_detection = pen_tracker.track(
                            frame,
                            hand_detection.hand_landmarks[0],
                        )
                    elif config.track_pen_without_hand or not config.track_hand:
                        pen_detection = pen_tracker.track(frame)
                    else:
                        pen_tracker.reset()

                tracking = FrameTrackingResult(
                    hand=hand_detection,
                    pen=pen_detection,
                )
                render_frame(
                    frame,
                    tracking,
                    draw_hand=config.track_hand,
                    draw_pen=config.track_pen,
                    overlay_color=config.overlay_bgr,
                    overlay_thickness=config.overlay_thickness,
                )

                stats.frames_processed += 1
                if tracking.has_hand:
                    stats.frames_with_hands += 1
                if tracking.has_pen:
                    stats.frames_with_pen += 1

                if writer is None and config.output_path is not None:
                    height, width = frame.shape[:2]
                    writer = create_video_writer(
                        config.output_path,
                        output_fps,
                        width,
                        height,
                    )

                if writer is not None:
                    writer.write(frame)

                if config.display:
                    if not supports_gui():
                        logger.warning(
                            "Display requested but no GUI is available; skipping preview."
                        )
                    else:
                        cv2.imshow(window_name, frame)
                        if cv2.waitKey(20) & 0xFF == ord(config.quit_key):
                            logger.info("Stopped by user (%s key).", config.quit_key)
                            break

                frame_index += 1

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if config.display:
            cv2.destroyAllWindows()

    logger.info(
        "Processed %s frames (%s with hand, %s with pen).",
        stats.frames_processed,
        stats.frames_with_hands,
        stats.frames_with_pen,
    )
    if config.output_path is not None:
        logger.info("Wrote output to %s", config.output_path)

    return stats
