"""Video processing pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2

from pen_tracking.config import TrackerConfig
from pen_tracking.frame_renderer import render_frame, resize_frame
from pen_tracking.hand_tracker import HandTracker

logger = logging.getLogger(__name__)


@dataclass
class ProcessingStats:
    frames_processed: int = 0
    frames_with_hands: int = 0


def process_video(config: TrackerConfig) -> ProcessingStats:
    """Run hand-occlusion tracking over a video file."""
    config.validate()
    stats = ProcessingStats()

    cap = cv2.VideoCapture(str(config.video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {config.video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = None
    window_name = "AInav Core Engine"

    try:
        with HandTracker(
            str(config.model_path),
            max_num_hands=config.max_num_hands,
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        ) as tracker:
            frame_index = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                small_frame = resize_frame(frame, config.target_height)
                rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                timestamp_ms = int((frame_index / fps) * 1000)

                detection = tracker.detect(rgb_frame, timestamp_ms)
                render_frame(small_frame, detection)

                stats.frames_processed += 1
                if detection.hand_landmarks:
                    stats.frames_with_hands += 1

                if writer is None and config.output_path is not None:
                    height, width = small_frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(
                        str(config.output_path),
                        fourcc,
                        fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(
                            f"Unable to create output video: {config.output_path}"
                        )

                if writer is not None:
                    writer.write(small_frame)

                if config.display:
                    cv2.imshow(window_name, small_frame)
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
        "Processed %s frames (%s with hand occlusion).",
        stats.frames_processed,
        stats.frames_with_hands,
    )
    if config.output_path is not None:
        logger.info("Wrote output to %s", config.output_path)

    return stats
