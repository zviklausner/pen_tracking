"""Unit tests for pen detection."""

import cv2

from pen_tracking.pen_tracker_cv import PenTracker
from pen_tracking.paths import DEFAULT_VIDEO


def _read_frame(frame_index: int = 40):
    cap = cv2.VideoCapture(str(DEFAULT_VIDEO))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    assert ret
    height, width = frame.shape[:2]
    return cv2.resize(frame, (int(width * 720 / height), 720))


def test_pen_tracker_detects_tip_on_sample_frame(hand_landmarks_from_video) -> None:
    frame = _read_frame(40)
    tracker = PenTracker()
    pen = tracker.detect(frame, hand_landmarks_from_video)
    assert pen is not None
    assert pen.tip[1] > pen.grip[1]
    assert pen.confidence > 0
