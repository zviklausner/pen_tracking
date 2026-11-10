"""Unit tests for pen detection."""

import cv2
import numpy as np
import pytest

from pen_tracking.pen_tracker_cv import PenDetectionResult, PenTracker
from pen_tracking.paths import DEFAULT_VIDEO


def _read_full_frame(frame_index: int):
    cap = cv2.VideoCapture(str(DEFAULT_VIDEO))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    assert ret
    return frame


def _read_frame(frame_index: int = 40):
    frame = _read_full_frame(frame_index)
    height, width = frame.shape[:2]
    return cv2.resize(frame, (int(width * 720 / height), 720))


def test_track_with_hand_puts_tip_on_the_pencil_point(hand_landmarks_from_video) -> None:
    pen = PenTracker().track(_read_frame(40), hand_landmarks_from_video)
    assert pen is not None
    assert not pen.from_hand
    # Graphite point of frame 40 at 720 px height, checked by eye.
    assert np.linalg.norm(np.array(pen.tip) - (235, 490)) < 8


def test_track_holds_last_tip_briefly_then_gives_up() -> None:
    tracker = PenTracker(max_missed_frames=2)
    assert tracker.track(_synthetic_pencil()) is not None
    empty = np.full((720, 480, 3), (110, 165, 205), np.uint8)
    assert tracker.track(empty) is not None
    assert tracker.track(empty) is not None
    assert tracker.track(empty) is None


def test_track_ignores_sudden_jump_then_accepts_it_if_it_persists() -> None:
    tracker = PenTracker(max_rejected_frames=2)
    steady = _synthetic_pencil()
    for _ in range(3):
        first = tracker.track(steady)
    jumped = _synthetic_pencil(tip=(150, 650), back=(40, 230))

    assert tracker.track(jumped).tip == first.tip
    assert tracker.track(jumped).tip == first.tip
    moved = tracker.track(jumped)
    assert np.linalg.norm(np.array(moved.tip) - (150, 650)) < 4


def test_track_rejects_a_tip_sitting_on_the_fingers(hand_landmarks_from_video) -> None:
    """Reproduce the drift screenshot: a short shaft whose tip lands on the thumb."""
    frame = _read_frame(40)
    height, width = frame.shape[:2]
    thumb = hand_landmarks_from_video[4]
    index = hand_landmarks_from_video[8]
    thumb_px = np.array([thumb.x * width, thumb.y * height])
    index_px = np.array([index.x * width, index.y * height])
    tip = (thumb_px + index_px) / 2.0
    grip = tip - np.array([0.0, 40.0])

    raw = PenDetectionResult(
        tip=(float(tip[0]), float(tip[1])),
        grip=(float(grip[0]), float(grip[1])),
        axis=(0.0, 1.0),
        confidence=0.5,
        from_hand=False,
    )
    assert PenTracker._tip_is_on_the_hand(raw, [thumb_px, index_px], height)
    # A tip well past the fingertips along the pencil is kept.
    good = PenDetectionResult(
        tip=(float(tip[0]), float(tip[1] + 120)),
        grip=(float(grip[0]), float(grip[1])),
        axis=(0.0, 1.0),
        confidence=0.5,
        from_hand=False,
    )
    assert not PenTracker._tip_is_on_the_hand(good, [thumb_px, index_px], height)


def test_track_ignores_dark_streak_without_a_sharpened_point() -> None:
    tracker = PenTracker()
    steady = _synthetic_pencil()
    for _ in range(3):
        first = tracker.track(steady)
    # A dark streak just above the real tip, like a gap between fingers.
    streak = np.full((720, 480, 3), (110, 165, 205), np.uint8)
    cv2.line(streak, (240, 330), (290, 500), (25, 25, 25), 14)

    assert PenTracker().track(streak) is None
    assert tracker.track(streak).tip == first.tip


def _synthetic_pencil(tip=(300, 520), back=(180, 80)):
    """Wood-coloured background with a dark shaft, light sharpened cone, and grey lead."""
    frame = np.full((720, 480, 3), (110, 165, 205), np.uint8)  # BGR light wood
    tip_arr, back_arr = np.array(tip, float), np.array(back, float)
    axis = (tip_arr - back_arr) / np.linalg.norm(tip_arr - back_arr)
    cone_start = tip_arr - axis * 30
    lead_start = tip_arr - axis * 10
    normal = np.array([-axis[1], axis[0]])

    def quad(a, b, width_a, width_b):
        return np.array(
            [a + normal * width_a, b + normal * width_b, b - normal * width_b, a - normal * width_a],
            np.int32,
        )

    cv2.fillPoly(frame, [quad(back_arr, cone_start, 7, 7)], (25, 25, 25))
    cv2.fillPoly(frame, [quad(cone_start, lead_start, 7, 3)], (225, 230, 232))
    cv2.fillPoly(frame, [quad(lead_start, tip_arr, 3, 0.5)], (90, 90, 95))
    return frame


def test_detect_without_hand_on_synthetic_pencil() -> None:
    pen = PenTracker().detect_without_hand(_synthetic_pencil())
    assert pen is not None
    assert not pen.from_hand
    assert np.linalg.norm(np.array(pen.tip) - (300, 520)) < 4
    assert pen.grip[1] < pen.tip[1]


def test_detect_without_hand_ignores_empty_scene() -> None:
    frame = np.full((720, 480, 3), (110, 165, 205), np.uint8)
    assert PenTracker().detect_without_hand(frame) is None


@pytest.mark.parametrize(
    ("frame_index", "expected_tip"),
    [(0, (502, 2503)), (45, (1376, 2627)), (83, (1902, 2661))],
)
def test_detect_without_hand_finds_tip_on_sample_frames(frame_index, expected_tip) -> None:
    pen = PenTracker().detect_without_hand(_read_full_frame(frame_index))
    assert pen is not None
    # Sample frames are 2160x3840; 30 px is about 5 px at detection resolution.
    assert np.linalg.norm(np.array(pen.tip) - expected_tip) < 30
    assert pen.grip[1] < pen.tip[1]


def test_pen_tracker_detects_tip_on_sample_frame(hand_landmarks_from_video) -> None:
    frame = _read_frame(40)
    tracker = PenTracker()
    pen = tracker.detect(frame, hand_landmarks_from_video)
    assert pen is not None
    assert pen.tip[1] > pen.grip[1]
    assert pen.confidence > 0
