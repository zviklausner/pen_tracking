"""The trail left by the moving pen tip."""

import cv2
import numpy as np

from pen_tracking.frame_renderer import PenPath, render_frame
from pen_tracking.hand_tracker import HandDetectionResult
from pen_tracking.tracking_result import FrameTrackingResult


def test_pen_path_starts_a_new_stroke_after_a_jump() -> None:
    path = PenPath()
    path.add((10, 10), frame_height=200)
    path.add((12, 11), frame_height=200)
    path.add((80, 80), frame_height=200)

    assert len(path.strokes) == 2
    assert path.strokes[0] == [(10, 10), (12, 11)]
    assert path.strokes[1] == [(80, 80)]


def test_pen_path_does_not_connect_across_a_missing_frame() -> None:
    path = PenPath()
    path.add((10, 10), frame_height=200)
    path.add((14, 12), frame_height=200)
    path.break_stroke()
    path.add((18, 16), frame_height=200)

    drawn = [stroke for stroke in path.strokes if len(stroke) >= 2]
    assert drawn == [[(10, 10), (14, 12)]]
    assert path.strokes[-1] == [(18, 16)]


def test_render_draws_the_trail_and_tip_in_their_own_colors() -> None:
    frame = np.full((90, 120, 3), 255, np.uint8)
    path = PenPath()
    path.add((10, 70), frame_height=2000)
    path.add((70, 70), frame_height=2000)
    pen = type("Pen", (), {"grip": (20.0, 10.0), "tip": (20.0, 40.0)})()
    tracking = FrameTrackingResult(
        hand=HandDetectionResult(hand_landmarks=[], handedness=[]),
        pen=pen,
    )

    render_frame(
        frame,
        tracking,
        draw_hand=False,
        draw_pen_path=True,
        pen_path=path,
        overlay_color="yellow",
        path_color="cyan",
        tip_color="red",
        overlay_thickness=3,
    )

    # Cyan trail along y=70. The shaft stays yellow. The tip marker is red.
    assert frame[70, 40, 0] > 200
    assert frame[70, 40, 1] > 200
    assert frame[70, 40, 2] < 40
    assert frame[20, 20, 1] > 200
    assert frame[20, 20, 2] > 200
    assert int(frame[40, 20, 2]) > 200
    assert int(frame[40, 20, 1]) < 40
