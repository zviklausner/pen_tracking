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


def test_render_draws_the_tip_path_in_the_overlay_color() -> None:
    frame = np.full((80, 120, 3), 255, np.uint8)
    path = PenPath()
    path.add((10, 40), frame_height=80)
    path.add((70, 40), frame_height=80)
    tracking = FrameTrackingResult(
        hand=HandDetectionResult(hand_landmarks=[], handedness=[]),
        pen=None,
    )

    render_frame(
        frame,
        tracking,
        draw_hand=False,
        draw_pen=False,
        draw_pen_path=True,
        pen_path=path,
        overlay_color="yellow",
        overlay_thickness=3,
    )

    # Yellow in BGR is (0, 255, 255), drawn along y=40 between the points.
    assert frame[40, 40, 1] > 200
    assert frame[40, 40, 2] > 200
