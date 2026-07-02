"""Smoke tests for the pen tracking pipeline."""

from pathlib import Path

import cv2

from pen_tracking.config import TrackerConfig
from pen_tracking.paths import DEFAULT_MODEL, DEFAULT_VIDEO, PROJECT_ROOT
from pen_tracking.video_processor import process_video


def test_process_sample_video_writes_output(tmp_path: Path) -> None:
    output_path = tmp_path / "tracked.mp4"
    config = TrackerConfig(
        video_path=DEFAULT_VIDEO,
        model_path=DEFAULT_MODEL,
        output_path=output_path,
        display=False,
    )

    stats = process_video(config)

    assert stats.frames_processed == 84
    assert stats.frames_with_hands == 84
    assert stats.frames_with_pen >= 80
    assert output_path.is_file()
    assert output_path.stat().st_size > 0

    cap = cv2.VideoCapture(str(output_path))
    assert cap.isOpened()
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 84
    assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 720
    cap.release()


def test_project_assets_exist() -> None:
    assert DEFAULT_VIDEO.is_file()
    assert DEFAULT_MODEL.is_file()
    assert PROJECT_ROOT.name == "pen_tracking"
