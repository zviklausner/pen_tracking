"""Smoke tests for the pen tracking pipeline."""

import threading
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

    assert stats.frames_processed == 42
    assert stats.frames_with_hands >= 41
    assert stats.frames_with_pen >= 41
    assert output_path.is_file()
    assert output_path.stat().st_size > 0

    cap = cv2.VideoCapture(str(output_path))
    assert cap.isOpened()
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 42
    assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 3840
    cap.release()


def test_process_video_reports_progress() -> None:
    calls: list[tuple[int, int]] = []
    config = TrackerConfig(
        video_path=DEFAULT_VIDEO,
        model_path=DEFAULT_MODEL,
        output_path=None,
        frame_stride=10,
        display=False,
    )

    process_video(config, progress=lambda done, total: calls.append((done, total)))

    assert calls
    assert calls[0][0] == 1
    assert [done for done, _ in calls] == list(range(1, len(calls) + 1))


def test_process_video_stops_when_requested(tmp_path: Path) -> None:
    stop_event = threading.Event()

    def progress(done: int, _total: int) -> None:
        if done >= 5:
            stop_event.set()

    config = TrackerConfig(
        video_path=DEFAULT_VIDEO,
        model_path=DEFAULT_MODEL,
        output_path=tmp_path / "partial.mp4",
        frame_stride=1,
        display=False,
    )

    stats = process_video(config, progress=progress, stop_event=stop_event)

    assert stats.frames_processed == 5
    assert (tmp_path / "partial.mp4").is_file()


def test_project_assets_exist() -> None:
    assert DEFAULT_VIDEO.is_file()
    assert DEFAULT_MODEL.is_file()
    assert PROJECT_ROOT.name == "pen_tracking"
