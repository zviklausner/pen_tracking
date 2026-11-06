"""Tests for the non-widget helpers of the Tkinter UI."""

from pathlib import Path

import pytest

from pen_tracking.gui import (
    STRIDE_OPTIONS,
    build_config,
    stride_label,
    suggest_output_path,
)
from pen_tracking.paths import DEFAULT_MODEL, DEFAULT_OUTPUT_DIR


def test_stride_options_cover_one_to_ten() -> None:
    assert list(STRIDE_OPTIONS.values()) == list(range(1, 11))


@pytest.mark.parametrize(
    ("stride", "label"),
    [
        (1, "Every frame (1)"),
        (2, "Every 2nd frame"),
        (3, "Every 3rd frame"),
        (4, "Every 4th frame"),
        (10, "Every 10th frame"),
    ],
)
def test_stride_label(stride: int, label: str) -> None:
    assert stride_label(stride) == label
    assert STRIDE_OPTIONS[label] == stride


def test_suggest_output_path() -> None:
    path = suggest_output_path(Path("C:/videos/palm1.MOV"), 3)
    assert path == DEFAULT_OUTPUT_DIR / "palm1_tracked_every3.mp4"


def test_build_config(tmp_path: Path) -> None:
    video = tmp_path / "in.mov"
    output = tmp_path / "out.mp4"
    config = build_config(video, 5, output)

    assert config.video_path == video
    assert config.output_path == output
    assert config.frame_stride == 5
    assert config.model_path == DEFAULT_MODEL
    assert config.display is False


def test_build_config_forces_mp4_suffix(tmp_path: Path) -> None:
    config = build_config(tmp_path / "in.mov", 2, tmp_path / "out.avi")
    assert config.output_path == tmp_path / "out.mp4"


@pytest.mark.parametrize("stride", [0, 11])
def test_build_config_rejects_out_of_range_stride(tmp_path: Path, stride: int) -> None:
    with pytest.raises(ValueError):
        build_config(tmp_path / "in.mov", stride, tmp_path / "out.mp4")
