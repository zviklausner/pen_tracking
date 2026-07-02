"""Project path helpers."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIDEO = PROJECT_ROOT / "sample.mp4"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "hand_landmarker.task"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
