"""Project path helpers."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIDEO = Path(r"C:\AI\Sampels\palm1.MOV")
DEFAULT_MODEL = PROJECT_ROOT / "models" / "hand_landmarker.task"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
DESKTOP_DIR = Path.home() / "Desktop"
DEFAULT_INPUT_DIR = DESKTOP_DIR / "sampels" / "video files for processing"
DEFAULT_PROCESSED_DIR = DESKTOP_DIR / "sampels" / "processed video files"


def ensure_media_dirs() -> None:
    """Create the default input and output folders used by the file dialogs."""
    DEFAULT_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
ICON_ICO = PROJECT_ROOT / "assets" / "icon.ico"
ICON_PNG = PROJECT_ROOT / "assets" / "icon.png"
LOGO_PNG = PROJECT_ROOT / "assets" / "ainav_logo.png"
LOGO_UI_PNG = PROJECT_ROOT / "assets" / "ainav_logo_ui.png"
