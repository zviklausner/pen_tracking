# Pen Tracking

Markerless hand and pen tracking for navigation video analysis (AInav Core).

Detects hand landmarks with MediaPipe and tracks the pen tip held in the hand using OpenCV color segmentation guided by finger geometry.

## Requirements

- Python **3.10+** (developed and tested on **3.13.7**)
- macOS, Linux, or Windows

## Quick start

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

python -m pen_tracking           # writes output/tracked_sample.mp4
python -m pen_tracking --display # live preview (press q to quit)
```

### Windows (PowerShell or CMD)

```powershell
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

python -m pen_tracking
python -m pen_tracking --display
```

### Cross-platform helper (any OS)

```bash
python scripts/run_tracker.py
```

On Windows you can also double-click or run:

```cmd
scripts\run_tracker.bat
```

## Project layout

```
pen_tracker.py          # original prototype (unchanged)
sample.mp4              # sample input video
models/                 # MediaPipe hand_landmarker.task
src/pen_tracking/       # package source
output/                 # generated annotated videos
tests/                  # pytest smoke tests
```

## Dependency files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Package metadata, installable entry point (`pen-tracker`), and core dependencies for `pip install -e .` |
| `requirements.txt` | Pinned direct runtime deps for venv setup |
| `requirements.lock` | Full transitive lockfile from `pip freeze` |

## Cross-platform notes

- Paths use `pathlib` and work on both `\` (Windows) and `/` (Mac/Linux).
- Default mode writes `output/tracked_sample.mp4` without opening a GUI window.
- `--display` uses OpenCV windows and works on Windows desktops when a display is available.
- Video output tries multiple codecs automatically if the first one is unavailable on your OS.

## Tests

```bash
pytest tests/ -v
```
