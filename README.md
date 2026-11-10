# Pen Tracking

Markerless hand and pen tracking for navigation video analysis (AInav Core).

Detects hand landmarks with MediaPipe and tracks the pen tip with OpenCV.

The pen is found by its own appearance in every frame: a dark, thin shaft on a brighter surface, followed by the light sharpened wood and the graphite point, which is then re-located on the full-resolution frame. When a hand is visible it only tells the tracker which dark shape is the held pen; when no hand is visible (for example at the start of a video) the pen is still followed. Each new tip is checked against the recent average motion, and a sudden jump is ignored unless it persists for several frames; a dark shape without a sharpened point (such as a gap between fingers) is never taken as the tip. If the pen is briefly lost, the last position is kept for a few frames. Pass `--no-pen-without-hand` to only track the pen while a hand is visible.

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

### Installing Python on Windows (PowerShell)

Check whether Python is already installed:

```powershell
py --version
python --version
```

If those commands fail, or `python` opens the Microsoft Store instead of printing a version, install Python with [winget](https://learn.microsoft.com/en-us/windows/package-manager/winget/):

```powershell
winget search Python.Python.3.13
winget install Python.Python.3.13 --accept-package-agreements --accept-source-agreements
```

To list other available Python versions:

```powershell
winget search Python.Python
```

After installation, **close and reopen your terminal** (or restart Cursor), then verify:

```powershell
py --version
py -0p
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

If `.venv\Scripts\activate` fails with a script execution policy error in PowerShell, either allow scripts for your user account:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Or skip activation and run the venv Python directly:

```powershell
.\.venv\Scripts\python.exe -m pen_tracking
```

### Desktop UI

A small Tkinter window lets you pick the input video, choose which frames to process (every frame up to every 10th frame), and choose where the annotated output `.mp4` is saved:

```bash
pen-tracker-gui                  # after pip install -e .
python -m pen_tracking.gui       # equivalent
```

On Windows you can also double-click `scripts\run_gui.bat`, or create a "Pen Tracking" Desktop shortcut (with the app icon from `assets/`) once:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\create_desktop_shortcut.ps1
```
 Processing runs in the background with a progress bar and can be cancelled; a cancelled run keeps the partial output.

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
