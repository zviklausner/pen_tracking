#!/usr/bin/env python3
"""Cross-platform setup-and-run helper for pen tracking."""

from __future__ import annotations

import subprocess
import sys
import venv
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    venv_dir = root / ".venv"

    if not venv_dir.is_dir():
        print("Creating virtual environment...")
        venv.create(venv_dir, with_pip=True)

    if sys.platform == "win32":
        python = venv_dir / "Scripts" / "python.exe"
    else:
        python = venv_dir / "bin" / "python"

    subprocess.check_call([str(python), "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=root)
    subprocess.check_call([str(python), "-m", "pip", "install", "-q", "-e", "."], cwd=root)
    return subprocess.call([str(python), "-m", "pen_tracking", *sys.argv[1:]], cwd=root)


if __name__ == "__main__":
    raise SystemExit(main())
