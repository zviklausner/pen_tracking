"""Tkinter desktop UI for pen tracking."""

from __future__ import annotations

import logging
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2

from pen_tracking.config import TrackerConfig
from pen_tracking.paths import DEFAULT_MODEL, DEFAULT_OUTPUT_DIR
from pen_tracking.video_processor import ProcessingStats, process_video

logger = logging.getLogger(__name__)

MIN_STRIDE = 1
MAX_STRIDE = 10
DEFAULT_STRIDE = 2
VIDEO_FILETYPES = [
    ("Video files", "*.mp4 *.mov *.MOV *.avi *.mkv"),
    ("All files", "*.*"),
]
POLL_INTERVAL_MS = 100


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def stride_label(stride: int) -> str:
    if stride == 1:
        return "Every frame (1)"
    return f"Every {_ordinal(stride)} frame"


STRIDE_OPTIONS: dict[str, int] = {
    stride_label(stride): stride for stride in range(MIN_STRIDE, MAX_STRIDE + 1)
}


def suggest_output_path(video_path: Path, stride: int) -> Path:
    return DEFAULT_OUTPUT_DIR / f"{video_path.stem}_tracked_every{stride}.mp4"


def build_config(video_path: Path, stride: int, output_path: Path) -> TrackerConfig:
    if not MIN_STRIDE <= stride <= MAX_STRIDE:
        raise ValueError(f"Frame stride must be between {MIN_STRIDE} and {MAX_STRIDE}")
    if output_path.suffix.lower() != ".mp4":
        output_path = output_path.with_suffix(".mp4")
    return TrackerConfig(
        video_path=video_path,
        model_path=DEFAULT_MODEL,
        output_path=output_path,
        frame_stride=stride,
        display=False,
    )


def read_video_fps(video_path: Path) -> float | None:
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return None
        return cap.get(cv2.CAP_PROP_FPS) or None
    finally:
        cap.release()


def open_folder(path: Path) -> None:
    if sys.platform == "win32":
        os.startfile(path)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)
    else:
        subprocess.run(["xdg-open", str(path)], check=False)


class PenTrackerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Pen Tracking")
        self.root.resizable(True, False)

        self.video_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.stride_var = tk.StringVar(value=stride_label(DEFAULT_STRIDE))
        self.fps_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Choose a video to start.")

        self._video_fps: float | None = None
        self._output_user_chosen = False
        self._events: queue.Queue[tuple] = queue.Queue()
        self._stop_event: threading.Event | None = None
        self._worker: threading.Thread | None = None

        self._build_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_widgets(self) -> None:
        frame = ttk.Frame(self.root, padding=12)
        frame.grid(sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Input video:").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.video_var, state="readonly", width=60).grid(
            row=0, column=1, sticky="ew", padx=6
        )
        self.browse_button = ttk.Button(frame, text="Browse...", command=self._choose_video)
        self.browse_button.grid(row=0, column=2, sticky="ew")

        ttk.Label(frame, text="Frames to process:").grid(row=1, column=0, sticky="w", pady=4)
        self.stride_box = ttk.Combobox(
            frame,
            textvariable=self.stride_var,
            values=list(STRIDE_OPTIONS),
            state="readonly",
            width=20,
        )
        self.stride_box.grid(row=1, column=1, sticky="w", padx=6)
        self.stride_box.bind("<<ComboboxSelected>>", lambda _event: self._on_stride_changed())
        ttk.Label(frame, textvariable=self.fps_var, foreground="gray").grid(
            row=2, column=1, sticky="w", padx=6
        )

        ttk.Label(frame, text="Output file:").grid(row=3, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.output_var, width=60).grid(
            row=3, column=1, sticky="ew", padx=6
        )
        self.save_button = ttk.Button(frame, text="Save as...", command=self._choose_output)
        self.save_button.grid(row=3, column=2, sticky="ew")

        self.progress_bar = ttk.Progressbar(frame, mode="determinate", maximum=100)
        self.progress_bar.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(12, 4))

        ttk.Label(frame, textvariable=self.status_var, wraplength=520).grid(
            row=5, column=0, columnspan=3, sticky="w"
        )

        buttons = ttk.Frame(frame)
        buttons.grid(row=6, column=0, columnspan=3, sticky="e", pady=(12, 0))
        self.open_folder_button = ttk.Button(
            buttons, text="Open output folder", command=self._open_output_folder, state="disabled"
        )
        self.open_folder_button.pack(side="left", padx=4)
        self.cancel_button = ttk.Button(
            buttons, text="Cancel", command=self._cancel, state="disabled"
        )
        self.cancel_button.pack(side="left", padx=4)
        self.run_button = ttk.Button(buttons, text="Run", command=self._run)
        self.run_button.pack(side="left", padx=4)

    @property
    def stride(self) -> int:
        return STRIDE_OPTIONS.get(self.stride_var.get(), DEFAULT_STRIDE)

    def _choose_video(self) -> None:
        filename = filedialog.askopenfilename(
            title="Choose input video", filetypes=VIDEO_FILETYPES
        )
        if not filename:
            return
        video_path = Path(filename)
        self.video_var.set(str(video_path))
        self._video_fps = read_video_fps(video_path)
        if not self._output_user_chosen:
            self.output_var.set(str(suggest_output_path(video_path, self.stride)))
        self._update_fps_label()
        self.status_var.set("Ready.")

    def _choose_output(self) -> None:
        current = Path(self.output_var.get()) if self.output_var.get() else None
        filename = filedialog.asksaveasfilename(
            title="Save output video as",
            defaultextension=".mp4",
            filetypes=[("MP4 video", "*.mp4")],
            initialdir=str(current.parent if current else DEFAULT_OUTPUT_DIR),
            initialfile=current.name if current else "",
        )
        if filename:
            self.output_var.set(filename)
            self._output_user_chosen = True

    def _on_stride_changed(self) -> None:
        if self.video_var.get() and not self._output_user_chosen:
            self.output_var.set(
                str(suggest_output_path(Path(self.video_var.get()), self.stride))
            )
        self._update_fps_label()

    def _update_fps_label(self) -> None:
        if self._video_fps:
            output_fps = self._video_fps / self.stride
            self.fps_var.set(
                f"Input {self._video_fps:.1f} fps -> output {output_fps:.1f} fps"
            )
        else:
            self.fps_var.set("")

    def _set_running(self, running: bool) -> None:
        idle_state = "disabled" if running else "normal"
        self.run_button.configure(state=idle_state)
        self.browse_button.configure(state=idle_state)
        self.save_button.configure(state=idle_state)
        self.stride_box.configure(state="disabled" if running else "readonly")
        self.cancel_button.configure(state="normal" if running else "disabled")

    def _run(self) -> None:
        if not self.video_var.get():
            messagebox.showwarning("Pen Tracking", "Please choose an input video.")
            return
        if not self.output_var.get():
            messagebox.showwarning("Pen Tracking", "Please choose an output file.")
            return

        try:
            config = build_config(
                Path(self.video_var.get()), self.stride, Path(self.output_var.get())
            )
        except ValueError as exc:
            messagebox.showerror("Pen Tracking", str(exc))
            return
        self.output_var.set(str(config.output_path))

        self._stop_event = threading.Event()
        self.progress_bar.configure(value=0)
        self.open_folder_button.configure(state="disabled")
        self.status_var.set("Processing...")
        self._set_running(True)

        self._worker = threading.Thread(
            target=self._worker_main, args=(config, self._stop_event), daemon=True
        )
        self._worker.start()
        self.root.after(POLL_INTERVAL_MS, self._poll_events)

    def _worker_main(self, config: TrackerConfig, stop_event: threading.Event) -> None:
        try:
            stats = process_video(
                config,
                progress=lambda done, total: self._events.put(("progress", done, total)),
                stop_event=stop_event,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            self._events.put(("error", str(exc)))
        except Exception as exc:  # noqa: BLE001 - surface unexpected failures in the UI
            logger.exception("Processing failed")
            self._events.put(("error", f"Unexpected error: {exc}"))
        else:
            self._events.put(("done", stats, stop_event.is_set()))

    def _poll_events(self) -> None:
        finished = False
        latest_progress: tuple | None = None
        while True:
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                break
            kind = event[0]
            if kind == "progress":
                latest_progress = event
            elif kind == "error":
                finished = True
                self.status_var.set("Failed.")
                messagebox.showerror("Pen Tracking", event[1])
            elif kind == "done":
                finished = True
                self._on_done(event[1], event[2])

        if latest_progress is not None and not finished:
            _, done, total = latest_progress
            if total > 0:
                self.progress_bar.configure(value=min(100.0, 100.0 * done / total))
                self.status_var.set(f"Processing frame {done} of {total}...")
            else:
                self.status_var.set(f"Processing frame {done}...")

        if finished:
            self._set_running(False)
            self._worker = None
        else:
            self.root.after(POLL_INTERVAL_MS, self._poll_events)

    def _on_done(self, stats: ProcessingStats, cancelled: bool) -> None:
        summary = (
            f"{stats.frames_processed} frames processed, "
            f"{stats.frames_with_hands} with hand, {stats.frames_with_pen} with pen."
        )
        if cancelled:
            self.status_var.set(f"Cancelled. {summary} Partial output saved.")
        elif stats.frames_processed == 0:
            self.status_var.set("No frames were processed.")
        else:
            self.progress_bar.configure(value=100)
            self.status_var.set(f"Done. {summary} Saved to {self.output_var.get()}")
        if stats.frames_processed > 0:
            self.open_folder_button.configure(state="normal")

    def _cancel(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
            self.status_var.set("Cancelling...")
            self.cancel_button.configure(state="disabled")

    def _open_output_folder(self) -> None:
        folder = Path(self.output_var.get()).parent
        if folder.is_dir():
            open_folder(folder)

    def _on_close(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            if not messagebox.askyesno(
                "Pen Tracking", "Processing is still running. Stop and exit?"
            ):
                return
            if self._stop_event is not None:
                self._stop_event.set()
            self._worker.join(timeout=5)
        self.root.destroy()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    root = tk.Tk()
    PenTrackerApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
