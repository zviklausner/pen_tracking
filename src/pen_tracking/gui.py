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

from pen_tracking.browse_dialog import list_roots
from pen_tracking.config import (
    DEFAULT_OVERLAY_COLOR,
    OVERLAY_COLORS_BGR,
    TRACK_MODES,
    OverlayColorName,
    TrackMode,
    TrackerConfig,
)
from pen_tracking.paths import (
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_DIR,
    ICON_ICO,
    ICON_PNG,
    LOGO_PNG,
    LOGO_UI_PNG,
)
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
APP_USER_MODEL_ID = "AInav.PenTracking.GUI"


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
    return video_path.parent / f"{video_path.stem}_tracked_every{stride}.mp4"


MODE_LABELS: dict[str, TrackMode] = {
    "Both hand and pen": "both",
    "Hand only": "hand",
    "Pen only": "pen",
}
MODE_LABEL_BY_VALUE: dict[TrackMode, str] = {
    mode: label for label, mode in MODE_LABELS.items()
}
DEFAULT_MODE: TrackMode = "both"
LINE_COLOR_LABELS: dict[str, OverlayColorName] = {
    "Yellow": "yellow",
    "Lime": "lime",
    "Cyan": "cyan",
    "Magenta": "magenta",
    "Orange": "orange",
    "White": "white",
    "Red": "red",
}
LINE_COLOR_LABEL_BY_VALUE = {value: label for label, value in LINE_COLOR_LABELS.items()}


def build_config(
    video_path: Path,
    stride: int,
    output_path: Path,
    *,
    track_mode: TrackMode = DEFAULT_MODE,
    track_pen_without_hand: bool = True,
    overlay_color: OverlayColorName = DEFAULT_OVERLAY_COLOR,
    open_output_when_done: bool = False,
) -> TrackerConfig:
    if not MIN_STRIDE <= stride <= MAX_STRIDE:
        raise ValueError(f"Frame stride must be between {MIN_STRIDE} and {MAX_STRIDE}")
    if track_mode not in TRACK_MODES:
        raise ValueError(f"Track mode must be one of {TRACK_MODES}")
    if overlay_color not in OVERLAY_COLORS_BGR:
        raise ValueError(f"Line color must be one of {tuple(OVERLAY_COLORS_BGR)}")
    if not output_path.is_absolute():
        # A bare name typed into the output box is saved next to the input video.
        output_path = video_path.parent / output_path
    if output_path.suffix.lower() != ".mp4":
        output_path = output_path.with_suffix(".mp4")
    if output_path.resolve() == video_path.resolve():
        raise ValueError("The output file would overwrite the input video; choose another name.")
    return TrackerConfig(
        video_path=video_path,
        model_path=DEFAULT_MODEL,
        output_path=output_path,
        frame_stride=stride,
        track_mode=track_mode,
        track_pen_without_hand=track_pen_without_hand,
        overlay_color=overlay_color,
        open_output_when_done=open_output_when_done,
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


def open_path(path: Path) -> None:
    if sys.platform == "win32":
        os.startfile(path)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)
    else:
        subprocess.run(["xdg-open", str(path)], check=False)


def open_folder(path: Path) -> None:
    open_path(path)


def set_windows_app_id() -> None:
    # Without its own AppUserModelID, Windows groups the window under python.exe
    # and shows the Python icon in the taskbar.
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)
    except (AttributeError, OSError):
        pass


def apply_window_icon(root: tk.Tk) -> None:
    try:
        if sys.platform == "win32" and ICON_ICO.is_file():
            root.iconbitmap(default=str(ICON_ICO))
        elif ICON_PNG.is_file():
            root.iconphoto(True, tk.PhotoImage(master=root, file=str(ICON_PNG)))
    except tk.TclError:
        logger.debug("Could not set window icon", exc_info=True)


class PenTrackerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("AInav")
        self.root.resizable(True, False)

        self.video_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.stride_var = tk.StringVar(value=stride_label(DEFAULT_STRIDE))
        self.fps_var = tk.StringVar()
        self.mode_var = tk.StringVar(value=MODE_LABEL_BY_VALUE[DEFAULT_MODE])
        self.pen_without_hand_var = tk.BooleanVar(value=True)
        self.line_color_var = tk.StringVar(
            value=LINE_COLOR_LABEL_BY_VALUE[DEFAULT_OVERLAY_COLOR]
        )
        self.open_when_done_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(
            value="Pick a start folder if you want, then Browse for a video (or paste its path)."
        )

        self._video_fps: float | None = None
        self._output_user_chosen = False
        self._browse_dir = self._default_browse_dir()
        self.root_var = tk.StringVar(value=str(self._browse_dir))
        self._events: queue.Queue[tuple] = queue.Queue()
        self._stop_event: threading.Event | None = None
        self._worker: threading.Thread | None = None
        self._logo_image: tk.PhotoImage | None = None

        self._build_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    @staticmethod
    def _default_browse_dir() -> Path:
        samples = Path(r"C:\AI\Sampels")
        if samples.is_dir():
            return samples
        roots = list_roots()
        return roots[0] if roots else Path.home()

    def _build_widgets(self) -> None:
        frame = ttk.Frame(self.root, padding=12)
        frame.grid(sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        self._place_logo(frame)

        ttk.Label(frame, text="Start folder:").grid(row=1, column=0, sticky="w", pady=4)
        self.root_box = ttk.Combobox(
            frame,
            textvariable=self.root_var,
            values=[str(path) for path in list_roots()],
            state="readonly",
            width=58,
        )
        self.root_box.grid(row=1, column=1, sticky="ew", padx=6)
        self.root_box.bind("<<ComboboxSelected>>", lambda _event: self._on_root_chosen())
        self.folder_button = ttk.Button(
            frame, text="Choose folder...", command=self._choose_folder
        )
        self.folder_button.grid(row=1, column=2, sticky="ew")

        ttk.Label(frame, text="Input video:").grid(row=2, column=0, sticky="w", pady=4)
        self.video_entry = ttk.Entry(frame, textvariable=self.video_var, width=60)
        self.video_entry.grid(row=2, column=1, sticky="ew", padx=6)
        self.video_entry.bind("<FocusOut>", lambda _event: self._on_video_path_edited())
        self.video_entry.bind("<Return>", lambda _event: self._on_video_path_edited())
        self.browse_button = ttk.Button(frame, text="Browse...", command=self._choose_video)
        self.browse_button.grid(row=2, column=2, sticky="ew")

        ttk.Label(frame, text="Output file:").grid(row=3, column=0, sticky="w", pady=4)
        output_entry = ttk.Entry(frame, textvariable=self.output_var, width=60)
        output_entry.grid(row=3, column=1, sticky="ew", padx=6)
        output_entry.bind("<Key>", lambda _event: self._mark_output_chosen())
        self.save_button = ttk.Button(frame, text="Browse...", command=self._choose_output)
        self.save_button.grid(row=3, column=2, sticky="ew")

        ttk.Label(frame, text="Frames to process:").grid(row=4, column=0, sticky="w", pady=4)
        self.stride_box = ttk.Combobox(
            frame,
            textvariable=self.stride_var,
            values=list(STRIDE_OPTIONS),
            state="readonly",
            width=20,
        )
        self.stride_box.grid(row=4, column=1, sticky="w", padx=6)
        self.stride_box.bind("<<ComboboxSelected>>", lambda _event: self._on_stride_changed())
        ttk.Label(frame, textvariable=self.fps_var, foreground="gray").grid(
            row=5, column=1, sticky="w", padx=6
        )

        ttk.Label(frame, text="Track:").grid(row=6, column=0, sticky="w", pady=4)
        mode_row = ttk.Frame(frame)
        mode_row.grid(row=6, column=1, sticky="w", padx=6)
        self.mode_buttons: list[ttk.Radiobutton] = []
        for label in MODE_LABELS:
            button = ttk.Radiobutton(
                mode_row,
                text=label,
                value=label,
                variable=self.mode_var,
                command=self._on_mode_changed,
            )
            button.pack(side="left", padx=(0, 12))
            self.mode_buttons.append(button)

        self.pen_without_hand_check = ttk.Checkbutton(
            frame,
            text="Follow the pen when no hand is visible",
            variable=self.pen_without_hand_var,
        )
        self.pen_without_hand_check.grid(row=7, column=1, sticky="w", padx=6, pady=(4, 0))

        ttk.Label(frame, text="Tracking line:").grid(row=8, column=0, sticky="w", pady=4)
        color_row = ttk.Frame(frame)
        color_row.grid(row=8, column=1, sticky="w", padx=6)
        self.line_color_box = ttk.Combobox(
            color_row,
            textvariable=self.line_color_var,
            values=list(LINE_COLOR_LABELS),
            state="readonly",
            width=12,
        )
        self.line_color_box.pack(side="left")
        self.line_color_box.bind("<<ComboboxSelected>>", lambda _event: self._update_color_swatch())
        self.color_swatch = tk.Canvas(color_row, width=28, height=18, highlightthickness=1)
        self.color_swatch.pack(side="left", padx=8)
        self._update_color_swatch()

        self.open_when_done_check = ttk.Checkbutton(
            frame,
            text="Open the output video automatically when done",
            variable=self.open_when_done_var,
        )
        self.open_when_done_check.grid(row=9, column=1, sticky="w", padx=6, pady=(4, 0))

        self.progress_bar = ttk.Progressbar(frame, mode="determinate", maximum=100)
        self.progress_bar.grid(row=10, column=0, columnspan=3, sticky="ew", pady=(12, 4))

        ttk.Label(frame, textvariable=self.status_var, wraplength=520).grid(
            row=11, column=0, columnspan=3, sticky="w"
        )

        buttons = ttk.Frame(frame)
        buttons.grid(row=12, column=0, columnspan=3, sticky="e", pady=(12, 0))
        self._on_mode_changed()
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
        self.close_button = ttk.Button(buttons, text="Close", command=self._on_close)
        self.close_button.pack(side="left", padx=4)

    def _place_logo(self, frame: ttk.Frame) -> None:
        """Show the AInav logo at the top of the window."""
        logo_path = LOGO_UI_PNG if LOGO_UI_PNG.is_file() else LOGO_PNG
        if not logo_path.is_file():
            return
        try:
            self._logo_image = tk.PhotoImage(master=self.root, file=str(logo_path))
        except tk.TclError:
            logger.debug("Could not load logo image", exc_info=True)
            return
        logo_label = ttk.Label(frame, image=self._logo_image)
        logo_label.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))

    @property
    def stride(self) -> int:
        return STRIDE_OPTIONS.get(self.stride_var.get(), DEFAULT_STRIDE)

    @property
    def track_mode(self) -> TrackMode:
        return MODE_LABELS.get(self.mode_var.get(), DEFAULT_MODE)

    @property
    def overlay_color(self) -> OverlayColorName:
        return LINE_COLOR_LABELS.get(self.line_color_var.get(), DEFAULT_OVERLAY_COLOR)

    def _update_color_swatch(self) -> None:
        bgr = OVERLAY_COLORS_BGR[self.overlay_color]
        # Canvas colors are hex RGB.
        hex_color = f"#{bgr[2]:02x}{bgr[1]:02x}{bgr[0]:02x}"
        self.color_swatch.configure(bg=hex_color)
        self.color_swatch.delete("all")
        self.color_swatch.create_rectangle(0, 0, 28, 18, fill=hex_color, outline="")

    def _on_mode_changed(self) -> None:
        # Hand-only mode has no pen to follow; pen-only always tracks without a hand.
        if self.track_mode == "hand":
            self.pen_without_hand_check.configure(state="disabled")
        elif self.track_mode == "pen":
            self.pen_without_hand_var.set(True)
            self.pen_without_hand_check.configure(state="disabled")
        else:
            self.pen_without_hand_check.configure(state="normal")

    def _current_start_folder(self) -> Path:
        chosen = Path(self.root_var.get()) if self.root_var.get() else self._browse_dir
        return chosen if chosen.is_dir() else self._default_browse_dir()

    def _remember_folder(self, folder: Path) -> None:
        if not folder.is_dir():
            return
        self._browse_dir = folder
        text = str(folder)
        self.root_var.set(text)
        values = list(self.root_box.cget("values"))
        if text not in values:
            self.root_box.configure(values=[*values, text])

    def _on_root_chosen(self) -> None:
        self._remember_folder(self._current_start_folder())

    def _choose_folder(self) -> None:
        folder = filedialog.askdirectory(
            parent=self.root,
            title="Choose start folder / drive",
            initialdir=str(self._current_start_folder()),
            mustexist=True,
        )
        if folder:
            self._remember_folder(Path(folder))

    def _on_video_path_edited(self) -> None:
        text = self.video_var.get().strip().strip('"')
        if not text:
            return
        video_path = Path(text)
        if not video_path.is_file():
            return
        self.video_var.set(str(video_path))
        self._remember_folder(video_path.parent)
        self._video_fps = read_video_fps(video_path)
        if not self._output_user_chosen:
            self.output_var.set(str(suggest_output_path(video_path, self.stride)))
        self._update_fps_label()
        self.status_var.set("Ready.")

    def _choose_video(self) -> None:
        filename = filedialog.askopenfilename(
            parent=self.root,
            title="Choose input video",
            initialdir=str(self._current_start_folder()),
            filetypes=VIDEO_FILETYPES,
        )
        if not filename:
            return
        video_path = Path(filename)
        self._remember_folder(video_path.parent)
        self.video_var.set(str(video_path))
        self._video_fps = read_video_fps(video_path)
        if not self._output_user_chosen:
            self.output_var.set(str(suggest_output_path(video_path, self.stride)))
        self._update_fps_label()
        self.status_var.set("Ready.")

    def _choose_output(self) -> None:
        current = Path(self.output_var.get()) if self.output_var.get() else None
        if current is not None and current.is_absolute():
            initial_dir = current.parent
            initial_name = current.name
        elif self.video_var.get():
            initial_dir = Path(self.video_var.get()).parent
            initial_name = suggest_output_path(Path(self.video_var.get()), self.stride).name
        else:
            initial_dir = self._current_start_folder()
            initial_name = "tracked.mp4"
        filename = filedialog.asksaveasfilename(
            parent=self.root,
            title="Choose output video",
            defaultextension=".mp4",
            filetypes=[("MP4 video", "*.mp4")],
            initialdir=str(initial_dir if initial_dir.is_dir() else self._current_start_folder()),
            initialfile=initial_name,
        )
        if filename:
            chosen = Path(filename)
            self._remember_folder(chosen.parent)
            self.output_var.set(str(chosen))
            self._output_user_chosen = True

    def _mark_output_chosen(self) -> None:
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
        self.folder_button.configure(state=idle_state)
        self.root_box.configure(state="disabled" if running else "readonly")
        self.stride_box.configure(state="disabled" if running else "readonly")
        self.line_color_box.configure(state="disabled" if running else "readonly")
        self.open_when_done_check.configure(state=idle_state)
        for button in self.mode_buttons:
            button.configure(state=idle_state)
        if running:
            self.pen_without_hand_check.configure(state="disabled")
        else:
            self._on_mode_changed()
        self.cancel_button.configure(state="normal" if running else "disabled")

    def _run(self) -> None:
        if not self.video_var.get():
            messagebox.showwarning("AInav", "Please choose an input video.")
            return
        if not self.output_var.get():
            messagebox.showwarning("AInav", "Please choose an output file.")
            return

        try:
            config = build_config(
                Path(self.video_var.get()),
                self.stride,
                Path(self.output_var.get()),
                track_mode=self.track_mode,
                track_pen_without_hand=self.pen_without_hand_var.get(),
                overlay_color=self.overlay_color,
                open_output_when_done=self.open_when_done_var.get(),
            )
        except ValueError as exc:
            messagebox.showerror("AInav", str(exc))
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
                messagebox.showerror("AInav", event[1])
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
            output = Path(self.output_var.get())
            if (
                not cancelled
                and self.open_when_done_var.get()
                and output.is_file()
            ):
                open_path(output)

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
                "AInav", "Processing is still running. Stop and exit?"
            ):
                return
            if self._stop_event is not None:
                self._stop_event.set()
            self._worker.join(timeout=5)
        self.root.destroy()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    set_windows_app_id()
    root = tk.Tk()
    apply_window_icon(root)
    PenTrackerApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
