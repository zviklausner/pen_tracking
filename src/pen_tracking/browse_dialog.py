"""Scrollable browse dialogs with an explicit drive/root picker."""

from __future__ import annotations

import string
import tkinter as tk
from pathlib import Path
from tkinter import ttk


def list_roots() -> list[Path]:
    """Return drives and common top-level folders the user can jump to."""
    roots: list[Path] = []
    if hasattr(Path, "home"):
        home = Path.home()
        for name in ("Desktop", "Documents", "Downloads", "Videos"):
            candidate = home / name
            if candidate.is_dir():
                roots.append(candidate)
        samples = Path(r"C:\AI\Sampels")
        if samples.is_dir():
            roots.append(samples)
        ai = Path(r"C:\AI")
        if ai.is_dir():
            roots.append(ai)

    if hasattr(Path, "home"):
        # Prefer Windows drive letters so the user can pick the root of any disk.
        for letter in string.ascii_uppercase:
            drive = Path(f"{letter}:/")
            if drive.exists():
                roots.append(drive)
    return roots


def list_entries(folder: Path) -> tuple[list[Path], list[Path]]:
    """Return (subfolders, files) for a folder, sorted by name."""
    try:
        children = list(folder.iterdir())
    except OSError:
        return [], []
    dirs = sorted(
        (p for p in children if p.is_dir() and not p.name.startswith(".")),
        key=lambda p: p.name.lower(),
    )
    files = sorted(
        (p for p in children if p.is_file() and not p.name.startswith(".")),
        key=lambda p: p.name.lower(),
    )
    return dirs, files


def _match_video(path: Path) -> bool:
    return path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}


class _BrowseDialog(tk.Toplevel):
    """Folder browser with a root chooser, Up button, and a scrollable list."""

    def __init__(
        self,
        parent: tk.Misc,
        *,
        title: str,
        initial_dir: Path,
        mode: str,
        initial_name: str = "",
    ) -> None:
        super().__init__(parent)
        self.title(title)
        self.transient(parent)
        self.grab_set()
        self.resizable(True, True)
        self.geometry("640x480")
        self.result: Path | None = None
        self._mode = mode  # "open" or "save"
        self._current = initial_dir if initial_dir.is_dir() else Path.home()
        self._roots = list_roots()

        body = ttk.Frame(self, padding=10)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(2, weight=1)

        root_row = ttk.Frame(body)
        root_row.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(root_row, text="Root:").pack(side="left")
        self.root_var = tk.StringVar()
        self.root_box = ttk.Combobox(
            root_row,
            textvariable=self.root_var,
            values=[str(path) for path in self._roots],
            state="readonly",
            width=70,
        )
        self.root_box.pack(side="left", fill="x", expand=True, padx=6)
        self.root_box.bind("<<ComboboxSelected>>", self._on_root_chosen)

        path_row = ttk.Frame(body)
        path_row.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        self.up_button = ttk.Button(path_row, text="Up", width=6, command=self._go_up)
        self.up_button.pack(side="left")
        self.path_var = tk.StringVar()
        ttk.Entry(path_row, textvariable=self.path_var, state="readonly").pack(
            side="left", fill="x", expand=True, padx=6
        )

        list_frame = ttk.Frame(body)
        list_frame.grid(row=2, column=0, sticky="nsew")
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        self.listbox = tk.Listbox(list_frame, activestyle="dotbox")
        scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=scroll.set)
        self.listbox.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")
        self.listbox.bind("<Double-Button-1>", self._on_activate)
        self.listbox.bind("<Return>", self._on_activate)
        self.listbox.bind("<<ListboxSelect>>", self._on_select)

        if mode == "save":
            name_row = ttk.Frame(body)
            name_row.grid(row=3, column=0, sticky="ew", pady=(8, 0))
            ttk.Label(name_row, text="File name:").pack(side="left")
            self.name_var = tk.StringVar(value=initial_name or "tracked.mp4")
            ttk.Entry(name_row, textvariable=self.name_var).pack(
                side="left", fill="x", expand=True, padx=6
            )
        else:
            self.name_var = None

        button_row = ttk.Frame(body)
        button_row.grid(row=4, column=0, sticky="e", pady=(10, 0))
        ttk.Button(button_row, text="Cancel", command=self._cancel).pack(side="right", padx=4)
        self.ok_button = ttk.Button(
            button_row, text="Open" if mode == "open" else "Save", command=self._confirm
        )
        self.ok_button.pack(side="right", padx=4)

        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self._entries: list[Path] = []
        self._set_folder(self._current)
        self.wait_window(self)

    def _closest_root(self, folder: Path) -> Path | None:
        resolved = folder.resolve()
        for root in self._roots:
            try:
                resolved.relative_to(root.resolve())
                return root
            except ValueError:
                continue
        return self._roots[0] if self._roots else None

    def _set_folder(self, folder: Path) -> None:
        try:
            folder = folder.resolve()
        except OSError:
            folder = Path.home()
        if not folder.is_dir():
            folder = folder.parent if folder.parent.is_dir() else Path.home()
        self._current = folder
        self.path_var.set(str(folder))
        root = self._closest_root(folder)
        if root is not None:
            self.root_var.set(str(root))
        self.up_button.configure(state="normal" if folder.parent != folder else "disabled")

        dirs, files = list_entries(folder)
        if self._mode == "open":
            files = [path for path in files if _match_video(path)]
        elif self._mode == "save":
            files = [path for path in files if path.suffix.lower() == ".mp4"]

        self._entries = dirs + files
        self.listbox.delete(0, tk.END)
        for path in dirs:
            self.listbox.insert(tk.END, f"[Folder]  {path.name}")
        for path in files:
            self.listbox.insert(tk.END, path.name)

    def _on_root_chosen(self, _event=None) -> None:
        chosen = Path(self.root_var.get())
        if chosen.is_dir():
            self._set_folder(chosen)

    def _go_up(self) -> None:
        if self._current.parent != self._current:
            self._set_folder(self._current.parent)

    def _selected_path(self) -> Path | None:
        selection = self.listbox.curselection()
        if not selection:
            return None
        return self._entries[selection[0]]

    def _on_select(self, _event=None) -> None:
        path = self._selected_path()
        if path is not None and path.is_file() and self.name_var is not None:
            self.name_var.set(path.name)

    def _on_activate(self, _event=None) -> None:
        path = self._selected_path()
        if path is None:
            return
        if path.is_dir():
            self._set_folder(path)
            return
        self._confirm()

    def _confirm(self) -> None:
        if self._mode == "open":
            path = self._selected_path()
            if path is None or not path.is_file():
                return
            self.result = path
            self.destroy()
            return

        name = (self.name_var.get() if self.name_var is not None else "").strip()
        if not name:
            return
        path = self._current / name
        if path.suffix.lower() != ".mp4":
            path = path.with_suffix(".mp4")
        self.result = path
        self.destroy()

    def _cancel(self) -> None:
        self.result = None
        self.destroy()


def browse_open_video(parent: tk.Misc, initial_dir: Path | None = None) -> Path | None:
    start = initial_dir if initial_dir and initial_dir.is_dir() else Path.home()
    dialog = _BrowseDialog(parent, title="Choose input video", initial_dir=start, mode="open")
    return dialog.result


def browse_save_video(
    parent: tk.Misc,
    *,
    initial_dir: Path | None = None,
    initial_name: str = "",
) -> Path | None:
    start = initial_dir if initial_dir and initial_dir.is_dir() else Path.home()
    dialog = _BrowseDialog(
        parent,
        title="Choose output video",
        initial_dir=start,
        mode="save",
        initial_name=initial_name,
    )
    return dialog.result
