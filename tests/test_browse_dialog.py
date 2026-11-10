"""Tests for the scrollable browse helpers."""

from pathlib import Path

from pen_tracking.browse_dialog import list_entries, list_roots


def test_list_roots_includes_at_least_one_path() -> None:
    roots = list_roots()
    assert roots
    assert all(isinstance(path, Path) for path in roots)


def test_list_entries_separates_folders_and_files(tmp_path: Path) -> None:
    (tmp_path / "folder_a").mkdir()
    (tmp_path / "folder_b").mkdir()
    (tmp_path / "clip.mp4").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("hi", encoding="utf-8")

    dirs, files = list_entries(tmp_path)

    assert [path.name for path in dirs] == ["folder_a", "folder_b"]
    assert [path.name for path in files] == ["clip.mp4", "notes.txt"]


def test_list_entries_handles_missing_folder(tmp_path: Path) -> None:
    dirs, files = list_entries(tmp_path / "missing")
    assert dirs == []
    assert files == []
