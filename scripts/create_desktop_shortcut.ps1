# Creates a "Pen Tracking" shortcut on the current user's Desktop that opens the UI.
# Usage: powershell -ExecutionPolicy Bypass -File scripts\create_desktop_shortcut.ps1

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$pythonw = Join-Path $root ".venv\Scripts\pythonw.exe"
$icon = Join-Path $root "assets\icon.ico"

if (-not (Test-Path $pythonw)) {
    Write-Error "Virtual environment not found at $pythonw. Run scripts\run_tracker.bat once to create it."
}

$desktop = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path $desktop "Pen Tracking.lnk"

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $pythonw
$shortcut.Arguments = "-m pen_tracking.gui"
$shortcut.WorkingDirectory = $root
$shortcut.Description = "Pen Tracking - hand and pen tracking for video"
if (Test-Path $icon) {
    $shortcut.IconLocation = "$icon,0"
}
$shortcut.Save()

Write-Host "Created $shortcutPath"
