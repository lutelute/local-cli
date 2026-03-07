#!/usr/bin/env python3
"""Launcher for local-cli: pick a folder and model, then start the agent."""

import os
import subprocess
import sys

# Add project root to path so we can import local_cli
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from local_cli.model_selector import select_model_interactive
from local_cli.ollama_client import OllamaClient, OllamaConnectionError


def _pick_directory_macos() -> str | None:
    """Open a macOS Finder dialog to pick a directory."""
    script = (
        'tell application "System Events"\n'
        '  activate\n'
        '  set theFolder to choose folder with prompt '
        '"local-cli: Select working directory"\n'
        '  return POSIX path of theFolder\n'
        'end tell'
    )
    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        return None
    path = result.stdout.strip().rstrip("/")
    return path if path and os.path.isdir(path) else None


def _pick_directory_windows() -> str | None:
    """Open a Windows folder picker dialog via PowerShell."""
    ps_script = (
        "Add-Type -AssemblyName System.Windows.Forms; "
        "$d = New-Object System.Windows.Forms.FolderBrowserDialog; "
        "$d.Description = 'local-cli: Select working directory'; "
        "$d.ShowNewFolderButton = $false; "
        "if ($d.ShowDialog() -eq 'OK') { $d.SelectedPath } "
        "else { exit 1 }"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps_script],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        return None
    path = result.stdout.strip()
    return path if path and os.path.isdir(path) else None


def _pick_directory_linux() -> str | None:
    """Open a Linux folder picker dialog via zenity or kdialog."""
    for cmd in (
        ["zenity", "--file-selection", "--directory",
         "--title=local-cli: Select working directory"],
        ["kdialog", "--getexistingdirectory", os.getcwd(),
         "--title", "local-cli: Select working directory"],
    ):
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0:
                path = result.stdout.strip()
                if path and os.path.isdir(path):
                    return path
        except FileNotFoundError:
            continue
    return None


def pick_directory() -> str | None:
    """Open a native folder picker dialog."""
    try:
        if sys.platform == "darwin":
            return _pick_directory_macos()
        elif sys.platform == "win32":
            return _pick_directory_windows()
        else:
            return _pick_directory_linux()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        sys.stderr.write("Error: Could not open folder picker.\n")
        return None


def main() -> None:
    # 1. Pick directory
    directory = pick_directory()
    if directory is None:
        return

    # 2. Connect to Ollama
    try:
        client = OllamaClient()
        client.get_version()
    except (OllamaConnectionError, Exception) as exc:
        sys.stderr.write(f"Error: Cannot connect to Ollama. Is it running?\n")
        sys.exit(1)

    # 3. Select model
    sys.stdout.write("\n")
    selected = select_model_interactive(client, current_model="")
    if selected is None:
        sys.stdout.write("Cancelled.\n")
        return

    # 4. Launch local-cli
    sys.stdout.write(f"\nStarting local-cli...\n")
    sys.stdout.write(f"  Directory: {directory}\n")
    sys.stdout.write(f"  Model:     {selected}\n\n")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = project_root + os.pathsep + pythonpath if pythonpath else project_root

    os.chdir(directory)
    cmd = [sys.executable, "-m", "local_cli", "--model", selected]
    if sys.platform == "win32":
        raise SystemExit(subprocess.call(cmd, env=env))
    else:
        os.execvpe(sys.executable, cmd, env)


if __name__ == "__main__":
    main()
