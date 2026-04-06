"""Self-update functionality for local-cli.

Updates the installation by running ``git pull`` in the project directory.
"""

import os
import subprocess
import sys


def get_project_root() -> str:
    """Return the root directory of the local-cli project."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check_for_updates() -> tuple[bool, str]:
    """Check if updates are available from the remote repository.

    Returns:
        A tuple of (has_updates, message).
    """
    root = get_project_root()

    try:
        # Fetch latest from remote without merging.
        subprocess.run(
            ["git", "fetch"],
            cwd=root,
            capture_output=True,
            timeout=30,
        )

        # Compare local HEAD with remote tracking branch.
        result = subprocess.run(
            ["git", "status", "-uno"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = result.stdout

        if "Your branch is behind" in output:
            # Extract how many commits behind.
            return True, "Updates available."
        elif "Your branch is up to date" in output:
            return False, "Already up to date."
        else:
            return False, "Could not determine update status."

    except FileNotFoundError:
        return False, "git is not installed."
    except subprocess.TimeoutExpired:
        return False, "Update check timed out."
    except Exception as exc:
        return False, f"Update check failed: {exc}"


def perform_update() -> tuple[bool, str]:
    """Pull the latest changes from the remote repository.

    Returns:
        A tuple of (success, message).
    """
    root = get_project_root()

    try:
        # Check for uncommitted changes.
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if status.stdout.strip():
            return False, "Uncommitted changes detected. Commit or stash them first."

        # Get current version before update.
        from local_cli import __version__ as old_version

        # Pull latest changes.
        result = subprocess.run(
            ["git", "pull", "--ff-only"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            stderr = result.stderr.strip()
            if "not possible to fast-forward" in stderr:
                return False, "Cannot fast-forward. Local changes diverge from remote."
            return False, f"git pull failed: {stderr}"

        output = result.stdout.strip()

        if "Already up to date" in output:
            return True, f"Already up to date (v{old_version})."

        # Always re-install after update to ensure new modules are
        # registered and version metadata is refreshed.
        pip_cmd = [sys.executable, "-m", "pip", "install", "-e", "."]
        # PEP 668 (Python 3.12+): add --break-system-packages if needed.
        try:
            probe = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--dry-run", "-e", "."],
                cwd=root, capture_output=True, text=True, timeout=30,
            )
            if "externally-managed" in probe.stderr:
                pip_cmd.insert(4, "--break-system-packages")
        except Exception:
            pass

        subprocess.run(pip_cmd, cwd=root, capture_output=True, timeout=120)

        # Read new version from the updated source.
        try:
            new_ver_result = subprocess.run(
                [sys.executable, "-c",
                 "from local_cli import __version__; print(__version__)"],
                cwd=root, capture_output=True, text=True, timeout=10,
            )
            new_version = new_ver_result.stdout.strip() or "?"
        except Exception:
            new_version = "?"

        return True, f"Updated v{old_version} → v{new_version}. Restart to use the new version."

    except FileNotFoundError:
        return False, "git is not installed."
    except subprocess.TimeoutExpired:
        return False, "Update timed out."
    except Exception as exc:
        return False, f"Update failed: {exc}"
