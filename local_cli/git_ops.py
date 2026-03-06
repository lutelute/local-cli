"""Git checkpoint and rollback operations for local-cli.

Provides :class:`GitOps` for creating tagged checkpoint commits and
rolling back to previous checkpoints.  Uses ``subprocess.run`` with
``['git', ...]`` for all git operations.
"""

import subprocess
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Prefix for auto-generated checkpoint tag names.
_TAG_PREFIX = "local-cli-checkpoint-"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class GitNotInstalledError(Exception):
    """Raised when the ``git`` executable is not found."""


class GitError(Exception):
    """Raised when a git command fails."""


# ---------------------------------------------------------------------------
# GitOps
# ---------------------------------------------------------------------------


class GitOps:
    """Git checkpoint and rollback operations.

    Wraps common git commands via ``subprocess.run(['git', ...])`` to
    provide checkpoint creation (tagged commits) and rollback
    functionality.  All operations target the current working directory.

    Example::

        ops = GitOps()
        if ops.is_git_repo():
            tag = ops.create_checkpoint("before refactor")
            # ... make risky changes ...
            ops.rollback_to_checkpoint(tag)
    """

    def __init__(self) -> None:
        self._git_available: bool | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_git_available(self) -> None:
        """Verify that ``git`` is installed and accessible.

        Raises:
            GitNotInstalledError: If ``git`` is not found on the system.
        """
        if self._git_available is True:
            return

        try:
            result = subprocess.run(
                ["git", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            self._git_available = result.returncode == 0
        except FileNotFoundError:
            self._git_available = False

        if not self._git_available:
            raise GitNotInstalledError(
                "git is not installed or not found in PATH. "
                "Install git to use checkpoint/rollback features."
            )

    def _run_git(self, *args: str) -> subprocess.CompletedProcess[str]:
        """Run a git command and return the result.

        Args:
            *args: Arguments to pass to ``git`` (e.g. ``'status'``,
                ``'--porcelain'``).

        Returns:
            The completed process result.

        Raises:
            GitNotInstalledError: If git is not available.
            GitError: If the git command exits with a non-zero status.
        """
        self._check_git_available()

        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            stderr = result.stderr.strip()
            raise GitError(
                f"git {' '.join(args)} failed: {stderr}"
            )

        return result

    def _generate_tag_name(self) -> str:
        """Generate a unique checkpoint tag name based on the current time.

        Format: ``local-cli-checkpoint-YYYYMMDD-HHMMSS-ffffff`` (UTC).
        Includes microseconds to avoid collisions on rapid calls.

        Returns:
            A tag name string.
        """
        now = datetime.now(timezone.utc)
        timestamp = now.strftime("%Y%m%d-%H%M%S-%f")
        return f"{_TAG_PREFIX}{timestamp}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_git_repo(self) -> bool:
        """Check whether the current working directory is inside a git repo.

        Returns:
            True if the cwd is inside a git repository, False otherwise.
            Also returns False if git is not installed.
        """
        try:
            self._check_git_available()
        except GitNotInstalledError:
            return False

        try:
            result = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0 and result.stdout.strip() == "true"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def create_checkpoint(self, message: str = "") -> str:
        """Stage all changes and create a tagged checkpoint commit.

        Stages all tracked and untracked files (``git add -A``), creates
        a commit, and tags it with an auto-generated tag name.

        If there are no changes to commit, creates an empty commit so
        that the checkpoint tag still marks a known state.

        Args:
            message: Optional message to include in the commit.  If empty,
                a default checkpoint message is used.

        Returns:
            The tag name of the created checkpoint.

        Raises:
            GitNotInstalledError: If git is not available.
            GitError: If a git command fails.
        """
        tag_name = self._generate_tag_name()

        commit_message = message if message else f"Checkpoint: {tag_name}"

        # Stage all changes.
        self._run_git("add", "-A")

        # Create commit (allow empty in case there are no changes).
        try:
            self._run_git("commit", "-m", commit_message)
        except GitError:
            # If commit fails (e.g. nothing to commit), create an empty
            # commit so the tag still marks a point in history.
            self._run_git("commit", "--allow-empty", "-m", commit_message)

        # Tag the commit.
        self._run_git("tag", tag_name)

        return tag_name

    def rollback_to_checkpoint(self, tag: str) -> None:
        """Roll back the repository to a checkpoint tag.

        Performs a hard reset (``git reset --hard <tag>``) to restore
        the working directory to the state at the given checkpoint.

        Args:
            tag: The checkpoint tag name to roll back to.

        Raises:
            GitNotInstalledError: If git is not available.
            GitError: If the tag does not exist or the reset fails.
        """
        # Verify the tag exists.
        try:
            self._run_git("rev-parse", tag)
        except GitError:
            raise GitError(f"Checkpoint tag '{tag}' not found.")

        self._run_git("reset", "--hard", tag)

    def list_checkpoints(self) -> list[str]:
        """List available checkpoint tags.

        Returns tags matching the ``local-cli-checkpoint-*`` pattern,
        sorted newest first (lexicographic descending on the timestamp
        portion).

        Returns:
            A list of checkpoint tag name strings.  Returns an empty
            list if git is not available or the cwd is not a git repo.
        """
        try:
            self._check_git_available()
        except GitNotInstalledError:
            return []

        try:
            result = subprocess.run(
                ["git", "tag", "-l", f"{_TAG_PREFIX}*"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return []

        if result.returncode != 0:
            return []

        tags = [
            line.strip()
            for line in result.stdout.splitlines()
            if line.strip()
        ]

        # Sort descending (newest first by timestamp in tag name).
        tags.sort(reverse=True)
        return tags
