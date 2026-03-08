"""Tests for GitOps.undo_last_change() method."""

import os
import subprocess
import tempfile
import unittest

from local_cli.git_ops import (
    GitError,
    GitOps,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _init_temp_repo(tmpdir: str) -> str:
    """Initialize a git repo in a temporary directory.

    Creates an initial commit so the repo has a valid HEAD.

    Args:
        tmpdir: Directory to initialize the git repo in.

    Returns:
        The path to the repo directory.
    """
    subprocess.run(
        ["git", "init", tmpdir],
        capture_output=True,
        text=True,
        check=True,
    )
    subprocess.run(
        ["git", "-C", tmpdir, "config", "user.email", "test@test.com"],
        capture_output=True,
        text=True,
        check=True,
    )
    subprocess.run(
        ["git", "-C", tmpdir, "config", "user.name", "Test User"],
        capture_output=True,
        text=True,
        check=True,
    )
    # Create initial commit.
    readme = os.path.join(tmpdir, "README.md")
    with open(readme, "w") as f:
        f.write("# Test Repo\n")
    subprocess.run(
        ["git", "-C", tmpdir, "add", "-A"],
        capture_output=True,
        text=True,
        check=True,
    )
    subprocess.run(
        ["git", "-C", tmpdir, "commit", "-m", "Initial commit"],
        capture_output=True,
        text=True,
        check=True,
    )
    return tmpdir


# ---------------------------------------------------------------------------
# Tests: undo_last_change()
# ---------------------------------------------------------------------------


class TestUndoLastChange(unittest.TestCase):
    """Tests for GitOps.undo_last_change()."""

    def test_undo_reverts_unstaged_modified_file(self) -> None:
        """undo_last_change() restores an unstaged modified file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _init_temp_repo(tmpdir)
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                ops = GitOps()

                # Modify the committed README.
                readme = os.path.join(tmpdir, "README.md")
                with open(readme, "w") as f:
                    f.write("Modified content\n")

                result = ops.undo_last_change()

                # File should be restored.
                with open(readme, "r") as f:
                    content = f.read()
                self.assertEqual(content, "# Test Repo\n")
                self.assertIn("Reverted 1 file(s)", result)
                self.assertIn("README.md", result)
            finally:
                os.chdir(original_cwd)

    def test_undo_reverts_staged_modified_file(self) -> None:
        """undo_last_change() restores a staged (git add) modified file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _init_temp_repo(tmpdir)
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                ops = GitOps()

                # Modify and stage the file.
                readme = os.path.join(tmpdir, "README.md")
                with open(readme, "w") as f:
                    f.write("Staged modification\n")
                subprocess.run(
                    ["git", "add", "README.md"],
                    capture_output=True,
                    text=True,
                    check=True,
                )

                result = ops.undo_last_change()

                # File should be restored.
                with open(readme, "r") as f:
                    content = f.read()
                self.assertEqual(content, "# Test Repo\n")
                self.assertIn("Reverted 1 file(s)", result)
            finally:
                os.chdir(original_cwd)

    def test_undo_reverts_both_staged_and_unstaged(self) -> None:
        """undo_last_change() reverts mixed staged and unstaged changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _init_temp_repo(tmpdir)
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                ops = GitOps()

                # Create and commit a second file.
                file_b = os.path.join(tmpdir, "file_b.txt")
                with open(file_b, "w") as f:
                    f.write("original B\n")
                subprocess.run(
                    ["git", "add", "-A"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                subprocess.run(
                    ["git", "commit", "-m", "add file_b"],
                    capture_output=True,
                    text=True,
                    check=True,
                )

                # Stage a change to README.
                readme = os.path.join(tmpdir, "README.md")
                with open(readme, "w") as f:
                    f.write("staged change\n")
                subprocess.run(
                    ["git", "add", "README.md"],
                    capture_output=True,
                    text=True,
                    check=True,
                )

                # Make an unstaged change to file_b.
                with open(file_b, "w") as f:
                    f.write("unstaged change\n")

                result = ops.undo_last_change()

                # Both files should be restored.
                with open(readme, "r") as f:
                    self.assertEqual(f.read(), "# Test Repo\n")
                with open(file_b, "r") as f:
                    self.assertEqual(f.read(), "original B\n")
                self.assertIn("Reverted 2 file(s)", result)
            finally:
                os.chdir(original_cwd)

    def test_undo_no_changes_returns_message(self) -> None:
        """undo_last_change() returns helpful message when nothing to undo."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _init_temp_repo(tmpdir)
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                ops = GitOps()

                result = ops.undo_last_change()

                self.assertIn("No changes to undo", result)
            finally:
                os.chdir(original_cwd)

    def test_undo_does_not_remove_untracked_files(self) -> None:
        """undo_last_change() does NOT remove untracked (new) files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _init_temp_repo(tmpdir)
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                ops = GitOps()

                # Create an untracked file.
                new_file = os.path.join(tmpdir, "new_file.txt")
                with open(new_file, "w") as f:
                    f.write("untracked content\n")

                result = ops.undo_last_change()

                # Untracked file should still exist.
                self.assertTrue(os.path.exists(new_file))
                with open(new_file, "r") as f:
                    self.assertEqual(f.read(), "untracked content\n")
            finally:
                os.chdir(original_cwd)

    def test_undo_notes_untracked_files_exist(self) -> None:
        """undo_last_change() mentions untracked files in the summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _init_temp_repo(tmpdir)
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                ops = GitOps()

                # Create an untracked file (no tracked changes).
                new_file = os.path.join(tmpdir, "new_file.txt")
                with open(new_file, "w") as f:
                    f.write("untracked\n")

                result = ops.undo_last_change()

                self.assertIn("No changes to undo", result)
                self.assertIn("untracked file(s) exist", result)
                self.assertIn("not removed by undo", result)
            finally:
                os.chdir(original_cwd)

    def test_undo_notes_untracked_when_reverting(self) -> None:
        """Summary includes untracked note when reverting tracked changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _init_temp_repo(tmpdir)
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                ops = GitOps()

                # Create an untracked file.
                new_file = os.path.join(tmpdir, "new_file.txt")
                with open(new_file, "w") as f:
                    f.write("untracked\n")

                # Modify a tracked file.
                readme = os.path.join(tmpdir, "README.md")
                with open(readme, "w") as f:
                    f.write("modified\n")

                result = ops.undo_last_change()

                self.assertIn("Reverted 1 file(s)", result)
                self.assertIn("untracked file(s) exist", result)
            finally:
                os.chdir(original_cwd)

    def test_undo_confirmation_needed_for_many_files(self) -> None:
        """undo_last_change() requires confirmation when >3 files changed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _init_temp_repo(tmpdir)
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                ops = GitOps()

                # Create and commit 4 files.
                for i in range(4):
                    fpath = os.path.join(tmpdir, f"file_{i}.txt")
                    with open(fpath, "w") as f:
                        f.write(f"original {i}\n")
                subprocess.run(
                    ["git", "add", "-A"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                subprocess.run(
                    ["git", "commit", "-m", "add 4 files"],
                    capture_output=True,
                    text=True,
                    check=True,
                )

                # Modify all 4 files.
                for i in range(4):
                    fpath = os.path.join(tmpdir, f"file_{i}.txt")
                    with open(fpath, "w") as f:
                        f.write(f"modified {i}\n")

                # Without confirmation, should NOT revert.
                result = ops.undo_last_change(confirmed=False)

                self.assertIn("4 files have changes", result)
                self.assertIn("Run with confirmation", result)
                # Files should still be modified.
                with open(os.path.join(tmpdir, "file_0.txt"), "r") as f:
                    self.assertEqual(f.read(), "modified 0\n")
            finally:
                os.chdir(original_cwd)

    def test_undo_confirmed_reverts_many_files(self) -> None:
        """undo_last_change(confirmed=True) reverts even when >3 files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _init_temp_repo(tmpdir)
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                ops = GitOps()

                # Create and commit 5 files.
                for i in range(5):
                    fpath = os.path.join(tmpdir, f"file_{i}.txt")
                    with open(fpath, "w") as f:
                        f.write(f"original {i}\n")
                subprocess.run(
                    ["git", "add", "-A"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                subprocess.run(
                    ["git", "commit", "-m", "add 5 files"],
                    capture_output=True,
                    text=True,
                    check=True,
                )

                # Modify all 5 files.
                for i in range(5):
                    fpath = os.path.join(tmpdir, f"file_{i}.txt")
                    with open(fpath, "w") as f:
                        f.write(f"modified {i}\n")

                # With confirmation, should revert all.
                result = ops.undo_last_change(confirmed=True)

                self.assertIn("Reverted 5 file(s)", result)
                for i in range(5):
                    fpath = os.path.join(tmpdir, f"file_{i}.txt")
                    with open(fpath, "r") as f:
                        self.assertEqual(f.read(), f"original {i}\n")
            finally:
                os.chdir(original_cwd)

    def test_undo_three_files_does_not_require_confirmation(self) -> None:
        """undo_last_change() reverts without confirmation for exactly 3 files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _init_temp_repo(tmpdir)
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                ops = GitOps()

                # Create and commit 3 files (README already exists).
                for i in range(2):
                    fpath = os.path.join(tmpdir, f"file_{i}.txt")
                    with open(fpath, "w") as f:
                        f.write(f"original {i}\n")
                subprocess.run(
                    ["git", "add", "-A"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                subprocess.run(
                    ["git", "commit", "-m", "add 2 more files"],
                    capture_output=True,
                    text=True,
                    check=True,
                )

                # Modify all 3 tracked files.
                readme = os.path.join(tmpdir, "README.md")
                with open(readme, "w") as f:
                    f.write("modified\n")
                for i in range(2):
                    fpath = os.path.join(tmpdir, f"file_{i}.txt")
                    with open(fpath, "w") as f:
                        f.write(f"modified {i}\n")

                # Should revert without confirmation (3 files, not >3).
                result = ops.undo_last_change(confirmed=False)

                self.assertIn("Reverted 3 file(s)", result)
            finally:
                os.chdir(original_cwd)

    def test_undo_outside_git_repo_raises(self) -> None:
        """undo_last_change() raises GitError outside a git repository."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                ops = GitOps()
                with self.assertRaises(GitError):
                    ops.undo_last_change()
            finally:
                os.chdir(original_cwd)

    def test_undo_deduplicates_staged_and_unstaged(self) -> None:
        """Files appearing in both staged and unstaged diffs are listed once."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _init_temp_repo(tmpdir)
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                ops = GitOps()

                # Stage a change to README, then modify it again (unstaged).
                readme = os.path.join(tmpdir, "README.md")
                with open(readme, "w") as f:
                    f.write("staged version\n")
                subprocess.run(
                    ["git", "add", "README.md"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                with open(readme, "w") as f:
                    f.write("unstaged version on top\n")

                result = ops.undo_last_change()

                self.assertIn("Reverted 1 file(s)", result)
                # File should be back to committed state.
                with open(readme, "r") as f:
                    self.assertEqual(f.read(), "# Test Repo\n")
            finally:
                os.chdir(original_cwd)

    def test_undo_confirmation_lists_all_files(self) -> None:
        """Confirmation message lists all affected file names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _init_temp_repo(tmpdir)
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                ops = GitOps()

                # Create and commit 4 files.
                filenames = ["alpha.txt", "beta.txt", "gamma.txt", "delta.txt"]
                for name in filenames:
                    fpath = os.path.join(tmpdir, name)
                    with open(fpath, "w") as f:
                        f.write(f"original {name}\n")
                subprocess.run(
                    ["git", "add", "-A"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                subprocess.run(
                    ["git", "commit", "-m", "add files"],
                    capture_output=True,
                    text=True,
                    check=True,
                )

                # Modify all 4.
                for name in filenames:
                    fpath = os.path.join(tmpdir, name)
                    with open(fpath, "w") as f:
                        f.write(f"modified {name}\n")

                result = ops.undo_last_change(confirmed=False)

                for name in filenames:
                    self.assertIn(name, result)
            finally:
                os.chdir(original_cwd)

    def test_undo_confirmation_with_untracked_note(self) -> None:
        """Confirmation message includes untracked file note."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _init_temp_repo(tmpdir)
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                ops = GitOps()

                # Create and commit 4 files.
                for i in range(4):
                    fpath = os.path.join(tmpdir, f"file_{i}.txt")
                    with open(fpath, "w") as f:
                        f.write(f"original {i}\n")
                subprocess.run(
                    ["git", "add", "-A"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                subprocess.run(
                    ["git", "commit", "-m", "add files"],
                    capture_output=True,
                    text=True,
                    check=True,
                )

                # Modify all 4 + create an untracked file.
                for i in range(4):
                    fpath = os.path.join(tmpdir, f"file_{i}.txt")
                    with open(fpath, "w") as f:
                        f.write(f"modified {i}\n")
                untracked = os.path.join(tmpdir, "untracked.txt")
                with open(untracked, "w") as f:
                    f.write("new file\n")

                result = ops.undo_last_change(confirmed=False)

                self.assertIn("4 files have changes", result)
                self.assertIn("untracked file(s) exist", result)
            finally:
                os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()
