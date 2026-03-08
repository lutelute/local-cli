"""Tests for local_cli.git_ops module."""

import os
import subprocess
import tempfile
import time
import unittest

from local_cli.git_ops import (
    GitError,
    GitNotInstalledError,
    GitOps,
    _TAG_PREFIX,
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
# Tests: is_git_repo()
# ---------------------------------------------------------------------------


class TestIsGitRepo(unittest.TestCase):
    """Tests for GitOps.is_git_repo()."""

    def test_returns_true_in_git_repo(self) -> None:
        """is_git_repo() returns True inside a git repository."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _init_temp_repo(tmpdir)
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                ops = GitOps()
                self.assertTrue(ops.is_git_repo())
            finally:
                os.chdir(original_cwd)

    def test_returns_false_outside_git_repo(self) -> None:
        """is_git_repo() returns False outside a git repository."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # tmpdir is NOT a git repo.
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                ops = GitOps()
                self.assertFalse(ops.is_git_repo())
            finally:
                os.chdir(original_cwd)

    def test_returns_true_in_subdirectory(self) -> None:
        """is_git_repo() returns True in a subdirectory of a git repo."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _init_temp_repo(tmpdir)
            sub = os.path.join(tmpdir, "subdir")
            os.makedirs(sub)
            original_cwd = os.getcwd()
            try:
                os.chdir(sub)
                ops = GitOps()
                self.assertTrue(ops.is_git_repo())
            finally:
                os.chdir(original_cwd)


# ---------------------------------------------------------------------------
# Tests: create_checkpoint()
# ---------------------------------------------------------------------------


class TestCreateCheckpoint(unittest.TestCase):
    """Tests for GitOps.create_checkpoint()."""

    def test_creates_tagged_commit(self) -> None:
        """create_checkpoint() creates a commit and a tag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _init_temp_repo(tmpdir)
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                ops = GitOps()

                # Create a change.
                with open(os.path.join(tmpdir, "new_file.txt"), "w") as f:
                    f.write("new content\n")

                tag = ops.create_checkpoint("test checkpoint")
                self.assertTrue(tag.startswith(_TAG_PREFIX))

                # Verify tag exists in git.
                result = subprocess.run(
                    ["git", "tag", "-l", tag],
                    capture_output=True,
                    text=True,
                )
                self.assertIn(tag, result.stdout)
            finally:
                os.chdir(original_cwd)

    def test_tag_name_format(self) -> None:
        """Tag name follows 'local-cli-checkpoint-YYYYMMDD-HHMMSS' format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _init_temp_repo(tmpdir)
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                ops = GitOps()
                tag = ops.create_checkpoint()

                self.assertTrue(tag.startswith(_TAG_PREFIX))
                # Extract timestamp portion.
                timestamp = tag[len(_TAG_PREFIX):]
                # Format: YYYYMMDD-HHMMSS-ffffff
                parts = timestamp.split("-")
                self.assertEqual(len(parts), 3)
                self.assertEqual(len(parts[0]), 8)  # YYYYMMDD
                self.assertEqual(len(parts[1]), 6)  # HHMMSS
                self.assertEqual(len(parts[2]), 6)  # ffffff (microseconds)
                self.assertTrue(parts[0].isdigit())
                self.assertTrue(parts[1].isdigit())
                self.assertTrue(parts[2].isdigit())
            finally:
                os.chdir(original_cwd)

    def test_checkpoint_without_changes(self) -> None:
        """create_checkpoint() works even with no pending changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _init_temp_repo(tmpdir)
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                ops = GitOps()
                # No new changes — should still create an empty commit.
                tag = ops.create_checkpoint("no changes")
                self.assertTrue(tag.startswith(_TAG_PREFIX))

                # Verify tag exists.
                result = subprocess.run(
                    ["git", "rev-parse", tag],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0)
            finally:
                os.chdir(original_cwd)

    def test_checkpoint_with_custom_message(self) -> None:
        """create_checkpoint() uses the provided message."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _init_temp_repo(tmpdir)
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                ops = GitOps()
                tag = ops.create_checkpoint("before dangerous refactor")

                # Verify the commit message.
                result = subprocess.run(
                    ["git", "log", "-1", "--format=%s"],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(
                    result.stdout.strip(), "before dangerous refactor"
                )
            finally:
                os.chdir(original_cwd)


# ---------------------------------------------------------------------------
# Tests: rollback_to_checkpoint()
# ---------------------------------------------------------------------------


class TestRollbackToCheckpoint(unittest.TestCase):
    """Tests for GitOps.rollback_to_checkpoint()."""

    def test_rollback_restores_state(self) -> None:
        """rollback_to_checkpoint() restores working directory to checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _init_temp_repo(tmpdir)
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                ops = GitOps()

                # Create a file and checkpoint.
                file_a = os.path.join(tmpdir, "file_a.txt")
                with open(file_a, "w") as f:
                    f.write("original content\n")
                tag = ops.create_checkpoint("safe point")

                # Make a new change after the checkpoint.
                with open(file_a, "w") as f:
                    f.write("modified content\n")
                # Commit the change so rollback has something to undo.
                subprocess.run(
                    ["git", "add", "-A"],
                    capture_output=True,
                    text=True,
                )
                subprocess.run(
                    ["git", "commit", "-m", "post-checkpoint change"],
                    capture_output=True,
                    text=True,
                )

                # Rollback.
                ops.rollback_to_checkpoint(tag)

                # File should be restored to checkpoint state.
                with open(file_a, "r") as f:
                    content = f.read()
                self.assertEqual(content, "original content\n")
            finally:
                os.chdir(original_cwd)

    def test_rollback_nonexistent_tag_raises(self) -> None:
        """rollback_to_checkpoint() raises GitError for unknown tag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _init_temp_repo(tmpdir)
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                ops = GitOps()
                with self.assertRaises(GitError):
                    ops.rollback_to_checkpoint("nonexistent-tag-xyz")
            finally:
                os.chdir(original_cwd)

    def test_rollback_removes_new_files(self) -> None:
        """Files created after the checkpoint are removed on rollback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _init_temp_repo(tmpdir)
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                ops = GitOps()

                # Checkpoint with only README.
                tag = ops.create_checkpoint("clean state")

                # Create a new file after checkpoint.
                new_file = os.path.join(tmpdir, "new_file.txt")
                with open(new_file, "w") as f:
                    f.write("should be removed\n")
                subprocess.run(
                    ["git", "add", "-A"],
                    capture_output=True,
                    text=True,
                )
                subprocess.run(
                    ["git", "commit", "-m", "add new file"],
                    capture_output=True,
                    text=True,
                )
                self.assertTrue(os.path.exists(new_file))

                # Rollback.
                ops.rollback_to_checkpoint(tag)
                self.assertFalse(os.path.exists(new_file))
            finally:
                os.chdir(original_cwd)


# ---------------------------------------------------------------------------
# Tests: list_checkpoints()
# ---------------------------------------------------------------------------


class TestListCheckpoints(unittest.TestCase):
    """Tests for GitOps.list_checkpoints()."""

    def test_empty_when_no_checkpoints(self) -> None:
        """Returns empty list when no checkpoint tags exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _init_temp_repo(tmpdir)
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                ops = GitOps()
                checkpoints = ops.list_checkpoints()
                self.assertEqual(checkpoints, [])
            finally:
                os.chdir(original_cwd)

    def test_lists_created_checkpoints(self) -> None:
        """Created checkpoint tags appear in the list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _init_temp_repo(tmpdir)
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                ops = GitOps()

                tag1 = ops.create_checkpoint("first")
                time.sleep(1.1)  # Ensure unique timestamp-based tag name.
                tag2 = ops.create_checkpoint("second")

                checkpoints = ops.list_checkpoints()
                self.assertIn(tag1, checkpoints)
                self.assertIn(tag2, checkpoints)
            finally:
                os.chdir(original_cwd)

    def test_sorted_newest_first(self) -> None:
        """Checkpoints are sorted with newest first."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _init_temp_repo(tmpdir)
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                ops = GitOps()

                # Create checkpoints and manually set predictable tag names.
                subprocess.run(
                    ["git", "commit", "--allow-empty", "-m", "cp1"],
                    capture_output=True,
                    text=True,
                )
                subprocess.run(
                    ["git", "tag", f"{_TAG_PREFIX}20250101-120000"],
                    capture_output=True,
                    text=True,
                )
                subprocess.run(
                    ["git", "commit", "--allow-empty", "-m", "cp2"],
                    capture_output=True,
                    text=True,
                )
                subprocess.run(
                    ["git", "tag", f"{_TAG_PREFIX}20260601-120000"],
                    capture_output=True,
                    text=True,
                )

                checkpoints = ops.list_checkpoints()
                self.assertEqual(len(checkpoints), 2)
                # Newest first.
                self.assertEqual(
                    checkpoints[0], f"{_TAG_PREFIX}20260601-120000"
                )
                self.assertEqual(
                    checkpoints[1], f"{_TAG_PREFIX}20250101-120000"
                )
            finally:
                os.chdir(original_cwd)

    def test_ignores_non_checkpoint_tags(self) -> None:
        """Tags not matching the checkpoint prefix are excluded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _init_temp_repo(tmpdir)
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                ops = GitOps()

                # Create a non-checkpoint tag.
                subprocess.run(
                    ["git", "tag", "v1.0.0"],
                    capture_output=True,
                    text=True,
                )
                # Create a checkpoint tag.
                tag = ops.create_checkpoint("test")

                checkpoints = ops.list_checkpoints()
                self.assertIn(tag, checkpoints)
                self.assertNotIn("v1.0.0", checkpoints)
            finally:
                os.chdir(original_cwd)

    def test_returns_empty_outside_git_repo(self) -> None:
        """Returns empty list when not in a git repo."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                ops = GitOps()
                checkpoints = ops.list_checkpoints()
                self.assertEqual(checkpoints, [])
            finally:
                os.chdir(original_cwd)


# ---------------------------------------------------------------------------
# Tests: error handling
# ---------------------------------------------------------------------------


class TestGitOpsErrors(unittest.TestCase):
    """Tests for GitOps error handling."""

    def test_create_checkpoint_outside_repo_raises(self) -> None:
        """create_checkpoint() raises GitError outside a git repo."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                ops = GitOps()
                with self.assertRaises(GitError):
                    ops.create_checkpoint("test")
            finally:
                os.chdir(original_cwd)

    def test_rollback_outside_repo_raises(self) -> None:
        """rollback_to_checkpoint() raises GitError outside a git repo."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                ops = GitOps()
                with self.assertRaises(GitError):
                    ops.rollback_to_checkpoint("any-tag")
            finally:
                os.chdir(original_cwd)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestGitOpsIntegration(unittest.TestCase):
    """Integration: checkpoint -> modify -> rollback -> verify cycle."""

    def test_full_checkpoint_rollback_cycle(self) -> None:
        """Full cycle: create file, checkpoint, modify, rollback, verify."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _init_temp_repo(tmpdir)
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                ops = GitOps()

                # 1. Create a file with known content.
                test_file = os.path.join(tmpdir, "data.txt")
                with open(test_file, "w") as f:
                    f.write("version 1\n")

                # 2. Create checkpoint.
                tag = ops.create_checkpoint("v1 checkpoint")
                self.assertTrue(ops.is_git_repo())

                # 3. Modify the file.
                with open(test_file, "w") as f:
                    f.write("version 2\n")
                subprocess.run(
                    ["git", "add", "-A"],
                    capture_output=True,
                    text=True,
                )
                subprocess.run(
                    ["git", "commit", "-m", "update to v2"],
                    capture_output=True,
                    text=True,
                )

                # 4. Verify modification.
                with open(test_file, "r") as f:
                    self.assertEqual(f.read(), "version 2\n")

                # 5. Rollback.
                ops.rollback_to_checkpoint(tag)

                # 6. Verify rollback.
                with open(test_file, "r") as f:
                    self.assertEqual(f.read(), "version 1\n")

                # 7. Tag should still be in list.
                self.assertIn(tag, ops.list_checkpoints())
            finally:
                os.chdir(original_cwd)

    def test_multiple_checkpoints(self) -> None:
        """Multiple checkpoints can be created and listed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _init_temp_repo(tmpdir)
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                ops = GitOps()

                tags = []
                for i in range(3):
                    with open(os.path.join(tmpdir, f"file_{i}.txt"), "w") as f:
                        f.write(f"content {i}\n")
                    if i > 0:
                        time.sleep(1.1)  # Ensure unique timestamp-based tag.
                    tag = ops.create_checkpoint(f"checkpoint {i}")
                    tags.append(tag)

                checkpoints = ops.list_checkpoints()
                for tag in tags:
                    self.assertIn(tag, checkpoints)

                self.assertEqual(len(checkpoints), 3)
            finally:
                os.chdir(original_cwd)


# ---------------------------------------------------------------------------
# Tests: diff_working_tree()
# ---------------------------------------------------------------------------


class TestDiffWorkingTree(unittest.TestCase):
    """Tests for GitOps.diff_working_tree()."""

    def test_diff_no_changes(self) -> None:
        """diff_working_tree() returns message when there are no changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _init_temp_repo(tmpdir)
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                ops = GitOps()
                result = ops.diff_working_tree()
                self.assertEqual(result, "No uncommitted changes.")
            finally:
                os.chdir(original_cwd)

    def test_diff_unstaged_modification(self) -> None:
        """diff_working_tree() shows unstaged modifications."""
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

                result = ops.diff_working_tree(color=False)

                self.assertIn("README.md", result)
                self.assertIn("-# Test Repo", result)
                self.assertIn("+Modified content", result)
            finally:
                os.chdir(original_cwd)

    def test_diff_staged_modification(self) -> None:
        """diff_working_tree() shows staged modifications."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _init_temp_repo(tmpdir)
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                ops = GitOps()

                # Modify and stage the README.
                readme = os.path.join(tmpdir, "README.md")
                with open(readme, "w") as f:
                    f.write("Staged change\n")
                subprocess.run(
                    ["git", "add", "README.md"],
                    capture_output=True,
                    text=True,
                    check=True,
                )

                result = ops.diff_working_tree(color=False)

                self.assertIn("README.md", result)
                self.assertIn("-# Test Repo", result)
                self.assertIn("+Staged change", result)
            finally:
                os.chdir(original_cwd)

    def test_diff_untracked_file(self) -> None:
        """diff_working_tree() shows new untracked files as additions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _init_temp_repo(tmpdir)
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                ops = GitOps()

                # Create a new untracked file.
                new_file = os.path.join(tmpdir, "new_file.txt")
                with open(new_file, "w") as f:
                    f.write("new content\n")

                result = ops.diff_working_tree(color=False)

                self.assertIn("new_file.txt", result)
                self.assertIn("+new content", result)
            finally:
                os.chdir(original_cwd)

    def test_diff_deleted_file(self) -> None:
        """diff_working_tree() shows deleted files as all removals."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _init_temp_repo(tmpdir)
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                ops = GitOps()

                # Delete the README.
                readme = os.path.join(tmpdir, "README.md")
                os.remove(readme)

                result = ops.diff_working_tree(color=False)

                self.assertIn("README.md", result)
                self.assertIn("-# Test Repo", result)
            finally:
                os.chdir(original_cwd)

    def test_diff_multiple_files(self) -> None:
        """diff_working_tree() includes all changed files."""
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

                # Modify both files.
                readme = os.path.join(tmpdir, "README.md")
                with open(readme, "w") as f:
                    f.write("changed readme\n")
                with open(file_b, "w") as f:
                    f.write("changed B\n")

                result = ops.diff_working_tree(color=False)

                self.assertIn("README.md", result)
                self.assertIn("file_b.txt", result)
                self.assertIn("+changed readme", result)
                self.assertIn("+changed B", result)
            finally:
                os.chdir(original_cwd)

    def test_diff_binary_file_placeholder(self) -> None:
        """diff_working_tree() shows placeholder for binary files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _init_temp_repo(tmpdir)
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                ops = GitOps()

                # Create a binary file (contains null bytes).
                bin_file = os.path.join(tmpdir, "image.bin")
                with open(bin_file, "wb") as f:
                    f.write(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")

                result = ops.diff_working_tree(color=False)

                self.assertIn("[binary file: image.bin]", result)
            finally:
                os.chdir(original_cwd)

    def test_diff_color_disabled(self) -> None:
        """diff_working_tree(color=False) omits ANSI escape codes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _init_temp_repo(tmpdir)
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                ops = GitOps()

                readme = os.path.join(tmpdir, "README.md")
                with open(readme, "w") as f:
                    f.write("modified\n")

                result = ops.diff_working_tree(color=False)

                self.assertNotIn("\033[", result)
            finally:
                os.chdir(original_cwd)

    def test_diff_color_enabled(self) -> None:
        """diff_working_tree(color=True) includes ANSI escape codes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _init_temp_repo(tmpdir)
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                ops = GitOps()

                readme = os.path.join(tmpdir, "README.md")
                with open(readme, "w") as f:
                    f.write("modified\n")

                result = ops.diff_working_tree(color=True)

                self.assertIn("\033[", result)
            finally:
                os.chdir(original_cwd)

    def test_diff_truncation(self) -> None:
        """diff_working_tree() truncates large diffs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _init_temp_repo(tmpdir)
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                ops = GitOps()

                # Create a file with many lines.
                big_file = os.path.join(tmpdir, "big.txt")
                with open(big_file, "w") as f:
                    for i in range(200):
                        f.write(f"line {i}\n")
                subprocess.run(
                    ["git", "add", "-A"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                subprocess.run(
                    ["git", "commit", "-m", "add big file"],
                    capture_output=True,
                    text=True,
                    check=True,
                )

                # Modify every line.
                with open(big_file, "w") as f:
                    for i in range(200):
                        f.write(f"changed {i}\n")

                result = ops.diff_working_tree(color=False, max_lines=10)

                self.assertIn("... truncated", result)
            finally:
                os.chdir(original_cwd)

    def test_diff_outside_git_repo_raises(self) -> None:
        """diff_working_tree() raises GitError outside a git repository."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                ops = GitOps()
                with self.assertRaises(GitError):
                    ops.diff_working_tree()
            finally:
                os.chdir(original_cwd)

    def test_diff_mixed_staged_unstaged_and_untracked(self) -> None:
        """diff_working_tree() handles mix of staged, unstaged, and untracked."""
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

                # Staged change to README.
                readme = os.path.join(tmpdir, "README.md")
                with open(readme, "w") as f:
                    f.write("staged readme\n")
                subprocess.run(
                    ["git", "add", "README.md"],
                    capture_output=True,
                    text=True,
                    check=True,
                )

                # Unstaged change to file_b.
                with open(file_b, "w") as f:
                    f.write("unstaged B\n")

                # New untracked file.
                new_file = os.path.join(tmpdir, "new.txt")
                with open(new_file, "w") as f:
                    f.write("brand new\n")

                result = ops.diff_working_tree(color=False)

                self.assertIn("README.md", result)
                self.assertIn("file_b.txt", result)
                self.assertIn("new.txt", result)
                self.assertIn("+staged readme", result)
                self.assertIn("+unstaged B", result)
                self.assertIn("+brand new", result)
            finally:
                os.chdir(original_cwd)

    def test_diff_newly_staged_file(self) -> None:
        """diff_working_tree() shows files staged for the first time."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _init_temp_repo(tmpdir)
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                ops = GitOps()

                # Create a new file and stage it (not committed).
                new_file = os.path.join(tmpdir, "added.txt")
                with open(new_file, "w") as f:
                    f.write("newly added\n")
                subprocess.run(
                    ["git", "add", "added.txt"],
                    capture_output=True,
                    text=True,
                    check=True,
                )

                result = ops.diff_working_tree(color=False)

                self.assertIn("added.txt", result)
                self.assertIn("+newly added", result)
            finally:
                os.chdir(original_cwd)

    def test_diff_deduplicates_files(self) -> None:
        """Files appearing in both staged and unstaged are only shown once."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _init_temp_repo(tmpdir)
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                ops = GitOps()

                # Stage a change, then modify again (appears in both lists).
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
                    f.write("unstaged version\n")

                result = ops.diff_working_tree(color=False)

                # The file header should appear once.
                # Count occurrences of the file in diff headers.
                count = result.count("--- a/README.md")
                self.assertEqual(count, 1)
            finally:
                os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()
