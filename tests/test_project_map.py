"""Tests for project map injection (project_map.py).

Small models waste their first iterations exploring; a capped, sorted
file listing injected at session start hands them exact paths instead.
"""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from local_cli.project_map import (
    _MAX_ENTRIES,
    build_project_map,
    project_map_enabled,
    project_map_message,
)


def _make_tree(root: Path) -> None:
    (root / "app.py").write_text("x", encoding="utf-8")
    (root / "README.md").write_text("x", encoding="utf-8")
    (root / "src").mkdir()
    (root / "src" / "util.py").write_text("x", encoding="utf-8")
    (root / "node_modules").mkdir()
    (root / "node_modules" / "junk.js").write_text("x", encoding="utf-8")
    (root / ".hidden").write_text("x", encoding="utf-8")


class TestBuildProjectMap(unittest.TestCase):
    def test_walk_lists_sorted_and_prunes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _make_tree(Path(tmp))
            listing = build_project_map(tmp)
            self.assertEqual(
                listing.splitlines(),
                ["README.md", "app.py", "src/util.py"],
            )

    def test_empty_dir_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(build_project_map(tmp))

    def test_entry_cap_with_overflow_note(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(_MAX_ENTRIES + 30):
                (Path(tmp) / f"f{i:04d}.txt").write_text("x", encoding="utf-8")
            listing = build_project_map(tmp)
            lines = listing.splitlines()
            self.assertEqual(len(lines), _MAX_ENTRIES + 1)
            self.assertIn("more files", lines[-1])

    def test_git_repo_respects_gitignore(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(
                ["git", "init", "-q"], cwd=tmp, check=True,
                capture_output=True,
            )
            (root / "keep.py").write_text("x", encoding="utf-8")
            (root / "secret.log").write_text("x", encoding="utf-8")
            (root / ".gitignore").write_text("*.log\n", encoding="utf-8")
            listing = build_project_map(tmp)
            self.assertIn("keep.py", listing)
            self.assertNotIn("secret.log", listing)


class TestProjectMapMessage(unittest.TestCase):
    def test_message_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _make_tree(Path(tmp))
            msg = project_map_message(tmp)
            self.assertEqual(msg["role"], "system")
            self.assertIn("PROJECT MAP", msg["content"])
            self.assertIn("src/util.py", msg["content"])
            self.assertIn("END PROJECT MAP", msg["content"])

    def test_none_for_empty_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(project_map_message(tmp))

    def test_env_kill_switch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _make_tree(Path(tmp))
            with patch.dict(os.environ, {"LOCAL_CLI_PROJECT_MAP": "0"}):
                self.assertFalse(project_map_enabled())
                self.assertIsNone(project_map_message(tmp))

    def test_env_default_on(self) -> None:
        with patch.dict(os.environ):
            os.environ.pop("LOCAL_CLI_PROJECT_MAP", None)
            self.assertTrue(project_map_enabled())


if __name__ == "__main__":
    unittest.main()
