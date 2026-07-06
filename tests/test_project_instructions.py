"""Tests for project instruction files (LOCAL_CLI.md / AGENTS.md / CLAUDE.md).

The per-project steering lever: a file the user drops into the project
is injected into every session as a system message.  Small models need
the re-assertion — they forget project conventions between turns.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from local_cli.project_instructions import (
    _MAX_CHARS,
    build_instruction_message,
    find_instruction_file,
    load_project_instructions,
    project_instruction_message,
    project_instructions_enabled,
)


class TestFindInstructionFile(unittest.TestCase):
    def test_no_file_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / ".git").mkdir()  # fence the walk inside tmp
            self.assertIsNone(find_instruction_file(tmp))

    def test_finds_each_name(self) -> None:
        for name in ("LOCAL_CLI.md", "AGENTS.md", "CLAUDE.md"):
            with tempfile.TemporaryDirectory() as tmp:
                (Path(tmp) / ".git").mkdir()
                (Path(tmp) / name).write_text("rules", encoding="utf-8")
                found = find_instruction_file(tmp)
                self.assertIsNotNone(found)
                self.assertEqual(found.name, name)

    def test_precedence_local_cli_over_agents_over_claude(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / ".git").mkdir()
            (Path(tmp) / "CLAUDE.md").write_text("c", encoding="utf-8")
            (Path(tmp) / "AGENTS.md").write_text("a", encoding="utf-8")
            self.assertEqual(find_instruction_file(tmp).name, "AGENTS.md")
            (Path(tmp) / "LOCAL_CLI.md").write_text("l", encoding="utf-8")
            self.assertEqual(find_instruction_file(tmp).name, "LOCAL_CLI.md")

    def test_walks_up_to_git_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".git").mkdir()
            (root / "AGENTS.md").write_text("root rules", encoding="utf-8")
            sub = root / "src" / "pkg"
            sub.mkdir(parents=True)
            found = find_instruction_file(str(sub))
            self.assertIsNotNone(found)
            self.assertEqual(found.read_text(encoding="utf-8"), "root rules")

    def test_nearest_wins_over_ancestor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".git").mkdir()
            (root / "AGENTS.md").write_text("root", encoding="utf-8")
            sub = root / "sub"
            sub.mkdir()
            (sub / "LOCAL_CLI.md").write_text("near", encoding="utf-8")
            found = find_instruction_file(str(sub))
            self.assertEqual(found.read_text(encoding="utf-8"), "near")

    def test_does_not_climb_past_git_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            outer = Path(tmp)
            (outer / "AGENTS.md").write_text("outer", encoding="utf-8")
            repo = outer / "repo"
            repo.mkdir()
            (repo / ".git").mkdir()
            self.assertIsNone(find_instruction_file(str(repo)))

    def test_home_dir_itself_never_checked(self) -> None:
        """A ~/CLAUDE.md meant for other tools must not leak in."""
        with tempfile.TemporaryDirectory() as tmp:
            fake_home = Path(tmp).resolve()
            (fake_home / "CLAUDE.md").write_text(
                "ssh secrets", encoding="utf-8",
            )
            sub = fake_home / "docs" / "proj"
            sub.mkdir(parents=True)
            with patch(
                "local_cli.project_instructions.Path.home",
                return_value=fake_home,
            ):
                self.assertIsNone(find_instruction_file(str(sub)))
                # A file in the project itself still wins.
                (sub / "AGENTS.md").write_text("mine", encoding="utf-8")
                found = find_instruction_file(str(sub))
                self.assertEqual(found.name, "AGENTS.md")


class TestLoadProjectInstructions(unittest.TestCase):
    def test_returns_source_and_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / ".git").mkdir()
            (Path(tmp) / "AGENTS.md").write_text(
                "常に日本語で回答する", encoding="utf-8",
            )
            source, content = load_project_instructions(tmp)
            self.assertEqual(source, "AGENTS.md")
            self.assertEqual(content, "常に日本語で回答する")

    def test_empty_file_counts_as_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / ".git").mkdir()
            (Path(tmp) / "AGENTS.md").write_text("  \n", encoding="utf-8")
            self.assertIsNone(load_project_instructions(tmp))

    def test_overlong_content_clipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / ".git").mkdir()
            (Path(tmp) / "AGENTS.md").write_text(
                "x" * (_MAX_CHARS + 500), encoding="utf-8",
            )
            _, content = load_project_instructions(tmp)
            self.assertLess(len(content), _MAX_CHARS + 100)
            self.assertIn("[instructions truncated]", content)

    def test_env_kill_switch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / ".git").mkdir()
            (Path(tmp) / "AGENTS.md").write_text("rules", encoding="utf-8")
            with patch.dict(
                os.environ, {"LOCAL_CLI_PROJECT_INSTRUCTIONS": "0"},
            ):
                self.assertFalse(project_instructions_enabled())
                self.assertIsNone(load_project_instructions(tmp))

    def test_env_default_on(self) -> None:
        with patch.dict(os.environ):
            os.environ.pop("LOCAL_CLI_PROJECT_INSTRUCTIONS", None)
            self.assertTrue(project_instructions_enabled())


class TestInstructionMessage(unittest.TestCase):
    def test_message_shape(self) -> None:
        msg = build_instruction_message("AGENTS.md", "use tabs")
        self.assertEqual(msg["role"], "system")
        self.assertIn("PROJECT INSTRUCTIONS (AGENTS.md)", msg["content"])
        self.assertIn("use tabs", msg["content"])
        self.assertIn("END PROJECT INSTRUCTIONS", msg["content"])

    def test_one_call_helper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / ".git").mkdir()
            (Path(tmp) / "LOCAL_CLI.md").write_text("rule#1", encoding="utf-8")
            msg = project_instruction_message(tmp)
            self.assertEqual(msg["role"], "system")
            self.assertIn("rule#1", msg["content"])

    def test_one_call_helper_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / ".git").mkdir()
            self.assertIsNone(project_instruction_message(tmp))


if __name__ == "__main__":
    unittest.main()
