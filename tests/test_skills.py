"""Tests for local_cli.skills module."""

import os
import tempfile
import unittest
from pathlib import Path

from local_cli.skills import (
    Skill,
    SkillNotFoundError,
    SkillParseError,
    SkillsError,
    SkillsLoader,
)


def _write_skill(skills_dir: str, name: str, content: str) -> Path:
    """Helper to write a SKILL.md file for testing.

    Args:
        skills_dir: Base skills directory.
        name: Skill subdirectory name.
        content: Full content of the SKILL.md file.

    Returns:
        Path to the created SKILL.md file.
    """
    skill_dir = Path(skills_dir) / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(content, encoding="utf-8")
    return skill_file


class TestSkillsLoaderInit(unittest.TestCase):
    """Tests for SkillsLoader construction."""

    def test_stores_skills_dir_as_path(self) -> None:
        """skills_dir argument is stored as a Path object."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = SkillsLoader(tmpdir)
            self.assertIsInstance(loader._skills_dir, Path)

    def test_tilde_expansion(self) -> None:
        """Skills dir path with ~ is expanded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = SkillsLoader(tmpdir)
            self.assertTrue(loader._skills_dir.is_absolute())

    def test_default_skills_dir(self) -> None:
        """Default skills_dir is '.agents/skills'."""
        loader = SkillsLoader()
        self.assertEqual(loader._skills_dir.name, "skills")

    def test_starts_with_empty_skills(self) -> None:
        """Loader starts with no skills loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = SkillsLoader(tmpdir)
            self.assertEqual(loader.list_skills(), [])


class TestDiscoverSkills(unittest.TestCase):
    """Tests for SkillsLoader.discover_skills()."""

    def test_discovers_valid_skill(self) -> None:
        """discover_skills finds and parses a valid SKILL.md file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_skill(
                tmpdir,
                "my-skill",
                "---\n"
                "name: my-skill\n"
                "triggers: [python, code]\n"
                "description: A test skill\n"
                "---\n"
                "\n"
                "# Skill Content\n"
                "\n"
                "Use this for Python code generation.\n",
            )
            loader = SkillsLoader(tmpdir)
            skills = loader.discover_skills()

            self.assertEqual(len(skills), 1)
            self.assertEqual(skills[0].name, "my-skill")
            self.assertEqual(skills[0].triggers, ["python", "code"])
            self.assertEqual(skills[0].description, "A test skill")
            self.assertIn("Skill Content", skills[0].content)

    def test_discovers_multiple_skills(self) -> None:
        """discover_skills finds all valid skills in subdirectories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_skill(
                tmpdir,
                "skill-a",
                "---\nname: skill-a\ntriggers: [alpha]\ndescription: Alpha\n---\nContent A",
            )
            _write_skill(
                tmpdir,
                "skill-b",
                "---\nname: skill-b\ntriggers: [beta]\ndescription: Beta\n---\nContent B",
            )
            loader = SkillsLoader(tmpdir)
            skills = loader.discover_skills()

            self.assertEqual(len(skills), 2)
            names = [s.name for s in skills]
            self.assertIn("skill-a", names)
            self.assertIn("skill-b", names)

    def test_sorted_by_name(self) -> None:
        """Discovered skills are sorted alphabetically by name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_skill(
                tmpdir,
                "zebra",
                "---\nname: zebra\ntriggers: [z]\ndescription: Z\n---\nZ",
            )
            _write_skill(
                tmpdir,
                "apple",
                "---\nname: apple\ntriggers: [a]\ndescription: A\n---\nA",
            )
            loader = SkillsLoader(tmpdir)
            skills = loader.discover_skills()

            names = [s.name for s in skills]
            self.assertEqual(names, ["apple", "zebra"])

    def test_missing_directory_returns_empty(self) -> None:
        """Non-existent skills directory returns empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            skills_dir = os.path.join(tmpdir, "nonexistent")
            loader = SkillsLoader(skills_dir)
            skills = loader.discover_skills()
            self.assertEqual(skills, [])

    def test_empty_directory_returns_empty(self) -> None:
        """Empty skills directory returns empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = SkillsLoader(tmpdir)
            skills = loader.discover_skills()
            self.assertEqual(skills, [])

    def test_directory_without_skill_file_skipped(self) -> None:
        """Subdirectories without SKILL.md are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_skill(
                tmpdir,
                "valid",
                "---\nname: valid\ntriggers: [test]\ndescription: Test\n---\nContent",
            )
            # Create directory without SKILL.md.
            empty_dir = Path(tmpdir) / "no-skill"
            empty_dir.mkdir()

            loader = SkillsLoader(tmpdir)
            skills = loader.discover_skills()

            self.assertEqual(len(skills), 1)
            self.assertEqual(skills[0].name, "valid")

    def test_malformed_frontmatter_skipped(self) -> None:
        """Skills with malformed frontmatter are silently skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_skill(
                tmpdir,
                "valid",
                "---\nname: valid\ntriggers: [test]\ndescription: Test\n---\nContent",
            )
            # No frontmatter delimiters.
            _write_skill(
                tmpdir,
                "bad",
                "This has no frontmatter at all.\nJust content.",
            )
            loader = SkillsLoader(tmpdir)
            skills = loader.discover_skills()

            self.assertEqual(len(skills), 1)
            self.assertEqual(skills[0].name, "valid")

    def test_rediscover_clears_previous(self) -> None:
        """Calling discover_skills again clears previously loaded skills."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_skill(
                tmpdir,
                "first",
                "---\nname: first\ntriggers: [x]\ndescription: X\n---\nContent",
            )
            loader = SkillsLoader(tmpdir)
            loader.discover_skills()
            self.assertEqual(len(loader.list_skills()), 1)

            # Remove the first skill and add a different one.
            import shutil

            shutil.rmtree(str(Path(tmpdir) / "first"))
            _write_skill(
                tmpdir,
                "second",
                "---\nname: second\ntriggers: [y]\ndescription: Y\n---\nContent",
            )
            loader.discover_skills()
            self.assertEqual(len(loader.list_skills()), 1)
            self.assertEqual(loader.list_skills()[0].name, "second")

    def test_files_in_root_ignored(self) -> None:
        """Regular files in the skills root directory are ignored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_skill(
                tmpdir,
                "valid",
                "---\nname: valid\ntriggers: [test]\ndescription: Test\n---\nContent",
            )
            # Create a stray file in root.
            stray = Path(tmpdir) / "stray.md"
            stray.write_text("not a skill")

            loader = SkillsLoader(tmpdir)
            skills = loader.discover_skills()
            self.assertEqual(len(skills), 1)


class TestFrontmatterParsing(unittest.TestCase):
    """Tests for SKILL.md frontmatter parsing."""

    def test_bracketed_triggers(self) -> None:
        """Triggers in bracket format [a, b, c] are parsed correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_skill(
                tmpdir,
                "test",
                "---\nname: test\ntriggers: [python, code review, testing]\ndescription: Test\n---\nContent",
            )
            loader = SkillsLoader(tmpdir)
            skills = loader.discover_skills()

            self.assertEqual(
                skills[0].triggers, ["python", "code review", "testing"]
            )

    def test_bare_triggers(self) -> None:
        """Triggers without brackets are parsed correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_skill(
                tmpdir,
                "test",
                "---\nname: test\ntriggers: python, code review, testing\ndescription: Test\n---\nContent",
            )
            loader = SkillsLoader(tmpdir)
            skills = loader.discover_skills()

            self.assertEqual(
                skills[0].triggers, ["python", "code review", "testing"]
            )

    def test_single_trigger(self) -> None:
        """A single trigger value is parsed correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_skill(
                tmpdir,
                "test",
                "---\nname: test\ntriggers: [python]\ndescription: Test\n---\nContent",
            )
            loader = SkillsLoader(tmpdir)
            skills = loader.discover_skills()

            self.assertEqual(skills[0].triggers, ["python"])

    def test_empty_triggers(self) -> None:
        """Empty triggers value results in empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_skill(
                tmpdir,
                "test",
                "---\nname: test\ntriggers: \ndescription: Test\n---\nContent",
            )
            loader = SkillsLoader(tmpdir)
            skills = loader.discover_skills()

            self.assertEqual(skills[0].triggers, [])

    def test_empty_brackets_triggers(self) -> None:
        """Empty brackets [] results in empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_skill(
                tmpdir,
                "test",
                "---\nname: test\ntriggers: []\ndescription: Test\n---\nContent",
            )
            loader = SkillsLoader(tmpdir)
            skills = loader.discover_skills()

            self.assertEqual(skills[0].triggers, [])

    def test_name_from_frontmatter(self) -> None:
        """Name is taken from frontmatter when present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_skill(
                tmpdir,
                "dir-name",
                "---\nname: custom-name\ntriggers: [x]\ndescription: Test\n---\nContent",
            )
            loader = SkillsLoader(tmpdir)
            skills = loader.discover_skills()

            self.assertEqual(skills[0].name, "custom-name")

    def test_name_fallback_to_dir_name(self) -> None:
        """If name is missing in frontmatter, directory name is used."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_skill(
                tmpdir,
                "dir-name",
                "---\ntriggers: [x]\ndescription: Test\n---\nContent",
            )
            loader = SkillsLoader(tmpdir)
            skills = loader.discover_skills()

            self.assertEqual(skills[0].name, "dir-name")

    def test_description_parsed(self) -> None:
        """Description is extracted from frontmatter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_skill(
                tmpdir,
                "test",
                "---\nname: test\ntriggers: [x]\ndescription: A helpful skill\n---\nContent",
            )
            loader = SkillsLoader(tmpdir)
            skills = loader.discover_skills()

            self.assertEqual(skills[0].description, "A helpful skill")

    def test_body_content_extracted(self) -> None:
        """Body content after frontmatter is preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            body = "# Instructions\n\nUse this skill for Python code.\n\n## Details\n\nMore info here."
            _write_skill(
                tmpdir,
                "test",
                f"---\nname: test\ntriggers: [x]\ndescription: Test\n---\n\n{body}\n",
            )
            loader = SkillsLoader(tmpdir)
            skills = loader.discover_skills()

            self.assertIn("# Instructions", skills[0].content)
            self.assertIn("## Details", skills[0].content)

    def test_no_closing_delimiter_skipped(self) -> None:
        """Frontmatter without closing --- is treated as no frontmatter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_skill(
                tmpdir,
                "test",
                "---\nname: test\ntriggers: [x]\nSome content without closing delimiter",
            )
            loader = SkillsLoader(tmpdir)
            skills = loader.discover_skills()

            # Should be skipped (no valid frontmatter).
            self.assertEqual(len(skills), 0)

    def test_key_case_insensitive(self) -> None:
        """Frontmatter keys are case-insensitive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_skill(
                tmpdir,
                "test",
                "---\nName: test\nTriggers: [python]\nDescription: Test\n---\nContent",
            )
            loader = SkillsLoader(tmpdir)
            skills = loader.discover_skills()

            self.assertEqual(len(skills), 1)
            self.assertEqual(skills[0].name, "test")
            self.assertEqual(skills[0].triggers, ["python"])

    def test_colon_in_description_value(self) -> None:
        """Description with colons is parsed correctly (split on first colon)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_skill(
                tmpdir,
                "test",
                "---\nname: test\ntriggers: [x]\ndescription: Use for: Python, Go, Rust\n---\nContent",
            )
            loader = SkillsLoader(tmpdir)
            skills = loader.discover_skills()

            self.assertEqual(
                skills[0].description, "Use for: Python, Go, Rust"
            )

    def test_whitespace_in_frontmatter(self) -> None:
        """Leading whitespace in frontmatter is handled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_skill(
                tmpdir,
                "test",
                "---\n  name: test\n  triggers: [python]\n  description: Test\n---\nContent",
            )
            loader = SkillsLoader(tmpdir)
            skills = loader.discover_skills()

            self.assertEqual(len(skills), 1)
            self.assertEqual(skills[0].name, "test")


class TestGetMatchingSkills(unittest.TestCase):
    """Tests for SkillsLoader.get_matching_skills()."""

    def _setup_loader(self, tmpdir: str) -> SkillsLoader:
        """Create a loader with several test skills."""
        _write_skill(
            tmpdir,
            "python",
            "---\nname: python-skill\ntriggers: [python, django, flask]\ndescription: Python\n---\nPython content",
        )
        _write_skill(
            tmpdir,
            "rust",
            "---\nname: rust-skill\ntriggers: [rust, cargo]\ndescription: Rust\n---\nRust content",
        )
        _write_skill(
            tmpdir,
            "testing",
            "---\nname: testing-skill\ntriggers: [test, pytest, unittest]\ndescription: Testing\n---\nTest content",
        )
        loader = SkillsLoader(tmpdir)
        loader.discover_skills()
        return loader

    def test_exact_trigger_match(self) -> None:
        """Input containing an exact trigger keyword matches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = self._setup_loader(tmpdir)
            matched = loader.get_matching_skills("Help me with python")
            names = [s.name for s in matched]
            self.assertIn("python-skill", names)

    def test_partial_trigger_match(self) -> None:
        """Trigger substring within input matches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = self._setup_loader(tmpdir)
            matched = loader.get_matching_skills("I need a pytest fixture")
            names = [s.name for s in matched]
            self.assertIn("testing-skill", names)

    def test_case_insensitive_match(self) -> None:
        """Matching is case-insensitive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = self._setup_loader(tmpdir)
            matched = loader.get_matching_skills("PYTHON code review")
            names = [s.name for s in matched]
            self.assertIn("python-skill", names)

    def test_no_match(self) -> None:
        """Input that doesn't match any triggers returns empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = self._setup_loader(tmpdir)
            matched = loader.get_matching_skills("Help me with JavaScript")
            self.assertEqual(matched, [])

    def test_empty_input_returns_empty(self) -> None:
        """Empty input returns empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = self._setup_loader(tmpdir)
            matched = loader.get_matching_skills("")
            self.assertEqual(matched, [])

    def test_whitespace_only_input_returns_empty(self) -> None:
        """Whitespace-only input returns empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = self._setup_loader(tmpdir)
            matched = loader.get_matching_skills("   ")
            self.assertEqual(matched, [])

    def test_multiple_skills_match(self) -> None:
        """Multiple skills can match the same input."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = self._setup_loader(tmpdir)
            matched = loader.get_matching_skills("Write python test with pytest")
            names = [s.name for s in matched]
            self.assertIn("python-skill", names)
            self.assertIn("testing-skill", names)

    def test_matched_sorted_by_name(self) -> None:
        """Matched skills are sorted alphabetically by name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = self._setup_loader(tmpdir)
            matched = loader.get_matching_skills("python test pytest")
            names = [s.name for s in matched]
            self.assertEqual(names, sorted(names))

    def test_no_skills_loaded(self) -> None:
        """Returns empty when no skills have been discovered."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = SkillsLoader(tmpdir)
            matched = loader.get_matching_skills("python")
            self.assertEqual(matched, [])


class TestGetSkillContent(unittest.TestCase):
    """Tests for SkillsLoader.get_skill_content()."""

    def test_returns_skill_content(self) -> None:
        """get_skill_content returns the body content of a skill."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_skill(
                tmpdir,
                "my-skill",
                "---\nname: my-skill\ntriggers: [x]\ndescription: Test\n---\n\n# Instructions\n\nDo the thing.",
            )
            loader = SkillsLoader(tmpdir)
            loader.discover_skills()
            content = loader.get_skill_content("my-skill")
            self.assertIn("# Instructions", content)
            self.assertIn("Do the thing.", content)

    def test_nonexistent_skill_raises(self) -> None:
        """get_skill_content raises SkillNotFoundError for unknown skill."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = SkillsLoader(tmpdir)
            loader.discover_skills()
            with self.assertRaises(SkillNotFoundError):
                loader.get_skill_content("does-not-exist")


class TestListSkills(unittest.TestCase):
    """Tests for SkillsLoader.list_skills()."""

    def test_returns_all_skills(self) -> None:
        """list_skills returns all discovered skills."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_skill(
                tmpdir,
                "a",
                "---\nname: a\ntriggers: [x]\ndescription: A\n---\nContent A",
            )
            _write_skill(
                tmpdir,
                "b",
                "---\nname: b\ntriggers: [y]\ndescription: B\n---\nContent B",
            )
            loader = SkillsLoader(tmpdir)
            loader.discover_skills()
            skills = loader.list_skills()

            self.assertEqual(len(skills), 2)
            names = [s.name for s in skills]
            self.assertEqual(names, ["a", "b"])

    def test_empty_before_discover(self) -> None:
        """list_skills returns empty before discover_skills is called."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = SkillsLoader(tmpdir)
            self.assertEqual(loader.list_skills(), [])

    def test_sorted_by_name(self) -> None:
        """Skills are sorted alphabetically by name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_skill(
                tmpdir,
                "z",
                "---\nname: z\ntriggers: [x]\ndescription: Z\n---\nZ",
            )
            _write_skill(
                tmpdir,
                "a",
                "---\nname: a\ntriggers: [y]\ndescription: A\n---\nA",
            )
            loader = SkillsLoader(tmpdir)
            loader.discover_skills()
            skills = loader.list_skills()

            names = [s.name for s in skills]
            self.assertEqual(names, ["a", "z"])


class TestSkillDataContainer(unittest.TestCase):
    """Tests for the Skill data container."""

    def test_slots_defined(self) -> None:
        """Skill uses __slots__ for memory efficiency."""
        self.assertTrue(hasattr(Skill, "__slots__"))

    def test_all_attributes_accessible(self) -> None:
        """All Skill attributes are accessible."""
        skill = Skill(
            name="test",
            triggers=["a", "b"],
            description="desc",
            content="body",
            path=Path("/tmp/test"),
        )
        self.assertEqual(skill.name, "test")
        self.assertEqual(skill.triggers, ["a", "b"])
        self.assertEqual(skill.description, "desc")
        self.assertEqual(skill.content, "body")
        self.assertEqual(skill.path, Path("/tmp/test"))


class TestExceptionHierarchy(unittest.TestCase):
    """Tests for the skills exception hierarchy."""

    def test_skill_not_found_is_skills_error(self) -> None:
        """SkillNotFoundError is a subclass of SkillsError."""
        self.assertTrue(issubclass(SkillNotFoundError, SkillsError))

    def test_skill_parse_error_is_skills_error(self) -> None:
        """SkillParseError is a subclass of SkillsError."""
        self.assertTrue(issubclass(SkillParseError, SkillsError))

    def test_skills_error_is_exception(self) -> None:
        """SkillsError is a subclass of Exception."""
        self.assertTrue(issubclass(SkillsError, Exception))

    def test_catch_base_catches_not_found(self) -> None:
        """Catching SkillsError catches SkillNotFoundError."""
        with self.assertRaises(SkillsError):
            raise SkillNotFoundError("test")

    def test_catch_base_catches_parse_error(self) -> None:
        """Catching SkillsError catches SkillParseError."""
        with self.assertRaises(SkillsError):
            raise SkillParseError("test")


class TestSkillsUnicode(unittest.TestCase):
    """Tests for Unicode handling in skills."""

    def test_unicode_content_preserved(self) -> None:
        """Unicode content in SKILL.md is preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_skill(
                tmpdir,
                "unicode",
                "---\nname: unicode\ntriggers: [日本語]\ndescription: Unicode skill\n---\n\n# 日本語スキル\n\n使い方の説明 🎉",
            )
            loader = SkillsLoader(tmpdir)
            skills = loader.discover_skills()

            self.assertEqual(len(skills), 1)
            self.assertEqual(skills[0].triggers, ["日本語"])
            self.assertIn("日本語スキル", skills[0].content)

    def test_unicode_trigger_matching(self) -> None:
        """Unicode triggers are matched case-insensitively."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_skill(
                tmpdir,
                "unicode",
                "---\nname: unicode\ntriggers: [日本語]\ndescription: Unicode skill\n---\nContent",
            )
            loader = SkillsLoader(tmpdir)
            loader.discover_skills()
            matched = loader.get_matching_skills("日本語で書いてください")
            self.assertEqual(len(matched), 1)


if __name__ == "__main__":
    unittest.main()
