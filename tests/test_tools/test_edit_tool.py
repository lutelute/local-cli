"""Tests for local_cli.tools.edit_tool module."""

import os
import tempfile
import unittest
from pathlib import Path

from local_cli.tools.edit_tool import EditTool


class TestEditToolMetadata(unittest.TestCase):
    """Tests for EditTool metadata properties."""

    def setUp(self) -> None:
        self.tool = EditTool()

    def test_name(self) -> None:
        """Tool name is 'edit'."""
        self.assertEqual(self.tool.name, "edit")

    def test_description_is_nonempty(self) -> None:
        """Description is a non-empty string."""
        self.assertIsInstance(self.tool.description, str)
        self.assertTrue(len(self.tool.description) > 0)

    def test_parameters_schema(self) -> None:
        """Parameters schema defines required fields."""
        params = self.tool.parameters
        self.assertEqual(params["type"], "object")
        self.assertIn("file_path", params["properties"])
        self.assertIn("old_text", params["properties"])
        self.assertIn("new_text", params["properties"])
        self.assertIn("file_path", params["required"])
        self.assertIn("old_text", params["required"])
        self.assertIn("new_text", params["required"])

    def test_to_ollama_tool(self) -> None:
        """to_ollama_tool returns correct function-calling format."""
        tool_def = self.tool.to_ollama_tool()
        self.assertEqual(tool_def["type"], "function")
        self.assertEqual(tool_def["function"]["name"], "edit")


class TestEditToolReplacement(unittest.TestCase):
    """Tests for text replacement."""

    def setUp(self) -> None:
        self.tool = EditTool()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_file(self, name: str, content: str) -> str:
        """Create a temp file with the given content and return its path."""
        path = os.path.join(self.tmpdir, name)
        Path(path).write_text(content, encoding="utf-8")
        return path

    def test_simple_replacement(self) -> None:
        """Replace a single occurrence of text."""
        path = self._create_file("test.txt", "hello world\n")
        result = self.tool.execute(
            file_path=path, old_text="hello", new_text="goodbye"
        )
        content = Path(path).read_text(encoding="utf-8")
        self.assertEqual(content, "goodbye world\n")
        # Result should show diff-like output.
        self.assertIn("-hello", result)
        self.assertIn("+goodbye", result)

    def test_multiline_replacement(self) -> None:
        """Replace multiline text."""
        original = "line1\nline2\nline3\n"
        path = self._create_file("multi.txt", original)
        result = self.tool.execute(
            file_path=path, old_text="line1\nline2", new_text="replaced"
        )
        content = Path(path).read_text(encoding="utf-8")
        self.assertEqual(content, "replaced\nline3\n")

    def test_replace_with_empty_string(self) -> None:
        """Replace text with an empty string (deletion)."""
        path = self._create_file("delete.txt", "keep this remove this end\n")
        result = self.tool.execute(
            file_path=path, old_text="remove this ", new_text=""
        )
        content = Path(path).read_text(encoding="utf-8")
        self.assertEqual(content, "keep this end\n")

    def test_default_replaces_only_first_occurrence(self) -> None:
        """By default, only the first occurrence is replaced."""
        path = self._create_file("first.txt", "aaa bbb aaa bbb\n")
        self.tool.execute(
            file_path=path, old_text="aaa", new_text="XXX"
        )
        content = Path(path).read_text(encoding="utf-8")
        self.assertEqual(content, "XXX bbb aaa bbb\n")

    def test_diff_output_shows_occurrence_count(self) -> None:
        """Diff output includes the number of replacements."""
        path = self._create_file("count.txt", "abc def\n")
        result = self.tool.execute(
            file_path=path, old_text="abc", new_text="xyz"
        )
        self.assertIn("1 occurrence", result)


class TestEditToolNoMatch(unittest.TestCase):
    """Tests for no-match error handling."""

    def setUp(self) -> None:
        self.tool = EditTool()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_file(self, name: str, content: str) -> str:
        path = os.path.join(self.tmpdir, name)
        Path(path).write_text(content, encoding="utf-8")
        return path

    def test_no_match_returns_error(self) -> None:
        """Attempting to replace text that doesn't exist returns an error."""
        path = self._create_file("nomatch.txt", "hello world\n")
        result = self.tool.execute(
            file_path=path, old_text="nonexistent", new_text="replacement"
        )
        self.assertIn("Error", result)
        self.assertIn("not found", result)

    def test_file_unchanged_on_no_match(self) -> None:
        """File content is not modified when old_text is not found."""
        original = "original content\n"
        path = self._create_file("unchanged.txt", original)
        self.tool.execute(
            file_path=path, old_text="missing", new_text="new"
        )
        content = Path(path).read_text(encoding="utf-8")
        self.assertEqual(content, original)


class TestEditToolMultipleMatches(unittest.TestCase):
    """Tests for multiple match handling with replace_all."""

    def setUp(self) -> None:
        self.tool = EditTool()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_file(self, name: str, content: str) -> str:
        path = os.path.join(self.tmpdir, name)
        Path(path).write_text(content, encoding="utf-8")
        return path

    def test_replace_all_replaces_every_occurrence(self) -> None:
        """replace_all=True replaces every occurrence."""
        path = self._create_file("all.txt", "foo bar foo baz foo\n")
        result = self.tool.execute(
            file_path=path,
            old_text="foo",
            new_text="XXX",
            replace_all=True,
        )
        content = Path(path).read_text(encoding="utf-8")
        self.assertEqual(content, "XXX bar XXX baz XXX\n")
        self.assertIn("3 occurrence", result)

    def test_replace_all_false_replaces_only_first(self) -> None:
        """replace_all=False explicitly replaces only the first occurrence."""
        path = self._create_file("first_only.txt", "aaa bbb aaa\n")
        self.tool.execute(
            file_path=path,
            old_text="aaa",
            new_text="ZZZ",
            replace_all=False,
        )
        content = Path(path).read_text(encoding="utf-8")
        self.assertEqual(content, "ZZZ bbb aaa\n")

    def test_replace_all_with_single_occurrence(self) -> None:
        """replace_all=True with only one occurrence works correctly."""
        path = self._create_file("single_all.txt", "unique text here\n")
        result = self.tool.execute(
            file_path=path,
            old_text="unique",
            new_text="common",
            replace_all=True,
        )
        content = Path(path).read_text(encoding="utf-8")
        self.assertEqual(content, "common text here\n")
        self.assertIn("1 occurrence", result)


class TestEditToolMissingFile(unittest.TestCase):
    """Tests for missing file handling."""

    def setUp(self) -> None:
        self.tool = EditTool()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_missing_file_returns_error(self) -> None:
        """Editing a non-existent file returns an error."""
        path = os.path.join(self.tmpdir, "nonexistent.txt")
        result = self.tool.execute(
            file_path=path, old_text="old", new_text="new"
        )
        self.assertIn("Error", result)
        self.assertIn("not found", result)

    def test_directory_returns_error(self) -> None:
        """Editing a directory path returns an error."""
        result = self.tool.execute(
            file_path=self.tmpdir, old_text="old", new_text="new"
        )
        self.assertIn("Error", result)
        self.assertIn("not a regular file", result)


class TestEditToolParameterValidation(unittest.TestCase):
    """Tests for parameter validation."""

    def setUp(self) -> None:
        self.tool = EditTool()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_file(self, name: str, content: str) -> str:
        path = os.path.join(self.tmpdir, name)
        Path(path).write_text(content, encoding="utf-8")
        return path

    def test_missing_file_path_returns_error(self) -> None:
        """Missing file_path parameter returns an error."""
        result = self.tool.execute(old_text="old", new_text="new")
        self.assertIn("Error", result)

    def test_empty_file_path_returns_error(self) -> None:
        """Empty file_path string returns an error."""
        result = self.tool.execute(file_path="", old_text="old", new_text="new")
        self.assertIn("Error", result)

    def test_missing_old_text_returns_error(self) -> None:
        """Missing old_text parameter returns an error."""
        path = self._create_file("test.txt", "content")
        result = self.tool.execute(file_path=path, new_text="new")
        self.assertIn("Error", result)

    def test_empty_old_text_returns_error(self) -> None:
        """Empty old_text string returns an error."""
        path = self._create_file("test.txt", "content")
        result = self.tool.execute(file_path=path, old_text="", new_text="new")
        self.assertIn("Error", result)
        self.assertIn("empty", result.lower())

    def test_missing_new_text_returns_error(self) -> None:
        """Missing new_text parameter returns an error."""
        path = self._create_file("test.txt", "content")
        result = self.tool.execute(file_path=path, old_text="old")
        self.assertIn("Error", result)

    def test_invalid_replace_all_uses_default(self) -> None:
        """Non-boolean replace_all falls back to False (default)."""
        path = self._create_file("invalid_flag.txt", "abc abc abc\n")
        self.tool.execute(
            file_path=path,
            old_text="abc",
            new_text="xyz",
            replace_all="yes",  # Not a bool.
        )
        content = Path(path).read_text(encoding="utf-8")
        # Only the first occurrence should be replaced (default=False).
        self.assertEqual(content, "xyz abc abc\n")


class TestEditToolLineEndingFallback(unittest.TestCase):
    """Tests for the CRLF/LF normalization fallback."""

    def test_lf_old_text_matches_crlf_file(self) -> None:
        """old_text emitted with LF matches a file stored with CRLF."""
        fd, path = tempfile.mkstemp(suffix=".txt")
        os.close(fd)
        try:
            with open(path, "w", newline="") as f:
                f.write("line one\r\nline two\r\nline three\r\n")
            result = EditTool().execute(
                file_path=path,
                old_text="line one\nline two",
                new_text="line one\nLINE TWO",
            )
            self.assertNotIn("not found", result)
            with open(path, encoding="utf-8") as f:
                self.assertIn("LINE TWO", f.read())
        finally:
            os.unlink(path)

    def test_genuinely_missing_text_still_errors(self) -> None:
        """Text absent even after normalization still returns not-found."""
        fd, path = tempfile.mkstemp(suffix=".txt")
        os.close(fd)
        try:
            with open(path, "w") as f:
                f.write("hello world\n")
            result = EditTool().execute(
                file_path=path, old_text="nonexistent xyz", new_text="q",
            )
            self.assertIn("not found", result)
        finally:
            os.unlink(path)


class TestEditToolAtomicWrite(unittest.TestCase):
    """Tests for atomic-write behavior (permission preservation)."""

    def test_preserves_file_mode(self) -> None:
        """An edit preserves the file's permission bits."""
        import stat as _stat
        fd, path = tempfile.mkstemp(suffix=".sh")
        os.close(fd)
        try:
            with open(path, "w") as f:
                f.write("echo hi\n")
            os.chmod(path, 0o755)
            EditTool().execute(file_path=path, old_text="hi", new_text="bye")
            mode = _stat.S_IMODE(os.stat(path).st_mode)
            self.assertEqual(mode, 0o755)
            with open(path, encoding="utf-8") as f:
                self.assertIn("bye", f.read())
        finally:
            os.unlink(path)


class TestEditToolClosestMatchHint(unittest.TestCase):
    """Tests for the closest-match hint on old_text-not-found errors."""

    def setUp(self) -> None:
        self.tool = EditTool()
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write(self, text: str) -> str:
        path = self.dir / "code.py"
        path.write_text(text, encoding="utf-8")
        return str(path)

    def test_wrong_indentation_gets_hint_with_actual_text(self) -> None:
        """A near-miss (wrong indentation) shows the real block."""
        path = self._write(
            "class Foo:\n"
            "    def bar(self):\n"
            "        return 42\n"
        )
        # The model guessed the body without the class indentation.
        result = self.tool.execute(
            file_path=path,
            old_text="def bar(self):\n    return 42",
            new_text="def bar(self):\n    return 43",
        )
        self.assertIn("not found", result)
        self.assertIn("most similar text", result)
        self.assertIn("    def bar(self):", result)
        self.assertIn("line 2", result)
        self.assertIn("EXACTLY", result)

    def test_typo_in_old_text_gets_hint(self) -> None:
        path = self._write("value = compute_total(items)\n")
        result = self.tool.execute(
            file_path=path,
            old_text="value = computeTotal(items)",
            new_text="value = compute_sum(items)",
        )
        self.assertIn("not found", result)
        self.assertIn("compute_total(items)", result)

    def test_completely_unrelated_text_gets_no_hint(self) -> None:
        """Below the similarity floor, only the plain error is returned."""
        path = self._write("alpha beta gamma\n")
        result = self.tool.execute(
            file_path=path,
            old_text="zzzzzz qqqqqq wwwwww",
            new_text="x",
        )
        self.assertIn("not found", result)
        self.assertNotIn("most similar", result)

    def test_file_not_modified_when_hint_shown(self) -> None:
        original = "def f():\n    return 1\n"
        path = self._write(original)
        self.tool.execute(
            file_path=path,
            old_text="def f():\n  return 1",  # wrong indent
            new_text="def f():\n  return 2",
        )
        self.assertEqual(Path(path).read_text(encoding="utf-8"), original)

    def test_huge_old_text_skips_hint(self) -> None:
        path = self._write("short file\n")
        result = self.tool.execute(
            file_path=path,
            old_text="x" * 5_000,
            new_text="y",
        )
        self.assertIn("not found", result)
        self.assertNotIn("most similar", result)

    def test_multiline_hint_reports_start_line(self) -> None:
        lines = [f"line_{i} = {i}" for i in range(1, 21)]
        path = self._write("\n".join(lines) + "\n")
        result = self.tool.execute(
            file_path=path,
            old_text="line_10 = 10\nline_11 = eleven",  # second line wrong
            new_text="replacement",
        )
        self.assertIn("not found", result)
        self.assertIn("line 10", result)
        self.assertIn("line_10 = 10", result)


if __name__ == "__main__":
    unittest.main()
