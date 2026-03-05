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


if __name__ == "__main__":
    unittest.main()
