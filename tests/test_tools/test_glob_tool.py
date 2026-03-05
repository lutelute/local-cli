"""Tests for local_cli.tools.glob_tool module."""

import os
import tempfile
import time
import unittest
from pathlib import Path

from local_cli.tools.glob_tool import GlobTool


class TestGlobToolMetadata(unittest.TestCase):
    """Tests for GlobTool metadata properties."""

    def setUp(self) -> None:
        self.tool = GlobTool()

    def test_name(self) -> None:
        """Tool name is 'glob'."""
        self.assertEqual(self.tool.name, "glob")

    def test_description_is_nonempty(self) -> None:
        """Description is a non-empty string."""
        self.assertIsInstance(self.tool.description, str)
        self.assertTrue(len(self.tool.description) > 0)

    def test_parameters_schema(self) -> None:
        """Parameters schema defines 'pattern' as required."""
        params = self.tool.parameters
        self.assertEqual(params["type"], "object")
        self.assertIn("pattern", params["properties"])
        self.assertIn("pattern", params["required"])

    def test_to_ollama_tool(self) -> None:
        """to_ollama_tool returns correct function-calling format."""
        tool_def = self.tool.to_ollama_tool()
        self.assertEqual(tool_def["type"], "function")
        self.assertEqual(tool_def["function"]["name"], "glob")


class TestGlobToolPatternMatching(unittest.TestCase):
    """Tests for glob pattern matching."""

    def setUp(self) -> None:
        self.tool = GlobTool()
        self.tmpdir = tempfile.mkdtemp()
        # Create a set of test files.
        self._create_file("file1.py", "python file 1")
        self._create_file("file2.py", "python file 2")
        self._create_file("file3.txt", "text file")
        self._create_file("README.md", "readme")

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_file(self, name: str, content: str) -> str:
        path = os.path.join(self.tmpdir, name)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(content, encoding="utf-8")
        return path

    def test_match_py_files(self) -> None:
        """Match *.py files in a directory."""
        result = self.tool.execute(pattern="*.py", path=self.tmpdir)
        self.assertIn("file1.py", result)
        self.assertIn("file2.py", result)
        self.assertNotIn("file3.txt", result)
        self.assertNotIn("README.md", result)

    def test_match_txt_files(self) -> None:
        """Match *.txt files in a directory."""
        result = self.tool.execute(pattern="*.txt", path=self.tmpdir)
        self.assertIn("file3.txt", result)
        self.assertNotIn("file1.py", result)

    def test_match_md_files(self) -> None:
        """Match *.md files in a directory."""
        result = self.tool.execute(pattern="*.md", path=self.tmpdir)
        self.assertIn("README.md", result)

    def test_match_all_files(self) -> None:
        """Match all files with *."""
        result = self.tool.execute(pattern="*", path=self.tmpdir)
        self.assertIn("file1.py", result)
        self.assertIn("file2.py", result)
        self.assertIn("file3.txt", result)
        self.assertIn("README.md", result)

    def test_no_matches_returns_message(self) -> None:
        """No matches returns a descriptive message."""
        result = self.tool.execute(pattern="*.xyz", path=self.tmpdir)
        self.assertIn("No files matched", result)

    def test_recursive_pattern(self) -> None:
        """Recursive ** pattern matches files in subdirectories."""
        self._create_file("sub/deep/nested.py", "nested python")
        result = self.tool.execute(pattern="**/*.py", path=self.tmpdir)
        self.assertIn("nested.py", result)
        self.assertIn("file1.py", result)


class TestGlobToolDirectoryFiltering(unittest.TestCase):
    """Tests for directory filtering."""

    def setUp(self) -> None:
        self.tool = GlobTool()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_custom_path(self) -> None:
        """Search in a custom directory path."""
        subdir = os.path.join(self.tmpdir, "custom_dir")
        os.makedirs(subdir)
        Path(os.path.join(subdir, "target.py")).write_text(
            "found", encoding="utf-8"
        )
        Path(os.path.join(self.tmpdir, "outside.py")).write_text(
            "not found", encoding="utf-8"
        )

        result = self.tool.execute(pattern="*.py", path=subdir)
        self.assertIn("target.py", result)
        self.assertNotIn("outside.py", result)

    def test_nonexistent_directory_returns_error(self) -> None:
        """Searching a non-existent directory returns an error."""
        fake_path = os.path.join(self.tmpdir, "nonexistent")
        result = self.tool.execute(pattern="*.py", path=fake_path)
        self.assertIn("Error", result)
        self.assertIn("not found", result)

    def test_file_path_as_directory_returns_error(self) -> None:
        """Using a file path as the search directory returns an error."""
        file_path = os.path.join(self.tmpdir, "file.txt")
        Path(file_path).write_text("content", encoding="utf-8")
        result = self.tool.execute(pattern="*.py", path=file_path)
        self.assertIn("Error", result)
        self.assertIn("not a directory", result)

    def test_results_sorted_by_mtime(self) -> None:
        """Results are sorted by modification time (most recent first)."""
        # Create files with staggered modification times.
        path_old = os.path.join(self.tmpdir, "old.py")
        Path(path_old).write_text("old", encoding="utf-8")
        # Set mtime in the past.
        os.utime(path_old, (1000000, 1000000))

        path_new = os.path.join(self.tmpdir, "new.py")
        Path(path_new).write_text("new", encoding="utf-8")
        # This file has the current mtime so it's newer.

        result = self.tool.execute(pattern="*.py", path=self.tmpdir)
        lines = result.strip().split("\n")
        self.assertEqual(len(lines), 2)
        # Most recent file should appear first.
        self.assertIn("new.py", lines[0])
        self.assertIn("old.py", lines[1])


class TestGlobToolParameterValidation(unittest.TestCase):
    """Tests for parameter validation."""

    def setUp(self) -> None:
        self.tool = GlobTool()

    def test_missing_pattern_returns_error(self) -> None:
        """Missing pattern parameter returns an error."""
        result = self.tool.execute()
        self.assertIn("Error", result)

    def test_empty_pattern_returns_error(self) -> None:
        """Empty pattern string returns an error."""
        result = self.tool.execute(pattern="")
        self.assertIn("Error", result)

    def test_whitespace_pattern_returns_error(self) -> None:
        """Whitespace-only pattern returns an error."""
        result = self.tool.execute(pattern="   ")
        self.assertIn("Error", result)

    def test_non_string_pattern_returns_error(self) -> None:
        """Non-string pattern returns an error."""
        result = self.tool.execute(pattern=123)
        self.assertIn("Error", result)

    def test_default_path_uses_cwd(self) -> None:
        """When path is not provided, current working directory is used."""
        # This should not return an error (cwd always exists).
        result = self.tool.execute(pattern="*.nonexistent_extension_xyz")
        # Should get no matches, not a directory error.
        self.assertIn("No files matched", result)


if __name__ == "__main__":
    unittest.main()
