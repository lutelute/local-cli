"""Tests for local_cli.tools.write_tool module."""

import os
import tempfile
import unittest
from pathlib import Path

from local_cli.tools.write_tool import WriteTool


class TestWriteToolMetadata(unittest.TestCase):
    """Tests for WriteTool metadata properties."""

    def setUp(self) -> None:
        self.tool = WriteTool()

    def test_name(self) -> None:
        """Tool name is 'write'."""
        self.assertEqual(self.tool.name, "write")

    def test_description_is_nonempty(self) -> None:
        """Description is a non-empty string."""
        self.assertIsInstance(self.tool.description, str)
        self.assertTrue(len(self.tool.description) > 0)

    def test_parameters_schema(self) -> None:
        """Parameters schema defines 'file_path' and 'content' as required."""
        params = self.tool.parameters
        self.assertEqual(params["type"], "object")
        self.assertIn("file_path", params["properties"])
        self.assertIn("content", params["properties"])
        self.assertIn("file_path", params["required"])
        self.assertIn("content", params["required"])

    def test_to_ollama_tool(self) -> None:
        """to_ollama_tool returns correct function-calling format."""
        tool_def = self.tool.to_ollama_tool()
        self.assertEqual(tool_def["type"], "function")
        self.assertEqual(tool_def["function"]["name"], "write")


class TestWriteToolFileCreation(unittest.TestCase):
    """Tests for file creation."""

    def setUp(self) -> None:
        self.tool = WriteTool()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_create_new_file(self) -> None:
        """Create a new file with content."""
        path = os.path.join(self.tmpdir, "new_file.txt")
        result = self.tool.execute(file_path=path, content="hello world")
        self.assertIn("Successfully wrote", result)
        self.assertTrue(Path(path).exists())
        self.assertEqual(Path(path).read_text(encoding="utf-8"), "hello world")

    def test_create_file_with_multiline_content(self) -> None:
        """Create a file with multiline content."""
        path = os.path.join(self.tmpdir, "multi.txt")
        content = "line 1\nline 2\nline 3\n"
        result = self.tool.execute(file_path=path, content=content)
        self.assertIn("Successfully wrote", result)
        self.assertEqual(Path(path).read_text(encoding="utf-8"), content)

    def test_create_empty_file(self) -> None:
        """Create a file with empty content."""
        path = os.path.join(self.tmpdir, "empty.txt")
        result = self.tool.execute(file_path=path, content="")
        self.assertIn("Successfully wrote", result)
        self.assertTrue(Path(path).exists())
        self.assertEqual(Path(path).read_text(encoding="utf-8"), "")

    def test_result_reports_bytes_and_lines(self) -> None:
        """Result message includes byte count and line count."""
        path = os.path.join(self.tmpdir, "stats.txt")
        content = "abc\ndef\n"
        result = self.tool.execute(file_path=path, content=content)
        self.assertIn("8 bytes", result)  # 8 bytes in "abc\ndef\n"
        self.assertIn("2 lines", result)


class TestWriteToolParentDirectoryCreation(unittest.TestCase):
    """Tests for automatic parent directory creation."""

    def setUp(self) -> None:
        self.tool = WriteTool()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_creates_parent_directories(self) -> None:
        """Parent directories are created automatically."""
        path = os.path.join(self.tmpdir, "a", "b", "c", "deep_file.txt")
        result = self.tool.execute(file_path=path, content="deep content")
        self.assertIn("Successfully wrote", result)
        self.assertTrue(Path(path).exists())
        self.assertEqual(
            Path(path).read_text(encoding="utf-8"), "deep content"
        )

    def test_existing_parent_directory_ok(self) -> None:
        """Writing to an existing directory does not fail."""
        subdir = os.path.join(self.tmpdir, "existing")
        os.makedirs(subdir)
        path = os.path.join(subdir, "file.txt")
        result = self.tool.execute(file_path=path, content="content")
        self.assertIn("Successfully wrote", result)


class TestWriteToolOverwrite(unittest.TestCase):
    """Tests for file overwrite behavior."""

    def setUp(self) -> None:
        self.tool = WriteTool()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_overwrite_existing_file(self) -> None:
        """Overwrite an existing file with new content."""
        path = os.path.join(self.tmpdir, "existing.txt")
        Path(path).write_text("original content", encoding="utf-8")

        result = self.tool.execute(file_path=path, content="new content")
        self.assertIn("Successfully wrote", result)
        self.assertEqual(
            Path(path).read_text(encoding="utf-8"), "new content"
        )

    def test_overwrite_replaces_entirely(self) -> None:
        """Overwriting replaces the file completely, not appending."""
        path = os.path.join(self.tmpdir, "replace.txt")
        Path(path).write_text("old old old", encoding="utf-8")

        self.tool.execute(file_path=path, content="new")
        self.assertEqual(Path(path).read_text(encoding="utf-8"), "new")

    def test_writing_to_directory_returns_error(self) -> None:
        """Writing to a path that is a directory returns an error."""
        result = self.tool.execute(file_path=self.tmpdir, content="content")
        self.assertIn("Error", result)
        self.assertIn("directory", result.lower())


class TestWriteToolParameterValidation(unittest.TestCase):
    """Tests for parameter validation."""

    def setUp(self) -> None:
        self.tool = WriteTool()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_missing_file_path_returns_error(self) -> None:
        """Missing file_path parameter returns an error."""
        result = self.tool.execute(content="hello")
        self.assertIn("Error", result)

    def test_empty_file_path_returns_error(self) -> None:
        """Empty file_path returns an error."""
        result = self.tool.execute(file_path="", content="hello")
        self.assertIn("Error", result)

    def test_missing_content_returns_error(self) -> None:
        """Missing content parameter returns an error."""
        path = os.path.join(self.tmpdir, "no_content.txt")
        result = self.tool.execute(file_path=path)
        self.assertIn("Error", result)

    def test_non_string_content_returns_error(self) -> None:
        """Non-string content parameter returns an error."""
        path = os.path.join(self.tmpdir, "bad_content.txt")
        result = self.tool.execute(file_path=path, content=123)
        self.assertIn("Error", result)

    def test_non_string_file_path_returns_error(self) -> None:
        """Non-string file_path returns an error."""
        result = self.tool.execute(file_path=123, content="hello")
        self.assertIn("Error", result)


class TestWriteToolSecurity(unittest.TestCase):
    """Tests for directory traversal prevention."""

    def setUp(self) -> None:
        self.tool = WriteTool()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_directory_traversal_blocked(self) -> None:
        """Relative paths with '..' that escape cwd are blocked."""
        result = self.tool.execute(
            file_path="../../etc/passwd", content="hacked"
        )
        self.assertIn("Error", result)
        self.assertIn("traversal", result.lower())

    def test_absolute_path_to_etc_blocked(self) -> None:
        """Absolute paths to /etc/ are blocked."""
        result = self.tool.execute(
            file_path="/etc/passwd", content="hacked"
        )
        self.assertIn("Error", result)

    def test_absolute_path_to_dev_blocked(self) -> None:
        """Absolute paths to /dev/ are blocked."""
        result = self.tool.execute(
            file_path="/dev/sda", content="hacked"
        )
        self.assertIn("Error", result)

    def test_absolute_path_to_boot_blocked(self) -> None:
        """Absolute paths to /boot/ are blocked."""
        result = self.tool.execute(
            file_path="/boot/vmlinuz", content="hacked"
        )
        self.assertIn("Error", result)


if __name__ == "__main__":
    unittest.main()
