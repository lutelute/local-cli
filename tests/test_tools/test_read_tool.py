"""Tests for local_cli.tools.read_tool module."""

import os
import tempfile
import unittest
from pathlib import Path

from local_cli.tools.read_tool import ReadTool


class TestReadToolMetadata(unittest.TestCase):
    """Tests for ReadTool metadata properties."""

    def setUp(self) -> None:
        self.tool = ReadTool()

    def test_name(self) -> None:
        """Tool name is 'read'."""
        self.assertEqual(self.tool.name, "read")

    def test_description_is_nonempty(self) -> None:
        """Description is a non-empty string."""
        self.assertIsInstance(self.tool.description, str)
        self.assertTrue(len(self.tool.description) > 0)

    def test_parameters_schema(self) -> None:
        """Parameters schema defines 'file_path' as required."""
        params = self.tool.parameters
        self.assertEqual(params["type"], "object")
        self.assertIn("file_path", params["properties"])
        self.assertIn("file_path", params["required"])

    def test_to_ollama_tool(self) -> None:
        """to_ollama_tool returns correct function-calling format."""
        tool_def = self.tool.to_ollama_tool()
        self.assertEqual(tool_def["type"], "function")
        self.assertEqual(tool_def["function"]["name"], "read")


class TestReadToolTextFiles(unittest.TestCase):
    """Tests for reading text files."""

    def setUp(self) -> None:
        self.tool = ReadTool()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        # Clean up temp files.
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_file(self, name: str, content: str) -> str:
        """Create a temp file with the given content and return its path."""
        path = os.path.join(self.tmpdir, name)
        Path(path).write_text(content, encoding="utf-8")
        return path

    def test_read_simple_file(self) -> None:
        """Read a simple text file and get content."""
        path = self._create_file("test.txt", "hello world\n")
        result = self.tool.execute(file_path=path)
        self.assertIn("hello world", result)

    def test_line_numbers_present(self) -> None:
        """Output includes line numbers."""
        path = self._create_file("numbered.txt", "line one\nline two\nline three\n")
        result = self.tool.execute(file_path=path)
        # Line numbers should be present (1-based).
        self.assertIn("1", result)
        self.assertIn("line one", result)
        self.assertIn("2", result)
        self.assertIn("line two", result)
        self.assertIn("3", result)
        self.assertIn("line three", result)

    def test_line_number_format(self) -> None:
        """Line numbers are formatted with tab separator."""
        path = self._create_file("fmt.txt", "alpha\nbeta\n")
        result = self.tool.execute(file_path=path)
        lines = result.split("\n")
        # Each line should contain a tab between line number and content.
        for line in lines:
            if line.strip():
                self.assertIn("\t", line)

    def test_read_multiline_file(self) -> None:
        """Read a file with multiple lines."""
        content = "first\nsecond\nthird\nfourth\nfifth\n"
        path = self._create_file("multi.txt", content)
        result = self.tool.execute(file_path=path)
        self.assertIn("first", result)
        self.assertIn("fifth", result)

    def test_read_empty_file(self) -> None:
        """Read an empty file returns empty result (no error)."""
        path = self._create_file("empty.txt", "")
        result = self.tool.execute(file_path=path)
        # Should not contain "Error".
        self.assertNotIn("Error", result)

    def test_read_single_line_no_newline(self) -> None:
        """Read a file with a single line and no trailing newline."""
        path = self._create_file("single.txt", "only line")
        result = self.tool.execute(file_path=path)
        self.assertIn("only line", result)


class TestReadToolOffset(unittest.TestCase):
    """Tests for offset and limit parameters."""

    def setUp(self) -> None:
        self.tool = ReadTool()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_file(self, name: str, content: str) -> str:
        path = os.path.join(self.tmpdir, name)
        Path(path).write_text(content, encoding="utf-8")
        return path

    def test_offset_skips_lines(self) -> None:
        """Offset parameter skips the first N-1 lines."""
        content = "line1\nline2\nline3\nline4\nline5\n"
        path = self._create_file("offset.txt", content)
        result = self.tool.execute(file_path=path, offset=3)
        self.assertNotIn("line1", result)
        self.assertNotIn("line2", result)
        self.assertIn("line3", result)
        self.assertIn("line4", result)

    def test_limit_restricts_lines(self) -> None:
        """Limit parameter restricts the number of lines returned."""
        content = "line1\nline2\nline3\nline4\nline5\n"
        path = self._create_file("limit.txt", content)
        result = self.tool.execute(file_path=path, limit=2)
        self.assertIn("line1", result)
        self.assertIn("line2", result)
        self.assertNotIn("line3", result)

    def test_offset_and_limit_combined(self) -> None:
        """Offset and limit work together."""
        content = "a\nb\nc\nd\ne\n"
        path = self._create_file("combined.txt", content)
        result = self.tool.execute(file_path=path, offset=2, limit=2)
        self.assertNotIn("\ta\n", result)
        self.assertIn("b", result)
        self.assertIn("c", result)
        self.assertNotIn("\td\n", result)

    def test_offset_beyond_file_returns_error(self) -> None:
        """Offset beyond end of file returns an error."""
        content = "line1\nline2\n"
        path = self._create_file("beyond.txt", content)
        result = self.tool.execute(file_path=path, offset=100)
        self.assertIn("Error", result)
        self.assertIn("beyond", result.lower())


class TestReadToolBinaryDetection(unittest.TestCase):
    """Tests for binary file detection."""

    def setUp(self) -> None:
        self.tool = ReadTool()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_binary_file_detected(self) -> None:
        """Files with null bytes are detected as binary."""
        path = os.path.join(self.tmpdir, "binary.bin")
        Path(path).write_bytes(b"some\x00binary\x00data")
        result = self.tool.execute(file_path=path)
        self.assertIn("Error", result)
        self.assertIn("binary", result.lower())

    def test_text_file_not_detected_as_binary(self) -> None:
        """Regular text files are not flagged as binary."""
        path = os.path.join(self.tmpdir, "text.txt")
        Path(path).write_text("just regular text\n", encoding="utf-8")
        result = self.tool.execute(file_path=path)
        self.assertNotIn("binary", result.lower())
        self.assertIn("regular text", result)


class TestReadToolMissingFile(unittest.TestCase):
    """Tests for handling missing or invalid files."""

    def setUp(self) -> None:
        self.tool = ReadTool()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_missing_file_returns_error(self) -> None:
        """Reading a non-existent file returns an error."""
        path = os.path.join(self.tmpdir, "nonexistent.txt")
        result = self.tool.execute(file_path=path)
        self.assertIn("Error", result)
        self.assertIn("not found", result)

    def test_directory_returns_error(self) -> None:
        """Reading a directory returns an error."""
        result = self.tool.execute(file_path=self.tmpdir)
        self.assertIn("Error", result)
        self.assertIn("not a regular file", result)


class TestReadToolParameterValidation(unittest.TestCase):
    """Tests for parameter validation."""

    def setUp(self) -> None:
        self.tool = ReadTool()

    def test_missing_file_path_returns_error(self) -> None:
        """Missing file_path parameter returns an error."""
        result = self.tool.execute()
        self.assertIn("Error", result)

    def test_empty_file_path_returns_error(self) -> None:
        """Empty file_path string returns an error."""
        result = self.tool.execute(file_path="")
        self.assertIn("Error", result)

    def test_non_string_file_path_returns_error(self) -> None:
        """Non-string file_path returns an error."""
        result = self.tool.execute(file_path=123)
        self.assertIn("Error", result)


if __name__ == "__main__":
    unittest.main()
