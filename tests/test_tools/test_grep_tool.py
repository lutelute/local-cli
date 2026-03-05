"""Tests for local_cli.tools.grep_tool module."""

import os
import tempfile
import unittest
from pathlib import Path

from local_cli.tools.grep_tool import GrepTool


class TestGrepToolMetadata(unittest.TestCase):
    """Tests for GrepTool metadata properties."""

    def setUp(self) -> None:
        self.tool = GrepTool()

    def test_name(self) -> None:
        """Tool name is 'grep'."""
        self.assertEqual(self.tool.name, "grep")

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
        self.assertEqual(tool_def["function"]["name"], "grep")


class TestGrepToolRegexSearch(unittest.TestCase):
    """Tests for regex pattern searching."""

    def setUp(self) -> None:
        self.tool = GrepTool()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_file(self, name: str, content: str) -> str:
        """Create a temp file with the given content and return its path."""
        path = os.path.join(self.tmpdir, name)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(content, encoding="utf-8")
        return path

    def test_simple_string_match(self) -> None:
        """Match a simple string pattern."""
        self._create_file("test.py", "def hello():\n    return 'world'\n")
        result = self.tool.execute(pattern="hello", path=self.tmpdir)
        self.assertIn("hello", result)
        self.assertIn("test.py", result)

    def test_regex_pattern_match(self) -> None:
        """Match a regex pattern with special characters."""
        self._create_file("test.py", "def func_123():\n    pass\n")
        result = self.tool.execute(pattern=r"func_\d+", path=self.tmpdir)
        self.assertIn("func_123", result)

    def test_line_number_in_output(self) -> None:
        """Output includes line numbers."""
        self._create_file(
            "numbered.py",
            "line one\nline two\ntarget line\nline four\n",
        )
        result = self.tool.execute(pattern="target", path=self.tmpdir)
        # Format should be file:line_number:content.
        self.assertIn(":3:", result)
        self.assertIn("target line", result)

    def test_multiple_matches_in_file(self) -> None:
        """Find multiple matches in a single file."""
        self._create_file(
            "multi.txt",
            "apple\nbanana\napple pie\ncherry\napple sauce\n",
        )
        result = self.tool.execute(pattern="apple", path=self.tmpdir)
        # Should find 3 lines with 'apple'.
        lines = [l for l in result.split("\n") if "apple" in l]
        self.assertEqual(len(lines), 3)

    def test_matches_across_files(self) -> None:
        """Find matches across multiple files."""
        self._create_file("a.py", "import os\n")
        self._create_file("b.py", "import sys\n")
        self._create_file("c.py", "import os\nimport re\n")
        result = self.tool.execute(pattern="import os", path=self.tmpdir)
        self.assertIn("a.py", result)
        self.assertIn("c.py", result)
        self.assertNotIn("b.py", result)

    def test_no_matches_returns_message(self) -> None:
        """No matches returns a descriptive message."""
        self._create_file("empty_match.txt", "nothing here\n")
        result = self.tool.execute(
            pattern="nonexistent_pattern_xyz", path=self.tmpdir
        )
        self.assertIn("No matches found", result)

    def test_search_single_file(self) -> None:
        """Search within a specific file (not a directory)."""
        path = self._create_file("single.txt", "find this line\nskip this\n")
        result = self.tool.execute(pattern="find", path=path)
        self.assertIn("find this line", result)

    def test_invalid_regex_returns_error(self) -> None:
        """Invalid regex pattern returns an error."""
        self._create_file("test.txt", "content\n")
        result = self.tool.execute(pattern="[invalid", path=self.tmpdir)
        self.assertIn("Error", result)
        self.assertIn("invalid regex", result.lower())


class TestGrepToolCaseInsensitive(unittest.TestCase):
    """Tests for case-insensitive search mode."""

    def setUp(self) -> None:
        self.tool = GrepTool()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_file(self, name: str, content: str) -> str:
        path = os.path.join(self.tmpdir, name)
        Path(path).write_text(content, encoding="utf-8")
        return path

    def test_case_insensitive_matches_upper(self) -> None:
        """Case-insensitive search matches uppercase text."""
        self._create_file(
            "case.txt", "Hello World\nhello world\nHELLO WORLD\n"
        )
        result = self.tool.execute(
            pattern="hello", path=self.tmpdir, case_insensitive=True
        )
        lines = [l for l in result.split("\n") if l.strip()]
        self.assertEqual(len(lines), 3)

    def test_case_sensitive_default(self) -> None:
        """Default search is case-sensitive."""
        self._create_file(
            "case.txt", "Hello\nhello\nHELLO\n"
        )
        result = self.tool.execute(pattern="hello", path=self.tmpdir)
        # Only 'hello' should match (case-sensitive by default).
        lines = [l for l in result.split("\n") if "hello" in l.lower() and l.strip()]
        # Check that we don't match all three.
        self.assertIn("hello", result)
        # 'Hello' and 'HELLO' should NOT match when case-sensitive.
        match_lines = [l for l in result.split("\n") if l.strip()]
        self.assertEqual(len(match_lines), 1)

    def test_case_insensitive_regex(self) -> None:
        """Case-insensitive mode works with regex patterns."""
        self._create_file("regex_case.txt", "FooBar\nfoobar\nFOOBAR\n")
        result = self.tool.execute(
            pattern=r"foo.*bar",
            path=self.tmpdir,
            case_insensitive=True,
        )
        lines = [l for l in result.split("\n") if l.strip()]
        self.assertEqual(len(lines), 3)


class TestGrepToolFileTypeFiltering(unittest.TestCase):
    """Tests for file type filtering with include parameter."""

    def setUp(self) -> None:
        self.tool = GrepTool()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_file(self, name: str, content: str) -> str:
        path = os.path.join(self.tmpdir, name)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(content, encoding="utf-8")
        return path

    def test_include_py_files_only(self) -> None:
        """Include filter restricts search to *.py files."""
        self._create_file("code.py", "import target\n")
        self._create_file("notes.txt", "target text\n")
        self._create_file("data.json", '{"target": true}\n')
        result = self.tool.execute(
            pattern="target", path=self.tmpdir, include="*.py"
        )
        self.assertIn("code.py", result)
        self.assertNotIn("notes.txt", result)
        self.assertNotIn("data.json", result)

    def test_include_txt_files_only(self) -> None:
        """Include filter restricts search to *.txt files."""
        self._create_file("file.py", "match here\n")
        self._create_file("file.txt", "match here too\n")
        result = self.tool.execute(
            pattern="match", path=self.tmpdir, include="*.txt"
        )
        self.assertIn("file.txt", result)
        self.assertNotIn("file.py", result)

    def test_include_with_subdirectories(self) -> None:
        """Include filter works with recursive subdirectory search."""
        self._create_file("src/main.py", "find me\n")
        self._create_file("src/style.css", "find me too\n")
        self._create_file("lib/util.py", "also find me\n")
        result = self.tool.execute(
            pattern="find me", path=self.tmpdir, include="*.py"
        )
        self.assertIn("main.py", result)
        self.assertIn("util.py", result)
        self.assertNotIn("style.css", result)

    def test_no_include_searches_all_files(self) -> None:
        """Without include filter, all files are searched."""
        self._create_file("a.py", "common\n")
        self._create_file("b.txt", "common\n")
        self._create_file("c.md", "common\n")
        result = self.tool.execute(pattern="common", path=self.tmpdir)
        self.assertIn("a.py", result)
        self.assertIn("b.txt", result)
        self.assertIn("c.md", result)


class TestGrepToolBinarySkipping(unittest.TestCase):
    """Tests for binary file skipping."""

    def setUp(self) -> None:
        self.tool = GrepTool()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_binary_files_skipped(self) -> None:
        """Binary files (with null bytes) are skipped during search."""
        # Create a text file with a match.
        text_path = os.path.join(self.tmpdir, "text.txt")
        Path(text_path).write_text("find this\n", encoding="utf-8")
        # Create a binary file with the same content plus null bytes.
        bin_path = os.path.join(self.tmpdir, "binary.bin")
        Path(bin_path).write_bytes(b"find this\x00more data")

        result = self.tool.execute(pattern="find", path=self.tmpdir)
        self.assertIn("text.txt", result)
        self.assertNotIn("binary.bin", result)


class TestGrepToolMissingPath(unittest.TestCase):
    """Tests for missing path handling."""

    def setUp(self) -> None:
        self.tool = GrepTool()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_nonexistent_path_returns_error(self) -> None:
        """Searching a non-existent path returns an error."""
        fake_path = os.path.join(self.tmpdir, "nonexistent")
        result = self.tool.execute(pattern="test", path=fake_path)
        self.assertIn("Error", result)
        self.assertIn("not found", result)


class TestGrepToolParameterValidation(unittest.TestCase):
    """Tests for parameter validation."""

    def setUp(self) -> None:
        self.tool = GrepTool()

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


if __name__ == "__main__":
    unittest.main()
