"""Tests for local_cli.diff_preview module."""

import unittest

from local_cli.diff_preview import (
    DiffPreviewError,
    _BINARY_CHECK_SIZE,
    _COLOR_BOLD,
    _COLOR_CYAN,
    _COLOR_GREEN,
    _COLOR_RED,
    _COLOR_RESET,
    _MAX_DIFF_LINES,
    binary_file_placeholder,
    colorize_diff,
    format_diff_output,
    generate_diff,
    generate_multi_file_diff,
    is_binary_content,
    truncate_diff,
)


class TestIsBinaryContent(unittest.TestCase):
    """Tests for is_binary_content()."""

    def test_empty_bytes_is_not_binary(self) -> None:
        """Empty bytes are not considered binary."""
        self.assertFalse(is_binary_content(b""))

    def test_plain_text_is_not_binary(self) -> None:
        """Plain ASCII text is not considered binary."""
        self.assertFalse(is_binary_content(b"Hello, world!\n"))

    def test_utf8_text_is_not_binary(self) -> None:
        """UTF-8 encoded text without null bytes is not binary."""
        self.assertFalse(is_binary_content("日本語テスト\n".encode("utf-8")))

    def test_null_byte_is_binary(self) -> None:
        """Content with a null byte is considered binary."""
        self.assertTrue(is_binary_content(b"hello\x00world"))

    def test_null_byte_at_start_is_binary(self) -> None:
        """Content starting with a null byte is binary."""
        self.assertTrue(is_binary_content(b"\x00hello"))

    def test_null_byte_beyond_check_size_is_not_binary(self) -> None:
        """Null bytes beyond _BINARY_CHECK_SIZE are ignored."""
        data = b"x" * _BINARY_CHECK_SIZE + b"\x00"
        self.assertFalse(is_binary_content(data))

    def test_null_byte_within_check_size_is_binary(self) -> None:
        """Null bytes within _BINARY_CHECK_SIZE are detected."""
        data = b"x" * (_BINARY_CHECK_SIZE - 1) + b"\x00"
        self.assertTrue(is_binary_content(data))

    def test_rejects_non_bytes_input(self) -> None:
        """Raises TypeError when given a non-bytes argument."""
        with self.assertRaises(TypeError):
            is_binary_content("not bytes")  # type: ignore[arg-type]

    def test_rejects_int_input(self) -> None:
        """Raises TypeError when given an integer."""
        with self.assertRaises(TypeError):
            is_binary_content(42)  # type: ignore[arg-type]

    def test_png_header_is_binary(self) -> None:
        """PNG file header (contains null bytes) is detected as binary."""
        png_header = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        self.assertTrue(is_binary_content(png_header))


class TestGenerateDiff(unittest.TestCase):
    """Tests for generate_diff()."""

    def test_identical_files_return_empty(self) -> None:
        """No diff is produced when files are identical."""
        lines = ["line 1", "line 2", "line 3"]
        result = generate_diff(lines, lines)
        self.assertEqual(result, "")

    def test_single_line_added(self) -> None:
        """Diff shows a single added line."""
        from_lines = ["line 1", "line 2"]
        to_lines = ["line 1", "line 2", "line 3"]
        result = generate_diff(from_lines, to_lines)
        self.assertIn("+line 3", result)
        self.assertNotIn("-line", result)

    def test_single_line_removed(self) -> None:
        """Diff shows a single removed line."""
        from_lines = ["line 1", "line 2", "line 3"]
        to_lines = ["line 1", "line 2"]
        result = generate_diff(from_lines, to_lines)
        self.assertIn("-line 3", result)

    def test_single_line_modified(self) -> None:
        """Diff shows a modification (old line removed, new line added)."""
        from_lines = ["line 1", "original", "line 3"]
        to_lines = ["line 1", "modified", "line 3"]
        result = generate_diff(from_lines, to_lines)
        self.assertIn("-original", result)
        self.assertIn("+modified", result)

    def test_unified_diff_has_file_headers(self) -> None:
        """Diff output contains --- and +++ file headers."""
        from_lines = ["old"]
        to_lines = ["new"]
        result = generate_diff(from_lines, to_lines, from_file="a/foo.py", to_file="b/foo.py")
        self.assertIn("--- a/foo.py", result)
        self.assertIn("+++ b/foo.py", result)

    def test_unified_diff_has_hunk_header(self) -> None:
        """Diff output contains @@ hunk headers."""
        from_lines = ["old"]
        to_lines = ["new"]
        result = generate_diff(from_lines, to_lines)
        self.assertIn("@@", result)

    def test_custom_file_labels(self) -> None:
        """Custom from_file and to_file labels appear in output."""
        from_lines = ["a"]
        to_lines = ["b"]
        result = generate_diff(
            from_lines, to_lines,
            from_file="original.txt",
            to_file="modified.txt",
        )
        self.assertIn("--- original.txt", result)
        self.assertIn("+++ modified.txt", result)

    def test_empty_from_lines_shows_all_added(self) -> None:
        """When from_lines is empty, all to_lines are shown as additions."""
        to_lines = ["new line 1", "new line 2"]
        result = generate_diff([], to_lines)
        self.assertIn("+new line 1", result)
        self.assertIn("+new line 2", result)

    def test_empty_to_lines_shows_all_removed(self) -> None:
        """When to_lines is empty, all from_lines are shown as deletions."""
        from_lines = ["old line 1", "old line 2"]
        result = generate_diff(from_lines, [])
        self.assertIn("-old line 1", result)
        self.assertIn("-old line 2", result)

    def test_both_empty_returns_empty(self) -> None:
        """When both from and to are empty, no diff is produced."""
        result = generate_diff([], [])
        self.assertEqual(result, "")

    def test_context_lines_parameter(self) -> None:
        """context_lines controls the number of context lines around changes."""
        from_lines = [f"line {i}" for i in range(20)]
        to_lines = list(from_lines)
        to_lines[10] = "CHANGED"
        # With 1 context line, fewer unchanged lines are shown.
        result_1 = generate_diff(from_lines, to_lines, context_lines=1)
        # With 5 context lines, more unchanged lines are shown.
        result_5 = generate_diff(from_lines, to_lines, context_lines=5)
        self.assertGreater(len(result_5), len(result_1))

    def test_diff_ends_with_newline(self) -> None:
        """Non-empty diff output ends with a newline character."""
        result = generate_diff(["a"], ["b"])
        self.assertTrue(result.endswith("\n"))

    def test_multiline_changes(self) -> None:
        """Diff correctly handles multiple changed lines."""
        from_lines = ["A", "B", "C", "D", "E"]
        to_lines = ["A", "X", "Y", "D", "E"]
        result = generate_diff(from_lines, to_lines)
        self.assertIn("-B", result)
        self.assertIn("-C", result)
        self.assertIn("+X", result)
        self.assertIn("+Y", result)


class TestColorizeDiff(unittest.TestCase):
    """Tests for colorize_diff()."""

    def test_empty_string_unchanged(self) -> None:
        """Empty input is returned as-is."""
        self.assertEqual(colorize_diff(""), "")

    def test_addition_lines_are_green(self) -> None:
        """Lines starting with '+' (not '+++') get green color."""
        diff = "+added line\n"
        result = colorize_diff(diff)
        self.assertIn(_COLOR_GREEN, result)
        self.assertIn("+added line", result)
        self.assertIn(_COLOR_RESET, result)

    def test_deletion_lines_are_red(self) -> None:
        """Lines starting with '-' (not '---') get red color."""
        diff = "-removed line\n"
        result = colorize_diff(diff)
        self.assertIn(_COLOR_RED, result)
        self.assertIn("-removed line", result)
        self.assertIn(_COLOR_RESET, result)

    def test_hunk_headers_are_cyan(self) -> None:
        """Lines starting with '@@' get cyan color."""
        diff = "@@ -1,3 +1,3 @@\n"
        result = colorize_diff(diff)
        self.assertIn(_COLOR_CYAN, result)
        self.assertIn("@@", result)

    def test_file_headers_are_bold(self) -> None:
        """Lines starting with '---' or '+++' get bold formatting."""
        diff = "--- a/file.py\n+++ b/file.py\n"
        result = colorize_diff(diff)
        self.assertIn(_COLOR_BOLD, result)
        # File headers should NOT use red/green
        lines = result.splitlines()
        for line in lines:
            if "--- a/file.py" in line:
                self.assertNotIn(_COLOR_RED, line)
            if "+++ b/file.py" in line:
                self.assertNotIn(_COLOR_GREEN, line)

    def test_context_lines_unchanged(self) -> None:
        """Context lines (no +/-/@ prefix) are not coloured."""
        diff = " unchanged line\n"
        result = colorize_diff(diff)
        self.assertNotIn(_COLOR_GREEN, result)
        self.assertNotIn(_COLOR_RED, result)
        self.assertNotIn(_COLOR_CYAN, result)

    def test_full_diff_colourised(self) -> None:
        """A complete diff is colourised correctly."""
        diff = (
            "--- a/test.py\n"
            "+++ b/test.py\n"
            "@@ -1,3 +1,3 @@\n"
            " line 1\n"
            "-old line\n"
            "+new line\n"
            " line 3\n"
        )
        result = colorize_diff(diff)
        # Check each line type is coloured appropriately.
        self.assertIn(f"{_COLOR_BOLD}--- a/test.py{_COLOR_RESET}", result)
        self.assertIn(f"{_COLOR_BOLD}+++ b/test.py{_COLOR_RESET}", result)
        self.assertIn(f"{_COLOR_RED}-old line{_COLOR_RESET}", result)
        self.assertIn(f"{_COLOR_GREEN}+new line{_COLOR_RESET}", result)

    def test_result_ends_with_newline(self) -> None:
        """Colourised output ends with a newline."""
        diff = "+added\n"
        result = colorize_diff(diff)
        self.assertTrue(result.endswith("\n"))


class TestTruncateDiff(unittest.TestCase):
    """Tests for truncate_diff()."""

    def test_empty_string_unchanged(self) -> None:
        """Empty input is returned as-is."""
        self.assertEqual(truncate_diff(""), "")

    def test_short_diff_unchanged(self) -> None:
        """Diff within the limit is returned unchanged."""
        diff = "line 1\nline 2\nline 3\n"
        result = truncate_diff(diff, max_lines=10)
        self.assertEqual(result, diff)

    def test_exact_limit_unchanged(self) -> None:
        """Diff with exactly max_lines lines is not truncated."""
        lines = [f"line {i}" for i in range(10)]
        diff = "\n".join(lines) + "\n"
        result = truncate_diff(diff, max_lines=10)
        self.assertEqual(result, diff)

    def test_exceeds_limit_is_truncated(self) -> None:
        """Diff exceeding the limit is truncated with a notice."""
        lines = [f"line {i}" for i in range(20)]
        diff = "\n".join(lines) + "\n"
        result = truncate_diff(diff, max_lines=10)
        result_lines = result.splitlines()
        # Should have 10 kept lines + 1 truncation notice = 11 lines.
        self.assertEqual(len(result_lines), 11)
        self.assertIn("... truncated", result_lines[-1])
        self.assertIn("10 more lines", result_lines[-1])

    def test_truncation_notice_shows_correct_count(self) -> None:
        """Truncation notice shows the correct number of omitted lines."""
        lines = [f"line {i}" for i in range(100)]
        diff = "\n".join(lines) + "\n"
        result = truncate_diff(diff, max_lines=30)
        self.assertIn("70 more lines", result)

    def test_default_max_lines(self) -> None:
        """Default max_lines is _MAX_DIFF_LINES (500)."""
        lines = [f"line {i}" for i in range(600)]
        diff = "\n".join(lines) + "\n"
        result = truncate_diff(diff)
        self.assertIn("... truncated", result)
        self.assertIn("100 more lines", result)

    def test_truncated_result_ends_with_newline(self) -> None:
        """Truncated output ends with a newline."""
        lines = [f"line {i}" for i in range(20)]
        diff = "\n".join(lines) + "\n"
        result = truncate_diff(diff, max_lines=5)
        self.assertTrue(result.endswith("\n"))


class TestFormatDiffOutput(unittest.TestCase):
    """Tests for format_diff_output()."""

    def test_empty_string_unchanged(self) -> None:
        """Empty input is returned as-is."""
        self.assertEqual(format_diff_output(""), "")

    def test_color_and_truncation_applied(self) -> None:
        """Both colour and truncation are applied."""
        lines = ["+added"] * 20
        diff = "\n".join(lines) + "\n"
        result = format_diff_output(diff, color=True, max_lines=5)
        # Should be truncated.
        self.assertIn("... truncated", result)
        # Should be coloured.
        self.assertIn(_COLOR_GREEN, result)

    def test_no_color_when_disabled(self) -> None:
        """No ANSI codes when color=False."""
        diff = "+added line\n-removed line\n"
        result = format_diff_output(diff, color=False)
        self.assertNotIn(_COLOR_GREEN, result)
        self.assertNotIn(_COLOR_RED, result)
        self.assertNotIn(_COLOR_RESET, result)

    def test_no_truncation_when_zero(self) -> None:
        """No truncation when max_lines=0."""
        lines = [f"line {i}" for i in range(1000)]
        diff = "\n".join(lines) + "\n"
        result = format_diff_output(diff, color=False, max_lines=0)
        self.assertNotIn("truncated", result)
        # All lines should be present.
        self.assertEqual(len(result.splitlines()), 1000)

    def test_truncation_before_colouring(self) -> None:
        """Truncation is applied before colour so only visible lines are coloured."""
        lines = ["+line"] * 10
        diff = "\n".join(lines) + "\n"
        result = format_diff_output(diff, color=True, max_lines=3)
        # The truncation notice line should not have colour codes.
        result_lines = result.splitlines()
        truncation_line = result_lines[-1]
        self.assertIn("truncated", truncation_line)


class TestGenerateMultiFileDiff(unittest.TestCase):
    """Tests for generate_multi_file_diff()."""

    def test_empty_changes_returns_empty(self) -> None:
        """No changes produces an empty string."""
        result = generate_multi_file_diff([])
        self.assertEqual(result, "")

    def test_single_file_change(self) -> None:
        """Single file change produces a valid diff."""
        changes = [("test.py", ["old"], ["new"])]
        result = generate_multi_file_diff(changes, color=False)
        self.assertIn("--- a/test.py", result)
        self.assertIn("+++ b/test.py", result)
        self.assertIn("-old", result)
        self.assertIn("+new", result)

    def test_multiple_file_changes(self) -> None:
        """Multiple file changes produce a combined diff."""
        changes = [
            ("file1.py", ["a"], ["b"]),
            ("file2.py", ["x"], ["y"]),
        ]
        result = generate_multi_file_diff(changes, color=False)
        self.assertIn("--- a/file1.py", result)
        self.assertIn("+++ b/file1.py", result)
        self.assertIn("--- a/file2.py", result)
        self.assertIn("+++ b/file2.py", result)

    def test_new_file_shows_additions(self) -> None:
        """New file (empty from_lines) shows all lines as additions."""
        changes = [("new_file.py", [], ["line 1", "line 2"])]
        result = generate_multi_file_diff(changes, color=False)
        self.assertIn("+line 1", result)
        self.assertIn("+line 2", result)

    def test_deleted_file_shows_removals(self) -> None:
        """Deleted file (empty to_lines) shows all lines as removals."""
        changes = [("deleted.py", ["line 1", "line 2"], [])]
        result = generate_multi_file_diff(changes, color=False)
        self.assertIn("-line 1", result)
        self.assertIn("-line 2", result)

    def test_identical_files_excluded(self) -> None:
        """Identical files produce no diff section."""
        changes = [
            ("changed.py", ["old"], ["new"]),
            ("same.py", ["unchanged"], ["unchanged"]),
        ]
        result = generate_multi_file_diff(changes, color=False)
        self.assertIn("changed.py", result)
        self.assertNotIn("same.py", result)

    def test_color_applied_to_output(self) -> None:
        """Colour codes are applied when color=True."""
        changes = [("test.py", ["old"], ["new"])]
        result = generate_multi_file_diff(changes, color=True)
        self.assertIn(_COLOR_GREEN, result)
        self.assertIn(_COLOR_RED, result)

    def test_truncation_applied_to_output(self) -> None:
        """Large combined diffs are truncated."""
        # Create a change with many lines.
        from_lines = [f"old {i}" for i in range(300)]
        to_lines = [f"new {i}" for i in range(300)]
        changes = [("big.py", from_lines, to_lines)]
        result = generate_multi_file_diff(changes, color=False, max_lines=50)
        self.assertIn("... truncated", result)

    def test_file_path_prefixed_with_a_b(self) -> None:
        """File headers use a/ and b/ prefixes."""
        changes = [("src/main.py", ["x"], ["y"])]
        result = generate_multi_file_diff(changes, color=False)
        self.assertIn("--- a/src/main.py", result)
        self.assertIn("+++ b/src/main.py", result)


class TestBinaryFilePlaceholder(unittest.TestCase):
    """Tests for binary_file_placeholder()."""

    def test_returns_placeholder_with_path(self) -> None:
        """Placeholder contains the file path."""
        result = binary_file_placeholder("image.png")
        self.assertEqual(result, "[binary file: image.png]\n")

    def test_placeholder_with_nested_path(self) -> None:
        """Placeholder works with nested directory paths."""
        result = binary_file_placeholder("assets/icons/logo.png")
        self.assertEqual(result, "[binary file: assets/icons/logo.png]\n")

    def test_placeholder_ends_with_newline(self) -> None:
        """Placeholder ends with a newline for consistent formatting."""
        result = binary_file_placeholder("file.bin")
        self.assertTrue(result.endswith("\n"))


class TestDiffPreviewError(unittest.TestCase):
    """Tests for the DiffPreviewError exception."""

    def test_is_exception_subclass(self) -> None:
        """DiffPreviewError inherits from Exception."""
        self.assertTrue(issubclass(DiffPreviewError, Exception))

    def test_can_be_raised_and_caught(self) -> None:
        """DiffPreviewError can be raised and caught normally."""
        with self.assertRaises(DiffPreviewError):
            raise DiffPreviewError("test error")

    def test_message_is_preserved(self) -> None:
        """Exception message is preserved."""
        with self.assertRaises(DiffPreviewError) as ctx:
            raise DiffPreviewError("diff generation failed")
        self.assertEqual(str(ctx.exception), "diff generation failed")


class TestConstants(unittest.TestCase):
    """Tests for module-level constants."""

    def test_max_diff_lines_is_500(self) -> None:
        """Default max diff lines is 500."""
        self.assertEqual(_MAX_DIFF_LINES, 500)

    def test_binary_check_size_positive(self) -> None:
        """Binary check size is a positive integer."""
        self.assertGreater(_BINARY_CHECK_SIZE, 0)

    def test_color_codes_are_strings(self) -> None:
        """All colour codes are non-empty strings."""
        for code in (_COLOR_RED, _COLOR_GREEN, _COLOR_CYAN, _COLOR_BOLD, _COLOR_RESET):
            self.assertIsInstance(code, str)
            self.assertGreater(len(code), 0)


class TestEdgeCases(unittest.TestCase):
    """Additional edge-case tests for diff_preview module."""

    def test_generate_diff_whitespace_only_changes(self) -> None:
        """Diff detects whitespace-only changes."""
        from_lines = ["hello world"]
        to_lines = ["hello  world"]
        result = generate_diff(from_lines, to_lines)
        self.assertNotEqual(result, "")
        self.assertIn("-hello world", result)
        self.assertIn("+hello  world", result)

    def test_generate_diff_unicode_content(self) -> None:
        """Diff handles Unicode content correctly."""
        from_lines = ["日本語のテスト"]
        to_lines = ["日本語の確認"]
        result = generate_diff(from_lines, to_lines)
        self.assertIn("-日本語のテスト", result)
        self.assertIn("+日本語の確認", result)

    def test_colorize_diff_preserves_content(self) -> None:
        """Colourisation does not alter the original diff content."""
        diff = "+added line\n-removed line\n context\n"
        result = colorize_diff(diff)
        # Strip all ANSI codes and verify content is preserved.
        import re
        stripped = re.sub(r"\033\[[0-9;]*m", "", result)
        self.assertEqual(stripped, diff)

    def test_truncate_single_line_over_limit(self) -> None:
        """Truncation of a two-line diff with max_lines=1."""
        diff = "line 1\nline 2\n"
        result = truncate_diff(diff, max_lines=1)
        result_lines = result.splitlines()
        self.assertEqual(len(result_lines), 2)  # 1 kept + 1 notice
        self.assertEqual(result_lines[0], "line 1")
        self.assertIn("1 more lines", result_lines[1])

    def test_multi_file_diff_preserves_order(self) -> None:
        """Files appear in the diff in the same order as input."""
        changes = [
            ("z_file.py", ["a"], ["b"]),
            ("a_file.py", ["x"], ["y"]),
        ]
        result = generate_multi_file_diff(changes, color=False)
        z_pos = result.index("z_file.py")
        a_pos = result.index("a_file.py")
        self.assertLess(z_pos, a_pos)

    def test_generate_diff_large_file(self) -> None:
        """Diff handles large files without error."""
        from_lines = [f"line {i}" for i in range(1000)]
        to_lines = list(from_lines)
        to_lines[500] = "CHANGED LINE"
        result = generate_diff(from_lines, to_lines)
        self.assertIn("-line 500", result)
        self.assertIn("+CHANGED LINE", result)


if __name__ == "__main__":
    unittest.main()
