"""Tests for local_cli.tools._fileio (atomic write helper)."""

import os
import shutil
import stat
import tempfile
import unittest
from pathlib import Path

from local_cli.tools._fileio import (
    atomic_write_text,
    find_similar_path,
    not_found_error,
)


class TestAtomicWriteText(unittest.TestCase):
    """Tests for atomic_write_text()."""

    def test_writes_content(self) -> None:
        """Content is written and read back verbatim."""
        fd, p = tempfile.mkstemp()
        os.close(fd)
        try:
            atomic_write_text(Path(p), "hello\nworld\n")
            self.assertEqual(Path(p).read_text(encoding="utf-8"), "hello\nworld\n")
        finally:
            os.unlink(p)

    def test_overwrite_preserves_existing_mode(self) -> None:
        """Overwriting an existing file keeps its permission bits."""
        fd, p = tempfile.mkstemp()
        os.close(fd)
        try:
            os.chmod(p, 0o755)
            atomic_write_text(Path(p), "replaced")
            self.assertEqual(stat.S_IMODE(os.stat(p).st_mode), 0o755)
            self.assertEqual(Path(p).read_text(encoding="utf-8"), "replaced")
        finally:
            os.unlink(p)

    def test_new_file_respects_umask(self) -> None:
        """A newly created file gets 0666 & ~umask, not mkstemp's 0600."""
        d = tempfile.mkdtemp()
        try:
            p = Path(d) / "new.txt"
            atomic_write_text(p, "data")
            mode = stat.S_IMODE(os.stat(p).st_mode)
            current_umask = os.umask(0)
            os.umask(current_umask)
            self.assertEqual(mode, 0o666 & ~current_umask)
        finally:
            shutil.rmtree(d)

    def test_overwrite_leaves_no_temp_files(self) -> None:
        """After overwriting, only the target file remains in the dir."""
        d = tempfile.mkdtemp()
        try:
            p = Path(d) / "f.txt"
            atomic_write_text(p, "a")
            atomic_write_text(p, "b")
            self.assertEqual(p.read_text(encoding="utf-8"), "b")
            self.assertEqual(list(Path(d).iterdir()), [p])
        finally:
            shutil.rmtree(d)


class TestFindSimilarPath(unittest.TestCase):
    """Tests for the not-found path suggestion."""

    def setUp(self) -> None:
        self._dir = tempfile.mkdtemp()
        self._old_cwd = os.getcwd()
        os.chdir(self._dir)

    def tearDown(self) -> None:
        os.chdir(self._old_cwd)
        shutil.rmtree(self._dir, ignore_errors=True)

    def test_hallucinated_absolute_prefix_resolves_to_cwd_file(self) -> None:
        """The live 4B failure: /app/app.py for a file that is ./app.py."""
        Path("app.py").write_text("x = 1\n", encoding="utf-8")
        self.assertEqual(find_similar_path("/app/app.py"), "app.py")

    def test_one_level_down(self) -> None:
        Path("src").mkdir()
        (Path("src") / "main.py").write_text("", encoding="utf-8")
        self.assertEqual(find_similar_path("/main.py"), str(Path("src") / "main.py"))

    def test_two_levels_down(self) -> None:
        deep = Path("a") / "b"
        deep.mkdir(parents=True)
        (deep / "mod.py").write_text("", encoding="utf-8")
        self.assertEqual(find_similar_path("mod.py"), str(deep / "mod.py"))

    def test_ambiguous_match_gives_no_hint(self) -> None:
        for d in ("x", "y"):
            Path(d).mkdir()
            (Path(d) / "dup.py").write_text("", encoding="utf-8")
        self.assertIsNone(find_similar_path("/somewhere/dup.py"))

    def test_no_match_gives_no_hint(self) -> None:
        self.assertIsNone(find_similar_path("/nope/never.py"))

    def test_empty_name(self) -> None:
        self.assertIsNone(find_similar_path(""))


class TestNotFoundError(unittest.TestCase):
    """Tests for the composed not-found error message."""

    def setUp(self) -> None:
        self._dir = tempfile.mkdtemp()
        self._old_cwd = os.getcwd()
        os.chdir(self._dir)

    def tearDown(self) -> None:
        os.chdir(self._old_cwd)
        shutil.rmtree(self._dir, ignore_errors=True)

    def test_hint_included_when_file_exists_elsewhere(self) -> None:
        Path("app.py").write_text("", encoding="utf-8")
        message = not_found_error("/app/app.py")
        self.assertIn("Error: file not found: /app/app.py", message)
        self.assertIn("'app.py'", message)
        self.assertIn("use that path", message)

    def test_plain_error_when_no_similar_file(self) -> None:
        message = not_found_error("/gone/missing.py")
        self.assertEqual(message, "Error: file not found: /gone/missing.py")

    def test_no_self_referential_hint(self) -> None:
        """A relative request that simply does not exist gets no hint."""
        message = not_found_error("absent.py")
        self.assertEqual(message, "Error: file not found: absent.py")


if __name__ == "__main__":
    unittest.main()
