"""Tests for post-write merge-conflict-marker detection.

The verify_writes gate previously only touched ``.py``/``.json``/``.toml``,
so a broken file of any other type got no feedback.  Merge-conflict
markers break a file of *any* type and are almost never legitimate, so
they are now flagged for every suffix — the one check that spans all
file types.
"""

from local_cli.harness import verify_file_write

_CONFLICT = (
    "line one\n"
    "<<<<<<< HEAD\n"
    "ours\n"
    "=======\n"
    "theirs\n"
    ">>>>>>> branch\n"
)


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_conflict_markers_flagged_in_non_syntax_type(tmp_path):
    fp = _write(tmp_path, "notes.md", _CONFLICT)
    warning = verify_file_write("write", {"file_path": fp}, "wrote")
    assert warning is not None
    assert "merge-conflict" in warning
    assert "notes.md" in warning


def test_conflict_markers_flagged_in_python(tmp_path):
    fp = _write(tmp_path, "mod.py", _CONFLICT)
    warning = verify_file_write("edit", {"file_path": fp}, "edited")
    assert warning is not None
    assert "merge-conflict" in warning


def test_clean_markdown_passes(tmp_path):
    fp = _write(tmp_path, "ok.md", "# Title\n\nSome prose.\n")
    assert verify_file_write("write", {"file_path": fp}, "wrote") is None


def test_decorative_equals_rule_not_flagged(tmp_path):
    # A bare "=======" separator (no <<<<<<< / >>>>>>> pair) must not fire.
    fp = _write(tmp_path, "doc.md", "Section\n=======\n\nBody text.\n")
    assert verify_file_write("write", {"file_path": fp}, "wrote") is None


def test_valid_python_still_passes(tmp_path):
    fp = _write(tmp_path, "good.py", "def f():\n    return 1\n")
    assert verify_file_write("write", {"file_path": fp}, "wrote") is None


def test_broken_python_still_syntax_warns(tmp_path):
    fp = _write(tmp_path, "bad.py", "def f(:\n    pass\n")
    warning = verify_file_write("write", {"file_path": fp}, "wrote")
    assert warning is not None
    assert "syntax error" in warning
