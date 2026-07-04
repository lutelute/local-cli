"""Tests for the post-write TOML syntax verification gate.

The verify_writes harness gate syntax-checks a file right after a
write/edit so a small model hears about a broken file immediately.  It
already covered ``.py`` and ``.json``; this extends it to ``.toml`` (a
common config format), using stdlib ``tomllib`` when available.
"""

import pytest

from local_cli.harness import tomllib, verify_file_write

_HAS_TOMLLIB = tomllib is not None


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_valid_toml_passes(tmp_path):
    fp = _write(tmp_path, "cfg.toml", 'title = "demo"\n[owner]\nname = "x"\n')
    assert verify_file_write("write", {"file_path": fp}, "wrote") is None


@pytest.mark.skipif(not _HAS_TOMLLIB, reason="tomllib requires Python 3.11+")
def test_invalid_toml_warns(tmp_path):
    fp = _write(tmp_path, "bad.toml", "title = \n[unclosed\n")
    warning = verify_file_write("write", {"file_path": fp}, "wrote")
    assert warning is not None
    assert "TOML" in warning
    assert "bad.toml" in warning


def test_toml_only_for_write_edit(tmp_path):
    fp = _write(tmp_path, "bad.toml", "= nope\n")
    # A read result is not a mutation; nothing to verify.
    assert verify_file_write("read", {"file_path": fp}, "contents") is None


def test_toml_skipped_on_error_result(tmp_path):
    fp = _write(tmp_path, "bad.toml", "= nope\n")
    # The tool itself already failed; do not second-guess it.
    assert verify_file_write("write", {"file_path": fp}, "Error: boom") is None


def test_existing_json_verification_untouched(tmp_path):
    fp = _write(tmp_path, "broken.json", '{"a": 1,,}')
    warning = verify_file_write("write", {"file_path": fp}, "wrote")
    assert warning is not None
    assert "JSON" in warning
