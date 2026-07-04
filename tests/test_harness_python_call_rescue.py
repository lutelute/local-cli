"""Tests for text tool-call rescue of Python-style call syntax.

Some small models without structured tool-calling emit a call as Python
call syntax — ``write(file_path="a.py", content="print(1)")`` — rather
than JSON.  The rescue recovers keyword=constant calls, gated by the
resolver, and only as a fallback when no JSON-shaped call was found.
"""

from local_cli.harness import _scan_python_calls, extract_text_tool_calls


def _resolver(known):
    return lambda name: name if name in known else None


def test_python_call_with_kwargs_rescued():
    calls = extract_text_tool_calls(
        'write(file_path="a.py", content="print(1)")',
        _resolver({"write"}),
    )
    assert len(calls) == 1
    fn = calls[0]["function"]
    assert fn["name"] == "write"
    assert fn["arguments"] == {"file_path": "a.py", "content": "print(1)"}


def test_python_call_embedded_in_prose():
    calls = extract_text_tool_calls(
        'Sure — I will read(file_path="b.py") to check it.',
        _resolver({"read"}),
    )
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "read"
    assert calls[0]["function"]["arguments"] == {"file_path": "b.py"}


def test_string_with_parens_is_handled():
    # The balanced-paren scan is string-aware: parens inside a string
    # literal do not end the call early.
    calls = extract_text_tool_calls(
        'bash(command="echo (hello world)")',
        _resolver({"bash"}),
    )
    assert len(calls) == 1
    assert calls[0]["function"]["arguments"] == {"command": "echo (hello world)"}


def test_positional_only_call_not_rescued():
    # No keyword args -> too ambiguous to map to tool parameters.
    calls = extract_text_tool_calls('read("a.py")', _resolver({"read"}))
    assert calls == []


def test_non_constant_value_not_rescued():
    calls = extract_text_tool_calls(
        "write(file_path=some_variable)", _resolver({"write"})
    )
    assert calls == []


def test_unknown_tool_rejected_by_resolver():
    calls = extract_text_tool_calls(
        'frobnicate(x="1")', _resolver({"write", "read"})
    )
    assert calls == []


def test_json_call_takes_precedence_over_python_scan():
    # When a JSON-shaped call is present, it is used; the Python fallback
    # only runs if JSON rescue found nothing.
    calls = extract_text_tool_calls(
        '{"name": "read", "arguments": {"file_path": "b.py"}}',
        _resolver({"read", "write"}),
    )
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "read"


def test_scan_python_calls_unit():
    got = _scan_python_calls('write(file_path="a.py", content="x")')
    assert got == [("write", {"file_path": "a.py", "content": "x"})]
    assert _scan_python_calls("nothing here") == []
    assert _scan_python_calls("noop()") == []  # no kwargs
