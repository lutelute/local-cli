"""Tests for text tool-call rescue of inlined (top-level) arguments.

Some small models emit a tool call as a flat JSON object with the
arguments as top-level keys — ``{"name": "write", "file_path": "a.py",
"content": "..."}`` — instead of nesting them under ``"arguments"``.
The rescue must recover those arguments rather than dropping them, so
the tool actually receives what the model intended.
"""

from local_cli.harness import extract_text_tool_calls


def _resolver(known):
    return lambda name: name if name in known else None


def test_inlined_args_are_recovered():
    calls = extract_text_tool_calls(
        '{"name": "write", "file_path": "a.py", "content": "print(1)"}',
        _resolver({"write"}),
    )
    assert len(calls) == 1
    fn = calls[0]["function"]
    assert fn["name"] == "write"
    assert fn["arguments"] == {"file_path": "a.py", "content": "print(1)"}


def test_nested_args_still_work():
    calls = extract_text_tool_calls(
        '{"name": "read", "arguments": {"file_path": "b.py"}}',
        _resolver({"read"}),
    )
    assert len(calls) == 1
    fn = calls[0]["function"]
    assert fn["name"] == "read"
    assert fn["arguments"] == {"file_path": "b.py"}


def test_alt_args_key_takes_precedence_over_inlining():
    # When an explicit args key exists, it wins; top-level keys are not
    # scavenged (avoids double-counting).
    calls = extract_text_tool_calls(
        '{"tool": "bash", "args": {"command": "ls"}, "stray": "ignore"}',
        _resolver({"bash"}),
    )
    assert len(calls) == 1
    assert calls[0]["function"]["arguments"] == {"command": "ls"}


def test_name_only_still_empty_args():
    # A bare name with no arguments anywhere stays empty (nothing to scavenge).
    calls = extract_text_tool_calls(
        '{"name": "glob"}',
        _resolver({"glob"}),
    )
    assert len(calls) == 1
    assert calls[0]["function"]["arguments"] == {}


def test_inlined_unknown_tool_is_rejected():
    # Name must resolve to a real tool; otherwise no rescue.
    calls = extract_text_tool_calls(
        '{"name": "frobnicate", "x": 1}',
        _resolver({"write", "read"}),
    )
    assert calls == []
