"""Tests for text tool-call rescue of the "tool-name-as-key" shape.

Some small models without structured tool-calling emit a call as a
single-key object whose key is the tool name and whose value is the
arguments — ``{"write": {"file_path": "a.py", "content": "..."}}`` —
rather than ``{"name": "write", "arguments": {...}}``.  The rescue must
recover these, gated by the resolver so unknown keys are not mistaken
for calls.
"""

from local_cli.harness import extract_text_tool_calls


def _resolver(known):
    return lambda name: name if name in known else None


def test_toolname_as_key_is_rescued():
    calls = extract_text_tool_calls(
        '{"write": {"file_path": "a.py", "content": "print(1)"}}',
        _resolver({"write"}),
    )
    assert len(calls) == 1
    fn = calls[0]["function"]
    assert fn["name"] == "write"
    assert fn["arguments"] == {"file_path": "a.py", "content": "print(1)"}


def test_unknown_key_is_not_mistaken_for_a_call():
    # A single-key object whose key is not a real tool must be rejected
    # (this is the false-positive guard via the resolver).
    calls = extract_text_tool_calls(
        '{"result": {"value": 42}}',
        _resolver({"write", "read", "bash"}),
    )
    assert calls == []


def test_multi_key_object_not_treated_as_namekey():
    # More than one key means it is not the tool-name-as-key shape; the
    # normal name/args detection applies (here no name key -> no rescue).
    calls = extract_text_tool_calls(
        '{"write": {"file_path": "a.py"}, "note": "hi"}',
        _resolver({"write"}),
    )
    assert calls == []


def test_namekey_value_must_be_dict():
    # {"write": "stuff"} is not a call (value is not an arguments dict).
    calls = extract_text_tool_calls(
        '{"write": "just some text"}',
        _resolver({"write"}),
    )
    assert calls == []


def test_explicit_name_key_still_wins():
    # The canonical shape is unaffected by the new fallback.
    calls = extract_text_tool_calls(
        '{"name": "read", "arguments": {"file_path": "b.py"}}',
        _resolver({"read"}),
    )
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "read"
    assert calls[0]["function"]["arguments"] == {"file_path": "b.py"}
