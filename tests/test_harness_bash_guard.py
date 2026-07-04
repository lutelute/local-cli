"""Tests for the bash exit-code failure guard.

The bash tool reports a failing command with an ``[exit code: N]`` marker
rather than an ``Error:`` prefix, so ``last_tool_result_errored`` must
recognise a non-zero exit as a failure the model should not finish on.
Without this, a small model can run a red test, watch it fail, and still
declare the task done.
"""

from local_cli.harness import _bash_exit_failed, last_tool_result_errored


def test_bash_exit_failed_nonzero():
    assert _bash_exit_failed("boom\n[exit code: 1]") is True
    assert _bash_exit_failed("x\n[exit code: 2]") is True
    assert _bash_exit_failed("[exit code: 127]") is True


def test_bash_exit_failed_zero_or_absent():
    assert _bash_exit_failed("ok\n[exit code: 0]") is False
    assert _bash_exit_failed("(no output)") is False
    assert _bash_exit_failed("passed") is False
    assert _bash_exit_failed("") is False


def test_guard_fires_on_failing_bash():
    messages = [
        {"role": "user", "content": "run the tests"},
        {
            "role": "tool",
            "content": "AssertionError: 7 != 12\n[stderr]\nTraceback\n[exit code: 1]",
            "tool_name": "bash",
        },
        {"role": "assistant", "content": "The tests pass now."},
    ]
    assert last_tool_result_errored(messages) is True


def test_guard_silent_on_passing_bash():
    messages = [
        {"role": "user", "content": "run it"},
        {"role": "tool", "content": "(no output)", "tool_name": "bash"},
        {"role": "assistant", "content": "done"},
    ]
    assert last_tool_result_errored(messages) is False


def test_guard_ignores_plain_warning_word_in_bash():
    # A stray "WARNING" in build output (a bash result, not write/edit)
    # must not fire — only a non-zero exit code does.
    messages = [
        {"role": "user", "content": "build"},
        {"role": "tool", "content": "build WARNING: deprecated", "tool_name": "bash"},
        {"role": "assistant", "content": "done"},
    ]
    assert last_tool_result_errored(messages) is False
