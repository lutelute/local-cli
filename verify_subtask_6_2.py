#!/usr/bin/env python3
"""Subtask 6-2 verification script.

Verifies:
1. All new module imports work correctly
2. No external dependencies were introduced
3. Backward compatibility: old sessions load without errors
4. Existing tool execution unaffected by cacheable property
5. All new slash commands are registered
"""

import ast
import json
import os
import pathlib
import sys
import tempfile

PASS = 0
FAIL = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {label}")
    else:
        FAIL += 1
        msg = f"  ✗ {label}"
        if detail:
            msg += f" — {detail}"
        print(msg)


def test_imports() -> None:
    """Test 1: Verify all new module imports work correctly."""
    print("\n=== Test 1: New Module Imports ===")

    try:
        from local_cli.clipboard import copy_to_clipboard
        check("clipboard.copy_to_clipboard", True)
    except Exception as e:
        check("clipboard.copy_to_clipboard", False, str(e))

    try:
        from local_cli.clipboard import ClipboardError, ClipboardUnavailableError
        check("clipboard.ClipboardError + ClipboardUnavailableError", True)
    except Exception as e:
        check("clipboard.ClipboardError + ClipboardUnavailableError", False, str(e))

    try:
        from local_cli.health_check import run_health_check, format_health_check
        check("health_check.run_health_check + format_health_check", True)
    except Exception as e:
        check("health_check.run_health_check + format_health_check", False, str(e))

    try:
        from local_cli.health_check import (
            check_ollama_connectivity,
            check_model_availability,
            check_disk_space,
            CheckResult,
            STATUS_OK,
            STATUS_WARNING,
            STATUS_ERROR,
        )
        check("health_check sub-imports", True)
    except Exception as e:
        check("health_check sub-imports", False, str(e))

    try:
        from local_cli.diff_preview import generate_diff, generate_multi_file_diff
        check("diff_preview.generate_diff + generate_multi_file_diff", True)
    except Exception as e:
        check("diff_preview.generate_diff + generate_multi_file_diff", False, str(e))

    try:
        from local_cli.diff_preview import (
            colorize_diff,
            truncate_diff,
            format_diff_output,
            is_binary_content,
            DiffPreviewError,
        )
        check("diff_preview sub-imports", True)
    except Exception as e:
        check("diff_preview sub-imports", False, str(e))

    try:
        from local_cli.token_tracker import TokenTracker, TokenUsage
        check("token_tracker.TokenTracker + TokenUsage", True)
    except Exception as e:
        check("token_tracker.TokenTracker + TokenUsage", False, str(e))

    try:
        from local_cli.tool_cache import ToolCache
        check("tool_cache.ToolCache", True)
    except Exception as e:
        check("tool_cache.ToolCache", False, str(e))

    # Verify instantiation works
    try:
        from local_cli.token_tracker import TokenTracker
        t = TokenTracker()
        check("TokenTracker() instantiation", True)
    except Exception as e:
        check("TokenTracker() instantiation", False, str(e))

    try:
        from local_cli.tool_cache import ToolCache
        c = ToolCache()
        check("ToolCache() instantiation", True)
    except Exception as e:
        check("ToolCache() instantiation", False, str(e))


def test_no_external_deps() -> None:
    """Test 2: Verify no external dependencies were introduced."""
    print("\n=== Test 2: Stdlib-Only Check ===")

    stdlib_prefixes = (
        "local_cli", "os", "sys", "json", "pathlib", "subprocess", "tempfile",
        "unittest", "abc", "argparse", "readline", "hashlib", "math", "sqlite3",
        "uuid", "datetime", "typing", "collections", "difflib", "shutil", "re",
        "io", "socket", "threading", "urllib", "time", "textwrap", "platform",
        "struct", "signal", "_thread", "__future__", "functools", "dataclasses",
        "enum", "contextlib", "copy", "string", "logging", "traceback",
        "importlib", "inspect", "glob", "fnmatch", "http", "email", "base64",
        "binascii", "secrets", "stat", "errno", "html", "curses", "ssl",
    )

    externals = []
    for f in sorted(pathlib.Path("local_cli").rglob("*.py")):
        try:
            tree = ast.parse(f.read_text())
        except SyntaxError:
            check(f"Parse {f}", False, "SyntaxError")
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if not alias.name.startswith(stdlib_prefixes):
                        externals.append(f"{alias.name} in {f}")
            elif isinstance(node, ast.ImportFrom):
                if node.module and not node.module.startswith(stdlib_prefixes):
                    externals.append(f"{node.module} in {f}")

    if externals:
        for ext in externals:
            check(f"No external: {ext}", False)
    else:
        check("No external dependencies in local_cli/", True)

    # Also check new test files
    new_test_files = [
        "tests/test_clipboard.py",
        "tests/test_diff_preview.py",
        "tests/test_token_tracker.py",
        "tests/test_tool_cache.py",
        "tests/test_health_check.py",
        "tests/test_undo.py",
        "tests/test_context_command.py",
        "tests/test_usage_command.py",
    ]
    test_externals = []
    test_stdlib = stdlib_prefixes + ("unittest",)
    for tf in new_test_files:
        p = pathlib.Path(tf)
        if not p.exists():
            check(f"Test file exists: {tf}", False)
            continue
        try:
            tree = ast.parse(p.read_text())
        except SyntaxError:
            check(f"Parse {tf}", False, "SyntaxError")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if not alias.name.startswith(test_stdlib):
                        test_externals.append(f"{alias.name} in {tf}")
            elif isinstance(node, ast.ImportFrom):
                if node.module and not node.module.startswith(test_stdlib):
                    test_externals.append(f"{node.module} in {tf}")

    if test_externals:
        for ext in test_externals:
            check(f"No external in tests: {ext}", False)
    else:
        check("No external dependencies in new test files", True)


def test_backward_compat_sessions() -> None:
    """Test 3: Verify old sessions load without errors."""
    print("\n=== Test 3: Session Backward Compatibility ===")

    from local_cli.session import SessionManager

    with tempfile.TemporaryDirectory() as tmpdir:
        # SessionManager expects a state_dir and creates sessions/ subdir
        sm = SessionManager(tmpdir)

        # Old-style messages (no token_usage metadata)
        old_messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]

        # Save old-style session (no token_tracker)
        try:
            session_id = sm.save_session(old_messages)
            check("Save old-style session (no token_tracker)", True)
        except Exception as e:
            check("Save old-style session (no token_tracker)", False, str(e))
            return

        # Load it back
        try:
            loaded = sm.load_session(session_id)
            check("Load old-style session", True)
            check(
                "Old session message count",
                len(loaded) == len(old_messages),
                f"expected {len(old_messages)}, got {len(loaded)}",
            )
            check(
                "Old session content intact",
                loaded[1]["content"] == "Hello"
                and loaded[4]["content"] == "4",
            )
        except Exception as e:
            check("Load old-style session", False, str(e))

        # Save new-style session with token_tracker
        try:
            from local_cli.token_tracker import TokenTracker, TokenUsage
            tracker = TokenTracker()
            tracker.record(TokenUsage(input_tokens=100, output_tokens=50, provider="ollama"))
            tracker.record(TokenUsage(input_tokens=200, output_tokens=100, provider="ollama"))

            new_session_id = sm.save_session(
                old_messages, token_tracker=tracker
            )
            check("Save new-style session with token data", True)

            loaded_new = sm.load_session(new_session_id)
            check(
                "Load new-style session",
                len(loaded_new) == len(old_messages),
            )
        except Exception as e:
            check("Save/load new-style session", False, str(e))

        # Manually create an old JSONL session file (pre-token_usage era) and load it
        try:
            sessions_subdir = os.path.join(tmpdir, "sessions")
            old_session_file = os.path.join(sessions_subdir, "old_format_test.jsonl")
            with open(old_session_file, "w") as f:
                for msg in old_messages:
                    f.write(json.dumps(msg) + "\n")

            loaded_old = sm.load_session("old_format_test")
            check(
                "Load manually-created old-format JSONL",
                len(loaded_old) == len(old_messages),
            )
        except Exception as e:
            check("Load manually-created old-format JSONL", False, str(e))


def test_cacheable_property() -> None:
    """Test 4: Verify cacheable property doesn't affect existing tools."""
    print("\n=== Test 4: Tool Cacheable Property ===")

    from local_cli.tools.read_tool import ReadTool
    from local_cli.tools.glob_tool import GlobTool
    from local_cli.tools.grep_tool import GrepTool
    from local_cli.tools.bash_tool import BashTool
    from local_cli.tools.write_tool import WriteTool
    from local_cli.tools.edit_tool import EditTool

    # Cacheable tools
    check("ReadTool.cacheable == True", ReadTool().cacheable is True)
    check("GlobTool.cacheable == True", GlobTool().cacheable is True)
    check("GrepTool.cacheable == True", GrepTool().cacheable is True)

    # Non-cacheable tools
    check("BashTool.cacheable == False", BashTool().cacheable is False)
    check("WriteTool.cacheable == False", WriteTool().cacheable is False)
    check("EditTool.cacheable == False", EditTool().cacheable is False)

    # Verify all tool properties still work
    tool_classes = [ReadTool, GlobTool, GrepTool, BashTool, WriteTool, EditTool]
    for ToolClass in tool_classes:
        t = ToolClass()
        name_ok = isinstance(t.name, str) and len(t.name) > 0
        desc_ok = isinstance(t.description, str) and len(t.description) > 0
        params_ok = isinstance(t.parameters, dict)
        cache_ok = isinstance(t.cacheable, bool)
        all_ok = name_ok and desc_ok and params_ok and cache_ok
        check(
            f"{ToolClass.__name__} properties intact (name, description, parameters, cacheable)",
            all_ok,
        )


def test_slash_commands() -> None:
    """Test 5: Verify all new slash commands are registered."""
    print("\n=== Test 5: Slash Command Registration ===")

    from local_cli.cli import _SLASH_COMMANDS

    new_commands = ["/undo", "/diff", "/usage", "/context", "/copy"]
    for cmd in new_commands:
        check(
            f"{cmd} registered",
            cmd in _SLASH_COMMANDS,
            f"missing from _SLASH_COMMANDS" if cmd not in _SLASH_COMMANDS else "",
        )
        if cmd in _SLASH_COMMANDS:
            check(
                f"{cmd} has description",
                isinstance(_SLASH_COMMANDS[cmd], str) and len(_SLASH_COMMANDS[cmd]) > 0,
            )

    check(f"Total slash commands: {len(_SLASH_COMMANDS)}", len(_SLASH_COMMANDS) >= 10)


def test_cross_module_integration() -> None:
    """Test 6: Verify cross-module integration points."""
    print("\n=== Test 6: Cross-Module Integration ===")

    # Verify agent.py accepts cache and tracker parameters
    try:
        import inspect
        from local_cli.agent import agent_loop
        sig = inspect.signature(agent_loop)
        check("agent_loop has 'cache' parameter", "cache" in sig.parameters)
        check("agent_loop has 'tracker' parameter", "tracker" in sig.parameters)

        # Check defaults are None (backward compatible)
        cache_param = sig.parameters.get("cache")
        tracker_param = sig.parameters.get("tracker")
        if cache_param:
            check(
                "agent_loop cache default is None",
                cache_param.default is None,
            )
        if tracker_param:
            check(
                "agent_loop tracker default is None",
                tracker_param.default is None,
            )
    except Exception as e:
        check("agent_loop parameter inspection", False, str(e))

    # Verify collect_streaming_response accepts tracker
    try:
        from local_cli.agent import collect_streaming_response
        sig = inspect.signature(collect_streaming_response)
        check(
            "collect_streaming_response has 'tracker' parameter",
            "tracker" in sig.parameters,
        )
    except Exception as e:
        check("collect_streaming_response inspection", False, str(e))

    # Verify _ReplContext has new slots
    try:
        from local_cli.cli import _ReplContext
        ctx_slots = _ReplContext.__slots__
        check("_ReplContext has 'token_tracker' slot", "token_tracker" in ctx_slots)
        check("_ReplContext has 'tool_cache' slot", "tool_cache" in ctx_slots)
    except Exception as e:
        check("_ReplContext slot inspection", False, str(e))

    # Verify session.save_session accepts token_tracker
    try:
        from local_cli.session import SessionManager
        sig = inspect.signature(SessionManager.save_session)
        check(
            "save_session has 'token_tracker' parameter",
            "token_tracker" in sig.parameters,
        )
    except Exception as e:
        check("save_session parameter inspection", False, str(e))


def test_new_test_files_exist() -> None:
    """Test 7: Verify all expected test files exist."""
    print("\n=== Test 7: New Test Files Exist ===")

    expected_files = [
        "tests/test_clipboard.py",
        "tests/test_diff_preview.py",
        "tests/test_token_tracker.py",
        "tests/test_tool_cache.py",
        "tests/test_health_check.py",
        "tests/test_undo.py",
        "tests/test_context_command.py",
        "tests/test_usage_command.py",
    ]

    for f in expected_files:
        exists = pathlib.Path(f).exists()
        check(f"{f} exists", exists)
        if exists:
            content = pathlib.Path(f).read_text()
            has_tests = "class Test" in content and "def test_" in content
            check(f"{f} has test classes and methods", has_tests)


def test_new_source_files_exist() -> None:
    """Test 8: Verify all expected source files exist."""
    print("\n=== Test 8: New Source Files Exist ===")

    expected_files = [
        "local_cli/clipboard.py",
        "local_cli/health_check.py",
        "local_cli/diff_preview.py",
        "local_cli/token_tracker.py",
        "local_cli/tool_cache.py",
    ]

    for f in expected_files:
        exists = pathlib.Path(f).exists()
        check(f"{f} exists", exists)
        if exists:
            content = pathlib.Path(f).read_text()
            has_docstring = '"""' in content or "'''" in content
            check(f"{f} has docstrings", has_docstring)


if __name__ == "__main__":
    print("=" * 60)
    print("Subtask 6-2: Integration Verification")
    print("=" * 60)

    test_imports()
    test_no_external_deps()
    test_backward_compat_sessions()
    test_cacheable_property()
    test_slash_commands()
    test_cross_module_integration()
    test_new_test_files_exist()
    test_new_source_files_exist()

    print("\n" + "=" * 60)
    print(f"RESULTS: {PASS} passed, {FAIL} failed")
    print("=" * 60)

    sys.exit(1 if FAIL > 0 else 0)
