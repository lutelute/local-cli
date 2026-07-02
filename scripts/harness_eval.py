#!/usr/bin/env python3
"""Harness evaluation: drive real local models through agentic tasks.

Measures what the deterministic harness actually delivers on small
models: task success rate, harness intervention counts, iterations,
tool calls, and wall time.  Each task runs in a fresh temp directory
with the same tool set, system prompt, and inference options the CLI
uses (minus interactive tools).

Usage (from the repo root):

    python3 scripts/harness_eval.py --models qwen3:0.6b qwen3.5:4b
    python3 scripts/harness_eval.py --models qwen3:0.6b --timeout 120 \
        --out eval_results.json

Requires a running Ollama with the requested models installed.
"""

import argparse
import ast
import contextlib
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Allow running as `python3 scripts/harness_eval.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from local_cli.agent import run_agent  # noqa: E402
from local_cli.harness import AgentEvent, HarnessConfig  # noqa: E402
from local_cli.model_presets import (  # noqa: E402
    SUPPORTS_THINKING,
    get_model_family,
)
from local_cli.ollama_client import OllamaClient  # noqa: E402
from local_cli.prompts import build_system_prompt  # noqa: E402
from local_cli.tools import get_default_tools  # noqa: E402

# ---------------------------------------------------------------------------
# Tasks: (setup, prompt, machine check)
# ---------------------------------------------------------------------------


def _check_create(workdir: Path) -> tuple[bool, str]:
    path = workdir / "greet.py"
    if not path.is_file():
        return False, "greet.py not created"
    try:
        ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        return False, f"syntax error: {exc}"
    return True, "ok"


def _setup_edit(workdir: Path) -> None:
    (workdir / "app.py").write_text(
        "def add(a, b):\n    return a - b\n", encoding="utf-8",
    )


def _check_edit(workdir: Path) -> tuple[bool, str]:
    namespace: dict = {}
    try:
        # Suppress stray prints in case the model rewrote app.py into a
        # script (observed live).
        with contextlib.redirect_stdout(io.StringIO()):
            exec((workdir / "app.py").read_text(encoding="utf-8"), namespace)
        result = namespace["add"](2, 3)
    except Exception as exc:
        return False, f"app.py broken: {exc}"
    if result != 5:
        return False, f"add(2, 3) returned {result}, expected 5"
    return True, "ok"


def _setup_syntax(workdir: Path) -> None:
    (workdir / "broken.py").write_text(
        "def greet(name:\n"
        "    print('hi ' + name)\n"
        "\n"
        "greet('world')\n",
        encoding="utf-8",
    )


def _check_syntax(workdir: Path) -> tuple[bool, str]:
    try:
        ast.parse((workdir / "broken.py").read_text(encoding="utf-8"))
    except SyntaxError as exc:
        return False, f"still broken: {exc}"
    return True, "ok"


def _check_multi(workdir: Path) -> tuple[bool, str]:
    one = workdir / "one.txt"
    two = workdir / "two.txt"
    if not one.is_file():
        return False, "one.txt missing"
    if not two.is_file():
        return False, "two.txt missing"
    if "first" not in one.read_text(encoding="utf-8"):
        return False, "one.txt content wrong"
    if "second" not in two.read_text(encoding="utf-8"):
        return False, "two.txt content wrong"
    return True, "ok"


def _setup_search_fix(workdir: Path) -> None:
    """A small project: the bug is in one of several files."""
    src = workdir / "src"
    src.mkdir()
    (src / "utils.py").write_text(
        "def clamp(value, low, high):\n"
        "    return max(low, min(high, value))\n"
        "\n"
        "\n"
        "def calc_total(items):\n"
        "    total = 1\n"
        "    for item in items:\n"
        "        total *= item\n"
        "    return total\n",
        encoding="utf-8",
    )
    (src / "main.py").write_text(
        "from utils import calc_total\n"
        "\n"
        "print(calc_total([1, 2, 3]))\n",
        encoding="utf-8",
    )
    (workdir / "README.md").write_text(
        "# demo project\n\nutilities live under src/\n", encoding="utf-8",
    )


def _check_search_fix(workdir: Path) -> tuple[bool, str]:
    namespace: dict = {}
    path = workdir / "src" / "utils.py"
    if not path.is_file():
        return False, "src/utils.py missing"
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(path.read_text(encoding="utf-8"), namespace)
        result = namespace["calc_total"]([2, 3])
    except Exception as exc:
        return False, f"utils.py broken: {exc}"
    if result != 5:
        return False, f"calc_total([2, 3]) returned {result}, expected 5"
    return True, "ok"


def _setup_test_fix(workdir: Path) -> None:
    (workdir / "calc.py").write_text(
        "def mul(a, b):\n    return a + b\n", encoding="utf-8",
    )
    (workdir / "test_calc.py").write_text(
        "from calc import mul\n"
        "\n"
        "assert mul(3, 4) == 12, f'mul(3, 4) = {mul(3, 4)}, expected 12'\n"
        "print('test passed')\n",
        encoding="utf-8",
    )


def _check_test_fix(workdir: Path) -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, "test_calc.py"],
        cwd=str(workdir), capture_output=True, text=True, timeout=30,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip().splitlines()
        return False, f"test still failing: {detail[-1] if detail else '?'}"
    return True, "ok"


def _check_todo_multi(workdir: Path) -> tuple[bool, str]:
    expected = {"a.txt": "alpha", "b.txt": "beta", "c.txt": "gamma"}
    for name, word in expected.items():
        path = workdir / name
        if not path.is_file():
            return False, f"{name} missing"
        if word not in path.read_text(encoding="utf-8"):
            return False, f"{name} content wrong"
    return True, "ok"


TASKS = [
    {
        "name": "create_file",
        "prompt": "Create a file called greet.py that prints 'hello'.",
        "setup": None,
        "check": _check_create,
    },
    {
        "name": "edit_file",
        "prompt": (
            "In app.py, the add function is wrong: it subtracts. "
            "Fix it so add returns a + b."
        ),
        "setup": _setup_edit,
        "check": _check_edit,
    },
    {
        "name": "fix_syntax",
        "prompt": "broken.py has a syntax error. Find it and fix it.",
        "setup": _setup_syntax,
        "check": _check_syntax,
    },
    {
        "name": "multi_file",
        "prompt": (
            "Create two files: one.txt containing the word 'first' and "
            "two.txt containing the word 'second'."
        ),
        "setup": None,
        "check": _check_multi,
    },
    {
        "name": "search_fix",
        "prompt": (
            "Somewhere in this project, calc_total is wrong: "
            "calc_total([2, 3]) should return 5 but returns 6. Find the "
            "function and fix it."
        ),
        "setup": _setup_search_fix,
        "check": _check_search_fix,
    },
    {
        "name": "test_fix",
        "prompt": (
            "Run test_calc.py with python. It fails. Fix calc.py so the "
            "test passes, and run it again to confirm."
        ),
        "setup": _setup_test_fix,
        "check": _check_test_fix,
    },
    {
        "name": "todo_multi",
        "prompt": (
            "Create three files: a.txt containing 'alpha', b.txt "
            "containing 'beta', and c.txt containing 'gamma'. Use the "
            "todo_write tool to track your progress through the three "
            "files."
        ),
        "setup": None,
        "check": _check_todo_multi,
    },
]

# Harness intervention events worth counting.
HARNESS_EVENTS = (
    "rescue",
    "tools_fallback",
    "loop_warning",
    "loop_break",
    "verify_warning",
    "nudge",
    "error_stop",
    "reminder",
    "limit",
    "retry",
)


class _TaskTimeout(Exception):
    pass


def _run_task(
    client: OllamaClient,
    model: str,
    task: dict,
    timeout_s: int,
    max_iterations: int,
) -> dict:
    """Run one task against one model and machine-check the result."""
    workdir = Path(tempfile.mkdtemp(prefix=f"heval_{task['name']}_"))
    old_cwd = os.getcwd()
    os.chdir(workdir)

    counts: dict[str, int] = {}
    iterations = 0
    tool_calls = 0

    def emit(event: AgentEvent) -> None:
        nonlocal iterations, tool_calls
        if event.kind == "llm_start":
            iterations = event.data.get("iteration", iterations)
        elif event.kind == "tool_result":
            tool_calls += 1
        if event.kind in HARNESS_EVENTS:
            counts[event.kind] = counts.get(event.kind, 0) + 1

    # Interactive/spawning tools are excluded (no stdin; keep runs flat).
    tools = [
        t for t in get_default_tools() if t.name not in ("ask_user", "agent")
    ]
    if task["setup"] is not None:
        task["setup"](workdir)

    messages = [
        {"role": "system", "content": build_system_prompt(tools)},
        {"role": "user", "content": task["prompt"]},
    ]

    family = get_model_family(model)
    think = False if family in SUPPORTS_THINKING else None

    def _on_alarm(signum, frame):  # noqa: ANN001
        raise _TaskTimeout()

    signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(timeout_s)
    start = time.monotonic()
    error = ""
    try:
        run_agent(
            client, model, tools, messages,
            emit=emit,
            harness=HarnessConfig(max_iterations=max_iterations),
            # Deterministic decoding so intervention changes are A/B
            # comparable across eval runs.
            options={"num_ctx": 8192, "temperature": 0, "seed": 42},
            think=think,
        )
    except _TaskTimeout:
        error = f"timeout after {timeout_s}s"
    except Exception as exc:  # noqa: BLE001 — record, don't crash the sweep
        error = f"{type(exc).__name__}: {exc}"
    finally:
        signal.alarm(0)
        os.chdir(old_cwd)

    elapsed = time.monotonic() - start
    ok, detail = task["check"](workdir)

    return {
        "task": task["name"],
        "success": ok,
        "detail": detail,
        "error": error,
        "seconds": round(elapsed, 1),
        "iterations": iterations,
        "tool_calls": tool_calls,
        "interventions": counts,
        "workdir": str(workdir),
        # Full conversation (minus the system prompt) for failure analysis.
        "transcript": messages[1:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--timeout", type=int, default=120,
                        help="per-task timeout in seconds")
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--out", default="",
                        help="write full JSON results to this path")
    parser.add_argument("--tasks", nargs="*", default=[],
                        help="task names to run (default: all)")
    args = parser.parse_args()

    tasks = TASKS
    if args.tasks:
        tasks = [t for t in TASKS if t["name"] in set(args.tasks)]

    client = OllamaClient()
    results: dict[str, list[dict]] = {}

    for model in args.models:
        results[model] = []
        for task in tasks:
            print(f"[{model}] {task['name']} ...", flush=True)
            record = _run_task(
                client, model, task, args.timeout, args.max_iterations,
            )
            status = "PASS" if record["success"] else "FAIL"
            interventions = ", ".join(
                f"{k}x{v}" for k, v in sorted(record["interventions"].items())
            ) or "-"
            print(
                f"[{model}] {task['name']}: {status} "
                f"({record['seconds']}s, {record['iterations']} iters, "
                f"{record['tool_calls']} calls; {interventions})"
                + (f" error={record['error']}" if record["error"] else "")
                + (f" detail={record['detail']}" if not record["success"] else ""),
                flush=True,
            )
            results[model].append(record)

    # Summary table.
    print("\n=== SUMMARY ===")
    for model, records in results.items():
        passed = sum(1 for r in records if r["success"])
        total_interventions: dict[str, int] = {}
        for r in records:
            for k, v in r["interventions"].items():
                total_interventions[k] = total_interventions.get(k, 0) + v
        summary = ", ".join(
            f"{k}x{v}" for k, v in sorted(total_interventions.items())
        ) or "none"
        print(f"{model}: {passed}/{len(records)} passed | interventions: {summary}")

    if args.out:
        Path(args.out).write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nFull results: {args.out}")


if __name__ == "__main__":
    main()
