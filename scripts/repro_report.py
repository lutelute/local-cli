#!/usr/bin/env python3
"""Field repro: "read this folder, write a report .md" — layer diagnosis.

Drives the real agent loop (run_agent) against a real Ollama model in a
fresh temp project, then reports what actually happened: tool calls,
files created, whether any .md exists, which harness interventions
fired, and the size of the final chat answer.

The --think flag is the harness/brain separator observed in the field:
the same model, task and machine flips between PASS and FAIL on this
flag alone, because thinking-default models swallow the reply into the
thinking channel when the flag is left unset.  A/B usage:

    python3 scripts/repro_report.py --model qwen3.5:9b-q4_K_M --think product
    python3 scripts/repro_report.py --model qwen3.5:9b-q4_K_M --think auto

If PASS/FAIL flips on the flag alone, the failure is harness-layer.
Exit code 0 = a .md was created AND the final answer is non-empty.
Requires a running Ollama with the model installed.
"""

import argparse
import json
import os
import signal
import sys
import tempfile
import time
from pathlib import Path

# Allow running as `python3 scripts/repro_report.py` from the repo root.
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

FIELD_PROMPT = (
    "このフォルダのファイルを読んで、プロジェクトの内容をまとめた報告書を "
    "mdファイルで作成してください。"
)


def _resolve_think(mode: str, model: str) -> bool | None:
    if mode == "on":
        return True
    if mode == "off":
        return False
    if mode == "auto":
        return None
    # "product": what the CLI/server ship — explicit False for thinking
    # families so the reply is not swallowed; unset otherwise.
    return False if get_model_family(model) in SUPPORTS_THINKING else None


class _Timeout(Exception):
    pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default=FIELD_PROMPT)
    parser.add_argument(
        "--think", choices=("product", "auto", "on", "off"), default="product",
        help="product = ship behavior (False for thinking families); "
             "auto = leave unset (pre-fix field behavior); on/off = force",
    )
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=300,
                        help="wall-clock limit in seconds")
    args = parser.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="repro_report_"))
    seed_files = {
        "app.py": "def add(a, b):\n    return a + b\n\nprint(add(2, 3))\n",
        "utils.py": "def clamp(v, lo, hi):\n    return max(lo, min(hi, v))\n",
        "README.txt": "demo project: small math utilities\n",
    }
    for name, text in seed_files.items():
        (workdir / name).write_text(text, encoding="utf-8")
    os.chdir(workdir)

    events: list[AgentEvent] = []

    def emit(ev: AgentEvent) -> None:
        events.append(ev)
        if ev.kind == "tool_start":
            arg_str = json.dumps(
                ev.data.get("arguments", {}), ensure_ascii=False)[:100]
            print(f"  [tool] {ev.data.get('tool_name')} {arg_str}")
        elif ev.kind in (
            "rescue", "nudge", "error_stop", "empty_response",
            "tools_fallback", "write_deferred", "deliverable_nudge",
            "read_gate", "loop_warning", "loop_break", "limit",
        ):
            print(f"  [harness] {ev.kind}")

    tools = [
        t for t in get_default_tools() if t.name not in ("ask_user", "agent")
    ]
    messages = [
        {"role": "system", "content": build_system_prompt(tools)},
        {"role": "user", "content": args.prompt},
    ]
    think = _resolve_think(args.think, args.model)

    print(f"model={args.model} think={args.think}({think}) workdir={workdir}")

    def _on_alarm(signum, frame):  # noqa: ANN001
        raise _Timeout()

    signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(args.timeout)
    start = time.monotonic()
    final = ""
    error = ""
    try:
        final = run_agent(
            OllamaClient(), args.model, tools, messages,
            emit=emit,
            harness=HarnessConfig(max_iterations=args.max_iterations),
            # Deterministic decoding so A/B runs differ only in the flag
            # under test.
            options={"num_ctx": 8192, "temperature": 0, "seed": 42},
            think=think,
        )
    except _Timeout:
        error = f"timeout after {args.timeout}s"
    finally:
        signal.alarm(0)
    elapsed = time.monotonic() - start

    md_files = [p.name for p in workdir.glob("*.md")]
    created = [p.name for p in workdir.iterdir() if p.name not in seed_files]
    kinds: dict[str, int] = {}
    for ev in events:
        if ev.kind not in ("thinking_delta", "content_delta"):
            kinds[ev.kind] = kinds.get(ev.kind, 0) + 1

    print(f"\n=== RESULT ({elapsed:.0f}s) ===")
    if error:
        print(f"error: {error}")
    print(f"md_created: {md_files or 'NONE'}")
    print(f"new_files: {created or 'NONE'}")
    print(f"events: {dict(sorted(kinds.items()))}")
    prose_dump = "YES" if len(final) > 400 and not md_files else "no"
    print(f"final_answer_chars: {len(final)} "
          f"(prose report dumped to chat instead of file? {prose_dump})")
    verdict = bool(md_files) and bool(final.strip())
    print(f"VERDICT: {'PASS' if verdict else 'FAIL'} "
          f"(md exists AND final answer non-empty)")
    sys.exit(0 if verdict else 1)


if __name__ == "__main__":
    main()
