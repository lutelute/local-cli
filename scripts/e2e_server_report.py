#!/usr/bin/env python3
"""E2E: drive the real JSON-line server (the desktop path) end to end.

Spawns `python -m local_cli --server` in a fresh temp project, sends the
field report task over the wire, and checks what a GUI would have
received: stream chars (the visible answer), thinking chars, harness
events, and whether a .md landed on disk.

This is the server-layer counterpart to scripts/repro_report.py — run
both to localize a field failure: repro_report.py exercises run_agent
directly; this script exercises the full server protocol on top of it.

    python3 scripts/e2e_server_report.py --model qwen3.5:9b-q4_K_M

Exit code 0 = a .md was created AND the visible answer is non-empty.
Requires a running Ollama with the model installed.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

FIELD_PROMPT = (
    "このフォルダのファイルを読んで、プロジェクトの内容をまとめた報告書を "
    "mdファイルで作成してください。"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default=FIELD_PROMPT)
    parser.add_argument("--timeout", type=int, default=300,
                        help="wall-clock limit in seconds")
    args = parser.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="e2e_server_"))
    (workdir / "app.py").write_text(
        "def add(a, b):\n    return a + b\n\nprint(add(2, 3))\n",
        encoding="utf-8")
    (workdir / "utils.py").write_text(
        "def clamp(v, lo, hi):\n    return max(lo, min(hi, v))\n",
        encoding="utf-8")
    (workdir / "README.txt").write_text(
        "demo project: small math utilities\n", encoding="utf-8")

    env = {**os.environ}
    env["PYTHONPATH"] = str(REPO) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    proc = subprocess.Popen(
        [sys.executable, "-m", "local_cli", "--server", "--model", args.model],
        cwd=str(workdir),
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, env=env,
    )

    types_seen: dict[str, int] = {}
    stream_chars = 0
    thinking_chars = 0
    harness_events: list[str] = []
    done = False

    def reader() -> None:
        nonlocal stream_chars, thinking_chars, done
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            t = msg.get("type", "?")
            types_seen[t] = types_seen.get(t, 0) + 1
            if t == "ready":
                proc.stdin.write(json.dumps(
                    {"id": 1, "type": "chat", "content": args.prompt}) + "\n")
                proc.stdin.flush()
            elif t == "stream":
                stream_chars += len(msg.get("content", ""))
            elif t == "thinking":
                thinking_chars += len(msg.get("content", ""))
            elif t == "tool_call":
                print(f"  [tool_call] {msg.get('name')}")
            elif t == "harness":
                harness_events.append(msg.get("event", "?"))
                print(f"  [harness] {msg.get('event')}")
            elif t in ("done", "stopped", "error"):
                if t == "error":
                    print(f"  [error] {msg.get('message', '')[:200]}")
                done = True
                return

    print(f"model={args.model} workdir={workdir}")
    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    deadline = time.monotonic() + args.timeout
    while not done and time.monotonic() < deadline:
        time.sleep(0.5)
    proc.terminate()

    md_files = [p.name for p in workdir.glob("*.md")]
    print("\n=== E2E RESULT ===")
    print(f"types_seen: {types_seen}")
    print(f"stream_chars(final visible answer): {stream_chars}")
    print(f"thinking_chars(surfaced to GUI): {thinking_chars}")
    print(f"harness_events: {harness_events or '-'}")
    print(f"md_created: {md_files or 'NONE'}")
    print(f"timed_out: {not done}")
    verdict = bool(md_files) and stream_chars > 0
    print(f"VERDICT: {'PASS' if verdict else 'FAIL'} "
          f"(md exists AND visible answer non-empty)")
    sys.exit(0 if verdict else 1)


if __name__ == "__main__":
    main()
