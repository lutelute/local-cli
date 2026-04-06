"""Quick benchmark for local-cli models.

Measures speed, knowledge accuracy, and tool-calling ability.

Usage::

    local-cli --bench                # bench current model
    local-cli --bench --model X      # bench specific model
"""

import json
import time
from typing import Any

from local_cli.config import Config
from local_cli.providers import LLMProvider


# ---------------------------------------------------------------------------
# Benchmark questions
# ---------------------------------------------------------------------------

_KNOWLEDGE_QS: list[dict[str, str]] = [
    {
        "q": "What is the capital of France? Answer with just the city name.",
        "a": "Paris",
        "cat": "basic",
    },
    {
        "q": "日本で一番高い山は何ですか？山の名前だけ答えてください。",
        "a": "富士山",
        "cat": "japanese",
    },
    {
        "q": "What is 17 * 24? Answer with just the number.",
        "a": "408",
        "cat": "math",
    },
    {
        "q": "Who wrote Romeo and Juliet? Answer with just the name.",
        "a": "Shakespeare",
        "cat": "knowledge",
    },
    {
        "q": "水の化学式は何ですか？化学式だけ答えてください。",
        "a": "H2O",
        "cat": "japanese",
    },
    {
        "q": "What is the derivative of x^3? Answer in the form ax^b.",
        "a": "3x^2",
        "cat": "math",
    },
    {
        "q": "Sort these numbers ascending: 8, 2, 5, 1, 9. Answer as comma-separated.",
        "a": "1, 2, 5, 8, 9",
        "cat": "reasoning",
    },
    {
        "q": "光の速度はおよそ秒速何万キロメートルですか？数字だけ答えてください。",
        "a": "30",
        "cat": "japanese",
    },
]

_TOOL_QS: list[dict[str, Any]] = [
    {
        "q": "Read the file README.md",
        "expect_tool": "read",
        "expect_args": {"file_path": "README.md"},
    },
    {
        "q": "Find all Python files in the src directory",
        "expect_tool": "glob",
        "expect_args_contains": "*.py",
    },
    {
        "q": "Run the command 'echo hello'",
        "expect_tool": "bash",
        "expect_args": {"command": "echo hello"},
    },
]

_TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a file and return its contents",
            "parameters": {
                "type": "object",
                "properties": {"file_path": {"type": "string"}},
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find files matching a pattern",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def _check_answer(response: str, expected: str) -> bool:
    """Fuzzy check if response contains the expected answer."""
    import unicodedata
    # Normalize unicode (e.g. H₂O → H2O)
    r = unicodedata.normalize("NFKC", response).lower().strip()
    e = unicodedata.normalize("NFKC", expected).lower().strip()
    return e in r


def _check_tool_call(
    result: dict[str, Any], expect_tool: str, expect_args: dict | None = None,
    expect_args_contains: str | None = None,
) -> bool:
    """Check if the model made the correct tool call."""
    msg = result.get("message", {})
    tool_calls = msg.get("tool_calls", [])
    if not tool_calls:
        return False

    tc = tool_calls[0]
    func = tc.get("function", {})
    name = func.get("name", "")
    args = func.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (json.JSONDecodeError, ValueError):
            pass

    if name != expect_tool:
        return False

    if expect_args is not None:
        for k, v in expect_args.items():
            if str(args.get(k, "")).lower() != str(v).lower():
                return False

    if expect_args_contains is not None:
        args_str = json.dumps(args).lower()
        if expect_args_contains.lower() not in args_str:
            return False

    return True


def run_bench(provider: LLMProvider, model: str, verbose: bool = True) -> dict[str, Any]:
    """Run the benchmark suite.

    Returns a summary dict with scores and timing data.
    """
    results: dict[str, Any] = {
        "model": model,
        "provider": provider.name,
        "speed": {},
        "knowledge": {"total": 0, "correct": 0, "details": []},
        "tools": {"total": 0, "correct": 0, "details": []},
    }

    if verbose:
        print(f"\n  local-cli Benchmark")
        print(f"  Model: {model} | Provider: {provider.name}")
        print(f"  {'='*50}\n")

    # --- Speed benchmark ---
    if verbose:
        print("  [1/3] Speed Test")

    prompt = "Write a short paragraph about artificial intelligence in exactly 3 sentences."
    messages = [{"role": "user", "content": prompt}]

    # TTFT (Time to First Token)
    t0 = time.monotonic()
    first_token_time = None
    token_count = 0
    full_text = ""

    try:
        for chunk in provider.chat_stream(model, messages, max_tokens=256):
            msg = chunk.get("message", {})
            delta = msg.get("content", "")
            if delta:
                if first_token_time is None:
                    first_token_time = time.monotonic()
                token_count += 1
                full_text += delta
            if chunk.get("done"):
                break
    except Exception as exc:
        if verbose:
            print(f"    Speed test failed: {exc}")
        first_token_time = t0

    t_end = time.monotonic()
    ttft = (first_token_time - t0) if first_token_time else 0
    gen_time = t_end - (first_token_time or t_end)
    tps = token_count / gen_time if gen_time > 0 else 0

    results["speed"] = {
        "ttft_ms": round(ttft * 1000),
        "tokens": token_count,
        "gen_time_s": round(gen_time, 2),
        "tokens_per_sec": round(tps, 1),
    }

    if verbose:
        print(f"    TTFT:       {ttft*1000:.0f} ms")
        print(f"    Speed:      {tps:.1f} tok/s")
        print(f"    Tokens:     {token_count}")
        print()

    # --- Knowledge benchmark ---
    if verbose:
        print("  [2/3] Knowledge Test")

    for i, kq in enumerate(_KNOWLEDGE_QS):
        try:
            resp = provider.chat(
                model,
                [{"role": "user", "content": kq["q"]}],
                max_tokens=64,
            )
            answer = resp.get("message", {}).get("content", "")
            correct = _check_answer(answer, kq["a"])
        except Exception:
            answer = ""
            correct = False

        results["knowledge"]["total"] += 1
        if correct:
            results["knowledge"]["correct"] += 1

        results["knowledge"]["details"].append({
            "q": kq["q"][:40],
            "expected": kq["a"],
            "got": answer.strip()[:60],
            "correct": correct,
            "cat": kq["cat"],
        })

        mark = "✓" if correct else "✗"
        if verbose:
            print(f"    {mark} [{kq['cat']}] {kq['a']:>15s} {'← correct' if correct else '← got: ' + answer.strip()[:30]}")

    k = results["knowledge"]
    if verbose:
        print(f"    Score: {k['correct']}/{k['total']}")
        print()

    # --- Tool-calling benchmark ---
    if verbose:
        print("  [3/3] Tool Calling Test")

    for tq in _TOOL_QS:
        try:
            resp = provider.chat(
                model,
                [{"role": "user", "content": tq["q"]}],
                tools=_TOOL_DEFS,
                max_tokens=128,
            )
            correct = _check_tool_call(
                resp,
                tq["expect_tool"],
                tq.get("expect_args"),
                tq.get("expect_args_contains"),
            )
        except Exception:
            resp = {}
            correct = False

        results["tools"]["total"] += 1
        if correct:
            results["tools"]["correct"] += 1

        # Extract what was actually called
        tc_info = ""
        msg = resp.get("message", {})
        tcs = msg.get("tool_calls", [])
        if tcs:
            f = tcs[0].get("function", {})
            tc_info = f"{f.get('name', '?')}({json.dumps(f.get('arguments', {}), ensure_ascii=False)[:50]})"
        else:
            tc_info = "(no tool call)"

        results["tools"]["details"].append({
            "q": tq["q"],
            "expected": tq["expect_tool"],
            "got": tc_info,
            "correct": correct,
        })

        mark = "✓" if correct else "✗"
        if verbose:
            print(f"    {mark} {tq['expect_tool']:>6s} → {tc_info}")

    t = results["tools"]
    if verbose:
        print(f"    Score: {t['correct']}/{t['total']}")
        print()

    # --- Summary ---
    total_score = k["correct"] + t["correct"]
    total_max = k["total"] + t["total"]
    pct = total_score / total_max * 100 if total_max > 0 else 0

    results["summary"] = {
        "total_score": total_score,
        "total_max": total_max,
        "pct": round(pct, 1),
        "speed_tps": results["speed"]["tokens_per_sec"],
        "ttft_ms": results["speed"]["ttft_ms"],
    }

    if verbose:
        print(f"  {'='*50}")
        print(f"  Summary: {model}")
        print(f"  {'─'*50}")
        print(f"  Speed:      {results['speed']['tokens_per_sec']} tok/s (TTFT {results['speed']['ttft_ms']}ms)")
        print(f"  Knowledge:  {k['correct']}/{k['total']}")
        print(f"  Tools:      {t['correct']}/{t['total']}")
        print(f"  Total:      {total_score}/{total_max} ({pct:.0f}%)")
        print(f"  {'='*50}\n")

    return results
