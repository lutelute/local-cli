"""Sub-agent execution for local-cli.

Provides the :class:`SubAgent` execution unit, :class:`SubAgentResult`
result container, and :class:`SubAgentRunner` parallel execution manager
for the multi-agent system.  Each sub-agent runs an independent agent
loop with its own isolated context (messages, provider, tools) and
returns structured results.

Sub-agents run in threads via :class:`SubAgentRunner` and operate
silently -- no stdout streaming, no spinners, no interactive I/O.
"""

import json
import os
import subprocess
import tempfile
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from local_cli.providers.base import LLMProvider, ProviderStreamError
from local_cli.tools.base import Tool


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class SubAgentResult:
    """Structured result from a sub-agent execution.

    Contains the final output, execution metadata, and status information.
    Returned by :meth:`SubAgent.run` after the agent loop completes.

    Attributes:
        agent_id: Unique identifier for this sub-agent execution.
        description: Short description of the task (3-5 words).
        content: Final output text from the sub-agent (last assistant
            message content).  Empty string on error/timeout if no
            content was produced.
        status: Execution status -- one of ``'success'``, ``'error'``,
            or ``'timeout'``.
        duration_seconds: Wall-clock execution time in seconds.
        messages_count: Total number of messages in the sub-agent's
            conversation history (including system, user, assistant,
            and tool messages).
        tool_calls_count: Total number of tool calls executed during
            the sub-agent's run.
        error_message: Error details when status is ``'error'`` or
            ``'timeout'``.  Empty string on success.
    """

    agent_id: str
    description: str
    content: str
    status: str  # "success" | "error" | "timeout"
    duration_seconds: float
    messages_count: int
    tool_calls_count: int
    error_message: str = ""
    worktree_path: str = ""

    def format_result(self) -> str:
        """Format the result as a human-readable string for the LLM.

        Returns:
            A formatted string summarizing the sub-agent's execution
            and its output content.
        """
        lines = [
            f"Sub-agent '{self.description}' ({self.agent_id})",
            f"Status: {self.status}",
            f"Duration: {self.duration_seconds:.1f}s",
            f"Messages: {self.messages_count}, "
            f"Tool calls: {self.tool_calls_count}",
        ]
        if self.error_message:
            lines.append(f"Error: {self.error_message}")
        if self.worktree_path:
            lines.append(f"Worktree: {self.worktree_path}")
        if self.content:
            lines.append(f"\nResult:\n{self.content}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sub-agent execution unit
# ---------------------------------------------------------------------------


# Default system prompt for sub-agents.
_SUB_AGENT_SYSTEM_PROMPT = (
    "You are a sub-agent working on a specific task. "
    "Complete the task thoroughly and return a clear, concise result. "
    "You have access to tools to help you accomplish the task. "
    "Focus only on the assigned task."
)

# Default timeout for sub-agent execution (seconds).
_DEFAULT_TIMEOUT = 300.0


class SubAgent:
    """Independent agent execution unit with isolated context.

    Each sub-agent has its own message history, provider instance, and
    tool list.  It runs a silent agent loop (no stdout streaming) and
    returns a :class:`SubAgentResult` when complete.

    Sub-agents are designed to be submitted to a :class:`SubAgentRunner`
    for concurrent execution via ``ThreadPoolExecutor``.

    Args:
        provider: An :class:`~local_cli.providers.base.LLMProvider`
            instance for this sub-agent.  Must be a **fresh** instance
            (not shared with other agents) for thread safety.
        model: Model name to use (e.g. ``'qwen3:8b'``).
        tools: List of :class:`~local_cli.tools.base.Tool` instances
            available to the sub-agent.  Should **not** include
            ``AgentTool`` (prevents recursion) or ``AskUserTool``
            (prevents stdin blocking).
        prompt: The task prompt for the sub-agent to execute.
        description: Short description of the task (3-5 words).
        agent_id: Optional unique identifier.  If ``None``, one is
            generated automatically.
        timeout: Maximum execution time in seconds.  Defaults to 300.
        isolation: Optional isolation mode.  When set to
            ``'worktree'``, the sub-agent runs in a temporary git
            worktree.  Worktree creation is serialized by
            :class:`SubAgentRunner` to avoid races on the shared
            ``.git`` directory.  If the worktree has changes after
            execution, it is preserved and its path is returned in
            :attr:`SubAgentResult.worktree_path`.  Defaults to
            ``None`` (no isolation).
    """

    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        tools: list[Tool],
        prompt: str,
        description: str = "",
        agent_id: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
        isolation: str | None = None,
    ) -> None:
        self._provider = provider
        self._model = model
        self._tools = list(tools)
        self._prompt = prompt
        self._description = description or "sub-agent task"
        self._agent_id = agent_id or self._generate_agent_id()
        self._timeout = timeout
        self._isolation = isolation

        # Each sub-agent gets its own isolated message list.
        self._messages: list[dict[str, Any]] = []
        self._tool_calls_count = 0

        # Worktree isolation state -- populated by _setup_worktree().
        self._worktree_path: str = ""
        self._worktree_branch: str = ""
        self._worktree_base_commit: str = ""

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def agent_id(self) -> str:
        """Unique identifier for this sub-agent."""
        return self._agent_id

    @property
    def description(self) -> str:
        """Short description of the sub-agent's task."""
        return self._description

    @property
    def isolation(self) -> str | None:
        """Isolation mode for this sub-agent (``'worktree'`` or ``None``)."""
        return self._isolation

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self) -> SubAgentResult:
        """Execute the sub-agent's task and return the result.

        Runs a silent agent loop (no stdout/stderr output) with the
        configured provider, model, tools, and prompt.  The loop
        continues until the LLM responds without tool calls or the
        timeout is reached.

        When worktree isolation is active (``_worktree_path`` is set by
        :meth:`_setup_worktree`), the working directory is changed to
        the worktree before the agent loop runs and restored afterward.
        After completion, the worktree is cleaned up if no changes were
        detected; otherwise it is preserved and its path included in
        the result.

        This method is designed to be called from a thread and is
        safe for concurrent execution (each sub-agent uses its own
        provider instance and message list).

        Returns:
            A :class:`SubAgentResult` containing the final output,
            status, and execution metadata.
        """
        start_time = time.monotonic()

        # Initialize the message list with system prompt and user task.
        self._messages = [
            {"role": "system", "content": _SUB_AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": self._prompt},
        ]
        self._tool_calls_count = 0
        final_content = ""

        # Change to worktree directory if isolation is active.
        original_cwd = ""
        if self._worktree_path:
            original_cwd = os.getcwd()
            os.chdir(self._worktree_path)

        try:
            final_content = self._run_agent_loop(start_time)
            status = "success"
            error_message = ""
        except _SubAgentTimeout as exc:
            status = "timeout"
            error_message = str(exc)
            # Preserve any partial content accumulated before timeout.
            final_content = self._extract_last_assistant_content()
        except Exception as exc:
            status = "error"
            error_message = f"{type(exc).__name__}: {exc}"
            final_content = self._extract_last_assistant_content()
        finally:
            # Restore original working directory.
            if original_cwd:
                try:
                    os.chdir(original_cwd)
                except OSError:
                    pass

        duration = time.monotonic() - start_time

        # Clean up worktree; preserve if changes were detected.
        preserved_worktree = ""
        if self._worktree_path:
            if self._teardown_worktree():
                preserved_worktree = self._worktree_path

        return SubAgentResult(
            agent_id=self._agent_id,
            description=self._description,
            content=final_content,
            status=status,
            duration_seconds=round(duration, 2),
            messages_count=len(self._messages),
            tool_calls_count=self._tool_calls_count,
            error_message=error_message,
            worktree_path=preserved_worktree,
        )

    # ------------------------------------------------------------------
    # Internal: silent agent loop
    # ------------------------------------------------------------------

    def _run_agent_loop(self, start_time: float) -> str:
        """Run the silent agent loop until completion or timeout.

        This is a simplified, silent variant of the main
        :func:`~local_cli.agent.agent_loop` that does not write to
        stdout/stderr and does not use spinners.

        Args:
            start_time: Monotonic timestamp of when execution started,
                used for timeout checking.

        Returns:
            The final assistant message content string.

        Raises:
            _SubAgentTimeout: If the timeout is exceeded.
        """
        tool_map: dict[str, Tool] = {t.name: t for t in self._tools}
        tool_defs: list[dict[str, Any]] = self._provider.format_tools(
            self._tools,
        )

        final_content = ""

        while True:
            # Check timeout before each LLM call.
            self._check_timeout(start_time)

            # Send messages to the LLM and collect the response silently.
            stream = self._provider.chat_stream(
                self._model,
                self._messages,
                tools=tool_defs,
            )
            full_response = self._collect_silent_response(stream)

            # Check timeout after response collection (streaming can
            # take a long time).
            self._check_timeout(start_time)

            # Append the assistant message to the conversation history.
            assistant_message = full_response["message"]
            self._messages.append(assistant_message)
            final_content = assistant_message.get("content", "")

            # Check for tool calls.  If none, we're done.
            tool_calls = assistant_message.get("tool_calls", [])
            if not tool_calls:
                break

            # Execute each tool call and append results.
            for tc in tool_calls:
                self._check_timeout(start_time)

                func = tc.get("function", {})
                tool_name = func.get("name", "")
                arguments = func.get("arguments", {})

                # Ollama sometimes returns arguments as a JSON string.
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except (json.JSONDecodeError, ValueError):
                        arguments = {}

                tool_call_id = tc.get("id")
                tool = tool_map.get(tool_name)

                if tool is None:
                    result = f"Error: unknown tool '{tool_name}'"
                else:
                    try:
                        result = tool.execute(**arguments)
                    except Exception as exc:
                        result = f"Error: {type(exc).__name__}: {exc}"

                self._tool_calls_count += 1

                # Append tool result as a 'tool' role message.
                tool_msg: dict[str, Any] = {
                    "role": "tool",
                    "tool_name": tool_name,
                    "content": result,
                }
                if tool_call_id is not None:
                    tool_msg["tool_call_id"] = tool_call_id
                self._messages.append(tool_msg)

        return final_content

    @staticmethod
    def _collect_silent_response(
        stream: Any,
    ) -> dict[str, Any]:
        """Accumulate a streaming response silently (no stdout/stderr).

        Silent variant of
        :func:`~local_cli.agent.collect_streaming_response` that
        accumulates content without writing to stdout, creating
        spinners, or printing debug output.

        Args:
            stream: A generator yielding streaming chunks from the
                provider.

        Returns:
            A dictionary with the assembled response in the same
            format as :func:`collect_streaming_response`.
        """
        content_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        last_chunk: dict[str, Any] = {}

        try:
            for chunk in stream:
                last_chunk = chunk
                message = chunk.get("message", {})

                delta = message.get("content", "")
                if delta:
                    content_parts.append(delta)

                chunk_tool_calls = message.get("tool_calls")
                if chunk_tool_calls:
                    tool_calls.extend(chunk_tool_calls)

        except KeyboardInterrupt:
            # Propagate interrupt -- the caller (run()) handles it.
            raise
        except ProviderStreamError:
            # Stream error -- re-raise for the caller to handle.
            raise

        # Build the assembled response.
        assembled_message: dict[str, Any] = {
            "role": "assistant",
            "content": "".join(content_parts),
        }
        if tool_calls:
            assembled_message["tool_calls"] = tool_calls

        result: dict[str, Any] = dict(last_chunk)
        result["message"] = assembled_message

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_timeout(self, start_time: float) -> None:
        """Raise :class:`_SubAgentTimeout` if the timeout has been exceeded.

        Args:
            start_time: Monotonic timestamp of when execution started.

        Raises:
            _SubAgentTimeout: If the elapsed time exceeds the timeout.
        """
        elapsed = time.monotonic() - start_time
        if elapsed > self._timeout:
            raise _SubAgentTimeout(
                f"Sub-agent timed out after {elapsed:.1f}s "
                f"(limit: {self._timeout}s)"
            )

    def _extract_last_assistant_content(self) -> str:
        """Extract the content from the last assistant message, if any.

        Used to preserve partial output on timeout or error.

        Returns:
            The content string, or empty string if no assistant
            message exists.
        """
        for msg in reversed(self._messages):
            if msg.get("role") == "assistant":
                return msg.get("content", "")
        return ""

    # ------------------------------------------------------------------
    # Internal: worktree isolation
    # ------------------------------------------------------------------

    def _setup_worktree(self) -> str:
        """Create a temporary git worktree for this sub-agent.

        Creates a new branch and worktree directory using
        ``git worktree add``.  Records the base commit hash so that
        :meth:`_has_worktree_changes` can detect new commits.

        This method is called by :class:`SubAgentRunner` in the main
        thread (serialized) before the sub-agent is submitted to the
        thread pool, to avoid races on the shared ``.git`` directory.

        Returns:
            The filesystem path to the created worktree directory.

        Raises:
            _WorktreeError: If ``git worktree add`` fails.
        """
        branch_name = f"sub-agent-{self._agent_id}"
        worktree_dir = os.path.join(
            tempfile.gettempdir(),
            f"local-cli-worktree-{self._agent_id}",
        )

        result = subprocess.run(
            ["git", "worktree", "add", "-b", branch_name, worktree_dir],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise _WorktreeError(
                f"git worktree add failed: {result.stderr.strip()}"
            )

        # Record base commit for change detection.
        rev_result = subprocess.run(
            ["git", "-C", worktree_dir, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        self._worktree_path = worktree_dir
        self._worktree_branch = branch_name
        self._worktree_base_commit = rev_result.stdout.strip()
        return worktree_dir

    def _teardown_worktree(self) -> bool:
        """Clean up the worktree if no changes were detected.

        Checks for both uncommitted changes and new commits.  If the
        worktree is clean, removes it via ``git worktree remove`` and
        deletes the associated branch.  If changes are detected, the
        worktree is preserved.

        Cleanup failures are handled gracefully (silently ignored) to
        avoid masking the sub-agent's actual result.

        Returns:
            ``True`` if changes were detected and the worktree was
            preserved; ``False`` if the worktree was cleaned up (or
            was not set).
        """
        if not self._worktree_path:
            return False

        has_changes = self._has_worktree_changes()

        if not has_changes:
            # Remove the worktree directory.
            try:
                subprocess.run(
                    ["git", "worktree", "remove", self._worktree_path],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            except (subprocess.TimeoutExpired, OSError):
                pass  # Graceful cleanup -- don't fail.

            # Delete the temporary branch.
            try:
                subprocess.run(
                    ["git", "branch", "-d", self._worktree_branch],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
            except (subprocess.TimeoutExpired, OSError):
                pass  # Graceful cleanup -- don't fail.

            self._worktree_path = ""
            return False

        return True

    def _has_worktree_changes(self) -> bool:
        """Check if the worktree has uncommitted or committed changes.

        Detects two kinds of changes:

        1. **Uncommitted changes** -- via ``git status --porcelain``.
        2. **New commits** -- by comparing the current HEAD to the
           base commit recorded at worktree creation time.

        Returns ``True`` on any error as a safe default (preserves
        the worktree rather than accidentally deleting work).

        Returns:
            ``True`` if changes were detected, ``False`` if the
            worktree is clean.
        """
        if not self._worktree_path:
            return False

        try:
            # Check for uncommitted changes.
            status_result = subprocess.run(
                ["git", "-C", self._worktree_path,
                 "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if status_result.stdout.strip():
                return True

            # Check if HEAD has moved (new commits).
            rev_result = subprocess.run(
                ["git", "-C", self._worktree_path,
                 "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            current_commit = rev_result.stdout.strip()
            return current_commit != self._worktree_base_commit

        except (subprocess.TimeoutExpired, OSError):
            # Assume changes on error (safe default).
            return True

    @staticmethod
    def _generate_agent_id() -> str:
        """Generate a unique agent identifier.

        Format: ``agent-YYYYMMDD-HHMMSS-<short_uuid>`` (UTC).
        Follows the same pattern as
        :meth:`~local_cli.session.SessionManager.generate_session_id`.

        Returns:
            A unique agent identifier string.
        """
        now = datetime.now(timezone.utc)
        timestamp = now.strftime("%Y%m%d-%H%M%S")
        short_id = uuid.uuid4().hex[:8]
        return f"agent-{timestamp}-{short_id}"


# ---------------------------------------------------------------------------
# Parallel execution manager
# ---------------------------------------------------------------------------


class SubAgentRunner:
    """Manages concurrent sub-agent execution via ``ThreadPoolExecutor``.

    Supports foreground (blocking) and background (non-blocking) execution
    modes.  Background agents are tracked by their agent ID and can be
    polled for completion via :meth:`get_background_result`.

    The thread pool is created lazily on first submission to avoid
    resource waste when no sub-agents are needed.

    Args:
        max_workers: Maximum number of concurrent sub-agent threads.
            Capped to ``min(max_workers, OLLAMA_NUM_PARALLEL)`` to
            avoid Ollama queue saturation.  Defaults to 3.
    """

    # Default max concurrent sub-agents.
    _DEFAULT_MAX_WORKERS = 3

    def __init__(self, max_workers: int = _DEFAULT_MAX_WORKERS) -> None:
        ollama_parallel = self._get_ollama_num_parallel()
        self._max_workers = min(max_workers, ollama_parallel)
        self._executor: ThreadPoolExecutor | None = None
        self._background: dict[str, Future[SubAgentResult]] = {}
        # Serializes worktree creation to avoid races on .git.
        self._worktree_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, sub_agent: "SubAgent") -> SubAgentResult:
        """Submit a sub-agent for execution and block until completion.

        If the sub-agent has ``isolation='worktree'``, its worktree is
        created (serialized) in the calling thread before the agent is
        submitted to the thread pool.

        Creates a thread in the pool to run the sub-agent's
        :meth:`~SubAgent.run` method and waits for the result.

        Args:
            sub_agent: The sub-agent to execute.

        Returns:
            The sub-agent's execution result.
        """
        self._prepare_isolation(sub_agent)
        executor = self._ensure_executor()
        future = executor.submit(sub_agent.run)
        return future.result()

    def submit_background(self, sub_agent: "SubAgent") -> str:
        """Submit a sub-agent for background execution.

        If the sub-agent has ``isolation='worktree'``, its worktree is
        created (serialized) in the calling thread before the agent is
        submitted to the thread pool.

        Returns immediately with the agent ID.  The result can be
        retrieved later via :meth:`get_background_result`.

        Args:
            sub_agent: The sub-agent to execute in the background.

        Returns:
            The agent ID for tracking the background execution.
        """
        self._prepare_isolation(sub_agent)
        executor = self._ensure_executor()
        future = executor.submit(sub_agent.run)
        self._background[sub_agent.agent_id] = future
        return sub_agent.agent_id

    def get_background_result(
        self,
        agent_id: str,
    ) -> SubAgentResult | None:
        """Retrieve the result of a background sub-agent.

        Returns ``None`` if the agent is still running or the
        agent ID is not recognized.

        Args:
            agent_id: The agent ID returned by :meth:`submit_background`.

        Returns:
            The sub-agent result if completed, or ``None``.
        """
        future = self._background.get(agent_id)
        if future is None:
            return None
        if not future.done():
            return None
        return future.result()

    def list_background_agents(self) -> list[dict]:
        """List all background agents and their current status.

        Returns:
            A list of dicts, each with ``agent_id`` and ``status``
            keys.  Status is one of ``'running'``, ``'completed'``,
            or ``'error'``.
        """
        agents: list[dict] = []
        for agent_id, future in self._background.items():
            if not future.done():
                status = "running"
            elif future.exception() is not None:
                status = "error"
            else:
                status = "completed"
            agents.append({"agent_id": agent_id, "status": status})
        return agents

    def shutdown(self) -> None:
        """Shut down the thread pool executor.

        Cancels pending futures and waits for running threads to
        complete.  Safe to call multiple times.
        """
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_isolation(self, sub_agent: "SubAgent") -> None:
        """Set up worktree isolation if the sub-agent requests it.

        Worktree creation is serialized via a lock to avoid races on
        the shared ``.git`` directory.  Called in the main thread
        before the sub-agent is submitted to the thread pool.

        Args:
            sub_agent: The sub-agent whose isolation to prepare.

        Raises:
            _WorktreeError: If worktree creation fails.
        """
        if sub_agent.isolation == "worktree":
            with self._worktree_lock:
                sub_agent._setup_worktree()

    def _ensure_executor(self) -> ThreadPoolExecutor:
        """Return the thread pool executor, creating it lazily if needed.

        Returns:
            The ``ThreadPoolExecutor`` instance.
        """
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self._max_workers,
            )
        return self._executor

    @staticmethod
    def _get_ollama_num_parallel() -> int:
        """Read ``OLLAMA_NUM_PARALLEL`` from the environment.

        Returns:
            The configured parallelism limit, or ``4`` if the
            variable is unset or not a valid integer.
        """
        raw = os.environ.get("OLLAMA_NUM_PARALLEL", "4")
        try:
            value = int(raw)
            return max(value, 1)
        except (ValueError, TypeError):
            return 4


# ---------------------------------------------------------------------------
# Internal exceptions
# ---------------------------------------------------------------------------


class _SubAgentTimeout(Exception):
    """Raised internally when a sub-agent exceeds its timeout."""


class _WorktreeError(Exception):
    """Raised when git worktree creation or management fails."""
