"""Sub-agent execution for local-cli.

Provides the :class:`SubAgent` execution unit and :class:`SubAgentResult`
result container for the multi-agent system.  Each sub-agent runs an
independent agent loop with its own isolated context (messages, provider,
tools) and returns structured results.

Sub-agents are designed to run in threads via :class:`SubAgentRunner`
(added later) and operate silently -- no stdout streaming, no spinners,
no interactive I/O.
"""

import json
import time
import uuid
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
    ) -> None:
        self._provider = provider
        self._model = model
        self._tools = list(tools)
        self._prompt = prompt
        self._description = description or "sub-agent task"
        self._agent_id = agent_id or self._generate_agent_id()
        self._timeout = timeout

        # Each sub-agent gets its own isolated message list.
        self._messages: list[dict[str, Any]] = []
        self._tool_calls_count = 0

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

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self) -> SubAgentResult:
        """Execute the sub-agent's task and return the result.

        Runs a silent agent loop (no stdout/stderr output) with the
        configured provider, model, tools, and prompt.  The loop
        continues until the LLM responds without tool calls or the
        timeout is reached.

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

        duration = time.monotonic() - start_time

        return SubAgentResult(
            agent_id=self._agent_id,
            description=self._description,
            content=final_content,
            status=status,
            duration_seconds=round(duration, 2),
            messages_count=len(self._messages),
            tool_calls_count=self._tool_calls_count,
            error_message=error_message,
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
# Internal exceptions
# ---------------------------------------------------------------------------


class _SubAgentTimeout(Exception):
    """Raised internally when a sub-agent exceeds its timeout."""
