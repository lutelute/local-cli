"""Ideation engine for local-cli.

Provides a tool-free brainstorming mode optimized for open-ended ideation
and creative thinking.  Multi-turn conversations use ``/api/chat`` without
the ``tools`` parameter, while single-shot prompts use ``/api/generate``
for lightweight one-off generation.

The ideation engine maintains its own message history, completely separate
from the main agent conversation, so brainstorming context does not pollute
the agent loop.
"""

import sys
from typing import Any, Generator

from local_cli.agent import collect_streaming_response
from local_cli.ollama_client import OllamaClient
from local_cli.providers.base import (
    ProviderRequestError,
    ProviderStreamError,
)
from local_cli.spinner import Spinner


# ---------------------------------------------------------------------------
# Default ideation system prompt
# ---------------------------------------------------------------------------

_DEFAULT_SYSTEM_PROMPT = (
    "You are a creative brainstorming assistant. Focus on generating ideas, "
    "exploring concepts, and thinking deeply about problems. You do not have "
    "access to any tools -- respond with free-form text only."
)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class IdeationError(Exception):
    """Base exception for ideation operations."""


# ---------------------------------------------------------------------------
# Streaming generate response collector
# ---------------------------------------------------------------------------


def collect_generate_response(
    stream: Generator[dict[str, Any], None, None],
    spinner: Spinner | None = None,
) -> str:
    """Accumulate a streaming generate response and print tokens as they arrive.

    Iterates over NDJSON chunks from :meth:`OllamaClient.generate_stream`,
    concatenating response deltas.  Tokens are printed to stdout immediately
    for a responsive user experience.

    Unlike :func:`~local_cli.agent.collect_streaming_response`, this handles
    the ``/api/generate`` response format where content is in
    ``chunk["response"]`` rather than ``chunk["message"]["content"]``.

    Args:
        stream: A generator yielding parsed NDJSON chunks from the Ollama
            streaming generate API.
        spinner: Optional spinner to stop once the first content arrives.

    Returns:
        The complete generated text.

    Raises:
        ProviderStreamError: If the stream yields an error chunk.
        KeyboardInterrupt: If the user presses Ctrl+C during streaming.
            Partial content accumulated so far is returned.
    """
    content_parts: list[str] = []
    spinner_stopped = False

    try:
        for chunk in stream:
            # Generate API uses "response" key for content deltas.
            delta = chunk.get("response", "")
            if delta:
                # Stop the spinner on first content token.
                if spinner and not spinner_stopped:
                    spinner.stop()
                    spinner_stopped = True
                content_parts.append(delta)
                sys.stdout.write(delta)
                sys.stdout.flush()

    except KeyboardInterrupt:
        # User interrupted streaming.  Return what we have so far.
        if spinner and not spinner_stopped:
            spinner.stop()
        sys.stdout.write("\n")
        sys.stdout.flush()

    except ProviderStreamError:
        # Mid-stream error from the provider.  Print a newline to cleanly
        # separate any partial output, then re-raise so the caller can
        # decide how to handle it.
        if spinner and not spinner_stopped:
            spinner.stop()
        sys.stdout.write("\n")
        sys.stdout.flush()
        raise

    # Ensure spinner is stopped after stream completes.
    if spinner and not spinner_stopped:
        spinner.stop()

    # Print a trailing newline after streamed content (if any was printed).
    if content_parts:
        sys.stdout.write("\n")
        sys.stdout.flush()

    return "".join(content_parts)


# ---------------------------------------------------------------------------
# Ideation engine
# ---------------------------------------------------------------------------


class IdeationEngine:
    """Tool-free brainstorming engine with separate message history.

    Supports two modes of operation:

    * **Multi-turn chat** -- uses ``/api/chat`` without the ``tools``
      parameter to maintain a conversational brainstorming session.
    * **Single-shot generate** -- uses ``/api/generate`` for lightweight
      one-off ideation prompts.

    The engine maintains its own message history list, completely separate
    from the main agent conversation, so brainstorming context does not
    pollute the agent loop.

    Args:
        client: An :class:`OllamaClient` instance for LLM communication.
    """

    def __init__(self, client: OllamaClient) -> None:
        self._client = client
        self._messages: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def start_session(self, system_prompt: str | None = None) -> None:
        """Initialize or reset the ideation session with a system prompt.

        If the session already has messages, this replaces the system
        prompt but preserves conversation history.  If the session is
        empty, a new system message is inserted at the start.

        Args:
            system_prompt: The system prompt for ideation context.  If
                ``None``, uses the default brainstorming prompt.
        """
        prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT
        system_message: dict[str, Any] = {
            "role": "system",
            "content": prompt,
        }

        if self._messages and self._messages[0].get("role") == "system":
            # Replace existing system prompt.
            self._messages[0] = system_message
        else:
            # Insert system prompt at the start.
            self._messages.insert(0, system_message)

    def clear_history(self) -> None:
        """Reset the ideation message history.

        Removes all messages including the system prompt.  Call
        :meth:`start_session` again to begin a new session.
        """
        self._messages.clear()

    def get_history(self) -> list[dict[str, Any]]:
        """Return the current ideation message history.

        Returns:
            A list of message dicts (system, user, assistant).  This is
            the internal list, not a copy -- callers should not mutate it
            directly.
        """
        return self._messages

    @property
    def has_session(self) -> bool:
        """Whether an ideation session is active (has a system prompt)."""
        return bool(
            self._messages
            and self._messages[0].get("role") == "system"
        )

    # ------------------------------------------------------------------
    # Multi-turn chat (no tools)
    # ------------------------------------------------------------------

    def chat_turn(
        self,
        user_input: str,
        model: str,
        think: bool | None = True,
    ) -> str:
        """Send a user message and stream the assistant's response.

        Uses ``/api/chat`` **without** the ``tools`` parameter so the
        model focuses on free-form text generation.  The ``think``
        parameter enables chain-of-thought reasoning when the model
        supports it; if the model does not support it, the request is
        retried transparently without ``think``.

        Both the user message and assistant response are appended to the
        internal message history.

        Args:
            user_input: The user's brainstorming prompt or follow-up.
            model: Ollama model name (e.g. ``"qwen3:8b"``).
            think: Enable chain-of-thought reasoning.  Defaults to
                ``True``.  Set to ``None`` to omit the parameter entirely.

        Returns:
            The assistant's response text.

        Raises:
            IdeationError: If the chat request fails after fallback
                attempts.
        """
        # Ensure a session is active.
        if not self.has_session:
            self.start_session()

        # Append user message.
        self._messages.append({
            "role": "user",
            "content": user_input,
        })

        # Attempt streaming chat without tools.
        try:
            content = self._do_chat_stream(model, think=think)
        except ProviderRequestError:
            if think is not None:
                # Model may not support `think` parameter -- retry without.
                sys.stderr.write(
                    "Model does not support thinking mode, "
                    "falling back to standard generation\n"
                )
                try:
                    content = self._do_chat_stream(model, think=None)
                except (ProviderRequestError, ProviderStreamError) as exc:
                    # Remove the user message on failure.
                    self._messages.pop()
                    raise IdeationError(
                        f"Ideation chat failed: {exc}"
                    ) from exc
            else:
                # Already without think -- nothing to fall back to.
                self._messages.pop()
                raise
        except ProviderStreamError as exc:
            # Mid-stream error -- remove the user message.
            self._messages.pop()
            raise IdeationError(
                f"Ideation chat stream error: {exc}"
            ) from exc

        # Append assistant response to history.
        self._messages.append({
            "role": "assistant",
            "content": content,
        })

        return content

    def _do_chat_stream(
        self,
        model: str,
        think: bool | None = None,
    ) -> str:
        """Execute a streaming chat request and collect the response.

        Args:
            model: Ollama model name.
            think: Optional ``think`` parameter for the chat request.

        Returns:
            The assistant's response text.

        Raises:
            ProviderRequestError: If the API returns an error.
            ProviderStreamError: If an error occurs mid-stream.
        """
        thinking_spinner = Spinner("Thinking")
        thinking_spinner.start()

        try:
            stream = self._client.chat_stream(
                model, self._messages, think=think,
            )
            full_response = collect_streaming_response(
                stream, spinner=thinking_spinner,
            )
        except (ProviderRequestError, ProviderStreamError):
            thinking_spinner.stop()
            raise
        except KeyboardInterrupt:
            thinking_spinner.stop()
            sys.stderr.write("\nInterrupted.\n")
            return ""

        return full_response["message"].get("content", "")

    # ------------------------------------------------------------------
    # Single-shot generate (no chat context)
    # ------------------------------------------------------------------

    def single_shot(
        self,
        prompt: str,
        model: str,
        think: bool | None = None,
    ) -> str:
        """Generate a single response without chat context.

        Uses ``/api/generate`` for a lightweight one-off generation.
        This does **not** use or modify the ideation message history.

        Args:
            prompt: The prompt string for generation.
            model: Ollama model name (e.g. ``"qwen3:8b"``).
            think: Optional flag to enable chain-of-thought reasoning.

        Returns:
            The generated text.

        Raises:
            IdeationError: If the generation request fails after
                fallback attempts.
        """
        kwargs: dict[str, Any] = {}
        if think is not None:
            kwargs["think"] = think

        try:
            content = self._do_generate_stream(prompt, model, **kwargs)
        except ProviderRequestError:
            if think is not None:
                # Model may not support `think` -- retry without.
                sys.stderr.write(
                    "Model does not support thinking mode, "
                    "falling back to standard generation\n"
                )
                try:
                    content = self._do_generate_stream(prompt, model)
                except (ProviderRequestError, ProviderStreamError) as exc:
                    raise IdeationError(
                        f"Ideation generation failed: {exc}"
                    ) from exc
            else:
                raise
        except ProviderStreamError as exc:
            raise IdeationError(
                f"Ideation generation stream error: {exc}"
            ) from exc

        return content

    def _do_generate_stream(
        self,
        prompt: str,
        model: str,
        **kwargs: Any,
    ) -> str:
        """Execute a streaming generate request and collect the response.

        Args:
            prompt: The prompt string.
            model: Ollama model name.
            **kwargs: Additional parameters for the generate API.

        Returns:
            The generated text.

        Raises:
            ProviderRequestError: If the API returns an error.
            ProviderStreamError: If an error occurs mid-stream.
        """
        generating_spinner = Spinner("Generating")
        generating_spinner.start()

        try:
            stream = self._client.generate_stream(
                model, prompt, **kwargs,
            )
            content = collect_generate_response(
                stream, spinner=generating_spinner,
            )
        except (ProviderRequestError, ProviderStreamError):
            generating_spinner.stop()
            raise
        except KeyboardInterrupt:
            generating_spinner.stop()
            sys.stderr.write("\nInterrupted.\n")
            return ""

        return content
