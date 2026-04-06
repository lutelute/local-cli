"""Agent tool for spawning independent sub-agents.

Allows the LLM to dynamically spawn sub-agents that execute tasks
autonomously with their own isolated context, tools, and provider
instance.  Analogous to Claude Code's ``Agent`` tool.

Unlike other tools (which use no-arg constructors), ``AgentTool``
requires runtime dependencies injected at construction time: the
:class:`~local_cli.sub_agent.SubAgentRunner`, a template
:class:`~local_cli.providers.base.LLMProvider`, the model name, and
the sub-agent tool list.
"""

from local_cli.providers.base import LLMProvider
from local_cli.sub_agent import SubAgent, SubAgentRunner
from local_cli.tools.base import Tool


class AgentTool(Tool):
    """LLM-callable tool that spawns independent sub-agents.

    Each invocation creates a new :class:`~local_cli.sub_agent.SubAgent`
    with a **fresh** :class:`~local_cli.providers.base.LLMProvider`
    instance (for thread safety) and submits it to the
    :class:`~local_cli.sub_agent.SubAgentRunner`.

    Supports two execution modes:

    - **Foreground** (default): Blocks until the sub-agent completes and
      returns the formatted result.
    - **Background**: Returns immediately with the agent ID.  The result
      can be retrieved later via the runner.

    Args:
        runner: The :class:`SubAgentRunner` that manages concurrent
            sub-agent execution.
        provider: A template :class:`LLMProvider` instance used to
            determine the provider type and configuration.  This
            instance is **not** shared with sub-agents -- a fresh
            instance is created for each sub-agent.
        model: Model name to use for sub-agents (e.g. ``'qwen3:8b'``).
        sub_agent_tools: List of :class:`Tool` instances available to
            sub-agents.  Should **not** include ``AgentTool`` (prevents
            recursive spawning) or ``AskUserTool`` (prevents stdin
            blocking in silent threads).
    """

    def __init__(
        self,
        runner: SubAgentRunner,
        provider: LLMProvider,
        model: str,
        sub_agent_tools: list[Tool],
    ) -> None:
        self._runner = runner
        self._provider = provider
        self._model = model
        self._sub_agent_tools = sub_agent_tools

    # ------------------------------------------------------------------
    # Tool interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "agent"

    @property
    def description(self) -> str:
        return (
            "Spawn an independent sub-agent to handle a task autonomously. "
            "The sub-agent runs with its own isolated context and has "
            "access to tools (bash, read, write, edit, glob, grep, "
            "web_fetch). Use this to delegate tasks that can be worked "
            "on independently. Provide a clear, detailed prompt so the "
            "sub-agent can work autonomously."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": (
                        "Short description of the task (3-5 words)."
                    ),
                },
                "prompt": {
                    "type": "string",
                    "description": (
                        "Detailed task for the sub-agent to perform."
                    ),
                },
                "run_in_background": {
                    "type": "boolean",
                    "description": (
                        "If true, run in background and return "
                        "immediately with an agent ID."
                    ),
                },
            },
            "required": ["description", "prompt"],
        }

    def execute(self, **kwargs: object) -> str:
        """Spawn a sub-agent to execute the given task.

        Creates a new :class:`SubAgent` with a fresh provider instance
        and submits it to the runner.  In foreground mode, blocks until
        the sub-agent completes and returns the formatted result.  In
        background mode, returns the agent ID immediately.

        Args:
            **kwargs: Must include ``description`` (str) and ``prompt``
                (str).  May include ``run_in_background`` (bool,
                default ``False``).

        Returns:
            The formatted sub-agent result string (foreground) or the
            agent ID string (background).
        """
        description = kwargs.get("description", "")
        if not isinstance(description, str) or not description.strip():
            return "Error: 'description' parameter is required and must be a non-empty string."

        prompt = kwargs.get("prompt", "")
        if not isinstance(prompt, str) or not prompt.strip():
            return "Error: 'prompt' parameter is required and must be a non-empty string."

        run_in_background = kwargs.get("run_in_background", False)
        if not isinstance(run_in_background, bool):
            run_in_background = False

        # Create a fresh provider instance for thread safety.
        try:
            fresh_provider = self._create_fresh_provider()
        except Exception as exc:
            return f"Error: failed to create provider for sub-agent: {exc}"

        sub_agent = SubAgent(
            provider=fresh_provider,
            model=self._model,
            tools=self._sub_agent_tools,
            prompt=prompt.strip(),
            description=description.strip(),
        )

        if run_in_background:
            agent_id = self._runner.submit_background(sub_agent)
            return (
                f"Sub-agent '{description.strip()}' started in background. "
                f"Agent ID: {agent_id}"
            )

        # Foreground: block until completion.
        result = self._runner.submit(sub_agent)
        return result.format_result()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_fresh_provider(self) -> LLMProvider:
        """Create a new provider instance for a sub-agent.

        Uses the template provider's :pyattr:`name` to determine which
        concrete provider to create, then calls
        :func:`~local_cli.providers.get_provider` with the appropriate
        configuration extracted from the template.

        Returns:
            A new :class:`LLMProvider` instance.

        Raises:
            ValueError: If the provider type is unknown.
        """
        from local_cli.providers import get_provider

        provider_name = self._provider.name

        if provider_name == "ollama":
            # Extract the base URL from the template provider's client.
            base_url = "http://localhost:11434"
            if hasattr(self._provider, "client") and hasattr(
                self._provider.client, "base_url"
            ):
                base_url = self._provider.client.base_url
            return get_provider("ollama", base_url=base_url)

        if provider_name == "claude":
            return get_provider("claude")

        if provider_name == "llama-server":
            base_url = getattr(self._provider, "_base_url", "http://localhost:8090")
            return get_provider("llama-server", base_url=base_url)

        # Fallback: attempt to create by name with no extra kwargs.
        return get_provider(provider_name)
