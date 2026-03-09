"""Top-level orchestrator for local-cli.

Manages provider instances, consults the model registry to route tasks
to appropriate models, and supports runtime provider/brain switching.
The "brain" model handles high-level planning and delegation while
worker models handle specific tasks.

The orchestrator caches provider instances so that switching back to a
previously used provider does not incur re-initialization costs.

Sub-agent support: the orchestrator can spawn independent sub-agents
that run in parallel via :class:`~local_cli.sub_agent.SubAgentRunner`.
Each sub-agent gets a fresh :class:`~local_cli.providers.base.LLMProvider`
instance for thread safety.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from local_cli.config import Config
from local_cli.model_registry import ModelRegistry, TaskType
from local_cli.providers.base import LLMProvider

if TYPE_CHECKING:
    from local_cli.sub_agent import SubAgentResult, SubAgentRunner


class Orchestrator:
    """Top-level controller for multi-provider LLM orchestration.

    Manages the lifecycle of :class:`~local_cli.providers.base.LLMProvider`
    instances, consults a :class:`~local_cli.model_registry.ModelRegistry`
    to route tasks to appropriate model+provider combinations, and supports
    runtime switching of the active provider and brain model.

    Provider instances are cached by name so that switching between
    providers does not re-create them.

    Args:
        config: Application configuration providing default provider,
            model, and orchestrator model settings.
        registry: Optional model registry for task-based routing.
            If ``None``, routing falls back to the config defaults.
    """

    def __init__(
        self,
        config: Config,
        registry: ModelRegistry | None = None,
    ) -> None:
        self._config: Config = config
        self._registry: ModelRegistry | None = registry
        self._providers: dict[str, LLMProvider] = {}
        self._active_provider_name: str = config.provider
        self._brain_model: str = config.orchestrator_model or config.model

        # Sub-agent runner is lazily created on first use.
        self._sub_agent_runner: SubAgentRunner | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_provider(self, name: str) -> LLMProvider:
        """Create a new provider instance by name.

        Uses lazy imports to avoid circular dependencies and to keep
        provider modules optional.

        Args:
            name: Provider name (``'ollama'`` or ``'claude'``).

        Returns:
            A :class:`LLMProvider` instance.

        Raises:
            ValueError: If *name* is not a recognized provider.
            ValueError: If the provider cannot be initialized (e.g.
                missing API key for Claude).
        """
        from local_cli.providers import get_provider

        if name == "ollama":
            return get_provider("ollama", base_url=self._config.ollama_host)

        if name == "claude":
            return get_provider("claude")

        raise ValueError(
            f"Unknown provider: {name!r}. "
            f"Supported providers: 'ollama', 'claude'."
        )

    # ------------------------------------------------------------------
    # Provider management
    # ------------------------------------------------------------------

    def get_provider(self, name: str | None = None) -> LLMProvider:
        """Get or create a provider instance by name.

        Provider instances are cached so that repeated calls with the
        same name return the same instance.  If *name* is ``None``, the
        active provider is returned.

        Args:
            name: Provider name, or ``None`` for the active provider.

        Returns:
            A :class:`LLMProvider` instance.

        Raises:
            ValueError: If the provider name is unknown or the provider
                cannot be initialized.
        """
        if name is None:
            name = self._active_provider_name

        if name not in self._providers:
            self._providers[name] = self._create_provider(name)

        return self._providers[name]

    def get_active_provider(self) -> LLMProvider:
        """Return the currently active provider instance.

        This is a convenience wrapper around
        ``get_provider(None)`` that always returns the active provider.

        Returns:
            The active :class:`LLMProvider` instance.

        Raises:
            ValueError: If the active provider cannot be initialized.
        """
        return self.get_provider(self._active_provider_name)

    def get_active_provider_name(self) -> str:
        """Return the name of the currently active provider.

        Returns:
            The active provider name string (e.g. ``'ollama'``).
        """
        return self._active_provider_name

    def switch_provider(self, name: str) -> LLMProvider:
        """Switch the active provider.

        Creates the provider if it has not been used before.  On failure,
        the active provider remains unchanged and a warning is printed
        to stderr.  If the fallback provider also fails, the original
        error is re-raised.

        Args:
            name: Provider name to switch to.

        Returns:
            The newly active :class:`LLMProvider` instance.

        Raises:
            ValueError: If neither the requested provider nor the
                fallback can be initialized.
        """
        try:
            provider = self.get_provider(name)
            self._active_provider_name = name
            return provider
        except (ValueError, Exception) as exc:
            # Attempt fallback to the other provider.
            fallback = "ollama" if name != "ollama" else "claude"
            sys.stderr.write(
                f"Warning: Failed to switch to provider {name!r}: {exc}\n"
                f"Attempting fallback to {fallback!r}...\n"
            )

            try:
                provider = self.get_provider(fallback)
                self._active_provider_name = fallback
                sys.stderr.write(
                    f"Fallback to {fallback!r} successful.\n"
                )
                return provider
            except (ValueError, Exception):
                raise ValueError(
                    f"Cannot switch to provider {name!r} and "
                    f"fallback to {fallback!r} also failed: {exc}"
                ) from exc

    # ------------------------------------------------------------------
    # Brain model management
    # ------------------------------------------------------------------

    def get_brain_model(self) -> str:
        """Return the current brain model name.

        The brain model is the model used for high-level planning and
        delegation tasks.

        Returns:
            The brain model name string.
        """
        return self._brain_model

    def set_brain_model(self, model: str) -> None:
        """Set the brain model.

        Args:
            model: Model name to use as the brain (e.g. ``'qwen3:8b'``
                or ``'claude-sonnet-4-5'``).

        Raises:
            ValueError: If the model name is empty.
        """
        if not model:
            raise ValueError("Brain model name cannot be empty")
        self._brain_model = model

    # ------------------------------------------------------------------
    # Task routing
    # ------------------------------------------------------------------

    def get_model_for_task(
        self,
        task_type: TaskType,
    ) -> tuple[str, str]:
        """Look up the best provider+model pair for a task type.

        Consults the model registry if available.  If no registry is
        configured or the registry has no mapping for the task type,
        falls back to the config defaults (active provider + config model).

        Args:
            task_type: The type of task to route.

        Returns:
            A tuple of ``(provider_name, model_name)``.
        """
        if self._registry is not None:
            return self._registry.get_model_for_task(task_type)

        return (self._active_provider_name, self._config.model)

    def get_provider_for_task(
        self,
        task_type: TaskType,
    ) -> tuple[LLMProvider, str]:
        """Get the provider instance and model for a task type.

        Combines :meth:`get_model_for_task` with :meth:`get_provider` to
        return both the provider instance and the model name.  If the
        preferred provider cannot be initialized, falls back to the
        active provider with a warning.

        Args:
            task_type: The type of task to route.

        Returns:
            A tuple of ``(provider_instance, model_name)``.
        """
        provider_name, model_name = self.get_model_for_task(task_type)

        try:
            provider = self.get_provider(provider_name)
            return (provider, model_name)
        except (ValueError, Exception) as exc:
            sys.stderr.write(
                f"Warning: Cannot create provider {provider_name!r} "
                f"for task {task_type.value!r}: {exc}\n"
                f"Falling back to active provider "
                f"{self._active_provider_name!r}.\n"
            )
            provider = self.get_active_provider()
            return (provider, self._config.model)

    # ------------------------------------------------------------------
    # Registry access
    # ------------------------------------------------------------------

    @property
    def registry(self) -> ModelRegistry | None:
        """The model registry, or ``None`` if not configured."""
        return self._registry

    @registry.setter
    def registry(self, value: ModelRegistry | None) -> None:
        """Set or clear the model registry.

        Args:
            value: A :class:`ModelRegistry` instance, or ``None`` to
                clear the registry and fall back to config defaults.
        """
        self._registry = value

    # ------------------------------------------------------------------
    # Sub-agent support
    # ------------------------------------------------------------------

    def create_fresh_provider(
        self,
        name: str | None = None,
    ) -> LLMProvider:
        """Create a new, uncached provider instance for sub-agent use.

        Unlike :meth:`get_provider` (which caches instances by name),
        this method always creates a **new** provider instance.  This
        ensures that each sub-agent thread gets its own
        :class:`LLMProvider` with its own underlying HTTP client, which
        is necessary for thread safety.

        Args:
            name: Provider name to create.  If ``None``, uses the
                active provider name.

        Returns:
            A new :class:`LLMProvider` instance (not added to the cache).

        Raises:
            ValueError: If the provider name is unknown or the provider
                cannot be initialized.
        """
        if name is None:
            name = self._active_provider_name
        return self._create_provider(name)

    def _ensure_sub_agent_runner(self) -> SubAgentRunner:
        """Return the sub-agent runner, creating it lazily if needed.

        Returns:
            The :class:`SubAgentRunner` instance.
        """
        if self._sub_agent_runner is None:
            from local_cli.sub_agent import SubAgentRunner

            self._sub_agent_runner = SubAgentRunner()
        return self._sub_agent_runner

    def spawn_agent(
        self,
        prompt: str,
        description: str = "",
        model: str | None = None,
        run_in_background: bool = False,
    ) -> SubAgentResult | str:
        """Spawn a sub-agent to execute a task.

        Convenience method that creates a :class:`SubAgent` using the
        active provider configuration and model, then submits it to the
        :class:`SubAgentRunner`.

        A **fresh** provider instance is created for thread safety (via
        :meth:`create_fresh_provider`).  The sub-agent gets the default
        sub-agent tools (no ``AgentTool``, no ``AskUserTool``).

        Args:
            prompt: Detailed task prompt for the sub-agent.
            description: Short description of the task (3-5 words).
                Defaults to ``"sub-agent task"`` if empty.
            model: Model name to use.  If ``None``, uses the config
                model (``self._config.model``).
            run_in_background: If ``True``, returns immediately with
                the agent ID string.  If ``False`` (default), blocks
                until the sub-agent completes and returns the result.

        Returns:
            A :class:`SubAgentResult` when ``run_in_background`` is
            ``False``, or the agent ID string when ``True``.

        Raises:
            ValueError: If *prompt* is empty.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Sub-agent prompt cannot be empty")

        from local_cli.sub_agent import SubAgent
        from local_cli.tools import get_sub_agent_tools

        runner = self._ensure_sub_agent_runner()
        fresh_provider = self.create_fresh_provider()
        agent_model = model or self._config.model
        sub_agent_tools = get_sub_agent_tools()

        sub_agent = SubAgent(
            provider=fresh_provider,
            model=agent_model,
            tools=sub_agent_tools,
            prompt=prompt.strip(),
            description=description or "sub-agent task",
        )

        if run_in_background:
            return runner.submit_background(sub_agent)

        return runner.submit(sub_agent)

    def get_background_result(
        self,
        agent_id: str,
    ) -> SubAgentResult | None:
        """Retrieve the result of a background sub-agent.

        Delegates to :meth:`SubAgentRunner.get_background_result`.
        Returns ``None`` if the runner has not been created, the agent
        ID is not recognized, or the agent is still running.

        Args:
            agent_id: The agent ID returned by :meth:`spawn_agent`
                with ``run_in_background=True``.

        Returns:
            The sub-agent result if completed, or ``None``.
        """
        if self._sub_agent_runner is None:
            return None
        return self._sub_agent_runner.get_background_result(agent_id)

    def shutdown_agents(self) -> None:
        """Shut down the sub-agent runner and its thread pool.

        Cancels pending futures and waits for running threads to
        complete.  Safe to call multiple times or when no runner
        has been created.
        """
        if self._sub_agent_runner is not None:
            self._sub_agent_runner.shutdown()
            self._sub_agent_runner = None
