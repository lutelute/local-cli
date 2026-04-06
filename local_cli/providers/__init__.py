"""LLM provider abstraction layer for local-cli.

Provides :class:`LLMProvider`, the abstract base class that normalizes the
interface across different LLM backends (Ollama, Claude API, etc.), and a
provider-agnostic exception hierarchy.

Use :func:`get_provider` to obtain a concrete provider instance by name.
"""

from local_cli.providers.base import (
    LLMProvider,
    ProviderConnectionError,
    ProviderRequestError,
    ProviderStreamError,
)


def get_provider(name: str, **kwargs: object) -> LLMProvider:
    """Return a provider instance for the given backend name.

    Providers are imported lazily so that this module can be imported
    even before every provider module exists.  As new provider modules
    are created they should be imported and instantiated here.

    Args:
        name: Provider backend name (``'ollama'`` or ``'claude'``).
        **kwargs: Additional keyword arguments forwarded to the
            provider constructor.

    Returns:
        A :class:`LLMProvider` instance.

    Raises:
        ValueError: If *name* is not a recognized provider.
    """
    if name == "ollama":
        from local_cli.providers.ollama_provider import OllamaProvider

        return OllamaProvider(**kwargs)

    if name == "claude":
        from local_cli.providers.claude_provider import ClaudeProvider

        return ClaudeProvider(**kwargs)

    if name == "llama-server":
        from local_cli.providers.llama_server_provider import LlamaServerProvider

        return LlamaServerProvider(**kwargs)

    raise ValueError(
        f"Unknown provider: {name!r}. Supported providers: 'ollama', 'claude', 'llama-server'."
    )


__all__ = [
    "LLMProvider",
    "ProviderConnectionError",
    "ProviderRequestError",
    "ProviderStreamError",
    "get_provider",
]
