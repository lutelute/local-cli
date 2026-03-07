"""Abstract base class for all built-in tools.

Every tool inherits from ``Tool`` and implements the four abstract members:
``name``, ``description``, ``parameters``, and ``execute``.  The concrete
``to_ollama_tool()`` and ``to_claude_tool()`` methods convert the tool
definition into provider-specific formats consumed by their respective APIs.
"""

from abc import ABC, abstractmethod


class Tool(ABC):
    """Base class for all built-in tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name used in function calls (e.g. ``"bash"``)."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description shown to the LLM."""
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """JSON Schema dict describing the tool's accepted parameters.

        Example::

            {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute.",
                    },
                },
                "required": ["command"],
            }
        """
        ...

    @abstractmethod
    def execute(self, **kwargs: object) -> str:
        """Execute the tool and return the result as a string.

        All tool results are serialised to strings so that they can be
        passed back to the LLM as ``role: "tool"`` message content.

        Args:
            **kwargs: Keyword arguments matching the JSON Schema defined
                by :pyattr:`parameters`.

        Returns:
            A string representation of the tool's result.
        """
        ...

    def to_ollama_tool(self) -> dict:
        """Convert to the Ollama / OpenAI function-calling format.

        Returns:
            A dictionary suitable for the ``tools`` field of a
            ``/api/chat`` request::

                {
                    "type": "function",
                    "function": {
                        "name": "...",
                        "description": "...",
                        "parameters": { ... },
                    },
                }
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_claude_tool(self) -> dict:
        """Convert to the Anthropic Claude Messages API tool format.

        Returns:
            A dictionary suitable for the ``tools`` field of a
            ``/v1/messages`` request::

                {
                    "name": "...",
                    "description": "...",
                    "input_schema": { ... },
                }

        Note:
            Claude uses ``input_schema`` where Ollama / OpenAI use
            ``parameters``, and the Claude format is a flat dictionary
            rather than being wrapped in ``{"type": "function", ...}``.
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }
