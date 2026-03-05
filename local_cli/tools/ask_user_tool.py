"""Ask user tool for interactive prompts.

Presents a question to the user via ``input()`` and returns their
response.  Useful when the agent needs clarification or confirmation
before proceeding with an action.
"""

from local_cli.tools.base import Tool


class AskUserTool(Tool):
    """Prompt the user with a question and return their answer."""

    @property
    def name(self) -> str:
        return "ask_user"

    @property
    def description(self) -> str:
        return (
            "Ask the user a question and wait for their response. "
            "Use this when you need clarification, confirmation, or "
            "additional information to proceed."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask the user.",
                },
            },
            "required": ["question"],
        }

    def execute(self, **kwargs: object) -> str:
        """Prompt the user with a question and return their answer.

        Args:
            **kwargs: Must include ``question`` (str).

        Returns:
            The user's response as a string, or an error message if
            the prompt fails.
        """
        question = kwargs.get("question")
        if not isinstance(question, str) or not question.strip():
            return "Error: 'question' parameter is required and must be a non-empty string."

        try:
            answer = input(f"\n{question}\n> ")
        except EOFError:
            return "Error: no input available (non-interactive environment)."
        except KeyboardInterrupt:
            return "Error: user cancelled the prompt."

        return answer
