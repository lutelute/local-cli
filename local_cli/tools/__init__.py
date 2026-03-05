"""Tool system for local-cli.

Provides :func:`get_default_tools` to obtain instances of every built-in
tool, and :func:`get_tool_map` for name-based lookup.
"""

from local_cli.tools.base import Tool


def get_default_tools() -> list[Tool]:
    """Return a list of all default tool instances.

    Tools are imported lazily so that this module can be imported even
    before every tool module exists.  As new tool modules are created
    they should be imported and instantiated here.

    Returns:
        A list of :class:`Tool` instances.
    """
    from local_cli.tools.bash_tool import BashTool
    from local_cli.tools.edit_tool import EditTool
    from local_cli.tools.glob_tool import GlobTool
    from local_cli.tools.grep_tool import GrepTool
    from local_cli.tools.read_tool import ReadTool
    from local_cli.tools.write_tool import WriteTool

    tools: list[Tool] = [
        BashTool(),
        ReadTool(),
        WriteTool(),
        EditTool(),
        GlobTool(),
        GrepTool(),
    ]
    return tools


def get_tool_map() -> dict[str, Tool]:
    """Return a mapping of tool names to tool instances.

    Convenience wrapper around :func:`get_default_tools` for fast
    lookup by name during the agent loop.

    Returns:
        A dictionary mapping each tool's ``name`` to its instance.
    """
    return {tool.name: tool for tool in get_default_tools()}
