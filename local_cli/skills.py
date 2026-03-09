"""Skills system for local-cli.

Provides :class:`SkillsLoader` for discovering, parsing, and matching
skill definitions stored as ``SKILL.md`` files with YAML-like frontmatter.
Skills are auto-discovered from a configurable directory and matched
against user input for contextual injection into the system prompt.
"""

from pathlib import Path


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SkillsError(Exception):
    """Base exception for skills operations."""


class SkillNotFoundError(SkillsError):
    """Raised when a referenced skill does not exist."""


class SkillParseError(SkillsError):
    """Raised when a SKILL.md file has malformed frontmatter."""


# ---------------------------------------------------------------------------
# Skill data container
# ---------------------------------------------------------------------------


class Skill:
    """Lightweight container for a parsed skill definition.

    Attributes:
        name: Unique skill name (from frontmatter or directory name).
        triggers: List of trigger keywords/phrases for matching.
        description: Brief description of what the skill enables.
        content: The body content of the SKILL.md (below frontmatter).
        path: Filesystem path to the SKILL.md file.
    """

    __slots__ = ("name", "triggers", "description", "content", "path")

    def __init__(
        self,
        name: str,
        triggers: list[str],
        description: str,
        content: str,
        path: Path,
    ) -> None:
        self.name = name
        self.triggers = triggers
        self.description = description
        self.content = content
        self.path = path


# ---------------------------------------------------------------------------
# SkillsLoader
# ---------------------------------------------------------------------------


class SkillsLoader:
    """Discovers and manages skill definitions from SKILL.md files.

    Skills are stored in subdirectories of a configurable base directory
    (default ``.agents/skills/``).  Each skill directory contains a
    ``SKILL.md`` file with YAML-like frontmatter defining its name,
    triggers, and description.

    Directory layout::

        .agents/skills/
            my-skill/
                SKILL.md

    SKILL.md format::

        ---
        name: my-skill
        triggers: [keyword1, keyword2, phrase]
        description: What this skill enables
        ---

        # Skill Content

        Instructions and context the agent receives when triggered.

    Args:
        skills_dir: Base directory for skill storage.  Missing directory
            is handled gracefully (no error, empty skill list).
    """

    # Expected filename within each skill subdirectory.
    _SKILL_FILE = "SKILL.md"

    def __init__(self, skills_dir: str = ".agents/skills") -> None:
        self._skills_dir = Path(skills_dir).expanduser()
        self._skills: dict[str, Skill] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def discover_skills(self) -> list[Skill]:
        """Scan the skills directory and load all valid skill definitions.

        Iterates over subdirectories of the configured skills directory,
        parsing each ``SKILL.md`` file found.  Skills with missing or
        malformed frontmatter are silently skipped.

        Returns:
            A list of discovered :class:`Skill` objects, sorted by name.
        """
        self._skills.clear()

        if not self._skills_dir.is_dir():
            return []

        try:
            for entry in sorted(self._skills_dir.iterdir()):
                if not entry.is_dir():
                    continue

                skill_file = entry / self._SKILL_FILE
                if not skill_file.is_file():
                    continue

                try:
                    skill = self._parse_skill_file(skill_file, entry.name)
                    self._skills[skill.name] = skill
                except (SkillParseError, OSError):
                    # Skip malformed or unreadable skill files gracefully.
                    continue
        except OSError:
            return []

        return list(self._skills.values())

    def get_matching_skills(self, user_input: str) -> list[Skill]:
        """Return skills whose triggers match the user input.

        Matching is case-insensitive substring matching: a skill matches
        if any of its trigger keywords/phrases appear anywhere in the
        user input string.

        Args:
            user_input: The user's input text to match against.

        Returns:
            A list of matching :class:`Skill` objects, sorted by name.
        """
        if not user_input or not user_input.strip():
            return []

        input_lower = user_input.lower()
        matched: list[Skill] = []

        for skill in self._skills.values():
            for trigger in skill.triggers:
                if trigger.lower() in input_lower:
                    matched.append(skill)
                    break

        matched.sort(key=lambda s: s.name)
        return matched

    def get_skill_content(self, name: str) -> str:
        """Return the body content of a skill by name.

        Args:
            name: The skill name to look up.

        Returns:
            The skill body content string.

        Raises:
            SkillNotFoundError: If no skill with the given name exists.
        """
        skill = self._skills.get(name)
        if skill is None:
            raise SkillNotFoundError(f"Skill '{name}' not found.")
        return skill.content

    def list_skills(self) -> list[Skill]:
        """Return all discovered skills, sorted by name.

        Returns:
            A list of :class:`Skill` objects.
        """
        skills = list(self._skills.values())
        skills.sort(key=lambda s: s.name)
        return skills

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_skill_file(self, path: Path, dir_name: str) -> Skill:
        """Parse a SKILL.md file into a :class:`Skill` object.

        Reads the file, extracts YAML-like frontmatter between ``---``
        delimiters, and parses it as simple ``key: value`` lines.  List
        values (e.g. ``triggers: [a, b, c]``) are parsed by stripping
        brackets and splitting on commas.

        Args:
            path: Path to the SKILL.md file.
            dir_name: The skill directory name (used as fallback name).

        Returns:
            A :class:`Skill` object.

        Raises:
            SkillParseError: If the file has no valid frontmatter.
            OSError: If the file cannot be read.
        """
        text = path.read_text(encoding="utf-8")
        frontmatter, body = self._split_frontmatter(text)

        if frontmatter is None:
            raise SkillParseError(
                f"No frontmatter found in '{path}'."
            )

        metadata = self._parse_frontmatter(frontmatter)

        name = metadata.get("name", dir_name).strip()
        if not name:
            name = dir_name

        triggers_raw = metadata.get("triggers", "")
        triggers = self._parse_list_value(triggers_raw)

        description = metadata.get("description", "").strip()

        return Skill(
            name=name,
            triggers=triggers,
            description=description,
            content=body.strip(),
            path=path,
        )

    def _split_frontmatter(self, text: str) -> tuple[str | None, str]:
        """Split a file's text into frontmatter and body sections.

        Frontmatter is delimited by ``---`` lines at the very start
        of the file.

        Args:
            text: The full file content.

        Returns:
            A tuple of ``(frontmatter_text, body_text)``.  If no valid
            frontmatter delimiters are found, returns ``(None, text)``.
        """
        lines = text.split("\n")

        # First non-empty line must be '---'.
        start_idx = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped == "---":
                start_idx = i
                break
            # First non-empty line is not '---' — no frontmatter.
            return None, text

        if start_idx is None:
            return None, text

        # Find the closing '---'.
        end_idx = None
        for i in range(start_idx + 1, len(lines)):
            if lines[i].strip() == "---":
                end_idx = i
                break

        if end_idx is None:
            return None, text

        frontmatter_lines = lines[start_idx + 1 : end_idx]
        body_lines = lines[end_idx + 1 :]

        return "\n".join(frontmatter_lines), "\n".join(body_lines)

    def _parse_frontmatter(self, frontmatter: str) -> dict[str, str]:
        """Parse YAML-like frontmatter into a dictionary.

        Handles simple ``key: value`` lines.  Values are kept as raw
        strings (list parsing is deferred to :meth:`_parse_list_value`).

        Args:
            frontmatter: The frontmatter text (without ``---`` delimiters).

        Returns:
            A dictionary of ``key: value`` string pairs.
        """
        metadata: dict[str, str] = {}

        for line in frontmatter.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Split on first colon only.
            colon_idx = line.find(":")
            if colon_idx < 0:
                continue

            key = line[:colon_idx].strip().lower()
            value = line[colon_idx + 1 :].strip()

            if key:
                metadata[key] = value

        return metadata

    def _parse_list_value(self, raw: str) -> list[str]:
        """Parse a list value string into a list of trimmed strings.

        Handles both bracketed (``[a, b, c]``) and bare (``a, b, c``)
        formats.  Empty items are filtered out.

        Args:
            raw: The raw value string.

        Returns:
            A list of trimmed, non-empty strings.
        """
        if not raw:
            return []

        # Strip surrounding brackets if present.
        value = raw.strip()
        if value.startswith("[") and value.endswith("]"):
            value = value[1:-1]

        items = [item.strip() for item in value.split(",")]
        return [item for item in items if item]
