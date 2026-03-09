"""Plan management for local-cli.

Provides :class:`PlanManager` for creating, listing, showing, updating,
activating, and abandoning task plans stored as markdown files.  Plans
use human-readable markdown with checkbox steps and metadata headers.
"""

import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Valid plan statuses.
_VALID_STATUSES = frozenset({"draft", "active", "complete", "abandoned"})

# File extension for plan files.
_PLAN_EXT = ".md"

# Regex for extracting plan ID from filename (e.g. "001.md" -> "001").
_PLAN_ID_RE = re.compile(r"^(\d{3,})$")

# Width of zero-padded plan IDs.
_ID_PAD_WIDTH = 3


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PlanError(Exception):
    """Base exception for plan operations."""


class PlanNotFoundError(PlanError):
    """Raised when a referenced plan does not exist."""


class PlanParseError(PlanError):
    """Raised when a plan file has malformed markdown."""


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


class Plan:
    """In-memory representation of a plan.

    Attributes:
        plan_id: Zero-padded string identifier (e.g. ``"001"``).
        title: Plan title.
        status: One of ``draft``, ``active``, ``complete``, ``abandoned``.
        created: ISO-8601 creation timestamp.
        model: Model name used when the plan was created.
        description: Free-text description.
        steps: List of ``(done, text)`` tuples where *done* is a bool.
        notes: Free-text notes section.
    """

    __slots__ = (
        "plan_id",
        "title",
        "status",
        "created",
        "model",
        "description",
        "steps",
        "notes",
    )

    def __init__(
        self,
        plan_id: str,
        title: str,
        status: str = "draft",
        created: str = "",
        model: str = "",
        description: str = "",
        steps: list[tuple[bool, str]] | None = None,
        notes: str = "",
    ) -> None:
        self.plan_id = plan_id
        self.title = title
        self.status = status
        self.created = created
        self.model = model
        self.description = description
        self.steps = steps if steps is not None else []
        self.notes = notes


# ---------------------------------------------------------------------------
# PlanManager
# ---------------------------------------------------------------------------


class PlanManager:
    """Manages task plans as markdown files.

    Plans are stored as markdown files in a configurable directory
    (default ``.agents/plans/``).  Each plan has a zero-padded sequential
    ID (e.g. ``001``, ``002``) derived by scanning existing files.

    Args:
        plans_dir: Directory where plan files are stored.  Created
            automatically if it does not exist.
    """

    def __init__(self, plans_dir: str = ".agents/plans") -> None:
        self._plans_dir = Path(plans_dir).expanduser()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_plan(
        self,
        title: str,
        description: str = "",
        steps: list[str] | None = None,
        model: str = "",
    ) -> Plan:
        """Create a new plan and write it to disk.

        The plan is assigned the next available sequential ID and written
        as a markdown file with ``draft`` status.

        Args:
            title: Plan title.
            description: Optional description text.
            steps: Optional list of step description strings.
            model: Optional model name to record.

        Returns:
            The newly created :class:`Plan`.

        Raises:
            PlanError: If the plan file cannot be written.
        """
        self._plans_dir.mkdir(parents=True, exist_ok=True)

        plan_id = self._next_id()
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        step_tuples = [(False, s) for s in (steps or [])]

        plan = Plan(
            plan_id=plan_id,
            title=title,
            status="draft",
            created=now,
            model=model,
            description=description,
            steps=step_tuples,
            notes="",
        )

        self._write_plan(plan)
        return plan

    def list_plans(self) -> list[Plan]:
        """Return all plans sorted by ID (ascending).

        Malformed plan files are silently skipped.

        Returns:
            A list of :class:`Plan` objects.
        """
        if not self._plans_dir.is_dir():
            return []

        plans: list[Plan] = []
        try:
            for entry in self._plans_dir.iterdir():
                if entry.is_file() and entry.suffix == _PLAN_EXT:
                    stem = entry.stem
                    if _PLAN_ID_RE.match(stem):
                        try:
                            plan = self._read_plan(stem)
                            plans.append(plan)
                        except (PlanParseError, PlanNotFoundError):
                            # Skip malformed or unreadable plans.
                            continue
        except OSError:
            return []

        plans.sort(key=lambda p: p.plan_id)
        return plans

    def show_plan(self, plan_id: str) -> Plan:
        """Load and return a single plan.

        Args:
            plan_id: The plan identifier (e.g. ``"001"``).

        Returns:
            The :class:`Plan` object.

        Raises:
            PlanNotFoundError: If the plan file does not exist.
            PlanParseError: If the plan file is malformed.
        """
        plan_id = self._normalize_id(plan_id)
        return self._read_plan(plan_id)

    def update_step(self, plan_id: str, step_number: int, done: bool) -> Plan:
        """Mark a plan step as done or not done.

        Step numbers are 1-based to match the display format.

        Args:
            plan_id: The plan identifier.
            step_number: 1-based step index.
            done: Whether the step is complete.

        Returns:
            The updated :class:`Plan`.

        Raises:
            PlanNotFoundError: If the plan does not exist.
            PlanParseError: If the plan is malformed.
            PlanError: If the step number is out of range.
        """
        plan_id = self._normalize_id(plan_id)
        plan = self._read_plan(plan_id)

        if step_number < 1 or step_number > len(plan.steps):
            raise PlanError(
                f"Step {step_number} is out of range "
                f"(plan has {len(plan.steps)} steps)."
            )

        idx = step_number - 1
        _done, text = plan.steps[idx]
        plan.steps[idx] = (done, text)

        # Auto-complete plan if all steps are done.
        if all(s[0] for s in plan.steps) and plan.steps:
            plan.status = "complete"

        self._write_plan(plan)
        return plan

    def activate_plan(self, plan_id: str) -> Plan:
        """Set a plan's status to ``active``.

        Args:
            plan_id: The plan identifier.

        Returns:
            The updated :class:`Plan`.

        Raises:
            PlanNotFoundError: If the plan does not exist.
            PlanParseError: If the plan is malformed.
            PlanError: If the plan cannot be activated (e.g. already
                abandoned or complete).
        """
        plan_id = self._normalize_id(plan_id)
        plan = self._read_plan(plan_id)

        if plan.status in ("abandoned", "complete"):
            raise PlanError(
                f"Cannot activate plan {plan_id} with status "
                f"'{plan.status}'."
            )

        plan.status = "active"
        self._write_plan(plan)
        return plan

    def abandon_plan(self, plan_id: str) -> Plan:
        """Set a plan's status to ``abandoned``.

        Args:
            plan_id: The plan identifier.

        Returns:
            The updated :class:`Plan`.

        Raises:
            PlanNotFoundError: If the plan does not exist.
            PlanParseError: If the plan is malformed.
        """
        plan_id = self._normalize_id(plan_id)
        plan = self._read_plan(plan_id)
        plan.status = "abandoned"
        self._write_plan(plan)
        return plan

    def update_notes(self, plan_id: str, notes: str) -> Plan:
        """Replace the notes section of a plan.

        Args:
            plan_id: The plan identifier.
            notes: New notes content.

        Returns:
            The updated :class:`Plan`.

        Raises:
            PlanNotFoundError: If the plan does not exist.
            PlanParseError: If the plan is malformed.
        """
        plan_id = self._normalize_id(plan_id)
        plan = self._read_plan(plan_id)
        plan.notes = notes
        self._write_plan(plan)
        return plan

    def get_plan_content(self, plan_id: str) -> str:
        """Return the raw markdown content of a plan file.

        Args:
            plan_id: The plan identifier.

        Returns:
            The markdown content as a string.

        Raises:
            PlanNotFoundError: If the plan file does not exist.
        """
        plan_id = self._normalize_id(plan_id)
        file_path = self._plan_path(plan_id)

        if not file_path.exists():
            raise PlanNotFoundError(
                f"Plan '{plan_id}' not found."
            )

        try:
            return file_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise PlanError(f"Failed to read plan '{plan_id}': {exc}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _plan_path(self, plan_id: str) -> Path:
        """Return the filesystem path for a given plan identifier."""
        return self._plans_dir / f"{plan_id}{_PLAN_EXT}"

    def _normalize_id(self, plan_id: str) -> str:
        """Normalize a plan ID to zero-padded format.

        Accepts both ``"1"`` and ``"001"`` and returns ``"001"``.

        Args:
            plan_id: Raw plan identifier string.

        Returns:
            Zero-padded plan ID string.
        """
        try:
            num = int(plan_id)
        except ValueError:
            return plan_id
        return str(num).zfill(_ID_PAD_WIDTH)

    def _next_id(self) -> str:
        """Determine the next available sequential plan ID.

        Scans existing plan files and returns one higher than the
        current maximum.

        Returns:
            A zero-padded plan ID string (e.g. ``"001"``).
        """
        max_id = 0

        if self._plans_dir.is_dir():
            try:
                for entry in self._plans_dir.iterdir():
                    if entry.is_file() and entry.suffix == _PLAN_EXT:
                        stem = entry.stem
                        if _PLAN_ID_RE.match(stem):
                            try:
                                num = int(stem)
                                if num > max_id:
                                    max_id = num
                            except ValueError:
                                continue
            except OSError:
                pass

        return str(max_id + 1).zfill(_ID_PAD_WIDTH)

    def _write_plan(self, plan: Plan) -> None:
        """Write a plan to disk using atomic write (temp + rename).

        Args:
            plan: The :class:`Plan` to persist.

        Raises:
            PlanError: If the file cannot be written.
        """
        content = self._render_markdown(plan)
        file_path = self._plan_path(plan.plan_id)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Atomic write: write to temp file then rename.
            fd, tmp_path = tempfile.mkstemp(
                dir=str(self._plans_dir),
                suffix=".tmp",
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(content)
                os.replace(tmp_path, str(file_path))
            except BaseException:
                # Clean up temp file on failure.
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except OSError as exc:
            raise PlanError(f"Failed to write plan '{plan.plan_id}': {exc}")

    def _render_markdown(self, plan: Plan) -> str:
        """Render a plan as a markdown string.

        Args:
            plan: The :class:`Plan` to render.

        Returns:
            A markdown-formatted string.
        """
        lines: list[str] = []

        lines.append(f"# Plan: {plan.title}")
        lines.append("")
        lines.append(f"**Status**: {plan.status}")
        lines.append(f"**Created**: {plan.created}")
        lines.append(f"**Model**: {plan.model}")
        lines.append("")
        lines.append("## Description")
        lines.append("")
        lines.append(plan.description if plan.description else "")
        lines.append("")
        lines.append("## Steps")
        lines.append("")

        for done, text in plan.steps:
            checkbox = "[x]" if done else "[ ]"
            lines.append(f"- {checkbox} {text}")

        lines.append("")
        lines.append("## Notes")
        lines.append("")
        lines.append(plan.notes if plan.notes else "")
        lines.append("")

        return "\n".join(lines)

    def _read_plan(self, plan_id: str) -> Plan:
        """Parse a plan markdown file into a :class:`Plan` object.

        Args:
            plan_id: The plan identifier.

        Returns:
            A :class:`Plan` instance.

        Raises:
            PlanNotFoundError: If the file does not exist.
            PlanParseError: If the markdown structure is invalid.
        """
        file_path = self._plan_path(plan_id)

        if not file_path.exists():
            raise PlanNotFoundError(f"Plan '{plan_id}' not found.")

        try:
            content = file_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise PlanError(f"Failed to read plan '{plan_id}': {exc}")

        return self._parse_markdown(plan_id, content)

    def _parse_markdown(self, plan_id: str, content: str) -> Plan:
        """Parse markdown content into a :class:`Plan`.

        Args:
            plan_id: The plan identifier.
            content: Raw markdown string.

        Returns:
            A :class:`Plan` instance.

        Raises:
            PlanParseError: If required fields are missing or malformed.
        """
        lines = content.splitlines()

        # -- Parse title from first heading --
        title = ""
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("# Plan:"):
                title = stripped[len("# Plan:"):].strip()
                break

        if not title:
            raise PlanParseError(
                f"Plan '{plan_id}' is missing '# Plan: <title>' heading."
            )

        # -- Parse metadata fields --
        status = self._extract_field(lines, "Status") or "draft"
        created = self._extract_field(lines, "Created") or ""
        model = self._extract_field(lines, "Model") or ""

        if status not in _VALID_STATUSES:
            raise PlanParseError(
                f"Plan '{plan_id}' has invalid status: '{status}'."
            )

        # -- Parse sections --
        description = self._extract_section(lines, "Description")
        notes = self._extract_section(lines, "Notes")

        # -- Parse steps --
        steps = self._parse_steps(lines)

        return Plan(
            plan_id=plan_id,
            title=title,
            status=status,
            created=created,
            model=model,
            description=description,
            steps=steps,
            notes=notes,
        )

    def _extract_field(self, lines: list[str], field_name: str) -> str:
        """Extract a ``**Field**: value`` metadata field from plan lines.

        Args:
            lines: Plan file lines.
            field_name: Field name to look for (e.g. ``"Status"``).

        Returns:
            The field value, or an empty string if not found.
        """
        prefix = f"**{field_name}**:"
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(prefix):
                return stripped[len(prefix):].strip()
        return ""

    def _extract_section(self, lines: list[str], section_name: str) -> str:
        """Extract the content of a ``## Section`` from plan lines.

        Collects all lines between the section heading and the next ``##``
        heading (or end of file).

        Args:
            lines: Plan file lines.
            section_name: Section heading text (e.g. ``"Description"``).

        Returns:
            The section content as a string, stripped of leading/trailing
            whitespace.
        """
        heading = f"## {section_name}"
        in_section = False
        section_lines: list[str] = []

        for line in lines:
            stripped = line.strip()
            if stripped == heading:
                in_section = True
                continue
            if in_section:
                if stripped.startswith("## "):
                    break
                section_lines.append(line)

        return "\n".join(section_lines).strip()

    def _parse_steps(self, lines: list[str]) -> list[tuple[bool, str]]:
        """Parse checkbox steps from the ``## Steps`` section.

        Recognizes ``- [ ] text`` (incomplete) and ``- [x] text``
        (complete) patterns.

        Args:
            lines: Plan file lines.

        Returns:
            A list of ``(done, text)`` tuples.
        """
        in_steps = False
        steps: list[tuple[bool, str]] = []

        for line in lines:
            stripped = line.strip()
            if stripped == "## Steps":
                in_steps = True
                continue
            if in_steps:
                if stripped.startswith("## "):
                    break
                if stripped.startswith("- [x] ") or stripped.startswith("- [X] "):
                    steps.append((True, stripped[6:]))
                elif stripped.startswith("- [ ] "):
                    steps.append((False, stripped[6:]))

        return steps
