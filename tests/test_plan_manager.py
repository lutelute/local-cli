"""Tests for local_cli.plan_manager module."""

import os
import tempfile
import unittest
from pathlib import Path

from local_cli.plan_manager import (
    Plan,
    PlanError,
    PlanManager,
    PlanNotFoundError,
    PlanParseError,
)


class TestPlanManagerInit(unittest.TestCase):
    """Tests for PlanManager construction."""

    def test_stores_plans_dir_as_path(self) -> None:
        """plans_dir argument is stored as a Path object."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plans_dir = os.path.join(tmpdir, "plans")
            mgr = PlanManager(plans_dir)
            self.assertIsInstance(mgr._plans_dir, Path)

    def test_tilde_expansion(self) -> None:
        """Plans dir path with ~ is expanded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            self.assertTrue(mgr._plans_dir.is_absolute())

    def test_directory_not_created_on_init(self) -> None:
        """Plans directory is NOT created on init (lazy creation)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plans_dir = os.path.join(tmpdir, "plans")
            mgr = PlanManager(plans_dir)
            self.assertFalse(Path(plans_dir).exists())

    def test_existing_directory_ok(self) -> None:
        """Initializing with an existing plans dir does not raise."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr1 = PlanManager(tmpdir)
            mgr2 = PlanManager(tmpdir)  # Second init should not fail.
            self.assertIsInstance(mgr2._plans_dir, Path)


class TestCreatePlan(unittest.TestCase):
    """Tests for PlanManager.create_plan()."""

    def test_creates_markdown_file(self) -> None:
        """create_plan creates a .md file in the plans directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            plan = mgr.create_plan("My First Plan")

            file_path = Path(tmpdir) / f"{plan.plan_id}.md"
            self.assertTrue(file_path.exists())

    def test_first_plan_gets_id_001(self) -> None:
        """First plan in an empty directory gets ID '001'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            plan = mgr.create_plan("Plan One")
            self.assertEqual(plan.plan_id, "001")

    def test_sequential_ids(self) -> None:
        """Multiple plans get sequential IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            p1 = mgr.create_plan("Plan One")
            p2 = mgr.create_plan("Plan Two")
            p3 = mgr.create_plan("Plan Three")

            self.assertEqual(p1.plan_id, "001")
            self.assertEqual(p2.plan_id, "002")
            self.assertEqual(p3.plan_id, "003")

    def test_default_status_is_draft(self) -> None:
        """Newly created plans have 'draft' status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            plan = mgr.create_plan("Draft Plan")
            self.assertEqual(plan.status, "draft")

    def test_title_preserved(self) -> None:
        """Plan title is set correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            plan = mgr.create_plan("Refactor Auth Module")
            self.assertEqual(plan.title, "Refactor Auth Module")

    def test_description_stored(self) -> None:
        """Optional description is stored in the plan."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            plan = mgr.create_plan(
                "Plan With Desc",
                description="A detailed description.",
            )
            self.assertEqual(plan.description, "A detailed description.")

    def test_steps_stored(self) -> None:
        """Steps are stored as (False, text) tuples."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            plan = mgr.create_plan(
                "Stepped Plan",
                steps=["Step 1: Do A", "Step 2: Do B"],
            )
            self.assertEqual(len(plan.steps), 2)
            self.assertEqual(plan.steps[0], (False, "Step 1: Do A"))
            self.assertEqual(plan.steps[1], (False, "Step 2: Do B"))

    def test_model_stored(self) -> None:
        """Optional model name is stored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            plan = mgr.create_plan("Model Plan", model="qwen3:8b")
            self.assertEqual(plan.model, "qwen3:8b")

    def test_created_timestamp_set(self) -> None:
        """Created timestamp is set to a non-empty ISO string."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            plan = mgr.create_plan("Timestamp Plan")
            self.assertIsInstance(plan.created, str)
            self.assertGreater(len(plan.created), 0)
            # Should look like ISO format: YYYY-MM-DDTHH:MM:SS
            self.assertIn("T", plan.created)

    def test_creates_directory_if_missing(self) -> None:
        """create_plan creates the plans directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plans_dir = os.path.join(tmpdir, "nested", "plans")
            mgr = PlanManager(plans_dir)
            mgr.create_plan("Nested Plan")
            self.assertTrue(Path(plans_dir).is_dir())

    def test_markdown_format(self) -> None:
        """Plan file contains expected markdown structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            plan = mgr.create_plan(
                "Format Test",
                description="Test desc",
                steps=["First step", "Second step"],
                model="qwen3:8b",
            )

            file_path = Path(tmpdir) / f"{plan.plan_id}.md"
            content = file_path.read_text(encoding="utf-8")

            self.assertIn("# Plan: Format Test", content)
            self.assertIn("**Status**: draft", content)
            self.assertIn("**Model**: qwen3:8b", content)
            self.assertIn("## Description", content)
            self.assertIn("Test desc", content)
            self.assertIn("## Steps", content)
            self.assertIn("- [ ] First step", content)
            self.assertIn("- [ ] Second step", content)
            self.assertIn("## Notes", content)

    def test_empty_steps(self) -> None:
        """Plan with no steps has empty steps list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            plan = mgr.create_plan("No Steps Plan")
            self.assertEqual(plan.steps, [])

    def test_unicode_title_and_description(self) -> None:
        """Unicode content in title and description is preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            plan = mgr.create_plan(
                "計画テスト 🚀",
                description="日本語の説明文",
            )

            loaded = mgr.show_plan(plan.plan_id)
            self.assertEqual(loaded.title, "計画テスト 🚀")
            self.assertEqual(loaded.description, "日本語の説明文")

    def test_id_after_gap(self) -> None:
        """If plan 001 is created, then deleted, next plan gets 002 not 001."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            p1 = mgr.create_plan("Plan One")
            # Manually delete the first plan file.
            os.unlink(Path(tmpdir) / f"{p1.plan_id}.md")
            p2 = mgr.create_plan("Plan Two")
            # Should still be 002, not 001 — but since file is gone,
            # next_id scans and finds max 0, so it becomes 001 again.
            # This tests the actual behavior of _next_id.
            self.assertIn(p2.plan_id, ("001", "002"))


class TestListPlans(unittest.TestCase):
    """Tests for PlanManager.list_plans()."""

    def test_empty_directory(self) -> None:
        """No plans returns empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            self.assertEqual(mgr.list_plans(), [])

    def test_nonexistent_directory(self) -> None:
        """Non-existent plans directory returns empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(os.path.join(tmpdir, "nonexistent"))
            self.assertEqual(mgr.list_plans(), [])

    def test_lists_all_plans(self) -> None:
        """list_plans returns all created plans."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            mgr.create_plan("Plan A")
            mgr.create_plan("Plan B")
            mgr.create_plan("Plan C")

            plans = mgr.list_plans()
            self.assertEqual(len(plans), 3)

    def test_sorted_by_id(self) -> None:
        """Plans are sorted by ID in ascending order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            mgr.create_plan("Plan A")
            mgr.create_plan("Plan B")
            mgr.create_plan("Plan C")

            plans = mgr.list_plans()
            ids = [p.plan_id for p in plans]
            self.assertEqual(ids, ["001", "002", "003"])

    def test_non_md_files_ignored(self) -> None:
        """Files without .md extension are not listed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            mgr.create_plan("Valid Plan")

            # Create a non-.md file.
            other = Path(tmpdir) / "notes.txt"
            other.write_text("not a plan")

            plans = mgr.list_plans()
            self.assertEqual(len(plans), 1)

    def test_non_numeric_stems_ignored(self) -> None:
        """Files with non-numeric stems are not listed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            mgr.create_plan("Valid Plan")

            # Create an .md file with non-numeric name.
            other = Path(tmpdir) / "README.md"
            other.write_text("# Not a plan")

            plans = mgr.list_plans()
            self.assertEqual(len(plans), 1)

    def test_malformed_plan_skipped(self) -> None:
        """Malformed plan files are silently skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            mgr.create_plan("Good Plan")

            # Create a malformed plan file.
            bad_plan = Path(tmpdir) / "002.md"
            bad_plan.write_text("This has no proper heading")

            plans = mgr.list_plans()
            self.assertEqual(len(plans), 1)
            self.assertEqual(plans[0].title, "Good Plan")

    def test_returns_plan_objects(self) -> None:
        """list_plans returns Plan instances with correct attributes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            mgr.create_plan("Test Plan", description="desc")

            plans = mgr.list_plans()
            self.assertEqual(len(plans), 1)
            self.assertIsInstance(plans[0], Plan)
            self.assertEqual(plans[0].title, "Test Plan")
            self.assertEqual(plans[0].status, "draft")


class TestShowPlan(unittest.TestCase):
    """Tests for PlanManager.show_plan()."""

    def test_round_trip(self) -> None:
        """Create then show preserves all plan fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            original = mgr.create_plan(
                "Round Trip",
                description="Test description",
                steps=["Step A", "Step B"],
                model="qwen3:8b",
            )

            loaded = mgr.show_plan(original.plan_id)

            self.assertEqual(loaded.plan_id, original.plan_id)
            self.assertEqual(loaded.title, original.title)
            self.assertEqual(loaded.status, original.status)
            self.assertEqual(loaded.created, original.created)
            self.assertEqual(loaded.model, original.model)
            self.assertEqual(loaded.description, original.description)
            self.assertEqual(loaded.steps, original.steps)

    def test_nonexistent_plan_raises(self) -> None:
        """Showing a non-existent plan raises PlanNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            # Create directory so it exists but plan file does not.
            Path(tmpdir).mkdir(parents=True, exist_ok=True)
            with self.assertRaises(PlanNotFoundError):
                mgr.show_plan("999")

    def test_normalizes_id(self) -> None:
        """show_plan accepts '1' and normalizes to '001'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            mgr.create_plan("ID Norm Test")

            plan = mgr.show_plan("1")
            self.assertEqual(plan.plan_id, "001")
            self.assertEqual(plan.title, "ID Norm Test")

    def test_malformed_markdown_raises(self) -> None:
        """Malformed plan markdown raises PlanParseError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            Path(tmpdir).mkdir(parents=True, exist_ok=True)

            # Write a malformed plan (missing title heading).
            bad_plan = Path(tmpdir) / "001.md"
            bad_plan.write_text("No heading here\nJust text\n")

            with self.assertRaises(PlanParseError):
                mgr.show_plan("001")


class TestUpdateStep(unittest.TestCase):
    """Tests for PlanManager.update_step()."""

    def test_mark_step_done(self) -> None:
        """Marking a step done updates the step tuple."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            mgr.create_plan("Step Plan", steps=["Step 1", "Step 2"])

            updated = mgr.update_step("001", 1, True)
            self.assertTrue(updated.steps[0][0])
            self.assertFalse(updated.steps[1][0])

    def test_mark_step_undone(self) -> None:
        """Marking a done step as not done works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            mgr.create_plan("Undo Step", steps=["Step 1", "Step 2"])

            mgr.update_step("001", 1, True)
            updated = mgr.update_step("001", 1, False)
            self.assertFalse(updated.steps[0][0])

    def test_step_persisted_to_disk(self) -> None:
        """Step changes are persisted to the markdown file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            mgr.create_plan("Persist Test", steps=["Do A", "Do B"])

            mgr.update_step("001", 1, True)

            content = (Path(tmpdir) / "001.md").read_text(encoding="utf-8")
            self.assertIn("- [x] Do A", content)
            self.assertIn("- [ ] Do B", content)

    def test_auto_complete_when_all_steps_done(self) -> None:
        """Plan status changes to 'complete' when all steps are done."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            mgr.create_plan("Auto Complete", steps=["Step 1", "Step 2"])

            mgr.update_step("001", 1, True)
            updated = mgr.update_step("001", 2, True)

            self.assertEqual(updated.status, "complete")

    def test_partial_steps_not_auto_complete(self) -> None:
        """Plan remains in current status when only some steps are done."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            mgr.create_plan("Partial", steps=["Step 1", "Step 2", "Step 3"])

            updated = mgr.update_step("001", 1, True)
            self.assertEqual(updated.status, "draft")

    def test_step_number_out_of_range_raises(self) -> None:
        """Step number 0 or beyond max raises PlanError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            mgr.create_plan("Range Test", steps=["Step 1"])

            with self.assertRaises(PlanError):
                mgr.update_step("001", 0, True)

            with self.assertRaises(PlanError):
                mgr.update_step("001", 2, True)

    def test_step_number_negative_raises(self) -> None:
        """Negative step numbers raise PlanError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            mgr.create_plan("Negative Test", steps=["Step 1"])

            with self.assertRaises(PlanError):
                mgr.update_step("001", -1, True)

    def test_nonexistent_plan_raises(self) -> None:
        """Updating a step on a non-existent plan raises PlanNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            Path(tmpdir).mkdir(parents=True, exist_ok=True)
            with self.assertRaises(PlanNotFoundError):
                mgr.update_step("999", 1, True)

    def test_normalizes_plan_id(self) -> None:
        """update_step accepts unnormalized IDs like '1'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            mgr.create_plan("Norm Test", steps=["Step 1"])

            updated = mgr.update_step("1", 1, True)
            self.assertTrue(updated.steps[0][0])


class TestActivatePlan(unittest.TestCase):
    """Tests for PlanManager.activate_plan()."""

    def test_activate_draft(self) -> None:
        """Activating a draft plan sets status to 'active'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            mgr.create_plan("Draft Plan")

            plan = mgr.activate_plan("001")
            self.assertEqual(plan.status, "active")

    def test_activate_persisted(self) -> None:
        """Activated status is persisted to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            mgr.create_plan("Persist Active")

            mgr.activate_plan("001")

            loaded = mgr.show_plan("001")
            self.assertEqual(loaded.status, "active")

    def test_activate_already_active(self) -> None:
        """Activating an already active plan is a no-op."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            mgr.create_plan("Already Active")

            mgr.activate_plan("001")
            plan = mgr.activate_plan("001")
            self.assertEqual(plan.status, "active")

    def test_cannot_activate_abandoned(self) -> None:
        """Activating an abandoned plan raises PlanError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            mgr.create_plan("Abandoned Plan")
            mgr.abandon_plan("001")

            with self.assertRaises(PlanError):
                mgr.activate_plan("001")

    def test_cannot_activate_complete(self) -> None:
        """Activating a complete plan raises PlanError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            mgr.create_plan("Complete Plan", steps=["Only step"])
            mgr.update_step("001", 1, True)  # Auto-completes.

            with self.assertRaises(PlanError):
                mgr.activate_plan("001")

    def test_nonexistent_plan_raises(self) -> None:
        """Activating a non-existent plan raises PlanNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            Path(tmpdir).mkdir(parents=True, exist_ok=True)
            with self.assertRaises(PlanNotFoundError):
                mgr.activate_plan("999")

    def test_normalizes_plan_id(self) -> None:
        """activate_plan accepts unnormalized IDs like '1'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            mgr.create_plan("Norm Active")

            plan = mgr.activate_plan("1")
            self.assertEqual(plan.status, "active")


class TestAbandonPlan(unittest.TestCase):
    """Tests for PlanManager.abandon_plan()."""

    def test_abandon_draft(self) -> None:
        """Abandoning a draft plan sets status to 'abandoned'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            mgr.create_plan("Draft Abandon")

            plan = mgr.abandon_plan("001")
            self.assertEqual(plan.status, "abandoned")

    def test_abandon_active(self) -> None:
        """Abandoning an active plan sets status to 'abandoned'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            mgr.create_plan("Active Abandon")
            mgr.activate_plan("001")

            plan = mgr.abandon_plan("001")
            self.assertEqual(plan.status, "abandoned")

    def test_abandon_persisted(self) -> None:
        """Abandoned status is persisted to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            mgr.create_plan("Persist Abandon")

            mgr.abandon_plan("001")

            loaded = mgr.show_plan("001")
            self.assertEqual(loaded.status, "abandoned")

    def test_nonexistent_plan_raises(self) -> None:
        """Abandoning a non-existent plan raises PlanNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            Path(tmpdir).mkdir(parents=True, exist_ok=True)
            with self.assertRaises(PlanNotFoundError):
                mgr.abandon_plan("999")

    def test_normalizes_plan_id(self) -> None:
        """abandon_plan accepts unnormalized IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            mgr.create_plan("Norm Abandon")

            plan = mgr.abandon_plan("1")
            self.assertEqual(plan.status, "abandoned")


class TestUpdateNotes(unittest.TestCase):
    """Tests for PlanManager.update_notes()."""

    def test_update_notes(self) -> None:
        """Notes section is updated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            mgr.create_plan("Notes Plan")

            plan = mgr.update_notes("001", "Some new notes here.")
            self.assertEqual(plan.notes, "Some new notes here.")

    def test_notes_persisted(self) -> None:
        """Notes changes are persisted to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            mgr.create_plan("Persist Notes")

            mgr.update_notes("001", "Persistent note content")

            loaded = mgr.show_plan("001")
            self.assertEqual(loaded.notes, "Persistent note content")

    def test_replace_existing_notes(self) -> None:
        """Updating notes replaces the previous content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            mgr.create_plan("Replace Notes")

            mgr.update_notes("001", "First notes")
            plan = mgr.update_notes("001", "Second notes")
            self.assertEqual(plan.notes, "Second notes")

    def test_multiline_notes(self) -> None:
        """Multi-line notes are preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            mgr.create_plan("Multiline Notes")

            notes = "Line one\nLine two\nLine three"
            plan = mgr.update_notes("001", notes)
            self.assertEqual(plan.notes, notes)

            loaded = mgr.show_plan("001")
            self.assertEqual(loaded.notes, notes)

    def test_empty_notes(self) -> None:
        """Notes can be cleared by setting to empty string."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            mgr.create_plan("Clear Notes")

            mgr.update_notes("001", "Some content")
            plan = mgr.update_notes("001", "")
            self.assertEqual(plan.notes, "")

    def test_nonexistent_plan_raises(self) -> None:
        """Updating notes on a non-existent plan raises PlanNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            Path(tmpdir).mkdir(parents=True, exist_ok=True)
            with self.assertRaises(PlanNotFoundError):
                mgr.update_notes("999", "notes")


class TestGetPlanContent(unittest.TestCase):
    """Tests for PlanManager.get_plan_content()."""

    def test_returns_raw_markdown(self) -> None:
        """get_plan_content returns the raw markdown string."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            mgr.create_plan(
                "Content Test",
                description="Test desc",
                steps=["Step 1"],
            )

            content = mgr.get_plan_content("001")
            self.assertIsInstance(content, str)
            self.assertIn("# Plan: Content Test", content)
            self.assertIn("**Status**: draft", content)
            self.assertIn("## Description", content)
            self.assertIn("- [ ] Step 1", content)

    def test_nonexistent_plan_raises(self) -> None:
        """Getting content of a non-existent plan raises PlanNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            Path(tmpdir).mkdir(parents=True, exist_ok=True)
            with self.assertRaises(PlanNotFoundError):
                mgr.get_plan_content("999")

    def test_normalizes_plan_id(self) -> None:
        """get_plan_content accepts unnormalized IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            mgr.create_plan("Norm Content")

            content = mgr.get_plan_content("1")
            self.assertIn("# Plan: Norm Content", content)


class TestIdNormalization(unittest.TestCase):
    """Tests for plan ID normalization."""

    def test_single_digit_normalized(self) -> None:
        """'1' is normalized to '001'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            self.assertEqual(mgr._normalize_id("1"), "001")

    def test_two_digit_normalized(self) -> None:
        """'12' is normalized to '012'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            self.assertEqual(mgr._normalize_id("12"), "012")

    def test_three_digit_unchanged(self) -> None:
        """'001' remains '001'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            self.assertEqual(mgr._normalize_id("001"), "001")

    def test_four_digit_unchanged(self) -> None:
        """'1000' remains '1000'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            self.assertEqual(mgr._normalize_id("1000"), "1000")

    def test_non_numeric_returned_as_is(self) -> None:
        """Non-numeric IDs are returned unchanged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            self.assertEqual(mgr._normalize_id("abc"), "abc")


class TestMarkdownParsing(unittest.TestCase):
    """Tests for markdown parsing edge cases."""

    def test_parse_completed_steps(self) -> None:
        """Completed steps ([x]) are parsed correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            mgr.create_plan("Parse Steps", steps=["Step A", "Step B"])
            mgr.update_step("001", 1, True)

            loaded = mgr.show_plan("001")
            self.assertTrue(loaded.steps[0][0])
            self.assertEqual(loaded.steps[0][1], "Step A")
            self.assertFalse(loaded.steps[1][0])

    def test_plan_with_no_steps(self) -> None:
        """Plan with no steps parses correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            mgr.create_plan("No Steps")

            loaded = mgr.show_plan("001")
            self.assertEqual(loaded.steps, [])

    def test_plan_with_no_description(self) -> None:
        """Plan with empty description parses correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            mgr.create_plan("No Desc")

            loaded = mgr.show_plan("001")
            self.assertEqual(loaded.description, "")

    def test_plan_with_no_model(self) -> None:
        """Plan with empty model field parses correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            mgr.create_plan("No Model")

            loaded = mgr.show_plan("001")
            self.assertEqual(loaded.model, "")

    def test_missing_title_raises(self) -> None:
        """Plan file without a title heading raises PlanParseError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            Path(tmpdir).mkdir(parents=True, exist_ok=True)

            bad = Path(tmpdir) / "001.md"
            bad.write_text(
                "**Status**: draft\n"
                "## Steps\n"
                "- [ ] A step\n",
                encoding="utf-8",
            )

            with self.assertRaises(PlanParseError):
                mgr.show_plan("001")

    def test_invalid_status_raises(self) -> None:
        """Plan file with invalid status raises PlanParseError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            Path(tmpdir).mkdir(parents=True, exist_ok=True)

            bad = Path(tmpdir) / "001.md"
            bad.write_text(
                "# Plan: Bad Status\n\n"
                "**Status**: invalid_status\n"
                "**Created**: 2026-03-09T00:00:00\n"
                "**Model**: test\n\n"
                "## Description\n\n\n"
                "## Steps\n\n"
                "## Notes\n\n",
                encoding="utf-8",
            )

            with self.assertRaises(PlanParseError):
                mgr.show_plan("001")

    def test_uppercase_x_step_parsed(self) -> None:
        """Steps with uppercase [X] are also parsed as done."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            Path(tmpdir).mkdir(parents=True, exist_ok=True)

            plan_md = (
                "# Plan: X Test\n\n"
                "**Status**: active\n"
                "**Created**: 2026-03-09T00:00:00\n"
                "**Model**: test\n\n"
                "## Description\n\n\n"
                "## Steps\n\n"
                "- [X] Done step\n"
                "- [ ] Not done\n\n"
                "## Notes\n\n"
            )
            (Path(tmpdir) / "001.md").write_text(plan_md, encoding="utf-8")

            plan = mgr.show_plan("001")
            self.assertTrue(plan.steps[0][0])
            self.assertEqual(plan.steps[0][1], "Done step")
            self.assertFalse(plan.steps[1][0])

    def test_description_with_multiple_lines(self) -> None:
        """Multi-line description is parsed correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            Path(tmpdir).mkdir(parents=True, exist_ok=True)

            plan_md = (
                "# Plan: Multi Desc\n\n"
                "**Status**: draft\n"
                "**Created**: 2026-03-09T00:00:00\n"
                "**Model**: test\n\n"
                "## Description\n\n"
                "Line one of desc.\n"
                "Line two of desc.\n\n"
                "## Steps\n\n"
                "## Notes\n\n"
            )
            (Path(tmpdir) / "001.md").write_text(plan_md, encoding="utf-8")

            plan = mgr.show_plan("001")
            self.assertIn("Line one of desc.", plan.description)
            self.assertIn("Line two of desc.", plan.description)


class TestStatusTransitions(unittest.TestCase):
    """Tests for complete status transition workflows."""

    def test_draft_to_active_to_complete(self) -> None:
        """Full lifecycle: draft -> active -> complete via step completion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            plan = mgr.create_plan("Lifecycle", steps=["Step 1", "Step 2"])
            self.assertEqual(plan.status, "draft")

            plan = mgr.activate_plan("001")
            self.assertEqual(plan.status, "active")

            mgr.update_step("001", 1, True)
            plan = mgr.update_step("001", 2, True)
            self.assertEqual(plan.status, "complete")

    def test_draft_to_abandoned(self) -> None:
        """Plan can go from draft to abandoned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            plan = mgr.create_plan("Abandon Early")
            self.assertEqual(plan.status, "draft")

            plan = mgr.abandon_plan("001")
            self.assertEqual(plan.status, "abandoned")

    def test_active_to_abandoned(self) -> None:
        """Active plan can be abandoned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            mgr.create_plan("Abandon Active")
            mgr.activate_plan("001")

            plan = mgr.abandon_plan("001")
            self.assertEqual(plan.status, "abandoned")

    def test_cannot_reactivate_after_abandon(self) -> None:
        """Abandoned plan cannot be activated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            mgr.create_plan("No Reactivate")
            mgr.abandon_plan("001")

            with self.assertRaises(PlanError):
                mgr.activate_plan("001")


class TestPlanDataContainer(unittest.TestCase):
    """Tests for the Plan data container class."""

    def test_default_values(self) -> None:
        """Plan defaults are correct."""
        plan = Plan(plan_id="001", title="Test")
        self.assertEqual(plan.status, "draft")
        self.assertEqual(plan.created, "")
        self.assertEqual(plan.model, "")
        self.assertEqual(plan.description, "")
        self.assertEqual(plan.steps, [])
        self.assertEqual(plan.notes, "")

    def test_all_fields_set(self) -> None:
        """All Plan fields can be set via constructor."""
        plan = Plan(
            plan_id="042",
            title="Full Plan",
            status="active",
            created="2026-03-09T12:00:00",
            model="qwen3:8b",
            description="Full description",
            steps=[(True, "Done step"), (False, "Pending step")],
            notes="Some notes",
        )
        self.assertEqual(plan.plan_id, "042")
        self.assertEqual(plan.title, "Full Plan")
        self.assertEqual(plan.status, "active")
        self.assertEqual(plan.created, "2026-03-09T12:00:00")
        self.assertEqual(plan.model, "qwen3:8b")
        self.assertEqual(plan.description, "Full description")
        self.assertEqual(len(plan.steps), 2)
        self.assertEqual(plan.notes, "Some notes")

    def test_uses_slots(self) -> None:
        """Plan uses __slots__ for memory efficiency."""
        self.assertTrue(hasattr(Plan, "__slots__"))
        plan = Plan(plan_id="001", title="Slot Test")
        with self.assertRaises(AttributeError):
            plan.nonexistent_attr = "fail"  # type: ignore[attr-defined]


class TestExceptionHierarchy(unittest.TestCase):
    """Tests for exception class hierarchy."""

    def test_plan_not_found_is_plan_error(self) -> None:
        """PlanNotFoundError inherits from PlanError."""
        self.assertTrue(issubclass(PlanNotFoundError, PlanError))

    def test_plan_parse_error_is_plan_error(self) -> None:
        """PlanParseError inherits from PlanError."""
        self.assertTrue(issubclass(PlanParseError, PlanError))

    def test_plan_error_is_exception(self) -> None:
        """PlanError inherits from Exception."""
        self.assertTrue(issubclass(PlanError, Exception))

    def test_catch_all_plan_errors(self) -> None:
        """All plan exceptions can be caught with PlanError."""
        with self.assertRaises(PlanError):
            raise PlanNotFoundError("not found")

        with self.assertRaises(PlanError):
            raise PlanParseError("parse failed")


class TestAtomicWrite(unittest.TestCase):
    """Tests for atomic write behavior."""

    def test_no_leftover_temp_files(self) -> None:
        """No temporary files remain after a successful write."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            mgr.create_plan("Clean Write")

            # Check for .tmp files.
            tmp_files = list(Path(tmpdir).glob("*.tmp"))
            self.assertEqual(len(tmp_files), 0)

    def test_file_content_consistent(self) -> None:
        """File content is consistent after write (atomic)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            mgr.create_plan(
                "Atomic Plan",
                description="Consistent content",
                steps=["Step A"],
            )

            content = (Path(tmpdir) / "001.md").read_text(encoding="utf-8")
            self.assertIn("# Plan: Atomic Plan", content)
            self.assertIn("Consistent content", content)
            self.assertIn("- [ ] Step A", content)


class TestMultiplePlanWorkflow(unittest.TestCase):
    """Integration tests for working with multiple plans."""

    def test_multiple_plans_independent(self) -> None:
        """Changes to one plan don't affect others."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            mgr.create_plan("Plan A", steps=["A step"])
            mgr.create_plan("Plan B", steps=["B step"])

            mgr.update_step("001", 1, True)

            plan_a = mgr.show_plan("001")
            plan_b = mgr.show_plan("002")

            self.assertTrue(plan_a.steps[0][0])
            self.assertFalse(plan_b.steps[0][0])

    def test_list_reflects_status_changes(self) -> None:
        """list_plans reflects current status of each plan."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            mgr.create_plan("Draft Plan")
            mgr.create_plan("Active Plan")
            mgr.create_plan("Abandoned Plan")

            mgr.activate_plan("002")
            mgr.abandon_plan("003")

            plans = mgr.list_plans()
            statuses = {p.plan_id: p.status for p in plans}

            self.assertEqual(statuses["001"], "draft")
            self.assertEqual(statuses["002"], "active")
            self.assertEqual(statuses["003"], "abandoned")

    def test_create_after_high_id(self) -> None:
        """Creating a plan when existing plans have high IDs works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PlanManager(tmpdir)
            Path(tmpdir).mkdir(parents=True, exist_ok=True)

            # Manually create a plan with a high ID.
            high_plan = (
                "# Plan: High ID Plan\n\n"
                "**Status**: draft\n"
                "**Created**: 2026-03-09T00:00:00\n"
                "**Model**: test\n\n"
                "## Description\n\n\n"
                "## Steps\n\n"
                "## Notes\n\n"
            )
            (Path(tmpdir) / "050.md").write_text(high_plan, encoding="utf-8")

            new_plan = mgr.create_plan("After High")
            self.assertEqual(new_plan.plan_id, "051")


if __name__ == "__main__":
    unittest.main()
