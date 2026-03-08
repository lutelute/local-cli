"""Tests for local_cli.tool_cache module."""

import os
import tempfile
import time
import unittest

from local_cli.tool_cache import (
    ToolCache,
    _CacheEntry,
    _DEFAULT_MAX_ENTRIES,
)


# ---------------------------------------------------------------------------
# Cache key generation tests
# ---------------------------------------------------------------------------


class TestMakeKey(unittest.TestCase):
    """Tests for ToolCache._make_key() static method."""

    def test_simple_key(self) -> None:
        """Generates a key from tool name and simple args."""
        key = ToolCache._make_key("read", {"path": "/tmp/test.py"})
        self.assertEqual(key, 'read:{"path": "/tmp/test.py"}')

    def test_empty_args(self) -> None:
        """Generates a key with empty args dict."""
        key = ToolCache._make_key("glob", {})
        self.assertEqual(key, "glob:{}")

    def test_sort_keys_deterministic(self) -> None:
        """Argument order does not affect the cache key."""
        key1 = ToolCache._make_key("grep", {"pattern": "foo", "path": "/bar"})
        key2 = ToolCache._make_key("grep", {"path": "/bar", "pattern": "foo"})
        self.assertEqual(key1, key2)

    def test_different_tools_different_keys(self) -> None:
        """Different tool names produce different keys for the same args."""
        key1 = ToolCache._make_key("read", {"path": "/tmp/a.py"})
        key2 = ToolCache._make_key("glob", {"path": "/tmp/a.py"})
        self.assertNotEqual(key1, key2)

    def test_different_args_different_keys(self) -> None:
        """Different arguments produce different keys for the same tool."""
        key1 = ToolCache._make_key("read", {"path": "/tmp/a.py"})
        key2 = ToolCache._make_key("read", {"path": "/tmp/b.py"})
        self.assertNotEqual(key1, key2)

    def test_nested_args(self) -> None:
        """Nested dict/list arguments are serialized deterministically."""
        args = {"options": {"recursive": True}, "paths": ["/a", "/b"]}
        key = ToolCache._make_key("glob", args)
        self.assertIn("glob:", key)
        self.assertIn('"recursive": true', key)


# ---------------------------------------------------------------------------
# Basic cache operations
# ---------------------------------------------------------------------------


class TestCacheBasicOps(unittest.TestCase):
    """Tests for basic ToolCache get/put/clear operations."""

    def test_empty_cache_returns_none(self) -> None:
        """get() returns None on an empty cache."""
        cache = ToolCache()
        result = cache.get("read", {"path": "/tmp/test.py"})
        self.assertIsNone(result)

    def test_put_and_get(self) -> None:
        """put() then get() returns the cached value."""
        cache = ToolCache()
        cache.put("read", {"path": "/tmp/test.py"}, "file contents here")
        result = cache.get("read", {"path": "/tmp/test.py"})
        self.assertEqual(result, "file contents here")

    def test_get_nonexistent_key(self) -> None:
        """get() returns None for a key that was never put."""
        cache = ToolCache()
        cache.put("read", {"path": "/tmp/a.py"}, "content a")
        result = cache.get("read", {"path": "/tmp/b.py"})
        self.assertIsNone(result)

    def test_put_overwrites_existing(self) -> None:
        """put() with the same key overwrites the previous value."""
        cache = ToolCache()
        args = {"path": "/tmp/test.py"}
        cache.put("read", args, "old content")
        cache.put("read", args, "new content")
        result = cache.get("read", args)
        self.assertEqual(result, "new content")

    def test_clear_removes_all_entries(self) -> None:
        """clear() empties the cache."""
        cache = ToolCache()
        cache.put("read", {"path": "/tmp/a.py"}, "a")
        cache.put("glob", {"pattern": "*.py"}, "b")
        self.assertEqual(cache.size, 2)
        cache.clear()
        self.assertEqual(cache.size, 0)
        self.assertIsNone(cache.get("read", {"path": "/tmp/a.py"}))

    def test_clear_resets_counters(self) -> None:
        """clear() resets hit and miss counters."""
        cache = ToolCache()
        cache.put("read", {"path": "/tmp/a.py"}, "a")
        cache.get("read", {"path": "/tmp/a.py"})  # hit
        cache.get("read", {"path": "/tmp/b.py"})  # miss
        self.assertGreater(cache.hits, 0)
        self.assertGreater(cache.misses, 0)
        cache.clear()
        self.assertEqual(cache.hits, 0)
        self.assertEqual(cache.misses, 0)

    def test_size_property(self) -> None:
        """size property reflects number of cached entries."""
        cache = ToolCache()
        self.assertEqual(cache.size, 0)
        cache.put("read", {"path": "/tmp/a.py"}, "a")
        self.assertEqual(cache.size, 1)
        cache.put("read", {"path": "/tmp/b.py"}, "b")
        self.assertEqual(cache.size, 2)

    def test_max_entries_property(self) -> None:
        """max_entries property returns the configured maximum."""
        cache = ToolCache(max_entries=100)
        self.assertEqual(cache.max_entries, 100)

    def test_default_max_entries(self) -> None:
        """Default max_entries is _DEFAULT_MAX_ENTRIES (256)."""
        cache = ToolCache()
        self.assertEqual(cache.max_entries, _DEFAULT_MAX_ENTRIES)
        self.assertEqual(cache.max_entries, 256)


# ---------------------------------------------------------------------------
# LRU eviction tests
# ---------------------------------------------------------------------------


class TestLRUEviction(unittest.TestCase):
    """Tests for LRU eviction when cache exceeds max_entries."""

    def test_evicts_oldest_when_full(self) -> None:
        """Oldest entry is evicted when cache exceeds max_entries."""
        cache = ToolCache(max_entries=3)
        cache.put("read", {"path": "/a"}, "a")
        cache.put("read", {"path": "/b"}, "b")
        cache.put("read", {"path": "/c"}, "c")
        # Cache is full at 3. Adding a 4th should evict "/a".
        cache.put("read", {"path": "/d"}, "d")
        self.assertEqual(cache.size, 3)
        self.assertIsNone(cache.get("read", {"path": "/a"}))
        self.assertEqual(cache.get("read", {"path": "/b"}), "b")
        self.assertEqual(cache.get("read", {"path": "/c"}), "c")
        self.assertEqual(cache.get("read", {"path": "/d"}), "d")

    def test_get_promotes_entry(self) -> None:
        """Accessing an entry promotes it, protecting it from eviction."""
        cache = ToolCache(max_entries=3)
        cache.put("read", {"path": "/a"}, "a")
        cache.put("read", {"path": "/b"}, "b")
        cache.put("read", {"path": "/c"}, "c")
        # Access "/a" to promote it to most-recently-used.
        cache.get("read", {"path": "/a"})
        # Now add "/d" — "/b" should be evicted (it's now the LRU).
        cache.put("read", {"path": "/d"}, "d")
        self.assertEqual(cache.size, 3)
        self.assertIsNone(cache.get("read", {"path": "/b"}))
        self.assertEqual(cache.get("read", {"path": "/a"}), "a")

    def test_put_overwrites_does_not_evict(self) -> None:
        """Overwriting an existing key does not trigger eviction."""
        cache = ToolCache(max_entries=3)
        cache.put("read", {"path": "/a"}, "a")
        cache.put("read", {"path": "/b"}, "b")
        cache.put("read", {"path": "/c"}, "c")
        # Overwrite "/a" — no eviction should happen.
        cache.put("read", {"path": "/a"}, "a_updated")
        self.assertEqual(cache.size, 3)
        self.assertEqual(cache.get("read", {"path": "/a"}), "a_updated")
        self.assertEqual(cache.get("read", {"path": "/b"}), "b")
        self.assertEqual(cache.get("read", {"path": "/c"}), "c")

    def test_max_entries_one(self) -> None:
        """Cache with max_entries=1 keeps only the last put."""
        cache = ToolCache(max_entries=1)
        cache.put("read", {"path": "/a"}, "a")
        cache.put("read", {"path": "/b"}, "b")
        self.assertEqual(cache.size, 1)
        self.assertIsNone(cache.get("read", {"path": "/a"}))
        self.assertEqual(cache.get("read", {"path": "/b"}), "b")

    def test_eviction_order_with_multiple_puts(self) -> None:
        """Multiple entries are evicted in correct LRU order."""
        cache = ToolCache(max_entries=2)
        cache.put("read", {"path": "/a"}, "a")
        cache.put("read", {"path": "/b"}, "b")
        # Evict "/a" by adding "/c".
        cache.put("read", {"path": "/c"}, "c")
        self.assertIsNone(cache.get("read", {"path": "/a"}))
        # Now "/b" is the LRU. Adding "/d" should evict "/b".
        cache.put("read", {"path": "/d"}, "d")
        self.assertIsNone(cache.get("read", {"path": "/b"}))
        self.assertEqual(cache.get("read", {"path": "/c"}), "c")
        self.assertEqual(cache.get("read", {"path": "/d"}), "d")


# ---------------------------------------------------------------------------
# File-mtime-based invalidation tests
# ---------------------------------------------------------------------------


class TestMtimeInvalidation(unittest.TestCase):
    """Tests for file-mtime-based cache invalidation."""

    def test_valid_entry_with_unchanged_file(self) -> None:
        """Cache entry is valid when file mtime has not changed."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("original content")
            f.flush()
            tmp_path = f.name

        try:
            cache = ToolCache()
            cache.put("read", {"path": tmp_path}, "original content", file_path=tmp_path)
            result = cache.get("read", {"path": tmp_path})
            self.assertEqual(result, "original content")
        finally:
            os.unlink(tmp_path)

    def test_stale_entry_when_file_modified(self) -> None:
        """Cache entry is invalidated when file mtime changes."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("original content")
            f.flush()
            tmp_path = f.name

        try:
            cache = ToolCache()
            cache.put("read", {"path": tmp_path}, "original content", file_path=tmp_path)

            # Modify the file to change its mtime.
            # Use os.utime to ensure mtime is different.
            original_mtime = os.path.getmtime(tmp_path)
            os.utime(tmp_path, (original_mtime + 1, original_mtime + 1))

            result = cache.get("read", {"path": tmp_path})
            self.assertIsNone(result)
        finally:
            os.unlink(tmp_path)

    def test_stale_entry_when_file_deleted(self) -> None:
        """Cache entry is invalidated when file is deleted."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("content")
            f.flush()
            tmp_path = f.name

        cache = ToolCache()
        cache.put("read", {"path": tmp_path}, "content", file_path=tmp_path)

        # Delete the file.
        os.unlink(tmp_path)

        result = cache.get("read", {"path": tmp_path})
        self.assertIsNone(result)

    def test_stale_entry_removed_from_cache(self) -> None:
        """A stale entry is removed from the cache on access."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("content")
            f.flush()
            tmp_path = f.name

        try:
            cache = ToolCache()
            cache.put("read", {"path": tmp_path}, "content", file_path=tmp_path)
            self.assertEqual(cache.size, 1)

            # Modify the file.
            original_mtime = os.path.getmtime(tmp_path)
            os.utime(tmp_path, (original_mtime + 1, original_mtime + 1))

            # Access triggers removal.
            cache.get("read", {"path": tmp_path})
            self.assertEqual(cache.size, 0)
        finally:
            os.unlink(tmp_path)

    def test_entry_without_file_path_never_stale(self) -> None:
        """Cache entries without a file_path are never invalidated by mtime."""
        cache = ToolCache()
        cache.put("glob", {"pattern": "*.py"}, "file1.py\nfile2.py")
        result = cache.get("glob", {"pattern": "*.py"})
        self.assertEqual(result, "file1.py\nfile2.py")

    def test_get_mtime_nonexistent_file(self) -> None:
        """_get_mtime returns None for nonexistent files."""
        mtime = ToolCache._get_mtime("/nonexistent/path/file.txt")
        self.assertIsNone(mtime)

    def test_get_mtime_existing_file(self) -> None:
        """_get_mtime returns a float for existing files."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            tmp_path = f.name

        try:
            mtime = ToolCache._get_mtime(tmp_path)
            self.assertIsInstance(mtime, float)
            self.assertGreater(mtime, 0)
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# File invalidation by path
# ---------------------------------------------------------------------------


class TestInvalidateFile(unittest.TestCase):
    """Tests for ToolCache.invalidate_file()."""

    def test_invalidate_removes_matching_entries(self) -> None:
        """invalidate_file() removes all entries for the given file path."""
        cache = ToolCache()
        cache.put("read", {"path": "/tmp/a.py"}, "a", file_path="/tmp/a.py")
        cache.put("grep", {"path": "/tmp/a.py", "pattern": "foo"}, "match", file_path="/tmp/a.py")
        cache.put("read", {"path": "/tmp/b.py"}, "b", file_path="/tmp/b.py")

        removed = cache.invalidate_file("/tmp/a.py")
        self.assertEqual(removed, 2)
        self.assertEqual(cache.size, 1)
        self.assertIsNone(cache.get("read", {"path": "/tmp/a.py"}))
        self.assertEqual(cache.get("read", {"path": "/tmp/b.py"}), "b")

    def test_invalidate_nonexistent_file(self) -> None:
        """invalidate_file() returns 0 when no entries match."""
        cache = ToolCache()
        cache.put("read", {"path": "/tmp/a.py"}, "a", file_path="/tmp/a.py")
        removed = cache.invalidate_file("/tmp/nonexistent.py")
        self.assertEqual(removed, 0)
        self.assertEqual(cache.size, 1)

    def test_invalidate_empty_cache(self) -> None:
        """invalidate_file() on empty cache returns 0."""
        cache = ToolCache()
        removed = cache.invalidate_file("/tmp/a.py")
        self.assertEqual(removed, 0)


# ---------------------------------------------------------------------------
# Hit/miss statistics tests
# ---------------------------------------------------------------------------


class TestCacheStatistics(unittest.TestCase):
    """Tests for ToolCache hit/miss tracking and statistics."""

    def test_initial_stats_zero(self) -> None:
        """New cache has zero hits and misses."""
        cache = ToolCache()
        self.assertEqual(cache.hits, 0)
        self.assertEqual(cache.misses, 0)

    def test_hit_increments(self) -> None:
        """Successful get() increments the hit counter."""
        cache = ToolCache()
        cache.put("read", {"path": "/a"}, "content")
        cache.get("read", {"path": "/a"})
        self.assertEqual(cache.hits, 1)
        self.assertEqual(cache.misses, 0)

    def test_miss_increments(self) -> None:
        """Failed get() increments the miss counter."""
        cache = ToolCache()
        cache.get("read", {"path": "/a"})
        self.assertEqual(cache.hits, 0)
        self.assertEqual(cache.misses, 1)

    def test_stale_entry_counts_as_miss(self) -> None:
        """Accessing a stale entry counts as a miss."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("content")
            f.flush()
            tmp_path = f.name

        try:
            cache = ToolCache()
            cache.put("read", {"path": tmp_path}, "content", file_path=tmp_path)

            # Modify the file to make the entry stale.
            original_mtime = os.path.getmtime(tmp_path)
            os.utime(tmp_path, (original_mtime + 1, original_mtime + 1))

            cache.get("read", {"path": tmp_path})
            self.assertEqual(cache.hits, 0)
            self.assertEqual(cache.misses, 1)
        finally:
            os.unlink(tmp_path)

    def test_hit_rate_zero_no_lookups(self) -> None:
        """hit_rate returns 0.0 when no lookups have been done."""
        cache = ToolCache()
        self.assertEqual(cache.hit_rate, 0.0)

    def test_hit_rate_all_hits(self) -> None:
        """hit_rate is 1.0 when all lookups are hits."""
        cache = ToolCache()
        cache.put("read", {"path": "/a"}, "a")
        cache.get("read", {"path": "/a"})
        cache.get("read", {"path": "/a"})
        self.assertAlmostEqual(cache.hit_rate, 1.0)

    def test_hit_rate_all_misses(self) -> None:
        """hit_rate is 0.0 when all lookups are misses."""
        cache = ToolCache()
        cache.get("read", {"path": "/a"})
        cache.get("read", {"path": "/b"})
        self.assertAlmostEqual(cache.hit_rate, 0.0)

    def test_hit_rate_mixed(self) -> None:
        """hit_rate is correct for mixed hits and misses."""
        cache = ToolCache()
        cache.put("read", {"path": "/a"}, "a")
        cache.get("read", {"path": "/a"})  # hit
        cache.get("read", {"path": "/b"})  # miss
        cache.get("read", {"path": "/a"})  # hit
        cache.get("read", {"path": "/c"})  # miss
        # 2 hits, 2 misses = 50%
        self.assertAlmostEqual(cache.hit_rate, 0.5)

    def test_format_stats_empty(self) -> None:
        """format_stats() shows 'Cache: empty' for fresh cache."""
        cache = ToolCache()
        self.assertEqual(cache.format_stats(), "Cache: empty")

    def test_format_stats_with_data(self) -> None:
        """format_stats() shows entries, hits, and misses."""
        cache = ToolCache()
        cache.put("read", {"path": "/a"}, "a")
        cache.get("read", {"path": "/a"})  # hit
        cache.get("read", {"path": "/b"})  # miss
        stats = cache.format_stats()
        self.assertIn("1 entries", stats)
        self.assertIn("1 hits", stats)
        self.assertIn("1 misses", stats)
        self.assertIn("50.0%", stats)

    def test_format_stats_all_hits(self) -> None:
        """format_stats() shows 100.0% for all hits."""
        cache = ToolCache()
        cache.put("read", {"path": "/a"}, "a")
        cache.get("read", {"path": "/a"})
        stats = cache.format_stats()
        self.assertIn("100.0%", stats)


# ---------------------------------------------------------------------------
# Cache entry tests
# ---------------------------------------------------------------------------


class TestCacheEntry(unittest.TestCase):
    """Tests for the _CacheEntry internal class."""

    def test_defaults(self) -> None:
        """_CacheEntry defaults file_path and mtime to None."""
        entry = _CacheEntry(result="content")
        self.assertEqual(entry.result, "content")
        self.assertIsNone(entry.file_path)
        self.assertIsNone(entry.mtime)

    def test_with_file_metadata(self) -> None:
        """_CacheEntry stores file path and mtime."""
        entry = _CacheEntry(result="content", file_path="/tmp/a.py", mtime=1234567890.0)
        self.assertEqual(entry.file_path, "/tmp/a.py")
        self.assertEqual(entry.mtime, 1234567890.0)

    def test_slots(self) -> None:
        """_CacheEntry uses __slots__ for memory efficiency."""
        entry = _CacheEntry(result="content")
        self.assertTrue(hasattr(entry, "__slots__"))
        with self.assertRaises(AttributeError):
            entry.extra_attr = "should fail"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and boundary conditions."""

    def test_empty_string_result(self) -> None:
        """Cache stores and retrieves empty string results."""
        cache = ToolCache()
        cache.put("read", {"path": "/empty"}, "")
        result = cache.get("read", {"path": "/empty"})
        self.assertEqual(result, "")

    def test_large_result_string(self) -> None:
        """Cache handles large result strings."""
        cache = ToolCache()
        large_content = "x" * 100_000
        cache.put("read", {"path": "/large"}, large_content)
        result = cache.get("read", {"path": "/large"})
        self.assertEqual(result, large_content)

    def test_special_characters_in_args(self) -> None:
        """Cache handles special characters in argument values."""
        cache = ToolCache()
        args = {"path": "/tmp/file with spaces & special chars!.py"}
        cache.put("read", args, "content")
        result = cache.get("read", args)
        self.assertEqual(result, "content")

    def test_unicode_in_result(self) -> None:
        """Cache handles Unicode characters in results."""
        cache = ToolCache()
        content = "日本語テスト 🎉 émojis"
        cache.put("read", {"path": "/unicode"}, content)
        result = cache.get("read", {"path": "/unicode"})
        self.assertEqual(result, content)

    def test_multiple_tools_same_path(self) -> None:
        """Different tools with the same path have separate cache entries."""
        cache = ToolCache()
        cache.put("read", {"path": "/a.py"}, "file content", file_path="/a.py")
        cache.put("grep", {"path": "/a.py", "pattern": "def"}, "grep result", file_path="/a.py")
        self.assertEqual(cache.size, 2)
        self.assertEqual(cache.get("read", {"path": "/a.py"}), "file content")
        self.assertEqual(
            cache.get("grep", {"path": "/a.py", "pattern": "def"}), "grep result"
        )

    def test_put_with_nonexistent_file_path(self) -> None:
        """put() with a nonexistent file_path stores None mtime."""
        cache = ToolCache()
        cache.put(
            "read",
            {"path": "/nonexistent"},
            "content",
            file_path="/nonexistent/file.py",
        )
        # File didn't exist at cache time and still doesn't — not stale.
        result = cache.get("read", {"path": "/nonexistent"})
        self.assertEqual(result, "content")

    def test_stale_when_file_created_after_cache(self) -> None:
        """Entry is stale if the tracked file is created after caching."""
        cache = ToolCache()
        tmp_path = os.path.join(tempfile.gettempdir(), "_tool_cache_test_new.py")
        # Ensure file doesn't exist.
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

        cache.put("read", {"path": tmp_path}, "empty", file_path=tmp_path)

        try:
            # Create the file — entry should now be stale.
            with open(tmp_path, "w") as f:
                f.write("new content")
            result = cache.get("read", {"path": tmp_path})
            self.assertIsNone(result)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_default_max_entries_constant(self) -> None:
        """_DEFAULT_MAX_ENTRIES is 256."""
        self.assertEqual(_DEFAULT_MAX_ENTRIES, 256)


# ---------------------------------------------------------------------------
# Integration-style tests
# ---------------------------------------------------------------------------


class TestCacheWorkflow(unittest.TestCase):
    """Integration-style tests simulating real tool cache usage patterns."""

    def test_read_cache_hit_workflow(self) -> None:
        """Simulates: read file -> cache -> read same file -> cache hit."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def hello(): pass")
            f.flush()
            tmp_path = f.name

        try:
            cache = ToolCache()
            args = {"path": tmp_path}

            # First read — miss.
            result1 = cache.get("read", args)
            self.assertIsNone(result1)

            # Simulate tool execution and caching.
            cache.put("read", args, "def hello(): pass", file_path=tmp_path)

            # Second read — hit.
            result2 = cache.get("read", args)
            self.assertEqual(result2, "def hello(): pass")
            self.assertEqual(cache.hits, 1)
            self.assertEqual(cache.misses, 1)
        finally:
            os.unlink(tmp_path)

    def test_write_invalidates_read_cache(self) -> None:
        """Simulates: read file -> cache -> write file -> read invalidated."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("original")
            f.flush()
            tmp_path = f.name

        try:
            cache = ToolCache()
            args = {"path": tmp_path}

            # Cache a read result.
            cache.put("read", args, "original", file_path=tmp_path)
            self.assertEqual(cache.get("read", args), "original")

            # Simulate a write tool modifying the file.
            cache.invalidate_file(tmp_path)

            # Read should now miss.
            result = cache.get("read", args)
            self.assertIsNone(result)
        finally:
            os.unlink(tmp_path)

    def test_session_clear_workflow(self) -> None:
        """Simulates: cache entries -> /clear -> cache empty."""
        cache = ToolCache()
        cache.put("read", {"path": "/a"}, "a")
        cache.put("glob", {"pattern": "*.py"}, "files")
        cache.put("grep", {"pattern": "def"}, "matches")
        self.assertEqual(cache.size, 3)

        # User runs /clear.
        cache.clear()
        self.assertEqual(cache.size, 0)
        self.assertEqual(cache.hits, 0)
        self.assertEqual(cache.misses, 0)

    def test_glob_without_file_path(self) -> None:
        """glob results cached without file_path never go stale from mtime."""
        cache = ToolCache()
        cache.put("glob", {"pattern": "*.py"}, "a.py\nb.py")

        # Multiple reads should all hit.
        for _ in range(5):
            result = cache.get("glob", {"pattern": "*.py"})
            self.assertEqual(result, "a.py\nb.py")

        self.assertEqual(cache.hits, 5)
        self.assertEqual(cache.misses, 0)


if __name__ == "__main__":
    unittest.main()
