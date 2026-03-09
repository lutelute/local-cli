"""LRU cache for idempotent tool results with file-mtime-based invalidation.

Caches results of read-only tools (read, glob, grep) to avoid redundant
file I/O during a single agent session.  Cache entries are automatically
invalidated when the underlying file's modification time changes.

The cache is session-scoped — call :meth:`ToolCache.clear` when the user
runs ``/clear`` to reset the session.

Design choices:
- Uses ``os.path.getmtime()`` for fast O(1) invalidation (no file reads).
- LRU eviction when the cache exceeds ``max_entries`` (default 256).
- Cache key is ``tool_name:json.dumps(args, sort_keys=True)`` for
  deterministic keying regardless of dict insertion order.
- Thread-safety is **not** required — the CLI runs a single-threaded REPL.
"""

import json
import os
from collections import OrderedDict
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default maximum number of cache entries before LRU eviction.
_DEFAULT_MAX_ENTRIES: int = 256


# ---------------------------------------------------------------------------
# Cache entry
# ---------------------------------------------------------------------------


class _CacheEntry:
    """Internal cache entry storing a tool result and optional file metadata.

    Attributes:
        result: The cached tool result string.
        file_path: The file path tracked for mtime invalidation, or ``None``
            if no specific file is associated.
        mtime: The file's modification time at cache insertion, or ``None``
            if *file_path* is ``None``.
    """

    __slots__ = ("result", "file_path", "mtime")

    def __init__(
        self,
        result: str,
        file_path: str | None = None,
        mtime: float | None = None,
    ) -> None:
        self.result: str = result
        self.file_path: str | None = file_path
        self.mtime: float | None = mtime


# ---------------------------------------------------------------------------
# Tool cache
# ---------------------------------------------------------------------------


class ToolCache:
    """LRU-style cache for idempotent tool results with file-change invalidation.

    The cache maps deterministic keys (derived from tool name and arguments)
    to result strings.  For file-dependent tools the file's ``mtime`` is
    tracked and the entry is considered stale if the file has been modified
    since caching.

    Args:
        max_entries: Maximum number of entries before the least-recently-used
            entry is evicted.  Defaults to :data:`_DEFAULT_MAX_ENTRIES`.
    """

    def __init__(self, max_entries: int = _DEFAULT_MAX_ENTRIES) -> None:
        self._cache: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._max_entries: int = max_entries
        self._hits: int = 0
        self._misses: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, tool_name: str, args: dict[str, Any]) -> str | None:
        """Return a cached result, or ``None`` if stale or missing.

        On a cache hit the entry is promoted to the most-recently-used
        position.  On a cache miss (or stale entry) the miss counter is
        incremented.

        Args:
            tool_name: Name of the tool (e.g. ``"read"``, ``"glob"``).
            args: The tool's argument dictionary.

        Returns:
            The cached result string, or ``None`` if the entry does not
            exist or has been invalidated.
        """
        key = self._make_key(tool_name, args)
        entry = self._cache.get(key)

        if entry is None:
            self._misses += 1
            return None

        # Check mtime-based invalidation.
        if entry.file_path is not None:
            if self._is_stale(entry):
                # Remove the stale entry and report a miss.
                del self._cache[key]
                self._misses += 1
                return None

        # Cache hit — promote to most-recently-used.
        self._cache.move_to_end(key)
        self._hits += 1
        return entry.result

    def put(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: str,
        file_path: str | None = None,
    ) -> None:
        """Store a tool result in the cache.

        If *file_path* is provided, the file's current ``mtime`` is recorded
        for later invalidation checks.  If the cache is full the
        least-recently-used entry is evicted.

        Args:
            tool_name: Name of the tool.
            args: The tool's argument dictionary.
            result: The tool result string to cache.
            file_path: Optional file path to track for mtime invalidation.
        """
        key = self._make_key(tool_name, args)

        # Compute mtime if a file path is provided.
        mtime: float | None = None
        if file_path is not None:
            mtime = self._get_mtime(file_path)

        entry = _CacheEntry(result=result, file_path=file_path, mtime=mtime)

        # If key already exists, update in-place and promote.
        if key in self._cache:
            self._cache[key] = entry
            self._cache.move_to_end(key)
        else:
            # Evict LRU entry if at capacity.
            if len(self._cache) >= self._max_entries:
                self._cache.popitem(last=False)
            self._cache[key] = entry

    def clear(self) -> None:
        """Remove all entries from the cache.

        Called when the user runs ``/clear`` to reset the session.
        Also resets hit/miss counters.
        """
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def invalidate_file(self, file_path: str) -> int:
        """Remove all cache entries associated with a specific file.

        Useful when a write or edit tool modifies a file and all cached
        reads for that file should be discarded.

        Args:
            file_path: The file path whose entries should be removed.

        Returns:
            The number of entries removed.
        """
        keys_to_remove = [
            key for key, entry in self._cache.items()
            if entry.file_path == file_path
        ]
        for key in keys_to_remove:
            del self._cache[key]
        return len(keys_to_remove)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Current number of entries in the cache."""
        return len(self._cache)

    @property
    def max_entries(self) -> int:
        """Maximum number of entries before LRU eviction."""
        return self._max_entries

    @property
    def hits(self) -> int:
        """Total number of cache hits since creation or last clear."""
        return self._hits

    @property
    def misses(self) -> int:
        """Total number of cache misses since creation or last clear."""
        return self._misses

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a fraction in [0.0, 1.0].

        Returns 0.0 if no lookups have been performed.
        """
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    def format_stats(self) -> str:
        """Format cache statistics as a human-readable string.

        Returns:
            A string like ``"Cache: 12 entries, 45 hits / 5 misses (90.0%)"``
            or ``"Cache: empty"`` when the cache has no entries.
        """
        if self.size == 0 and self._hits == 0 and self._misses == 0:
            return "Cache: empty"

        total = self._hits + self._misses
        if total > 0:
            pct = f"{self.hit_rate * 100:.1f}%"
        else:
            pct = "N/A"

        return (
            f"Cache: {self.size} entries, "
            f"{self._hits} hits / {self._misses} misses ({pct})"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_key(tool_name: str, args: dict[str, Any]) -> str:
        """Generate a deterministic cache key from tool name and arguments.

        Uses ``json.dumps`` with ``sort_keys=True`` so that equivalent
        argument dictionaries always produce the same key, regardless of
        insertion order.

        Args:
            tool_name: Name of the tool.
            args: The tool's argument dictionary.

        Returns:
            A string key in the form ``"tool_name:{json_args}"``.
        """
        return f"{tool_name}:{json.dumps(args, sort_keys=True)}"

    @staticmethod
    def _get_mtime(file_path: str) -> float | None:
        """Get the modification time of a file.

        Args:
            file_path: Path to the file.

        Returns:
            The file's mtime as a float, or ``None`` if the file does
            not exist or cannot be accessed.
        """
        try:
            return os.path.getmtime(file_path)
        except OSError:
            return None

    @staticmethod
    def _is_stale(entry: _CacheEntry) -> bool:
        """Check whether a cache entry is stale.

        An entry is stale if its tracked file has been modified since
        the entry was created (i.e. the current mtime differs from the
        recorded mtime).

        Special cases:
        - If the file did not exist at cache time (mtime is ``None``) and
          still does not exist, the entry is **not** stale.
        - If the file did not exist at cache time but now exists, the entry
          **is** stale (the file was created after caching).
        - If the file existed at cache time but no longer exists, the entry
          **is** stale.

        Args:
            entry: The cache entry to check.

        Returns:
            ``True`` if the entry is stale and should be discarded.
        """
        if entry.file_path is None:
            return False

        try:
            current_mtime = os.path.getmtime(entry.file_path)
        except OSError:
            # File does not exist now.  If it also didn't exist at
            # cache time (mtime is None), the entry is still valid.
            return entry.mtime is not None

        # File exists now.  If it didn't exist at cache time, it's stale.
        if entry.mtime is None:
            return True

        return current_mtime != entry.mtime
