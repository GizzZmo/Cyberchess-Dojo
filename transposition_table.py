"""
Transposition Table for Cyberchess-Dojo.

A fixed-capacity hash map that caches position evaluations to avoid
re-computing identical board states across different search branches.

Entry flags follow the standard alpha-beta convention:
  EXACT  — the stored score is the true minimax value.
  LOWER  — the score is a lower bound (failed high / beta cutoff).
  UPPER  — the score is an upper bound (failed low / all-node).

Replacement policy: depth-preferred — a new entry replaces an existing one
only if the new search depth is at least as deep, ensuring that shallow results
never evict hard-won deep evaluations.  When the table is full the oldest
entry (lowest insertion index) is evicted.

Usage::

    from transposition_table import TranspositionTable, TTFlag

    tt = TranspositionTable(max_size=1_000_000)

    # Store a result
    tt.store(board.fen(), score=42, depth=6, move="e2e4", flag=TTFlag.EXACT)

    # Look up a position
    if tt.contains(fen) and tt.get_depth(fen) >= required_depth:
        cached_score = tt.get_score(fen)
        best_move    = tt.get_move(fen)
"""

from __future__ import annotations

from collections import OrderedDict
from enum import IntEnum
from typing import Optional


class TTFlag(IntEnum):
    """Type of bound stored in a transposition-table entry."""
    EXACT = 0   # true minimax value
    LOWER = 1   # lower bound (beta cutoff)
    UPPER = 2   # upper bound (alpha node)


class _TTEntry:
    """A single transposition-table record."""

    __slots__ = ("score", "depth", "move", "flag")

    def __init__(self, score: int, depth: int, move: Optional[str], flag: TTFlag) -> None:
        self.score = score
        self.depth = depth
        self.move  = move
        self.flag  = flag


class TranspositionTable:
    """
    Fixed-capacity transposition table using a depth-preferred replacement policy.

    Args:
        max_size: Maximum number of entries.  Defaults to 1 000 000.
                  Set to a smaller value for memory-constrained environments.
    """

    def __init__(self, max_size: int = 1_000_000) -> None:
        self._max_size = max(1, max_size)
        self._table: OrderedDict[str, _TTEntry] = OrderedDict()
        self._hits   = 0
        self._misses = 0
        self._stores = 0

    # ------------------------------------------------------------------
    # Core interface (mirrored after the C++ pseudo-code in the spec)
    # ------------------------------------------------------------------

    def contains(self, key: str) -> bool:
        """Return True if *key* has an entry in the table."""
        hit = key in self._table
        if hit:
            self._hits += 1
        else:
            self._misses += 1
        return hit

    def get_depth(self, key: str) -> int:
        """Return the search depth stored for *key* (0 if not present)."""
        entry = self._table.get(key)
        return entry.depth if entry is not None else 0

    def get_score(self, key: str) -> Optional[int]:
        """Return the cached score for *key*, or ``None`` if absent."""
        entry = self._table.get(key)
        return entry.score if entry is not None else None

    def get_move(self, key: str) -> Optional[str]:
        """Return the best move (UCI string) stored for *key*, or ``None``."""
        entry = self._table.get(key)
        return entry.move if entry is not None else None

    def get_flag(self, key: str) -> Optional[TTFlag]:
        """Return the bound flag for *key*, or ``None`` if absent."""
        entry = self._table.get(key)
        return entry.flag if entry is not None else None

    def store(
        self,
        key: str,
        score: int,
        depth: int,
        move: Optional[str] = None,
        flag: TTFlag = TTFlag.EXACT,
    ) -> None:
        """
        Insert or update a table entry.

        Depth-preferred policy: an existing deeper entry is kept; a shallower
        (or equal-depth) entry is replaced.  When the table is full the oldest
        entry is evicted.
        """
        existing = self._table.get(key)
        if existing is not None and existing.depth > depth:
            return  # keep the deeper result

        if key not in self._table and len(self._table) >= self._max_size:
            # Evict the oldest entry (FIFO among equal-depth candidates).
            self._table.popitem(last=False)

        self._table[key] = _TTEntry(score=score, depth=depth, move=move, flag=flag)
        self._stores += 1

    def clear(self) -> None:
        """Remove all entries."""
        self._table.clear()
        self._hits = self._misses = self._stores = 0

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Current number of entries."""
        return len(self._table)

    @property
    def hits(self) -> int:
        """Cumulative TT-hit count."""
        return self._hits

    @property
    def misses(self) -> int:
        """Cumulative TT-miss count."""
        return self._misses

    @property
    def stores(self) -> int:
        """Cumulative store count."""
        return self._stores

    def hit_rate(self) -> float:
        """Cache hit rate (0.0–1.0). Returns 0.0 if no lookups yet."""
        total = self._hits + self._misses
        return self._hits / total if total else 0.0

    def __repr__(self) -> str:
        return (
            f"TranspositionTable(size={self.size}/{self._max_size}, "
            f"hit_rate={self.hit_rate():.1%})"
        )
