"""Compile cache: avoids recompiling identical configs across episodes.

Key insight: the (template_id, BLOCK_SIZE, num_warps, num_stages, vectorize,
op_name, dtype) tuple uniquely identifies a compile result. Caching eliminates
redundant JIT overhead.
"""

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

from triton_agent.core.spec import OpAction


CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS compile_cache (
    cache_key TEXT PRIMARY KEY,
    template_id TEXT NOT NULL,
    op_name TEXT NOT NULL,
    block_size INTEGER,
    num_warps INTEGER,
    num_stages INTEGER,
    vectorize INTEGER,
    dtype TEXT,
    compile_time_s REAL,
    success INTEGER,
    error_log TEXT,
    created_at REAL
)
"""


class CompileCache:
    """SQLite-backed compile result cache.

    Eliminates redundant Triton JIT compilation for configurations
    that have been compiled before, even across different optimization runs.
    """

    def __init__(self, db_path: str | Path = "leaderboard/compile_cache.sqlite") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(str(self.db_path))
        conn.executescript(CACHE_SCHEMA)
        conn.commit()
        conn.close()

    def _make_key(
        self,
        template_id: str,
        action: OpAction,
        op_name: str,
        dtype: str,
        B: int, T: int, D: int,
    ) -> str:
        """Produce a stable cache key from compile-influencing parameters."""
        raw = (
            f"{template_id}|{op_name}|{action.block_d}|{action.num_warps}"
            f"|{action.num_stages}|{int(action.vectorize)}|{dtype}|{B}|{T}|{D}"
        )
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(
        self,
        template_id: str,
        action: OpAction,
        op_name: str,
        dtype: str,
        B: int, T: int, D: int,
    ) -> Optional[dict]:
        """Retrieve a cached compile result, or None."""
        key = self._make_key(template_id, action, op_name, dtype, B, T, D)
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM compile_cache WHERE cache_key = ?", (key,)
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    def put(
        self,
        template_id: str,
        action: OpAction,
        op_name: str,
        dtype: str,
        B: int, T: int, D: int,
        compile_time_s: float,
        success: bool,
        error_log: str = "",
    ) -> None:
        """Store a compile result in the cache."""
        key = self._make_key(template_id, action, op_name, dtype, B, T, D)
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            INSERT OR REPLACE INTO compile_cache
                (cache_key, template_id, op_name, block_size, num_warps,
                 num_stages, vectorize, dtype, compile_time_s, success, error_log, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            key, template_id, op_name, action.block_d, action.num_warps,
            action.num_stages, 1 if action.vectorize else 0, dtype,
            compile_time_s, 1 if success else 0, error_log, time.time(),
        ))
        conn.commit()
        conn.close()

    def stats(self) -> dict:
        """Return cache statistics."""
        conn = sqlite3.connect(str(self.db_path))
        total = conn.execute("SELECT COUNT(*) FROM compile_cache").fetchone()[0]
        hits = conn.execute("SELECT COUNT(*) FROM compile_cache WHERE success = 1").fetchone()[0]
        conn.close()
        return {"total_entries": total, "successful": hits, "failed": total - hits}

    def clear(self) -> None:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("DELETE FROM compile_cache")
        conn.commit()
        conn.close()
