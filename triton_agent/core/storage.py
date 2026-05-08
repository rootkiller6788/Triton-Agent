"""Episode and leaderboard storage: SQLite + JSONL."""

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from triton_agent.core.contract import OpContract
from triton_agent.core.spec import OpState, OpAction, CandidateResult


class EpisodeStore:
    """JSONL-based episode log: appends one record per candidate evaluation."""

    def __init__(self, episodes_dir: str | Path = "episodes") -> None:
        self.episodes_dir = Path(episodes_dir)
        self.episodes_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        op_name: str,
        episode_id: str,
        state: OpState,
        action: OpAction,
        result: CandidateResult,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """Save an episode to a JSONL file under episodes/<op>/<episode_id>/."""
        op_dir = self.episodes_dir / op_name
        op_dir.mkdir(parents=True, exist_ok=True)
        episode_dir = op_dir / episode_id
        episode_dir.mkdir(parents=True, exist_ok=True)

        record = {
            "episode_id": episode_id,
            "timestamp": time.time(),
            "state": state.to_dict(),
            "action": action.to_dict(),
            "result": result.to_dict(),
            "extra": extra or {},
        }

        jsonl_path = episode_dir / "record.json"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, default=str)

        return episode_dir

    def load(self, op_name: str, episode_id: str) -> dict | None:
        """Load a single episode."""
        jsonl_path = self.episodes_dir / op_name / episode_id / "record.json"
        if not jsonl_path.exists():
            return None
        with open(jsonl_path, encoding="utf-8") as f:
            return json.load(f)

    def list_episodes(self, op_name: str) -> list[str]:
        """List episode IDs for an operator."""
        op_dir = self.episodes_dir / op_name
        if not op_dir.exists():
            return []
        return sorted(
            [d.name for d in op_dir.iterdir() if (d / "record.json").exists()]
        )


class LeaderboardStore:
    """SQLite-based leaderboard tracking best configs per-op per-shape."""

    def __init__(self, db_path: str | Path = "leaderboard/leaderboard.sqlite") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS leaderboard (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                op_name TEXT NOT NULL,
                shape TEXT NOT NULL,
                dtype TEXT NOT NULL,
                device TEXT NOT NULL,
                template_id TEXT,
                block_d INTEGER,
                num_warps INTEGER,
                num_stages INTEGER,
                vectorize INTEGER,
                latency_us_p50 REAL,
                speedup REAL,
                reward REAL,
                promoted INTEGER,
                episode_id TEXT,
                timestamp REAL,
                UNIQUE(op_name, shape, dtype, device, template_id)
            )
        """)
        conn.commit()
        conn.close()

    def upsert(
        self,
        op_name: str,
        shape: str,
        dtype: str,
        device: str,
        action: OpAction,
        result: CandidateResult,
        episode_id: str,
    ) -> None:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            INSERT INTO leaderboard
                (op_name, shape, dtype, device, template_id, block_d, num_warps,
                 num_stages, vectorize, latency_us_p50, speedup, reward, promoted, episode_id, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(op_name, shape, dtype, device, template_id) DO UPDATE SET
                block_d=excluded.block_d,
                num_warps=excluded.num_warps,
                num_stages=excluded.num_stages,
                vectorize=excluded.vectorize,
                latency_us_p50=excluded.latency_us_p50,
                speedup=excluded.speedup,
                reward=excluded.reward,
                promoted=excluded.promoted,
                episode_id=excluded.episode_id,
                timestamp=excluded.timestamp
        """, (
            op_name, shape, dtype, device, action.template_id,
            action.block_d, action.num_warps, action.num_stages,
            1 if action.vectorize else 0,
            result.latency_us_p50, result.speedup, result.reward,
            1 if result.promoted else 0, episode_id, time.time(),
        ))
        conn.commit()
        conn.close()

    def query(
        self,
        op_name: str,
        shape: str | None = None,
        dtype: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        where = ["op_name = ?"]
        params: list[Any] = [op_name]
        if shape:
            where.append("shape = ?")
            params.append(shape)
        if dtype:
            where.append("dtype = ?")
            params.append(dtype)
        rows = conn.execute(
            f"SELECT * FROM leaderboard WHERE {' AND '.join(where)} ORDER BY speedup DESC LIMIT ?",
            params + [limit],
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def best_config(self, op_name: str, shape: str, dtype: str) -> dict | None:
        rows = self.query(op_name, shape=shape, dtype=dtype, limit=1)
        return rows[0] if rows else None
