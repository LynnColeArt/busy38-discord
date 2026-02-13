"""
Discord transcript logger for Busy38.

Writes observed Discord messages into Busy38's DuckDB chat log DB so other
systems ("boards", analyzers, memory tooling) can see the raw channel flow.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

try:
    import duckdb
except Exception:  # pragma: no cover
    duckdb = None

logger = logging.getLogger(__name__)


class DiscordTranscriptLogger:
    def __init__(self, data_dir: str = "./data/memory"):
        self.data_dir = Path(data_dir)
        self._conn = None

    def connect(self) -> None:
        if duckdb is None:
            raise RuntimeError("duckdb not installed")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        path = self.data_dir / "chat_logs.duckdb"
        self._conn = duckdb.connect(str(path))
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_entries (
                id VARCHAR PRIMARY KEY, timestamp TIMESTAMP, content TEXT,
                vector TEXT, project_id VARCHAR, participants TEXT, topics TEXT,
                metadata TEXT, expires_at TIMESTAMP, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS chat_timestamp_idx ON chat_entries(timestamp)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS chat_project_idx ON chat_entries(project_id)")

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
        self._conn = None

    def log_message(
        self,
        *,
        source_id: str,
        timestamp: Optional[datetime],
        content: str,
        project_id: str,
        metadata: Dict[str, Any],
        participants: Optional[list[int]] = None,
        topics: Optional[list[str]] = None,
        expires_at: Optional[datetime] = None,
    ) -> None:
        if self._conn is None:
            self.connect()

        ts = timestamp or datetime.now(timezone.utc)

        # Insert idempotently. If the same Discord message is seen twice, ignore.
        try:
            self._conn.execute(
                """
                INSERT INTO chat_entries
                  (id, timestamp, content, vector, project_id, participants, topics, metadata, expires_at)
                SELECT ?, ?, ?, ?, ?, ?, ?, ?, ?
                WHERE NOT EXISTS (SELECT 1 FROM chat_entries WHERE id = ?)
                """,
                [
                    source_id,
                    ts.isoformat(),
                    content,
                    "[]",
                    project_id,
                    json.dumps(participants or []),
                    json.dumps(topics or []),
                    json.dumps(metadata, separators=(",", ":"), sort_keys=True),
                    expires_at.isoformat() if expires_at else None,
                    source_id,
                ],
            )
        except Exception as e:
            logger.debug(f"Failed to log discord message {source_id}: {e}")

    def search(
        self,
        *,
        query: str,
        project_id: Optional[str] = None,
        project_id_prefix: str = "discord:",
        since: Optional[datetime] = None,
        max_messages: int = 2000,
        context: int = 80,
        case_sensitive: bool = True,
        regex: bool = False,
        snippets_per_message: int = 3,
        max_results: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Pattern search across chat_entries.

        Modeled after RangeWriter4-a's workspace/search:
        - compile regex (or escape for literal)
        - scan content
        - return small snippets around match sites
        """
        if self._conn is None:
            self.connect()

        if not query:
            return []

        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            pattern = re.compile(query if regex else re.escape(query), flags)
        except re.error as exc:
            raise ValueError(f"invalid regex: {exc}")

        where = "project_id = ?" if project_id else "project_id LIKE ?"
        param = project_id if project_id else f"{project_id_prefix}%"

        clauses = [where]
        params: list[Any] = [param]
        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(since.isoformat())
        params.append(int(max_messages))
        where_sql = " AND ".join(clauses)

        rows = self._conn.execute(
            f"""
            SELECT id, timestamp, content, project_id, metadata
            FROM chat_entries
            WHERE {where_sql}
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            params,
        ).fetchall()

        results: List[Dict[str, Any]] = []
        for row in rows:
            rid, ts, content, pid, meta = row
            if not content:
                continue
            matches = list(pattern.finditer(content))
            if not matches:
                continue

            snippets: List[str] = []
            for m in matches[:snippets_per_message]:
                start = max(0, m.start() - context)
                end = min(len(content), m.end() + context)
                snippets.append(content[start:end])

            try:
                meta_obj = json.loads(meta) if isinstance(meta, str) and meta else {}
            except Exception:
                meta_obj = {}

            results.append(
                {
                    "id": rid,
                    "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                    "project_id": pid,
                    "snippets": snippets,
                    "metadata": meta_obj,
                }
            )
            if len(results) >= max_results:
                break

        return results

    def context_around(
        self,
        *,
        source_id: str,
        before: int = 8,
        after: int = 8,
    ) -> List[Dict[str, Any]]:
        """Return message context around a specific chat_entries.id (discord:<message_id>)."""
        if self._conn is None:
            self.connect()

        row = self._conn.execute(
            "SELECT id, timestamp, content, project_id, metadata FROM chat_entries WHERE id = ?",
            [source_id],
        ).fetchone()
        if not row:
            return []

        rid, ts, content, pid, meta = row

        before_rows = self._conn.execute(
            """
            SELECT id, timestamp, content, project_id, metadata
            FROM chat_entries
            WHERE project_id = ? AND timestamp < ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            [pid, ts, int(before)],
        ).fetchall()
        after_rows = self._conn.execute(
            """
            SELECT id, timestamp, content, project_id, metadata
            FROM chat_entries
            WHERE project_id = ? AND timestamp > ?
            ORDER BY timestamp ASC
            LIMIT ?
            """,
            [pid, ts, int(after)],
        ).fetchall()

        combined = list(reversed(before_rows)) + [row] + list(after_rows)
        out: List[Dict[str, Any]] = []
        for r in combined:
            rid2, ts2, content2, pid2, meta2 = r
            try:
                meta_obj = json.loads(meta2) if isinstance(meta2, str) and meta2 else {}
            except Exception:
                meta_obj = {}
            out.append(
                {
                    "id": rid2,
                    "timestamp": ts2.isoformat() if hasattr(ts2, "isoformat") else str(ts2),
                    "project_id": pid2,
                    "content": content2 or "",
                    "metadata": meta_obj,
                }
            )
        return out

    def recent_messages(
        self,
        *,
        project_id: str,
        since: Optional[datetime] = None,
        max_age_hours: int = 72,
        now: Optional[datetime] = None,
        limit: int = 800,
    ) -> List[Dict[str, Any]]:
        """
        Fetch recent messages for one discord project/channel in chronological order.
        """
        if self._conn is None:
            self.connect()

        if since is None and int(max_age_hours) > 0:
            ref = now or datetime.now(timezone.utc)
            since = ref - timedelta(hours=int(max_age_hours))

        clauses = ["project_id = ?"]
        params: List[Any] = [project_id]
        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(since.isoformat())
        params.append(int(limit))

        rows = self._conn.execute(
            f"""
            SELECT id, timestamp, content, metadata
            FROM chat_entries
            WHERE {' AND '.join(clauses)}
            ORDER BY timestamp ASC
            LIMIT ?
            """,
            params,
        ).fetchall()

        out: List[Dict[str, Any]] = []
        for rid, ts, content, meta in rows:
            try:
                meta_obj = json.loads(meta) if isinstance(meta, str) and meta else {}
            except Exception:
                meta_obj = {}
            out.append(
                {
                    "id": rid,
                    "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                    "content": content or "",
                    "metadata": meta_obj,
                }
            )
        return out
