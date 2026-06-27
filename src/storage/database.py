"""
SQLite database for session state and experiment results.

Verified: sqlite3 is Python stdlib, no external dependency.
All SQL is parameterized (? placeholders) to prevent injection.
"""

from __future__ import annotations

import json
import sqlite3
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "data/cryptodrift.db"


class CryptoDriftDB:
    """SQLite database manager for CryptoDrift."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self) -> None:
        """Open database connection and create tables if needed."""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()
        logger.info("Database connected: %s", self._db_path)

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def _create_tables(self) -> None:
        """Create all tables if they don't exist."""
        assert self._conn is not None
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                start_time REAL NOT NULL,
                end_time REAL,
                strategy TEXT NOT NULL,
                corpus_sample TEXT NOT NULL,
                corpus_category TEXT DEFAULT '',
                total_iterations INTEGER DEFAULT 0,
                status TEXT DEFAULT 'running'
            );

            CREATE TABLE IF NOT EXISTS iterations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(id),
                iteration_num INTEGER NOT NULL,
                code_before TEXT NOT NULL,
                code_after TEXT NOT NULL,
                prompt TEXT NOT NULL,
                model_response TEXT,
                context_tokens INTEGER DEFAULT 0,
                timestamp REAL NOT NULL,
                duration_seconds REAL DEFAULT 0.0
            );

            CREATE TABLE IF NOT EXISTS vuln_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                iteration_id INTEGER NOT NULL REFERENCES iterations(id),
                vuln_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                line INTEGER DEFAULT 0,
                col INTEGER DEFAULT 0,
                description TEXT NOT NULL,
                confidence TEXT DEFAULT 'HIGH',
                is_introduced INTEGER DEFAULT 0,
                is_fixed INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS attention_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                iteration_id INTEGER NOT NULL REFERENCES iterations(id),
                snapshot_path TEXT,
                crypto_token_count INTEGER DEFAULT 0,
                mean_entropy REAL DEFAULT 0.0,
                min_entropy_layer INTEGER DEFAULT 0,
                min_entropy_head INTEGER DEFAULT 0,
                entropy_matrix_json TEXT
            );

            CREATE TABLE IF NOT EXISTS correlations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(id),
                vuln_iteration INTEGER NOT NULL,
                vuln_type TEXT NOT NULL,
                predictive_layer INTEGER NOT NULL,
                predictive_head INTEGER NOT NULL,
                signal_type TEXT NOT NULL,
                magnitude REAL DEFAULT 0.0,
                confidence REAL DEFAULT 0.0
            );

            CREATE INDEX IF NOT EXISTS idx_iterations_session
                ON iterations(session_id);
            CREATE INDEX IF NOT EXISTS idx_vuln_scores_iteration
                ON vuln_scores(iteration_id);
            CREATE INDEX IF NOT EXISTS idx_attention_iteration
                ON attention_snapshots(iteration_id);
            CREATE INDEX IF NOT EXISTS idx_correlations_session
                ON correlations(session_id);
        """)
        self._conn.commit()

    # ── Session Operations ────────────────────────────────────────

    def insert_session(
        self,
        session_id: str,
        start_time: float,
        strategy: str,
        corpus_sample: str,
        corpus_category: str = "",
    ) -> None:
        """Insert a new session record."""
        assert self._conn is not None
        self._conn.execute(
            "INSERT INTO sessions (id, start_time, strategy, corpus_sample, corpus_category) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, start_time, strategy, corpus_sample, corpus_category),
        )
        self._conn.commit()

    def update_session_status(
        self,
        session_id: str,
        status: str,
        end_time: Optional[float] = None,
        total_iterations: Optional[int] = None,
    ) -> None:
        """Update session status."""
        assert self._conn is not None
        updates = ["status = ?"]
        params: list[Any] = [status]

        if end_time is not None:
            updates.append("end_time = ?")
            params.append(end_time)
        if total_iterations is not None:
            updates.append("total_iterations = ?")
            params.append(total_iterations)

        params.append(session_id)
        self._conn.execute(
            f"UPDATE sessions SET {', '.join(updates)} WHERE id = ?",
            params,
        )
        self._conn.commit()

    def get_session(self, session_id: str) -> Optional[dict]:
        """Get a session by ID."""
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_sessions(self, status: Optional[str] = None) -> list[dict]:
        """List sessions, optionally filtered by status."""
        assert self._conn is not None
        if status:
            rows = self._conn.execute(
                "SELECT * FROM sessions WHERE status = ? ORDER BY start_time DESC",
                (status,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM sessions ORDER BY start_time DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Iteration Operations ──────────────────────────────────────

    def insert_iteration(
        self,
        session_id: str,
        iteration_num: int,
        code_before: str,
        code_after: str,
        prompt: str,
        timestamp: float,
        model_response: str = "",
        context_tokens: int = 0,
        duration_seconds: float = 0.0,
    ) -> int:
        """Insert an iteration record. Returns the iteration ID."""
        assert self._conn is not None
        cursor = self._conn.execute(
            "INSERT INTO iterations "
            "(session_id, iteration_num, code_before, code_after, "
            "prompt, model_response, context_tokens, timestamp, duration_seconds) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session_id, iteration_num, code_before, code_after,
                prompt, model_response, context_tokens, timestamp,
                duration_seconds,
            ),
        )
        self._conn.commit()
        return cursor.lastrowid

    def get_iterations(self, session_id: str) -> list[dict]:
        """Get all iterations for a session."""
        assert self._conn is not None
        rows = self._conn.execute(
            "SELECT * FROM iterations WHERE session_id = ? "
            "ORDER BY iteration_num",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Vulnerability Operations ──────────────────────────────────

    def insert_vuln_finding(
        self,
        iteration_id: int,
        vuln_type: str,
        severity: str,
        line: int,
        col: int,
        description: str,
        confidence: str = "HIGH",
        is_introduced: bool = False,
        is_fixed: bool = False,
    ) -> None:
        """Insert a vulnerability finding."""
        assert self._conn is not None
        self._conn.execute(
            "INSERT INTO vuln_scores "
            "(iteration_id, vuln_type, severity, line, col, description, "
            "confidence, is_introduced, is_fixed) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                iteration_id, vuln_type, severity, line, col,
                description, confidence,
                1 if is_introduced else 0,
                1 if is_fixed else 0,
            ),
        )
        self._conn.commit()

    # ── Attention Snapshot Operations ──────────────────────────────

    def insert_attention_snapshot(
        self,
        iteration_id: int,
        snapshot_path: str = "",
        crypto_token_count: int = 0,
        mean_entropy: float = 0.0,
        min_entropy_layer: int = 0,
        min_entropy_head: int = 0,
        entropy_matrix_json: str = "",
    ) -> int:
        """Insert attention snapshot metadata."""
        assert self._conn is not None
        cursor = self._conn.execute(
            "INSERT INTO attention_snapshots "
            "(iteration_id, snapshot_path, crypto_token_count, mean_entropy, "
            "min_entropy_layer, min_entropy_head, entropy_matrix_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                iteration_id, snapshot_path, crypto_token_count,
                mean_entropy, min_entropy_layer, min_entropy_head,
                entropy_matrix_json,
            ),
        )
        self._conn.commit()
        return cursor.lastrowid

    # ── Correlation Operations ────────────────────────────────────

    def insert_correlation(
        self,
        session_id: str,
        vuln_iteration: int,
        vuln_type: str,
        predictive_layer: int,
        predictive_head: int,
        signal_type: str,
        magnitude: float = 0.0,
        confidence: float = 0.0,
    ) -> None:
        """Insert a correlation finding."""
        assert self._conn is not None
        self._conn.execute(
            "INSERT INTO correlations "
            "(session_id, vuln_iteration, vuln_type, predictive_layer, "
            "predictive_head, signal_type, magnitude, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session_id, vuln_iteration, vuln_type,
                predictive_layer, predictive_head,
                signal_type, magnitude, confidence,
            ),
        )
        self._conn.commit()

    # ── Export ─────────────────────────────────────────────────────

    def export_session_csv(
        self, session_id: str, output_path: str
    ) -> None:
        """Export session data to CSV."""
        import csv

        iterations = self.get_iterations(session_id)
        if not iterations:
            return

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=iterations[0].keys())
            writer.writeheader()
            writer.writerows(iterations)

        logger.info("Exported %d iterations to %s", len(iterations), output_path)
