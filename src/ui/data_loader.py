"""
Database-backed data loader for the CryptoDrift UI.

Reads completed experiment data from SQLite and provides
it to the UI panels for visualization.
"""

from __future__ import annotations

import json
import sqlite3
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent.parent / "data" / "cryptodrift.db"


@dataclass
class SessionView:
    """UI-friendly view of a completed session."""
    session_id: str
    corpus_sample: str
    strategy: str
    total_iterations: int
    iterations: list[IterationView] = field(default_factory=list)


@dataclass
class IterationView:
    """UI-friendly view of a single iteration."""
    iteration_num: int
    code_before: str
    code_after: str
    mean_entropy: float
    entropy_matrix: Optional[np.ndarray]  # 32x32 or None
    vuln_count: int
    vuln_types: dict[str, int]  # {type: count}
    vulns: list[dict]  # raw vuln records


@dataclass
class DriftAlertView:
    """UI-friendly drift alert."""
    timestamp: float
    severity: str
    layer: int
    head: int
    predicted_vuln_type: str
    message: str


class ExperimentDataLoader:
    """Loads experiment results from SQLite for the UI."""

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = db_path or str(DB_PATH)

    def get_sessions(self) -> list[SessionView]:
        """Get all completed sessions with their iteration data."""
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row

        sessions = []
        session_rows = conn.execute("""
            SELECT id, corpus_sample, strategy, total_iterations
            FROM sessions WHERE status = 'completed'
            ORDER BY start_time
        """).fetchall()

        for sr in session_rows:
            session = SessionView(
                session_id=sr["id"],
                corpus_sample=sr["corpus_sample"],
                strategy=sr["strategy"],
                total_iterations=sr["total_iterations"],
            )

            # Load iterations
            iter_rows = conn.execute("""
                SELECT i.iteration_num, i.code_before, i.code_after,
                       a.mean_entropy, a.entropy_matrix_json
                FROM iterations i
                LEFT JOIN attention_snapshots a ON a.iteration_id = i.id
                WHERE i.session_id = ?
                ORDER BY i.iteration_num
            """, (sr["id"],)).fetchall()

            for ir in iter_rows:
                # Load vulns for this iteration
                vuln_rows = conn.execute("""
                    SELECT vuln_type, severity, description
                    FROM vuln_scores
                    WHERE iteration_id = (
                        SELECT id FROM iterations 
                        WHERE session_id = ? AND iteration_num = ?
                    )
                """, (sr["id"], ir["iteration_num"])).fetchall()

                vuln_types: dict[str, int] = {}
                vulns = []
                for vr in vuln_rows:
                    vuln_types[vr["vuln_type"]] = vuln_types.get(vr["vuln_type"], 0) + 1
                    vulns.append(dict(vr))

                # Parse entropy matrix
                entropy_matrix = None
                if ir["entropy_matrix_json"]:
                    try:
                        matrix_data = json.loads(ir["entropy_matrix_json"])
                        entropy_matrix = np.array(matrix_data, dtype=np.float32)
                    except (json.JSONDecodeError, ValueError):
                        pass

                # Clean code for display
                code_before = ir["code_before"] or ""
                code_after = ir["code_after"] or ""

                # If code_after looks like prose, fall back to code_before
                if code_after and not any(
                    kw in code_after[:200]
                    for kw in ("import ", "from ", "def ", "class ")
                ):
                    code_after = f"# [Model returned prose, not code]\n# Showing previous iteration's code\n\n{code_before}"

                session.iterations.append(IterationView(
                    iteration_num=ir["iteration_num"],
                    code_before=code_before,
                    code_after=code_after,
                    mean_entropy=ir["mean_entropy"] or 0.0,
                    entropy_matrix=entropy_matrix,
                    vuln_count=len(vulns),
                    vuln_types=vuln_types,
                    vulns=vulns,
                ))

            sessions.append(session)

        conn.close()
        return sessions

    def get_session_labels(self) -> list[str]:
        """Get display labels for all completed sessions."""
        conn = sqlite3.connect(self._db_path)
        rows = conn.execute("""
            SELECT corpus_sample, strategy FROM sessions 
            WHERE status = 'completed' ORDER BY start_time
        """).fetchall()
        conn.close()
        return [f"{r[0]} x {r[1]}" for r in rows]
