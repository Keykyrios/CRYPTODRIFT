"""
Session manager for multi-turn MCP refinement sessions.

Tracks conversation state, context window accumulation,
and iteration history across a complete refinement session.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class IterationRecord:
    """Record of a single refinement iteration."""
    iteration_num: int
    code_before: str
    code_after: str
    prompt: str
    strategy: str
    timestamp: float = field(default_factory=time.time)
    context_tokens: int = 0
    attention_snapshot_id: Optional[int] = None
    vuln_score: Optional[dict] = None
    model_response_raw: Optional[str] = None
    duration_seconds: float = 0.0


@dataclass
class SessionState:
    """
    Complete state of a multi-turn refinement session.

    Tracks:
    - Session metadata (id, strategy, corpus sample)
    - All iteration records
    - Accumulated context window size
    - Timing information
    """
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    strategy: str = ""
    corpus_sample: str = ""
    corpus_category: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    status: str = "running"  # running, completed, failed, aborted
    iterations: list[IterationRecord] = field(default_factory=list)
    total_context_tokens: int = 0

    @property
    def current_iteration(self) -> int:
        return len(self.iterations)

    @property
    def current_code(self) -> str:
        """Get the most recent version of the code."""
        if not self.iterations:
            return ""
        return self.iterations[-1].code_after

    @property
    def initial_code(self) -> str:
        """Get the original code before any refinement."""
        if not self.iterations:
            return ""
        return self.iterations[0].code_before

    @property
    def duration(self) -> float:
        """Total session duration in seconds."""
        end = self.end_time or time.time()
        return end - self.start_time

    def add_iteration(self, record: IterationRecord) -> None:
        """Add an iteration record to the session."""
        self.iterations.append(record)
        self.total_context_tokens = record.context_tokens

    def complete(self) -> None:
        """Mark the session as completed."""
        self.end_time = time.time()
        self.status = "completed"

    def fail(self, reason: str = "") -> None:
        """Mark the session as failed."""
        self.end_time = time.time()
        self.status = "failed"

    def abort(self) -> None:
        """Mark the session as aborted."""
        self.end_time = time.time()
        self.status = "aborted"

    def get_vuln_timeline(self) -> list[dict]:
        """Get vulnerability scores across all iterations."""
        timeline = []
        for record in self.iterations:
            if record.vuln_score is not None:
                timeline.append({
                    "iteration": record.iteration_num,
                    "vuln_score": record.vuln_score,
                })
        return timeline

    def to_dict(self) -> dict:
        """Serialize session state to a dictionary."""
        return {
            "session_id": self.session_id,
            "strategy": self.strategy,
            "corpus_sample": self.corpus_sample,
            "corpus_category": self.corpus_category,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "status": self.status,
            "total_iterations": len(self.iterations),
            "total_context_tokens": self.total_context_tokens,
            "duration_seconds": self.duration,
            "iterations": [
                {
                    "iteration_num": r.iteration_num,
                    "code_before_len": len(r.code_before),
                    "code_after_len": len(r.code_after),
                    "prompt": r.prompt,
                    "strategy": r.strategy,
                    "timestamp": r.timestamp,
                    "context_tokens": r.context_tokens,
                    "vuln_score": r.vuln_score,
                    "duration_seconds": r.duration_seconds,
                }
                for r in self.iterations
            ],
        }


class SessionManager:
    """
    Manages multiple refinement sessions.

    Provides session creation, lookup, and lifecycle management.
    """

    def __init__(self):
        self._sessions: dict[str, SessionState] = {}
        self._active_session_id: Optional[str] = None

    @property
    def active_session(self) -> Optional[SessionState]:
        if self._active_session_id is None:
            return None
        return self._sessions.get(self._active_session_id)

    def create_session(
        self,
        strategy: str,
        corpus_sample: str,
        corpus_category: str = "",
    ) -> SessionState:
        """Create a new session and set it as active."""
        session = SessionState(
            strategy=strategy,
            corpus_sample=corpus_sample,
            corpus_category=corpus_category,
        )
        self._sessions[session.session_id] = session
        self._active_session_id = session.session_id
        return session

    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    def list_sessions(
        self, status: Optional[str] = None
    ) -> list[SessionState]:
        """List all sessions, optionally filtered by status."""
        sessions = list(self._sessions.values())
        if status is not None:
            sessions = [s for s in sessions if s.status == status]
        return sessions

    def complete_active_session(self) -> Optional[SessionState]:
        """Mark the active session as completed."""
        session = self.active_session
        if session is not None:
            session.complete()
            self._active_session_id = None
        return session
