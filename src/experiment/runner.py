"""
Experiment runner — orchestrates the full pipeline.

Runs iterative refinement sessions:
1. Load baseline code from corpus
2. For each iteration (default 5):
   a. Send code + strategy prompt to model via MCP
   b. Capture attention weights via hooks
   c. Run vulnerability detection on refined code
   d. Create attention snapshot
   e. Run early warning system
3. After session: run correlation analysis
4. Store everything in SQLite

The runner works without MCP transport too (direct mode) for
simpler testing and debugging.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np

from ..mcp.session import SessionManager, SessionState, IterationRecord
from ..vulns.detector import CryptoVulnDetector, VulnReport, IterationDelta
from ..vulns.complexity import compute_complexity, compute_complexity_delta
from ..attention.snapshot import AttentionSnapshot, create_snapshot
from ..attention.crypto_tokens import find_crypto_token_positions
from ..correlation.correlator import DriftCorrelator, IterationData
from ..correlation.early_warning import EarlyWarningSystem, DriftWarning
from ..storage.database import CryptoDriftDB
from .strategies import get_strategy, get_iteration_prompt
from .corpus import CorpusSample

logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    """Configuration for an experiment run."""
    max_iterations: int = 5
    strategies: list[str] = None  # Default: all 4
    db_path: str = "data/cryptodrift.db"
    snapshot_dir: str = "data/snapshots"
    use_mcp: bool = False  # Direct mode by default
    use_gpu: bool = True
    max_new_tokens: int = 1024
    temperature: float = 0.7

    def __post_init__(self):
        if self.strategies is None:
            self.strategies = ["EF", "FF", "SF", "AI"]


# Type for refinement function (allows swapping model/mock)
RefinementFn = Callable[
    [str, str, list[dict], int, float],
    tuple[str, int],  # (generated_text, total_tokens)
]


class ExperimentRunner:
    """
    Orchestrates iterative code refinement experiments.

    Can run in two modes:
    1. Direct mode: Calls model inference directly
    2. MCP mode: Uses MCP client/server transport
    """

    def __init__(self, config: RunConfig):
        self.config = config
        self._session_mgr = SessionManager()
        self._detector = CryptoVulnDetector()
        self._correlator = DriftCorrelator()
        self._early_warning = EarlyWarningSystem()
        self._db = CryptoDriftDB(config.db_path)
        self._refinement_fn: Optional[RefinementFn] = None
        self._tokenizer = None

        # Callbacks for UI integration
        self._on_iteration: Optional[Callable] = None
        self._on_warning: Optional[Callable] = None
        self._on_session_complete: Optional[Callable] = None
        self._running = False

    def set_refinement_fn(self, fn: RefinementFn) -> None:
        """Set the function used to generate code refinements."""
        self._refinement_fn = fn

    def set_tokenizer(self, tokenizer) -> None:
        """Set the tokenizer for crypto token position finding."""
        self._tokenizer = tokenizer

    def set_callbacks(
        self,
        on_iteration: Optional[Callable] = None,
        on_warning: Optional[Callable] = None,
        on_session_complete: Optional[Callable] = None,
    ) -> None:
        """Set UI callback functions."""
        self._on_iteration = on_iteration
        self._on_warning = on_warning
        self._on_session_complete = on_session_complete

    def setup(self) -> None:
        """Initialize database connection."""
        self._db.connect()

    def teardown(self) -> None:
        """Clean up resources."""
        self._db.close()

    def stop(self) -> None:
        """Signal the runner to stop after current iteration."""
        self._running = False

    def run_session(
        self,
        sample: CorpusSample,
        strategy_code: str,
        probe=None,
    ) -> SessionState:
        """
        Run a single refinement session.

        Args:
            sample: Baseline code sample.
            strategy_code: Strategy code (EF/FF/SF/AI).
            probe: Optional AttentionProbe for attention capture.

        Returns:
            Completed SessionState.
        """
        self._running = True
        strategy = get_strategy(strategy_code)

        # Create session
        session = self._session_mgr.create_session(
            strategy=strategy_code,
            corpus_sample=sample.name,
            corpus_category=sample.category,
        )

        # Store session in DB
        self._db.insert_session(
            session_id=session.session_id,
            start_time=session.start_time,
            strategy=strategy_code,
            corpus_sample=sample.name,
            corpus_category=sample.category,
        )

        logger.info(
            "Session %s: %s × %s (%d iterations)",
            session.session_id[:8],
            sample.name,
            strategy_code,
            self.config.max_iterations,
        )

        current_code = sample.code
        conversation_history: list[dict] = []
        iteration_data_list: list[IterationData] = []

        # Baseline vulnerability analysis
        baseline_report = self._detector.analyze(current_code)
        logger.info(
            "Baseline: %d vulns (expected %d)",
            baseline_report.total_count,
            sample.expected_vulns,
        )

        for iter_num in range(self.config.max_iterations):
            if not self._running:
                logger.info("Run stopped at iteration %d", iter_num)
                session.abort()
                break

            iter_start = time.time()

            # Get refinement prompt
            prompt = get_iteration_prompt(strategy_code, iter_num)

            # Generate refinement
            if self._refinement_fn is None:
                logger.error("No refinement function set")
                session.fail("No refinement function")
                break

            try:
                generated_text, total_tokens = self._refinement_fn(
                    current_code,
                    prompt,
                    conversation_history,
                    self.config.max_new_tokens,
                    self.config.temperature,
                )
            except Exception as e:
                logger.error("Refinement failed at iteration %d: %s", iter_num, e)
                session.fail(str(e))
                break

            # Extract code from response
            from ..attention.model_loader import extract_code_from_response
            refined_code = extract_code_from_response(generated_text)

            if not refined_code.strip():
                logger.warning("Empty refinement at iteration %d, using previous", iter_num)
                refined_code = current_code

            iter_duration = time.time() - iter_start

            # Vulnerability analysis
            vuln_report = self._detector.analyze(refined_code)
            vuln_delta = self._detector.compare(current_code, refined_code)

            # Complexity analysis
            complexity = compute_complexity_delta(current_code, refined_code)

            # Attention snapshot (if probe is available)
            snapshot: Optional[AttentionSnapshot] = None
            crypto_positions: dict[str, list[int]] = {}

            if probe is not None and probe.enabled:
                layer_snapshots = probe.get_snapshots()
                if layer_snapshots and self._tokenizer is not None:
                    crypto_positions = find_crypto_token_positions(
                        self._tokenizer, refined_code
                    )
                    snapshot = create_snapshot(
                        layer_snapshots,
                        crypto_positions,
                        iter_num,
                        save_dir=self.config.snapshot_dir,
                    )
                probe.clear_snapshots()

            # Build iteration record
            record = IterationRecord(
                iteration_num=iter_num,
                code_before=current_code,
                code_after=refined_code,
                prompt=prompt,
                strategy=strategy_code,
                context_tokens=total_tokens,
                vuln_score=vuln_report.to_dict(),
                model_response_raw=generated_text,
                duration_seconds=iter_duration,
            )
            session.add_iteration(record)

            # Store in DB
            iter_id = self._db.insert_iteration(
                session_id=session.session_id,
                iteration_num=iter_num,
                code_before=current_code,
                code_after=refined_code,
                prompt=prompt,
                timestamp=record.timestamp,
                model_response=generated_text,
                context_tokens=total_tokens,
                duration_seconds=iter_duration,
            )

            # Store vulnerability findings
            for finding in vuln_report.findings:
                self._db.insert_vuln_finding(
                    iteration_id=iter_id,
                    vuln_type=finding.vuln_type,
                    severity=finding.severity.value,
                    line=finding.line,
                    col=finding.col,
                    description=finding.description,
                    confidence=finding.confidence,
                )

            # Store attention snapshot metadata
            if snapshot is not None:
                self._db.insert_attention_snapshot(
                    iteration_id=iter_id,
                    snapshot_path=snapshot.snapshot_path or "",
                    crypto_token_count=snapshot.crypto_token_count,
                    mean_entropy=snapshot.mean_entropy,
                    entropy_matrix_json=json.dumps(
                        snapshot.get_entropy_matrix().tolist()
                    ),
                )

            # Build iteration data for correlation
            entropy_matrix = None
            if snapshot is not None:
                entropy_matrix = snapshot.get_entropy_matrix()

            iter_data = IterationData(
                iteration=iter_num,
                entropy_matrix=entropy_matrix,
                cross_attention=(
                    snapshot.cross_attention_scores if snapshot else None
                ),
                vuln_types_introduced=[
                    f.vuln_type for f in vuln_delta.introduced
                ],
                net_vuln_delta=vuln_delta.net_delta,
            )
            iteration_data_list.append(iter_data)

            # Early warning check
            warnings = []
            if entropy_matrix is not None:
                warnings = self._early_warning.check(entropy_matrix, iter_num)
                for w in warnings:
                    if self._on_warning:
                        self._on_warning(w)

            # UI callback
            if self._on_iteration:
                self._on_iteration({
                    "session_id": session.session_id,
                    "iteration": iter_num,
                    "code": refined_code,
                    "vuln_report": vuln_report.to_dict(),
                    "vuln_delta": vuln_delta.to_dict(),
                    "complexity": complexity,
                    "snapshot": snapshot.to_dict() if snapshot else None,
                    "warnings": [w.to_dict() for w in warnings],
                    "context_tokens": total_tokens,
                    "duration": iter_duration,
                })

            logger.info(
                "Iteration %d: vulns=%d (Δ%+d), tokens=%d, %.1fs",
                iter_num,
                vuln_report.total_count,
                vuln_delta.net_delta,
                total_tokens,
                iter_duration,
            )

            # Update context for next iteration
            conversation_history.append({"role": "user", "content": prompt})
            conversation_history.append({
                "role": "assistant",
                "content": generated_text,
            })
            current_code = refined_code

        # Complete session
        if session.status == "running":
            session.complete()

        self._db.update_session_status(
            session_id=session.session_id,
            status=session.status,
            end_time=session.end_time,
            total_iterations=len(session.iterations),
        )

        # Run correlation analysis
        if len(iteration_data_list) >= 2:
            correlation = self._correlator.correlate(
                session.session_id, iteration_data_list
            )
            logger.info(
                "Correlation: %d predictive heads, %d vuln events",
                len(correlation.predictive_heads),
                correlation.total_vuln_events,
            )

            # Store correlations
            for head in correlation.predictive_heads:
                for signal in head.signals:
                    self._db.insert_correlation(
                        session_id=session.session_id,
                        vuln_iteration=signal.vuln_iteration,
                        vuln_type=signal.vuln_type,
                        predictive_layer=signal.layer,
                        predictive_head=signal.head,
                        signal_type=signal.signal_type,
                        magnitude=signal.magnitude,
                        confidence=0.0,
                    )

        if self._on_session_complete:
            self._on_session_complete(session)

        logger.info(
            "Session %s completed: %d iterations, %.1fs",
            session.session_id[:8],
            len(session.iterations),
            session.duration,
        )

        return session

    def run_full_experiment(
        self,
        samples: list[CorpusSample],
        probe=None,
    ) -> list[SessionState]:
        """
        Run the full experiment: all samples × all strategies.

        Returns list of completed SessionStates.
        """
        results = []
        total_sessions = len(samples) * len(self.config.strategies)
        session_num = 0

        for sample in samples:
            for strategy in self.config.strategies:
                session_num += 1
                logger.info(
                    "=== Session %d/%d: %s × %s ===",
                    session_num, total_sessions,
                    sample.name, strategy,
                )

                try:
                    session = self.run_session(sample, strategy, probe)
                    results.append(session)
                except Exception as e:
                    logger.error(
                        "Session failed: %s × %s: %s",
                        sample.name, strategy, e,
                    )

                if not self._running:
                    break
            if not self._running:
                break

        return results
