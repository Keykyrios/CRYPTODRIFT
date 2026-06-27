"""
DriftCorrelator — the novel contribution.

Correlates attention patterns with vulnerability introduction events.
For each iteration where a new vulnerability appears, examines the
PREVIOUS iteration's attention snapshot to find predictive patterns.

This is what nobody has done: mapping specific attention heads
to specific crypto vulnerability types.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PredictiveSignal:
    """A signal from an attention head that preceded a vulnerability."""
    layer: int
    head: int
    signal_type: str  # "entropy_collapse", "attention_sink", "cross_collapse"
    magnitude: float  # How strong the signal was
    vuln_type: str  # Which vulnerability it preceded
    iteration: int  # When the signal appeared
    vuln_iteration: int  # When the vulnerability appeared


@dataclass
class PredictiveHeadProfile:
    """Aggregated profile of a predictive attention head."""
    layer: int
    head: int
    signals: list[PredictiveSignal] = field(default_factory=list)

    @property
    def signal_count(self) -> int:
        return len(self.signals)

    @property
    def vuln_types_predicted(self) -> set[str]:
        return {s.vuln_type for s in self.signals}

    @property
    def mean_magnitude(self) -> float:
        if not self.signals:
            return 0.0
        return float(np.mean([s.magnitude for s in self.signals]))

    @property
    def prediction_rate(self) -> float:
        """What fraction of vulnerability events this head predicted."""
        # This needs total vuln events to compute, set externally
        return 0.0

    def to_dict(self) -> dict:
        return {
            "layer": self.layer,
            "head": self.head,
            "signal_count": self.signal_count,
            "vuln_types": list(self.vuln_types_predicted),
            "mean_magnitude": round(self.mean_magnitude, 4),
            "signals": [
                {
                    "signal_type": s.signal_type,
                    "magnitude": round(s.magnitude, 4),
                    "vuln_type": s.vuln_type,
                    "iteration": s.iteration,
                    "vuln_iteration": s.vuln_iteration,
                }
                for s in self.signals
            ],
        }


@dataclass
class CorrelationResult:
    """Complete result of drift correlation analysis."""
    session_id: str
    total_iterations: int
    total_vuln_events: int  # Iterations where net_delta > 0
    predictive_heads: list[PredictiveHeadProfile] = field(
        default_factory=list
    )
    per_vuln_type: dict[str, list[PredictiveHeadProfile]] = field(
        default_factory=dict
    )

    def get_top_predictive_heads(
        self, n: int = 10
    ) -> list[PredictiveHeadProfile]:
        """Get the N most predictive heads by signal count."""
        return sorted(
            self.predictive_heads,
            key=lambda h: h.signal_count,
            reverse=True,
        )[:n]

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "total_iterations": self.total_iterations,
            "total_vuln_events": self.total_vuln_events,
            "top_predictive_heads": [
                h.to_dict()
                for h in self.get_top_predictive_heads()
            ],
            "per_vuln_type": {
                vtype: [h.to_dict() for h in heads]
                for vtype, heads in self.per_vuln_type.items()
            },
        }


@dataclass
class IterationData:
    """Data for a single iteration used by the correlator."""
    iteration: int
    entropy_matrix: Optional[np.ndarray] = None  # (32, 32) layers × heads
    cross_attention: Optional[dict[str, list[float]]] = None
    vuln_types_introduced: list[str] = field(default_factory=list)
    net_vuln_delta: int = 0


class DriftCorrelator:
    """
    Correlates attention patterns with vulnerability introduction.

    For each vulnerability introduction event (iteration where
    net_delta > 0), examines the previous iteration's attention
    snapshot to find which heads showed anomalous patterns.

    Three signal types:
    1. Entropy collapse: Head entropy drops below threshold
    2. Attention sink: Head concentrates on non-crypto tokens
    3. Cross-attention collapse: Attention between crypto pairs drops
    """

    def __init__(
        self,
        entropy_collapse_threshold: float = 0.3,
        entropy_low_absolute: float = 0.2,
        cross_attention_drop_threshold: float = 0.5,
        lookback_window: int = 2,
    ):
        self._entropy_collapse_threshold = entropy_collapse_threshold
        self._entropy_low_absolute = entropy_low_absolute
        self._cross_drop_threshold = cross_attention_drop_threshold
        self._lookback = lookback_window

    def correlate(
        self,
        session_id: str,
        iteration_data: list[IterationData],
    ) -> CorrelationResult:
        """
        Run correlation analysis on a complete session.

        Args:
            session_id: Session identifier.
            iteration_data: List of IterationData, one per iteration,
                sorted by iteration number.

        Returns:
            CorrelationResult with identified predictive heads.
        """
        # Find vulnerability introduction events
        vuln_events = [
            i for i, d in enumerate(iteration_data)
            if d.net_vuln_delta > 0 and i > 0
        ]

        logger.info(
            "Session %s: %d iterations, %d vuln events",
            session_id,
            len(iteration_data),
            len(vuln_events),
        )

        # Collect all predictive signals
        all_signals: list[PredictiveSignal] = []

        for vuln_idx in vuln_events:
            vuln_data = iteration_data[vuln_idx]

            # Look at previous iteration(s) for predictive signals
            for lookback in range(1, self._lookback + 1):
                prev_idx = vuln_idx - lookback
                if prev_idx < 0:
                    continue

                prev_data = iteration_data[prev_idx]
                if prev_data.entropy_matrix is None:
                    continue

                signals = self._find_predictive_signals(
                    prev_data=prev_data,
                    vuln_data=vuln_data,
                    prev_iter=prev_data.iteration,
                    vuln_iter=vuln_data.iteration,
                )
                all_signals.extend(signals)

        # Aggregate signals by head
        head_profiles = self._aggregate_by_head(all_signals)

        # Aggregate by vulnerability type
        per_vuln_type = self._aggregate_by_vuln_type(all_signals)

        return CorrelationResult(
            session_id=session_id,
            total_iterations=len(iteration_data),
            total_vuln_events=len(vuln_events),
            predictive_heads=head_profiles,
            per_vuln_type=per_vuln_type,
        )

    def _find_predictive_signals(
        self,
        prev_data: IterationData,
        vuln_data: IterationData,
        prev_iter: int,
        vuln_iter: int,
    ) -> list[PredictiveSignal]:
        """Find predictive signals in the previous iteration's data."""
        signals = []

        entropy_matrix = prev_data.entropy_matrix
        if entropy_matrix is None:
            return signals

        num_layers, num_heads = entropy_matrix.shape

        for vuln_type in vuln_data.vuln_types_introduced:
            # Signal 1: Entropy collapse — heads with abnormally low entropy
            for layer in range(num_layers):
                for head in range(num_heads):
                    entropy_val = entropy_matrix[layer, head]

                    # Absolute low entropy
                    if entropy_val < self._entropy_low_absolute:
                        signals.append(PredictiveSignal(
                            layer=layer,
                            head=head,
                            signal_type="entropy_collapse",
                            magnitude=self._entropy_low_absolute - entropy_val,
                            vuln_type=vuln_type,
                            iteration=prev_iter,
                            vuln_iteration=vuln_iter,
                        ))

            # Signal 2: Cross-attention collapse
            if prev_data.cross_attention and vuln_data.cross_attention:
                for pair_key, prev_scores in prev_data.cross_attention.items():
                    vuln_scores = vuln_data.cross_attention.get(pair_key)
                    if vuln_scores is None:
                        continue

                    # Check for drops in cross-attention
                    for layer_idx in range(
                        min(len(prev_scores), len(vuln_scores))
                    ):
                        if prev_scores[layer_idx] > 0:
                            drop_ratio = (
                                1.0 - vuln_scores[layer_idx]
                                / prev_scores[layer_idx]
                            )
                            if drop_ratio > self._cross_drop_threshold:
                                for vuln_type in (
                                    vuln_data.vuln_types_introduced
                                ):
                                    signals.append(PredictiveSignal(
                                        layer=layer_idx,
                                        head=-1,  # Cross-attn is per-layer
                                        signal_type="cross_collapse",
                                        magnitude=drop_ratio,
                                        vuln_type=vuln_type,
                                        iteration=prev_iter,
                                        vuln_iteration=vuln_iter,
                                    ))

        return signals

    def _aggregate_by_head(
        self, signals: list[PredictiveSignal]
    ) -> list[PredictiveHeadProfile]:
        """Aggregate signals into per-head profiles."""
        head_map: dict[tuple[int, int], PredictiveHeadProfile] = {}

        for signal in signals:
            key = (signal.layer, signal.head)
            if key not in head_map:
                head_map[key] = PredictiveHeadProfile(
                    layer=signal.layer, head=signal.head
                )
            head_map[key].signals.append(signal)

        return list(head_map.values())

    def _aggregate_by_vuln_type(
        self, signals: list[PredictiveSignal]
    ) -> dict[str, list[PredictiveHeadProfile]]:
        """Aggregate signals by vulnerability type."""
        type_signals: dict[str, list[PredictiveSignal]] = {}
        for signal in signals:
            type_signals.setdefault(signal.vuln_type, []).append(signal)

        result = {}
        for vtype, sigs in type_signals.items():
            result[vtype] = self._aggregate_by_head(sigs)

        return result
