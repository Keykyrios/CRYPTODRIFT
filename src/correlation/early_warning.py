"""
Early warning system — real-time prediction of crypto vulnerabilities
based on attention patterns.

Takes current iteration's attention snapshot, compares against
known predictive signatures, emits warnings.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DriftWarning:
    """A warning emitted by the early warning system."""
    timestamp: float = field(default_factory=time.time)
    layer: int = 0
    head: int = 0
    predicted_vuln_type: str = ""
    confidence: float = 0.0  # 0.0 to 1.0
    signal_type: str = ""  # entropy_collapse, cross_collapse
    message: str = ""
    entropy_value: float = 0.0

    @property
    def severity(self) -> str:
        if self.confidence >= 0.8:
            return "CRITICAL"
        elif self.confidence >= 0.5:
            return "HIGH"
        elif self.confidence >= 0.3:
            return "MEDIUM"
        return "LOW"

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "layer": self.layer,
            "head": self.head,
            "predicted_vuln_type": self.predicted_vuln_type,
            "confidence": round(self.confidence, 3),
            "severity": self.severity,
            "signal_type": self.signal_type,
            "message": self.message,
            "entropy_value": round(self.entropy_value, 4),
        }


@dataclass
class PredictiveSignature:
    """A known predictive signature pattern."""
    name: str
    layer: int
    head: int
    signal_type: str
    vuln_type: str
    threshold: float  # Entropy below this triggers warning
    confidence: float  # Historical prediction accuracy


class EarlyWarningSystem:
    """
    Real-time vulnerability prediction system.

    Compares live attention patterns against known predictive
    signatures to emit warnings before vulnerabilities appear.
    """

    def __init__(self):
        self._signatures: list[PredictiveSignature] = []
        self._warnings: list[DriftWarning] = []
        self._entropy_history: list[np.ndarray] = []
        self._listeners: list = []

    @property
    def warnings(self) -> list[DriftWarning]:
        return list(self._warnings)

    def add_signature(self, signature: PredictiveSignature) -> None:
        """Add a known predictive signature."""
        self._signatures.append(signature)

    def load_signatures_from_correlation(
        self, correlation_result
    ) -> int:
        """
        Build signatures from a completed correlation analysis.

        Extracts the most predictive heads and creates signature
        patterns from them.
        """
        count = 0
        for head_profile in correlation_result.get_top_predictive_heads(20):
            if head_profile.signal_count < 3:
                continue

            # Find the dominant signal type and vuln type
            signal_types = {}
            vuln_types = {}
            magnitudes = []

            for signal in head_profile.signals:
                signal_types[signal.signal_type] = (
                    signal_types.get(signal.signal_type, 0) + 1
                )
                vuln_types[signal.vuln_type] = (
                    vuln_types.get(signal.vuln_type, 0) + 1
                )
                magnitudes.append(signal.magnitude)

            dominant_signal = max(signal_types, key=signal_types.get)
            dominant_vuln = max(vuln_types, key=vuln_types.get)

            sig = PredictiveSignature(
                name=f"L{head_profile.layer}_H{head_profile.head}_{dominant_vuln}",
                layer=head_profile.layer,
                head=head_profile.head,
                signal_type=dominant_signal,
                vuln_type=dominant_vuln,
                threshold=float(np.mean(magnitudes)),
                confidence=min(0.95, head_profile.signal_count / 20.0),
            )
            self._signatures.append(sig)
            count += 1

        logger.info("Loaded %d predictive signatures", count)
        return count

    def check(
        self,
        entropy_matrix: np.ndarray,
        iteration: int,
    ) -> list[DriftWarning]:
        """
        Check current attention state against predictive signatures.

        Args:
            entropy_matrix: Shape (32, 32) — normalized entropy
                per layer and head.
            iteration: Current iteration number.

        Returns:
            List of new warnings emitted.
        """
        self._entropy_history.append(entropy_matrix.copy())
        new_warnings = []

        # Check signature-based warnings
        for sig in self._signatures:
            if sig.layer >= entropy_matrix.shape[0]:
                continue
            if sig.head >= entropy_matrix.shape[1]:
                continue

            entropy_val = entropy_matrix[sig.layer, sig.head]

            if sig.signal_type == "entropy_collapse":
                if entropy_val < sig.threshold:
                    warning = DriftWarning(
                        layer=sig.layer,
                        head=sig.head,
                        predicted_vuln_type=sig.vuln_type,
                        confidence=sig.confidence,
                        signal_type="entropy_collapse",
                        entropy_value=entropy_val,
                        message=(
                            f"Layer {sig.layer} Head {sig.head}: "
                            f"entropy collapse ({entropy_val:.3f} < "
                            f"{sig.threshold:.3f}). "
                            f"{sig.vuln_type} likely in next iteration."
                        ),
                    )
                    new_warnings.append(warning)

        # Check for rapid entropy drops (signature-independent)
        if len(self._entropy_history) >= 2:
            prev = self._entropy_history[-2]
            curr = self._entropy_history[-1]
            drops = prev - curr

            for layer in range(min(drops.shape[0], 32)):
                for head in range(min(drops.shape[1], 32)):
                    if drops[layer, head] > 0.4:
                        warning = DriftWarning(
                            layer=layer,
                            head=head,
                            predicted_vuln_type="UNKNOWN",
                            confidence=0.3,
                            signal_type="rapid_entropy_drop",
                            entropy_value=float(curr[layer, head]),
                            message=(
                                f"Layer {layer} Head {head}: "
                                f"rapid entropy drop "
                                f"({drops[layer, head]:.3f}). "
                                "Possible degradation signal."
                            ),
                        )
                        new_warnings.append(warning)

        self._warnings.extend(new_warnings)

        # Notify listeners
        for listener in self._listeners:
            for warning in new_warnings:
                listener(warning)

        return new_warnings

    def add_listener(self, callback) -> None:
        """Add a callback that receives new warnings."""
        self._listeners.append(callback)

    def clear_history(self) -> None:
        """Clear entropy history and warnings."""
        self._entropy_history.clear()
        self._warnings.clear()
