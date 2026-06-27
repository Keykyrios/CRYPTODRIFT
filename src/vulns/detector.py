"""
CryptoVulnDetector — orchestrates all 8 detection rules.

Parses Python code via ast.parse() and runs each rule's
NodeVisitor against the AST. Returns categorized findings.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from typing import Optional

from .rules import VulnFinding, Severity, BaseRule
from .rules.iv_reuse import IVReuseRule
from .rules.weak_kdf import WeakKDFRule
from .rules.timing_vuln import TimingVulnRule
from .rules.ecb_mode import ECBModeRule
from .rules.hardcoded_key import HardcodedKeyRule
from .rules.predictable_rng import PredictableRNGRule
from .rules.mac_order import MACOrderRule
from .rules.weak_hash import WeakHashRule

logger = logging.getLogger(__name__)


@dataclass
class VulnReport:
    """Complete vulnerability report for a code sample."""
    findings: list[VulnFinding] = field(default_factory=list)
    parse_error: Optional[str] = None

    @property
    def total_count(self) -> int:
        return len(self.findings)

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.CRITICAL)

    @property
    def high_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.HIGH)

    @property
    def weighted_score(self) -> int:
        """Weighted vulnerability score."""
        return sum(f.severity.weight for f in self.findings)

    def by_type(self) -> dict[str, list[VulnFinding]]:
        """Group findings by vulnerability type."""
        grouped: dict[str, list[VulnFinding]] = {}
        for f in self.findings:
            grouped.setdefault(f.vuln_type, []).append(f)
        return grouped

    def to_dict(self) -> dict:
        return {
            "total": self.total_count,
            "critical": self.critical_count,
            "high": self.high_count,
            "weighted_score": self.weighted_score,
            "parse_error": self.parse_error,
            "findings": [f.to_dict() for f in self.findings],
        }


@dataclass
class IterationDelta:
    """Vulnerability delta between two iterations."""
    introduced: list[VulnFinding] = field(default_factory=list)
    fixed: list[VulnFinding] = field(default_factory=list)
    net_delta: int = 0
    score_delta: int = 0

    def to_dict(self) -> dict:
        return {
            "introduced_count": len(self.introduced),
            "fixed_count": len(self.fixed),
            "net_delta": self.net_delta,
            "score_delta": self.score_delta,
            "introduced": [f.to_dict() for f in self.introduced],
            "fixed": [f.to_dict() for f in self.fixed],
        }


class CryptoVulnDetector:
    """
    Orchestrates all cryptographic vulnerability detection rules.

    Usage:
        detector = CryptoVulnDetector()
        report = detector.analyze(code)
        delta = detector.compare(code_before, code_after)
    """

    def __init__(self):
        self._rules: list[BaseRule] = [
            IVReuseRule(),
            WeakKDFRule(),
            TimingVulnRule(),
            ECBModeRule(),
            HardcodedKeyRule(),
            PredictableRNGRule(),
            MACOrderRule(),
            WeakHashRule(),
        ]

    def analyze(self, code: str) -> VulnReport:
        """
        Analyze code for cryptographic vulnerabilities.

        Args:
            code: Python source code string.

        Returns:
            VulnReport with all findings.
        """
        report = VulnReport()

        # Parse the code
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            report.parse_error = f"SyntaxError: {e}"
            logger.warning("Failed to parse code: %s", e)
            return report

        # Run each rule
        for rule in self._rules:
            rule.reset()
            try:
                rule.visit(tree)
                report.findings.extend(rule.findings)
            except Exception as e:
                logger.warning(
                    "Rule %s failed: %s", rule.rule_id, e
                )

        return report

    def compare(
        self, code_before: str, code_after: str
    ) -> IterationDelta:
        """
        Compare vulnerability reports between two code versions.

        Identifies which vulnerabilities were introduced and
        which were fixed.
        """
        report_before = self.analyze(code_before)
        report_after = self.analyze(code_after)

        # Build sets of (vuln_type, description) for comparison
        # We use (type, line_normalized_desc) as identity
        before_set = {
            (f.vuln_type, f.description) for f in report_before.findings
        }
        after_set = {
            (f.vuln_type, f.description) for f in report_after.findings
        }

        # Find introduced vulnerabilities (in after but not before)
        introduced_keys = after_set - before_set
        introduced = [
            f for f in report_after.findings
            if (f.vuln_type, f.description) in introduced_keys
        ]

        # Find fixed vulnerabilities (in before but not after)
        fixed_keys = before_set - after_set
        fixed = [
            f for f in report_before.findings
            if (f.vuln_type, f.description) in fixed_keys
        ]

        return IterationDelta(
            introduced=introduced,
            fixed=fixed,
            net_delta=len(report_after.findings) - len(report_before.findings),
            score_delta=report_after.weighted_score - report_before.weighted_score,
        )
