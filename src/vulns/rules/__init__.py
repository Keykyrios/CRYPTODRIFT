"""
Cryptographic vulnerability detection rules.

Each rule is an ast.NodeVisitor subclass that detects a specific
class of cryptographic vulnerability in Python source code.

Verified detection patterns against:
- OWASP Cryptographic Failures guidelines
- PyCryptodome API (Crypto.Cipher.AES, etc.)
- Python cryptography library API (cryptography.hazmat.*)
- Python stdlib (hashlib, hmac, secrets, os.urandom)
"""

from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Severity(Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

    @property
    def weight(self) -> int:
        return {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}[self.value]


@dataclass
class VulnFinding:
    """A single vulnerability finding."""
    vuln_type: str
    severity: Severity
    line: int
    col: int
    description: str
    confidence: str = "HIGH"  # HIGH, MEDIUM, LOW

    def to_dict(self) -> dict:
        return {
            "vuln_type": self.vuln_type,
            "severity": self.severity.value,
            "line": self.line,
            "col": self.col,
            "description": self.description,
            "confidence": self.confidence,
        }


class BaseRule(ast.NodeVisitor, ABC):
    """Base class for all vulnerability detection rules."""

    def __init__(self):
        self.findings: list[VulnFinding] = []

    @property
    @abstractmethod
    def rule_id(self) -> str:
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        ...

    def reset(self) -> None:
        self.findings.clear()

    def _add_finding(
        self,
        severity: Severity,
        line: int,
        col: int,
        description: str,
        confidence: str = "HIGH",
    ) -> None:
        self.findings.append(VulnFinding(
            vuln_type=self.rule_id,
            severity=severity,
            line=line,
            col=col,
            description=description,
            confidence=confidence,
        ))

    def _get_call_name(self, node: ast.Call) -> str:
        """Extract the full dotted name of a function call."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            parts = []
            current = node.func
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            return ".".join(reversed(parts))
        return ""

    def _is_constant_value(self, node: ast.expr) -> bool:
        """Check if a node is a constant (literal) value."""
        # ast.Constant covers str, bytes, int, float, bool, None
        # since Python 3.8+
        return isinstance(node, ast.Constant)

    def _get_keyword_arg(
        self, node: ast.Call, keyword: str
    ) -> Optional[ast.expr]:
        """Get a keyword argument value from a call node."""
        for kw in node.keywords:
            if kw.arg == keyword:
                return kw.value
        return None
