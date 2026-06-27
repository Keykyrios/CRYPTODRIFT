"""
TIMING_UNSAFE_COMPARE detection rule.

Detects timing-unsafe comparisons of cryptographic values
(HMAC tags, hashes, tokens) using == instead of
hmac.compare_digest() or secrets.compare_digest().

Verified: hmac.compare_digest exists since Python 3.3.
          secrets.compare_digest exists since Python 3.6.
"""

from __future__ import annotations

import ast

from . import BaseRule, Severity


# Variable names that suggest cryptographic values
CRYPTO_VALUE_NAMES = {
    "mac", "tag", "hmac", "digest", "hash", "signature", "sig",
    "token", "auth_tag", "expected_mac", "computed_mac",
    "expected_tag", "computed_tag", "expected_hash", "computed_hash",
    "expected_hmac", "computed_hmac", "verify_tag",
}


class TimingVulnRule(BaseRule):

    @property
    def rule_id(self) -> str:
        return "TIMING_UNSAFE_COMPARE"

    @property
    def description(self) -> str:
        return "Timing-unsafe comparison of cryptographic values"

    def visit_Compare(self, node: ast.Compare) -> None:
        """Check for == comparisons involving crypto values."""
        # Only concerned with == and != operators
        for op in node.ops:
            if not isinstance(op, (ast.Eq, ast.NotEq)):
                self.generic_visit(node)
                return

        # Check if either side of the comparison looks like a crypto value
        all_operands = [node.left] + node.comparators

        crypto_operand_found = False
        for operand in all_operands:
            if self._is_crypto_value(operand):
                crypto_operand_found = True
                break

        if crypto_operand_found:
            self._add_finding(
                severity=Severity.HIGH,
                line=node.lineno,
                col=node.col_offset,
                description=(
                    "Timing-unsafe comparison of cryptographic value. "
                    "Use hmac.compare_digest() instead of == to prevent "
                    "timing side-channel attacks."
                ),
            )

        self.generic_visit(node)

    def _is_crypto_value(self, node: ast.expr) -> bool:
        """Check if a node likely represents a cryptographic value."""
        if isinstance(node, ast.Name):
            return node.id.lower() in CRYPTO_VALUE_NAMES

        if isinstance(node, ast.Attribute):
            return node.attr.lower() in CRYPTO_VALUE_NAMES

        # Check for method calls like .digest(), .hexdigest()
        if isinstance(node, ast.Call):
            call_name = self._get_call_name(node)
            return any(
                call_name.endswith(suffix)
                for suffix in (".digest", ".hexdigest", ".finalize", ".tag")
            )

        return False
