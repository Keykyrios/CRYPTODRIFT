"""
ECB_MODE detection rule.

Detects usage of ECB (Electronic Codebook) cipher mode, which
does not provide semantic security (identical plaintext blocks
produce identical ciphertext blocks).

Verified:
- PyCryptodome: AES.MODE_ECB is the integer constant 1
- cryptography: modes.ECB() is a class
"""

from __future__ import annotations

import ast

from . import BaseRule, Severity


class ECBModeRule(BaseRule):

    @property
    def rule_id(self) -> str:
        return "ECB_MODE"

    @property
    def description(self) -> str:
        return "Use of ECB cipher mode (no semantic security)"

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Check for AES.MODE_ECB or similar constants."""
        if node.attr == "MODE_ECB":
            self._add_finding(
                severity=Severity.CRITICAL,
                line=node.lineno,
                col=node.col_offset,
                description=(
                    "ECB mode provides no semantic security. "
                    "Identical plaintext blocks produce identical "
                    "ciphertext. Use AES-GCM or AES-CBC with HMAC."
                ),
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Check for modes.ECB() constructor or AES.new with mode=1."""
        call_name = self._get_call_name(node)

        # cryptography library: modes.ECB()
        if call_name == "modes.ECB" or call_name.endswith(".ECB"):
            self._add_finding(
                severity=Severity.CRITICAL,
                line=node.lineno,
                col=node.col_offset,
                description=(
                    "ECB mode constructor used. ECB provides no "
                    "semantic security. Use GCM, CBC+HMAC, or CTR."
                ),
            )

        # PyCryptodome: AES.new(key, AES.MODE_ECB) or AES.new(key, 1)
        if call_name in ("AES.new", "DES.new", "DES3.new"):
            if len(node.args) >= 2:
                mode_arg = node.args[1]
                # Check for literal 1 (MODE_ECB constant value)
                if (
                    isinstance(mode_arg, ast.Constant)
                    and mode_arg.value == 1
                ):
                    self._add_finding(
                        severity=Severity.CRITICAL,
                        line=node.lineno,
                        col=node.col_offset,
                        description=(
                            "Cipher constructed with mode=1 (ECB). "
                            "Use a secure mode like GCM (mode=6) or "
                            "CBC (mode=2) with HMAC."
                        ),
                    )

            # Also check mode= keyword
            mode_kw = self._get_keyword_arg(node, "mode")
            if mode_kw is not None:
                if isinstance(mode_kw, ast.Constant) and mode_kw.value == 1:
                    self._add_finding(
                        severity=Severity.CRITICAL,
                        line=node.lineno,
                        col=node.col_offset,
                        description="Cipher constructed with mode=ECB (1).",
                    )

        self.generic_visit(node)
