"""
IV_REUSE detection rule.

Detects initialization vectors that are reused, static, or predictable.

Patterns detected:
- AES/cipher construction without os.urandom() or secrets.token_bytes()
- Hardcoded IV values (bytes literals assigned to iv/nonce variables)
- Same variable used as IV in multiple encrypt calls
- time-based IV seeding
"""

from __future__ import annotations

import ast
from typing import Optional

from . import BaseRule, Severity


class IVReuseRule(BaseRule):

    @property
    def rule_id(self) -> str:
        return "IV_REUSE"

    @property
    def description(self) -> str:
        return "Initialization vector reuse or predictable IV generation"

    def __init__(self):
        super().__init__()
        # Track variable names assigned as IVs
        self._iv_sources: dict[str, ast.AST] = {}
        # Track IV variable names used in encrypt calls
        self._iv_usages: list[tuple[str, int]] = []

    def reset(self) -> None:
        super().reset()
        self._iv_sources.clear()
        self._iv_usages.clear()

    def visit_Assign(self, node: ast.Assign) -> None:
        """Check for hardcoded or static IV assignments."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                name_lower = target.id.lower()
                if name_lower in ("iv", "nonce", "initialization_vector", "ctr"):
                    self._iv_sources[target.id] = node.value

                    # Check if the value is a constant (hardcoded)
                    if self._is_constant_value(node.value):
                        self._add_finding(
                            severity=Severity.CRITICAL,
                            line=node.lineno,
                            col=node.col_offset,
                            description=(
                                f"Hardcoded IV/nonce assigned to '{target.id}'. "
                                "IVs must be generated fresh using "
                                "os.urandom() or secrets.token_bytes()."
                            ),
                        )

                    # Check for time-based IV
                    if isinstance(node.value, ast.Call):
                        call_name = self._get_call_name(node.value)
                        if call_name in ("time.time", "time.time_ns",
                                         "int", "str"):
                            # int(time.time()) or similar
                            self._add_finding(
                                severity=Severity.HIGH,
                                line=node.lineno,
                                col=node.col_offset,
                                description=(
                                    f"Time-based IV/nonce in '{target.id}'. "
                                    "Time values are predictable and "
                                    "must not be used as IVs."
                                ),
                                confidence="MEDIUM",
                            )

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Check cipher construction calls for IV handling."""
        call_name = self._get_call_name(node)

        # PyCryptodome: AES.new(key, mode, iv=...)
        # cryptography: Cipher(algorithms.AES(key), modes.CBC(iv))
        if call_name in ("AES.new", "DES.new", "DES3.new",
                         "Blowfish.new", "ChaCha20.new"):
            self._check_pycryptodome_cipher(node)
        elif call_name in ("modes.CBC", "modes.CTR", "modes.CFB",
                           "modes.OFB", "modes.GCM"):
            self._check_cryptography_mode(node)

        self.generic_visit(node)

    def _check_pycryptodome_cipher(self, node: ast.Call) -> None:
        """Check PyCryptodome cipher construction for IV issues."""
        # Look for iv= or nonce= keyword argument
        iv_arg = self._get_keyword_arg(node, "iv")
        nonce_arg = self._get_keyword_arg(node, "nonce")

        iv_node = iv_arg or nonce_arg

        if iv_node is not None:
            if self._is_constant_value(iv_node):
                self._add_finding(
                    severity=Severity.CRITICAL,
                    line=node.lineno,
                    col=node.col_offset,
                    description=(
                        "Hardcoded IV/nonce passed directly to cipher. "
                        "Use os.urandom(16) or secrets.token_bytes(16)."
                    ),
                )
            elif isinstance(iv_node, ast.Name):
                # Track the usage for reuse detection
                self._iv_usages.append((iv_node.id, node.lineno))

        # Also check positional args (3rd arg is often IV in PyCryptodome)
        if len(node.args) >= 3:
            potential_iv = node.args[2]
            if self._is_constant_value(potential_iv):
                self._add_finding(
                    severity=Severity.CRITICAL,
                    line=node.lineno,
                    col=node.col_offset,
                    description=(
                        "Hardcoded value in cipher IV position (3rd argument)."
                    ),
                )

    def _check_cryptography_mode(self, node: ast.Call) -> None:
        """Check cryptography library mode construction for IV."""
        if node.args:
            iv_arg = node.args[0]
            if self._is_constant_value(iv_arg):
                self._add_finding(
                    severity=Severity.CRITICAL,
                    line=node.lineno,
                    col=node.col_offset,
                    description=(
                        "Hardcoded IV/nonce passed to cipher mode constructor."
                    ),
                )
