"""
WEAK_KDF detection rule.

Detects weak key derivation function usage.

Patterns detected:
- PBKDF2 with iterations < 600000 (OWASP 2024 recommendation for SHA-256)
- scrypt with N < 2^14 (16384)
- Plain SHA/MD5 used for password hashing
- bcrypt with rounds < 12
"""

from __future__ import annotations

import ast

from . import BaseRule, Severity


# OWASP 2024 minimum iterations for PBKDF2-HMAC-SHA256
# https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html
PBKDF2_MIN_ITERATIONS = 600_000
SCRYPT_MIN_N = 16384  # 2^14


class WeakKDFRule(BaseRule):

    @property
    def rule_id(self) -> str:
        return "WEAK_KDF"

    @property
    def description(self) -> str:
        return "Weak key derivation function parameters"

    def visit_Call(self, node: ast.Call) -> None:
        call_name = self._get_call_name(node)

        if call_name in ("hashlib.pbkdf2_hmac", "PBKDF2",
                         "pbkdf2.PBKDF2HMAC"):
            self._check_pbkdf2(node, call_name)
        elif call_name in ("hashlib.scrypt", "scrypt"):
            self._check_scrypt(node, call_name)
        elif call_name in ("bcrypt.hashpw", "bcrypt.gensalt"):
            self._check_bcrypt(node, call_name)
        elif call_name in ("hashlib.sha256", "hashlib.sha512",
                           "hashlib.sha1", "hashlib.md5"):
            self._check_plain_hash_for_password(node, call_name)

        self.generic_visit(node)

    def _check_pbkdf2(self, node: ast.Call, call_name: str) -> None:
        """Check PBKDF2 iteration count."""
        iterations_value = None

        if call_name == "hashlib.pbkdf2_hmac":
            # hashlib.pbkdf2_hmac(hash_name, password, salt, iterations)
            # iterations is the 4th positional arg or 'iterations' keyword
            if len(node.args) >= 4:
                iter_node = node.args[3]
                if isinstance(iter_node, ast.Constant) and isinstance(
                    iter_node.value, int
                ):
                    iterations_value = iter_node.value

            iter_kw = self._get_keyword_arg(node, "iterations")
            if iter_kw is not None and isinstance(iter_kw, ast.Constant):
                if isinstance(iter_kw.value, int):
                    iterations_value = iter_kw.value

        elif call_name == "pbkdf2.PBKDF2HMAC":
            # cryptography library: PBKDF2HMAC(algorithm, length, salt, iterations)
            if len(node.args) >= 4:
                iter_node = node.args[3]
                if isinstance(iter_node, ast.Constant) and isinstance(
                    iter_node.value, int
                ):
                    iterations_value = iter_node.value

            iter_kw = self._get_keyword_arg(node, "iterations")
            if iter_kw is not None and isinstance(iter_kw, ast.Constant):
                if isinstance(iter_kw.value, int):
                    iterations_value = iter_kw.value

        if iterations_value is not None and iterations_value < PBKDF2_MIN_ITERATIONS:
            severity = (
                Severity.CRITICAL if iterations_value < 100_000
                else Severity.HIGH
            )
            self._add_finding(
                severity=severity,
                line=node.lineno,
                col=node.col_offset,
                description=(
                    f"PBKDF2 iterations={iterations_value} is below "
                    f"OWASP minimum ({PBKDF2_MIN_ITERATIONS}). "
                    "Low iteration counts enable brute-force attacks."
                ),
            )

    def _check_scrypt(self, node: ast.Call, call_name: str) -> None:
        """Check scrypt N parameter."""
        n_value = None

        # hashlib.scrypt(password, *, salt, n, r, p, ...)
        n_kw = self._get_keyword_arg(node, "n")
        if n_kw is not None and isinstance(n_kw, ast.Constant):
            if isinstance(n_kw.value, int):
                n_value = n_kw.value

        if n_value is not None and n_value < SCRYPT_MIN_N:
            self._add_finding(
                severity=Severity.HIGH,
                line=node.lineno,
                col=node.col_offset,
                description=(
                    f"scrypt N={n_value} is below minimum ({SCRYPT_MIN_N}). "
                    "Low cost factor enables brute-force attacks."
                ),
            )

    def _check_bcrypt(self, node: ast.Call, call_name: str) -> None:
        """Check bcrypt rounds."""
        if call_name == "bcrypt.gensalt":
            # bcrypt.gensalt(rounds=12)
            rounds_kw = self._get_keyword_arg(node, "rounds")
            if rounds_kw is not None and isinstance(rounds_kw, ast.Constant):
                if isinstance(rounds_kw.value, int) and rounds_kw.value < 12:
                    self._add_finding(
                        severity=Severity.MEDIUM,
                        line=node.lineno,
                        col=node.col_offset,
                        description=(
                            f"bcrypt rounds={rounds_kw.value} is low. "
                            "Minimum recommended is 12."
                        ),
                    )

    def _check_plain_hash_for_password(
        self, node: ast.Call, call_name: str
    ) -> None:
        """
        Flag plain hash usage that appears to be password-related.

        This uses heuristic analysis on nearby variable names.
        """
        # Check if the hash is being called on something that looks
        # like a password. This is a heuristic check.
        if node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Name):
                name_lower = arg.id.lower()
                if any(
                    kw in name_lower
                    for kw in ("password", "passwd", "pwd", "pass")
                ):
                    self._add_finding(
                        severity=Severity.CRITICAL,
                        line=node.lineno,
                        col=node.col_offset,
                        description=(
                            f"Plain hash ({call_name}) used for password. "
                            "Use a proper KDF like PBKDF2, scrypt, or argon2."
                        ),
                        confidence="MEDIUM",
                    )
