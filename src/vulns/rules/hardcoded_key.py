"""
HARDCODED_KEY detection rule.

Detects cryptographic keys, secrets, and passwords hardcoded
as string or bytes literals in source code.
"""

from __future__ import annotations

import ast
import re

from . import BaseRule, Severity


# Variable names suggesting cryptographic secrets
SECRET_NAMES = {
    "key", "secret", "secret_key", "api_key", "private_key",
    "encryption_key", "signing_key", "master_key", "password",
    "passphrase", "aes_key", "hmac_key", "auth_key", "token",
    "secret_token", "cipher_key",
}

# Additional patterns (partial matches)
SECRET_PATTERNS = [
    r"_key$", r"_secret$", r"_password$", r"_token$",
    r"^key_", r"^secret_", r"^password",
]


class HardcodedKeyRule(BaseRule):

    @property
    def rule_id(self) -> str:
        return "HARDCODED_KEY"

    @property
    def description(self) -> str:
        return "Hardcoded cryptographic key or secret"

    def visit_Assign(self, node: ast.Assign) -> None:
        """Check for secret-looking variable names with literal values."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                if self._is_secret_name(target.id):
                    if self._is_hardcoded_secret(node.value):
                        self._add_finding(
                            severity=Severity.CRITICAL,
                            line=node.lineno,
                            col=node.col_offset,
                            description=(
                                f"Hardcoded secret in variable '{target.id}'. "
                                "Keys and secrets must not be stored in source "
                                "code. Use environment variables or a "
                                "secrets manager."
                            ),
                        )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Check for hardcoded keys passed directly to crypto functions."""
        call_name = self._get_call_name(node)

        # Check AES constructors for hardcoded key (first arg)
        if call_name in ("AES.new", "DES.new", "DES3.new",
                         "Fernet", "ChaCha20.new"):
            if node.args and self._is_hardcoded_secret(node.args[0]):
                self._add_finding(
                    severity=Severity.CRITICAL,
                    line=node.lineno,
                    col=node.col_offset,
                    description=(
                        f"Hardcoded key passed to {call_name}(). "
                        "Keys must never be hardcoded in source."
                    ),
                )

        # Check keyword arguments named 'key'
        key_kw = self._get_keyword_arg(node, "key")
        if key_kw is not None and self._is_hardcoded_secret(key_kw):
            self._add_finding(
                severity=Severity.CRITICAL,
                line=node.lineno,
                col=node.col_offset,
                description=(
                    f"Hardcoded value for 'key' parameter in {call_name}()."
                ),
            )

        self.generic_visit(node)

    def _is_secret_name(self, name: str) -> bool:
        """Check if a variable name suggests a cryptographic secret."""
        name_lower = name.lower()
        if name_lower in SECRET_NAMES:
            return True
        for pattern in SECRET_PATTERNS:
            if re.search(pattern, name_lower):
                return True
        return False

    def _is_hardcoded_secret(self, node: ast.expr) -> bool:
        """Check if a node represents a hardcoded secret value."""
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (str, bytes)):
                # Ignore empty strings and very short strings (unlikely keys)
                val = node.value
                if isinstance(val, str) and len(val) >= 4:
                    return True
                if isinstance(val, bytes) and len(val) >= 4:
                    return True
        return False
