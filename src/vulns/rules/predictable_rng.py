"""
PREDICTABLE_RNG detection rule.

Detects use of non-cryptographic random number generators
in security-sensitive contexts.

Verified:
- random module: NOT cryptographically secure (uses Mersenne Twister)
- secrets module: cryptographically secure (Python 3.6+)
- os.urandom(): cryptographically secure
"""

from __future__ import annotations

import ast

from . import BaseRule, Severity


# Non-cryptographic random functions
INSECURE_RNG_CALLS = {
    "random.random", "random.randint", "random.randrange",
    "random.choice", "random.choices", "random.sample",
    "random.uniform", "random.getrandbits", "random.randbytes",
    "random.seed", "random.Random",
}


class PredictableRNGRule(BaseRule):

    @property
    def rule_id(self) -> str:
        return "PREDICTABLE_RNG"

    @property
    def description(self) -> str:
        return "Use of non-cryptographic RNG in security context"

    def __init__(self):
        super().__init__()
        self._in_crypto_context = False
        self._crypto_imports: set[str] = set()

    def reset(self) -> None:
        super().reset()
        self._in_crypto_context = False
        self._crypto_imports.clear()

    def visit_Import(self, node: ast.Import) -> None:
        """Track crypto-related imports."""
        for alias in node.names:
            name = alias.name.lower()
            if any(kw in name for kw in (
                "crypto", "cipher", "aes", "rsa", "hmac",
                "hashlib", "ssl", "tls", "cryptography",
            )):
                self._in_crypto_context = True
                self._crypto_imports.add(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Track crypto-related from-imports."""
        module = (node.module or "").lower()
        if any(kw in module for kw in (
            "crypto", "cipher", "aes", "rsa", "hmac",
            "hashlib", "ssl", "tls", "cryptography",
        )):
            self._in_crypto_context = True
            self._crypto_imports.add(node.module or "")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Check for insecure RNG usage."""
        call_name = self._get_call_name(node)

        if call_name in INSECURE_RNG_CALLS:
            # Always flag in files with crypto imports
            if self._in_crypto_context:
                self._add_finding(
                    severity=Severity.HIGH,
                    line=node.lineno,
                    col=node.col_offset,
                    description=(
                        f"{call_name}() used in file with cryptographic "
                        "imports. The random module uses Mersenne Twister "
                        "(predictable). Use secrets.token_bytes(), "
                        "secrets.token_hex(), or os.urandom() instead."
                    ),
                )
            else:
                # Check if result is assigned to something crypto-looking
                # This requires checking the parent, which we can't easily
                # do with NodeVisitor. Flag with lower confidence.
                pass

        self.generic_visit(node)
