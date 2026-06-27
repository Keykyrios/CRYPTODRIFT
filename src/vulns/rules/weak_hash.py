"""
WEAK_HASH detection rule.

Detects usage of MD5 or SHA-1 for security-sensitive operations.

Verified:
- MD5: Collision attacks demonstrated since 2004 (Wang et al.)
- SHA-1: SHAttered collision attack 2017 (Google/CWI)
- Both are still fine for checksums/non-security use, but NOT for:
  signatures, MACs, key derivation, certificate verification
"""

from __future__ import annotations

import ast

from . import BaseRule, Severity


# Weak hash functions
WEAK_HASH_CALLS = {
    "hashlib.md5": "MD5",
    "hashlib.sha1": "SHA-1",
    "MD5.new": "MD5",
    "SHA.new": "SHA-1",
    "SHA1.new": "SHA-1",
}

# Hash functions used in HMAC context (still weak for new applications)
WEAK_HMAC_HASHES = {"md5", "sha1", "MD5", "SHA1", "sha"}


class WeakHashRule(BaseRule):

    @property
    def rule_id(self) -> str:
        return "WEAK_HASH"

    @property
    def description(self) -> str:
        return "Use of broken hash function (MD5/SHA-1) for security"

    def visit_Call(self, node: ast.Call) -> None:
        call_name = self._get_call_name(node)

        # Direct weak hash usage
        if call_name in WEAK_HASH_CALLS:
            hash_name = WEAK_HASH_CALLS[call_name]
            self._add_finding(
                severity=Severity.HIGH,
                line=node.lineno,
                col=node.col_offset,
                description=(
                    f"{hash_name} is cryptographically broken. "
                    f"Collision attacks are practical. "
                    f"Use SHA-256 or SHA-3 for security applications."
                ),
            )

        # HMAC with weak hash
        if call_name in ("hmac.new", "HMAC.new", "hmac.digest"):
            self._check_hmac_hash(node)

        self.generic_visit(node)

    def _check_hmac_hash(self, node: ast.Call) -> None:
        """Check if HMAC uses a weak hash function."""
        # hmac.new(key, msg, digestmod) - digestmod is 3rd positional
        # or keyword 'digestmod'
        digestmod = None

        if len(node.args) >= 3:
            digestmod = node.args[2]

        dm_kw = self._get_keyword_arg(node, "digestmod")
        if dm_kw is not None:
            digestmod = dm_kw

        if digestmod is not None:
            # Check for string constant like "md5", "sha1"
            if isinstance(digestmod, ast.Constant):
                if isinstance(digestmod.value, str):
                    if digestmod.value.lower() in WEAK_HMAC_HASHES:
                        self._add_finding(
                            severity=Severity.HIGH,
                            line=node.lineno,
                            col=node.col_offset,
                            description=(
                                f"HMAC with weak hash '{digestmod.value}'. "
                                "Use 'sha256' or 'sha384' instead."
                            ),
                        )

            # Check for attribute like hashlib.md5
            if isinstance(digestmod, ast.Attribute):
                if digestmod.attr.lower() in WEAK_HMAC_HASHES:
                    self._add_finding(
                        severity=Severity.HIGH,
                        line=node.lineno,
                        col=node.col_offset,
                        description=(
                            f"HMAC with weak hash hashlib.{digestmod.attr}. "
                            "Use hashlib.sha256 instead."
                        ),
                    )
