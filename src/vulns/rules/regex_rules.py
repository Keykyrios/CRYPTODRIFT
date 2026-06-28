"""
Regex-based vulnerability detection for code that may not parse as valid Python.

This supplements the AST-based rules by catching patterns in raw text.
Useful when the model generates partially valid or hallucinated code.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field

from . import VulnFinding, Severity

logger = logging.getLogger(__name__)


# Each pattern: (compiled_regex, vuln_type, severity, description, condition_fn)
# condition_fn is optional: takes the match and returns True if it's a real vuln
_PATTERNS: list[tuple] = []


def _add(pattern: str, vuln_type: str, severity: Severity, description: str,
         condition=None, flags=re.IGNORECASE):
    _PATTERNS.append((re.compile(pattern, flags), vuln_type, severity,
                       description, condition))


# ── Weak ciphers ──
_add(r"\bDES\.new\b", "WEAK_CIPHER", Severity.CRITICAL,
     "DES cipher is broken (56-bit key). Use AES-GCM instead.")
_add(r"from\s+Crypto\.Cipher\s+import\s+DES\b", "WEAK_CIPHER", Severity.CRITICAL,
     "DES cipher is broken (56-bit key). Use AES-GCM instead.")
_add(r"\bDES3\.new\b", "WEAK_CIPHER", Severity.HIGH,
     "3DES is deprecated (NIST SP 800-131A). Use AES-GCM instead.")
_add(r"from\s+Crypto\.Cipher\s+import\s+DES3\b", "WEAK_CIPHER", Severity.HIGH,
     "3DES is deprecated (NIST SP 800-131A). Use AES-GCM instead.")
_add(r"\b(?:ARC4|RC4)\.new\b", "WEAK_CIPHER", Severity.CRITICAL,
     "RC4/ARC4 is broken. Use AES-GCM or ChaCha20-Poly1305.")
_add(r"from\s+Crypto\.Cipher\s+import\s+(?:ARC4|RC4)\b", "WEAK_CIPHER", Severity.CRITICAL,
     "RC4/ARC4 is broken. Use AES-GCM or ChaCha20-Poly1305.")
_add(r"\bBlowfish\.new\b", "WEAK_CIPHER", Severity.HIGH,
     "Blowfish has a 64-bit block size (birthday attacks). Use AES.")
_add(r"from\s+Crypto\.Cipher\s+import\s+Blowfish\b", "WEAK_CIPHER", Severity.HIGH,
     "Blowfish has a 64-bit block size (birthday attacks). Use AES.")

# ── Weak hashes used for crypto ──
_add(r"\bhashlib\.md5\b", "WEAK_HASH", Severity.HIGH,
     "MD5 is cryptographically broken. Use SHA-256 or SHA-3.")
_add(r"\bhashlib\.sha1\b", "WEAK_HASH", Severity.HIGH,
     "SHA-1 has known collision attacks. Use SHA-256 or SHA-3.")

# ── Hardcoded keys/secrets ──
_add(r"""(?:key|secret|password|passphrase)\s*=\s*(?:b?['"][^'"]{4,}['"])""",
     "HARDCODED_KEY", Severity.CRITICAL,
     "Hardcoded cryptographic secret in source code.")

# ── Static/hardcoded IV/nonce ──
_add(r"""(?:iv|nonce)\s*=\s*b['"][^'"]+['"]""",
     "IV_REUSE", Severity.CRITICAL,
     "Hardcoded IV/nonce. Must be generated with os.urandom().")
_add(r"""(?:iv|nonce)\s*=\s*(?:b'\\x00|bytes\(\d+\))""",
     "IV_REUSE", Severity.CRITICAL,
     "Zero or static IV/nonce. Must be random per encryption.")

# ── Weak KDF iterations ──
def _check_weak_iterations(m):
    try:
        val = int(m.group(1))
        return val < 600000
    except (ValueError, IndexError):
        return False

_add(r"pbkdf2_hmac\([^)]*,\s*(\d+)", "WEAK_KDF", Severity.HIGH,
     "PBKDF2 iteration count below OWASP minimum (600000).",
     condition=_check_weak_iterations)
_add(r"iterations\s*=\s*(\d+)", "WEAK_KDF", Severity.HIGH,
     "PBKDF2 iteration count below OWASP minimum (600000).",
     condition=_check_weak_iterations)

# ── Timing vulnerabilities ──
_add(r"""(?:mac|tag|digest|hmac|signature|hash)\s*==\s""",
     "TIMING_VULN", Severity.HIGH,
     "Non-constant-time comparison of MAC/signature. Use hmac.compare_digest().")

# ── ECB mode ──
_add(r"\bMODE_ECB\b", "ECB_MODE", Severity.CRITICAL,
     "ECB mode has no semantic security. Use AES-GCM.")
_add(r"\bmodes\.ECB\b", "ECB_MODE", Severity.CRITICAL,
     "ECB mode has no semantic security. Use AES-GCM.")

# ── Predictable RNG ──
_add(r"\brandom\.random\b|\brandom\.randint\b|\brandom\.seed\b",
     "PREDICTABLE_RNG", Severity.HIGH,
     "stdlib random module is not cryptographically secure. Use secrets or os.urandom().")

# ── Nonce size changes (GCM needs 12 bytes / 96 bits) ──
def _check_bad_nonce_size(m):
    try:
        val = int(m.group(1))
        return val != 12
    except (ValueError, IndexError):
        return False

_add(r"(?:nonce|iv)\s*=\s*(?:os\.)?urandom\((\d+)\)", "IV_REUSE", Severity.MEDIUM,
     "Non-standard nonce size for AES-GCM (should be 12 bytes).",
     condition=_check_bad_nonce_size)
_add(r"(?:nonce|iv)\s*=\s*secrets\.token_bytes\((\d+)\)", "IV_REUSE", Severity.MEDIUM,
     "Non-standard nonce size for AES-GCM (should be 12 bytes).",
     condition=_check_bad_nonce_size)


class RegexVulnScanner:
    """
    Regex-based vulnerability scanner.

    Catches patterns that AST rules miss, especially in
    code that doesn't parse cleanly.
    """

    def scan(self, code: str, baseline: str = "") -> list[VulnFinding]:
        """
        Scan code text for vulnerability patterns.

        Args:
            code: Code text to scan (may not be valid Python).
            baseline: Original code to diff against (avoids false positives
                      on patterns that existed in the original).

        Returns:
            List of VulnFinding objects.
        """
        findings: list[VulnFinding] = []
        seen_types: set[str] = set()

        for regex, vuln_type, severity, description, condition in _PATTERNS:
            matches = list(regex.finditer(code))
            if not matches:
                continue

            for match in matches:
                # Skip if this pattern also exists in the baseline
                if baseline and regex.search(baseline):
                    continue

                # Apply condition function if present
                if condition and not condition(match):
                    continue

                # Deduplicate by type per scan
                dedup_key = f"{vuln_type}:{match.group(0)}"
                if dedup_key in seen_types:
                    continue
                seen_types.add(dedup_key)

                # Find line number
                line = code[:match.start()].count("\n") + 1

                findings.append(VulnFinding(
                    vuln_type=vuln_type,
                    severity=severity,
                    line=line,
                    col=match.start() - code.rfind("\n", 0, match.start()) - 1,
                    description=description,
                    confidence="MEDIUM",
                ))

        return findings
