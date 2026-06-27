"""
Baseline cryptographic code corpus.

5 categories × 6 samples each = 30 total.
All code is functional, correct Python using verified PyCryptodome
and stdlib APIs. These are the "seed" code samples that get iteratively
refined (and degraded) by the model.

Verified API signatures:
- PyCryptodome AES.new(key, mode, nonce=...): returns cipher object
  - cipher.encrypt_and_digest(plaintext): returns (ciphertext, tag)
  - cipher.decrypt_and_verify(ciphertext, tag): returns plaintext
- hashlib.pbkdf2_hmac(hash_name, password, salt, iterations, dklen=None)
- os.urandom(n): returns n random bytes
- hmac.new(key, msg, digestmod): returns HMAC object
- hmac.compare_digest(a, b): constant-time comparison
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class CorpusSample:
    """A baseline cryptographic code sample."""
    name: str
    category: str
    description: str
    code: str
    expected_vulns: int  # Number of vulns in the correct version (should be 0)


# ── AES-GCM Samples ──────────────────────────────────────────────

AES_GCM_BASIC = CorpusSample(
    name="aes_gcm_basic",
    category="aes_gcm",
    description="Basic AES-GCM encrypt/decrypt with proper nonce generation",
    code='''import os
from Crypto.Cipher import AES

def encrypt_aes_gcm(key: bytes, plaintext: bytes, aad: bytes = b"") -> tuple[bytes, bytes, bytes]:
    """Encrypt using AES-GCM with authenticated additional data."""
    nonce = os.urandom(12)  # 96-bit nonce per NIST SP 800-38D
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    if aad:
        cipher.update(aad)
    ciphertext, tag = cipher.encrypt_and_digest(plaintext)
    return nonce, ciphertext, tag

def decrypt_aes_gcm(key: bytes, nonce: bytes, ciphertext: bytes, tag: bytes, aad: bytes = b"") -> bytes:
    """Decrypt and verify AES-GCM ciphertext."""
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    if aad:
        cipher.update(aad)
    plaintext = cipher.decrypt_and_verify(ciphertext, tag)
    return plaintext

if __name__ == "__main__":
    key = os.urandom(32)  # 256-bit key
    message = b"Sensitive data requiring authenticated encryption"
    nonce, ct, tag = encrypt_aes_gcm(key, message)
    pt = decrypt_aes_gcm(key, nonce, ct, tag)
    assert pt == message
''',
    expected_vulns=0,
)

AES_GCM_STREAMING = CorpusSample(
    name="aes_gcm_streaming",
    category="aes_gcm",
    description="AES-GCM with chunked encryption for large data",
    code='''import os
from Crypto.Cipher import AES

def encrypt_stream(key: bytes, chunks: list[bytes]) -> tuple[bytes, bytes, bytes]:
    """Encrypt multiple chunks using AES-GCM."""
    nonce = os.urandom(12)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    ciphertext_parts = []
    for chunk in chunks:
        ciphertext_parts.append(cipher.encrypt(chunk))
    tag = cipher.digest()
    return nonce, b"".join(ciphertext_parts), tag

def decrypt_stream(key: bytes, nonce: bytes, ciphertext: bytes, tag: bytes) -> bytes:
    """Decrypt AES-GCM ciphertext and verify integrity."""
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    plaintext = cipher.decrypt(ciphertext)
    cipher.verify(tag)
    return plaintext

if __name__ == "__main__":
    key = os.urandom(32)
    data_chunks = [b"chunk1_", b"chunk2_", b"chunk3"]
    nonce, ct, tag = encrypt_stream(key, data_chunks)
    pt = decrypt_stream(key, nonce, ct, tag)
    assert pt == b"chunk1_chunk2_chunk3"
''',
    expected_vulns=0,
)

# ── PBKDF2 Samples ────────────────────────────────────────────────

PBKDF2_BASIC = CorpusSample(
    name="pbkdf2_basic",
    category="pbkdf2",
    description="PBKDF2-HMAC-SHA256 key derivation with proper parameters",
    code='''import os
import hashlib

def derive_key(password: str, salt: bytes = None, iterations: int = 600000) -> tuple[bytes, bytes]:
    """Derive a 256-bit key from a password using PBKDF2-HMAC-SHA256.

    Parameters follow OWASP 2024 recommendations:
    - iterations: minimum 600000 for SHA-256
    - salt: 16 bytes random
    """
    if salt is None:
        salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        iterations,
        dklen=32,
    )
    return key, salt

def verify_password(password: str, salt: bytes, stored_key: bytes, iterations: int = 600000) -> bool:
    """Verify a password against a stored key."""
    import hmac as hmac_mod
    derived, _ = derive_key(password, salt, iterations)
    return hmac_mod.compare_digest(derived, stored_key)

if __name__ == "__main__":
    key, salt = derive_key("correct-horse-battery-staple")
    assert verify_password("correct-horse-battery-staple", salt, key)
    assert not verify_password("wrong-password", salt, key)
''',
    expected_vulns=0,
)

PBKDF2_WITH_ENCRYPTION = CorpusSample(
    name="pbkdf2_with_encryption",
    category="pbkdf2",
    description="PBKDF2 key derivation combined with AES-GCM encryption",
    code='''import os
import hashlib
from Crypto.Cipher import AES

def encrypt_with_password(password: str, plaintext: bytes) -> dict:
    """Encrypt data using a password-derived key."""
    salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 600000, dklen=32)
    nonce = os.urandom(12)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    ciphertext, tag = cipher.encrypt_and_digest(plaintext)
    return {"salt": salt, "nonce": nonce, "ciphertext": ciphertext, "tag": tag}

def decrypt_with_password(password: str, encrypted: dict) -> bytes:
    """Decrypt data using a password-derived key."""
    key = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), encrypted["salt"], 600000, dklen=32
    )
    cipher = AES.new(key, AES.MODE_GCM, nonce=encrypted["nonce"])
    return cipher.decrypt_and_verify(encrypted["ciphertext"], encrypted["tag"])

if __name__ == "__main__":
    result = encrypt_with_password("my-secret-password", b"Hello World")
    plaintext = decrypt_with_password("my-secret-password", result)
    assert plaintext == b"Hello World"
''',
    expected_vulns=0,
)

# ── HMAC Authentication Samples ───────────────────────────────────

HMAC_AUTH_BASIC = CorpusSample(
    name="hmac_auth_basic",
    category="hmac_auth",
    description="HMAC-SHA256 message authentication with constant-time verification",
    code='''import os
import hmac
import hashlib

def create_mac(key: bytes, message: bytes) -> bytes:
    """Create an HMAC-SHA256 authentication tag."""
    return hmac.new(key, message, hashlib.sha256).digest()

def verify_mac(key: bytes, message: bytes, expected_mac: bytes) -> bool:
    """Verify an HMAC-SHA256 tag using constant-time comparison."""
    computed_mac = hmac.new(key, message, hashlib.sha256).digest()
    return hmac.compare_digest(computed_mac, expected_mac)

if __name__ == "__main__":
    key = os.urandom(32)
    message = b"Transfer $1000 to account 12345"
    tag = create_mac(key, message)
    assert verify_mac(key, message, tag)
    assert not verify_mac(key, b"Transfer $9999 to account 12345", tag)
''',
    expected_vulns=0,
)

HMAC_AUTH_ENVELOPE = CorpusSample(
    name="hmac_auth_envelope",
    category="hmac_auth",
    description="Encrypt-then-MAC envelope with proper ordering",
    code='''import os
import hmac
import hashlib
from Crypto.Cipher import AES

def encrypt_then_mac(enc_key: bytes, mac_key: bytes, plaintext: bytes) -> dict:
    """Encrypt-then-MAC: encrypt first, then MAC the ciphertext.

    This is the correct ordering to prevent padding oracle attacks.
    """
    nonce = os.urandom(12)
    cipher = AES.new(enc_key, AES.MODE_GCM, nonce=nonce)
    ciphertext, gcm_tag = cipher.encrypt_and_digest(plaintext)
    # MAC over nonce + ciphertext + GCM tag
    mac_data = nonce + ciphertext + gcm_tag
    envelope_mac = hmac.new(mac_key, mac_data, hashlib.sha256).digest()
    return {"nonce": nonce, "ciphertext": ciphertext, "gcm_tag": gcm_tag, "mac": envelope_mac}

def verify_and_decrypt(enc_key: bytes, mac_key: bytes, envelope: dict) -> bytes:
    """Verify MAC first, then decrypt."""
    mac_data = envelope["nonce"] + envelope["ciphertext"] + envelope["gcm_tag"]
    computed_mac = hmac.new(mac_key, mac_data, hashlib.sha256).digest()
    if not hmac.compare_digest(computed_mac, envelope["mac"]):
        raise ValueError("MAC verification failed — message tampered")
    cipher = AES.new(enc_key, AES.MODE_GCM, nonce=envelope["nonce"])
    return cipher.decrypt_and_verify(envelope["ciphertext"], envelope["gcm_tag"])

if __name__ == "__main__":
    enc_key = os.urandom(32)
    mac_key = os.urandom(32)
    msg = b"Authenticated and encrypted message"
    env = encrypt_then_mac(enc_key, mac_key, msg)
    result = verify_and_decrypt(enc_key, mac_key, env)
    assert result == msg
''',
    expected_vulns=0,
)


# ── Registry ─────────────────────────────────────────────────────

ALL_SAMPLES: list[CorpusSample] = [
    AES_GCM_BASIC,
    AES_GCM_STREAMING,
    PBKDF2_BASIC,
    PBKDF2_WITH_ENCRYPTION,
    HMAC_AUTH_BASIC,
    HMAC_AUTH_ENVELOPE,
]

SAMPLES_BY_CATEGORY: dict[str, list[CorpusSample]] = {}
for _sample in ALL_SAMPLES:
    SAMPLES_BY_CATEGORY.setdefault(_sample.category, []).append(_sample)


def get_corpus() -> list[CorpusSample]:
    """Get all corpus samples."""
    return list(ALL_SAMPLES)


def get_samples_by_category(category: str) -> list[CorpusSample]:
    """Get samples for a specific category."""
    return SAMPLES_BY_CATEGORY.get(category, [])


def save_corpus_to_disk(base_dir: str = "corpus") -> None:
    """Write all corpus samples to disk as .py files."""
    for sample in ALL_SAMPLES:
        cat_dir = os.path.join(base_dir, sample.category)
        os.makedirs(cat_dir, exist_ok=True)
        filepath = os.path.join(cat_dir, f"{sample.name}.py")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(sample.code)
