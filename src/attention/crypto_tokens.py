"""
Crypto token vocabulary and position mapping.

Maps cryptographic keywords to their likely token positions
in the Llama 3.1 8B tokenizer output.

Llama 3.1 uses a BPE tokenizer with 128K vocab.
Crypto terms may be split into subwords:
- "PBKDF2" → ["P", "BK", "DF", "2"] or similar
- "AES" → might be a single token or split
- "os.urandom" → ["os", ".", "ur", "andom"] or similar

We use a prefix-matching strategy: for each crypto keyword,
we find all token positions where any subword of that keyword appears.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Crypto-relevant vocabulary organized by category
CRYPTO_TOKEN_VOCABULARY: dict[str, list[str]] = {
    "primitives": [
        "AES", "RSA", "Ed25519", "Kyber", "HMAC", "SHA",
        "PBKDF2", "scrypt", "argon2", "ChaCha20", "Poly1305",
        "DES", "Blowfish", "Fernet", "X25519", "ECDSA",
    ],
    "operations": [
        "encrypt", "decrypt", "sign", "verify", "derive",
        "hash", "digest", "finalize", "update", "mac",
    ],
    "parameters": [
        "key", "iv", "nonce", "salt", "iterations", "counter",
        "tag", "aad", "password", "secret", "token",
    ],
    "modes": [
        "GCM", "CBC", "ECB", "CTR", "CFB", "CCM", "SIV",
    ],
    "security": [
        "random", "urandom", "secrets", "compare_digest",
        "constant_time",
    ],
}

# Flattened list for quick lookup
ALL_CRYPTO_KEYWORDS = []
for category_terms in CRYPTO_TOKEN_VOCABULARY.values():
    ALL_CRYPTO_KEYWORDS.extend(category_terms)
ALL_CRYPTO_KEYWORDS_LOWER = {kw.lower() for kw in ALL_CRYPTO_KEYWORDS}


# Security-critical token pairs for cross-attention analysis
# Format: (token_a, token_b) — we measure attention between these
CRITICAL_TOKEN_PAIRS = [
    ("iv", "encrypt"),
    ("nonce", "encrypt"),
    ("key", "derive"),
    ("key", "encrypt"),
    ("salt", "PBKDF2"),
    ("salt", "scrypt"),
    ("random", "key"),
    ("random", "iv"),
    ("HMAC", "verify"),
    ("tag", "verify"),
    ("GCM", "nonce"),
    ("CBC", "iv"),
]


def find_crypto_token_positions(
    tokenizer: Any,
    code: str,
) -> dict[str, list[int]]:
    """
    Find positions of crypto-relevant tokens in tokenized code.

    Args:
        tokenizer: HuggingFace tokenizer instance.
        code: Source code string.

    Returns:
        Dict mapping crypto keyword -> list of token positions.
    """
    # Tokenize the code
    encoding = tokenizer(code, return_tensors="pt", add_special_tokens=False)
    input_ids = encoding["input_ids"][0]

    # Decode each token individually
    token_strings = []
    for i in range(len(input_ids)):
        token_str = tokenizer.decode(
            [input_ids[i].item()],
            skip_special_tokens=True,
        ).strip()
        token_strings.append(token_str.lower())

    # Match crypto keywords to token positions
    positions: dict[str, list[int]] = {}

    for keyword in ALL_CRYPTO_KEYWORDS:
        keyword_lower = keyword.lower()
        matched_positions = []

        for idx, token_str in enumerate(token_strings):
            if not token_str:
                continue
            # Exact match
            if token_str == keyword_lower:
                matched_positions.append(idx)
            # Substring match (for subword tokens)
            elif keyword_lower in token_str or token_str in keyword_lower:
                # Only match if it's a significant overlap
                if len(token_str) >= 2:
                    matched_positions.append(idx)

        if matched_positions:
            positions[keyword] = matched_positions

    return positions


def get_all_crypto_positions(
    positions: dict[str, list[int]],
) -> list[int]:
    """Get a flat sorted list of all crypto token positions."""
    all_pos: set[int] = set()
    for pos_list in positions.values():
        all_pos.update(pos_list)
    return sorted(all_pos)


def find_token_pair_positions(
    positions: dict[str, list[int]],
    pair: tuple[str, str],
) -> list[tuple[int, int]]:
    """
    Find position pairs for a critical token pair.

    Returns list of (pos_a, pos_b) tuples for all combinations
    of positions where both tokens appear.
    """
    keyword_a, keyword_b = pair

    # Find positions for each keyword (case-insensitive)
    pos_a = []
    pos_b = []

    for kw, positions_list in positions.items():
        if kw.lower() == keyword_a.lower():
            pos_a.extend(positions_list)
        if kw.lower() == keyword_b.lower():
            pos_b.extend(positions_list)

    if not pos_a or not pos_b:
        return []

    # Return all (a, b) pairs
    return [(a, b) for a in pos_a for b in pos_b if a != b]
