"""
Attention entropy computation.

Shannon entropy of attention distributions, computed per-head.

Verified math:
- Shannon entropy: H = -Σ p(i) * log2(p(i))
- For uniform distribution over N items: H_max = log2(N)
- Normalized entropy: H / H_max ∈ [0, 1]
- Low entropy = concentrated attention = model is "confident"
- Entropy collapse = sudden drop in entropy = danger signal

Uses numpy for computation (not scipy.stats.entropy, which uses
natural log by default — we want log2 for bits).
Actually scipy.stats.entropy(pk, base=2) would also work,
but numpy avoids the dependency for this simple computation.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


def attention_entropy(
    attn_weights: np.ndarray,
    axis: int = -1,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Compute Shannon entropy of attention distributions.

    Args:
        attn_weights: Attention weights, shape (..., seq_len).
            The last dimension should sum to ~1.0 (softmax output).
        axis: Axis along which to compute entropy.
        eps: Small constant to avoid log(0).

    Returns:
        Entropy values with the specified axis removed.
    """
    # Clip to avoid log(0) — attention weights should already be ≥ 0
    p = np.clip(attn_weights, eps, 1.0)

    # Renormalize (in case clipping changed the sum)
    p = p / p.sum(axis=axis, keepdims=True)

    # Shannon entropy: H = -Σ p * log2(p)
    log_p = np.log2(p)
    entropy = -np.sum(p * log_p, axis=axis)

    return entropy


def normalized_entropy(
    attn_weights: np.ndarray,
    axis: int = -1,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Compute normalized entropy ∈ [0, 1].

    Normalized by max possible entropy (uniform distribution).

    Args:
        attn_weights: Attention weights, shape (..., seq_len).
        axis: Axis along which to compute.

    Returns:
        Normalized entropy values.
    """
    raw = attention_entropy(attn_weights, axis=axis, eps=eps)
    seq_len = attn_weights.shape[axis]
    max_entropy = np.log2(seq_len) if seq_len > 1 else 1.0
    return raw / max_entropy


def per_head_entropy(
    layer_snapshot: np.ndarray,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Compute entropy for each attention head in a layer.

    Args:
        layer_snapshot: Shape (num_heads, seq_len, seq_len).
            Attention weights for all heads in one layer.

    Returns:
        Shape (num_heads,) — mean entropy per head across all
        query positions.
    """
    # layer_snapshot shape: (heads, query_len, key_len)
    # Compute entropy for each head at each query position
    # Then average across query positions
    per_position = attention_entropy(layer_snapshot, axis=-1)
    # per_position shape: (heads, query_len)
    return np.mean(per_position, axis=-1)  # shape: (heads,)


def per_head_normalized_entropy(
    layer_snapshot: np.ndarray,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Compute normalized entropy for each head.

    Args:
        layer_snapshot: Shape (num_heads, seq_len, seq_len).

    Returns:
        Shape (num_heads,) — mean normalized entropy per head.
    """
    per_position = normalized_entropy(layer_snapshot, axis=-1)
    return np.mean(per_position, axis=-1)


def crypto_token_entropy(
    layer_snapshot: np.ndarray,
    crypto_positions: list[int],
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Compute entropy of attention FROM crypto tokens only.

    Measures how the model distributes attention when generating
    at crypto-relevant positions.

    Args:
        layer_snapshot: Shape (num_heads, seq_len, seq_len).
        crypto_positions: Token positions corresponding to crypto keywords.

    Returns:
        Shape (num_heads,) — mean entropy at crypto positions per head.
    """
    if not crypto_positions:
        return np.zeros(layer_snapshot.shape[0])

    # Filter to valid positions
    seq_len = layer_snapshot.shape[1]
    valid_pos = [p for p in crypto_positions if p < seq_len]

    if not valid_pos:
        return np.zeros(layer_snapshot.shape[0])

    # Extract attention rows at crypto positions
    # Shape: (heads, num_crypto_pos, seq_len)
    crypto_attn = layer_snapshot[:, valid_pos, :]

    # Compute entropy per head, averaged over crypto positions
    per_position = attention_entropy(crypto_attn, axis=-1)
    return np.mean(per_position, axis=-1)


def cross_attention_strength(
    layer_snapshot: np.ndarray,
    positions_a: list[int],
    positions_b: list[int],
) -> np.ndarray:
    """
    Compute cross-attention strength between two sets of token positions.

    Measures how strongly tokens at positions_a attend to tokens at
    positions_b (and vice versa). This captures relationships like
    IV↔encrypt, key↔derive.

    Args:
        layer_snapshot: Shape (num_heads, seq_len, seq_len).
        positions_a: First set of token positions.
        positions_b: Second set of token positions.

    Returns:
        Shape (num_heads,) — mean cross-attention strength per head.
    """
    if not positions_a or not positions_b:
        return np.zeros(layer_snapshot.shape[0])

    seq_len = layer_snapshot.shape[1]
    valid_a = [p for p in positions_a if p < seq_len]
    valid_b = [p for p in positions_b if p < seq_len]

    if not valid_a or not valid_b:
        return np.zeros(layer_snapshot.shape[0])

    # A→B: attention FROM positions_a TO positions_b
    # Shape: (heads, len(a), len(b))
    cross_ab = layer_snapshot[:, valid_a, :][:, :, valid_b]

    # Mean attention mass from A to B, per head
    strength = np.mean(cross_ab, axis=(1, 2))

    return strength


def detect_entropy_collapse(
    entropy_history: list[np.ndarray],
    threshold_drop: float = 0.3,
    window: int = 2,
) -> list[tuple[int, int, float]]:
    """
    Detect entropy collapse events in a history of per-head entropies.

    An entropy collapse is when a head's normalized entropy drops
    by more than threshold_drop within a window of iterations.

    Args:
        entropy_history: List of arrays, each shape (num_heads,).
        threshold_drop: Minimum entropy drop to trigger collapse.
        window: Number of iterations to look back.

    Returns:
        List of (iteration, head_idx, drop_magnitude) tuples.
    """
    collapses = []

    for i in range(window, len(entropy_history)):
        current = entropy_history[i]
        previous = entropy_history[i - window]

        drops = previous - current  # positive = entropy decreased
        for head_idx in range(len(current)):
            if drops[head_idx] > threshold_drop:
                collapses.append((i, head_idx, float(drops[head_idx])))

    return collapses
