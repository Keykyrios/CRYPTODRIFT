"""
Attention snapshot storage and serialization.

Stores crypto-token submatrices and per-head summary statistics.
Full attention matrices are NOT stored (too large for 6GB VRAM budget).

Storage format: compressed numpy .npz files.
Metadata: stored in SQLite (via storage.database).
"""

from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .entropy import (
    per_head_normalized_entropy,
    crypto_token_entropy,
    cross_attention_strength,
)
from .crypto_tokens import (
    CRITICAL_TOKEN_PAIRS,
    get_all_crypto_positions,
    find_token_pair_positions,
)

logger = logging.getLogger(__name__)


@dataclass
class HeadSummary:
    """Summary statistics for a single attention head."""
    layer: int
    head: int
    mean_entropy: float
    crypto_entropy: float
    max_attention: float
    attention_mass_on_crypto: float

    def to_dict(self) -> dict:
        return {
            "layer": self.layer,
            "head": self.head,
            "mean_entropy": round(self.mean_entropy, 6),
            "crypto_entropy": round(self.crypto_entropy, 6),
            "max_attention": round(self.max_attention, 6),
            "attention_mass_on_crypto": round(
                self.attention_mass_on_crypto, 6
            ),
        }


@dataclass
class AttentionSnapshot:
    """
    A complete attention snapshot for one iteration.

    Contains:
    - Per-head summary statistics for all 32 layers × 32 heads
    - Crypto-token submatrices (only rows/cols for crypto tokens)
    - Cross-attention scores between critical token pairs
    """
    iteration: int
    timestamp: float = field(default_factory=time.time)
    head_summaries: list[HeadSummary] = field(default_factory=list)
    cross_attention_scores: dict[str, list[float]] = field(
        default_factory=dict
    )
    crypto_token_count: int = 0
    # Full crypto submatrices stored separately in .npz file
    snapshot_path: Optional[str] = None

    @property
    def min_entropy_head(self) -> Optional[HeadSummary]:
        """Find the head with lowest crypto entropy."""
        if not self.head_summaries:
            return None
        return min(self.head_summaries, key=lambda h: h.crypto_entropy)

    @property
    def mean_entropy(self) -> float:
        """Mean entropy across all heads."""
        if not self.head_summaries:
            return 0.0
        return float(np.mean([h.mean_entropy for h in self.head_summaries]))

    def get_entropy_matrix(self) -> np.ndarray:
        """
        Get entropy as a 32×32 matrix (layers × heads).

        This is what gets displayed in the heatmap.
        """
        matrix = np.zeros((32, 32))
        for summary in self.head_summaries:
            if summary.layer < 32 and summary.head < 32:
                matrix[summary.layer, summary.head] = summary.crypto_entropy
        return matrix

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "timestamp": self.timestamp,
            "crypto_token_count": self.crypto_token_count,
            "mean_entropy": self.mean_entropy,
            "min_entropy_head": (
                self.min_entropy_head.to_dict()
                if self.min_entropy_head
                else None
            ),
            "cross_attention_scores": self.cross_attention_scores,
        }


def create_snapshot(
    layer_snapshots: dict[int, np.ndarray],
    crypto_positions: dict[str, list[int]],
    iteration: int,
    save_dir: Optional[str] = None,
) -> AttentionSnapshot:
    """
    Create an AttentionSnapshot from raw attention weight data.

    Args:
        layer_snapshots: Dict mapping layer_idx -> attention weights
            of shape (num_heads, seq_len, seq_len).
        crypto_positions: Dict mapping crypto keyword -> token positions.
        iteration: Current iteration number.
        save_dir: Directory to save .npz files. If None, no file saved.

    Returns:
        AttentionSnapshot with computed summaries.
    """
    all_crypto_pos = get_all_crypto_positions(crypto_positions)
    snapshot = AttentionSnapshot(
        iteration=iteration,
        crypto_token_count=len(all_crypto_pos),
    )

    head_summaries = []
    crypto_submatrices = {}

    for layer_idx in sorted(layer_snapshots.keys()):
        weights = layer_snapshots[layer_idx]
        # weights shape: (num_heads, seq_len, seq_len)

        num_heads = weights.shape[0]

        # Per-head normalized entropy (full sequence)
        mean_entropies = per_head_normalized_entropy(weights)

        # Per-head crypto-token entropy
        crypto_entropies = crypto_token_entropy(weights, all_crypto_pos)

        for head_idx in range(num_heads):
            # Max attention value for this head
            max_attn = float(np.max(weights[head_idx]))

            # Attention mass on crypto tokens (how much attention
            # goes to crypto positions from all positions)
            if all_crypto_pos:
                seq_len = weights.shape[1]
                valid_pos = [p for p in all_crypto_pos if p < seq_len]
                if valid_pos:
                    mass = float(np.mean(
                        weights[head_idx, :, valid_pos].sum(axis=-1)
                    ))
                else:
                    mass = 0.0
            else:
                mass = 0.0

            head_summaries.append(HeadSummary(
                layer=layer_idx,
                head=head_idx,
                mean_entropy=float(mean_entropies[head_idx]),
                crypto_entropy=float(crypto_entropies[head_idx]),
                max_attention=max_attn,
                attention_mass_on_crypto=mass,
            ))

        # Save crypto submatrices for this layer
        if all_crypto_pos:
            seq_len = weights.shape[1]
            valid_pos = [p for p in all_crypto_pos if p < seq_len]
            if valid_pos:
                # Extract submatrix: rows and columns at crypto positions
                submatrix = weights[:, valid_pos, :][:, :, valid_pos]
                crypto_submatrices[f"layer_{layer_idx}"] = submatrix

    snapshot.head_summaries = head_summaries

    # Compute cross-attention scores for critical token pairs
    for pair in CRITICAL_TOKEN_PAIRS:
        pair_positions = find_token_pair_positions(crypto_positions, pair)
        if pair_positions:
            pair_key = f"{pair[0]}_{pair[1]}"
            scores = []
            for layer_idx, weights in sorted(layer_snapshots.items()):
                pos_a = [p[0] for p in pair_positions]
                pos_b = [p[1] for p in pair_positions]
                strength = cross_attention_strength(weights, pos_a, pos_b)
                scores.append(float(np.mean(strength)))
            snapshot.cross_attention_scores[pair_key] = scores

    # Save to .npz if directory provided
    if save_dir and crypto_submatrices:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"snapshot_iter{iteration:04d}_{int(time.time())}.npz"
        filepath = os.path.join(save_dir, filename)

        np.savez_compressed(filepath, **crypto_submatrices)
        snapshot.snapshot_path = filepath
        logger.info(
            "Saved attention snapshot: %s (%d layers)",
            filepath,
            len(crypto_submatrices),
        )

    return snapshot
