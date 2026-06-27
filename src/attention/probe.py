"""
AttentionProbe — registers forward hooks on Llama 3.1 8B attention layers.

Verified:
- Hook registration: model.model.layers[i].self_attn.register_forward_hook()
- Output format with eager attention + output_attentions=True:
  self_attn forward returns tuple: (attn_output, attn_weights, past_key_value)
  attn_weights shape: (batch_size, num_query_heads, seq_len, seq_len)
  = (1, 32, seq_len, seq_len) for Llama 3.1 8B

- IMPORTANT: attn_weights at index 1 is ONLY present when
  output_attentions=True is passed to the model forward call.
  If not set, output may only contain (attn_output, None, past_kv).

- We use register_forward_hook which gives us (module, input, output).
  The output tuple order depends on the implementation.
  We add a debug check on first call to verify the index.

- Memory: Each attention matrix is (1, 32, S, S) float32.
  For S=512: 32*512*512*4 = 32MB per layer. Total for 32 layers = 1GB.
  We MUST .detach().cpu() immediately and NOT hold all layers on GPU.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class AttentionProbe:
    """
    Captures attention weight matrices from all layers during inference.

    Usage:
        probe = AttentionProbe(model)
        probe.enable()
        model(input_ids, output_attentions=True)
        snapshots = probe.get_snapshots()
        probe.disable()
    """

    def __init__(self, model: Any):
        """
        Args:
            model: A loaded LlamaForCausalLM model.
        """
        self._model = model
        self._hooks: list[Any] = []  # torch hook handles
        self._snapshots: dict[int, np.ndarray] = {}  # layer_idx -> weights
        self._enabled = False
        self._verified_output_index = False
        self._attn_weight_index = 1  # Default: (output, attn_weights, kv)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        """Register forward hooks on all attention layers."""
        if self._enabled:
            return

        # Verify model structure
        # Llama model path: model.model.layers[i].self_attn
        try:
            layers = self._model.model.layers
        except AttributeError:
            raise RuntimeError(
                "Model does not have .model.layers attribute. "
                "Expected LlamaForCausalLM or similar architecture."
            )

        for layer_idx, layer in enumerate(layers):
            try:
                attn_module = layer.self_attn
            except AttributeError:
                logger.warning(
                    "Layer %d has no self_attn module, skipping", layer_idx
                )
                continue

            hook = attn_module.register_forward_hook(
                self._make_hook(layer_idx)
            )
            self._hooks.append(hook)

        self._enabled = True
        logger.info(
            "AttentionProbe enabled: %d hooks registered", len(self._hooks)
        )

    def disable(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._enabled = False
        self._snapshots.clear()
        logger.info("AttentionProbe disabled")

    def clear_snapshots(self) -> None:
        """Clear captured attention snapshots without removing hooks."""
        self._snapshots.clear()

    def get_snapshots(self) -> dict[int, np.ndarray]:
        """
        Get captured attention weight snapshots.

        Returns:
            Dict mapping layer_idx -> numpy array of shape
            (num_heads, seq_len, seq_len).
        """
        return dict(self._snapshots)

    def get_layer_snapshot(self, layer_idx: int) -> Optional[np.ndarray]:
        """Get attention weights for a specific layer."""
        return self._snapshots.get(layer_idx)

    def _make_hook(self, layer_idx: int):
        """
        Create a forward hook for a specific layer.

        The hook captures attention weights from the module output,
        immediately detaches them from the computation graph,
        and moves them to CPU to prevent GPU memory accumulation.
        """

        def hook(module, input, output):
            if not self._enabled:
                return

            # Output is a tuple. For LlamaAttention with output_attentions=True:
            # (attn_output, attn_weights, past_key_value)
            # attn_weights is at index 1
            #
            # First call: verify the output structure
            if not self._verified_output_index:
                self._verify_output_structure(output, layer_idx)

            try:
                attn_weights = output[self._attn_weight_index]

                if attn_weights is None:
                    # output_attentions=True was not set
                    if layer_idx == 0:
                        logger.warning(
                            "Attention weights are None. Ensure "
                            "output_attentions=True is passed to model.forward()"
                        )
                    return

                # Immediately detach and move to CPU
                # Shape: (batch, num_heads, seq_len, seq_len)
                weights_cpu = attn_weights.detach().cpu().float().numpy()

                # Remove batch dimension (we always use batch=1)
                # Result shape: (num_heads, seq_len, seq_len)
                if weights_cpu.ndim == 4:
                    weights_cpu = weights_cpu[0]

                self._snapshots[layer_idx] = weights_cpu

            except (IndexError, TypeError) as e:
                if layer_idx == 0:
                    logger.warning(
                        "Failed to extract attention weights from layer %d: %s",
                        layer_idx,
                        e,
                    )

        return hook

    def _verify_output_structure(self, output, layer_idx: int) -> None:
        """
        Verify the output tuple structure on first hook call.

        Different model implementations may place attention weights
        at different indices. This auto-detects the correct index.
        """
        import torch

        self._verified_output_index = True

        if not isinstance(output, tuple):
            logger.warning(
                "Layer %d output is not a tuple (type: %s). "
                "Attention extraction may not work.",
                layer_idx,
                type(output).__name__,
            )
            return

        logger.info(
            "Layer %d output tuple length: %d, types: %s",
            layer_idx,
            len(output),
            [type(x).__name__ if x is not None else "None" for x in output],
        )

        # Find the attention weights tensor
        # It should be a tensor of shape (batch, heads, seq, seq)
        for idx, item in enumerate(output):
            if item is None:
                continue
            if isinstance(item, torch.Tensor) and item.ndim == 4:
                expected_heads = 32  # Llama 3.1 8B query heads
                if item.shape[1] == expected_heads:
                    if idx != self._attn_weight_index:
                        logger.info(
                            "Attention weights found at index %d "
                            "(expected %d), adjusting",
                            idx,
                            self._attn_weight_index,
                        )
                        self._attn_weight_index = idx
                    else:
                        logger.info(
                            "Confirmed: attention weights at index %d, "
                            "shape %s",
                            idx,
                            list(item.shape),
                        )
                    return

        logger.warning(
            "Could not find attention weight tensor with expected shape. "
            "Output items: %s",
            [
                (type(x).__name__, list(x.shape) if hasattr(x, "shape") else "N/A")
                for x in output
            ],
        )
