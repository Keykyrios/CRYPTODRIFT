"""
Llama 3.1 8B model loader with 4-bit quantization.

Verified configuration:
- Model: meta-llama/Llama-3.1-8B-Instruct
- Quantization: BitsAndBytesConfig(load_in_4bit=True)
- bnb_4bit_compute_dtype: torch.bfloat16 (for RTX 4050 support)
- bnb_4bit_quant_type: "nf4" (Normal Float 4, recommended)
- attn_implementation: "eager" (REQUIRED for attention weight extraction)
  - flash_attention_2 does NOT return attention weights
  - sdpa may not return them either
- device_map: "auto"

Architecture (Llama 3.1 8B):
- 32 transformer decoder layers
- 32 query heads, 8 KV heads (GQA ratio 4:1)
- Hidden dim: 4096
- FFN dim: 14336
- Vocab: 128000
- RoPE with base 500000

VRAM budget (RTX 4050 6GB GDDR6):
- Model weights (4-bit): ~5.0 GB
- KV cache + attention capture: ~0.5 GB
- Total: ~5.5 GB (0.5 GB headroom)
"""

from __future__ import annotations

import gc
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Model constants — verified against Llama 3.1 8B config
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
NUM_LAYERS = 32
NUM_QUERY_HEADS = 32
NUM_KV_HEADS = 8
HIDDEN_DIM = 4096
MAX_SEQ_LEN = 2048  # Conservative limit for 6GB VRAM

# Lazy imports to avoid loading torch at module import time
_model = None
_tokenizer = None


def get_model_and_tokenizer(
    model_id: str = MODEL_ID,
    max_memory: Optional[dict] = None,
):
    """
    Load (or return cached) model and tokenizer.

    Uses singleton pattern — model is only loaded once.
    """
    global _model, _tokenizer

    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    logger.info("Loading model: %s", model_id)
    logger.info("This may take 1-2 minutes on first load...")

    # 4-bit quantization config — verified against HuggingFace docs
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,  # Extra memory savings
    )

    if max_memory is None:
        max_memory = {0: "5.5GiB"}

    # Load model with eager attention (REQUIRED for attention weights)
    _model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory=max_memory,
        attn_implementation="eager",  # NOT flash_attention_2!
        torch_dtype=torch.bfloat16,
    )

    # Load tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Ensure pad token is set (Llama uses EOS as pad)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token
        _model.config.pad_token_id = _tokenizer.eos_token_id

    logger.info(
        "Model loaded: %d layers, %d query heads, %d KV heads",
        NUM_LAYERS,
        NUM_QUERY_HEADS,
        NUM_KV_HEADS,
    )

    return _model, _tokenizer


def unload_model() -> None:
    """Unload the model to free VRAM."""
    global _model, _tokenizer
    import torch

    if _model is not None:
        del _model
        _model = None
    if _tokenizer is not None:
        del _tokenizer
        _tokenizer = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Model unloaded, VRAM freed")


def get_vram_usage() -> dict:
    """Get current VRAM usage statistics."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"available": False}
        return {
            "available": True,
            "allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
            "reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
            "max_allocated_mb": (
                torch.cuda.max_memory_allocated() / 1024 / 1024
            ),
            "total_mb": (
                torch.cuda.get_device_properties(0).total_mem / 1024 / 1024
            ),
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


def generate_refinement(
    code: str,
    strategy_prompt: str,
    conversation_history: Optional[list[dict]] = None,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
) -> tuple[str, int]:
    """
    Generate a code refinement using the loaded model.

    Args:
        code: Current code to refine.
        strategy_prompt: The refinement instruction.
        conversation_history: Previous turns for multi-turn context.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.

    Returns:
        (generated_text, total_tokens_in_context)
    """
    import torch

    model, tokenizer = get_model_and_tokenizer()

    # Build the chat messages
    messages = []
    if conversation_history:
        messages.extend(conversation_history)

    messages.append({
        "role": "user",
        "content": (
            f"{strategy_prompt}\n\n"
            f"Here is the code:\n```python\n{code}\n```\n\n"
            "Return ONLY the improved Python code, wrapped in ```python blocks."
        ),
    })

    # Apply chat template
    # Verified: Llama 3.1 Instruct uses apply_chat_template
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LEN,
    )
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    total_input_tokens = input_ids.shape[1]

    # Generate with attention output
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the new tokens
    new_tokens = outputs[0][input_ids.shape[1]:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return generated_text, total_input_tokens


def extract_code_from_response(response: str) -> str:
    """
    Extract Python code from a model response.

    Handles ```python ... ``` blocks and bare code.
    """
    # Try to extract from markdown code blocks
    import re

    # Match ```python ... ``` or ``` ... ```
    pattern = r"```(?:python)?\s*\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)

    if matches:
        # Return the longest match (most likely the full code)
        return max(matches, key=len).strip()

    # If no code blocks, return the whole response stripped
    return response.strip()
