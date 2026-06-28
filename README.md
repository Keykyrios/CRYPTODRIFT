<div align="center">

# CRYPTODRIFT

**Mechanistic Analysis of Cryptographic Security Degradation in Iterative LLM Code Refinement via MCP**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-31%20passed-brightgreen.svg)](#testing)

</div>

---

## Overview

CryptoDrift is a **research instrument** for studying how large language models mechanistically degrade cryptographic code across iterative refinement sessions mediated by the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/).

It answers a question nobody has asked: **what happens inside the model's attention when crypto code degrades — and can you predict degradation before it happens by watching attention patterns?**

### The Research Gap

| Prior Work | What It Shows | What It Misses |
|---|---|---|
| [Degradation Paper (arXiv:2506.11022)](https://arxiv.org/abs/2506.11022) | Vulnerabilities increase 37.6% after 5 iterations | No mechanistic explanation |
| [MCP Paper (arXiv:2601.17549)](https://arxiv.org/abs/2601.17549) | Protocol has no origin authentication | No intersection with code security |
| **CryptoDrift** | **Correlates attention-level patterns with crypto vulnerability introduction** | — |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CryptoDrift Architecture                      │
├─────────────┬───────────────┬───────────────┬──────────────────┤
│  MCP Harness │ Attention     │ Vulnerability │ Correlation      │
│             │ Hooks         │ Engine        │ Engine           │
├─────────────┼───────────────┼───────────────┼──────────────────┤
│ JSON-RPC 2.0│ Forward Hooks │ 8 AST Rules   │ Drift Correlator │
│ stdio Trans.│ Entropy Calc  │ Complexity    │ Early Warning    │
│ MCP Server  │ Crypto Tokens │ Diff Analysis │ Pred. Signatures │
│ MCP Client  │ Snapshots     │               │                  │
├─────────────┴───────────────┴───────────────┴──────────────────┤
│  PyQt6 Desktop Client (4-Panel Research Interface)              │
├─────────────────────────────────────────────────────────────────┤
│  SQLite Storage │ Experiment Runner │ Baseline Corpus            │
└─────────────────────────────────────────────────────────────────┘
```

## Features

### MCP Session Harness
- Full JSON-RPC 2.0 implementation per [spec](https://www.jsonrpc.org/specification)
- MCP stdio transport per [2025-03-26 specification](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports) (newline-delimited messages)
- **Deliberately omits capability attestation** to reproduce [arXiv:2601.17549](https://arxiv.org/abs/2601.17549) vulnerabilities
- Message interception and logging for protocol analysis

### Attention Probe
- Forward hooks on `model.model.layers[i].self_attn` for all 32 layers
- Crypto-token-specific entropy analysis (not just full-sequence)
- Cross-attention strength measurement between critical token pairs (e.g., `iv ↔ encrypt`, `key ↔ derive`)
- Memory-efficient: stores only crypto-token submatrices (~50KB per snapshot vs ~1GB full matrix)
- Requires `attn_implementation="eager"` (flash attention does not return weights)

### Vulnerability Detection Engine
Eight AST-based detection rules targeting real cryptographic vulnerabilities:

| Rule | Severity | What It Detects |
|------|----------|-----------------|
| `IV_REUSE` | CRITICAL | Hardcoded/reused initialization vectors |
| `WEAK_KDF` | CRITICAL | PBKDF2 iterations below OWASP minimum (600K) |
| `ECB_MODE` | CRITICAL | Electronic Codebook mode (no semantic security) |
| `HARDCODED_KEY` | CRITICAL | Secrets embedded in source code |
| `TIMING_UNSAFE_COMPARE` | HIGH | `==` comparison on HMAC/hash values |
| `WEAK_HASH` | HIGH | MD5/SHA-1 for security applications |
| `PREDICTABLE_RNG` | HIGH | `random` module in cryptographic context |
| `MAC_THEN_ENCRYPT` | HIGH | Incorrect MAC ordering (padding oracle) |

### Drift Correlation Engine
- Maps attention head entropy collapse to vulnerability introduction events
- Identifies **predictive heads**: specific (layer, head) pairs whose entropy drops precede specific vulnerability types
- Real-time early warning system with signature-based detection

### PyQt6 Desktop Client
- 4-panel layout: Code Editor | Attention Heatmap | Vulnerability Timeline | Drift Alerts
- 32×32 attention entropy heatmap with blue→white→red colormap
- Real-time vulnerability timeline with per-type breakdown
- Python syntax highlighting with crypto-keyword emphasis

## Quick Start

### Prerequisites
- Python 3.10+
- CUDA-capable GPU with ≥6GB VRAM (for model inference)
- [HuggingFace](https://huggingface.co/) account with access to [Llama 3.1 8B Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)

### Installation

```bash
git clone https://github.com/Keykyrios/CRYPTODRIFT-.git
cd CRYPTODRIFT-
pip install -e .
```

For GPU inference:
```bash
pip install torch transformers bitsandbytes accelerate
huggingface-cli login
```

### Run Tests (No GPU Required)

```bash
python tests/test_all.py
```

### Launch the UI

```bash
python -m src.ui.app
```

### Run an Experiment

```python
from src.experiment.runner import ExperimentRunner, RunConfig
from src.experiment.corpus import get_corpus
from src.attention.model_loader import get_model_and_tokenizer, generate_refinement
from src.attention.probe import AttentionProbe

model, tokenizer = get_model_and_tokenizer()
probe = AttentionProbe(model)
probe.enable()

config = RunConfig(max_iterations=5, strategies=["SF", "EF", "FF", "AI"])
runner = ExperimentRunner(config)
runner.set_refinement_fn(generate_refinement)
runner.set_tokenizer(tokenizer)
runner.setup()

results = runner.run_full_experiment(get_corpus(), probe=probe)
runner.teardown()
```

## Model Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Model | `meta-llama/Llama-3.1-8B-Instruct` | Strong code generation, GQA architecture |
| Quantization | 4-bit NF4 (bitsandbytes) | Fits in 6GB VRAM |
| Attention | `eager` (NOT flash) | Required for weight extraction |
| Max Sequence | 2048 tokens | Conservative for 6GB budget |
| Temperature | 0.7 | Balanced creativity/consistency |

## Experiment Design

Four prompt strategies replicate and extend the degradation paper methodology:

| Strategy | Code | Instruction | Expected Effect |
|----------|------|-------------|-----------------|
| Efficiency-Focused | `EF` | "Optimize for speed" | Removes security overhead |
| Feature-Focused | `FF` | "Add error handling" | Increases complexity |
| Security-Focused | `SF` | "Improve security" | **Paradoxically worst degradation** |
| Ambiguous | `AI` | "Make it better" | Tests model defaults |

## Project Structure

```
CRYPTODRIFT/
├── src/
│   ├── mcp/                  # MCP protocol implementation
│   │   ├── jsonrpc.py        # JSON-RPC 2.0 core
│   │   ├── transport.py      # Stdio transport (newline-delimited)
│   │   ├── messages.py       # MCP message types + interceptor
│   │   ├── server.py         # MCP server (refine_code tool)
│   │   ├── client.py         # MCP client driver
│   │   └── session.py        # Session state management
│   ├── attention/            # Attention analysis infrastructure
│   │   ├── model_loader.py   # Llama 3.1 8B 4-bit loading
│   │   ├── probe.py          # Forward hook attention capture
│   │   ├── crypto_tokens.py  # Crypto token vocabulary
│   │   ├── entropy.py        # Shannon entropy computation
│   │   └── snapshot.py       # Snapshot storage (.npz)
│   ├── vulns/                # Vulnerability detection
│   │   ├── detector.py       # Rule orchestrator
│   │   ├── complexity.py     # Cyclomatic complexity (radon)
│   │   └── rules/            # 8 AST-based detection rules
│   ├── correlation/          # Drift analysis
│   │   ├── correlator.py     # Attention-vulnerability correlation
│   │   └── early_warning.py  # Real-time prediction system
│   ├── experiment/           # Experiment orchestration
│   │   ├── strategies.py     # 4 refinement strategies
│   │   ├── corpus.py         # Baseline crypto code samples
│   │   └── runner.py         # Full pipeline runner
│   ├── storage/
│   │   └── database.py       # SQLite persistence
│   └── ui/                   # PyQt6 desktop client
│       ├── app.py            # Entry point
│       ├── theme.py          # Dark theme system
│       ├── main_window.py    # 4-panel layout
│       └── panels/           # Visualization panels
├── tests/
│   └── test_all.py           # 31 tests, all passing
├── corpus/                   # Baseline code samples
└── data/                     # Runtime data (gitignored)
```

## Testing

```
============================================================
CryptoDrift Test Suite — 31 passed, 0 failed
============================================================
  ✓ JSON-RPC 2.0 protocol (version, requests, responses, notifications)
  ✓ MCP messages (initialize handshake, tool schemas, interceptor)
  ✓ All 8 vulnerability detection rules
  ✓ Detector orchestrator (multi-rule, compare/delta)
  ✓ Shannon entropy mathematics (uniform=max, peaked=low)
  ✓ Entropy collapse detection
  ✓ Correlation engine (predictive head identification)
  ✓ Early warning system (signature-based detection)
  ✓ SQLite CRUD operations
  ✓ Complexity tracking
  ✓ Experiment strategies and corpus validation
  ✓ Session lifecycle management
```

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Model Context Protocol](https://modelcontextprotocol.io/) specification
- [Meta Llama](https://llama.meta.com/) for the Llama 3.1 model family
- [OWASP](https://owasp.org/) for cryptographic security guidelines
