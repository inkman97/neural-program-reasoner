# Neural Program Reasoner (NPR)

**Learning Interpretable Rule Programs from Few Examples via Neural Primitive Composition and Cyclic Self-Improvement**

---

## Overview

The Neural Program Reasoner (NPR) is a novel architecture that bridges neural network flexibility and symbolic reasoning interpretability. Given a small set of input-output examples demonstrating a linguistic relationship, NPR:

1. **Perceives** the input through a frozen pre-trained language model (GPT-2)
2. **Synthesizes** an explicit program composed of learnable neural primitives
3. **Executes** the program on new inputs to produce predictions
4. **Memorizes** discovered programs for efficient reuse
5. **Self-improves** by compressing frequent primitive sequences into new higher-level operations

Unlike standard Transformer models that encode knowledge implicitly in billions of parameters, NPR discovers interpretable rule programs that can be **read**, **modified**, **transferred**, and **composed**.

## Key Results

| Metric | Cycle 1 (6 primitives) | Cycle 2 (8 primitives) | Cycle 3 (10 primitives) |
|--------|----------------------|----------------------|------------------------|
| Overall Accuracy | 16.7% | 46.7% | **69.3%** |
| Top-3 Accuracy | 38.0% | 62.7% | **87.3%** |

The model discovers **distinct, consistent programs** for each relation type:

```
opposite:    LOOKUP -> MORPH -> NEGATE
capital:     IDENTITY -> ASSOCIATE -> LOOKUP
plural:      MORPH -> ASSOCIATE -> NEGATE
past_tense:  NEGATE -> IDENTITY -> ASSOCIATE
comparative: LOOKUP -> BLEND -> IDENTITY
```

## Architecture

```
Input Text
    │
    ▼
┌──────────────────┐
│   PERCEIVER      │  GPT-2 Small (frozen, layer 8)
│   Text → R^768   │  Transforms text into semantic vectors
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│   PROGRAM SYNTHESIZER                         │
│                                               │
│   Examples → Self-Attention → Signature       │
│   Signature → Per-step MLP → Gumbel-Softmax  │
│   Output: sequence of primitive selections    │
└────────┬─────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│   PRIMITIVE LIBRARY                           │
│                                               │
│   IDENTITY  │ NEGATE  │ MORPH                 │
│   ASSOCIATE │ LOOKUP  │ BLEND                 │
│   + dynamically invented compressed primitives│
└────────┬─────────────────────────────────────┘
         │
         ▼
┌──────────────────┐
│   EXECUTOR       │  Applies selected primitives sequentially
│   + step embed   │  with learnable step embeddings
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   GENERATOR      │  Maps (transformed + original + signature)
│   → vocabulary   │  to prediction over restricted vocabulary
└──────────────────┘
```

## Three Key Mechanisms

### 1. Differentiable Program Synthesis (Gumbel-Softmax)

The Program Synthesizer selects primitives using Gumbel-Softmax, which produces near-discrete selections during training while remaining differentiable for gradient-based optimization. At inference time, hard argmax produces fully discrete, interpretable programs.

### 2. Program Memory

Discovered programs are cached in a non-parametric memory indexed by learned signature vectors. When new examples arrive, the system searches for similar signatures (cosine similarity > 0.85) and reuses cached programs, improving both speed and consistency.

### 3. Cyclic Self-Improvement

After each training cycle:
- **Analyze**: scan memorized programs for frequently co-occurring primitive pairs
- **Compress**: create new primitives that approximate the sequential application of frequent pairs
- **Expand**: add new primitives to the library and rebuild the Synthesizer's selection heads
- **Retrain**: train again with the expanded primitive set

This cycle enables automatic abstraction formation — the model builds increasingly powerful operations from simpler building blocks.

## Requirements

```
Python >= 3.8
PyTorch >= 2.0
transformers >= 4.30
```

## Installation

```bash
pip install torch transformers
```

## Usage

### Quick Start

```bash
python neural_program_reasoner.py
```

This runs the full training pipeline: 3 self-improvement cycles with evaluation after each cycle, followed by a final test. GPT-2 Small is downloaded automatically on first run (~500MB).

### Configuration

All hyperparameters are in the `CONFIG` dictionary at the top of the file:

```python
CONFIG = {
    "proj_dim": 256,            # Internal projection dimension
    "num_program_steps": 3,     # Steps per program
    "temperature": 0.8,         # Gumbel-Softmax temperature
    "grad_accumulation": 16,    # Simulated batch size
    "num_examples": 5,          # Examples per task
    "memory_capacity": 200,     # Max programs in memory
    "compression_threshold": 5, # Min frequency to compress
    "max_new_primitives": 2,    # Max new primitives per cycle
    "num_cycles": 3,            # Self-improvement cycles
    "iters_per_cycle": [2000, 1500, 1000],
    "lr_per_cycle": [5e-4, 3e-4, 1e-4],
    "perceiver_layer": 8,       # GPT-2 layer for representations
}
```

### Expected Output

The training produces output like:

```
CYCLE 1/3 | Primitives: 6 | Params: 18,746,056
  [C1] It    0 | L:5.4195 | Acc:  0.0% | [plural] 'tooth' -> 'spain' (want:'teeth') ✗
  [C1] It  200 | L:4.6452 | Acc:  3.0% | [opposite] 'loud' -> 'quiet' (want:'quiet') ✓
  ...

--- Compression ---
  [COMPRESS] Frequent: MORPH -> IDENTITY (17x)
  [COMPRESS] Created: MORPH_IDENTITY (= MORPH -> IDENTITY)
  [COMPRESS] Library: 8 primitives
```

### Expected Runtime

| Platform | Time per cycle | Total (3 cycles) |
|----------|---------------|-------------------|
| NVIDIA T4 (Colab/Kaggle) | ~10 min | ~30 min |
| CPU (modern laptop) | ~60 min | ~3 hours |

## Dataset

The benchmark consists of five linguistic relation types:

| Relation | Pairs | Example | Type |
|----------|-------|---------|------|
| opposite | 20 | hot → cold | Semantic transformation |
| capital | 17 | France → Paris | Factual association |
| plural | 20 | mouse → mice | Morphological (irregular) |
| past_tense | 20 | walk → walked | Morphological (regular) |
| comparative | 20 | big → bigger | Morphological (regular) |

The restricted vocabulary contains **175 unique words**. Tasks are generated procedurally with random sampling, so the model never sees the same task twice.

## Primitives

Each primitive is a small neural network with a specific architectural bias:

| Primitive | Architecture | Initial Gate | Purpose |
|-----------|-------------|-------------|---------|
| IDENTITY | Linear → Tanh | 0.01 | Near-passthrough |
| NEGATE | 768→1536→1536→768, GELU | 0.50 | Deep semantic inversion |
| MORPH | 768→768, LayerNorm, GELU | 0.30 | Morphological modification |
| ASSOCIATE | Self-attention (Q,K,V) | 0.40 | Associative lookup |
| LOOKUP | 768→1536→1536→768, GELU | 0.50 | Factual retrieval |
| BLEND | Concat with context → MLP | 0.30 | Context mixing |

All primitives use gated residual connections: `output = state + sigmoid(gate) * delta(state)`.

## Extending

### Adding New Relations

Add entries to the `RELATIONS` dictionary:

```python
RELATIONS["superlative"] = [
    ("big", "biggest"), ("small", "smallest"), ("fast", "fastest"), ...
]
```

Then rebuild the vocabulary and retrain.

### Adding New Base Primitives

1. Add the name to `BASE_PRIMITIVE_NAMES`
2. Implement the network in `PrimitiveLibrary.__init__`
3. Add the case in `PrimitiveLibrary._apply_base`
4. Add a gate value in `self.gates`

### Using a Different Perceiver

Replace `GPT2LMHeadModel` in `main()` with any HuggingFace model that supports `output_hidden_states=True`. Adjust `CONFIG["perceiver_layer"]` accordingly.

## Paper

See `NPR_Paper.pdf` for the full academic paper with detailed methodology, results, analysis, and references.

## License

MIT

## Citation

```bibtex
@article{npr2026,
  title={Neural Program Reasoner: Learning Interpretable Rule Programs 
         from Few Examples via Neural Primitive Composition 
         and Cyclic Self-Improvement},
  author={Alessandro Schino},
  year={2026}
}
```
