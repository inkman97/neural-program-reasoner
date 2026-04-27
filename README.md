# Neural Program Reasoner (NPR)

### *Rules Are All You Need*

**Learning Interpretable Neural Programs from Few Examples via Primitive Composition and Cyclic Self-Improvement**

*Alessandro Schino — April 2026*

---

## What is this?

NPR is an architecture that **discovers interpretable programs** from a few examples. Instead of encoding rules implicitly in billions of parameters like a Transformer, NPR builds explicit programs from composable neural primitives that you can **read**, **modify**, and **reuse**.

```
Input: 5 examples of a relationship (e.g., "hot→cold", "big→small", ...)
  ↓
NPR discovers: IDENTITY → MORPH  (a 2-step program)
  ↓
Applies to new input: "loud" → "quiet" ✓
```

## Key Results

### NPR vs GPT-2 Few-Shot on Google Analogy Benchmark

```
    Relation               |   NPR    t3 |  GPT2    t3
    -----------------------+-------------+------------
    capital-world          |    2%   10% |    0%    4%  [NPR]
    city-in-state          |   34%   56% |    0%   12%  [NPR]
    currency               |   54%   64% |    8%    8%  [NPR]
    gram1-adjective-to-adv |   46%   68% |    8%   46%  [NPR]
    gram2-opposite         |   58%   96% |    0%   28%  [NPR]
    gram3-comparative      |   50%   68% |   68%   98%  [GPT2]
    gram4-superlative      |   64%   84% |   38%   66%  [NPR]
    gram5-present-particip |   78%  100% |   72%   90%  [NPR]
    gram6-nationality-adje |   60%   80% |    6%   38%  [NPR]
    gram7-past-tense       |   54%   74% |   32%   82%  [NPR]
    gram8-plural           |   58%   82% |   16%   72%  [NPR]
    gram9-plural-verbs     |   70%   88% |   68%   90%  [NPR]

    NPR  OVERALL: 52.3% (top3: 72.5%)
    GPT2 OVERALL: 26.3% (top3: 52.8%)
```

**NPR wins on 10/12 relations**, nearly doubling GPT-2 overall accuracy.

### Cyclic Self-Improvement

| Stage | Primitives | Overall | Top-3 |
|-------|-----------|---------|-------|
| Cycle 1 (base) | 6 | 16.0% | 28.3% |
| Cycle 2 (+2 invented) | 8 | 40.5% | 62.5% |
| Cycle 3 (+2 invented) | 10 | 54.3% | 70.5% |
| Final test | 10 | 52.3% | 72.5% |

### Discovered Programs

```
    capital-world         : ASSOCIATE -> NEGATE              (K=2)
    city-in-state         : MORPH -> BLEND                   (K=2)
    currency              : ASSOCIATE -> NEGATE              (K=2)
    gram1-adjective-to-adv: ASSOCIATE -> NEGATE              (K=2)
    gram2-opposite        : IDENTITY -> MORPH                (K=2)
    gram3-comparative     : LOOKUP -> BLEND                  (K=2)
    gram4-superlative     : NEGATE -> BLEND                  (K=2)
    gram5-present-particip: ASSOCIATE -> LOOKUP_NEGATE       (K=2)  ← invented primitive!
    gram6-nationality-adje: LOOKUP -> BLEND                  (K=2)
    gram7-past-tense      : ASSOCIATE -> NEGATE              (K=2)
    gram8-plural          : BLEND -> MORPH                   (K=2)
    gram9-plural-verbs    : ASSOCIATE -> NEGATE              (K=2)
```

### Primitive Usage

```
    BLEND               :  23.7%
    NEGATE              :  22.5%
    ASSOCIATE           :  22.2%
    MORPH               :  14.5%
    LOOKUP              :  11.3%
    IDENTITY            :   4.2%
    NEGATE_MORPH        :   1.2%   ← invented
    LOOKUP_NEGATE       :   0.5%   ← invented
```

### Invented Primitives

```
    Cycle 1: MORPH_IDENTITY   (= MORPH → IDENTITY, seen 97x)
             LOOKUP_NEGATE    (= LOOKUP → NEGATE, seen 83x)
    Cycle 2: ASSOCIATE_NEGATE (= ASSOCIATE → NEGATE, seen 208x)
             NEGATE_MORPH     (= NEGATE → MORPH, seen 126x)
```

---

## Architecture

```
Input Text
    │
    ▼
┌───────────────────┐
│  PERCEIVER        │  GPT-2 Small frozen (layer 8)
│  + EmbeddingCache │  Pre-computed representations (~10s startup)
└────────┬──────────┘
         │
         ▼
┌────────────────────────────────────────────────┐
│  PROGRAM SYNTHESIZER                            │
│  Examples → 2x Self-Attention → Signature       │
│  Signature → Per-step MLP → Gumbel-Softmax     │
│  + Stop Predictor (variable K=2..6)             │
└────────┬───────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────┐
│  PRIMITIVE LIBRARY                              │
│  IDENTITY │ NEGATE │ MORPH │ ASSOCIATE          │
│  LOOKUP   │ BLEND  │ + invented primitives      │
└────────┬───────────────────────────────────────┘
         │
         ▼
┌───────────────────┐
│  EXECUTOR         │  Sequential primitive application
│  + step embeddings│  with learnable positional encoding
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  GENERATOR        │  (transformed + original + signature) → vocab
└───────────────────┘
```

## Three Key Mechanisms

### 1. Gumbel-Softmax Program Synthesis
Near-discrete but differentiable primitive selection. Temperature τ=0.8 (fixed). Hard argmax at inference.

### 2. Program Memory
Non-parametric cache indexed by signature vectors (cosine similarity > 0.85). Capacity: 500 entries with LRU eviction.

### 3. Cyclic Self-Improvement
```
Cycle 1: Train with 6 base primitives → Memorize programs
         → Find frequent pairs (MORPH→IDENTITY: 97x)
         → Create MORPH_IDENTITY primitive → Library grows to 8

Cycle 2: Retrain with 8 primitives → Memorize
         → Find frequent pairs (ASSOCIATE→NEGATE: 208x)  
         → Create ASSOCIATE_NEGATE primitive → Library grows to 10

Cycle 3: Final training with 10 primitives
```

---

## Requirements

```
Python >= 3.8
PyTorch >= 2.0
transformers >= 4.30
requests
```

## Quick Start

```bash
pip install torch transformers requests
python npr_scaled_fast.py
```

GPT-2 Small and the Google Analogy dataset are downloaded automatically on first run.

### Expected Runtime

| Platform | Cache | Training (3 cycles) | Total |
|----------|-------|-------------------|-------|
| NVIDIA T4 (Kaggle/Colab) | ~10s | ~30 min | ~30 min |
| CPU | ~10s | ~4 hours | ~4 hours |

## Files

| File | Description |
|------|-------------|
| `npr_scaled_fast.py` | **Main code** — Google Analogy, EmbeddingCache, GPT-2 baseline |
| `neural_program_reasoner.py` | Prototype — controlled 5-relation benchmark |
| `NPR_Paper.pdf` | Academic paper |
| `README.md` | This file |

## Configuration

All hyperparameters in the `CONFIG` dictionary:

```python
CONFIG = {
    "proj_dim": 256,            # Signature dimension
    "max_program_steps": 6,     # Maximum steps per program
    "min_program_steps": 2,     # Minimum before stop predictor activates
    "temperature": 0.8,         # Gumbel-Softmax temperature
    "grad_accumulation": 16,    # Simulated batch size
    "num_examples": 5,          # Examples per task
    "memory_capacity": 500,     # Max cached programs
    "compression_threshold": 8, # Min frequency to compress a pair
    "max_new_primitives": 2,    # Max new primitives per cycle
    "num_cycles": 3,            # Self-improvement cycles
    "iters_per_cycle": [4000, 3000, 2000],
    "lr_per_cycle": [5e-4, 3e-4, 1e-4],
    "perceiver_layer": 8,       # GPT-2 layer for extraction
}
```

## Loss Functions

Five loss terms work together:

| Loss | Weight | Purpose |
|------|--------|---------|
| Cross-entropy | 1.0 | Primary prediction signal |
| Diversity | 0.1 | Different primitives at different steps |
| Usage | 0.05 | Use all available primitives |
| Novelty | 0.1 | Use invented primitives (≥15% attention) |
| Stop | 0.03 | Encourage earlier termination |
| Length penalty | 0.02/step | Prefer shorter programs |

## Extending

### Adding relations
Add entries to `RELATIONS` dict or use a different dataset file.

### Adding base primitives
1. Add name to `BASE_NAMES`
2. Implement network in `PrimitiveLibrary.__init__`
3. Add case in `_base()` method
4. Add gate value

### Using a larger Perceiver
Replace GPT-2 with any HuggingFace model supporting `output_hidden_states=True`. Update `EmbeddingCache` and `CONFIG["perceiver_layer"]`.

---

## Citation

```bibtex
@article{schino2026rules,
  title={Rules Are All You Need: Learning Interpretable Neural Programs 
         from Few Examples via Primitive Composition and 
         Cyclic Self-Improvement},
  author={Schino, Alessandro},
  year={2026}
}
```

## License

MIT
