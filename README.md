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

It also handles **compositional reasoning** — chaining two relations together:

```
Input: "rome→italian", "tokyo→japanese", "moscow→russian"
  ↓
NPR discovers: LOOKUP → MORPH  (capital → country → nationality)
  ↓
Applies to: "bangkok" → "thai" ✓
```

## Key Results

### NPR vs GPT-2 Few-Shot on Google Analogy

```
    Relation               |   NPR    t3 |  GPT2    t3
    -----------------------+-------------+------------
    capital-world          |    4%    6% |    0%    4%  [NPR]
    city-in-state          |   24%   32% |    0%    8%  [NPR]
    currency               |   48%   68% |   10%   10%  [NPR]
    gram1-adjective-to-adv |   44%   76% |    8%   46%  [NPR]
    gram2-opposite         |   44%   44% |    0%   34%  [NPR]
    gram3-comparative      |   16%   42% |   64%   90%  [GPT2]
    gram4-superlative      |   32%   58% |   34%   56%  [TIE]
    gram5-present-particip |   54%   68% |   80%   98%  [GPT2]
    gram6-nationality-adje |   32%   54% |    6%   36%  [NPR]
    gram7-past-tense       |   68%   84% |   36%   80%  [NPR]
    gram8-plural           |   38%   48% |   26%   92%  [NPR]
    gram9-plural-verbs     |   30%   60% |   72%   94%  [GPT2]

    NPR  OVERALL: 36.2% (top3: 53.3%)
    GPT2 OVERALL: 28.0% (top3: 54.0%)
```

**NPR wins on 8/12 relations.** GPT-2 wins only on morphologically regular relations (comparative, present participle, plural verbs) where surface pattern matching suffices.

### Compositional Tasks (the strongest result)

```
    capital-world + currency:              89% (top3:  94%)
    capital-world + nationality:           79% (top3:  86%)
    present-participle + past-tense:       96% (top3: 100%)

    COMPOSITIONAL OVERALL: 86.5% (top3: 93.4%)
```

NPR composes two different relations in a single pass. Examples:
- `bucharest → romania → leu` (capital + currency) ✓
- `tokyo → japan → japanese` (capital + nationality) ✓
- `sit → sitting → sat` (participle + past tense, irregular) ✓

### Cyclic Self-Improvement

| Stage | Primitives | Standard | Compositional |
|-------|-----------|----------|---------------|
| Cycle 1 (base) | 6 | 14.0% | 27.1% |
| Cycle 2 (+2 invented) | 8 | 29.2% | 62.1% |
| Cycle 3 (+2 invented) | 10 | 39.3% | 87.2% |
| Final test | 10 | 36.2% | 86.5% |

### Primitive Probing Results

Empirical verification that primitives do what their names suggest:

| Test | Result | Verdict |
|------|--------|---------|
| NEGATE involution: ‖NEGATE(NEGATE(x))−x‖/‖x‖ | **0.17** | ✅ Near-involution |
| MORPH consistency: cosine similarity of deltas | **0.97** | ✅ Highly consistent |
| IDENTITY passthrough: ‖IDENTITY(x)−x‖/‖x‖ | **0.11** | ✅ Near-passthrough |
| LOOKUP directional (country→capital) | **0/20** | ❌ Not factual retrieval |
| Primitive distinctness: max off-diagonal cosine | **0.36** | ✅ All distinct |

Key finding: **MORPH applies nearly identical transformations** to different words of the same relation (cosine 0.97). This proves it learned a general morphological operation, not word-specific memorization.

### Discovered Programs

```
    capital-world         : NEGATE -> NEGATE_IDENTITY        (K=2)
    city-in-state         : LOOKUP -> BLEND                  (K=2)
    currency              : ASSOCIATE -> NEGATE              (K=2)
    gram1-adjective-to-adv: ASSOCIATE -> NEGATE              (K=2)
    gram2-opposite        : LOOKUP -> MORPH                  (K=2)
    gram3-comparative     : NEGATE -> MORPH                  (K=2)
    gram4-superlative     : NEGATE -> MORPH                  (K=2)
    gram5-present-particip: LOOKUP -> ASSOCIATE              (K=2)
    gram6-nationality-adje: LOOKUP -> MORPH                  (K=2)
    gram7-past-tense      : ASSOCIATE -> IDENTITY            (K=2)
    gram8-plural          : NEGATE -> ASSOCIATE              (K=2)
    gram9-plural-verbs    : ASSOCIATE -> NEGATE              (K=2)
```

Programs cluster into families:
- **Semantic relations** (currency, adverb, past-tense, plural-verbs): `ASSOCIATE → NEGATE`
- **Morphological relations** (comparative, superlative, nationality): `* → MORPH`

### Invented Primitives

```
    Cycle 1: MORPH_IDENTITY   (= MORPH → IDENTITY, seen 95x)
             NEGATE_IDENTITY  (= NEGATE → IDENTITY, seen 87x)
    Cycle 2: NEGATE_MORPH     (= NEGATE → MORPH, seen 223x)
             LOOKUP_BLEND     (= LOOKUP → BLEND, seen 158x)
```

`NEGATE_IDENTITY` appears in the final program for capital-world.

### Primitive Usage

```
    NEGATE              :  25.5%
    ASSOCIATE           :  22.3%
    MORPH               :  20.3%
    LOOKUP              :  19.8%
    BLEND               :   7.2%
    IDENTITY            :   3.2%
    NEGATE_IDENTITY     :   1.7%
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
         → Find frequent pairs → Create 2 new primitives → Library: 8

Cycle 2: Retrain with 8 primitives → Memorize
         → Find frequent pairs → Create 2 new → Library: 10

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

GPT-2 and the Google Analogy dataset are downloaded automatically.

### Expected Runtime

| Platform | Total |
|----------|-------|
| NVIDIA T4 (Kaggle/Colab) | ~30 min |
| CPU | ~4 hours |

## Files

| File | Description |
|------|-------------|
| `npr_scaled_fast.py` | **Main code** — Google Analogy + compositional tasks + probing + GPT-2 baseline |
| `neural_program_reasoner.py` | Prototype — controlled 5-relation benchmark (69.3% accuracy) |
| `NPR_Paper.pdf` | Academic paper |
| `README.md` | This file |

## Configuration

```python
CONFIG = {
    "proj_dim": 256,
    "max_program_steps": 6,
    "min_program_steps": 2,
    "temperature": 0.8,
    "grad_accumulation": 16,
    "num_examples": 5,
    "memory_capacity": 500,
    "compression_threshold": 8,
    "max_new_primitives": 2,
    "num_cycles": 3,
    "iters_per_cycle": [4000, 3000, 2000],
    "lr_per_cycle": [5e-4, 3e-4, 1e-4],
    "perceiver_layer": 8,
    "compositional_ratio": 0.3,    # 30% of training tasks are compositional
}
```

## Loss Functions

| Loss | Weight | Purpose |
|------|--------|---------|
| Cross-entropy | 1.0 | Primary prediction signal |
| Diversity | 0.1 | Different primitives at different steps |
| Usage | 0.05 | Use all available primitives |
| Novelty | 0.1 | Use invented primitives (≥15% attention) |
| Stop | 0.03 | Encourage earlier termination |
| Length penalty | 0.02/step | Prefer shorter programs |

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
