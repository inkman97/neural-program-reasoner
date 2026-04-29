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

## Two Modes

### Part I: Linguistic Analogy (Google Analogy Dataset)
NPR discovers rules for word relationships and composes them for multi-relation reasoning. **98.8% on compositional tasks.**

### Part II: World Model (Physical Rule Discovery)
NPR implements LeCun's three-module architecture (Perceiver, World Model, Reasoner) to discover physical rules, simulate state transitions, and plan actions — all in latent space. **100% single-step accuracy on 13 actions.**

---

## Key Results

### Part I: NPR vs GPT-2 on Google Analogy

**NPR wins on 8/12 relations.** Overall: NPR 36.2% vs GPT-2 28.0%.

Compositional tasks (chaining two relations):
```
    capital + currency:      100% (top3: 100%)  K=6
    capital + nationality:   100% (top3: 100%)  K=6
    participle + past:        96% (top3: 100%)  K=6
    OVERALL:                98.8% (top3: 100%)
```

### Part II: World Model

**100% single-step accuracy — all 13 actions at 100%**

| Metric | Result |
|--------|--------|
| Single-step (13/13 actions) | **100%** |
| Top-3 accuracy | **100%** |
| Depth-2 compositional chains | **64%** |
| Depth-3 compositional chains | **95%** |
| Chain effects (push fragile → breaks) | **100%** |
| Graded temperature (cold→warm→hot→boiling) | **100%** |
| Latent similarity (cosine to target) | **0.93** |
| Planning: first action correct (real) | **62%** |
| Planning: full plan correct (real) | **54%** |

The model learns all compositional structure **autonomously** — no hardcoded inverse pairs or architectural hacks.

---

## Architecture

### Part II: World Model (LeCun's Three Modules)

```
┌─────────────────────────────────────────────────────┐
│  MODULE 1 — PERCEIVER                                │
│  GPT-2 frozen → ObjectExtractor → SlotAttention      │
│  Discovers: object identity + property decomposition │
└────────┬────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  MODULE 2 — WORLD MODEL                              │
│  RuleSynthesizer (state-conditioned)                 │
│  PropertyUpdater (FiLM + Householder NEGATE)         │
│  LatentPredictor → 768-dim state vector              │
│  VocabDecoder (pred + obj + action + sig → vocab)    │
│  Per-step program synthesis for multi-step chains    │
└────────┬────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  MODULE 3 — REASONER                                 │
│  GoalEvaluator (contrastive state similarity)        │
│  ActionScorer (supervised action prediction)         │
│  Planning in latent space with repeat penalty        │
└─────────────────────────────────────────────────────┘
```

### Key Innovations

- **Householder NEGATE**: H(x) = x - 2(v·x)/(v·v)·v — structurally involutive by construction
- **Program Universality**: All 13 actions converge to same 2-step program, differentiated by FiLM conditioning
- **Cyclic Self-Improvement**: Frequent primitive pairs compressed into higher-level operations
- **Autonomous Learning**: Discovers inverse relationships from generic multi-step training — no hardcoded pairs

---

## Quick Start

```bash
pip install torch transformers

# Part I: Linguistic Analogy
python npr_scaled_fast.py

# Part II: World Model
python npr_world_model.py
```

### Expected Runtime

| Platform | Part I | Part II |
|----------|--------|---------|
| NVIDIA T4 (Kaggle/Colab) | ~30 min | ~40 min |
| CPU | ~4 hours | ~5 hours |

## Files

| File | Description |
|------|-------------|
| `npr_scaled_fast.py` | **Part I** — Google Analogy + compositional tasks + probing |
| `npr_world_model.py` | **Part II** — World Model with 13 actions, chain effects, planning |
| `NPR_Paper.pdf` | Academic paper (Part I + Part II) |
| `README.md` | This file |

---

## Citation

```bibtex
@article{schino2026rules,
  title={Rules Are All You Need: Learning Interpretable Neural Programs 
         from Few Examples via Primitive Composition and 
         Cyclic Self-Improvement},
  author={Schino, Alessandro},
  year={2026},
  url={https://zenodo.org/records/19862967}
}
```

## License

MIT
