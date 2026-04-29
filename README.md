# Neural Program Reasoner (NPR)

### *Rules Are All You Need*

**Learning Interpretable Neural Programs from Few Examples via Primitive Composition and Cyclic Self-Improvement**

*Alessandro Schino — April 2026*

---

## What is this?

NPR is an architecture that **discovers rule programs** from a few examples using composable neural primitives. Instead of encoding knowledge implicitly in billions of parameters, NPR builds explicit programs you can **trace**, **modify**, and **reuse**.

The project spans three parts, from linguistic analogy to a visual world model:

```
Part I:   Text → GPT-2 → discovers linguistic rules      → 98.8% compositional
Part II:  Text → GPT-2 → discovers physical rules         → 100% single-step, 95% depth-3
Part III: Image → ViT  → discovers physical rules (JEPA)  → 99.6% single-step, latent space only
```

---

## Three Parts

### Part I: Linguistic Analogy (Google Analogy Dataset)
NPR discovers rules for word relationships and composes them for multi-relation reasoning. **98.8% on compositional tasks.**

### Part II: Text World Model (Physical Rule Discovery)
NPR implements LeCun's three-module architecture (Perceiver, World Model, Reasoner) with GPT-2 as perceiver. Discovers physical rules from text, simulates state transitions, plans actions. **100% single-step accuracy on 13 actions.**

### Part III: JEPA Visual World Model
Replaces GPT-2 with frozen ViT. Learns from synthetic scene images. Predicts entirely in latent space — no vocabulary, no decoder. Model-based planning via World Model simulation. Includes generalization tests on held-out objects, counterfactual visuals, and cross-property transfer. **99.6% single-step, 0.986 latent similarity.**

---

## Key Results

### Part I: NPR vs GPT-2 on Google Analogy

**NPR wins on 8/12 relations.** Compositional tasks:
```
capital + currency:      100%    K=6
capital + nationality:   100%    K=6
participle + past:        96%    K=6
OVERALL:                98.8%    (top3: 100%)
```

### Part II: Text World Model

| Metric | Result |
|--------|--------|
| Single-step (13/13 actions) | **100%** |
| Depth-3 compositional chains | **95%** |
| Chain effects (push fragile → breaks) | **100%** |
| Graded temperature (cold→warm→hot→boiling) | **100%** |
| Latent similarity | **0.93** |
| Planning first action (real) | **62%** |

### Part III: JEPA Visual World Model

| Metric | Result |
|--------|--------|
| Single-step NN (nearest neighbor) | **99.6%** |
| Latent similarity (avg cosine) | **0.986** |
| Depth-2 compositional chains | **77-100%** |
| Depth-3 compositional chains | **78-87%** |
| Planning first action (real) | **57-60%** |

All learned autonomously — no hardcoded inverse pairs, no vocabulary, no architectural hacks.

---

## Architecture

### Part III: JEPA World Model

```
┌───────────────────────────────────────────────────────┐
│  PERCEIVER                                             │
│  ViT (frozen) → 768-dim CLS + 196 patch tokens        │
│  ObjectExtractor → object identity vector              │
│  SlotAttention → property decomposition (2 slots)      │
│  SlotSelector → which property does this action affect? │
└────────┬──────────────────────────────────────────────┘
         ↓
┌───────────────────────────────────────────────────────┐
│  WORLD MODEL                                           │
│  RuleSynthesizer → program from 3-5 examples           │
│  PrimitiveLibrary → 6 base + invented primitives       │
│     IDENTITY, NEGATE (Householder), MORPH, ASSOCIATE,  │
│     LOOKUP, BLEND + cyclic compressed primitives       │
│  PropertyUpdater → FiLM conditioning + program exec    │
│  LatentPredictor → predicted state vector (768-dim)    │
└────────┬──────────────────────────────────────────────┘
         ↓
┌───────────────────────────────────────────────────────┐
│  REASONER                                              │
│  GoalEvaluator → are we at the goal?                   │
│  Model-based planning → simulate all 13 actions,       │
│     pick the one closest to goal, repeat               │
│  No ActionScorer — World Model IS the planner          │
└───────────────────────────────────────────────────────┘
```

### Key Innovations

- **Householder NEGATE**: `H(x) = x - 2(v·x)/(v·v)·v` — structurally involutive by construction (H(H(x)) = x exactly)
- **Model-based planning**: simulates all actions through the real World Model, picks best — no separate action predictor
- **Predicts in latent space**: no vocabulary, no decoder — true JEPA-style prediction
- **Cyclic Self-Improvement**: frequent primitive pairs compressed into new operations
- **Structurally traceable**: every decision (object extraction, slot selection, program, FiLM parameters) is inspectable

### Generalization Tests (Part III)

Three tests address common criticisms of toy-world models:

- **Held-out objects**: 2 objects per property type excluded from training, tested with rules learned from other objects
- **Counterfactual visuals**: images rendered with wrong visual cues (e.g., "hot" rendered with cold colors) to detect renderer shortcuts
- **Cross-property transfer**: actions applied to objects from different property types (e.g., "the door is cold + heat")

---

## How it differs from existing JEPA implementations

| | V-JEPA 2 (Meta) | Causal-JEPA | NPR-JEPA (ours) |
|---|---|---|---|
| Training data | 1M+ hours video | Video sequences | 127 observations |
| Team | 30+ researchers | Research lab | 1 person |
| Compute | Large GPU cluster | Multi-GPU | Single T4, 40 min |
| Interpretability | Opaque predictor | Opaque | Traceable programs |
| Rule discovery | No | No | Yes (3-5 examples) |
| Primitive composition | No | No | Yes + cyclic compression |
| Householder NEGATE | No | No | Yes (exact involution) |
| Planning | MPC with learned model | N/A | Model-based search (13 actions) |

---

## Quick Start

```bash
pip install torch transformers Pillow

# Part I: Linguistic Analogy
python npr_scaled_fast.py

# Part II: Text World Model
python npr_world_model.py

# Part III: JEPA Visual World Model
python npr_jepa_world_model.py
```

### Expected Runtime

| Platform | Part I | Part II | Part III |
|----------|--------|---------|----------|
| NVIDIA T4 (Kaggle/Colab) | ~30 min | ~40 min | ~40 min |
| CPU | ~4 hours | ~5 hours | not recommended |

## Files

| File | Description |
|------|-------------|
| `npr_scaled_fast.py` | **Part I** — Google Analogy + compositional tasks + probing |
| `npr_world_model.py` | **Part II** — Text World Model (GPT-2 perceiver) |
| `npr_jepa_world_model.py` | **Part III** — JEPA Visual World Model (ViT perceiver) + generalization tests |
| `NPR_Paper.pdf` | Academic paper |
| `README.md` | This file |

---

## Limitations (honest)

- **Structurally traceable, not fully interpretable**: you can see which primitive is selected, but the 768-dim vector transformations are not semantically transparent
- **Controlled environment**: discrete states, synthetic images, no real-world noise
- **Renderer bias**: visual cues correlate with properties (steam = hot); counterfactual test measures this
- **Greedy planning**: 1-step lookahead model-based search, no beam search or MCTS
- **2 slots**: may not scale to complex multi-property scenes

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
