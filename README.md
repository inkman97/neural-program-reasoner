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
Part III: Image → ViT  → discovers physical rules (JEPA)  → 93.8% single-step, 67.7% held-out generalization
```

---

## Three Parts

### Part I: Linguistic Analogy (Google Analogy Dataset)
NPR discovers rules for word relationships and composes them for multi-relation reasoning. **98.8% on compositional tasks.**

### Part II: Text World Model (Physical Rule Discovery)
NPR implements LeCun's three-module architecture (Perceiver, World Model, Reasoner) with GPT-2 as perceiver. Discovers physical rules from text, simulates state transitions, plans actions. **100% single-step accuracy on 13 actions.**

### Part III: JEPA Visual World Model
Replaces GPT-2 with ViT + EMA target network. Learns from synthetic scene images. Predicts entirely in latent space — no vocabulary, no decoder. Model-based planning via World Model simulation. Includes three generalization tests: held-out objects, counterfactual visuals, and cross-property transfer. **93.8% single-step, 100% counterfactual.**

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
| Single-step NN (nearest neighbor) | **93.8%** |
| Latent similarity (avg cosine) | **0.974** |
| Depth-2 compositional chains | **94%** |
| Depth-3 compositional chains | **94%** |
| Planning first action (real) | **61%** |
| **Held-out objects** | **67.7% (21/31)** |
| **Counterfactual visuals** | **100% (6/6)** |
| **Cross-property transfer** | **0% (0/6)** |

---

## Architecture

### Part III: JEPA World Model

```
┌───────────────────────────────────────────────────────┐
│  PERCEIVER                                             │
│  ViT online (last 2 layers unfrozen, 14.2M params)    │
│  ViT target (EMA copy, decay=0.996, no gradients)     │
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
- **EMA target network**: online ViT adapts to domain, target ViT provides stable prediction targets — prevents representation collapse (follows BYOL/V-JEPA)
- **Model-based planning**: simulates all actions through the real World Model, picks best — no separate action predictor
- **Predicts in latent space**: no vocabulary, no decoder — true JEPA-style prediction
- **Cyclic Self-Improvement**: frequent primitive pairs compressed into new operations
- **Structurally traceable**: every decision (object extraction, slot selection, program, FiLM parameters) is inspectable

### Generalization Tests (Part III)

Three tests address common criticisms of toy-world models:

- **Held-out objects (67.7%)**: 2 objects per property type excluded from training. 9/13 actions generalize perfectly to unseen objects. Fails on position/containment/fullness where objects have unique visual renderings
- **Counterfactual visuals (100%)**: images rendered with wrong visual cues (e.g., "hot" rendered with cold colors). Model relies on rule structure, not visual shortcuts
- **Cross-property transfer (0%)**: actions applied to objects from different property types. Structural limit of ViT embedding space

---

## How it differs from existing JEPA implementations

| | V-JEPA 2 (Meta) | Causal-JEPA | NPR-JEPA (ours) |
|---|---|---|---|
| Training data | 1M+ hours video | Video sequences | 127 observations |
| Team | 30+ researchers | Research lab | 1 person |
| Compute | Large GPU cluster | Multi-GPU | Single T4, 55 min |
| Interpretability | Opaque predictor | Opaque | Traceable programs |
| Rule discovery | No | No | Yes (3-5 examples) |
| Primitive composition | No | No | Yes + cyclic compression |
| Householder NEGATE | No | No | Yes (exact involution) |
| EMA target | Yes | No | Yes |
| Planning | MPC with learned model | N/A | Model-based search (13 actions) |

---

## Quick Start

```bash
pip install torch transformers Pillow

# Part I: Linguistic Analogy
python npr_linguistic_reasoner.py

# Part II: Text World Model
python npr_jepa_world_model_text.py

# Part III: JEPA Visual World Model
python npr_jepa_world_model_visual.py
```

### Expected Runtime

| Platform | Part I | Part II | Part III |
|----------|--------|---------|----------|
| NVIDIA T4 (Kaggle/Colab) | ~30 min | ~40 min | ~55 min |
| CPU | ~4 hours | ~5 hours | not recommended |

## Files

| File                             | Description |
|----------------------------------|-------------|
| `npr_linguistic_model.py`        | **Part I** — Google Analogy + compositional tasks + probing |
| `npr_jepa_world_model_text.py`   | **Part II** — Text World Model (GPT-2 perceiver) |
| `npr_jepa_world_model_visual.py` | **Part III** — JEPA Visual World Model (ViT + EMA perceiver) + generalization tests |
| `NPR_Paper.pdf`                  | Academic paper (Part I + II + III) |
| `README.md`                      | This file |

---

## Limitations (honest)

- **Structurally traceable, not fully interpretable**: you can see which primitive is selected, but the 768-dim vector transformations are not semantically transparent
- **Controlled environment**: discrete states, synthetic images, no real-world noise
- **Renderer bias**: visual cues correlate with properties (steam = hot); counterfactual test (100%) confirms model doesn't rely on shortcuts
- **Greedy planning**: 1-step lookahead model-based search, no beam search or MCTS
- **2 slots**: may not scale to complex multi-property scenes
- **Cross-property transfer fails (0%)**: limitation of ViT embedding space, not World Model
- **GPT-2 baseline**: weak by 2026 standards; modern LLMs with CoT would likely outperform on linguistic tasks
- **Small scale**: 127 observations, 50+ objects, 13 actions — scaling untested

---

## Citation

```bibtex
@article{schino2026rules,
  title={Rules Are All You Need: Learning Interpretable Neural Programs 
         from Few Examples via Primitive Composition and 
         Cyclic Self-Improvement},
  author={Schino, Alessandro},
  year={2026},
  url={https://zenodo.org/records/19913249}
}
```

## License

MIT
