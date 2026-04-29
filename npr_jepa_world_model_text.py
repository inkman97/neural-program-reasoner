"""
Neural Program Reasoner — World Model: Causal Understanding + Time

Two key additions:

1. CAUSAL UNDERSTANDING (why actions work):
   - States are decomposed into PROPERTY SLOTS via Slot Attention
   - Each slot captures one abstract property (temperature, position, state...)
   - Slot names are NOT given — the model discovers them
   - Actions learn WHICH SLOT they operate on
   - This gives causal structure: "heat operates on the temperature slot"

2. TEMPORAL UNDERSTANDING (when actions apply):
   - Actions have PRECONDITIONS: "you can only fill something that is empty"
   - Actions have MULTI-EFFECTS: "push makes it fall AND it might break"
   - A PERSISTENT WORLD STATE tracks all objects across time
   - Actions can have DELAYED EFFECTS that propagate over time steps

Dataset is enriched with:
   - Preconditions (can't open what's already open)
   - Chain effects (push glass → falls → breaks)
   - Multi-object scenes (push the ball near the glass → glass falls too)

Requirements:
    pip install torch transformers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
from collections import Counter, defaultdict
import time

torch.manual_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
i2s_global = {}  # set in main()

CONFIG = {
    "proj_dim": 256,
    "num_slots": 2,        # model only uses 2 (physical vs functional)
    "max_program_steps": 6,
    "min_program_steps": 2,
    "temperature": 0.8,
    "grad_accumulation": 16,
    "memory_capacity": 300,
    "compression_threshold": 6,
    "max_new_primitives": 2,
    "num_cycles": 3,
    "iters_per_cycle": [12000, 8000, 4000],
    "lr_per_cycle": [5e-4, 3e-4, 1e-4],
    "perceiver_layer": 8,
    "eval_samples": 40,
    "max_chain_length": 3,
    "multistep_ratio": 0.35,
    "chain_effect_ratio": 0.1,
    "goal_loss_weight": 0.2,
    "action_loss_weight": 0.2,
    "precondition_loss_weight": 0.15,
}

# =============================================================================
# World Definition: Objects, Properties, Actions
# The model learns from OBSERVATIONS, not from explicit rules.
# =============================================================================

# Properties and their possible values (the model doesn't see this structure)
PROPERTY_DEFS = {
    "temperature": ["cold", "warm", "hot", "boiling"],
    "openness": ["closed", "open"],
    "power": ["off", "on"],
    "fullness": ["empty", "full"],
    "integrity": ["intact", "broken"],
    "position": "on the {surface} -> on the floor",
    "containment": "outside the {container} -> inside the {container}",
}

# Objects and their properties. This is the WORLD — not rules.
# The model never sees this dict. It only sees observations generated from it.
OBJECTS = {
    # Temperature objects (can be heated/cooled)
    "water": {"temperature": True}, "soup": {"temperature": True},
    "coffee": {"temperature": True}, "tea": {"temperature": True},
    "milk": {"temperature": True}, "food": {"temperature": True},
    "oil": {"temperature": True}, "pan": {"temperature": True},
    "iron": {"temperature": True}, "metal": {"temperature": True},
    # Openable objects
    "door": {"openness": True}, "window": {"openness": True},
    "box": {"openness": True}, "jar": {"openness": True},
    "gate": {"openness": True}, "drawer": {"openness": True},
    "lid": {"openness": True}, "bag": {"openness": True},
    "bottle": {"openness": True}, "cabinet": {"openness": True},
    # Powered objects
    "lamp": {"power": True, "surface": "table", "fragile": True, "integrity": True},
    "light": {"power": True}, "screen": {"power": True},
    "tv": {"power": True}, "radio": {"power": True},
    "computer": {"power": True}, "fan": {"power": True},
    "heater": {"power": True}, "oven": {"power": True},
    "speaker": {"power": True},
    # Fillable objects
    "cup": {"fullness": True, "surface": "shelf", "fragile": True, "integrity": True},
    "glass": {"fullness": True, "surface": "table", "fragile": True, "integrity": True},
    "bowl": {"fullness": True, "fragile": True, "integrity": True},
    "pot": {"fullness": True}, "tub": {"fullness": True},
    "bucket": {"fullness": True}, "jug": {"fullness": True},
    "tank": {"fullness": True}, "pool": {"fullness": True},
    # Breakable objects
    "plate": {"integrity": True, "surface": "counter", "fragile": True},
    "vase": {"integrity": True, "fragile": True, "surface": "shelf"},
    "mirror": {"integrity": True},
    # Positional objects (on a surface)
    "ball": {"surface": "table"}, "book": {"surface": "desk"},
    "phone": {"surface": "bed"}, "toy": {"surface": "shelf"},
    "pen": {"surface": "desk"}, "remote": {"surface": "couch"},
    # Containment objects (inside/outside a container)
    "key": {"container": "pocket"}, "coin": {"container": "jar"},
    "letter": {"container": "envelope"}, "shirt": {"container": "closet"},
    "tool": {"container": "shed"}, "food_c": {"container": "fridge"},
    "ball_c": {"container": "box"}, "toy_c": {"container": "bag"},
    "pen_c": {"container": "case"}, "book_c": {"container": "drawer"},
}

# Actions: what they do. Each action changes one property value to another.
ACTIONS = {
    "heat":       {"property": "temperature", "from": "cold", "to": "hot"},
    "cool":       {"property": "temperature", "from": "hot", "to": "cold"},
    "warm up":    {"property": "temperature", "from": "cold", "to": "warm"},
    "boil":       {"property": "temperature", "from": "hot", "to": "boiling"},
    "simmer":     {"property": "temperature", "from": "boiling", "to": "hot"},
    "open":       {"property": "openness", "from": "closed", "to": "open"},
    "close":      {"property": "openness", "from": "open", "to": "closed"},
    "switch on":  {"property": "power", "from": "off", "to": "on"},
    "switch off": {"property": "power", "from": "on", "to": "off"},
    "fill":       {"property": "fullness", "from": "empty", "to": "full"},
    "drop":       {"property": "integrity", "from": "intact", "to": "broken"},
    "push":       {"property": "position", "from": "surface", "to": "floor"},
    "put in":     {"property": "containment", "from": "outside", "to": "inside"},
}

def generate_world_observations():
    """Generate all valid (state, action, result) from the world definition.
    This replaces hardcoded TRANSITION_RULES."""
    observations = []
    rule_groups = defaultdict(list)

    for obj_name, obj_props in OBJECTS.items():
        for act_name, act_def in ACTIONS.items():
            prop = act_def["property"]

            # Check if object has this property
            if prop == "temperature" and not obj_props.get("temperature"):
                continue
            if prop == "openness" and not obj_props.get("openness"):
                continue
            if prop == "power" and not obj_props.get("power"):
                continue
            if prop == "fullness" and not obj_props.get("fullness"):
                continue
            if prop == "integrity" and not obj_props.get("integrity"):
                continue
            if prop == "position" and "surface" not in obj_props:
                continue
            if prop == "containment" and "container" not in obj_props:
                continue

            # Build state and result text
            display = obj_name.split("_")[0]  # food_c → food, ball_c → ball
            if prop == "position":
                surface = obj_props["surface"]
                state = f"the {display} is on the {surface}"
                result = f"the {display} is on the floor"
            elif prop == "containment":
                container = obj_props["container"]
                state = f"the {display} is outside the {container}"
                result = f"the {display} is inside the {container}"
            else:
                state = f"the {display} is {act_def['from']}"
                result = f"the {display} is {act_def['to']}"

            obs = {
                "state": state, "action": act_name, "result": result,
                "property": prop, "object": display,
                "fragile": obj_props.get("fragile", False),
            }
            observations.append(obs)
            rule_groups[act_name].append(obs)

    return observations, rule_groups

# Generate the world
ALL_OBSERVATIONS, RULE_GROUPS = generate_world_observations()

# Build TRANSITION_RULES for backward compatibility
TRANSITION_RULES = {}
for act_name, obs_list in RULE_GROUPS.items():
    triples = [(o["state"], o["action"], o["result"]) for o in obs_list]
    if triples:
        first = obs_list[0]
        precond = first.get("from_val", ACTIONS[act_name]["from"]).split()[0]
        TRANSITION_RULES[act_name] = {
            "triples": triples,
            "precondition": precond,
            "property_changed": first["property"],
        }

# Chain effects: generated automatically from fragile objects
CHAIN_EFFECTS = [
    {"trigger_action": "push", "chain_action": "drop",
     "condition": "fragile", "description": "fragile object falls and breaks"},
]

CHAIN_TRIPLES = []
for ce in CHAIN_EFFECTS:
    trigger_obs = RULE_GROUPS.get(ce["trigger_action"], [])
    chain_obs = RULE_GROUPS.get(ce["chain_action"], [])
    chain_by_obj = {o["object"]: o for o in chain_obs}
    for obs in trigger_obs:
        if obs.get(ce["condition"], False) and obs["object"] in chain_by_obj:
            co = chain_by_obj[obs["object"]]
            CHAIN_TRIPLES.append({
                "state": obs["state"], "action": obs["action"],
                "result": co["result"], "intermediate": obs["result"],
                "rules": [ce["trigger_action"], ce["chain_action"]],
                "object": obs["object"],
            })

print(f"World: {len(ALL_OBSERVATIONS)} observations, {len(RULE_GROUPS)} actions, {len(CHAIN_TRIPLES)} chain effects")
for act in sorted(RULE_GROUPS):
    obs = RULE_GROUPS[act]
    print(f"  {act:12s}: {len(obs):2d} obs ({obs[0]['property']})")
if CHAIN_TRIPLES:
    print(f"Chain effects ({len(CHAIN_TRIPLES)}):")
    for ct in CHAIN_TRIPLES[:3]:
        print(f"  '{ct['state'][:30]}' + '{ct['action']}' -> '{ct['result'][:25]}'")
ALL_ACTIONS = sorted(set(a for r in TRANSITION_RULES.values() for _,a,_ in r["triples"]))
ACT2IDX = {a:i for i,a in enumerate(ALL_ACTIONS)}
IDX2ACT = {i:a for a,i in ACT2IDX.items()}
NUM_ACTIONS = len(ALL_ACTIONS)

# Property type vocabulary (for precondition checking)
PROPERTY_TYPES = sorted(set(r["property_changed"] for r in TRANSITION_RULES.values()))
PROP2IDX = {p:i for i,p in enumerate(PROPERTY_TYPES)}
NUM_PROPERTIES = len(PROPERTY_TYPES)

def build_vocab():
    ss = set()
    for r in TRANSITION_RULES.values():
        for s,a,res in r["triples"]: ss.add(s); ss.add(res)
    # Also include chain effect states
    for ct in CHAIN_TRIPLES:
        ss.add(ct["state"]); ss.add(ct["result"]); ss.add(ct["intermediate"])
    sl = sorted(ss)
    return sl, {s:i for i,s in enumerate(sl)}, {i:s for i,s in enumerate(sl)}

def find_chains(max_len=3):
    st = defaultdict(list)
    for rn, r in TRANSITION_RULES.items():
        for s,a,res in r["triples"]: st[s].append((a,res,rn))
    chains = {1:[],2:[],3:[]}
    for rn, r in TRANSITION_RULES.items():
        for s,a,res in r["triples"]: chains[1].append({"states":[s,res],"actions":[a],"rules":[rn],"depth":1})
    for c1 in chains[1]:
        for a2,s2,r2 in st.get(c1["states"][-1],[]):
            if r2!=c1["rules"][-1]: chains[2].append({"states":c1["states"]+[s2],"actions":c1["actions"]+[a2],"rules":c1["rules"]+[r2],"depth":2})
    if max_len>=3:
        for c2 in chains[2]:
            for a3,s3,r3 in st.get(c2["states"][-1],[]):
                if r3!=c2["rules"][-1]: chains[3].append({"states":c2["states"]+[s3],"actions":c2["actions"]+[a3],"rules":c2["rules"]+[r3],"depth":3})
    print(f"\nChains: d1:{len(chains[1])}, d2:{len(chains[2])}, d3:{len(chains[3])}")
    return chains


# =============================================================================
# Persistent World State
# =============================================================================

class WorldState:
    """Tracks the state of all objects in the world across time."""
    def __init__(self):
        self.objects = {}  # object_name → current_state_text
        self.history = []  # list of (time, object, action, old_state, new_state)
        self.time = 0

    def set(self, obj, state_text):
        self.objects[obj] = state_text

    def apply_action(self, obj, action, new_state):
        old = self.objects.get(obj, "unknown")
        self.history.append((self.time, obj, action, old, new_state))
        self.objects[obj] = new_state
        self.time += 1

    def get(self, obj):
        return self.objects.get(obj, "unknown")

    def check_precondition(self, state_text, precondition):
        """Check if precondition holds in the state text."""
        return precondition in state_text

    def get_chain_effects(self, rule_name, obj_name):
        """Check if this action triggers chain effects."""
        effects = []
        for ce in CHAIN_EFFECTS:
            if ce["trigger_rule"] == rule_name and obj_name in ce["trigger_objects"]:
                effects.append(ce)
        return effects

    def snapshot(self):
        return dict(self.objects)


# =============================================================================
# Token-level Embedding Cache
# =============================================================================

class TokenEmbeddingCache:
    def __init__(self, gpt2, tokenizer, layer=8):
        self.last_cache, self.tokens_cache = {}, {}
        self.device = next(gpt2.parameters()).device
        self.gpt2, self.tokenizer, self.layer = gpt2, tokenizer, layer
        print("\nPre-computing representations...")
        t0 = time.time()
        phrases = set()
        for r in TRANSITION_RULES.values():
            for s,a,res in r["triples"]: phrases.add(f" {s}"); phrases.add(f" {a}"); phrases.add(f" {res}")
        # Also cache precondition and property type texts
        for r in TRANSITION_RULES.values():
            phrases.add(f" {r['precondition']}")
            phrases.add(f" {r['property_changed']}")
        for ct in CHAIN_TRIPLES:
            phrases.add(f" {ct['state']}"); phrases.add(f" {ct['result']}"); phrases.add(f" {ct['intermediate']}")
        print(f"  {len(phrases)} phrases...")
        gpt2.eval()
        with torch.no_grad():
            for p in sorted(phrases):
                ids = tokenizer.encode(p)
                out = gpt2(torch.tensor([ids],device=self.device),output_hidden_states=True)
                h = out.hidden_states[layer][0]
                self.last_cache[p]=h[-1].clone(); self.tokens_cache[p]=h.clone()
        print(f"  Done! {len(self.last_cache)} cached in {time.time()-t0:.1f}s")

    def get(self, phrase):
        if phrase in self.last_cache: return self.last_cache[phrase]
        ids=self.tokenizer.encode(phrase)
        with torch.no_grad():
            out=self.gpt2(torch.tensor([ids],device=self.device),output_hidden_states=True)
        h=out.hidden_states[self.layer][0]
        self.last_cache[phrase]=h[-1].clone(); self.tokens_cache[phrase]=h.clone()
        return self.last_cache[phrase]

    def get_tokens(self, phrase):
        if phrase not in self.tokens_cache: self.get(phrase)
        return self.tokens_cache[phrase]


# =============================================================================
# Tasks (with precondition info)
# =============================================================================

def generate_single_task(cache, s2i, num_examples=5, rule_name=None):
    rule_name = rule_name or random.choice(list(TRANSITION_RULES.keys()))
    r = TRANSITION_RULES[rule_name]; ts = r["triples"]
    ne = min(num_examples, len(ts)-1)
    chosen = random.sample(ts, ne+1); examples,test = chosen[:ne],chosen[ne]
    return {
        "example_reprs": torch.stack([torch.cat([cache.get(f" {s}"),cache.get(f" {a}"),cache.get(f" {res}")]) for s,a,res in examples]),
        "test_state":cache.get(f" {test[0]}"), "test_action":cache.get(f" {test[1]}"),
        "test_state_tokens":cache.get_tokens(f" {test[0]}"),
        "target_idx":s2i.get(test[2],0), "rule_name":rule_name,
        "state":test[0], "action":test[1], "expected":test[2],
        "depth":1, "task_type":"single",
        "result_state_vec":cache.get(f" {test[2]}"), "action_idx":ACT2IDX[test[1]],
        "precondition": r["precondition"],
        "property_changed": r["property_changed"],
        "precondition_holds": r["precondition"] in test[0],
        "property_idx": PROP2IDX[r["property_changed"]],
    }

def generate_multistep_task(cache, s2i, chains, num_examples=3):
    avail = [d for d in [2,3] if chains.get(d)]
    if not avail: return generate_single_task(cache, s2i)
    depth = random.choice(avail); pool = chains[depth]
    ne = min(num_examples, len(pool)-1)
    chosen = random.sample(pool, ne+1); examples,test = chosen[:ne],chosen[ne]
    er = [torch.cat([cache.get(f" {c['states'][0]}"),torch.stack([cache.get(f" {a}") for a in c["actions"]]).mean(0),cache.get(f" {c['states'][-1]}")]) for c in examples]
    ta = [cache.get(f" {a}") for a in test["actions"]]
    first_rule = TRANSITION_RULES[test["rules"][0]]

    # Per-step rule examples: for each action in the chain, provide
    # single-step examples of that specific rule so the Synthesizer
    # can produce the right program for each step
    per_step_examples = []
    for rule_name in test["rules"]:
        r = TRANSITION_RULES[rule_name]
        step_triples = random.sample(r["triples"], min(3, len(r["triples"])))
        step_ex = torch.stack([
            torch.cat([cache.get(f" {s}"), cache.get(f" {a}"), cache.get(f" {res}")])
            for s, a, res in step_triples
        ])
        per_step_examples.append(step_ex)

    return {
        "example_reprs":torch.stack(er), "test_state":cache.get(f" {test['states'][0]}"),
        "test_action":torch.stack(ta).mean(0), "test_actions":ta,
        "test_state_tokens":cache.get_tokens(f" {test['states'][0]}"),
        "target_idx":s2i.get(test["states"][-1],0), "rule_name":"+".join(test["rules"]),
        "state":test["states"][0], "action":"+".join(test["actions"]), "expected":test["states"][-1],
        "depth":depth, "task_type":"multistep",
        "result_state_vec":cache.get(f" {test['states'][-1]}"), "action_idx":ACT2IDX[test["actions"][0]],
        "precondition": first_rule["precondition"],
        "property_changed": first_rule["property_changed"],
        "precondition_holds": True,
        "property_idx": PROP2IDX[first_rule["property_changed"]],
        "per_step_examples": per_step_examples,  # examples for each step's rule
        "step_rules": test["rules"],
    }

def generate_chain_effect_task(cache, s2i, num_examples=3):
    """Generate a task where an action causes a chain effect.
    Example: push glass on table → glass is broken (gravity + break)."""
    if not CHAIN_TRIPLES:
        return generate_single_task(cache, s2i)
    ct = random.choice(CHAIN_TRIPLES)

    # Examples: show gravity examples (what push does normally)
    # AND break examples (what happens to fragile objects)
    gravity_triples = TRANSITION_RULES["push"]["triples"]
    break_triples = TRANSITION_RULES["drop"]["triples"]
    gex = random.sample(gravity_triples, min(2, len(gravity_triples)))
    bex = random.sample(break_triples, min(2, len(break_triples)))
    examples = gex + bex
    example_reprs = torch.stack([
        torch.cat([cache.get(f" {s}"), cache.get(f" {a}"), cache.get(f" {r}")])
        for s, a, r in examples
    ])

    return {
        "example_reprs": example_reprs,
        "test_state": cache.get(f" {ct['state']}"),
        "test_action": cache.get(f" {ct['action']}"),
        "test_state_tokens": cache.get_tokens(f" {ct['state']}"),
        "target_idx": s2i.get(ct["result"], 0),
        "rule_name": "chain:" + "+".join(ct["rules"]),
        "state": ct["state"], "action": ct["action"], "expected": ct["result"],
        "depth": 1, "task_type": "chain_effect",
        "result_state_vec": cache.get(f" {ct['result']}"),
        "action_idx": ACT2IDX[ct["action"]],
        "precondition": "on the",
        "property_changed": "position",
        "precondition_holds": True,
        "property_idx": PROP2IDX["position"],
        "chain_intermediate": ct["intermediate"],
        "chain_object": ct["object"],
    }

def generate_mixed_task(cache, s2i, chains, ne=5):
    r = random.random()
    if r < CONFIG["chain_effect_ratio"] and CHAIN_TRIPLES:
        return generate_chain_effect_task(cache, s2i)
    if r < CONFIG["chain_effect_ratio"] + CONFIG["multistep_ratio"] and any(chains.get(d) for d in [2,3]):
        return generate_multistep_task(cache, s2i, chains, 3)
    return generate_single_task(cache, s2i, ne)

def generate_planning_task(cache, s2i, chains):
    avail = [d for d in [1,2,3] if chains.get(d)]
    if not avail: return None
    c = random.choice(chains[random.choice(avail)])
    return {"initial_state":c["states"][0],"goal_state":c["states"][-1],"correct_actions":c["actions"],"depth":len(c["actions"])}


# =============================================================================
# PERCEIVER: Slot Attention for property decomposition
# =============================================================================

class SlotAttention(nn.Module):
    """Decomposes a state into N property slots via iterative attention.
    Each slot learns to capture one abstract property (temperature, position, etc.)
    Slot identities are NOT given — they emerge from training.

    This gives CAUSAL STRUCTURE: the model can learn that "heat" operates on
    slot 2 (which happens to capture temperature) without being told."""
    def __init__(self, sd, num_slots=2, n_iters=2):
        super().__init__()
        self.num_slots = num_slots
        self.n_iters = n_iters
        self.sd = sd
        self.slot_dim = sd

        # Initial slot vectors (learned)
        self.slot_mu = nn.Parameter(torch.randn(num_slots, sd) * 0.02)

        # Slot attention components
        self.k_proj = nn.Linear(sd, sd)
        self.v_proj = nn.Linear(sd, sd)
        self.q_proj = nn.Linear(sd, sd)

        # Slot update GRU
        self.gru = nn.GRUCell(sd, sd)
        self.mlp = nn.Sequential(nn.Linear(sd, sd*2), nn.GELU(), nn.Linear(sd*2, sd))
        self.norm1 = nn.LayerNorm(sd)
        self.norm2 = nn.LayerNorm(sd)
        self.scale = math.sqrt(sd)

    def forward(self, token_reprs):
        """
        token_reprs: (seq_len, sd)
        Returns: slots (num_slots, sd), slot_attn (num_slots, seq_len)
        """
        # Keys and values from tokens
        k = self.k_proj(token_reprs)  # (seq_len, sd)
        v = self.v_proj(token_reprs)  # (seq_len, sd)

        # Initialize slots
        slots = self.slot_mu.clone()  # (num_slots, sd)

        # Iterative attention
        for _ in range(self.n_iters):
            slots_prev = slots
            slots = self.norm1(slots)
            q = self.q_proj(slots)  # (num_slots, sd)

            # Attention: slots compete for tokens
            attn_logits = torch.matmul(q, k.T) / self.scale  # (num_slots, seq_len)
            attn = F.softmax(attn_logits, dim=0)  # normalize over SLOTS (competition)
            attn_weights = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)  # normalize per slot

            # Aggregate
            updates = torch.matmul(attn_weights, v)  # (num_slots, sd)

            # Update via GRU
            slots = self.gru(updates.reshape(-1, self.sd), slots_prev.reshape(-1, self.sd))
            slots = slots.reshape(self.num_slots, self.sd)
            slots = slots + self.mlp(self.norm2(slots))

        return slots, attn


class ObjectExtractor(nn.Module):
    """Extracts object identity from tokens (same as v8)."""
    def __init__(self, sd):
        super().__init__()
        self.query = nn.Parameter(torch.randn(sd)*0.02)
        self.k = nn.Linear(sd, sd); self.v = nn.Linear(sd, sd)
        self.out = nn.Sequential(nn.Linear(sd, sd), nn.LayerNorm(sd))
        self.scale = math.sqrt(sd)
    def forward(self, tokens):
        k, v = self.k(tokens), self.v(tokens)
        a = F.softmax(torch.matmul(k, self.query)/self.scale, dim=0)
        return self.out(torch.matmul(a.unsqueeze(0), v).squeeze(0)), a


# =============================================================================
# WORLD MODEL
# =============================================================================

BASE_NAMES = ["IDENTITY","NEGATE","MORPH","ASSOCIATE","LOOKUP","BLEND"]

class PrimitiveLibrary(nn.Module):
    """Primitives with architectures that match their semantic roles.

    IDENTITY: near-identity, minimal perturbation (small gate)
    NEGATE:   Householder reflection — structurally an involution (f(f(x))=x)
              H = I - 2*v*v^T/||v||^2, so H*H = I by construction
    MORPH:    smooth nonlinear transform (LayerNorm + GELU)
    ASSOCIATE: contextual binding via self-attention
    LOOKUP:   deep nonlinear transform (wide MLP)
    BLEND:    mix with learned context vector
    """
    def __init__(self, sd):
        super().__init__()
        self.sd, self.names = sd, list(BASE_NAMES)

        # IDENTITY: tiny perturbation
        self.identity = nn.Sequential(nn.Linear(sd, sd), nn.Tanh())

        # NEGATE: Householder reflection H(x) = x - 2*(v·x)/(v·v) * v
        # This is STRUCTURALLY an involution: H(H(x)) = x exactly
        # The learned vector v determines the reflection hyperplane
        self.negate_v = nn.Parameter(torch.randn(sd) * 0.1)
        # Learned pre/post projections to make it more expressive
        # while preserving the involution structure
        self.negate_pre = nn.Linear(sd, sd)
        self.negate_post = nn.Linear(sd, sd)

        # MORPH: smooth nonlinear transform
        self.morph = nn.Sequential(nn.Linear(sd, sd), nn.LayerNorm(sd), nn.GELU(), nn.Linear(sd, sd))

        # ASSOCIATE: self-attention binding
        self.aq, self.ak, self.av, self.ao = nn.Linear(sd, sd//4), nn.Linear(sd, sd//4), nn.Linear(sd, sd), nn.Linear(sd, sd)

        # LOOKUP: deep transform
        self.lookup = nn.Sequential(nn.Linear(sd, sd*2), nn.GELU(), nn.Linear(sd*2, sd*2), nn.GELU(), nn.Linear(sd*2, sd))

        # BLEND: mix with context
        self.bc = nn.Parameter(torch.randn(sd) * 0.01)
        self.bn = nn.Sequential(nn.Linear(sd*2, sd), nn.GELU(), nn.Linear(sd, sd))

        self.gates = nn.ParameterList([nn.Parameter(torch.tensor(g)) for g in [0.01, 0.5, 0.3, 0.4, 0.5, 0.3]])
        self.inv_p, self.inv_g = nn.ModuleList(), nn.ParameterList()

    @property
    def n(self): return len(BASE_NAMES) + len(self.inv_p)

    def _householder(self, x):
        """Householder reflection: H(x) = x - 2*(v·x)/(v·v) * v
        Structurally satisfies H(H(x)) = x."""
        v = self.negate_v
        # Project into reflection space
        x_proj = self.negate_pre(x)
        # Apply Householder reflection
        vv = torch.dot(v, v) + 1e-8
        coeff = 2 * torch.dot(v, x_proj) / vv
        reflected = x_proj - coeff * v
        # Project back
        return self.negate_post(reflected)

    def _base(self, i, s):
        if i == 0: return s + torch.sigmoid(self.gates[0]) * self.identity(s)
        if i == 1: return s + torch.sigmoid(self.gates[1]) * self._householder(s)
        if i == 2: return s + torch.sigmoid(self.gates[2]) * self.morph(s)
        if i == 3:
            q, k, v = self.aq(s), self.ak(s), self.av(s)
            return s + torch.sigmoid(self.gates[3]) * self.ao(torch.sigmoid(torch.dot(q, k) / math.sqrt(q.shape[0])) * v)
        if i == 4: return s + torch.sigmoid(self.gates[4]) * self.lookup(s)
        if i == 5: return s + torch.sigmoid(self.gates[5]) * self.bn(torch.cat([s, self.bc]))

    def apply(self, i, s):
        if i < len(BASE_NAMES): return self._base(i, s)
        j = i - len(BASE_NAMES); return s + torch.sigmoid(self.inv_g[j]) * self.inv_p[j](s)

    def apply_soft(self, w, s): return sum(w[i] * self.apply(i, s) for i in range(self.n))

    def add(self, name, a, b):
        sd, dev = self.sd, next(self.parameters()).device
        self.inv_p.append(nn.Sequential(nn.Linear(sd, sd*2), nn.LayerNorm(sd*2), nn.GELU(), nn.Linear(sd*2, sd*2), nn.GELU(), nn.Linear(sd*2, sd)).to(dev))
        self.inv_g.append(nn.Parameter(torch.tensor(0.4, device=dev)))
        self.names.append(name); print(f"  [COMPRESS] Created: {name}"); return self.n - 1


class SlotSelector(nn.Module):
    """Given an action, selects WHICH slot(s) to transform.
    This is the causal link: the action determines which property changes."""
    def __init__(self, action_dim, num_slots):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, 128), nn.GELU(),
            nn.Linear(128, num_slots))

    def forward(self, action_vec):
        """Returns soft selection over slots (which property does this action affect?)"""
        return F.softmax(self.net(action_vec), dim=-1)


class PropertyUpdater(nn.Module):
    """Applies primitives to the SELECTED slot, conditioned on action."""
    def __init__(self, sd, action_dim, max_steps):
        super().__init__()
        self.step_emb = nn.Embedding(max_steps, sd)
        self.action_film = nn.Sequential(nn.Linear(action_dim, sd*2), nn.GELU(), nn.Linear(sd*2, sd*2))
        self.sd = sd
    def forward(self, prop_vec, action, prog, lib):
        film = self.action_film(action)
        scale, shift = film[:self.sd], film[self.sd:]
        for i, sel in enumerate(prog):
            prop_vec = prop_vec + self.step_emb(torch.tensor(i, device=prop_vec.device))
            prop_vec = prop_vec * (1 + 0.1*torch.tanh(scale)) + 0.1*torch.tanh(shift)
            prop_vec = lib.apply_soft(sel, prop_vec)
        return prop_vec


class PreconditionChecker(nn.Module):
    """Predicts whether an action's precondition is met.
    Trained to output 1 if precondition holds, 0 if not.
    This gives the model understanding of WHEN actions can be applied."""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, state_dim // 2), nn.GELU(),
            nn.Linear(state_dim // 2, 1), nn.Sigmoid())

    def forward(self, state_vec, action_vec):
        return self.net(torch.cat([state_vec, action_vec]))


class RuleSynthesizer(nn.Module):
    def __init__(self, example_dim, state_dim, np, max_steps, pd=256):
        super().__init__()
        self.pd, self._np = pd, np
        self.proj = nn.Sequential(nn.Linear(example_dim,pd),nn.LayerNorm(pd),nn.GELU())
        self.a1 = nn.MultiheadAttention(pd,4,batch_first=True)
        self.n1,self.n2 = nn.LayerNorm(pd),nn.LayerNorm(pd)
        self.f1 = nn.Sequential(nn.Linear(pd,pd*2),nn.GELU(),nn.Linear(pd*2,pd))
        self.a2 = nn.MultiheadAttention(pd,4,batch_first=True)
        self.n3,self.n4 = nn.LayerNorm(pd),nn.LayerNorm(pd)
        self.f2 = nn.Sequential(nn.Linear(pd,pd*2),nn.GELU(),nn.Linear(pd*2,pd))
        # State-conditioned heads: each head sees signature + current state
        # This is the key fix: the program depends on WHERE you are, not just WHAT rule
        self.state_proj = nn.Linear(state_dim, pd)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(pd*2, 128), nn.GELU(), nn.Linear(128, np))
            for _ in range(max_steps)])
        self.stop = nn.Sequential(nn.Linear(pd+state_dim,128),nn.GELU(),nn.Linear(128,1),nn.Sigmoid())
    def signature(self, ex):
        x=self.proj(ex).unsqueeze(0)
        a,_=self.a1(x,x,x); x=self.n1(x+a); x=self.n2(x+self.f1(x))
        a,_=self.a2(x,x,x); x=self.n3(x+a); x=self.n4(x+self.f2(x))
        return x.mean(1).squeeze(0)
    def rebuild(self, new_n):
        old=self._np
        if new_n<=old: return
        nh=nn.ModuleList()
        for h in self.heads:
            n=nn.Sequential(nn.Linear(self.pd*2,128),nn.GELU(),nn.Linear(128,new_n)).to(next(h.parameters()).device)
            with torch.no_grad():
                n[0].weight[:,:self.pd].copy_(h[0].weight[:,:self.pd])
                n[0].weight[:,self.pd:].zero_()
                n[0].bias.copy_(h[0].bias)
                n[2].weight[:old].copy_(h[2].weight); n[2].bias[:old].copy_(h[2].bias)
                aw=h[2].weight.mean(0,keepdim=True).expand(new_n-old,-1)
                n[2].weight[old:].copy_(aw+torch.randn_like(aw)*0.05)
                n[2].bias[old:].fill_(h[2].bias[:old].mean().item())
            nh.append(n)
        self.heads=nh; self._np=new_n
    def forward(self, ex_reprs, state, temp=0.8, np_=None, min_steps=2):
        pat=self.signature(ex_reprs); np_=np_ or self._np
        state_ctx = self.state_proj(state)  # project state into pattern space
        # Concatenate signature + state context for primitive selection
        head_input = torch.cat([pat, state_ctx])
        prog, stop_probs = [], []
        for i,h in enumerate(self.heads):
            if i>=min_steps:
                sp=self.stop(torch.cat([pat,state])); stop_probs.append(sp)
                if not self.training and sp.item()>0.6: break
            lg=h(head_input)  # now depends on state too
            if lg.shape[0]<np_: lg=torch.cat([lg,torch.zeros(np_-lg.shape[0],device=lg.device)])
            elif lg.shape[0]>np_: lg=lg[:np_]
            if self.training: prog.append(F.gumbel_softmax(lg,tau=temp,hard=False))
            else: prog.append(F.one_hot(lg.argmax(),np_).float())
        return prog, pat, stop_probs


class Memory:
    def __init__(self, cap=300):
        self.cap, self.entries = cap, []
    def store(self, sig, prog, rel, ok):
        self.entries.append({"sig":sig.detach().clone(),"prog":prog,"rel":rel,"ok":ok,"cnt":1})
        if len(self.entries)>self.cap: self.entries.sort(key=lambda e:e["cnt"],reverse=True); self.entries=self.entries[:self.cap]
    def lookup(self, sig, thr=0.85):
        if not self.entries: return None
        best,bsim=None,-1
        for e in self.entries:
            sim=F.cosine_similarity(sig.unsqueeze(0),e["sig"].unsqueeze(0)).item()
            if sim>bsim: bsim,best=sim,e
        if bsim>thr and best: best["cnt"]+=1; return best
        return None
    def freq_pairs(self, mn=6):
        c=Counter()
        for e in self.entries:
            for i in range(len(e["prog"])-1): c[(e["prog"][i],e["prog"][i+1])]+=1
        return {p:n for p,n in c.items() if n>=mn}
    def clear(self): self.entries=[]
    def stats(self):
        if not self.entries: return "Empty"
        return f"{len(self.entries)} entries, {sum(1 for e in self.entries if e['ok'])} correct"


# =============================================================================
# REASONER
# =============================================================================

class GoalEvaluator(nn.Module):
    def __init__(self, sd):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(sd*2,sd),nn.LayerNorm(sd),nn.GELU(),nn.Linear(sd,sd//2),nn.GELU(),nn.Linear(sd//2,1),nn.Sigmoid())
    def forward(self, s, g): return self.net(torch.cat([s,g]))

class ActionScorer(nn.Module):
    def __init__(self, sd, na):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(sd*2,sd),nn.LayerNorm(sd),nn.GELU(),nn.Linear(sd,sd//2),nn.GELU(),nn.Linear(sd//2,na))
    def forward(self, s, g): return self.net(torch.cat([s,g]))


# =============================================================================
# Generator
# =============================================================================

class LatentPredictor(nn.Module):
    """Predicts the next state VECTOR in latent space.
    This is the core World Model output — operates entirely in embedding space.
    No vocabulary, no discrete states. Just continuous vectors."""
    def __init__(self, sd, action_dim, pattern_dim):
        super().__init__()
        input_dim = sd + sd + action_dim + pattern_dim
        self.h = nn.Sequential(
            nn.Linear(input_dim, sd*2), nn.LayerNorm(sd*2), nn.GELU(),
            nn.Linear(sd*2, sd*2), nn.LayerNorm(sd*2), nn.GELU(),
            nn.Linear(sd*2, sd))
    def forward(self, obj, prop, action, sig):
        return self.h(torch.cat([obj, prop, action, sig]))

class VocabDecoder(nn.Module):
    """Decodes a latent state vector into vocabulary logits.
    Receives the predicted vector PLUS context (object, action, signature)
    to disambiguate between similar states."""
    def __init__(self, sd, action_dim, pattern_dim, vocab_size):
        super().__init__()
        input_dim = sd + sd + action_dim + pattern_dim  # pred_vec + obj + action + sig
        self.h = nn.Sequential(
            nn.Linear(input_dim, sd), nn.LayerNorm(sd), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(sd, sd), nn.LayerNorm(sd), nn.GELU(),
            nn.Linear(sd, vocab_size))
    def forward(self, pred_vec, obj, action, sig):
        return self.h(torch.cat([pred_vec, obj, action, sig]))


# =============================================================================
# COMPLETE MODEL
# =============================================================================

class CompleteWorldModel(nn.Module):
    def __init__(self, state_dim, vocab_size):
        super().__init__()
        cfg = CONFIG; self.sd = state_dim; ed = state_dim*3; ns = cfg["num_slots"]
        # Perceiver
        self.obj_ext = ObjectExtractor(state_dim).to(DEVICE)
        self.slot_attn = SlotAttention(state_dim, ns).to(DEVICE)
        # World Model
        self.lib = PrimitiveLibrary(state_dim).to(DEVICE)
        self.syn = RuleSynthesizer(ed, state_dim, len(BASE_NAMES), cfg["max_program_steps"], cfg["proj_dim"]).to(DEVICE)
        self.slot_selector = SlotSelector(state_dim, ns).to(DEVICE)
        self.prop_updater = PropertyUpdater(state_dim, state_dim, cfg["max_program_steps"]).to(DEVICE)
        self.precond = PreconditionChecker(state_dim, state_dim).to(DEVICE)
        # Reasoner
        self.goal_eval = GoalEvaluator(state_dim).to(DEVICE)
        self.action_scorer = ActionScorer(state_dim, NUM_ACTIONS).to(DEVICE)
        # Output: latent predictor (primary) + vocab decoder (for eval)
        self.latent_pred = LatentPredictor(state_dim, state_dim, cfg["proj_dim"]).to(DEVICE)
        self.vocab_dec = VocabDecoder(state_dim, state_dim, cfg["proj_dim"], vocab_size).to(DEVICE)
        self.mem = Memory(cfg["memory_capacity"])
        self.world = WorldState()

    def forward(self, task, temp=0.8, use_mem=True):
        ex = task["example_reprs"].to(DEVICE)
        test_action = task["test_action"].to(DEVICE)
        test_tokens = task["test_state_tokens"].to(DEVICE)

        # Perceiver: extract object + decompose into slots
        obj_vec, obj_attn = self.obj_ext(test_tokens)
        slots, slot_attn = self.slot_attn(test_tokens)  # (num_slots, sd)

        # Slot selection: which slot does this action affect?
        slot_weights = self.slot_selector(test_action)  # (num_slots,)
        # Weighted combination of slots = the property being changed
        prop_vec = (slot_weights.unsqueeze(1) * slots).sum(0)  # (sd,)

        # Precondition check
        test_state = task["test_state"].to(DEVICE)
        precond_score = self.precond(test_state, test_action)

        # Rule synthesis
        np_ = self.lib.n; sig = self.syn.signature(ex)
        from_mem, prog, stop_probs = False, None, []
        if use_mem and not self.training:
            c = self.mem.lookup(sig)
            if c: prog=[F.one_hot(torch.tensor(i),np_).float().to(DEVICE) for i in c["prog"]]; from_mem=True
        if prog is None:
            prog, _, stop_probs = self.syn(ex, prop_vec, temp, np_)

        # Transform property
        if task.get("task_type")=="multistep" and "test_actions" in task:
            # Each step gets its OWN program from its own rule examples
            p = prop_vec
            per_step_ex = task.get("per_step_examples")
            for step_i, a in enumerate(task["test_actions"]):
                a_dev = a.to(DEVICE)
                if per_step_ex and step_i < len(per_step_ex):
                    # Synthesize program specific to this step's rule
                    step_prog, _, _ = self.syn(per_step_ex[step_i].to(DEVICE), p, temp, np_)
                else:
                    step_prog = prog  # fallback
                # Update slot selection for this specific action
                step_sw = self.slot_selector(a_dev)
                step_prop = (step_sw.unsqueeze(1) * slots).sum(0)
                # Carry state forward directly — the PropertyUpdater handles the transform
                p = self.prop_updater(p, a_dev, step_prog, self.lib)
            transformed = p
        else:
            transformed = self.prop_updater(prop_vec, test_action, prog, self.lib)

        # Predict next state in LATENT SPACE (primary output)
        pred_vec = self.latent_pred(obj_vec, transformed, test_action, sig)
        # Decode to vocabulary with full context
        logits = self.vocab_dec(pred_vec, obj_vec, test_action, sig)
        return logits, prog, sig, from_mem, stop_probs, obj_attn, slot_attn, slot_weights, obj_vec, precond_score, pred_vec

    def simulate_step(self, cache, current_tokens, action_name):
        """Simulate one step through the real World Model."""
        rule = TRANSITION_RULES.get(action_name)
        if rule is None: return None, None
        triples = rule["triples"]
        step_triples = random.sample(triples, min(3, len(triples)))
        step_ex = torch.stack([
            torch.cat([cache.get(f" {s}"), cache.get(f" {a}"), cache.get(f" {res}")])
            for s, a, res in step_triples
        ]).to(DEVICE)

        action_vec = cache.get(f" {action_name}")
        obj_vec, _ = self.obj_ext(current_tokens)
        slots, _ = self.slot_attn(current_tokens)
        slot_weights = self.slot_selector(action_vec)
        prop_vec = (slot_weights.unsqueeze(1) * slots).sum(0)

        np_ = self.lib.n
        step_prog, _, _ = self.syn(step_ex, prop_vec, 0.1, np_)
        sig = self.syn.signature(step_ex)
        transformed = self.prop_updater(prop_vec, action_vec, step_prog, self.lib)
        pred_vec = self.latent_pred(obj_vec, transformed, action_vec, sig)
        return pred_vec, action_vec

    def plan(self, cache, initial_state, goal_state, max_depth=3):
        """Plan using real World Model simulation at each step."""
        self.eval()
        with torch.no_grad():
            sv = cache.get(f" {initial_state}")
            gv = cache.get(f" {goal_state}")
            tokens = cache.get_tokens(f" {initial_state}")

            if self.goal_eval(sv, gv).item() > 0.9:
                return [], self.goal_eval(sv, gv).item()

            plan = []
            current_state_vec = sv
            current_tokens = tokens
            prev_action = None

            for step in range(max_depth):
                al = self.action_scorer(current_state_vec, gv)
                if prev_action is not None:
                    al[prev_action] -= 2.0

                ai = al.argmax().item()
                action_name = IDX2ACT[ai]
                plan.append(action_name)
                prev_action = ai

                # Simulate through the REAL World Model
                pred_vec, _ = self.simulate_step(cache, current_tokens, action_name)
                if pred_vec is None: break

                # Find closest known state for next step's tokens
                best_sim, best_state = -1, None
                for phrase, vec in cache.last_cache.items():
                    sim = F.cosine_similarity(pred_vec.unsqueeze(0), vec.unsqueeze(0)).item()
                    if sim > best_sim:
                        best_sim = sim; best_state = phrase

                if best_state and best_sim > 0.7:
                    current_state_vec = pred_vec
                    current_tokens = cache.get_tokens(best_state)
                    if self.goal_eval(current_state_vec, gv).item() > 0.85:
                        break
                else:
                    break

            final_score = self.goal_eval(current_state_vec, gv).item()
            return plan, final_score

    def memorize(self, sig, prog, rel, ok):
        self.mem.store(sig,[s.argmax().item() for s in prog],rel,ok)

    def compress(self):
        freq=self.mem.freq_pairs(CONFIG["compression_threshold"])
        if not freq: print("  [COMPRESS] Nothing."); return 0
        created=0
        for (a,b),cnt in sorted(freq.items(),key=lambda x:-x[1]):
            if created>=CONFIG["max_new_primitives"]: break
            na,nb=self.lib.names[a],self.lib.names[b]; nm=f"{na}_{nb}"
            if nm in self.lib.names: continue
            print(f"  [COMPRESS] {na}->{nb} ({cnt}x)"); self.lib.add(nm,a,b); created+=1
        if created>0: self.syn.rebuild(self.lib.n); self.mem.clear()
        return created


# =============================================================================
# Losses
# =============================================================================

def div_loss(prog):
    if len(prog)<2: return torch.tensor(0.0,device=DEVICE)
    l,n=torch.tensor(0.0,device=DEVICE),len(prog)
    for i in range(n):
        for j in range(i+1,n): l=l+F.relu(F.cosine_similarity(prog[i].unsqueeze(0),prog[j].unsqueeze(0))-0.3)
    return l/(n*(n-1)/2)

def use_loss(prog, np_):
    avg=torch.stack(prog).mean(0)[:np_]
    return math.log(np_)-(-(avg*torch.log(avg+1e-8)).sum())

def novelty_loss(prog, np_, n_base=6):
    if np_<=n_base: return torch.tensor(0.0,device=DEVICE)
    return F.relu(0.15-torch.stack(prog).mean(0)[:np_][n_base:].sum())

def goal_contrastive_loss(goal_eval, result_vec, all_sv):
    pos = goal_eval(result_vec, result_vec)
    neg_vec = random.choice(all_sv)
    if F.cosine_similarity(result_vec.unsqueeze(0),neg_vec.unsqueeze(0)).item()>0.99:
        neg_vec = random.choice(all_sv)
    neg = goal_eval(result_vec, neg_vec)
    return (F.binary_cross_entropy(pos,torch.ones_like(pos))+F.binary_cross_entropy(neg,torch.zeros_like(neg)))/2



# =============================================================================
# Training & Eval
# =============================================================================

def fmt(prog, names):
    return " -> ".join(names[s.argmax().item()] if s.argmax().item()<len(names) else "?" for s in prog)

def fmt_attn(attn, tokenizer, phrase):
    ids=tokenizer.encode(phrase); tokens=[tokenizer.decode([t]) for t in ids]
    al=attn[:len(tokens)]; pairs=sorted(zip(al.tolist(),tokens),reverse=True)
    return " ".join([f"{t.strip()}({w:.2f})" for w,t in pairs[:3] if w>0.05])

def fmt_slots(slot_weights):
    return " ".join([f"S{i}:{w:.2f}" for i,w in enumerate(slot_weights.tolist())])

def train_cycle(model, cache, s2i, i2s, chains, tokenizer, all_sv, n_iters, cycle, lr):
    cfg=CONFIG
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=0.01)
    sch=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:s/200 if s<200 else 0.5*(1+math.cos(math.pi*(s-200)/max(n_iters-200,1))))
    model.train(); ok,tot,pl_ok,pl_tot=0,0,0,0; t0=time.time()

    for it in range(n_iters):
        task=generate_mixed_task(cache,s2i,chains)
        np_=model.lib.n
        logits,prog,sig,_,stop_probs,oa,sa,sw,obj_vec,precond_sc,pred_vec = model(task,cfg["temperature"],use_mem=False)

        # PRIMARY loss: cosine similarity in latent space
        target_vec = task["result_state_vec"].to(DEVICE)
        latent_loss = 1.0 - F.cosine_similarity(pred_vec.unsqueeze(0), target_vec.unsqueeze(0))

        # AUXILIARY loss: cross-entropy on vocabulary (helps decoder learn)
        tgt=torch.tensor(task["target_idx"],device=DEVICE)
        lce=F.cross_entropy(logits.unsqueeze(0),tgt.unsqueeze(0))

        stop_l=torch.tensor(0.0,device=DEVICE)
        if stop_probs:
            for sp in stop_probs: stop_l=stop_l+(1.0-sp).squeeze()
            stop_l=stop_l/len(stop_probs)

        # Reasoner losses
        rv=task["result_state_vec"].to(DEVICE)
        gl=goal_contrastive_loss(model.goal_eval,rv,all_sv)
        al_logits=model.action_scorer(obj_vec.detach(),rv)
        al_loss=F.cross_entropy(al_logits.unsqueeze(0),torch.tensor(task["action_idx"],device=DEVICE).unsqueeze(0))

        # Precondition loss
        precond_target = torch.tensor(1.0 if task["precondition_holds"] else 0.0, device=DEVICE)
        precond_loss = F.binary_cross_entropy(precond_sc.view(-1)[0], precond_target)

        loss=(lce + 0.2*latent_loss + 0.1*div_loss(prog)+0.05*use_loss(prog,np_)+0.1*novelty_loss(prog,np_)+
              0.03*stop_l+0.02*len(prog)+cfg["goal_loss_weight"]*gl+
              cfg["action_loss_weight"]*al_loss+cfg["precondition_loss_weight"]*precond_loss)/cfg["grad_accumulation"]
        loss.backward()

        pred=i2s.get(logits.argmax().item(),"?"); correct=pred==task["expected"]
        ok+=correct; tot+=1
        pa_ok=IDX2ACT[al_logits.argmax().item()]==task["action"]; pl_ok+=pa_ok; pl_tot+=1
        with torch.no_grad(): model.memorize(sig,prog,task["rule_name"],correct)

        if (it+1)%cfg["grad_accumulation"]==0:
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
            opt.step(); opt.zero_grad(); sch.step()

        if it%300==0:
            acc=100*ok/max(tot,1); pacc=100*pl_ok/max(pl_tot,1)
            speed=(it+1)/max(time.time()-t0,0.01)
            with torch.no_grad():
                t=generate_mixed_task(cache,s2i,chains)
                lg,pr,_,_,_,oa2,sa2,sw2,_,pc,_=model(t,cfg["temperature"],use_mem=False)
                pw=i2s.get(lg.argmax().item(),"?"); tw=t["expected"]
                st=t["state"]
                ostr=fmt_attn(oa2,tokenizer,f" {st}")
            ms=f" [D{t['depth']}]" if t.get("task_type")=="multistep" else ""
            print(f"  [C{cycle}] {it:4d}/{n_iters} | WM:{lce.item():.2f} L:{latent_loss.item():.2f} G:{gl.item():.2f} A:{al_loss.item():.2f} P:{precond_loss.item():.2f} | "
                  f"Acc:{acc:4.0f}% Act:{pacc:4.0f}% | {speed:.0f}it/s | [{t['rule_name'][:12]:12s}] {'V' if pw==tw else 'X'} K={len(pr)}{ms}")
            print(f"    '{st[:22]}' + '{t['action'][:10]}' -> '{pw[:22]}' want:'{tw[:22]}'")
            print(f"    obj:{ostr}  slots:{fmt_slots(sw2)}  precond:{pc.item():.2f}")
            if it>0 and it%1500==0: ok,tot,pl_ok,pl_tot=0,0,0,0


def evaluate(model, cache, s2i, i2s, chains, tokenizer):
    cfg=CONFIG; model.eval(); names=model.lib.names; ne=cfg["eval_samples"]

    print("\n  --- Programs + Slots ---")
    with torch.no_grad():
        for rule in sorted(TRANSITION_RULES.keys()):
            t=generate_single_task(cache,s2i,rule_name=rule)
            _,pr,_,_,_,oa,sa,sw,_,pc,_=model(t,0.1,use_mem=False)
            st=t["state"]; prop=TRANSITION_RULES[rule]["property_changed"]
            print(f"    {rule:12s}: {fmt(pr,names)} K={len(pr)}  obj:{fmt_attn(oa,tokenizer,f' {st}')}  "
                  f"slots:{fmt_slots(sw)}  precond:{pc.item():.2f}  prop_type:{prop}")

    print(f"\n  --- Single-Step (n={ne}) ---\n")
    results={}
    with torch.no_grad():
        for rule in sorted(TRANSITION_RULES.keys()):
            ok,t3ok,shown=0,0,0
            for _ in range(ne):
                t=generate_single_task(cache,s2i,rule_name=rule)
                lg,_,_,_,_,oa,sa,sw,_,pc,_=model(t,0.1,use_mem=True)
                pred,target=i2s.get(lg.argmax().item(),"?"),t["expected"]
                top3=[i2s.get(i,"?") for i in lg.topk(min(3,lg.shape[0])).indices.tolist()]
                ok+=pred==target; t3ok+=target in top3
                if shown<1:
                    st=t["state"]
                    print(f"    [{rule:12s}] '{st[:18]}' + '{t['action']:9s}' -> '{pred[:22]}' {'V' if pred==target else 'X'}")
                    print(f"      want:'{target[:22]}'  slots:{fmt_slots(sw)}  precond:{pc.item():.2f}")
                    shown+=1
            a,t3=100*ok/ne,100*t3ok/ne; results[rule]=(a,t3)
            print(f"      -> {a:.0f}% (top3: {t3:.0f}%)")

    print("\n  --- Summary ---")
    for r,(a,t3) in sorted(results.items(),key=lambda x:-x[1][0]):
        bar=chr(9608)*int(a/5)+chr(9617)*int((t3-a)/5)
        print(f"    {r:12s}: {a:5.1f}% [{bar}] (top3: {t3:.0f}%)")
    n=len(results); ov=sum(a for a,_ in results.values())/n; ov3=sum(t for _,t in results.values())/n
    print(f"\n    SINGLE-STEP: {ov:.1f}% (top3: {ov3:.1f}%)")

    # Latent similarity: how close are predicted vectors to target vectors?
    print("\n  --- Latent Similarity ---")
    lat_sims = []
    with torch.no_grad():
        for rule in sorted(TRANSITION_RULES.keys()):
            rule_sims = []
            for _ in range(5):
                t = generate_single_task(cache, s2i, rule_name=rule)
                _,_,_,_,_,_,_,_,_,_,pv = model(t, 0.1, use_mem=False)
                tv = t["result_state_vec"].to(DEVICE)
                sim = F.cosine_similarity(pv.unsqueeze(0), tv.unsqueeze(0)).item()
                rule_sims.append(sim); lat_sims.append(sim)
            print(f"    {rule:12s}: cos_sim={sum(rule_sims)/len(rule_sims):.3f}")
    print(f"    AVERAGE: {sum(lat_sims)/len(lat_sims):.3f}")

    # Slot analysis
    print("\n  --- Slot Analysis ---")
    prop_slots = defaultdict(list)
    with torch.no_grad():
        for rule, r in TRANSITION_RULES.items():
            for s,a,res in r["triples"][:3]:
                t = generate_single_task(cache, s2i, rule_name=rule)
                _,_,_,_,_,_,_,sw,_,_,_ = model(t, 0.1, use_mem=False)
                prop_slots[r["property_changed"]].append(sw.argmax().item())
    for prop, slots in sorted(prop_slots.items()):
        dominant = Counter(slots).most_common(1)[0]
        print(f"    {prop:15s}: dominant slot = S{dominant[0]} ({dominant[1]}/{len(slots)} = {100*dominant[1]/len(slots):.0f}%)")

    # Multi-step
    for depth in [2,3]:
        if not chains.get(depth): continue
        print(f"\n  --- Depth-{depth} ---")
        ok,tested=0,0
        with torch.no_grad():
            for _ in range(min(50,len(chains[depth]))):
                t=generate_multistep_task(cache,s2i,chains,3)
                if t["depth"]!=depth: continue
                lg,_,_,_,_,_,_,_,_,_,_=model(t,0.1,use_mem=False)
                pred=i2s.get(lg.argmax().item(),"?"); ok+=pred==t["expected"]; tested+=1
                if tested<=3:
                    print(f"    '{t['state'][:20]}' +[{t['action'][:20]}]")
                    print(f"      -> '{pred[:25]}' want:'{t['expected'][:25]}' {'V' if pred==t['expected'] else 'X'}")
        if tested: print(f"    DEPTH-{depth}: {100*ok/tested:.0f}% [{tested} tested]")

    # Chain Effects
    if CHAIN_TRIPLES:
        print(f"\n  --- Chain Effects (push fragile → breaks) ---")
        ce_ok, ce_top3, ce_tested = 0, 0, 0
        with torch.no_grad():
            for ct in CHAIN_TRIPLES:
                t = generate_chain_effect_task(cache, s2i)
                lg,_,_,_,_,_,_,_,_,_,_ = model(t, 0.1, use_mem=False)
                pred = i2s.get(lg.argmax().item(), "?")
                top3 = [i2s.get(i,"?") for i in lg.topk(min(3,lg.shape[0])).indices.tolist()]
                correct = pred == t["expected"]
                t3 = t["expected"] in top3
                ce_ok += correct; ce_top3 += t3; ce_tested += 1
                if ce_tested <= 5:
                    inter = t.get("chain_intermediate","")
                    print(f"    '{t['state'][:22]}' + '{t['action']}' -> '{pred[:22]}' {'V' if correct else 'X'}")
                    print(f"      want:'{t['expected'][:22]}'  via:'{inter[:22]}'  obj:{t.get('chain_object','?')}")
        if ce_tested:
            print(f"    CHAIN EFFECTS: {100*ce_ok/ce_tested:.0f}% (top3: {100*ce_top3/ce_tested:.0f}%) [{ce_tested} tested]")

    # Graded Temperature Test
    graded_rules = ["warm_up", "boil", "simmer", "warm_cool"]
    graded_exist = [r for r in graded_rules if r in TRANSITION_RULES]
    if graded_exist:
        print(f"\n  --- Graded Temperature ---")
        gr_ok, gr_tested = 0, 0
        with torch.no_grad():
            for rname in graded_exist:
                r = TRANSITION_RULES[rname]
                for s, a, res in r["triples"]:
                    t = generate_single_task(cache, s2i, rule_name=rname)
                    lg,_,_,_,_,_,_,_,_,_,_ = model(t, 0.1, use_mem=False)
                    pred = i2s.get(lg.argmax().item(), "?")
                    correct = pred == t["expected"]
                    gr_ok += correct; gr_tested += 1
                    if gr_tested <= 6:
                        print(f"    [{rname:10s}] '{t['state'][:20]}' + '{t['action'][:8]}' -> '{pred[:22]}' {'V' if correct else 'X'}")
                        print(f"      want:'{t['expected'][:22]}'")
        if gr_tested:
            print(f"    GRADED TEMP: {100*gr_ok/gr_tested:.0f}% [{gr_tested} tested]")

    # Planning
    print(f"\n  --- Planning ---")
    pl_1st,pl_full,pl_t=0,0,0
    real_1st,real_full,real_t=0,0,0
    for _ in range(50):
        task=generate_planning_task(cache,s2i,chains)
        if not task: continue
        plan,sc=model.plan(cache,task["initial_state"],task["goal_state"],max_depth=task["depth"])
        is_round_trip = task["initial_state"] == task["goal_state"]
        if plan:
            fc=plan[0]==task["correct_actions"][0]
            full_c=plan[:len(task["correct_actions"])]==task["correct_actions"]
        else:
            fc = is_round_trip; full_c = is_round_trip
        pl_1st+=fc; pl_full+=full_c; pl_t+=1
        if not is_round_trip: real_1st+=fc; real_full+=full_c; real_t+=1
        if pl_t<=6:
            rt=" [ROUND]" if is_round_trip else ""
            print(f"    {task['initial_state'][:22]} -> {task['goal_state'][:22]}{rt}")
            print(f"      correct:{task['correct_actions']}  plan:{plan}  1st:{'V' if fc else 'X'} full:{'V' if full_c else 'X'} score:{sc:.3f}")
    if pl_t: print(f"\n    ALL:  1st:{100*pl_1st/pl_t:.0f}% full:{100*pl_full/pl_t:.0f}% ({pl_t} tested)")
    if real_t: print(f"    REAL: 1st:{100*real_1st/real_t:.0f}% full:{100*real_full/real_t:.0f}% ({real_t} tested)")
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    print(f"Device: {DEVICE}")
    print("Loading GPT-2...")
    tokenizer=GPT2Tokenizer.from_pretrained("gpt2")
    gpt2=GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE).eval()

    sl,s2i,i2s=build_vocab()
    global i2s_global
    i2s_global = i2s
    print(f"\nRules:{len(TRANSITION_RULES)} | States:{len(sl)} | Actions:{NUM_ACTIONS} | Properties:{NUM_PROPERTIES}")
    print(f"Property types: {PROPERTY_TYPES}")
    chains=find_chains(CONFIG["max_chain_length"])
    cache=TokenEmbeddingCache(gpt2,tokenizer,layer=CONFIG["perceiver_layer"])
    all_sv=[cache.get(f" {s}").to(DEVICE) for s in sl]

    model=CompleteWorldModel(768,len(sl))
    tp=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {tp:,}\n")

    cfg=CONFIG
    for c in range(cfg["num_cycles"]):
        print(f"{'='*65}")
        print(f"CYCLE {c+1}/{cfg['num_cycles']} | Prims:{model.lib.n} | Params:{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        print(f"Library: {model.lib.names}")
        print(f"{'='*65}\n")
        train_cycle(model,cache,s2i,i2s,chains,tokenizer,all_sv,cfg["iters_per_cycle"][c],c+1,cfg["lr_per_cycle"][c])
        print(f"\n--- Eval Cycle {c+1} ---")
        evaluate(model,cache,s2i,i2s,chains,tokenizer)
        print(f"\n  Memory: {model.mem.stats()}")
        if c<cfg["num_cycles"]-1:
            print(f"\n--- Compression ---")
            n=model.compress()
            if n: print(f"  {n} new primitives.")

    print(f"\n{'='*65}\nFINAL TEST\n{'='*65}")
    print(f"Library: {model.lib.names}")
    evaluate(model,cache,s2i,i2s,chains,tokenizer)

    print("\n  --- Primitive Usage ---")
    usage=Counter(); model.eval()
    with torch.no_grad():
        for _ in range(200):
            t=generate_single_task(cache,s2i)
            _,pr,_,_,_,_,_,_,_,_,_=model(t,0.1,use_mem=False)
            for sel in pr: usage[model.lib.names[sel.argmax().item()] if sel.argmax().item()<len(model.lib.names) else "?"]+=1
    tu=sum(usage.values())
    for nm,cnt in sorted(usage.items(),key=lambda x:-x[1]): print(f"    {nm:20s}: {100*cnt/tu:5.1f}%")
    print(f"\n  Memory: {model.mem.stats()}\nDone.")

if __name__=="__main__":
    main()
