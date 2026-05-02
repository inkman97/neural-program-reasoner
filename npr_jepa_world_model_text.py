"""
Neural Program Reasoner — World Model v2: Temporal Understanding

Extends the original World Model with three types of temporal effects:

1. DELAYED EFFECTS: "put in oven" starts cooking; "wait" steps complete it
2. NATURAL DECAY: battery drains, candle burns out over time
3. ACTION DURATION: "charge" takes 3 wait steps to reach full

Key insight: temporal progress is encoded IN THE STATE ("cooking_1", "cooking_2")
so the existing World Model architecture handles it without modification.
The "wait" action learns context-dependent transitions — this is the core
temporal reasoning challenge.

New objects: phone_t/laptop/tablet (battery), bread/cake/chicken (cooking),
            tomato/flower/herb (growth), candle/fire/torch (burning)
New actions: charge, put in oven, plant, light, wait
Wait transitions: 11 context-dependent effects

Requirements:
    pip install torch transformers
"""

import math
import random
import time
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
i2s_global = {}

CONFIG = {
    "proj_dim": 256,
    "num_slots": 2,
    "max_program_steps": 6,
    "min_program_steps": 2,
    "temperature": 0.8,
    "grad_accumulation": 16,
    "memory_capacity": 300,
    "compression_threshold": 6,
    "max_new_primitives": 2,
    "num_cycles": 3,
    "iters_per_cycle": [15000, 10000, 5000],
    "lr_per_cycle": [5e-4, 3e-4, 1e-4],
    "perceiver_layer": 8,
    "eval_samples": 40,
    "max_chain_length": 3,
    "multistep_ratio": 0.30,
    "chain_effect_ratio": 0.08,
    "temporal_ratio": 0.15,
    "goal_loss_weight": 0.2,
    "action_loss_weight": 0.2,
    "precondition_loss_weight": 0.15,
}

# =============================================================================
# World Definition: Original + Temporal Objects/Actions
# =============================================================================

PROPERTY_DEFS = {
    "temperature": ["cold", "warm", "hot", "boiling"],
    "openness": ["closed", "open"],
    "power": ["off", "on"],
    "fullness": ["empty", "full"],
    "integrity": ["intact", "broken"],
    "position": "on the {surface} -> on the floor",
    "containment": "outside the {container} -> inside the {container}",
    "battery": ["empty", "charging_1", "charging_2", "full", "half"],
    "cooking_state": ["raw", "cooking_1", "cooking_2", "cooked", "burnt"],
    "growth": ["seed", "growing_1", "growing_2", "grown"],
    "burn_state": ["unlit", "burning", "dim", "extinguished"],
}

OBJECTS = {
    # Original objects
    "water": {"temperature": True}, "soup": {"temperature": True},
    "coffee": {"temperature": True}, "tea": {"temperature": True},
    "milk": {"temperature": True}, "food": {"temperature": True},
    "oil": {"temperature": True}, "pan": {"temperature": True},
    "iron": {"temperature": True}, "metal": {"temperature": True},
    "door": {"openness": True}, "window": {"openness": True},
    "box": {"openness": True}, "jar": {"openness": True},
    "gate": {"openness": True}, "drawer": {"openness": True},
    "lid": {"openness": True}, "bag": {"openness": True},
    "bottle": {"openness": True}, "cabinet": {"openness": True},
    "lamp": {"power": True, "surface": "table", "fragile": True, "integrity": True},
    "light": {"power": True}, "screen": {"power": True},
    "tv": {"power": True}, "radio": {"power": True},
    "computer": {"power": True}, "fan": {"power": True},
    "heater": {"power": True}, "oven": {"power": True},
    "speaker": {"power": True},
    "cup": {"fullness": True, "surface": "shelf", "fragile": True, "integrity": True},
    "glass": {"fullness": True, "surface": "table", "fragile": True, "integrity": True},
    "bowl": {"fullness": True, "fragile": True, "integrity": True},
    "pot": {"fullness": True}, "tub": {"fullness": True},
    "bucket": {"fullness": True}, "jug": {"fullness": True},
    "tank": {"fullness": True}, "pool": {"fullness": True},
    "plate": {"integrity": True, "surface": "counter", "fragile": True},
    "vase": {"integrity": True, "fragile": True, "surface": "shelf"},
    "mirror": {"integrity": True},
    "ball": {"surface": "table"}, "book": {"surface": "desk"},
    "phone": {"surface": "bed"}, "toy": {"surface": "shelf"},
    "pen": {"surface": "desk"}, "remote": {"surface": "couch"},
    "key": {"container": "pocket"}, "coin": {"container": "jar"},
    "letter": {"container": "envelope"}, "shirt": {"container": "closet"},
    "tool": {"container": "shed"}, "food_c": {"container": "fridge"},
    "ball_c": {"container": "box"}, "toy_c": {"container": "bag"},
    "pen_c": {"container": "case"}, "book_c": {"container": "drawer"},
    # Temporal objects
    "phone_t": {"battery": True}, "laptop": {"battery": True}, "tablet": {"battery": True},
    "bread": {"cooking_state": True}, "cake": {"cooking_state": True}, "chicken": {"cooking_state": True},
    "tomato": {"growth": True}, "flower": {"growth": True}, "herb": {"growth": True},
    "candle": {"burn_state": True}, "fire": {"burn_state": True}, "torch": {"burn_state": True},
}

ACTIONS = {
    # Original actions
    "heat": {"property": "temperature", "from": "cold", "to": "hot"},
    "cool": {"property": "temperature", "from": "hot", "to": "cold"},
    "warm up": {"property": "temperature", "from": "cold", "to": "warm"},
    "boil": {"property": "temperature", "from": "hot", "to": "boiling"},
    "simmer": {"property": "temperature", "from": "boiling", "to": "hot"},
    "open": {"property": "openness", "from": "closed", "to": "open"},
    "close": {"property": "openness", "from": "open", "to": "closed"},
    "switch on": {"property": "power", "from": "off", "to": "on"},
    "switch off": {"property": "power", "from": "on", "to": "off"},
    "fill": {"property": "fullness", "from": "empty", "to": "full"},
    "drop": {"property": "integrity", "from": "intact", "to": "broken"},
    "push": {"property": "position", "from": "surface", "to": "floor"},
    "put in": {"property": "containment", "from": "outside", "to": "inside"},
    # Temporal initiating actions
    "charge": {"property": "battery", "from": "empty", "to": "charging_1"},
    "put in oven": {"property": "cooking_state", "from": "raw", "to": "cooking_1"},
    "plant": {"property": "growth", "from": "seed", "to": "growing_1"},
    "light": {"property": "burn_state", "from": "unlit", "to": "burning"},
    # Meta-action for temporal progression
    "wait": {"property": "temporal", "from": "any", "to": "next"},
}

WAIT_TRANSITIONS = {
    ("battery", "charging_1"): "charging_2", ("battery", "charging_2"): "full",
    ("battery", "full"): "half", ("battery", "half"): "empty",
    ("cooking_state", "cooking_1"): "cooking_2", ("cooking_state", "cooking_2"): "cooked",
    ("cooking_state", "cooked"): "burnt",
    ("growth", "growing_1"): "growing_2", ("growth", "growing_2"): "grown",
    ("burn_state", "burning"): "dim", ("burn_state", "dim"): "extinguished",
}

TEMPORAL_PROPERTIES = {"battery", "cooking_state", "growth", "burn_state"}


def generate_world_observations():
    observations = []
    rule_groups = defaultdict(list)
    for obj_name, obj_props in OBJECTS.items():
        for act_name, act_def in ACTIONS.items():
            prop = act_def["property"]
            if act_name == "wait": continue
            if prop == "temperature" and not obj_props.get("temperature"): continue
            if prop == "openness" and not obj_props.get("openness"): continue
            if prop == "power" and not obj_props.get("power"): continue
            if prop == "fullness" and not obj_props.get("fullness"): continue
            if prop == "integrity" and not obj_props.get("integrity"): continue
            if prop == "position" and "surface" not in obj_props: continue
            if prop == "containment" and "container" not in obj_props: continue
            if prop == "battery" and not obj_props.get("battery"): continue
            if prop == "cooking_state" and not obj_props.get("cooking_state"): continue
            if prop == "growth" and not obj_props.get("growth"): continue
            if prop == "burn_state" and not obj_props.get("burn_state"): continue
            display = obj_name.split("_")[0]
            if prop == "position":
                state = f"the {display} is on the {obj_props['surface']}";
                result = f"the {display} is on the floor"
            elif prop == "containment":
                state = f"the {display} is outside the {obj_props['container']}";
                result = f"the {display} is inside the {obj_props['container']}"
            else:
                state = f"the {display} is {act_def['from']}";
                result = f"the {display} is {act_def['to']}"
            obs = {"state": state, "action": act_name, "result": result, "property": prop, "object": display,
                   "fragile": obj_props.get("fragile", False), "temporal": prop in TEMPORAL_PROPERTIES}
            observations.append(obs);
            rule_groups[act_name].append(obs)
    # Generate wait transitions
    for obj_name, obj_props in OBJECTS.items():
        display = obj_name.split("_")[0]
        for (prop, from_val), to_val in WAIT_TRANSITIONS.items():
            if not obj_props.get(prop): continue
            state = f"the {display} is {from_val}";
            result = f"the {display} is {to_val}"
            obs = {"state": state, "action": "wait", "result": result, "property": prop,
                   "object": display, "fragile": False, "temporal": True}
            observations.append(obs);
            rule_groups["wait"].append(obs)
    return observations, rule_groups


ALL_OBSERVATIONS, RULE_GROUPS = generate_world_observations()
TRANSITION_RULES = {}
for act_name, obs_list in RULE_GROUPS.items():
    triples = [(o["state"], o["action"], o["result"]) for o in obs_list]
    if triples:
        first = obs_list[0]
        precond = "any" if act_name == "wait" else ACTIONS[act_name]["from"].split()[0]
        TRANSITION_RULES[act_name] = {"triples": triples, "precondition": precond,
                                      "property_changed": first["property"]}

CHAIN_EFFECTS = [{"trigger_action": "push", "chain_action": "drop", "condition": "fragile"}]
CHAIN_TRIPLES = []
for ce in CHAIN_EFFECTS:
    trigger_obs = RULE_GROUPS.get(ce["trigger_action"], [])
    chain_obs = RULE_GROUPS.get(ce["chain_action"], [])
    chain_by_obj = {o["object"]: o for o in chain_obs}
    for obs in trigger_obs:
        if obs.get(ce["condition"], False) and obs["object"] in chain_by_obj:
            co = chain_by_obj[obs["object"]]
            CHAIN_TRIPLES.append({"state": obs["state"], "action": obs["action"], "result": co["result"],
                                  "intermediate": obs["result"], "rules": [ce["trigger_action"], ce["chain_action"]],
                                  "object": obs["object"]})

# Build temporal chains
TEMPORAL_CHAINS = []
for obj_name, obj_props in OBJECTS.items():
    display = obj_name.split("_")[0]
    if obj_props.get("battery"):
        TEMPORAL_CHAINS.append({"states": [f"the {display} is empty", f"the {display} is charging_1",
                                           f"the {display} is charging_2", f"the {display} is full"],
                                "actions": ["charge", "wait", "wait"], "type": "duration", "property": "battery",
                                "depth": 3})
        TEMPORAL_CHAINS.append(
            {"states": [f"the {display} is full", f"the {display} is half", f"the {display} is empty"],
             "actions": ["wait", "wait"], "type": "decay", "property": "battery", "depth": 2})
    if obj_props.get("cooking_state"):
        TEMPORAL_CHAINS.append({"states": [f"the {display} is raw", f"the {display} is cooking_1",
                                           f"the {display} is cooking_2", f"the {display} is cooked"],
                                "actions": ["put in oven", "wait", "wait"], "type": "delayed",
                                "property": "cooking_state", "depth": 3})
        TEMPORAL_CHAINS.append({"states": [f"the {display} is cooked", f"the {display} is burnt"],
                                "actions": ["wait"], "type": "decay", "property": "cooking_state", "depth": 1})
    if obj_props.get("growth"):
        TEMPORAL_CHAINS.append({"states": [f"the {display} is seed", f"the {display} is growing_1",
                                           f"the {display} is growing_2", f"the {display} is grown"],
                                "actions": ["plant", "wait", "wait"], "type": "delayed", "property": "growth",
                                "depth": 3})
    if obj_props.get("burn_state"):
        TEMPORAL_CHAINS.append({"states": [f"the {display} is unlit", f"the {display} is burning",
                                           f"the {display} is dim", f"the {display} is extinguished"],
                                "actions": ["light", "wait", "wait"], "type": "duration_decay",
                                "property": "burn_state", "depth": 3})

print(
    f"World: {len(ALL_OBSERVATIONS)} observations, {len(RULE_GROUPS)} actions, {len(CHAIN_TRIPLES)} chain effects, {len(TEMPORAL_CHAINS)} temporal chains")
for act in sorted(RULE_GROUPS):
    obs = RULE_GROUPS[act]
    temporal_mark = " [TEMPORAL]" if any(o.get("temporal") for o in obs) else ""
    print(f"  {act:12s}: {len(obs):2d} obs ({obs[0]['property']}){temporal_mark}")

ALL_ACTIONS = sorted(set(a for r in TRANSITION_RULES.values() for _, a, _ in r["triples"]))
ACT2IDX = {a: i for i, a in enumerate(ALL_ACTIONS)}
IDX2ACT = {i: a for a, i in ACT2IDX.items()}
NUM_ACTIONS = len(ALL_ACTIONS)
PROPERTY_TYPES = sorted(set(r["property_changed"] for r in TRANSITION_RULES.values()))
PROP2IDX = {p: i for i, p in enumerate(PROPERTY_TYPES)}
NUM_PROPERTIES = len(PROPERTY_TYPES)


def build_vocab():
    ss = set()
    for r in TRANSITION_RULES.values():
        for s, a, res in r["triples"]: ss.add(s); ss.add(res)
    for ct in CHAIN_TRIPLES: ss.add(ct["state"]); ss.add(ct["result"]); ss.add(ct["intermediate"])
    for tc in TEMPORAL_CHAINS:
        for s in tc["states"]: ss.add(s)
    sl = sorted(ss)
    return sl, {s: i for i, s in enumerate(sl)}, {i: s for i, s in enumerate(sl)}


def find_chains(max_len=3):
    st = defaultdict(list)
    for rn, r in TRANSITION_RULES.items():
        for s, a, res in r["triples"]: st[s].append((a, res, rn))
    chains = {1: [], 2: [], 3: []}
    for rn, r in TRANSITION_RULES.items():
        for s, a, res in r["triples"]: chains[1].append({"states": [s, res], "actions": [a], "rules": [rn], "depth": 1})
    for c1 in chains[1]:
        for a2, s2, r2 in st.get(c1["states"][-1], []):
            if r2 != c1["rules"][-1]: chains[2].append(
                {"states": c1["states"] + [s2], "actions": c1["actions"] + [a2], "rules": c1["rules"] + [r2],
                 "depth": 2})
    if max_len >= 3:
        for c2 in chains[2]:
            for a3, s3, r3 in st.get(c2["states"][-1], []):
                if r3 != c2["rules"][-1]: chains[3].append(
                    {"states": c2["states"] + [s3], "actions": c2["actions"] + [a3], "rules": c2["rules"] + [r3],
                     "depth": 3})
    print(f"\nChains: d1:{len(chains[1])}, d2:{len(chains[2])}, d3:{len(chains[3])}, temporal:{len(TEMPORAL_CHAINS)}")
    return chains


class WorldState:
    def __init__(self):
        self.objects = {};
        self.history = [];
        self.time = 0

    def set(self, obj, state_text): self.objects[obj] = state_text

    def apply_action(self, obj, action, new_state):
        old = self.objects.get(obj, "unknown");
        self.history.append((self.time, obj, action, old, new_state));
        self.objects[obj] = new_state;
        self.time += 1

    def get(self, obj): return self.objects.get(obj, "unknown")

    def snapshot(self): return dict(self.objects)


class TokenEmbeddingCache:
    def __init__(self, gpt2, tokenizer, layer=8):
        self.last_cache, self.tokens_cache = {}, {}
        self.device = next(gpt2.parameters()).device
        self.gpt2, self.tokenizer, self.layer = gpt2, tokenizer, layer
        print("\nPre-computing representations...")
        t0 = time.time()
        phrases = set()
        for r in TRANSITION_RULES.values():
            for s, a, res in r["triples"]: phrases.add(f" {s}"); phrases.add(f" {a}"); phrases.add(f" {res}")
        for r in TRANSITION_RULES.values():
            phrases.add(f" {r['precondition']}");
            phrases.add(f" {r['property_changed']}")
        for ct in CHAIN_TRIPLES:
            phrases.add(f" {ct['state']}");
            phrases.add(f" {ct['result']}");
            phrases.add(f" {ct['intermediate']}")
        for tc in TEMPORAL_CHAINS:
            for s in tc["states"]: phrases.add(f" {s}")
            for a in tc["actions"]: phrases.add(f" {a}")
        print(f"  {len(phrases)} phrases...")
        gpt2.eval()
        with torch.no_grad():
            for p in sorted(phrases):
                ids = tokenizer.encode(p)
                out = gpt2(torch.tensor([ids], device=self.device), output_hidden_states=True)
                h = out.hidden_states[layer][0]
                self.last_cache[p] = h[-1].clone();
                self.tokens_cache[p] = h.clone()
        print(f"  Done! {len(self.last_cache)} cached in {time.time() - t0:.1f}s")

    def get(self, phrase):
        if phrase in self.last_cache: return self.last_cache[phrase]
        ids = self.tokenizer.encode(phrase)
        with torch.no_grad():
            out = self.gpt2(torch.tensor([ids], device=self.device), output_hidden_states=True)
        h = out.hidden_states[self.layer][0]
        self.last_cache[phrase] = h[-1].clone();
        self.tokens_cache[phrase] = h.clone()
        return self.last_cache[phrase]

    def get_tokens(self, phrase):
        if phrase not in self.tokens_cache: self.get(phrase)
        return self.tokens_cache[phrase]


# =============================================================================
# Tasks (with temporal)
# =============================================================================

def generate_single_task(cache, s2i, num_examples=5, rule_name=None):
    rule_name = rule_name or random.choice(list(TRANSITION_RULES.keys()))
    r = TRANSITION_RULES[rule_name];
    ts = r["triples"]
    ne = min(num_examples, len(ts) - 1)
    chosen = random.sample(ts, ne + 1);
    examples, test = chosen[:ne], chosen[ne]
    return {
        "example_reprs": torch.stack(
            [torch.cat([cache.get(f" {s}"), cache.get(f" {a}"), cache.get(f" {res}")]) for s, a, res in examples]),
        "test_state": cache.get(f" {test[0]}"), "test_action": cache.get(f" {test[1]}"),
        "test_state_tokens": cache.get_tokens(f" {test[0]}"),
        "target_idx": s2i.get(test[2], 0), "rule_name": rule_name,
        "state": test[0], "action": test[1], "expected": test[2],
        "depth": 1, "task_type": "single",
        "result_state_vec": cache.get(f" {test[2]}"), "action_idx": ACT2IDX[test[1]],
        "precondition": r["precondition"], "property_changed": r["property_changed"],
        "precondition_holds": r["precondition"] in test[0] or r["precondition"] == "any",
        "property_idx": PROP2IDX.get(r["property_changed"], 0),
    }


def generate_multistep_task(cache, s2i, chains, num_examples=3):
    avail = [d for d in [2, 3] if chains.get(d)]
    if not avail: return generate_single_task(cache, s2i)
    depth = random.choice(avail);
    pool = chains[depth]
    ne = min(num_examples, len(pool) - 1)
    chosen = random.sample(pool, ne + 1);
    examples, test = chosen[:ne], chosen[ne]
    er = [torch.cat([cache.get(f" {c['states'][0]}"), torch.stack([cache.get(f" {a}") for a in c["actions"]]).mean(0),
                     cache.get(f" {c['states'][-1]}")]) for c in examples]
    ta = [cache.get(f" {a}") for a in test["actions"]]
    first_rule = TRANSITION_RULES[test["rules"][0]]
    per_step_examples = []
    for rule_name in test["rules"]:
        r = TRANSITION_RULES[rule_name]
        step_triples = random.sample(r["triples"], min(3, len(r["triples"])))
        step_ex = torch.stack(
            [torch.cat([cache.get(f" {s}"), cache.get(f" {a}"), cache.get(f" {res}")]) for s, a, res in step_triples])
        per_step_examples.append(step_ex)
    return {
        "example_reprs": torch.stack(er), "test_state": cache.get(f" {test['states'][0]}"),
        "test_action": torch.stack(ta).mean(0), "test_actions": ta,
        "test_state_tokens": cache.get_tokens(f" {test['states'][0]}"),
        "target_idx": s2i.get(test["states"][-1], 0), "rule_name": "+".join(test["rules"]),
        "state": test["states"][0], "action": "+".join(test["actions"]), "expected": test["states"][-1],
        "depth": depth, "task_type": "multistep",
        "result_state_vec": cache.get(f" {test['states'][-1]}"), "action_idx": ACT2IDX[test["actions"][0]],
        "precondition": first_rule["precondition"], "property_changed": first_rule["property_changed"],
        "precondition_holds": True, "property_idx": PROP2IDX.get(first_rule["property_changed"], 0),
        "per_step_examples": per_step_examples, "step_rules": test["rules"],
    }


def generate_chain_effect_task(cache, s2i, num_examples=3):
    if not CHAIN_TRIPLES: return generate_single_task(cache, s2i)
    ct = random.choice(CHAIN_TRIPLES)
    gravity_triples = TRANSITION_RULES["push"]["triples"]
    break_triples = TRANSITION_RULES["drop"]["triples"]
    gex = random.sample(gravity_triples, min(2, len(gravity_triples)))
    bex = random.sample(break_triples, min(2, len(break_triples)))
    examples = gex + bex
    example_reprs = torch.stack(
        [torch.cat([cache.get(f" {s}"), cache.get(f" {a}"), cache.get(f" {r}")]) for s, a, r in examples])
    return {
        "example_reprs": example_reprs, "test_state": cache.get(f" {ct['state']}"),
        "test_action": cache.get(f" {ct['action']}"), "test_state_tokens": cache.get_tokens(f" {ct['state']}"),
        "target_idx": s2i.get(ct["result"], 0), "rule_name": "chain:" + "+".join(ct["rules"]),
        "state": ct["state"], "action": ct["action"], "expected": ct["result"],
        "depth": 1, "task_type": "chain_effect", "result_state_vec": cache.get(f" {ct['result']}"),
        "action_idx": ACT2IDX[ct["action"]], "precondition": "on the", "property_changed": "position",
        "precondition_holds": True, "property_idx": PROP2IDX["position"],
        "chain_intermediate": ct["intermediate"], "chain_object": ct["object"],
    }


def generate_temporal_task(cache, s2i, num_examples=3):
    """Generate a temporal task: a single step within a temporal chain.
    The model sees examples of the SAME temporal transition and must predict the next state."""
    if not TEMPORAL_CHAINS: return generate_single_task(cache, s2i)
    tc = random.choice(TEMPORAL_CHAINS)
    # Pick a random step within the chain
    step_idx = random.randint(0, len(tc["actions"]) - 1)
    state = tc["states"][step_idx]
    action = tc["actions"][step_idx]
    result = tc["states"][step_idx + 1]
    # Examples: other objects with the same temporal transition
    rule = TRANSITION_RULES.get(action)
    if not rule or len(rule["triples"]) < 2:
        return generate_single_task(cache, s2i)
    # Filter examples to same action, different object
    other_triples = [(s, a, r) for s, a, r in rule["triples"] if s != state]
    if len(other_triples) < 2: other_triples = rule["triples"]
    ne = min(num_examples, len(other_triples))
    chosen = random.sample(other_triples, ne)
    example_reprs = torch.stack(
        [torch.cat([cache.get(f" {s}"), cache.get(f" {a}"), cache.get(f" {res}")]) for s, a, res in chosen])
    return {
        "example_reprs": example_reprs, "test_state": cache.get(f" {state}"),
        "test_action": cache.get(f" {action}"), "test_state_tokens": cache.get_tokens(f" {state}"),
        "target_idx": s2i.get(result, 0), "rule_name": f"temporal:{action}",
        "state": state, "action": action, "expected": result,
        "depth": 1, "task_type": "temporal",
        "result_state_vec": cache.get(f" {result}"), "action_idx": ACT2IDX.get(action, 0),
        "precondition": "any", "property_changed": tc["property"],
        "precondition_holds": True, "property_idx": PROP2IDX.get(tc["property"], 0),
        "temporal_type": tc["type"], "temporal_depth": tc["depth"],
    }


def generate_mixed_task(cache, s2i, chains, ne=5):
    r = random.random()
    if r < CONFIG["temporal_ratio"] and TEMPORAL_CHAINS:
        return generate_temporal_task(cache, s2i)
    if r < CONFIG["temporal_ratio"] + CONFIG["chain_effect_ratio"] and CHAIN_TRIPLES:
        return generate_chain_effect_task(cache, s2i)
    if r < CONFIG["temporal_ratio"] + CONFIG["chain_effect_ratio"] + CONFIG["multistep_ratio"] and any(
            chains.get(d) for d in [2, 3]):
        return generate_multistep_task(cache, s2i, chains, 3)
    return generate_single_task(cache, s2i, ne)


def generate_planning_task(cache, s2i, chains):
    avail = [d for d in [1, 2, 3] if chains.get(d)]
    if not avail: return None
    c = random.choice(chains[random.choice(avail)])
    return {"initial_state": c["states"][0], "goal_state": c["states"][-1], "correct_actions": c["actions"],
            "depth": len(c["actions"])}


# =============================================================================
# PERCEIVER + WORLD MODEL (identical to original)
# =============================================================================

class SlotAttention(nn.Module):
    def __init__(self, sd, num_slots=2, n_iters=2):
        super().__init__();
        self.num_slots = num_slots;
        self.n_iters = n_iters;
        self.sd = sd;
        self.slot_dim = sd
        self.slot_mu = nn.Parameter(torch.randn(num_slots, sd) * 0.02);
        self.k_proj = nn.Linear(sd, sd);
        self.v_proj = nn.Linear(sd, sd);
        self.q_proj = nn.Linear(sd, sd)
        self.gru = nn.GRUCell(sd, sd);
        self.mlp = nn.Sequential(nn.Linear(sd, sd * 2), nn.GELU(), nn.Linear(sd * 2, sd))
        self.norm1 = nn.LayerNorm(sd);
        self.norm2 = nn.LayerNorm(sd);
        self.scale = math.sqrt(sd)

    def forward(self, token_reprs):
        k = self.k_proj(token_reprs);
        v = self.v_proj(token_reprs);
        slots = self.slot_mu.clone()
        for _ in range(self.n_iters):
            sp = slots;
            slots = self.norm1(slots);
            q = self.q_proj(slots)
            al = torch.matmul(q, k.T) / self.scale;
            attn = F.softmax(al, dim=0)
            aw = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8);
            updates = torch.matmul(aw, v)
            slots = self.gru(updates.reshape(-1, self.sd), sp.reshape(-1, self.sd))
            slots = slots.reshape(self.num_slots, self.sd);
            slots = slots + self.mlp(self.norm2(slots))
        return slots, attn


class ObjectExtractor(nn.Module):
    def __init__(self, sd):
        super().__init__();
        self.query = nn.Parameter(torch.randn(sd) * 0.02)
        self.k = nn.Linear(sd, sd);
        self.v = nn.Linear(sd, sd)
        self.out = nn.Sequential(nn.Linear(sd, sd), nn.LayerNorm(sd));
        self.scale = math.sqrt(sd)

    def forward(self, tokens):
        k, v = self.k(tokens), self.v(tokens);
        a = F.softmax(torch.matmul(k, self.query) / self.scale, dim=0)
        return self.out(torch.matmul(a.unsqueeze(0), v).squeeze(0)), a


BASE_NAMES = ["IDENTITY", "NEGATE", "MORPH", "ASSOCIATE", "LOOKUP", "BLEND"]


class PrimitiveLibrary(nn.Module):
    def __init__(self, sd):
        super().__init__();
        self.sd = sd;
        self.names = list(BASE_NAMES)
        self.identity = nn.Sequential(nn.Linear(sd, sd), nn.Tanh())
        self.negate_v = nn.Parameter(torch.randn(sd) * 0.1)
        self.negate_pre = nn.Linear(sd, sd);
        self.negate_post = nn.Linear(sd, sd)
        self.morph = nn.Sequential(nn.Linear(sd, sd), nn.LayerNorm(sd), nn.GELU(), nn.Linear(sd, sd))
        self.aq = nn.Linear(sd, sd // 4);
        self.ak = nn.Linear(sd, sd // 4);
        self.av = nn.Linear(sd, sd);
        self.ao = nn.Linear(sd, sd)
        self.lookup = nn.Sequential(nn.Linear(sd, sd * 2), nn.GELU(), nn.Linear(sd * 2, sd * 2), nn.GELU(),
                                    nn.Linear(sd * 2, sd))
        self.bc = nn.Parameter(torch.randn(sd) * 0.01);
        self.bn = nn.Sequential(nn.Linear(sd * 2, sd), nn.GELU(), nn.Linear(sd, sd))
        self.gates = nn.ParameterList([nn.Parameter(torch.tensor(g)) for g in [0.01, 0.5, 0.3, 0.4, 0.5, 0.3]])
        self.inv_p = nn.ModuleList();
        self.inv_g = nn.ParameterList()

    @property
    def n(self):
        return len(BASE_NAMES) + len(self.inv_p)

    def _householder(self, x):
        v = self.negate_v;
        x_proj = self.negate_pre(x)
        vv = torch.dot(v, v) + 1e-8;
        coeff = 2 * torch.dot(v, x_proj) / vv
        return self.negate_post(x_proj - coeff * v)

    def _base(self, i, s):
        if i == 0: return s + torch.sigmoid(self.gates[0]) * self.identity(s)
        if i == 1: return s + torch.sigmoid(self.gates[1]) * self._householder(s)
        if i == 2: return s + torch.sigmoid(self.gates[2]) * self.morph(s)
        if i == 3:
            q, k, v = self.aq(s), self.ak(s), self.av(s)
            return s + torch.sigmoid(self.gates[3]) * self.ao(
                torch.sigmoid(torch.dot(q, k) / math.sqrt(q.shape[0])) * v)
        if i == 4: return s + torch.sigmoid(self.gates[4]) * self.lookup(s)
        if i == 5: return s + torch.sigmoid(self.gates[5]) * self.bn(torch.cat([s, self.bc]))

    def apply(self, i, s):
        if i < len(BASE_NAMES): return self._base(i, s)
        j = i - len(BASE_NAMES);
        return s + torch.sigmoid(self.inv_g[j]) * self.inv_p[j](s)

    def apply_soft(self, w, s):
        return sum(w[i] * self.apply(i, s) for i in range(self.n))

    def add(self, name, a, b):
        sd, dev = self.sd, next(self.parameters()).device
        self.inv_p.append(
            nn.Sequential(nn.Linear(sd, sd * 2), nn.LayerNorm(sd * 2), nn.GELU(), nn.Linear(sd * 2, sd * 2), nn.GELU(),
                          nn.Linear(sd * 2, sd)).to(dev))
        self.inv_g.append(nn.Parameter(torch.tensor(0.4, device=dev)));
        self.names.append(name);
        return self.n - 1


class SlotSelector(nn.Module):
    def __init__(self, ad, ns):
        super().__init__();
        self.net = nn.Sequential(nn.Linear(ad, 128), nn.GELU(), nn.Linear(128, ns))

    def forward(self, a): return F.softmax(self.net(a), dim=-1)


class PropertyUpdater(nn.Module):
    def __init__(self, sd, ad, ms):
        super().__init__();
        self.step_emb = nn.Embedding(ms, sd)
        self.action_film = nn.Sequential(nn.Linear(ad, sd * 2), nn.GELU(), nn.Linear(sd * 2, sd * 2));
        self.sd = sd

    def forward(self, pv, action, prog, lib):
        film = self.action_film(action);
        scale, shift = film[:self.sd], film[self.sd:]
        for i, sel in enumerate(prog):
            pv = pv + self.step_emb(torch.tensor(i, device=pv.device))
            pv = pv * (1 + 0.1 * torch.tanh(scale)) + 0.1 * torch.tanh(shift);
            pv = lib.apply_soft(sel, pv)
        return pv


class PreconditionChecker(nn.Module):
    def __init__(self, sd, ad):
        super().__init__();
        self.net = nn.Sequential(nn.Linear(sd + ad, sd // 2), nn.GELU(), nn.Linear(sd // 2, 1), nn.Sigmoid())

    def forward(self, s, a): return self.net(torch.cat([s, a]))


class RuleSynthesizer(nn.Module):
    def __init__(self, ed, sd, np, ms, pd=256):
        super().__init__();
        self.pd = pd;
        self._np = np
        self.proj = nn.Sequential(nn.Linear(ed, pd), nn.LayerNorm(pd), nn.GELU())
        self.a1 = nn.MultiheadAttention(pd, 4, batch_first=True);
        self.n1 = nn.LayerNorm(pd);
        self.n2 = nn.LayerNorm(pd)
        self.f1 = nn.Sequential(nn.Linear(pd, pd * 2), nn.GELU(), nn.Linear(pd * 2, pd))
        self.a2 = nn.MultiheadAttention(pd, 4, batch_first=True);
        self.n3 = nn.LayerNorm(pd);
        self.n4 = nn.LayerNorm(pd)
        self.f2 = nn.Sequential(nn.Linear(pd, pd * 2), nn.GELU(), nn.Linear(pd * 2, pd))
        self.state_proj = nn.Linear(sd, pd)
        self.heads = nn.ModuleList(
            [nn.Sequential(nn.Linear(pd * 2, 128), nn.GELU(), nn.Linear(128, np)) for _ in range(ms)])
        self.stop = nn.Sequential(nn.Linear(pd + sd, 128), nn.GELU(), nn.Linear(128, 1), nn.Sigmoid())

    def signature(self, ex):
        x = self.proj(ex).unsqueeze(0);
        a, _ = self.a1(x, x, x);
        x = self.n1(x + a);
        x = self.n2(x + self.f1(x))
        a, _ = self.a2(x, x, x);
        x = self.n3(x + a);
        x = self.n4(x + self.f2(x));
        return x.mean(1).squeeze(0)

    def rebuild(self, new_n):
        old = self._np
        if new_n <= old: return
        nh = nn.ModuleList()
        for h in self.heads:
            n = nn.Sequential(nn.Linear(self.pd * 2, 128), nn.GELU(), nn.Linear(128, new_n)).to(
                next(h.parameters()).device)
            with torch.no_grad():
                n[0].weight[:, :self.pd].copy_(h[0].weight[:, :self.pd]);
                n[0].weight[:, self.pd:].zero_();
                n[0].bias.copy_(h[0].bias)
                n[2].weight[:old].copy_(h[2].weight);
                n[2].bias[:old].copy_(h[2].bias)
                aw = h[2].weight.mean(0, keepdim=True).expand(new_n - old, -1)
                n[2].weight[old:].copy_(aw + torch.randn_like(aw) * 0.05);
                n[2].bias[old:].fill_(h[2].bias[:old].mean().item())
            nh.append(n)
        self.heads = nh;
        self._np = new_n

    def forward(self, ex_reprs, state, temp=0.8, np_=None, min_steps=2):
        pat = self.signature(ex_reprs);
        np_ = np_ or self._np
        state_ctx = self.state_proj(state);
        head_input = torch.cat([pat, state_ctx])
        prog, stop_probs = [], []
        for i, h in enumerate(self.heads):
            if i >= min_steps:
                sp = self.stop(torch.cat([pat, state]));
                stop_probs.append(sp)
                if not self.training and sp.item() > 0.6: break
            lg = h(head_input)
            if lg.shape[0] < np_:
                lg = torch.cat([lg, torch.zeros(np_ - lg.shape[0], device=lg.device)])
            elif lg.shape[0] > np_:
                lg = lg[:np_]
            if self.training:
                prog.append(F.gumbel_softmax(lg, tau=temp, hard=False))
            else:
                prog.append(F.one_hot(lg.argmax(), np_).float())
        return prog, pat, stop_probs


class Memory:
    def __init__(self, cap=300):
        self.cap = cap; self.entries = []

    def store(self, sig, prog, rel, ok):
        self.entries.append({"sig": sig.detach().clone(), "prog": prog, "rel": rel, "ok": ok, "cnt": 1})
        if len(self.entries) > self.cap: self.entries.sort(key=lambda e: e["cnt"],
                                                           reverse=True); self.entries = self.entries[:self.cap]

    def lookup(self, sig, thr=0.85):
        if not self.entries: return None
        best, bsim = None, -1
        for e in self.entries:
            sim = F.cosine_similarity(sig.unsqueeze(0), e["sig"].unsqueeze(0)).item()
            if sim > bsim: bsim, best = sim, e
        if bsim > thr and best: best["cnt"] += 1; return best
        return None

    def freq_pairs(self, mn=6):
        c = Counter()
        for e in self.entries:
            for i in range(len(e["prog"]) - 1): c[(e["prog"][i], e["prog"][i + 1])] += 1
        return {p: n for p, n in c.items() if n >= mn}

    def clear(self):
        self.entries = []

    def stats(self):
        return f"{len(self.entries)} entries, {sum(1 for e in self.entries if e['ok'])} correct" if self.entries else "Empty"


class GoalEvaluator(nn.Module):
    def __init__(self, sd):
        super().__init__();
        self.net = nn.Sequential(nn.Linear(sd * 2, sd), nn.LayerNorm(sd), nn.GELU(), nn.Linear(sd, sd // 2), nn.GELU(),
                                 nn.Linear(sd // 2, 1), nn.Sigmoid())

    def forward(self, s, g): return self.net(torch.cat([s, g]))


class ActionScorer(nn.Module):
    def __init__(self, sd, na):
        super().__init__();
        self.net = nn.Sequential(nn.Linear(sd * 2, sd), nn.LayerNorm(sd), nn.GELU(), nn.Linear(sd, sd // 2), nn.GELU(),
                                 nn.Linear(sd // 2, na))

    def forward(self, s, g): return self.net(torch.cat([s, g]))


class LatentPredictor(nn.Module):
    def __init__(self, sd, ad, pd):
        super().__init__()
        self.h = nn.Sequential(nn.Linear(sd + sd + ad + pd, sd * 2), nn.LayerNorm(sd * 2), nn.GELU(),
                               nn.Linear(sd * 2, sd * 2), nn.LayerNorm(sd * 2), nn.GELU(), nn.Linear(sd * 2, sd))

    def forward(self, obj, prop, action, sig): return self.h(torch.cat([obj, prop, action, sig]))


class VocabDecoder(nn.Module):
    def __init__(self, sd, ad, pd, vs):
        super().__init__()
        self.h = nn.Sequential(nn.Linear(sd + sd + ad + pd, sd), nn.LayerNorm(sd), nn.GELU(), nn.Dropout(0.1),
                               nn.Linear(sd, sd), nn.LayerNorm(sd), nn.GELU(), nn.Linear(sd, vs))

    def forward(self, pv, obj, action, sig): return self.h(torch.cat([pv, obj, action, sig]))


class CompleteWorldModel(nn.Module):
    def __init__(self, state_dim, vocab_size):
        super().__init__();
        cfg = CONFIG;
        self.sd = state_dim;
        ed = state_dim * 3;
        ns = cfg["num_slots"]
        self.obj_ext = ObjectExtractor(state_dim).to(DEVICE);
        self.slot_attn = SlotAttention(state_dim, ns).to(DEVICE)
        self.lib = PrimitiveLibrary(state_dim).to(DEVICE)
        self.syn = RuleSynthesizer(ed, state_dim, len(BASE_NAMES), cfg["max_program_steps"], cfg["proj_dim"]).to(DEVICE)
        self.slot_selector = SlotSelector(state_dim, ns).to(DEVICE)
        self.prop_updater = PropertyUpdater(state_dim, state_dim, cfg["max_program_steps"]).to(DEVICE)
        self.precond = PreconditionChecker(state_dim, state_dim).to(DEVICE)
        self.goal_eval = GoalEvaluator(state_dim).to(DEVICE)
        self.action_scorer = ActionScorer(state_dim, NUM_ACTIONS).to(DEVICE)
        self.latent_pred = LatentPredictor(state_dim, state_dim, cfg["proj_dim"]).to(DEVICE)
        self.vocab_dec = VocabDecoder(state_dim, state_dim, cfg["proj_dim"], vocab_size).to(DEVICE)
        self.mem = Memory(cfg["memory_capacity"]);
        self.world = WorldState()

    def forward(self, task, temp=0.8, use_mem=True):
        ex = task["example_reprs"].to(DEVICE);
        ta = task["test_action"].to(DEVICE);
        tt = task["test_state_tokens"].to(DEVICE)
        obj_vec, oa = self.obj_ext(tt);
        slots, sa = self.slot_attn(tt)
        sw = self.slot_selector(ta);
        pv = (sw.unsqueeze(1) * slots).sum(0)
        ts = task["test_state"].to(DEVICE);
        pc = self.precond(ts, ta)
        np_ = self.lib.n;
        sig = self.syn.signature(ex);
        from_mem = False;
        prog = None;
        stop_probs = []
        if use_mem and not self.training:
            c = self.mem.lookup(sig)
            if c: prog = [F.one_hot(torch.tensor(i), np_).float().to(DEVICE) for i in c["prog"]]; from_mem = True
        if prog is None: prog, _, stop_probs = self.syn(ex, pv, temp, np_)
        if task.get("task_type") == "multistep" and "test_actions" in task:
            p = pv;
            pse = task.get("per_step_examples")
            for si, a in enumerate(task["test_actions"]):
                ad = a.to(DEVICE)
                if pse and si < len(pse):
                    sp, _, _ = self.syn(pse[si].to(DEVICE), p, temp, np_)
                else:
                    sp = prog
                step_sw = self.slot_selector(ad);
                step_prop = (step_sw.unsqueeze(1) * slots).sum(0)
                p = self.prop_updater(p, ad, sp, self.lib)
            transformed = p
        else:
            transformed = self.prop_updater(pv, ta, prog, self.lib)
        pred_vec = self.latent_pred(obj_vec, transformed, ta, sig)
        logits = self.vocab_dec(pred_vec, obj_vec, ta, sig)
        return logits, prog, sig, from_mem, stop_probs, oa, sa, sw, obj_vec, pc, pred_vec

    def plan(self, cache, initial_state, goal_state, max_depth=3):
        self.eval()
        with torch.no_grad():
            sv = cache.get(f" {initial_state}");
            gv = cache.get(f" {goal_state}")
            tokens = cache.get_tokens(f" {initial_state}")
            if self.goal_eval(sv, gv).item() > 0.9: return [], self.goal_eval(sv, gv).item()
            plan = [];
            csv = sv;
            ct = tokens;
            prev = None
            for step in range(max_depth):
                al = self.action_scorer(csv, gv)
                if prev is not None: al[prev] -= 2.0
                ai = al.argmax().item();
                an = IDX2ACT[ai];
                plan.append(an);
                prev = ai
                # Simulate step
                rule = TRANSITION_RULES.get(an)
                if rule is None: break
                triples = rule["triples"];
                st = random.sample(triples, min(3, len(triples)))
                step_ex = torch.stack(
                    [torch.cat([cache.get(f" {s}"), cache.get(f" {a}"), cache.get(f" {res}")]) for s, a, res in st]).to(
                    DEVICE)
                av = cache.get(f" {an}");
                ov, _ = self.obj_ext(ct);
                sl, _ = self.slot_attn(ct)
                sw_ = self.slot_selector(av);
                pv_ = (sw_.unsqueeze(1) * sl).sum(0)
                np_ = self.lib.n;
                sp, _, _ = self.syn(step_ex, pv_, 0.1, np_);
                sig_ = self.syn.signature(step_ex)
                tr = self.prop_updater(pv_, av, sp, self.lib);
                pv2 = self.latent_pred(ov, tr, av, sig_)
                best_sim, best_state = -1, None
                for phrase, vec in cache.last_cache.items():
                    sim = F.cosine_similarity(pv2.unsqueeze(0), vec.unsqueeze(0)).item()
                    if sim > best_sim: best_sim, best_state = sim, phrase
                if best_state and best_sim > 0.7:
                    csv = pv2;
                    ct = cache.get_tokens(best_state)
                    if self.goal_eval(csv, gv).item() > 0.85: break
                else:
                    break
            return plan, self.goal_eval(csv, gv).item()

    def memorize(self, sig, prog, rel, ok):
        self.mem.store(sig, [s.argmax().item() for s in prog], rel, ok)

    def compress(self):
        freq = self.mem.freq_pairs(CONFIG["compression_threshold"])
        if not freq: print("  [COMPRESS] Nothing."); return 0
        created = 0
        for (a, b), cnt in sorted(freq.items(), key=lambda x: -x[1]):
            if created >= CONFIG["max_new_primitives"]: break
            na, nb = self.lib.names[a], self.lib.names[b];
            nm = f"{na}_{nb}"
            if nm in self.lib.names: continue
            print(f"  [COMPRESS] {na}->{nb} ({cnt}x)");
            self.lib.add(nm, a, b);
            created += 1
        if created > 0: self.syn.rebuild(self.lib.n); self.mem.clear()
        return created


# =============================================================================
# Losses
# =============================================================================
def div_loss(prog):
    if len(prog) < 2: return torch.tensor(0.0, device=DEVICE)
    l, n = torch.tensor(0.0, device=DEVICE), len(prog)
    for i in range(n):
        for j in range(i + 1, n): l = l + F.relu(F.cosine_similarity(prog[i].unsqueeze(0), prog[j].unsqueeze(0)) - 0.3)
    return l / (n * (n - 1) / 2)


def use_loss(prog, np_): avg = torch.stack(prog).mean(0)[:np_]; return math.log(np_) - (
    -(avg * torch.log(avg + 1e-8)).sum())


def novelty_loss(prog, np_, n_base=6):
    if np_ <= n_base: return torch.tensor(0.0, device=DEVICE)
    return F.relu(0.15 - torch.stack(prog).mean(0)[:np_][n_base:].sum())


def goal_contrastive_loss(ge, rv, all_sv):
    pos = ge(rv, rv);
    nv = random.choice(all_sv)
    if F.cosine_similarity(rv.unsqueeze(0), nv.unsqueeze(0)).item() > 0.99: nv = random.choice(all_sv)
    neg = ge(rv, nv);
    return (F.binary_cross_entropy(pos, torch.ones_like(pos)) + F.binary_cross_entropy(neg, torch.zeros_like(neg))) / 2


# =============================================================================
# Training & Eval (with temporal)
# =============================================================================
def fmt(prog, names): return " -> ".join(
    names[s.argmax().item()] if s.argmax().item() < len(names) else "?" for s in prog)


def fmt_attn(attn, tokenizer, phrase):
    ids = tokenizer.encode(phrase);
    tokens = [tokenizer.decode([t]) for t in ids]
    al = attn[:len(tokens)];
    pairs = sorted(zip(al.tolist(), tokens), reverse=True)
    return " ".join([f"{t.strip()}({w:.2f})" for w, t in pairs[:3] if w > 0.05])


def fmt_slots(sw): return " ".join([f"S{i}:{w:.2f}" for i, w in enumerate(sw.tolist())])


def train_cycle(model, cache, s2i, i2s, chains, tokenizer, all_sv, n_iters, cycle, lr):
    cfg = CONFIG
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: s / 200 if s < 200 else 0.5 * (
                1 + math.cos(math.pi * (s - 200) / max(n_iters - 200, 1))))
    model.train();
    ok, tot, pl_ok, pl_tot = 0, 0, 0, 0;
    t0 = time.time()
    for it in range(n_iters):
        task = generate_mixed_task(cache, s2i, chains);
        np_ = model.lib.n
        logits, prog, sig, _, stop_probs, oa, sa, sw, obj_vec, precond_sc, pred_vec = model(task, cfg["temperature"],
                                                                                            use_mem=False)
        target_vec = task["result_state_vec"].to(DEVICE)
        latent_loss = 1.0 - F.cosine_similarity(pred_vec.unsqueeze(0), target_vec.unsqueeze(0))
        tgt = torch.tensor(task["target_idx"], device=DEVICE);
        lce = F.cross_entropy(logits.unsqueeze(0), tgt.unsqueeze(0))
        stop_l = torch.tensor(0.0, device=DEVICE)
        if stop_probs:
            for sp in stop_probs: stop_l = stop_l + (1.0 - sp).squeeze()
            stop_l = stop_l / len(stop_probs)
        rv = task["result_state_vec"].to(DEVICE);
        gl = goal_contrastive_loss(model.goal_eval, rv, all_sv)
        al_logits = model.action_scorer(obj_vec.detach(), rv)
        al_loss = F.cross_entropy(al_logits.unsqueeze(0), torch.tensor(task["action_idx"], device=DEVICE).unsqueeze(0))
        precond_target = torch.tensor(1.0 if task["precondition_holds"] else 0.0, device=DEVICE)
        precond_loss = F.binary_cross_entropy(precond_sc.view(-1)[0], precond_target)
        loss = (lce + 0.2 * latent_loss + 0.1 * div_loss(prog) + 0.05 * use_loss(prog, np_) + 0.1 * novelty_loss(prog,
                                                                                                                 np_) + 0.03 * stop_l + 0.02 * len(
            prog) + cfg["goal_loss_weight"] * gl + cfg["action_loss_weight"] * al_loss + cfg[
                    "precondition_loss_weight"] * precond_loss) / cfg["grad_accumulation"]
        loss.backward()
        pred = i2s.get(logits.argmax().item(), "?");
        correct = pred == task["expected"]
        ok += correct;
        tot += 1
        pa_ok = IDX2ACT[al_logits.argmax().item()] == task["action"];
        pl_ok += pa_ok;
        pl_tot += 1
        with torch.no_grad():
            model.memorize(sig, prog, task["rule_name"], correct)
        if (it + 1) % cfg["grad_accumulation"] == 0:
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step();
            opt.zero_grad();
            sch.step()
        if it % 500 == 0:
            acc = 100 * ok / max(tot, 1);
            speed = (it + 1) / max(time.time() - t0, 0.01)
            tt_mark = " [TEMP]" if task.get("task_type") == "temporal" else (
                " [D" + str(task['depth']) + "]" if task.get("task_type") == "multistep" else "")
            print(
                f"  [C{cycle}] {it:5d}/{n_iters} | WM:{lce.item():.2f} L:{latent_loss.item():.2f} | Acc:{acc:4.0f}% | {speed:.0f}it/s | [{task['rule_name'][:15]:15s}] {'V' if correct else 'X'} K={len(prog)}{tt_mark}")
            if it > 0 and it % 3000 == 0: ok, tot = 0, 0


def evaluate(model, cache, s2i, i2s, chains, tokenizer):
    cfg = CONFIG;
    model.eval();
    names = model.lib.names;
    ne = cfg["eval_samples"]
    print(f"\n  --- Single-Step (n={ne}) ---\n")
    results = {}
    with torch.no_grad():
        for rule in sorted(TRANSITION_RULES.keys()):
            ok, t3ok, shown = 0, 0, 0
            for _ in range(ne):
                t = generate_single_task(cache, s2i, rule_name=rule)
                lg, _, _, _, _, _, _, _, _, _, _ = model(t, 0.1, use_mem=True)
                pred, target = i2s.get(lg.argmax().item(), "?"), t["expected"]
                top3 = [i2s.get(i, "?") for i in lg.topk(min(3, lg.shape[0])).indices.tolist()]
                ok += pred == target;
                t3ok += target in top3
                if shown < 1:
                    is_temp = " [TEMP]" if rule in ["wait", "charge", "put in oven", "plant", "light"] else ""
                    print(
                        f"    [{rule:12s}] '{t['state'][:20]}' + '{t['action'][:10]}' -> '{pred[:22]}' {'V' if pred == target else 'X'}{is_temp}")
                    shown += 1
            a, t3 = 100 * ok / ne, 100 * t3ok / ne;
            results[rule] = (a, t3)
            print(f"      -> {a:.0f}% (top3: {t3:.0f}%)")
    print("\n  --- Summary ---")
    # Separate original vs temporal results
    original_actions = ["heat", "cool", "warm up", "boil", "simmer", "open", "close", "switch on", "switch off", "fill",
                        "drop", "push", "put in"]
    temporal_actions = ["charge", "put in oven", "plant", "light", "wait"]
    print("  ORIGINAL ACTIONS:")
    for r, (a, t3) in sorted(results.items(), key=lambda x: -x[1][0]):
        if r in original_actions:
            bar = chr(9608) * int(a / 5) + chr(9617) * int((t3 - a) / 5)
            print(f"    {r:12s}: {a:5.1f}% [{bar}] (top3: {t3:.0f}%)")
    print("  TEMPORAL ACTIONS:")
    for r, (a, t3) in sorted(results.items(), key=lambda x: -x[1][0]):
        if r in temporal_actions:
            bar = chr(9608) * int(a / 5) + chr(9617) * int((t3 - a) / 5)
            print(f"    {r:12s}: {a:5.1f}% [{bar}] (top3: {t3:.0f}%)")
    n = len(results);
    ov = sum(a for a, _ in results.values()) / n
    n_orig = sum(1 for r in results if r in original_actions)
    ov_orig = sum(a for r, (a, _) in results.items() if r in original_actions) / max(n_orig, 1)
    n_temp = sum(1 for r in results if r in temporal_actions)
    ov_temp = sum(a for r, (a, _) in results.items() if r in temporal_actions) / max(n_temp, 1)
    print(f"\n    ALL: {ov:.1f}% | ORIGINAL: {ov_orig:.1f}% | TEMPORAL: {ov_temp:.1f}%")

    # === TEMPORAL CHAIN EVALUATION ===
    print(f"\n  --- Temporal Chains ---")
    tc_results = defaultdict(list)
    with torch.no_grad():
        for tc in TEMPORAL_CHAINS:
            # Test full chain: can the model predict the final state from the initial state
            # through the sequence of actions?
            chain_ok = True
            for step_i in range(len(tc["actions"])):
                state = tc["states"][step_i];
                action = tc["actions"][step_i];
                expected = tc["states"][step_i + 1]
                rule_name = action
                rule = TRANSITION_RULES.get(rule_name)
                if not rule: chain_ok = False; break
                other_triples = [(s, a, r) for s, a, r in rule["triples"] if s != state]
                if len(other_triples) < 2: other_triples = rule["triples"]
                chosen = random.sample(other_triples, min(3, len(other_triples)))
                ex = torch.stack(
                    [torch.cat([cache.get(f" {s}"), cache.get(f" {a}"), cache.get(f" {res}")]) for s, a, res in chosen])
                task = {"example_reprs": ex, "test_state": cache.get(f" {state}"),
                        "test_action": cache.get(f" {action}"),
                        "test_state_tokens": cache.get_tokens(f" {state}"), "target_idx": s2i.get(expected, 0),
                        "rule_name": rule_name, "state": state, "action": action, "expected": expected, "depth": 1,
                        "task_type": "temporal", "result_state_vec": cache.get(f" {expected}"),
                        "action_idx": ACT2IDX.get(action, 0), "precondition": "any",
                        "property_changed": tc["property"], "precondition_holds": True,
                        "property_idx": PROP2IDX.get(tc["property"], 0)}
                lg, _, _, _, _, _, _, _, _, _, _ = model(task, 0.1, use_mem=False)
                pred = i2s.get(lg.argmax().item(), "?")
                if pred != expected: chain_ok = False
            tc_results[tc["type"]].append(chain_ok)
    for tc_type, results_list in sorted(tc_results.items()):
        n_ok = sum(results_list);
        n_total = len(results_list)
        print(f"    {tc_type:15s}: {100 * n_ok / n_total:.0f}% ({n_ok}/{n_total})")

    # === WAIT CONTEXT-DEPENDENCY TEST ===
    print(f"\n  --- Wait Context-Dependency ---")
    print(f"  (same 'wait' action, different effects based on state)")
    wait_rule = TRANSITION_RULES.get("wait")
    if wait_rule:
        w_ok, w_total = 0, 0
        with torch.no_grad():
            for s, a, res in wait_rule["triples"][:20]:  # test 20 wait transitions
                other = [(os, oa, ore) for os, oa, ore in wait_rule["triples"] if os != s]
                if len(other) < 2: continue
                chosen = random.sample(other, min(3, len(other)))
                ex = torch.stack(
                    [torch.cat([cache.get(f" {os}"), cache.get(f" {oa}"), cache.get(f" {ore}")]) for os, oa, ore in
                     chosen])
                task = {"example_reprs": ex, "test_state": cache.get(f" {s}"), "test_action": cache.get(f" {a}"),
                        "test_state_tokens": cache.get_tokens(f" {s}"), "target_idx": s2i.get(res, 0),
                        "rule_name": "wait", "state": s, "action": a, "expected": res, "depth": 1,
                        "task_type": "temporal", "result_state_vec": cache.get(f" {res}"),
                        "action_idx": ACT2IDX.get("wait", 0), "precondition": "any",
                        "property_changed": "temporal", "precondition_holds": True, "property_idx": 0}
                lg, _, _, _, _, _, _, _, _, _, _ = model(task, 0.1, use_mem=False)
                pred = i2s.get(lg.argmax().item(), "?");
                correct = pred == res
                w_ok += correct;
                w_total += 1
                if w_total <= 6:
                    print(f"    '{s[:22]}' + wait -> '{pred[:22]}' {'V' if correct else 'X'} (want: '{res[:22]}')")
        if w_total: print(f"    WAIT ACCURACY: {100 * w_ok / w_total:.0f}% ({w_ok}/{w_total})")

    # Planning
    print(f"\n  --- Planning ---")
    pl_1st, pl_full, pl_t = 0, 0, 0;
    real_1st, real_full, real_t = 0, 0, 0
    for _ in range(50):
        task = generate_planning_task(cache, s2i, chains)
        if not task: continue
        plan, sc = model.plan(cache, task["initial_state"], task["goal_state"], max_depth=task["depth"])
        is_rt = task["initial_state"] == task["goal_state"]
        if plan:
            fc = plan[0] == task["correct_actions"][0]; full_c = plan[:len(task["correct_actions"])] == task[
                "correct_actions"]
        else:
            fc = is_rt; full_c = is_rt
        pl_1st += fc;
        pl_full += full_c;
        pl_t += 1
        if not is_rt: real_1st += fc; real_full += full_c; real_t += 1
        if pl_t <= 4:
            print(f"    {task['initial_state'][:22]} -> {task['goal_state'][:22]}")
            print(f"      correct:{task['correct_actions']}  plan:{plan}  1st:{'V' if fc else 'X'}")
    if pl_t: print(f"\n    ALL:  1st:{100 * pl_1st / pl_t:.0f}% full:{100 * pl_full / pl_t:.0f}% ({pl_t})")
    if real_t: print(f"    REAL: 1st:{100 * real_1st / real_t:.0f}% full:{100 * real_full / real_t:.0f}% ({real_t})")
    return results


def main():
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    print(f"Device: {DEVICE}")
    print("Loading GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2 = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE).eval()
    sl, s2i, i2s = build_vocab()
    global i2s_global;
    i2s_global = i2s
    print(f"\nRules:{len(TRANSITION_RULES)} | States:{len(sl)} | Actions:{NUM_ACTIONS} | Properties:{NUM_PROPERTIES}")
    print(f"Property types: {PROPERTY_TYPES}")
    print(f"Temporal properties: {sorted(TEMPORAL_PROPERTIES)}")
    chains = find_chains(CONFIG["max_chain_length"])
    cache = TokenEmbeddingCache(gpt2, tokenizer, layer=CONFIG["perceiver_layer"])
    all_sv = [cache.get(f" {s}").to(DEVICE) for s in sl]
    model = CompleteWorldModel(768, len(sl))
    tp = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {tp:,}\n")
    cfg = CONFIG
    for c in range(cfg["num_cycles"]):
        print(f"{'=' * 65}")
        print(f"CYCLE {c + 1}/{cfg['num_cycles']} | Prims:{model.lib.n}")
        print(f"Library: {model.lib.names}")
        print(f"{'=' * 65}\n")
        train_cycle(model, cache, s2i, i2s, chains, tokenizer, all_sv, cfg["iters_per_cycle"][c], c + 1,
                    cfg["lr_per_cycle"][c])
        print(f"\n--- Eval Cycle {c + 1} ---")
        evaluate(model, cache, s2i, i2s, chains, tokenizer)
        print(f"\n  Memory: {model.mem.stats()}")
        if c < cfg["num_cycles"] - 1:
            print(f"\n--- Compression ---")
            n = model.compress()
            if n: print(f"  {n} new primitives.")
    print(f"\n{'=' * 65}\nFINAL TEST\n{'=' * 65}")
    print(f"Library: {model.lib.names}")
    evaluate(model, cache, s2i, i2s, chains, tokenizer)
    print(f"\n  Memory: {model.mem.stats()}\nDone.")


if __name__ == "__main__":
    main()
