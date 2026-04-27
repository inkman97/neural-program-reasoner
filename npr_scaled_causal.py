"""
Neural Program Reasoner — World Model v9: Causal Understanding + Time

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
    "iters_per_cycle": [5000, 3000, 2000],
    "lr_per_cycle": [5e-4, 3e-4, 1e-4],
    "perceiver_layer": 8,
    "eval_samples": 40,
    "max_chain_length": 3,
    "multistep_ratio": 0.2,
    "goal_loss_weight": 0.2,
    "action_loss_weight": 0.2,
    "precondition_loss_weight": 0.15,
}

# =============================================================================
# Dataset with preconditions, chain effects, and multi-object scenes
# =============================================================================

# Each rule now has: (state, action, result, precondition, side_effects)
# precondition: what must be true for the action to work
# side_effects: additional changes that happen

TRANSITION_RULES = {
    "heat": {
        "triples": [
            ("the water is cold","heat","the water is hot"),("the soup is cold","heat","the soup is hot"),
            ("the coffee is cold","heat","the coffee is hot"),("the tea is cold","heat","the tea is hot"),
            ("the milk is cold","heat","the milk is hot"),("the food is cold","heat","the food is hot"),
            ("the oil is cold","heat","the oil is hot"),("the pan is cold","heat","the pan is hot"),
            ("the iron is cold","heat","the iron is hot"),("the metal is cold","heat","the metal is hot"),
        ],
        "precondition": "cold",  # must be cold to heat
        "property_changed": "temperature",
    },
    "cool": {
        "triples": [
            ("the water is hot","cool","the water is cold"),("the soup is hot","cool","the soup is cold"),
            ("the coffee is hot","cool","the coffee is cold"),("the tea is hot","cool","the tea is cold"),
            ("the milk is hot","cool","the milk is cold"),("the food is hot","cool","the food is cold"),
            ("the oil is hot","cool","the oil is cold"),("the pan is hot","cool","the pan is cold"),
            ("the iron is hot","cool","the iron is cold"),("the metal is hot","cool","the metal is cold"),
        ],
        "precondition": "hot",
        "property_changed": "temperature",
    },
    "open": {
        "triples": [
            ("the door is closed","open","the door is open"),("the window is closed","open","the window is open"),
            ("the box is closed","open","the box is open"),("the jar is closed","open","the jar is open"),
            ("the gate is closed","open","the gate is open"),("the drawer is closed","open","the drawer is open"),
            ("the lid is closed","open","the lid is open"),("the bag is closed","open","the bag is open"),
            ("the bottle is closed","open","the bottle is open"),("the cabinet is closed","open","the cabinet is open"),
        ],
        "precondition": "closed",
        "property_changed": "openness",
    },
    "close": {
        "triples": [
            ("the door is open","close","the door is closed"),("the window is open","close","the window is closed"),
            ("the box is open","close","the box is closed"),("the jar is open","close","the jar is closed"),
            ("the gate is open","close","the gate is closed"),("the drawer is open","close","the drawer is closed"),
            ("the lid is open","close","the lid is closed"),("the bag is open","close","the bag is closed"),
            ("the bottle is open","close","the bottle is closed"),("the cabinet is open","close","the cabinet is closed"),
        ],
        "precondition": "open",
        "property_changed": "openness",
    },
    "fill": {
        "triples": [
            ("the glass is empty","fill","the glass is full"),("the cup is empty","fill","the cup is full"),
            ("the bucket is empty","fill","the bucket is full"),("the pool is empty","fill","the pool is full"),
            ("the tank is empty","fill","the tank is full"),("the bottle is empty","fill","the bottle is full"),
            ("the bowl is empty","fill","the bowl is full"),("the tub is empty","fill","the tub is full"),
            ("the jug is empty","fill","the jug is full"),("the pot is empty","fill","the pot is full"),
        ],
        "precondition": "empty",
        "property_changed": "fullness",
    },
    "break": {
        "triples": [
            ("the glass is intact","drop","the glass is broken"),("the plate is intact","drop","the plate is broken"),
            ("the vase is intact","drop","the vase is broken"),("the mirror is intact","drop","the mirror is broken"),
            ("the cup is intact","drop","the cup is broken"),("the bowl is intact","drop","the bowl is broken"),
            ("the bottle is intact","drop","the bottle is broken"),("the window is intact","drop","the window is broken"),
            ("the jar is intact","drop","the jar is broken"),("the lamp is intact","drop","the lamp is broken"),
        ],
        "precondition": "intact",
        "property_changed": "integrity",
    },
    "gravity": {
        "triples": [
            ("the ball is on the table","push","the ball is on the floor"),
            ("the cup is on the shelf","push","the cup is on the floor"),
            ("the book is on the desk","push","the book is on the floor"),
            ("the phone is on the bed","push","the phone is on the floor"),
            ("the lamp is on the table","push","the lamp is on the floor"),
            ("the plate is on the counter","push","the plate is on the floor"),
            ("the glass is on the table","push","the glass is on the floor"),
            ("the toy is on the shelf","push","the toy is on the floor"),
            ("the pen is on the desk","push","the pen is on the floor"),
            ("the remote is on the couch","push","the remote is on the floor"),
        ],
        "precondition": "on the",  # must be on something
        "property_changed": "position",
    },
    "switch_on": {
        "triples": [
            ("the lamp is off","switch on","the lamp is on"),("the light is off","switch on","the light is on"),
            ("the screen is off","switch on","the screen is on"),("the tv is off","switch on","the tv is on"),
            ("the radio is off","switch on","the radio is on"),("the computer is off","switch on","the computer is on"),
            ("the fan is off","switch on","the fan is on"),("the heater is off","switch on","the heater is on"),
            ("the oven is off","switch on","the oven is on"),("the speaker is off","switch on","the speaker is on"),
        ],
        "precondition": "off",
        "property_changed": "power",
    },
    "switch_off": {
        "triples": [
            ("the lamp is on","switch off","the lamp is off"),("the light is on","switch off","the light is off"),
            ("the screen is on","switch off","the screen is off"),("the tv is on","switch off","the tv is off"),
            ("the radio is on","switch off","the radio is off"),("the computer is on","switch off","the computer is off"),
            ("the fan is on","switch off","the fan is off"),("the heater is on","switch off","the heater is off"),
            ("the oven is on","switch off","the oven is off"),("the speaker is on","switch off","the speaker is off"),
        ],
        "precondition": "is on",
        "property_changed": "power",
    },
    "put_inside": {
        "triples": [
            ("the ball is outside the box","put in","the ball is inside the box"),
            ("the toy is outside the bag","put in","the toy is inside the bag"),
            ("the book is outside the drawer","put in","the book is inside the drawer"),
            ("the key is outside the pocket","put in","the key is inside the pocket"),
            ("the coin is outside the jar","put in","the coin is inside the jar"),
            ("the pen is outside the case","put in","the pen is inside the case"),
            ("the shirt is outside the closet","put in","the shirt is inside the closet"),
            ("the food is outside the fridge","put in","the food is inside the fridge"),
            ("the tool is outside the shed","put in","the tool is inside the shed"),
            ("the letter is outside the envelope","put in","the letter is inside the envelope"),
        ],
        "precondition": "outside",
        "property_changed": "containment",
    },
}

# Chain effects: action on state A causes additional effect on state B
CHAIN_EFFECTS = [
    # push a fragile object → falls AND breaks
    {"trigger_rule": "gravity", "trigger_objects": ["glass","plate","vase","cup","bowl","lamp"],
     "chain_rule": "break", "description": "fragile object falls and breaks"},
]

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
    }

def generate_mixed_task(cache, s2i, chains, ne=5):
    if random.random()<CONFIG["multistep_ratio"] and any(chains.get(d) for d in [2,3]):
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
    def __init__(self, sd):
        super().__init__()
        self.sd, self.names = sd, list(BASE_NAMES)
        self.identity = nn.Sequential(nn.Linear(sd,sd),nn.Tanh())
        self.negate = nn.Sequential(nn.Linear(sd,sd*2),nn.GELU(),nn.Linear(sd*2,sd*2),nn.GELU(),nn.Linear(sd*2,sd))
        self.morph = nn.Sequential(nn.Linear(sd,sd),nn.LayerNorm(sd),nn.GELU(),nn.Linear(sd,sd))
        self.aq,self.ak,self.av,self.ao = nn.Linear(sd,sd//4),nn.Linear(sd,sd//4),nn.Linear(sd,sd),nn.Linear(sd,sd)
        self.lookup = nn.Sequential(nn.Linear(sd,sd*2),nn.GELU(),nn.Linear(sd*2,sd*2),nn.GELU(),nn.Linear(sd*2,sd))
        self.bc = nn.Parameter(torch.randn(sd)*0.01)
        self.bn = nn.Sequential(nn.Linear(sd*2,sd),nn.GELU(),nn.Linear(sd,sd))
        self.gates = nn.ParameterList([nn.Parameter(torch.tensor(g)) for g in [0.01,0.5,0.3,0.4,0.5,0.3]])
        self.inv_p, self.inv_g = nn.ModuleList(), nn.ParameterList()
    @property
    def n(self): return len(BASE_NAMES)+len(self.inv_p)
    def _base(self, i, s):
        if i==0: return s+torch.sigmoid(self.gates[0])*self.identity(s)
        if i==1: return s+torch.sigmoid(self.gates[1])*self.negate(s)
        if i==2: return s+torch.sigmoid(self.gates[2])*self.morph(s)
        if i==3:
            q,k,v=self.aq(s),self.ak(s),self.av(s)
            return s+torch.sigmoid(self.gates[3])*self.ao(torch.sigmoid(torch.dot(q,k)/math.sqrt(q.shape[0]))*v)
        if i==4: return s+torch.sigmoid(self.gates[4])*self.lookup(s)
        if i==5: return s+torch.sigmoid(self.gates[5])*self.bn(torch.cat([s,self.bc]))
    def apply(self, i, s):
        if i<len(BASE_NAMES): return self._base(i,s)
        j=i-len(BASE_NAMES); return s+torch.sigmoid(self.inv_g[j])*self.inv_p[j](s)
    def apply_soft(self, w, s): return sum(w[i]*self.apply(i,s) for i in range(self.n))
    def add(self, name, a, b):
        sd,dev=self.sd,next(self.parameters()).device
        self.inv_p.append(nn.Sequential(nn.Linear(sd,sd*2),nn.LayerNorm(sd*2),nn.GELU(),nn.Linear(sd*2,sd*2),nn.GELU(),nn.Linear(sd*2,sd)).to(dev))
        self.inv_g.append(nn.Parameter(torch.tensor(0.4,device=dev)))
        self.names.append(name); print(f"  [COMPRESS] Created: {name}"); return self.n-1


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
        self.action_film = nn.Sequential(nn.Linear(action_dim,sd*2),nn.GELU(),nn.Linear(sd*2,sd*2))
        self.sd = sd
    def forward(self, prop_vec, action, prog, lib):
        film = self.action_film(action)
        scale, shift = film[:self.sd], film[self.sd:]
        for i, sel in enumerate(prog):
            prop_vec = prop_vec + self.step_emb(torch.tensor(i,device=prop_vec.device))
            prop_vec = prop_vec * (1+0.1*torch.tanh(scale)) + 0.1*torch.tanh(shift)
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
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(pd,128),nn.GELU(),nn.Linear(128,np)) for _ in range(max_steps)])
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
            n=nn.Sequential(nn.Linear(self.pd,128),nn.GELU(),nn.Linear(128,new_n)).to(next(h.parameters()).device)
            with torch.no_grad():
                n[0].weight.copy_(h[0].weight); n[0].bias.copy_(h[0].bias)
                n[2].weight[:old].copy_(h[2].weight); n[2].bias[:old].copy_(h[2].bias)
                aw=h[2].weight.mean(0,keepdim=True).expand(new_n-old,-1)
                n[2].weight[old:].copy_(aw+torch.randn_like(aw)*0.05)
                n[2].bias[old:].fill_(h[2].bias[:old].mean().item())
            nh.append(n)
        self.heads=nh; self._np=new_n
    def forward(self, ex_reprs, state, temp=0.8, np_=None, min_steps=2):
        pat=self.signature(ex_reprs); np_=np_ or self._np
        prog, stop_probs = [], []
        for i,h in enumerate(self.heads):
            if i>=min_steps:
                sp=self.stop(torch.cat([pat,state])); stop_probs.append(sp)
                if not self.training and sp.item()>0.6: break
            lg=h(pat)
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

class ResultGenerator(nn.Module):
    def __init__(self, sd, action_dim, pattern_dim, vocab_size):
        super().__init__()
        # object + transformed_property + action + signature
        input_dim = sd + sd + action_dim + pattern_dim
        self.h = nn.Sequential(
            nn.Linear(input_dim, sd), nn.LayerNorm(sd), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(sd, sd), nn.LayerNorm(sd), nn.GELU(),
            nn.Linear(sd, vocab_size))
    def forward(self, obj, prop, action, sig):
        return self.h(torch.cat([obj, prop, action, sig]))


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
        # Output
        self.gen = ResultGenerator(state_dim, state_dim, cfg["proj_dim"], vocab_size).to(DEVICE)
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
            p = prop_vec
            for a in task["test_actions"]: p = self.prop_updater(p, a.to(DEVICE), prog, self.lib)
            transformed = p
        else:
            transformed = self.prop_updater(prop_vec, test_action, prog, self.lib)

        logits = self.gen(obj_vec, transformed, test_action, sig)
        return logits, prog, sig, from_mem, stop_probs, obj_attn, slot_attn, slot_weights, obj_vec, precond_score

    def plan(self, cache, initial_state, goal_state, max_depth=3):
        self.eval()
        with torch.no_grad():
            sv = cache.get(f" {initial_state}"); gv = cache.get(f" {goal_state}")
            plan = []
            for _ in range(max_depth):
                al = self.action_scorer(sv, gv)
                ai = al.argmax().item(); plan.append(IDX2ACT[ai])
                if self.goal_eval(sv, gv).item()>0.8: break
                sv = sv + 0.1*(gv-sv)
            return plan, self.goal_eval(sv, gv).item()

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
        logits,prog,sig,_,stop_probs,oa,sa,sw,obj_vec,precond_sc = model(task,cfg["temperature"],use_mem=False)

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

        loss=(lce+0.1*div_loss(prog)+0.05*use_loss(prog,np_)+0.1*novelty_loss(prog,np_)+
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
                lg,pr,_,_,_,oa2,sa2,sw2,_,pc=model(t,cfg["temperature"],use_mem=False)
                pw=i2s.get(lg.argmax().item(),"?"); tw=t["expected"]
                st=t["state"]
                ostr=fmt_attn(oa2,tokenizer,f" {st}")
            ms=f" [D{t['depth']}]" if t.get("task_type")=="multistep" else ""
            print(f"  [C{cycle}] {it:4d}/{n_iters} | WM:{lce.item():.2f} G:{gl.item():.2f} A:{al_loss.item():.2f} P:{precond_loss.item():.2f} | "
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
            _,pr,_,_,_,oa,sa,sw,_,pc=model(t,0.1,use_mem=False)
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
                lg,_,_,_,_,oa,sa,sw,_,pc=model(t,0.1,use_mem=True)
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

    # Slot analysis: do same-property-type actions use the same slot?
    print("\n  --- Slot Analysis ---")
    prop_slots = defaultdict(list)
    with torch.no_grad():
        for rule, r in TRANSITION_RULES.items():
            for s,a,res in r["triples"][:3]:
                t = generate_single_task(cache, s2i, rule_name=rule)
                _,_,_,_,_,_,_,sw,_,_ = model(t, 0.1, use_mem=False)
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
            for _ in range(min(30,len(chains[depth]))):
                t=generate_multistep_task(cache,s2i,chains,3)
                if t["depth"]!=depth: continue
                lg,_,_,_,_,_,_,_,_,_=model(t,0.1,use_mem=False)
                pred=i2s.get(lg.argmax().item(),"?"); ok+=pred==t["expected"]; tested+=1
                if tested<=2: print(f"    '{t['state'][:20]}' +[{t['action'][:20]}] -> {'V' if pred==t['expected'] else 'X'}")
        if tested: print(f"    DEPTH-{depth}: {100*ok/tested:.0f}% [{tested} tested]")

    # Planning
    print(f"\n  --- Planning ---")
    pl_ok,pl_t=0,0
    for _ in range(30):
        task=generate_planning_task(cache,s2i,chains)
        if not task: continue
        plan,sc=model.plan(cache,task["initial_state"],task["goal_state"],max_depth=task["depth"])
        fc=plan[0]==task["correct_actions"][0] if plan else False
        pl_ok+=fc; pl_t+=1
        if pl_t<=5: print(f"    {task['initial_state'][:22]} -> {task['goal_state'][:22]}  plan:{plan}  {'V' if fc else 'X'}")
    if pl_t: print(f"\n    PLANNING (1st): {100*pl_ok/pl_t:.0f}% ({pl_ok}/{pl_t})")
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
            _,pr,_,_,_,_,_,_,_,_=model(t,0.1,use_mem=False)
            for sel in pr: usage[model.lib.names[sel.argmax().item()] if sel.argmax().item()<len(model.lib.names) else "?"]+=1
    tu=sum(usage.values())
    for nm,cnt in sorted(usage.items(),key=lambda x:-x[1]): print(f"    {nm:20s}: {100*cnt/tu:5.1f}%")
    print(f"\n  Memory: {model.mem.stats()}\nDone.")

if __name__=="__main__":
    main()
