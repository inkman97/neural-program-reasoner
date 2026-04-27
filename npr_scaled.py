"""
Neural Program Reasoner — Phase 2: Scaled + 3 Critical Fixes

Fixes over previous version:
1. COMPOSITIONAL TASKS: "plural of opposite", "past tense of comparative" etc.
   Forces K>2 programs by requiring multi-step reasoning.
2. PRIMITIVE PROBING: Empirically verifies what each primitive does by running
   systematic tests (involution for NEGATE, morphological consistency for MORPH, etc.)
3. GPT-2 FEW-SHOT BASELINE: Already present, kept and improved.

Requirements:
    pip install torch transformers requests
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import os
import requests
from collections import Counter, defaultdict
import time

torch.manual_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    "eval_samples_per_relation": 50,
    "compositional_ratio": 0.3,
}

# =============================================================================
# Dataset: Google Analogy + Compositional Tasks
# =============================================================================

ANALOGY_URL = "https://raw.githubusercontent.com/nicholas-leonard/word2vec/master/questions-words.txt"
ANALOGY_FILE = "questions-words.txt"


def download_analogy_dataset():
    if os.path.exists(ANALOGY_FILE):
        return
    print("Downloading Google Analogy Dataset...")
    r = requests.get(ANALOGY_URL)
    with open(ANALOGY_FILE, "w") as f:
        f.write(r.text)
    print("Done.")


def load_analogy_dataset(max_relations=15):
    download_analogy_dataset()
    relations = {}
    current = None
    with open(ANALOGY_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(":"):
                current = line[2:].strip().lower().replace(" ", "_")
                if current not in relations:
                    relations[current] = set()
                continue
            if not line or current is None:
                continue
            parts = line.lower().split()
            if len(parts) == 4:
                relations[current].add((parts[0], parts[1]))
                relations[current].add((parts[2], parts[3]))
    result = {}
    for name, pairs in relations.items():
        pl = list(pairs)
        if len(pl) >= 10:
            result[name] = pl
    result = dict(sorted(result.items(), key=lambda x: -len(x[1]))[:max_relations])
    total = sum(len(v) for v in result.values())
    print(f"Loaded {len(result)} relations, {total} pairs:")
    for n, p in sorted(result.items()):
        print(f"  {n:30s}: {len(p):4d} pairs  (e.g., {p[0][0]} -> {p[0][1]})")
    return result


def build_vocab(relations):
    words = set()
    for pairs in relations.values():
        for a, b in pairs:
            words.add(a)
            words.add(b)
    wl = sorted(words)
    return wl, {w: i for i, w in enumerate(wl)}, {i: w for i, w in enumerate(wl)}


# =============================================================================
# FIX 1: Compositional Task Generator
# =============================================================================

def find_compositional_tasks(relations):
    """
    Find word chains: word_a --rel_A--> word_b --rel_B--> word_c
    where word_b is shared between two different relations.
    """
    word_to_pairs = defaultdict(list)
    for rel_name, pairs in relations.items():
        for a, b in pairs:
            word_to_pairs[b].append((rel_name, a, b))

    compositional = []
    for rel_b_name, pairs_b in relations.items():
        for b_input, b_output in pairs_b:
            for rel_a_name, a_input, a_output in word_to_pairs.get(b_input, []):
                if rel_a_name != rel_b_name:
                    compositional.append({
                        "word_a": a_input, "word_b": b_input,
                        "word_c": b_output, "rel_a": rel_a_name,
                        "rel_b": rel_b_name,
                        "name": f"{rel_a_name}+{rel_b_name}",
                    })

    by_type = defaultdict(list)
    for c in compositional:
        by_type[c["name"]].append(c)

    valid = {k: v for k, v in by_type.items() if len(v) >= 5}

    if valid:
        print(f"\nFound {len(valid)} compositional task types:")
        for name, tasks in sorted(valid.items(), key=lambda x: -len(x[1]))[:10]:
            t = tasks[0]
            print(f"  {name[:40]:40s}: {len(tasks):3d} chains  "
                  f"(e.g., {t['word_a']} --{t['rel_a'][:10]}--> "
                  f"{t['word_b']} --{t['rel_b'][:10]}--> {t['word_c']})")
    else:
        print("\nNo compositional tasks found.")
    return valid


def generate_compositional_task(cache, w2i, relations, comp_tasks, num_examples=3):
    comp_type = random.choice(list(comp_tasks.keys()))
    chains = comp_tasks[comp_type]
    if len(chains) < num_examples + 1:
        num_examples = len(chains) - 1
    chosen = random.sample(chains, num_examples + 1)
    examples, test = chosen[:num_examples], chosen[num_examples]

    example_reprs = torch.stack([
        cache.get(f" {c['word_a']} means {c['word_c']}") for c in examples])
    test_repr = cache.get(f" {test['word_a']} means")

    return {
        "example_reprs": example_reprs, "test_repr": test_repr,
        "target_idx": w2i.get(test["word_c"], 0),
        "relation_name": f"COMP:{comp_type[:30]}",
        "test_word": test["word_a"], "expected_word": test["word_c"],
        "is_compositional": True, "rel_a": test["rel_a"],
        "rel_b": test["rel_b"], "intermediate": test["word_b"],
    }


# =============================================================================
# EmbeddingCache
# =============================================================================

class EmbeddingCache:
    def __init__(self, gpt2_model, tokenizer, relations, comp_tasks=None, layer=8):
        self.cache = {}
        self.device = next(gpt2_model.parameters()).device
        self.gpt2, self.tokenizer, self.layer = gpt2_model, tokenizer, layer

        print("\nPre-computing GPT-2 representations...")
        t0 = time.time()
        phrases = set()
        for pairs in relations.values():
            for a, b in pairs:
                phrases.add(f" {a} means {b}")
                phrases.add(f" {a} means")
        if comp_tasks:
            for chains in comp_tasks.values():
                for c in chains:
                    phrases.add(f" {c['word_a']} means {c['word_c']}")
                    phrases.add(f" {c['word_a']} means")

        print(f"  {len(phrases)} unique phrases to encode...")
        gpt2_model.eval()
        with torch.no_grad():
            for i, phrase in enumerate(sorted(phrases)):
                ids = tokenizer.encode(phrase)
                input_ids = torch.tensor([ids], device=self.device)
                out = gpt2_model(input_ids, output_hidden_states=True)
                self.cache[phrase] = out.hidden_states[layer][0, -1, :].clone()
                if (i + 1) % 500 == 0:
                    print(f"  Encoded {i+1}/{len(phrases)}...")
        print(f"  Done! {len(self.cache)} phrases cached in {time.time()-t0:.1f}s")

    def get(self, phrase):
        if phrase in self.cache:
            return self.cache[phrase]
        ids = self.tokenizer.encode(phrase)
        input_ids = torch.tensor([ids], device=self.device)
        with torch.no_grad():
            out = self.gpt2(input_ids, output_hidden_states=True)
        vec = out.hidden_states[self.layer][0, -1, :].clone()
        self.cache[phrase] = vec
        return vec


def generate_task_fast(cache, w2i, relations, num_examples=5, relation_name=None):
    rel = relation_name or random.choice(list(relations.keys()))
    pairs = relations[rel]
    if len(pairs) < num_examples + 1:
        num_examples = len(pairs) - 1
    chosen = random.sample(pairs, num_examples + 1)
    examples, test = chosen[:num_examples], chosen[num_examples]
    return {
        "example_reprs": torch.stack([cache.get(f" {a} means {b}") for a, b in examples]),
        "test_repr": cache.get(f" {test[0]} means"),
        "target_idx": w2i.get(test[1], 0),
        "relation_name": rel, "test_word": test[0], "expected_word": test[1],
        "is_compositional": False,
    }


def generate_mixed_task(cache, w2i, relations, comp_tasks, num_examples=5):
    if comp_tasks and random.random() < CONFIG["compositional_ratio"]:
        return generate_compositional_task(cache, w2i, relations, comp_tasks, num_examples=3)
    return generate_task_fast(cache, w2i, relations, num_examples)


# =============================================================================
# Primitive Library
# =============================================================================

BASE_NAMES = ["IDENTITY", "NEGATE", "MORPH", "ASSOCIATE", "LOOKUP", "BLEND"]

class PrimitiveLibrary(nn.Module):
    def __init__(self, sd):
        super().__init__()
        self.sd = sd
        self.names = list(BASE_NAMES)
        self.identity = nn.Sequential(nn.Linear(sd, sd), nn.Tanh())
        self.negate = nn.Sequential(nn.Linear(sd, sd*2), nn.GELU(), nn.Linear(sd*2, sd*2), nn.GELU(), nn.Linear(sd*2, sd))
        self.morph = nn.Sequential(nn.Linear(sd, sd), nn.LayerNorm(sd), nn.GELU(), nn.Linear(sd, sd))
        self.assoc_q, self.assoc_k = nn.Linear(sd, sd//4), nn.Linear(sd, sd//4)
        self.assoc_v, self.assoc_out = nn.Linear(sd, sd), nn.Linear(sd, sd)
        self.lookup = nn.Sequential(nn.Linear(sd, sd*2), nn.GELU(), nn.Linear(sd*2, sd*2), nn.GELU(), nn.Linear(sd*2, sd))
        self.blend_ctx = nn.Parameter(torch.randn(sd) * 0.01)
        self.blend_net = nn.Sequential(nn.Linear(sd*2, sd), nn.GELU(), nn.Linear(sd, sd))
        self.gates = nn.ParameterList([nn.Parameter(torch.tensor(g)) for g in [0.01, 0.5, 0.3, 0.4, 0.5, 0.3]])
        self.inv_prims, self.inv_gates = nn.ModuleList(), nn.ParameterList()

    @property
    def n(self): return len(BASE_NAMES) + len(self.inv_prims)

    def _base(self, i, s):
        if i == 0: return s + torch.sigmoid(self.gates[0]) * self.identity(s)
        if i == 1: return s + torch.sigmoid(self.gates[1]) * self.negate(s)
        if i == 2: return s + torch.sigmoid(self.gates[2]) * self.morph(s)
        if i == 3:
            q, k, v = self.assoc_q(s), self.assoc_k(s), self.assoc_v(s)
            return s + torch.sigmoid(self.gates[3]) * self.assoc_out(torch.sigmoid(torch.dot(q,k)/math.sqrt(q.shape[0])) * v)
        if i == 4: return s + torch.sigmoid(self.gates[4]) * self.lookup(s)
        if i == 5: return s + torch.sigmoid(self.gates[5]) * self.blend_net(torch.cat([s, self.blend_ctx]))

    def apply(self, i, s):
        if i < len(BASE_NAMES): return self._base(i, s)
        return s + torch.sigmoid(self.inv_gates[i-len(BASE_NAMES)]) * self.inv_prims[i-len(BASE_NAMES)](s)

    def apply_soft(self, w, s):
        return sum(w[i] * self.apply(i, s) for i in range(self.n))

    def add(self, name, a, b):
        sd = self.sd
        self.inv_prims.append(nn.Sequential(nn.Linear(sd,sd*2),nn.LayerNorm(sd*2),nn.GELU(),nn.Linear(sd*2,sd*2),nn.GELU(),nn.Linear(sd*2,sd)).to(next(self.parameters()).device))
        self.inv_gates.append(nn.Parameter(torch.tensor(0.4, device=next(self.parameters()).device)))
        self.names.append(name)
        print(f"  [COMPRESS] Created: {name}")
        return self.n - 1


# =============================================================================
# FIX 2: Primitive Probing
# =============================================================================

def probe_primitives(model, cache, relations, w2i, i2w):
    model.eval()
    lib = model.lib
    print("\n  === PRIMITIVE PROBING ===\n")

    all_words = list(w2i.keys())
    test_phrases = [f" {w} means" for w in random.sample(all_words, min(100, len(all_words)))]
    test_vecs = torch.stack([cache.get(p) for p in test_phrases]).to(DEVICE)

    with torch.no_grad():
        # 1. Magnitude
        print("  1. Transformation Magnitude (||p(x)-x|| / ||x||):")
        for i in range(lib.n):
            deltas = [(lib.apply(i,v)-v).norm().item()/max(v.norm().item(),1e-8) for v in test_vecs]
            avg = sum(deltas)/len(deltas)
            name = lib.names[i] if i < len(lib.names) else f"P{i}"
            print(f"    {name:20s}: {avg:.4f} [{'#'*int(avg*20)}]")

        # 2. NEGATE involution
        print("\n  2. NEGATE Involution (||NEGATE(NEGATE(x))-x|| / ||x||):")
        scores = [(lib.apply(1,lib.apply(1,v))-v).norm().item()/max(v.norm().item(),1e-8) for v in test_vecs[:50]]
        avg = sum(scores)/len(scores)
        print(f"    Distance: {avg:.4f}  {'GOOD: near-involution' if avg < 0.5 else 'NOT involution'}")

        # 3. IDENTITY passthrough
        print("\n  3. IDENTITY Passthrough:")
        scores = [(lib.apply(0,v)-v).norm().item()/max(v.norm().item(),1e-8) for v in test_vecs[:50]]
        avg = sum(scores)/len(scores)
        print(f"    Distance: {avg:.4f}  {'GOOD: passthrough' if avg < 0.1 else 'WARNING: modifies'}")

        # 4. MORPH consistency
        print("\n  4. MORPH Consistency (same-relation delta similarity):")
        morph_pairs = [(a,b,r) for r,ps in relations.items() if 'gram' in r for a,b in ps[:5]]
        if morph_pairs:
            sims = []
            for i in range(min(len(morph_pairs)-1, 20)):
                a1,_,r1 = morph_pairs[i]
                d1 = lib.apply(2, cache.get(f" {a1} means")) - cache.get(f" {a1} means")
                for j in range(i+1, min(len(morph_pairs), i+5)):
                    a2,_,r2 = morph_pairs[j]
                    if r1 == r2:
                        d2 = lib.apply(2, cache.get(f" {a2} means")) - cache.get(f" {a2} means")
                        sims.append(F.cosine_similarity(d1.unsqueeze(0), d2.unsqueeze(0)).item())
            if sims:
                avg = sum(sims)/len(sims)
                print(f"    Avg cosine: {avg:.4f}  {'GOOD: consistent' if avg > 0.3 else 'WEAK'}")

        # 5. LOOKUP directional test
        print("\n  5. LOOKUP Retrieval (moves country closer to capital?):")
        if "capital-world" in relations:
            ok = 0
            for a,b in relations["capital-world"][:20]:
                va, vb = cache.get(f" {a} means"), cache.get(f" {b} means")
                lu = lib.apply(4, va)
                ok += (lu-vb).norm().item() < (va-vb).norm().item()
            print(f"    Correct direction: {ok}/20 ({100*ok//20}%)  "
                  f"{'GOOD' if ok > 12 else 'WEAK'}")

        # 6. Distinctness matrix
        print("\n  6. Primitive Distinctness (cosine between avg deltas):")
        n_p = min(lib.n, len(BASE_NAMES))
        avg_d = [torch.stack([lib.apply(i,v)-v for v in test_vecs[:30]]).mean(0) for i in range(n_p)]
        print(f"    {'':12s}", end="")
        for i in range(n_p): print(f" {lib.names[i][:6]:>6s}", end="")
        print()
        for i in range(n_p):
            print(f"    {lib.names[i]:12s}", end="")
            for j in range(n_p):
                print(f" {F.cosine_similarity(avg_d[i].unsqueeze(0),avg_d[j].unsqueeze(0)).item():6.2f}", end="")
            print()

    print("\n  === END PROBING ===\n")


# =============================================================================
# Memory, Synthesizer, Executor, Generator, NPR (same architecture)
# =============================================================================

class Memory:
    def __init__(self, cap=500):
        self.cap, self.entries = cap, []
    def store(self, sig, prog, rel, ok):
        self.entries.append({"sig": sig.detach().clone(), "prog": prog, "rel": rel, "ok": ok, "cnt": 1})
        if len(self.entries) > self.cap:
            self.entries.sort(key=lambda e: e["cnt"], reverse=True)
            self.entries = self.entries[:self.cap]
    def lookup(self, sig, thr=0.85):
        if not self.entries: return None
        best, bsim = None, -1
        for e in self.entries:
            sim = F.cosine_similarity(sig.unsqueeze(0), e["sig"].unsqueeze(0)).item()
            if sim > bsim: bsim, best = sim, e
        if bsim > thr and best: best["cnt"] += 1; return best
        return None
    def freq_pairs(self, mn=8):
        c = Counter()
        for e in self.entries:
            for i in range(len(e["prog"])-1): c[(e["prog"][i], e["prog"][i+1])] += 1
        return {p: n for p, n in c.items() if n >= mn}
    def clear(self): self.entries = []
    def stats(self):
        if not self.entries: return "Empty"
        ok = sum(1 for e in self.entries if e["ok"])
        return f"{len(self.entries)} entries, {ok} correct ({100*ok//len(self.entries)}%)"

class Synthesizer(nn.Module):
    def __init__(self, sd, np, max_steps, pd=256):
        super().__init__()
        self.pd, self.ms, self._np = pd, max_steps, np
        self.proj = nn.Sequential(nn.Linear(sd, pd), nn.LayerNorm(pd), nn.GELU())
        self.a1 = nn.MultiheadAttention(pd, 4, batch_first=True)
        self.n1, self.n2 = nn.LayerNorm(pd), nn.LayerNorm(pd)
        self.f1 = nn.Sequential(nn.Linear(pd, pd*2), nn.GELU(), nn.Linear(pd*2, pd))
        self.a2 = nn.MultiheadAttention(pd, 4, batch_first=True)
        self.n3, self.n4 = nn.LayerNorm(pd), nn.LayerNorm(pd)
        self.f2 = nn.Sequential(nn.Linear(pd, pd*2), nn.GELU(), nn.Linear(pd*2, pd))
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(pd,128),nn.GELU(),nn.Linear(128,np)) for _ in range(max_steps)])
        self.stop = nn.Sequential(nn.Linear(pd+sd,128),nn.GELU(),nn.Linear(128,1),nn.Sigmoid())

    def signature(self, ex):
        x = self.proj(ex).unsqueeze(0)
        a,_ = self.a1(x,x,x); x = self.n1(x+a); x = self.n2(x+self.f1(x))
        a,_ = self.a2(x,x,x); x = self.n3(x+a); x = self.n4(x+self.f2(x))
        return x.mean(1).squeeze(0)

    def rebuild(self, new_n):
        old = self._np
        if new_n <= old: return
        nh = nn.ModuleList()
        for h in self.heads:
            n = nn.Sequential(nn.Linear(self.pd,128),nn.GELU(),nn.Linear(128,new_n))
            n = n.to(next(h.parameters()).device)
            with torch.no_grad():
                n[0].weight.copy_(h[0].weight); n[0].bias.copy_(h[0].bias)
                n[2].weight[:old].copy_(h[2].weight); n[2].bias[:old].copy_(h[2].bias)
                avg_w = h[2].weight.mean(0,keepdim=True).expand(new_n-old,-1)
                n[2].weight[old:].copy_(avg_w + torch.randn_like(avg_w)*0.05)
                n[2].bias[old:].fill_(h[2].bias[:old].mean().item())
            nh.append(n)
        self.heads = nh; self._np = new_n

    def forward(self, ex_reprs, state, temp=0.8, np_=None):
        pat = self.signature(ex_reprs)
        np_ = np_ or self._np
        prog, stop_probs = [], []
        for i, h in enumerate(self.heads):
            if i >= CONFIG["min_program_steps"]:
                sp = self.stop(torch.cat([pat, state]))
                stop_probs.append(sp)
                if not self.training and sp.item() > 0.6: break
            lg = h(pat)
            if lg.shape[0] < np_: lg = torch.cat([lg, torch.zeros(np_-lg.shape[0],device=lg.device)])
            elif lg.shape[0] > np_: lg = lg[:np_]
            if self.training: prog.append(F.gumbel_softmax(lg, tau=temp, hard=False))
            else: prog.append(F.one_hot(lg.argmax(), np_).float())
        return prog, pat, stop_probs

class Executor(nn.Module):
    def __init__(self, sd, ms):
        super().__init__()
        self.emb = nn.Embedding(ms, sd)
    def forward(self, s, prog, lib):
        tr = []
        for i, sel in enumerate(prog):
            s = lib.apply_soft(sel, s + self.emb(torch.tensor(i, device=s.device)))
            tr.append(sel.detach())
        return s, tr

class Generator(nn.Module):
    def __init__(self, sd, pd, vs):
        super().__init__()
        self.h = nn.Sequential(nn.Linear(sd*2+pd,sd),nn.LayerNorm(sd),nn.GELU(),nn.Dropout(0.1),nn.Linear(sd,sd//2),nn.LayerNorm(sd//2),nn.GELU(),nn.Linear(sd//2,vs))
    def forward(self, transformed, original, pattern):
        return self.h(torch.cat([transformed, original, pattern]))

class NPR(nn.Module):
    def __init__(self, state_dim, vocab_size):
        super().__init__()
        cfg = CONFIG
        self.lib = PrimitiveLibrary(state_dim).to(DEVICE)
        self.syn = Synthesizer(state_dim, len(BASE_NAMES), cfg["max_program_steps"], cfg["proj_dim"]).to(DEVICE)
        self.exe = Executor(state_dim, cfg["max_program_steps"]).to(DEVICE)
        self.gen = Generator(state_dim, cfg["proj_dim"], vocab_size).to(DEVICE)
        self.adapt = nn.Sequential(nn.Linear(state_dim,state_dim),nn.LayerNorm(state_dim),nn.GELU(),nn.Linear(state_dim,state_dim)).to(DEVICE)
        self.mem = Memory(cfg["memory_capacity"])

    def forward(self, task, temp=0.8, use_mem=True):
        ex = task["example_reprs"].to(DEVICE)
        tr = task["test_repr"].to(DEVICE)
        orig, state = tr.clone(), self.adapt(tr)
        np_ = self.lib.n
        sig = self.syn.signature(ex)
        from_mem, prog, stop_probs = False, None, []
        if use_mem and not self.training:
            c = self.mem.lookup(sig)
            if c:
                prog = [F.one_hot(torch.tensor(i),np_).float().to(DEVICE) for i in c["prog"]]
                from_mem = True
        if prog is None:
            prog, _, stop_probs = self.syn(ex, state, temp, np_)
        final, tr = self.exe(state, prog, self.lib)
        return self.gen(final, orig, sig), tr, prog, sig, from_mem, stop_probs

    def memorize(self, sig, prog, rel, ok):
        self.mem.store(sig, [s.argmax().item() for s in prog], rel, ok)

    def compress(self):
        freq = self.mem.freq_pairs(CONFIG["compression_threshold"])
        if not freq: print("  [COMPRESS] Nothing found."); return 0
        created = 0
        for (a,b),cnt in sorted(freq.items(), key=lambda x:-x[1]):
            if created >= CONFIG["max_new_primitives"]: break
            na, nb = self.lib.names[a], self.lib.names[b]
            nm = f"{na}_{nb}"
            if nm in self.lib.names: continue
            print(f"  [COMPRESS] {na} -> {nb} ({cnt}x)")
            self.lib.add(nm, a, b); created += 1
        if created > 0: self.syn.rebuild(self.lib.n); self.mem.clear()
        return created


# =============================================================================
# GPT-2 Baseline
# =============================================================================

def gpt2_baseline(gpt2, tokenizer, task, w2i, i2w, relations):
    if task.get("is_compositional"): return "?", []
    rel = task["relation_name"]
    pairs = relations.get(rel, [])
    ex = [p for p in pairs if p[0] != task["test_word"] and p[1] != task["expected_word"]][:CONFIG["num_examples"]]
    if not ex: return "?", []
    prompt = "\n".join([f"{a} means {b}" for a,b in ex]) + f"\n{task['test_word']} means"
    ids = tokenizer.encode(prompt)
    with torch.no_grad():
        logits = gpt2(torch.tensor([ids],device=DEVICE)).logits[0,-1,:]
    best_w, best_s = None, float('-inf')
    for w in w2i:
        tids = tokenizer.encode(f" {w}")
        if len(tids)==1:
            sc = logits[tids[0]].item()
            if sc > best_s: best_s, best_w = sc, w
    top3 = []
    for tid in logits.topk(100).indices.tolist():
        w = tokenizer.decode([tid]).strip().lower()
        if w in w2i and w not in top3: top3.append(w)
        if len(top3) >= 3: break
    return best_w or "?", top3


# =============================================================================
# Losses
# =============================================================================

def div_loss(prog):
    if len(prog)<2: return torch.tensor(0.0,device=DEVICE)
    l, n = torch.tensor(0.0,device=DEVICE), len(prog)
    for i in range(n):
        for j in range(i+1,n):
            l = l + F.relu(F.cosine_similarity(prog[i].unsqueeze(0),prog[j].unsqueeze(0))-0.3)
    return l/(n*(n-1)/2)

def use_loss(prog, np_):
    avg = torch.stack(prog).mean(0)[:np_]
    return math.log(np_) - (-(avg*torch.log(avg+1e-8)).sum())

def novelty_loss(prog, np_, n_base=6):
    if np_ <= n_base: return torch.tensor(0.0,device=DEVICE)
    return F.relu(0.15 - torch.stack(prog).mean(0)[:np_][n_base:].sum())


# =============================================================================
# Training & Eval
# =============================================================================

def fmt(prog, names):
    return " -> ".join(names[s.argmax().item()] if s.argmax().item()<len(names) else "?" for s in prog)

def train_cycle(model, cache, w2i, i2w, rels, comp_tasks, n_iters, cycle, lr):
    cfg = CONFIG
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: s/300 if s<300 else 0.5*(1+math.cos(math.pi*(s-300)/max(n_iters-300,1))))
    model.train()
    ok, tot, cok, ctot = 0, 0, 0, 0
    t0 = time.time()
    for it in range(n_iters):
        task = generate_mixed_task(cache, w2i, rels, comp_tasks, cfg["num_examples"])
        np_ = model.lib.n
        logits, tr, prog, sig, _, stop_probs = model(task, cfg["temperature"], use_mem=False)
        tgt = torch.tensor(task["target_idx"], device=DEVICE)
        lce = F.cross_entropy(logits.unsqueeze(0), tgt.unsqueeze(0))
        stop_l = torch.tensor(0.0, device=DEVICE)
        if stop_probs:
            for sp in stop_probs: stop_l = stop_l + (1.0 - sp).squeeze()
            stop_l = stop_l / len(stop_probs)
        loss = (lce + 0.1*div_loss(prog) + 0.05*use_loss(prog,np_) + 0.1*novelty_loss(prog,np_) + 0.03*stop_l + 0.02*len(prog)) / cfg["grad_accumulation"]
        loss.backward()
        pred = i2w[logits.argmax().item()]
        correct = pred == task["expected_word"]
        ok += correct; tot += 1
        if task.get("is_compositional"): cok += correct; ctot += 1
        with torch.no_grad(): model.memorize(sig, prog, task["relation_name"], correct)
        if (it+1)%cfg["grad_accumulation"]==0:
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); opt.zero_grad(); sch.step()
        if it%500==0:
            acc, cacc = 100*ok/max(tot,1), 100*cok/max(ctot,1) if ctot else 0
            speed = (it+1)/max(time.time()-t0,0.01)
            with torch.no_grad():
                t = generate_mixed_task(cache, w2i, rels, comp_tasks, cfg["num_examples"])
                lg, _, pr, _, _, _ = model(t, cfg["temperature"], use_mem=False)
                pw, tw = i2w[lg.argmax().item()], t["expected_word"]
                ct = " [COMP]" if t.get("is_compositional") else ""
            print(f"  [C{cycle}] {it:5d}/{n_iters} | L:{lce.item():.3f} | Acc:{acc:5.1f}% CompAcc:{cacc:4.0f}% | {speed:.0f}it/s | "
                  f"[{t['relation_name'][:15]:15s}] '{t['test_word']}'->'{pw}' (want:'{tw}') {'V' if pw==tw else 'X'} K={len(pr)}{ct}")
            print(f"           {fmt(pr, model.lib.names)}")
            if it>0 and it%2000==0: ok,tot,cok,ctot = 0,0,0,0

def evaluate(model, cache, w2i, i2w, rels, comp_tasks=None, gpt2=None, tokenizer=None):
    cfg = CONFIG
    model.eval()
    names = model.lib.names
    npr_res, g2_res = {}, {}
    n_eval = cfg["eval_samples_per_relation"]

    print("\n  --- Programs ---")
    with torch.no_grad():
        for rel in sorted(rels.keys()):
            if len(rels[rel])<cfg["num_examples"]+1: continue
            t = generate_task_fast(cache, w2i, rels, relation_name=rel)
            _, _, pr, _, _, _ = model(t, 0.1, use_mem=False)
            print(f"    {rel[:22]:22s}: {fmt(pr, names)} (K={len(pr)})")

    print(f"\n  --- Standard Relations (n={n_eval}) ---\n")
    hdr = f"    {'Relation':<22s} | {'NPR':>5s} {'t3':>5s}"
    if gpt2: hdr += f" | {'GPT2':>5s} {'t3':>5s}"
    print(hdr)
    print(f"    {'-'*22}-+-{'-'*5}-{'-'*5}" + (f"-+-{'-'*5}-{'-'*5}" if gpt2 else ""))

    with torch.no_grad():
        for rel in sorted(rels.keys()):
            if len(rels[rel])<cfg["num_examples"]+1: continue
            no, nt, go, gt = 0, 0, 0, 0
            for _ in range(n_eval):
                task = generate_task_fast(cache, w2i, rels, relation_name=rel)
                lg, _, _, _, _, _ = model(task, 0.1, use_mem=True)
                pred = i2w[lg.argmax().item()]
                top3 = [i2w[i] for i in lg.topk(min(3,lg.shape[0])).indices.tolist()]
                no += pred==task["expected_word"]; nt += task["expected_word"] in top3
                if gpt2 and tokenizer:
                    gp, gt3 = gpt2_baseline(gpt2, tokenizer, task, w2i, i2w, rels)
                    go += gp==task["expected_word"]; gt += task["expected_word"] in gt3
            na, nt3 = 100*no/n_eval, 100*nt/n_eval
            ga, gt3 = (100*go/n_eval, 100*gt/n_eval) if gpt2 else (0,0)
            npr_res[rel], g2_res[rel] = (na,nt3), (ga,gt3)
            line = f"    {rel[:22]:22s} | {na:4.0f}% {nt3:4.0f}%"
            if gpt2: line += f" | {ga:4.0f}% {gt3:4.0f}%  [{'NPR' if na>ga else ('GPT2' if ga>na else 'TIE')}]"
            print(line)

    n = len(npr_res)
    if n:
        print(f"\n    NPR  OVERALL: {sum(a for a,_ in npr_res.values())/n:.1f}% (top3: {sum(t for _,t in npr_res.values())/n:.1f}%)")
        if gpt2: print(f"    GPT2 OVERALL: {sum(a for a,_ in g2_res.values())/n:.1f}% (top3: {sum(t for _,t in g2_res.values())/n:.1f}%)")

    if comp_tasks:
        print(f"\n  --- Compositional Tasks ---\n")
        comp_res = {}
        with torch.no_grad():
            for cn in sorted(comp_tasks.keys()):
                chains = comp_tasks[cn]
                if len(chains)<4: continue
                ok, t3ok, nt = 0, 0, min(30, len(chains)-3)
                shown = 0
                for _ in range(nt):
                    task = generate_compositional_task(cache, w2i, rels, {cn: chains}, 3)
                    lg, _, pr, _, _, _ = model(task, 0.1, use_mem=False)
                    pred, target = i2w[lg.argmax().item()], task["expected_word"]
                    top3 = [i2w[i] for i in lg.topk(min(3,lg.shape[0])).indices.tolist()]
                    ok += pred==target; t3ok += target in top3
                    if shown<2:
                        print(f"    {task['test_word']} --{task['rel_a'][:10]}--> {task['intermediate']} --{task['rel_b'][:10]}--> {target} | pred: {pred} {'V' if pred==target else 'X'} K={len(pr)}")
                        print(f"      {fmt(pr, names)}")
                        shown += 1
                a, t3 = 100*ok/nt, 100*t3ok/nt
                comp_res[cn] = (a, t3)
                print(f"    {cn[:35]:35s}: {a:4.0f}% (top3: {t3:4.0f}%)")
        if comp_res:
            print(f"\n    COMP OVERALL: {sum(a for a,_ in comp_res.values())/len(comp_res):.1f}% "
                  f"(top3: {sum(t for _,t in comp_res.values())/len(comp_res):.1f}%)")
    return npr_res, g2_res


# =============================================================================
# Main
# =============================================================================

def main():
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    print(f"Device: {DEVICE}")
    print("Loading GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2 = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE).eval()

    rels = load_analogy_dataset(max_relations=12)
    wl, w2i, i2w = build_vocab(rels)
    print(f"\nVocab: {len(wl)} words | Primitives: {BASE_NAMES}")

    comp_tasks = find_compositional_tasks(rels)
    cache = EmbeddingCache(gpt2, tokenizer, rels, comp_tasks, layer=CONFIG["perceiver_layer"])

    model = NPR(768, len(wl))
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

    cfg = CONFIG
    for c in range(cfg["num_cycles"]):
        print(f"{'='*65}")
        print(f"CYCLE {c+1}/{cfg['num_cycles']} | Prims: {model.lib.n} | "
              f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        print(f"Library: {model.lib.names}")
        print(f"{'='*65}\n")
        train_cycle(model, cache, w2i, i2w, rels, comp_tasks, cfg["iters_per_cycle"][c], c+1, cfg["lr_per_cycle"][c])
        print(f"\n--- Eval Cycle {c+1} ---")
        evaluate(model, cache, w2i, i2w, rels, comp_tasks, gpt2=gpt2 if c==cfg["num_cycles"]-1 else None, tokenizer=tokenizer if c==cfg["num_cycles"]-1 else None)
        print(f"\n  Memory: {model.mem.stats()}")
        if c < cfg["num_cycles"]-1:
            print(f"\n--- Compression ---")
            n = model.compress()
            if n: print(f"  {n} new primitives.")

    print(f"\n{'='*65}\nFINAL TEST\n{'='*65}")
    print(f"Library: {model.lib.names}")
    evaluate(model, cache, w2i, i2w, rels, comp_tasks, gpt2=gpt2, tokenizer=tokenizer)

    probe_primitives(model, cache, rels, w2i, i2w)

    print("\n  --- Primitive Usage ---")
    usage = Counter()
    model.eval()
    with torch.no_grad():
        for _ in range(300):
            t = generate_task_fast(cache, w2i, rels)
            _, _, pr, _, _, _ = model(t, 0.1, use_mem=False)
            for s in pr:
                i = s.argmax().item()
                usage[model.lib.names[i] if i<len(model.lib.names) else f"P{i}"] += 1
    tu = sum(usage.values())
    for nm, cnt in sorted(usage.items(), key=lambda x:-x[1]):
        print(f"    {nm:20s}: {100*cnt/tu:5.1f}%")

    print(f"\n  --- Program Lengths ---")
    lens = Counter()
    with torch.no_grad():
        for _ in range(300):
            t = generate_mixed_task(cache, w2i, rels, comp_tasks)
            _, _, pr, _, _, _ = model(t, 0.1, use_mem=False)
            lens[len(pr)] += 1
    for k in sorted(lens): print(f"    K={k}: {100*lens[k]/300:.0f}%")

    print(f"\n  Memory: {model.mem.stats()}\nDone.")

if __name__ == "__main__":
    main()
