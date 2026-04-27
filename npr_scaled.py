"""
Neural Program Reasoner — Scaled + Fast Version

Key speedup: pre-compute ALL GPT-2 representations once at startup.
During training, only look up pre-computed vectors (instant) instead
of running GPT-2 forward pass every time (slow).

This makes training ~10-20x faster.

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
from collections import Counter
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
}

# =============================================================================
# Dataset
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
# Pre-computed Embedding Cache (THE KEY SPEEDUP)
# =============================================================================

class EmbeddingCache:
    """
    Pre-computes GPT-2 representations for ALL phrases we'll ever need.
    During training, we just look up vectors instead of running GPT-2.
    """

    def __init__(self, gpt2_model, tokenizer, relations, layer=8):
        self.cache = {}
        self.device = next(gpt2_model.parameters()).device

        print("\nPre-computing GPT-2 representations...")
        t0 = time.time()

        # Collect all unique phrases
        phrases = set()
        for pairs in relations.values():
            for a, b in pairs:
                phrases.add(f" {a} means {b}")  # example format
                phrases.add(f" {a} means")       # test format

        print(f"  {len(phrases)} unique phrases to encode...")

        # Encode all at once
        gpt2_model.eval()
        with torch.no_grad():
            for i, phrase in enumerate(sorted(phrases)):
                ids = tokenizer.encode(phrase)
                input_ids = torch.tensor([ids], device=self.device)
                out = gpt2_model(input_ids, output_hidden_states=True)
                self.cache[phrase] = out.hidden_states[layer][0, -1, :].clone()

                if (i + 1) % 500 == 0:
                    print(f"  Encoded {i+1}/{len(phrases)}...")

        elapsed = time.time() - t0
        print(f"  Done! {len(self.cache)} phrases cached in {elapsed:.1f}s")
        print(f"  Cache size: {sum(v.numel() for v in self.cache.values()) * 4 / 1e6:.1f} MB")

    def get(self, phrase):
        """Look up a pre-computed representation (instant)."""
        return self.cache[phrase]

    def get_batch(self, phrases):
        """Look up multiple representations."""
        return torch.stack([self.cache[p] for p in phrases])


def generate_task_fast(cache, w2i, relations, num_examples=5, relation_name=None):
    """Generate task using pre-computed cache (no GPT-2 calls)."""
    rel = relation_name or random.choice(list(relations.keys()))
    pairs = relations[rel]
    if len(pairs) < num_examples + 1:
        num_examples = len(pairs) - 1
    chosen = random.sample(pairs, num_examples + 1)
    examples, test = chosen[:num_examples], chosen[num_examples]

    # Look up pre-computed representations directly
    example_reprs = torch.stack([
        cache.get(f" {a} means {b}") for a, b in examples
    ])
    test_repr = cache.get(f" {test[0]} means")

    return {
        "example_reprs": example_reprs,
        "test_repr": test_repr,
        "target_idx": w2i.get(test[1], 0),
        "relation_name": rel,
        "test_word": test[0],
        "expected_word": test[1],
    }


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
        self.negate = nn.Sequential(
            nn.Linear(sd, sd*2), nn.GELU(), nn.Linear(sd*2, sd*2), nn.GELU(), nn.Linear(sd*2, sd))
        self.morph = nn.Sequential(
            nn.Linear(sd, sd), nn.LayerNorm(sd), nn.GELU(), nn.Linear(sd, sd))
        self.assoc_q = nn.Linear(sd, sd//4)
        self.assoc_k = nn.Linear(sd, sd//4)
        self.assoc_v = nn.Linear(sd, sd)
        self.assoc_out = nn.Linear(sd, sd)
        self.lookup = nn.Sequential(
            nn.Linear(sd, sd*2), nn.GELU(), nn.Linear(sd*2, sd*2), nn.GELU(), nn.Linear(sd*2, sd))
        self.blend_ctx = nn.Parameter(torch.randn(sd) * 0.01)
        self.blend_net = nn.Sequential(nn.Linear(sd*2, sd), nn.GELU(), nn.Linear(sd, sd))
        self.gates = nn.ParameterList([
            nn.Parameter(torch.tensor(0.01)), nn.Parameter(torch.tensor(0.5)),
            nn.Parameter(torch.tensor(0.3)),  nn.Parameter(torch.tensor(0.4)),
            nn.Parameter(torch.tensor(0.5)),  nn.Parameter(torch.tensor(0.3))])
        self.inv_prims = nn.ModuleList()
        self.inv_gates = nn.ParameterList()

    @property
    def n(self):
        return len(BASE_NAMES) + len(self.inv_prims)

    def _base(self, i, s):
        if i == 0: return s + torch.sigmoid(self.gates[0]) * self.identity(s)
        if i == 1: return s + torch.sigmoid(self.gates[1]) * self.negate(s)
        if i == 2: return s + torch.sigmoid(self.gates[2]) * self.morph(s)
        if i == 3:
            q, k = self.assoc_q(s), self.assoc_k(s)
            v = self.assoc_v(s)
            return s + torch.sigmoid(self.gates[3]) * self.assoc_out(
                torch.sigmoid(torch.dot(q, k) / math.sqrt(q.shape[0])) * v)
        if i == 4: return s + torch.sigmoid(self.gates[4]) * self.lookup(s)
        if i == 5: return s + torch.sigmoid(self.gates[5]) * self.blend_net(
            torch.cat([s, self.blend_ctx]))

    def apply(self, i, s):
        if i < len(BASE_NAMES): return self._base(i, s)
        j = i - len(BASE_NAMES)
        return s + torch.sigmoid(self.inv_gates[j]) * self.inv_prims[j](s)

    def apply_soft(self, w, s):
        return sum(w[i] * self.apply(i, s) for i in range(self.n))

    def add(self, name, a, b):
        sd = self.sd
        self.inv_prims.append(nn.Sequential(
            nn.Linear(sd, sd*2), nn.LayerNorm(sd*2), nn.GELU(),
            nn.Linear(sd*2, sd*2), nn.GELU(), nn.Linear(sd*2, sd)).to(
            next(self.parameters()).device))
        self.inv_gates.append(nn.Parameter(torch.tensor(0.4, device=next(self.parameters()).device)))
        self.names.append(name)
        print(f"  [COMPRESS] Created: {name}")
        return self.n - 1


# =============================================================================
# Program Memory
# =============================================================================

class Memory:
    def __init__(self, cap=500):
        self.cap, self.entries = cap, []

    def store(self, sig, prog, rel, ok):
        self.entries.append({"sig": sig.detach().clone(), "prog": prog,
                             "rel": rel, "ok": ok, "cnt": 1})
        if len(self.entries) > self.cap:
            self.entries.sort(key=lambda e: e["cnt"], reverse=True)
            self.entries = self.entries[:self.cap]

    def lookup(self, sig, thr=0.85):
        if not self.entries: return None
        best, bsim = None, -1
        for e in self.entries:
            sim = F.cosine_similarity(sig.unsqueeze(0), e["sig"].unsqueeze(0)).item()
            if sim > bsim: bsim, best = sim, e
        if bsim > thr and best:
            best["cnt"] += 1; return best
        return None

    def freq_pairs(self, mn=8):
        c = Counter()
        for e in self.entries:
            for i in range(len(e["prog"])-1):
                c[(e["prog"][i], e["prog"][i+1])] += 1
        return {p: n for p, n in c.items() if n >= mn}

    def clear(self): self.entries = []

    def stats(self):
        if not self.entries: return "Empty"
        ok = sum(1 for e in self.entries if e["ok"])
        return f"{len(self.entries)} entries, {ok} correct ({100*ok//len(self.entries)}%)"


# =============================================================================
# Program Synthesizer (with stop predictor)
# =============================================================================

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
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(pd, 128), nn.GELU(), nn.Linear(128, np))
            for _ in range(max_steps)])
        self.stop = nn.Sequential(nn.Linear(pd + sd, 128), nn.GELU(), nn.Linear(128, 1), nn.Sigmoid())

    def signature(self, ex):
        x = self.proj(ex).unsqueeze(0)
        a, _ = self.a1(x, x, x); x = self.n1(x + a); x = self.n2(x + self.f1(x))
        a, _ = self.a2(x, x, x); x = self.n3(x + a); x = self.n4(x + self.f2(x))
        return x.mean(1).squeeze(0)

    def rebuild(self, new_n):
        old = self._np
        if new_n <= old: return
        nh = nn.ModuleList()
        for h in self.heads:
            n = nn.Sequential(nn.Linear(self.pd, 128), nn.GELU(), nn.Linear(128, new_n))
            dev = next(h.parameters()).device
            n = n.to(dev)
            with torch.no_grad():
                n[0].weight.copy_(h[0].weight); n[0].bias.copy_(h[0].bias)
                n[2].weight[:old].copy_(h[2].weight); n[2].bias[:old].copy_(h[2].bias)
                # Initialize new primitives at the MEAN of existing ones (not near-zero)
                # This gives them a fair chance to compete with trained primitives
                avg_weight = h[2].weight.mean(dim=0, keepdim=True).expand(new_n - old, -1)
                avg_bias = h[2].bias[:old].mean().item()
                n[2].weight[old:].copy_(avg_weight + torch.randn_like(avg_weight) * 0.05)
                n[2].bias[old:].fill_(avg_bias)
            nh.append(n)
        self.heads = nh; self._np = new_n

    def forward(self, ex_reprs, state, temp=0.8, np_=None):
        pat = self.signature(ex_reprs)
        np_ = np_ or self._np
        prog = []
        stop_probs = []  # collect stop probabilities for loss
        for i, h in enumerate(self.heads):
            if i >= CONFIG["min_program_steps"]:
                sp = self.stop(torch.cat([pat, state]))
                stop_probs.append(sp)
                if not self.training and sp.item() > 0.6: break
            lg = h(pat)
            if lg.shape[0] < np_:
                lg = torch.cat([lg, torch.zeros(np_-lg.shape[0], device=lg.device)])
            elif lg.shape[0] > np_:
                lg = lg[:np_]
            if self.training:
                prog.append(F.gumbel_softmax(lg, tau=temp, hard=False))
            else:
                prog.append(F.one_hot(lg.argmax(), np_).float())
        return prog, pat, stop_probs


# =============================================================================
# Executor, Generator
# =============================================================================

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
        self.h = nn.Sequential(
            nn.Linear(sd*2+pd, sd), nn.LayerNorm(sd), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(sd, sd//2), nn.LayerNorm(sd//2), nn.GELU(), nn.Linear(sd//2, vs))

    def forward(self, transformed, original, pattern):
        return self.h(torch.cat([transformed, original, pattern]))


# =============================================================================
# Full Model
# =============================================================================

class NPR(nn.Module):
    def __init__(self, state_dim, vocab_size):
        super().__init__()
        cfg = CONFIG
        self.lib = PrimitiveLibrary(state_dim).to(DEVICE)
        self.syn = Synthesizer(state_dim, len(BASE_NAMES), cfg["max_program_steps"], cfg["proj_dim"]).to(DEVICE)
        self.exe = Executor(state_dim, cfg["max_program_steps"]).to(DEVICE)
        self.gen = Generator(state_dim, cfg["proj_dim"], vocab_size).to(DEVICE)
        self.adapt = nn.Sequential(
            nn.Linear(state_dim, state_dim), nn.LayerNorm(state_dim),
            nn.GELU(), nn.Linear(state_dim, state_dim)).to(DEVICE)
        self.mem = Memory(cfg["memory_capacity"])

    def forward(self, task, temp=0.8, use_mem=True):
        ex = task["example_reprs"].to(DEVICE)
        tr = task["test_repr"].to(DEVICE)
        orig = tr.clone()
        state = self.adapt(tr)
        np_ = self.lib.n

        sig = self.syn.signature(ex)

        from_mem = False
        prog = None
        if use_mem and not self.training:
            c = self.mem.lookup(sig)
            if c:
                prog = [F.one_hot(torch.tensor(i), np_).float().to(DEVICE) for i in c["prog"]]
                from_mem = True

        stop_probs = []
        if prog is None:
            prog, _, stop_probs = self.syn(ex, state, temp, np_)

        final, tr = self.exe(state, prog, self.lib)
        logits = self.gen(final, orig, sig)
        return logits, tr, prog, sig, from_mem, stop_probs

    def memorize(self, sig, prog, rel, ok):
        self.mem.store(sig, [s.argmax().item() for s in prog], rel, ok)

    def compress(self):
        cfg = CONFIG
        freq = self.mem.freq_pairs(cfg["compression_threshold"])
        if not freq:
            print("  [COMPRESS] Nothing found."); return 0
        created = 0
        for (a, b), cnt in sorted(freq.items(), key=lambda x: -x[1]):
            if created >= cfg["max_new_primitives"]: break
            na, nb = self.lib.names[a], self.lib.names[b]
            nm = f"{na}_{nb}"
            if nm in self.lib.names: continue
            print(f"  [COMPRESS] {na} -> {nb} ({cnt}x)")
            self.lib.add(nm, a, b)
            created += 1
        if created > 0:
            self.syn.rebuild(self.lib.n); self.mem.clear()
        return created


# =============================================================================
# GPT-2 Few-Shot Baseline
# =============================================================================

def gpt2_baseline(gpt2, tokenizer, task, w2i, i2w, relations):
    """GPT-2 few-shot: build prompt from same examples, predict next token."""
    rel = task["relation_name"]
    pairs = relations[rel]

    # Rebuild the text prompt from the task info
    # We need the actual word pairs, so we regenerate
    # (This is only called during eval, not training, so it's fine)
    examples_text = []
    all_pairs = relations[rel]
    test_word = task["test_word"]
    target = task["expected_word"]

    # Find some example pairs (not the test pair)
    ex_pairs = [p for p in all_pairs if p[0] != test_word and p[1] != target]
    ex_pairs = ex_pairs[:CONFIG["num_examples"]]

    prompt = "\n".join([f"{a} means {b}" for a, b in ex_pairs])
    prompt += f"\n{test_word} means"

    ids = tokenizer.encode(prompt)
    input_t = torch.tensor([ids], device=DEVICE)

    with torch.no_grad():
        out = gpt2(input_t)
        logits = out.logits[0, -1, :]

    # Find best word from our vocab
    best_w, best_s = None, float('-inf')
    for w in w2i:
        tids = tokenizer.encode(f" {w}")
        if len(tids) == 1:
            s = logits[tids[0]].item()
            if s > best_s: best_s, best_w = s, w

    # Top 3
    top_ids = logits.topk(100).indices.tolist()
    top3 = []
    for tid in top_ids:
        w = tokenizer.decode([tid]).strip().lower()
        if w in w2i and w not in top3:
            top3.append(w)
        if len(top3) >= 3: break

    return best_w or "?", top3


# =============================================================================
# Loss
# =============================================================================

def div_loss(prog):
    if len(prog) < 2: return torch.tensor(0.0, device=DEVICE)
    l, n = torch.tensor(0.0, device=DEVICE), len(prog)
    for i in range(n):
        for j in range(i+1, n):
            l = l + F.relu(F.cosine_similarity(prog[i].unsqueeze(0), prog[j].unsqueeze(0)) - 0.3)
    return l / (n*(n-1)/2)


def use_loss(prog, np_):
    avg = torch.stack(prog).mean(0)[:np_]
    return math.log(np_) - (-(avg * torch.log(avg + 1e-8)).sum())


def novelty_loss(prog, np_, n_base=6):
    """
    Encourage usage of invented primitives (indices >= n_base).
    If we have invented primitives, penalize if they get zero attention.
    """
    if np_ <= n_base:
        return torch.tensor(0.0, device=DEVICE)
    avg = torch.stack(prog).mean(0)[:np_]
    invented_usage = avg[n_base:].sum()
    # We want at least ~15% usage on invented primitives
    target = 0.15
    return F.relu(target - invented_usage)


# =============================================================================
# Training & Eval
# =============================================================================

def fmt(prog, names):
    return " -> ".join(names[s.argmax().item()] if s.argmax().item() < len(names)
                       else "?" for s in prog)


def train_cycle(model, cache, w2i, i2w, rels, n_iters, cycle, lr):
    cfg = CONFIG
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s:
    s/300 if s < 300 else 0.5*(1+math.cos(math.pi*(s-300)/max(n_iters-300,1))))

    model.train()
    ok, tot = 0, 0
    t0 = time.time()

    for it in range(n_iters):
        task = generate_task_fast(cache, w2i, rels, cfg["num_examples"])
        np_ = model.lib.n
        logits, tr, prog, sig, _, stop_probs = model(task, cfg["temperature"], use_mem=False)

        tgt = torch.tensor(task["target_idx"], device=DEVICE)
        lce = F.cross_entropy(logits.unsqueeze(0), tgt.unsqueeze(0))

        # Stop loss: encourage stopping when the program is already correct
        # Push later stop probs toward 1.0 (want to stop) via penalty
        stop_l = torch.tensor(0.0, device=DEVICE)
        if stop_probs:
            for sp in stop_probs:
                stop_l = stop_l + (1.0 - sp).squeeze()  # encourage stopping
            stop_l = stop_l / len(stop_probs)

        loss = (lce + 0.1*div_loss(prog) + 0.05*use_loss(prog, np_) + 0.1*novelty_loss(prog, np_) + 0.03*stop_l + 0.02*len(prog)) / cfg["grad_accumulation"]
        loss.backward()

        pred = i2w[logits.argmax().item()]
        correct = pred == task["expected_word"]
        ok += correct; tot += 1

        with torch.no_grad():
            model.memorize(sig, prog, task["relation_name"], correct)

        if (it+1) % cfg["grad_accumulation"] == 0:
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); opt.zero_grad(); sch.step()

        if it % 500 == 0:
            acc = 100*ok/max(tot,1)
            elapsed = time.time() - t0
            speed = (it+1) / max(elapsed, 0.01)
            with torch.no_grad():
                t = generate_task_fast(cache, w2i, rels, cfg["num_examples"])
                lg, _, pr, _, _, _ = model(t, cfg["temperature"], use_mem=False)
                pw, tw = i2w[lg.argmax().item()], t["expected_word"]
                m = "\u2713" if pw == tw else "\u2717"
                t3 = [i2w[i] for i in lg.topk(min(3, lg.shape[0])).indices.tolist()]
            print(f"  [C{cycle}] {it:5d}/{n_iters} | L:{lce.item():.3f} | "
                  f"Acc:{acc:5.1f}% | {speed:.0f} it/s | "
                  f"[{t['relation_name'][:15]:15s}] '{t['test_word']}' -> '{pw}' "
                  f"(want:'{tw}') {m} K={len(pr)} top3:{t3}")
            print(f"           {fmt(pr, model.lib.names)}")
            if it > 0 and it % 2000 == 0: ok, tot = 0, 0


def evaluate(model, cache, w2i, i2w, rels, gpt2=None, tokenizer=None):
    cfg = CONFIG
    model.eval()
    names = model.lib.names
    npr_res, g2_res = {}, {}
    n_eval = cfg["eval_samples_per_relation"]

    print("\n  --- Programs ---")
    with torch.no_grad():
        for rel in sorted(rels.keys()):
            if len(rels[rel]) < cfg["num_examples"]+1: continue
            t = generate_task_fast(cache, w2i, rels, relation_name=rel)
            _, _, pr, _, _, _ = model(t, 0.1, use_mem=False)
            print(f"    {rel[:22]:22s}: {fmt(pr, names)} (K={len(pr)})")

    print(f"\n  --- Accuracy (n={n_eval}) ---\n")
    hdr = f"    {'Relation':<22s} | {'NPR':>5s} {'t3':>5s}"
    if gpt2: hdr += f" | {'GPT2':>5s} {'t3':>5s}"
    print(hdr)
    print(f"    {'-'*22}-+-{'-'*5}-{'-'*5}" + (f"-+-{'-'*5}-{'-'*5}" if gpt2 else ""))

    with torch.no_grad():
        for rel in sorted(rels.keys()):
            if len(rels[rel]) < cfg["num_examples"]+1: continue
            no, nt, go, gt = 0, 0, 0, 0

            for _ in range(n_eval):
                task = generate_task_fast(cache, w2i, rels, relation_name=rel)
                target = task["expected_word"]

                lg, _, _, _, _, _ = model(task, 0.1, use_mem=True)
                pred = i2w[lg.argmax().item()]
                top3 = [i2w[i] for i in lg.topk(min(3, lg.shape[0])).indices.tolist()]
                no += pred == target
                nt += target in top3

                if gpt2 and tokenizer:
                    gp, gt3 = gpt2_baseline(gpt2, tokenizer, task, w2i, i2w, rels)
                    go += gp == target
                    gt += target in gt3

            na, nt3 = 100*no/n_eval, 100*nt/n_eval
            ga, gt3 = (100*go/n_eval, 100*gt/n_eval) if gpt2 else (0, 0)
            npr_res[rel] = (na, nt3)
            g2_res[rel] = (ga, gt3)

            line = f"    {rel[:22]:22s} | {na:4.0f}% {nt3:4.0f}%"
            if gpt2:
                w = "NPR" if na > ga else ("GPT2" if ga > na else "TIE")
                line += f" | {ga:4.0f}% {gt3:4.0f}%  [{w}]"
            print(line)

    # Summary
    n = len(npr_res)
    if n > 0:
        no = sum(a for a, _ in npr_res.values()) / n
        nt = sum(t for _, t in npr_res.values()) / n
        print(f"\n    NPR  OVERALL: {no:.1f}% (top3: {nt:.1f}%)")
        if gpt2:
            go = sum(a for a, _ in g2_res.values()) / n
            gt_ = sum(t for _, t in g2_res.values()) / n
            print(f"    GPT2 OVERALL: {go:.1f}% (top3: {gt_:.1f}%)")

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

    # Pre-compute all GPT-2 representations (THE KEY SPEEDUP)
    cache = EmbeddingCache(gpt2, tokenizer, rels, layer=CONFIG["perceiver_layer"])

    state_dim = 768  # GPT-2 hidden dim
    model = NPR(state_dim, len(wl))
    tp = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {tp:,}\n")

    cfg = CONFIG
    for c in range(cfg["num_cycles"]):
        print(f"{'='*65}")
        print(f"CYCLE {c+1}/{cfg['num_cycles']} | Prims: {model.lib.n} | "
              f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        print(f"Library: {model.lib.names}")
        print(f"{'='*65}\n")

        train_cycle(model, cache, w2i, i2w, rels,
                    cfg["iters_per_cycle"][c], c+1, cfg["lr_per_cycle"][c])

        print(f"\n--- Eval Cycle {c+1} ---")
        use_g2 = (c == cfg["num_cycles"] - 1)
        evaluate(model, cache, w2i, i2w, rels,
                 gpt2=gpt2 if use_g2 else None, tokenizer=tokenizer if use_g2 else None)
        print(f"\n  Memory: {model.mem.stats()}")

        if c < cfg["num_cycles"] - 1:
            print(f"\n--- Compression ---")
            n = model.compress()
            if n: print(f"  {n} new primitives.")

    print(f"\n{'='*65}")
    print("FINAL TEST (NPR vs GPT-2)")
    print(f"{'='*65}")
    print(f"Library: {model.lib.names}")
    evaluate(model, cache, w2i, i2w, rels, gpt2=gpt2, tokenizer=tokenizer)

    # Stats
    print("\n  --- Primitive Usage ---")
    usage = Counter()
    model.eval()
    with torch.no_grad():
        for _ in range(300):
            t = generate_task_fast(cache, w2i, rels)
            _, _, pr, _, _, _ = model(t, 0.1, use_mem=False)
            for s in pr:
                i = s.argmax().item()
                usage[model.lib.names[i] if i < len(model.lib.names) else f"P{i}"] += 1
    tu = sum(usage.values())
    for nm, cnt in sorted(usage.items(), key=lambda x: -x[1]):
        print(f"    {nm:20s}: {100*cnt/tu:5.1f}%")

    print(f"\n  --- Program Lengths ---")
    lens = Counter()
    with torch.no_grad():
        for _ in range(300):
            t = generate_task_fast(cache, w2i, rels)
            _, _, pr, _, _, _ = model(t, 0.1, use_mem=False)
            lens[len(pr)] += 1
    for k in sorted(lens): print(f"    K={k}: {100*lens[k]/300:.0f}%")

    print(f"\n  Memory: {model.mem.stats()}")
    print("\nDone.")


if __name__ == "__main__":
    main()
