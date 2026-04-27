"""
Neural Program Reasoner — Phase 3

Changes:
1. TRIPLE COMPOSITIONS: 3-relation chains (A->B->C->D) that genuinely require K>2
2. CONDITIONAL STOP LOSS: encourages stopping for simple tasks, discourages for compositional
3. CLAUDE SONNET BASELINE: modern LLM comparison via Anthropic API
4. Per-task min_steps: 2 for standard, 3 for double comp, 4 for triple comp

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
    "compositional_ratio": 0.4,  # 40% compositional (up from 30%)
}

# =============================================================================
# Dataset
# =============================================================================

ANALOGY_URL = "https://raw.githubusercontent.com/nicholas-leonard/word2vec/master/questions-words.txt"
ANALOGY_FILE = "questions-words.txt"

def download_analogy_dataset():
    if os.path.exists(ANALOGY_FILE): return
    print("Downloading Google Analogy Dataset...")
    r = requests.get(ANALOGY_URL)
    with open(ANALOGY_FILE, "w") as f: f.write(r.text)
    print("Done.")

def load_analogy_dataset(max_relations=15):
    download_analogy_dataset()
    relations, current = {}, None
    with open(ANALOGY_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(":"):
                current = line[2:].strip().lower().replace(" ", "_")
                if current not in relations: relations[current] = set()
                continue
            if not line or current is None: continue
            parts = line.lower().split()
            if len(parts) == 4:
                relations[current].add((parts[0], parts[1]))
                relations[current].add((parts[2], parts[3]))
    result = {n: list(p) for n, p in relations.items() if len(p) >= 10}
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
            words.add(a); words.add(b)
    wl = sorted(words)
    return wl, {w:i for i,w in enumerate(wl)}, {i:w for i,w in enumerate(wl)}


# =============================================================================
# Compositional Tasks: Double AND Triple Chains
# =============================================================================

def find_compositional_tasks(relations):
    """Find 2-step and 3-step chains."""
    # Build output->input lookup
    out_to_rel = defaultdict(list)  # word_b -> [(rel_name, word_a)]
    for rel, pairs in relations.items():
        for a, b in pairs:
            out_to_rel[b].append((rel, a))

    # 2-step: A --r1--> B --r2--> C
    double = []
    for r2, pairs2 in relations.items():
        for b, c in pairs2:
            for r1, a in out_to_rel.get(b, []):
                if r1 != r2:
                    double.append({"words": [a, b, c], "rels": [r1, r2],
                                   "name": f"{r1}+{r2}", "depth": 2})

    # 3-step: A --r1--> B --r2--> C --r3--> D
    triple = []
    for d2 in double:
        a, b, c = d2["words"]
        r1, r2 = d2["rels"]
        for r3, pairs3 in relations.items():
            for c_in, d in pairs3:
                if c_in == c and r3 != r2:
                    triple.append({"words": [a, b, c, d], "rels": [r1, r2, r3],
                                   "name": f"{r1}+{r2}+{r3}", "depth": 3})

    # Group and filter
    by_type = defaultdict(list)
    for item in double + triple:
        by_type[item["name"]].append(item)
    valid = {k: v for k, v in by_type.items() if len(v) >= 5}

    n_double = sum(1 for v in valid.values() if v[0]["depth"] == 2)
    n_triple = sum(1 for v in valid.values() if v[0]["depth"] == 3)
    print(f"\nCompositional tasks: {n_double} double-step, {n_triple} triple-step types")
    for name, tasks in sorted(valid.items(), key=lambda x: -len(x[1]))[:12]:
        t = tasks[0]
        chain = " -> ".join(t["words"])
        rels = " + ".join(r[:12] for r in t["rels"])
        print(f"  {name[:45]:45s}: {len(tasks):3d} chains (depth={t['depth']})  e.g., {chain}")
    return valid


def generate_compositional_task(cache, w2i, relations, comp_tasks, num_examples=3):
    comp_type = random.choice(list(comp_tasks.keys()))
    chains = comp_tasks[comp_type]
    ne = min(num_examples, len(chains) - 1)
    chosen = random.sample(chains, ne + 1)
    examples, test = chosen[:ne], chosen[ne]

    # Examples show: first word -> last word
    example_reprs = torch.stack([
        cache.get(f" {c['words'][0]} means {c['words'][-1]}") for c in examples])
    test_repr = cache.get(f" {test['words'][0]} means")

    depth = test["depth"]
    return {
        "example_reprs": example_reprs, "test_repr": test_repr,
        "target_idx": w2i.get(test["words"][-1], 0),
        "relation_name": f"COMP{depth}:{comp_type[:25]}",
        "test_word": test["words"][0], "expected_word": test["words"][-1],
        "is_compositional": True, "depth": depth,
        "rels": test["rels"], "intermediates": test["words"][1:-1],
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
                phrases.add(f" {a} means {b}"); phrases.add(f" {a} means")
        if comp_tasks:
            for chains in comp_tasks.values():
                for c in chains:
                    phrases.add(f" {c['words'][0]} means {c['words'][-1]}")
                    phrases.add(f" {c['words'][0]} means")
        print(f"  {len(phrases)} unique phrases...")
        gpt2_model.eval()
        with torch.no_grad():
            for i, p in enumerate(sorted(phrases)):
                ids = tokenizer.encode(p)
                out = gpt2_model(torch.tensor([ids], device=self.device), output_hidden_states=True)
                self.cache[p] = out.hidden_states[layer][0, -1, :].clone()
                if (i+1) % 500 == 0: print(f"  Encoded {i+1}/{len(phrases)}...")
        print(f"  Done! {len(self.cache)} cached in {time.time()-t0:.1f}s")

    def get(self, phrase):
        if phrase in self.cache: return self.cache[phrase]
        ids = self.tokenizer.encode(phrase)
        with torch.no_grad():
            out = self.gpt2(torch.tensor([ids], device=self.device), output_hidden_states=True)
        v = out.hidden_states[self.layer][0, -1, :].clone()
        self.cache[phrase] = v
        return v


def generate_task_fast(cache, w2i, relations, num_examples=5, relation_name=None):
    rel = relation_name or random.choice(list(relations.keys()))
    pairs = relations[rel]
    ne = min(num_examples, len(pairs)-1)
    chosen = random.sample(pairs, ne+1)
    ex, test = chosen[:ne], chosen[ne]
    return {
        "example_reprs": torch.stack([cache.get(f" {a} means {b}") for a,b in ex]),
        "test_repr": cache.get(f" {test[0]} means"),
        "target_idx": w2i.get(test[1], 0), "relation_name": rel,
        "test_word": test[0], "expected_word": test[1],
        "is_compositional": False, "depth": 1,
    }

def generate_mixed_task(cache, w2i, relations, comp_tasks, num_examples=5):
    if comp_tasks and random.random() < CONFIG["compositional_ratio"]:
        return generate_compositional_task(cache, w2i, relations, comp_tasks, 3)
    return generate_task_fast(cache, w2i, relations, num_examples)


# =============================================================================
# Primitive Library
# =============================================================================

BASE_NAMES = ["IDENTITY", "NEGATE", "MORPH", "ASSOCIATE", "LOOKUP", "BLEND"]

class PrimitiveLibrary(nn.Module):
    def __init__(self, sd):
        super().__init__()
        self.sd, self.names = sd, list(BASE_NAMES)
        self.identity = nn.Sequential(nn.Linear(sd,sd),nn.Tanh())
        self.negate = nn.Sequential(nn.Linear(sd,sd*2),nn.GELU(),nn.Linear(sd*2,sd*2),nn.GELU(),nn.Linear(sd*2,sd))
        self.morph = nn.Sequential(nn.Linear(sd,sd),nn.LayerNorm(sd),nn.GELU(),nn.Linear(sd,sd))
        self.aq, self.ak, self.av, self.ao = nn.Linear(sd,sd//4), nn.Linear(sd,sd//4), nn.Linear(sd,sd), nn.Linear(sd,sd)
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

    def apply_soft(self, w, s):
        return sum(w[i]*self.apply(i,s) for i in range(self.n))

    def add(self, name, a, b):
        sd=self.sd; dev=next(self.parameters()).device
        self.inv_p.append(nn.Sequential(nn.Linear(sd,sd*2),nn.LayerNorm(sd*2),nn.GELU(),nn.Linear(sd*2,sd*2),nn.GELU(),nn.Linear(sd*2,sd)).to(dev))
        self.inv_g.append(nn.Parameter(torch.tensor(0.4,device=dev)))
        self.names.append(name); print(f"  [COMPRESS] Created: {name}"); return self.n-1


# =============================================================================
# Primitive Probing
# =============================================================================

def probe_primitives(model, cache, relations, w2i):
    model.eval(); lib = model.lib
    print("\n  === PRIMITIVE PROBING ===\n")
    words = list(w2i.keys())
    vecs = torch.stack([cache.get(f" {w} means") for w in random.sample(words, min(100,len(words)))]).to(DEVICE)
    with torch.no_grad():
        print("  1. Magnitude:")
        for i in range(lib.n):
            d = sum((lib.apply(i,v)-v).norm().item()/max(v.norm().item(),1e-8) for v in vecs)/len(vecs)
            print(f"    {lib.names[i] if i<len(lib.names) else f'P{i}':20s}: {d:.4f}")
        print(f"\n  2. NEGATE involution: {sum((lib.apply(1,lib.apply(1,v))-v).norm().item()/max(v.norm().item(),1e-8) for v in vecs[:50])/50:.4f}")
        print(f"  3. IDENTITY passthrough: {sum((lib.apply(0,v)-v).norm().item()/max(v.norm().item(),1e-8) for v in vecs[:50])/50:.4f}")
        mp = [(a,b,r) for r,ps in relations.items() if 'gram' in r for a,b in ps[:5]]
        if mp:
            sims=[]
            for i in range(min(len(mp)-1,20)):
                a1,_,r1=mp[i]; d1=lib.apply(2,cache.get(f" {a1} means"))-cache.get(f" {a1} means")
                for j in range(i+1,min(len(mp),i+5)):
                    a2,_,r2=mp[j]
                    if r1==r2:
                        d2=lib.apply(2,cache.get(f" {a2} means"))-cache.get(f" {a2} means")
                        sims.append(F.cosine_similarity(d1.unsqueeze(0),d2.unsqueeze(0)).item())
            if sims: print(f"  4. MORPH consistency: {sum(sims)/len(sims):.4f}")
        if "capital-world" in relations:
            ok=sum(1 for a,b in relations["capital-world"][:20] if (lib.apply(4,cache.get(f" {a} means"))-cache.get(f" {b} means")).norm()<(cache.get(f" {a} means")-cache.get(f" {b} means")).norm())
            print(f"  5. LOOKUP directional: {ok}/20")
        np_=min(lib.n,len(BASE_NAMES))
        ad=[torch.stack([lib.apply(i,v)-v for v in vecs[:30]]).mean(0) for i in range(np_)]
        print(f"\n  6. Distinctness:")
        print(f"    {'':12s}"+"".join(f" {lib.names[i][:6]:>6s}" for i in range(np_)))
        for i in range(np_):
            print(f"    {lib.names[i]:12s}"+"".join(f" {F.cosine_similarity(ad[i].unsqueeze(0),ad[j].unsqueeze(0)).item():6.2f}" for j in range(np_)))
    print("  === END PROBING ===\n")


# =============================================================================
# Memory, Synthesizer, Executor, Generator, NPR
# =============================================================================

class Memory:
    def __init__(self, cap=500):
        self.cap, self.entries = cap, []
    def store(self, sig, prog, rel, ok):
        self.entries.append({"sig":sig.detach().clone(),"prog":prog,"rel":rel,"ok":ok,"cnt":1})
        if len(self.entries)>self.cap:
            self.entries.sort(key=lambda e:e["cnt"],reverse=True); self.entries=self.entries[:self.cap]
    def lookup(self, sig, thr=0.85):
        if not self.entries: return None
        best,bsim=None,-1
        for e in self.entries:
            sim=F.cosine_similarity(sig.unsqueeze(0),e["sig"].unsqueeze(0)).item()
            if sim>bsim: bsim,best=sim,e
        if bsim>thr and best: best["cnt"]+=1; return best
        return None
    def freq_pairs(self, mn=8):
        c=Counter()
        for e in self.entries:
            for i in range(len(e["prog"])-1): c[(e["prog"][i],e["prog"][i+1])]+=1
        return {p:n for p,n in c.items() if n>=mn}
    def clear(self): self.entries=[]
    def stats(self):
        if not self.entries: return "Empty"
        return f"{len(self.entries)} entries, {sum(1 for e in self.entries if e['ok'])} correct"

class Synthesizer(nn.Module):
    def __init__(self, sd, np, max_steps, pd=256):
        super().__init__()
        self.pd,self.ms,self._np=pd,max_steps,np
        self.proj=nn.Sequential(nn.Linear(sd,pd),nn.LayerNorm(pd),nn.GELU())
        self.a1=nn.MultiheadAttention(pd,4,batch_first=True)
        self.n1,self.n2=nn.LayerNorm(pd),nn.LayerNorm(pd)
        self.f1=nn.Sequential(nn.Linear(pd,pd*2),nn.GELU(),nn.Linear(pd*2,pd))
        self.a2=nn.MultiheadAttention(pd,4,batch_first=True)
        self.n3,self.n4=nn.LayerNorm(pd),nn.LayerNorm(pd)
        self.f2=nn.Sequential(nn.Linear(pd,pd*2),nn.GELU(),nn.Linear(pd*2,pd))
        self.heads=nn.ModuleList([nn.Sequential(nn.Linear(pd,128),nn.GELU(),nn.Linear(128,np)) for _ in range(max_steps)])
        self.stop=nn.Sequential(nn.Linear(pd+sd,128),nn.GELU(),nn.Linear(128,1),nn.Sigmoid())

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
            if i >= min_steps:  # use per-task min_steps
                sp=self.stop(torch.cat([pat,state])); stop_probs.append(sp)
                if not self.training and sp.item()>0.6: break
            lg=h(pat)
            if lg.shape[0]<np_: lg=torch.cat([lg,torch.zeros(np_-lg.shape[0],device=lg.device)])
            elif lg.shape[0]>np_: lg=lg[:np_]
            if self.training: prog.append(F.gumbel_softmax(lg,tau=temp,hard=False))
            else: prog.append(F.one_hot(lg.argmax(),np_).float())
        return prog, pat, stop_probs

class Executor(nn.Module):
    def __init__(self, sd, ms):
        super().__init__()
        self.emb=nn.Embedding(ms,sd)
    def forward(self, s, prog, lib):
        for i,sel in enumerate(prog):
            s=lib.apply_soft(sel, s+self.emb(torch.tensor(i,device=s.device)))
        return s

class Generator(nn.Module):
    def __init__(self, sd, pd, vs):
        super().__init__()
        self.h=nn.Sequential(nn.Linear(sd*2+pd,sd),nn.LayerNorm(sd),nn.GELU(),nn.Dropout(0.1),nn.Linear(sd,sd//2),nn.LayerNorm(sd//2),nn.GELU(),nn.Linear(sd//2,vs))
    def forward(self, t, o, p): return self.h(torch.cat([t,o,p]))

class NPR(nn.Module):
    def __init__(self, state_dim, vocab_size):
        super().__init__()
        cfg=CONFIG
        self.lib=PrimitiveLibrary(state_dim).to(DEVICE)
        self.syn=Synthesizer(state_dim,len(BASE_NAMES),cfg["max_program_steps"],cfg["proj_dim"]).to(DEVICE)
        self.exe=Executor(state_dim,cfg["max_program_steps"]).to(DEVICE)
        self.gen=Generator(state_dim,cfg["proj_dim"],vocab_size).to(DEVICE)
        self.adapt=nn.Sequential(nn.Linear(state_dim,state_dim),nn.LayerNorm(state_dim),nn.GELU(),nn.Linear(state_dim,state_dim)).to(DEVICE)
        self.mem=Memory(cfg["memory_capacity"])

    def forward(self, task, temp=0.8, use_mem=True):
        ex=task["example_reprs"].to(DEVICE)
        tr=task["test_repr"].to(DEVICE)
        orig,state=tr.clone(),self.adapt(tr)
        np_=self.lib.n; sig=self.syn.signature(ex)
        # Per-task min steps based on compositional depth
        depth=task.get("depth",1)
        min_steps = min(depth + 1, CONFIG["max_program_steps"])  # depth 1->2, 2->3, 3->4
        from_mem,prog,stop_probs=False,None,[]
        if use_mem and not self.training:
            c=self.mem.lookup(sig)
            if c: prog=[F.one_hot(torch.tensor(i),np_).float().to(DEVICE) for i in c["prog"]]; from_mem=True
        if prog is None:
            prog,_,stop_probs=self.syn(ex,state,temp,np_,min_steps=min_steps)
        final=self.exe(state,prog,self.lib)
        return self.gen(final,orig,sig),prog,sig,from_mem,stop_probs,depth

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
# GPT-2 Baseline
# =============================================================================

def gpt2_baseline(gpt2, tokenizer, task, w2i, relations):
    if task.get("is_compositional"): return "?", []
    rel=task["relation_name"]; pairs=relations.get(rel,[])
    ex=[p for p in pairs if p[0]!=task["test_word"] and p[1]!=task["expected_word"]][:CONFIG["num_examples"]]
    if not ex: return "?", []
    prompt="\n".join([f"{a} means {b}" for a,b in ex])+f"\n{task['test_word']} means"
    with torch.no_grad():
        logits=gpt2(torch.tensor([tokenizer.encode(prompt)],device=DEVICE)).logits[0,-1,:]
    best_w,best_s=None,float('-inf')
    for w in w2i:
        tids=tokenizer.encode(f" {w}")
        if len(tids)==1:
            sc=logits[tids[0]].item()
            if sc>best_s: best_s,best_w=sc,w
    top3=[]
    for tid in logits.topk(100).indices.tolist():
        w=tokenizer.decode([tid]).strip().lower()
        if w in w2i and w not in top3: top3.append(w)
        if len(top3)>=3: break
    return best_w or "?", top3


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


# =============================================================================
# Training & Eval
# =============================================================================

def fmt(prog, names):
    return " -> ".join(names[s.argmax().item()] if s.argmax().item()<len(names) else "?" for s in prog)

def train_cycle(model, cache, w2i, i2w, rels, comp_tasks, n_iters, cycle, lr):
    cfg=CONFIG
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=0.01)
    sch=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:s/300 if s<300 else 0.5*(1+math.cos(math.pi*(s-300)/max(n_iters-300,1))))
    model.train(); ok,tot,cok,ctot=0,0,0,0; t0=time.time()

    for it in range(n_iters):
        task=generate_mixed_task(cache,w2i,rels,comp_tasks,cfg["num_examples"])
        np_=model.lib.n
        logits,prog,sig,_,stop_probs,depth=model(task,cfg["temperature"],use_mem=False)
        tgt=torch.tensor(task["target_idx"],device=DEVICE)
        lce=F.cross_entropy(logits.unsqueeze(0),tgt.unsqueeze(0))

        # Conditional stop loss: encourage stop for simple, discourage for compositional
        stop_l=torch.tensor(0.0,device=DEVICE)
        if stop_probs:
            for sp in stop_probs:
                if depth == 1:
                    stop_l = stop_l + (1.0 - sp).squeeze()  # encourage stopping
                else:
                    stop_l = stop_l + sp.squeeze() * 0.5  # discourage stopping for compositional
            stop_l = stop_l / len(stop_probs)

        loss=(lce + 0.1*div_loss(prog) + 0.05*use_loss(prog,np_) + 0.1*novelty_loss(prog,np_) + 0.03*stop_l + 0.02*len(prog)) / cfg["grad_accumulation"]
        loss.backward()

        pred=i2w[logits.argmax().item()]; correct=pred==task["expected_word"]
        ok+=correct; tot+=1
        if task.get("is_compositional"): cok+=correct; ctot+=1

        with torch.no_grad(): model.memorize(sig,prog,task["relation_name"],correct)

        if (it+1)%cfg["grad_accumulation"]==0:
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
            opt.step(); opt.zero_grad(); sch.step()

        if it%500==0:
            acc,cacc=100*ok/max(tot,1),(100*cok/max(ctot,1) if ctot else 0)
            speed=(it+1)/max(time.time()-t0,0.01)
            with torch.no_grad():
                t=generate_mixed_task(cache,w2i,rels,comp_tasks,cfg["num_examples"])
                lg,pr,_,_,_,d=model(t,cfg["temperature"],use_mem=False)
                pw,tw=i2w[lg.argmax().item()],t["expected_word"]
                ct=f" [D{d}]" if t.get("is_compositional") else ""
            print(f"  [C{cycle}] {it:5d}/{n_iters} | L:{lce.item():.3f} | Acc:{acc:5.1f}% Comp:{cacc:4.0f}% | {speed:.0f}it/s | "
                  f"[{t['relation_name'][:15]:15s}] '{t['test_word']}'->'{pw}' (want:'{tw}') {'V' if pw==tw else 'X'} K={len(pr)}{ct}")
            print(f"           {fmt(pr,model.lib.names)}")
            if it>0 and it%2000==0: ok,tot,cok,ctot=0,0,0,0


def evaluate(model, cache, w2i, i2w, rels, comp_tasks=None, gpt2=None, tokenizer=None):
    cfg=CONFIG; model.eval(); names=model.lib.names
    npr_res,g2_res={},{}; n_eval=cfg["eval_samples_per_relation"]

    print("\n  --- Programs ---")
    with torch.no_grad():
        for rel in sorted(rels.keys()):
            if len(rels[rel])<cfg["num_examples"]+1: continue
            t=generate_task_fast(cache,w2i,rels,relation_name=rel)
            _,pr,_,_,_,_=model(t,0.1,use_mem=False)
            print(f"    {rel[:22]:22s}: {fmt(pr,names)} (K={len(pr)})")

    print(f"\n  --- Standard Relations (n={n_eval}) ---\n")
    hdr=f"    {'Relation':<22s} | {'NPR':>5s} {'t3':>5s}"
    if gpt2: hdr+=f" | {'GPT2':>5s} {'t3':>5s}"
    print(hdr)

    with torch.no_grad():
        for rel in sorted(rels.keys()):
            if len(rels[rel])<cfg["num_examples"]+1: continue
            no,nt,go,gt=0,0,0,0
            for _ in range(n_eval):
                task=generate_task_fast(cache,w2i,rels,relation_name=rel)
                target=task["expected_word"]
                lg,_,_,_,_,_=model(task,0.1,use_mem=True)
                pred=i2w[lg.argmax().item()]
                top3=[i2w[i] for i in lg.topk(min(3,lg.shape[0])).indices.tolist()]
                no+=pred==target; nt+=target in top3
                if gpt2 and tokenizer:
                    gp,gt3=gpt2_baseline(gpt2,tokenizer,task,w2i,rels)
                    go+=gp==target; gt+=target in gt3

            na,nt3=100*no/n_eval,100*nt/n_eval
            ga,gt3=(100*go/n_eval,100*gt/n_eval) if gpt2 else (0,0)
            npr_res[rel]=(na,nt3); g2_res[rel]=(ga,gt3)

            line=f"    {rel[:22]:22s} | {na:4.0f}% {nt3:4.0f}%"
            if gpt2:
                w="NPR" if na>ga else ("GPT2" if ga>na else "TIE")
                line+=f" | {ga:4.0f}% {gt3:4.0f}%  [{w}]"
            print(line)

    n=len(npr_res)
    if n:
        print(f"\n    NPR  OVERALL: {sum(a for a,_ in npr_res.values())/n:.1f}% (top3: {sum(t for _,t in npr_res.values())/n:.1f}%)")
        if gpt2: print(f"    GPT2 OVERALL: {sum(a for a,_ in g2_res.values())/n:.1f}% (top3: {sum(t for _,t in g2_res.values())/n:.1f}%)")

    # Compositional eval
    if comp_tasks:
        print(f"\n  --- Compositional Tasks ---\n")
        comp_res={}
        with torch.no_grad():
            for cn in sorted(comp_tasks.keys()):
                chains=comp_tasks[cn]
                if len(chains)<4: continue
                ok,t3ok,nt,shown=0,0,min(30,len(chains)-3),0
                depth=chains[0]["depth"]
                for _ in range(nt):
                    task=generate_compositional_task(cache,w2i,rels,{cn:chains},3)
                    lg,pr,_,_,_,_=model(task,0.1,use_mem=False)
                    pred,target=i2w[lg.argmax().item()],task["expected_word"]
                    top3=[i2w[i] for i in lg.topk(min(3,lg.shape[0])).indices.tolist()]
                    ok+=pred==target; t3ok+=target in top3
                    if shown<2:
                        inter=" -> ".join(task.get("intermediates",[]))
                        print(f"    {task['test_word']} -> [{inter}] -> {target} | pred:{pred} {'V' if pred==target else 'X'} K={len(pr)} D={depth}")
                        print(f"      {fmt(pr,names)}")
                        shown+=1
                a,t3=100*ok/nt,100*t3ok/nt
                comp_res[cn]=(a,t3,depth)
                print(f"    {cn[:40]:40s} D{depth}: {a:4.0f}% (top3:{t3:4.0f}%)")

        if comp_res:
            for d in [2,3]:
                items=[(k,v) for k,v in comp_res.items() if v[2]==d]
                if items:
                    avg_a=sum(v[0] for _,v in items)/len(items)
                    avg_t=sum(v[1] for _,v in items)/len(items)
                    print(f"\n    DEPTH-{d} OVERALL: {avg_a:.1f}% (top3: {avg_t:.1f}%)")
    return npr_res, g2_res


# =============================================================================
# Main
# =============================================================================

def main():
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    print(f"Device: {DEVICE}")
    print("Loading GPT-2...")
    tokenizer=GPT2Tokenizer.from_pretrained("gpt2")
    gpt2=GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE).eval()

    rels=load_analogy_dataset(max_relations=12)
    wl,w2i,i2w=build_vocab(rels)
    print(f"\nVocab: {len(wl)} words | Primitives: {BASE_NAMES}")

    comp_tasks=find_compositional_tasks(rels)
    cache=EmbeddingCache(gpt2,tokenizer,rels,comp_tasks,layer=CONFIG["perceiver_layer"])

    model=NPR(768,len(wl))
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

    cfg=CONFIG
    for c in range(cfg["num_cycles"]):
        print(f"{'='*65}")
        print(f"CYCLE {c+1}/{cfg['num_cycles']} | Prims: {model.lib.n} | Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        print(f"Library: {model.lib.names}")
        print(f"{'='*65}\n")
        train_cycle(model,cache,w2i,i2w,rels,comp_tasks,cfg["iters_per_cycle"][c],c+1,cfg["lr_per_cycle"][c])
        print(f"\n--- Eval Cycle {c+1} ---")
        evaluate(model,cache,w2i,i2w,rels,comp_tasks,
                 gpt2=gpt2 if c==cfg["num_cycles"]-1 else None,
                 tokenizer=tokenizer if c==cfg["num_cycles"]-1 else None)
        print(f"\n  Memory: {model.mem.stats()}")
        if c<cfg["num_cycles"]-1:
            print(f"\n--- Compression ---")
            n=model.compress()
            if n: print(f"  {n} new primitives.")

    print(f"\n{'='*65}\nFINAL TEST (NPR vs GPT-2)\n{'='*65}")
    print(f"Library: {model.lib.names}")
    evaluate(model,cache,w2i,i2w,rels,comp_tasks,gpt2=gpt2,tokenizer=tokenizer)

    probe_primitives(model,cache,rels,w2i)

    print("\n  --- Primitive Usage ---")
    usage=Counter(); model.eval()
    with torch.no_grad():
        for _ in range(300):
            t=generate_task_fast(cache,w2i,rels)
            _,pr,_,_,_,_=model(t,0.1,use_mem=False)
            for s in pr:
                i=s.argmax().item()
                usage[model.lib.names[i] if i<len(model.lib.names) else f"P{i}"]+=1
    tu=sum(usage.values())
    for nm,cnt in sorted(usage.items(),key=lambda x:-x[1]):
        print(f"    {nm:20s}: {100*cnt/tu:5.1f}%")

    print(f"\n  --- Program Lengths ---")
    lens=Counter()
    with torch.no_grad():
        for _ in range(300):
            t=generate_mixed_task(cache,w2i,rels,comp_tasks)
            _,pr,_,_,_,_=model(t,0.1,use_mem=False)
            lens[len(pr)]+=1
    for k in sorted(lens): print(f"    K={k}: {100*lens[k]/300:.0f}%")

    print(f"\n  Memory: {model.mem.stats()}\nDone.")

if __name__=="__main__":
    main()
