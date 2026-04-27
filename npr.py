"""
Neural Program Reasoner (NPR)
=============================
An architecture for learning interpretable rule programs from few examples
via neural primitive composition and cyclic self-improvement.

Given a few input-output examples demonstrating a linguistic relationship,
NPR synthesizes an explicit program composed of learnable neural primitives,
caches it for reuse, and periodically compresses frequent primitive sequences
into new higher-level primitives.

Requirements:
    pip install torch transformers

Usage:
    python neural_program_reasoner.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
from collections import Counter

torch.manual_seed(42)

# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    "proj_dim": 256,
    "num_program_steps": 3,
    "temperature": 0.8,
    "grad_accumulation": 16,
    "num_examples": 5,
    "memory_capacity": 200,
    "compression_threshold": 5,
    "max_new_primitives": 2,
    "num_cycles": 3,
    "iters_per_cycle": [2000, 1500, 1000],
    "lr_per_cycle": [5e-4, 3e-4, 1e-4],
    "perceiver_layer": 8,
}

# =============================================================================
# Dataset: Linguistic Relations
# =============================================================================

RELATIONS = {
    "opposite": [
        ("hot", "cold"), ("big", "small"), ("fast", "slow"),
        ("happy", "sad"), ("light", "dark"), ("old", "young"),
        ("tall", "short"), ("rich", "poor"), ("hard", "soft"),
        ("loud", "quiet"), ("strong", "weak"), ("dry", "wet"),
        ("clean", "dirty"), ("open", "closed"), ("full", "empty"),
        ("good", "bad"), ("early", "late"), ("deep", "shallow"),
        ("thick", "thin"), ("wide", "narrow"),
    ],
    "capital": [
        ("France", "Paris"), ("Italy", "Rome"), ("Japan", "Tokyo"),
        ("Germany", "Berlin"), ("Spain", "Madrid"), ("Egypt", "Cairo"),
        ("Brazil", "Brasilia"), ("China", "Beijing"), ("Russia", "Moscow"),
        ("India", "Delhi"), ("Canada", "Ottawa"), ("Australia", "Canberra"),
        ("Greece", "Athens"), ("Poland", "Warsaw"), ("Sweden", "Stockholm"),
        ("Norway", "Oslo"), ("Portugal", "Lisbon"),
    ],
    "plural": [
        ("cat", "cats"), ("dog", "dogs"), ("house", "houses"),
        ("car", "cars"), ("tree", "trees"), ("book", "books"),
        ("bird", "birds"), ("child", "children"), ("mouse", "mice"),
        ("man", "men"), ("woman", "women"), ("foot", "feet"),
        ("tooth", "teeth"), ("city", "cities"), ("baby", "babies"),
        ("leaf", "leaves"), ("wife", "wives"), ("life", "lives"),
        ("box", "boxes"), ("day", "days"),
    ],
    "past_tense": [
        ("walk", "walked"), ("talk", "talked"), ("play", "played"),
        ("jump", "jumped"), ("look", "looked"), ("work", "worked"),
        ("start", "started"), ("call", "called"), ("move", "moved"),
        ("live", "lived"), ("turn", "turned"), ("help", "helped"),
        ("ask", "asked"), ("need", "needed"), ("open", "opened"),
        ("want", "wanted"), ("show", "showed"), ("try", "tried"),
        ("use", "used"), ("seem", "seemed"),
    ],
    "comparative": [
        ("big", "bigger"), ("small", "smaller"), ("fast", "faster"),
        ("slow", "slower"), ("tall", "taller"), ("short", "shorter"),
        ("long", "longer"), ("strong", "stronger"), ("weak", "weaker"),
        ("old", "older"), ("young", "younger"), ("hard", "harder"),
        ("soft", "softer"), ("loud", "louder"), ("cold", "colder"),
        ("warm", "warmer"), ("dark", "darker"), ("light", "lighter"),
        ("clean", "cleaner"), ("deep", "deeper"),
    ],
}


def build_vocab():
    """Build a restricted vocabulary from all words in the dataset."""
    words = set()
    for pairs in RELATIONS.values():
        for a, b in pairs:
            words.add(a.lower())
            words.add(b.lower())
    word_list = sorted(words)
    word2idx = {w: i for i, w in enumerate(word_list)}
    idx2word = {i: w for w, i in word2idx.items()}
    return word_list, word2idx, idx2word


def generate_task(tokenizer, word2idx, num_examples=5, relation_name=None):
    """Generate a single few-shot analogy task."""
    rel_name = relation_name or random.choice(list(RELATIONS.keys()))
    pairs = RELATIONS[rel_name]
    chosen = random.sample(pairs, min(num_examples + 1, len(pairs)))
    examples, test_pair = chosen[:num_examples], chosen[num_examples]
    return {
        "example_ids": [[tokenizer.encode(f"{a} means {b}") for a, b in examples]],
        "test_input_ids": tokenizer.encode(f"{test_pair[0]} means"),
        "target_idx": word2idx.get(test_pair[1].lower(), 0),
        "relation_name": rel_name,
        "test_word": test_pair[0],
        "expected_word": test_pair[1],
    }


# =============================================================================
# Perceiver: Frozen GPT-2
# =============================================================================

class Perceiver(nn.Module):
    """Encodes text into dense semantic representations using frozen GPT-2."""

    def __init__(self, gpt2_model, layer=8):
        super().__init__()
        self.gpt2 = gpt2_model
        self.layer = layer
        self.hidden_dim = gpt2_model.config.n_embd
        for p in self.gpt2.parameters():
            p.requires_grad = False

    def encode_single(self, token_ids):
        ids = torch.tensor([token_ids], device=next(self.gpt2.parameters()).device)
        with torch.no_grad():
            out = self.gpt2(ids, output_hidden_states=True)
        return out.hidden_states[self.layer][0, -1, :]

    def encode_batch(self, token_ids_list):
        return torch.stack([self.encode_single(ids) for ids in token_ids_list])


# =============================================================================
# Primitive Library
# =============================================================================

BASE_PRIMITIVE_NAMES = [
    "IDENTITY",   # Minimal passthrough
    "NEGATE",     # Deep semantic inversion
    "MORPH",      # Morphological modification
    "ASSOCIATE",  # Self-referential associative lookup
    "LOOKUP",     # Deep factual retrieval
    "BLEND",      # Context mixing with learned global vector
]


class PrimitiveLibrary(nn.Module):
    """Library of neural primitives with dynamic expansion via compression."""

    def __init__(self, state_dim):
        super().__init__()
        self.state_dim = state_dim
        self.primitive_names = list(BASE_PRIMITIVE_NAMES)

        self.identity = nn.Sequential(nn.Linear(state_dim, state_dim), nn.Tanh())
        self.negate = nn.Sequential(
            nn.Linear(state_dim, state_dim * 2), nn.GELU(),
            nn.Linear(state_dim * 2, state_dim * 2), nn.GELU(),
            nn.Linear(state_dim * 2, state_dim),
        )
        self.morph = nn.Sequential(
            nn.Linear(state_dim, state_dim), nn.LayerNorm(state_dim),
            nn.GELU(), nn.Linear(state_dim, state_dim),
        )
        self.associate_q = nn.Linear(state_dim, state_dim // 4)
        self.associate_k = nn.Linear(state_dim, state_dim // 4)
        self.associate_v = nn.Linear(state_dim, state_dim)
        self.associate_out = nn.Linear(state_dim, state_dim)
        self.lookup = nn.Sequential(
            nn.Linear(state_dim, state_dim * 2), nn.GELU(),
            nn.Linear(state_dim * 2, state_dim * 2), nn.GELU(),
            nn.Linear(state_dim * 2, state_dim),
        )
        self.blend_context = nn.Parameter(torch.randn(state_dim) * 0.01)
        self.blend_net = nn.Sequential(
            nn.Linear(state_dim * 2, state_dim), nn.GELU(),
            nn.Linear(state_dim, state_dim),
        )
        self.gates = nn.ParameterList([
            nn.Parameter(torch.tensor(0.01)),
            nn.Parameter(torch.tensor(0.5)),
            nn.Parameter(torch.tensor(0.3)),
            nn.Parameter(torch.tensor(0.4)),
            nn.Parameter(torch.tensor(0.5)),
            nn.Parameter(torch.tensor(0.3)),
        ])
        self.invented_primitives = nn.ModuleList()
        self.invented_gates = nn.ParameterList()

    @property
    def num_primitives(self):
        return len(BASE_PRIMITIVE_NAMES) + len(self.invented_primitives)

    def _apply_base(self, idx, state):
        if idx == 0:
            return state + torch.sigmoid(self.gates[0]) * self.identity(state)
        elif idx == 1:
            return state + torch.sigmoid(self.gates[1]) * self.negate(state)
        elif idx == 2:
            return state + torch.sigmoid(self.gates[2]) * self.morph(state)
        elif idx == 3:
            q, k = self.associate_q(state), self.associate_k(state)
            v = self.associate_v(state)
            attn = torch.dot(q, k) / math.sqrt(q.shape[0])
            delta = self.associate_out(torch.sigmoid(attn) * v)
            return state + torch.sigmoid(self.gates[3]) * delta
        elif idx == 4:
            return state + torch.sigmoid(self.gates[4]) * self.lookup(state)
        elif idx == 5:
            delta = self.blend_net(torch.cat([state, self.blend_context]))
            return state + torch.sigmoid(self.gates[5]) * delta

    def apply_single(self, idx, state):
        if idx < len(BASE_PRIMITIVE_NAMES):
            return self._apply_base(idx, state)
        inv_idx = idx - len(BASE_PRIMITIVE_NAMES)
        delta = self.invented_primitives[inv_idx](state)
        return state + torch.sigmoid(self.invented_gates[inv_idx]) * delta

    def apply_soft(self, weights, state):
        return sum(weights[i] * self.apply_single(i, state)
                   for i in range(self.num_primitives))

    def add_primitive(self, name, src_a, src_b):
        sd = self.state_dim
        new_prim = nn.Sequential(
            nn.Linear(sd, sd * 2), nn.LayerNorm(sd * 2), nn.GELU(),
            nn.Linear(sd * 2, sd * 2), nn.GELU(), nn.Linear(sd * 2, sd),
        )
        self.invented_primitives.append(new_prim)
        self.invented_gates.append(nn.Parameter(torch.tensor(0.4)))
        self.primitive_names.append(name)
        print(f"  [COMPRESS] Created: {name} "
              f"(= {self.primitive_names[src_a]} -> {self.primitive_names[src_b]})")
        return self.num_primitives - 1


# =============================================================================
# Program Memory
# =============================================================================

class ProgramMemory:
    """Non-parametric memory for caching discovered programs."""

    def __init__(self, capacity=200):
        self.capacity = capacity
        self.entries = []

    def store(self, signature, program_indices, relation, correct):
        self.entries.append({
            "signature": signature.detach().clone(),
            "program": program_indices,
            "relation": relation,
            "correct": correct,
            "count": 1,
        })
        if len(self.entries) > self.capacity:
            self.entries.sort(key=lambda e: e["count"], reverse=True)
            self.entries = self.entries[:self.capacity]

    def lookup(self, signature, threshold=0.85):
        if not self.entries:
            return None
        best, best_sim = None, -1
        for e in self.entries:
            sim = F.cosine_similarity(
                signature.unsqueeze(0), e["signature"].unsqueeze(0)).item()
            if sim > best_sim:
                best_sim, best = sim, e
        if best_sim > threshold and best:
            best["count"] += 1
            return best
        return None

    def get_frequent_pairs(self, min_count=5):
        counts = Counter()
        for e in self.entries:
            for i in range(len(e["program"]) - 1):
                counts[(e["program"][i], e["program"][i + 1])] += 1
        return {p: c for p, c in counts.items() if c >= min_count}

    def clear(self):
        self.entries = []

    def stats(self):
        if not self.entries:
            return "Empty"
        c = sum(1 for e in self.entries if e["correct"])
        return (f"{len(self.entries)} entries, {c}/{len(self.entries)} correct "
                f"({100 * c / len(self.entries):.0f}%)")


# =============================================================================
# Program Synthesizer
# =============================================================================

class ProgramSynthesizer(nn.Module):
    """Produces a program (sequence of primitive selections) from examples."""

    def __init__(self, state_dim, num_primitives, num_steps, proj_dim=256):
        super().__init__()
        self.proj_dim = proj_dim
        self._num_primitives = num_primitives

        self.input_proj = nn.Sequential(
            nn.Linear(state_dim, proj_dim), nn.LayerNorm(proj_dim), nn.GELU())
        self.attn1 = nn.MultiheadAttention(proj_dim, 4, batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(proj_dim), nn.LayerNorm(proj_dim)
        self.ff1 = nn.Sequential(nn.Linear(proj_dim, proj_dim * 2), nn.GELU(),
                                 nn.Linear(proj_dim * 2, proj_dim))
        self.attn2 = nn.MultiheadAttention(proj_dim, 4, batch_first=True)
        self.norm3, self.norm4 = nn.LayerNorm(proj_dim), nn.LayerNorm(proj_dim)
        self.ff2 = nn.Sequential(nn.Linear(proj_dim, proj_dim * 2), nn.GELU(),
                                 nn.Linear(proj_dim * 2, proj_dim))
        self.step_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(proj_dim, 128), nn.GELU(),
                          nn.Linear(128, num_primitives))
            for _ in range(num_steps)
        ])

    def compute_signature(self, example_reprs):
        x = self.input_proj(example_reprs).unsqueeze(0)
        a, _ = self.attn1(x, x, x); x = self.norm1(x + a)
        x = self.norm2(x + self.ff1(x))
        a, _ = self.attn2(x, x, x); x = self.norm3(x + a)
        x = self.norm4(x + self.ff2(x))
        return x.mean(dim=1).squeeze(0)

    def rebuild_heads(self, new_num):
        old = self._num_primitives
        if new_num <= old:
            return
        new_heads = nn.ModuleList()
        for h in self.step_heads:
            nh = nn.Sequential(nn.Linear(self.proj_dim, 128), nn.GELU(),
                               nn.Linear(128, new_num))
            with torch.no_grad():
                nh[0].weight.copy_(h[0].weight); nh[0].bias.copy_(h[0].bias)
                nh[2].weight[:old].copy_(h[2].weight)
                nh[2].bias[:old].copy_(h[2].bias)
                nn.init.normal_(nh[2].weight[old:], std=0.01)
                nn.init.zeros_(nh[2].bias[old:])
            new_heads.append(nh)
        self.step_heads = new_heads
        self._num_primitives = new_num

    def forward(self, example_reprs, temperature=0.8, num_primitives=None):
        pattern = self.compute_signature(example_reprs)
        np_ = num_primitives or self._num_primitives
        program = []
        for head in self.step_heads:
            logits = head(pattern)
            if logits.shape[0] < np_:
                logits = torch.cat([logits, torch.zeros(np_ - logits.shape[0],
                                                        device=logits.device)])
            elif logits.shape[0] > np_:
                logits = logits[:np_]
            if self.training:
                program.append(F.gumbel_softmax(logits, tau=temperature, hard=False))
            else:
                program.append(F.one_hot(logits.argmax(), np_).float())
        return program, pattern


# =============================================================================
# Program Executor
# =============================================================================

class ProgramExecutor(nn.Module):
    """Applies a program to a state by sequentially executing selected primitives."""

    def __init__(self, state_dim, num_steps):
        super().__init__()
        self.step_embedding = nn.Embedding(num_steps, state_dim)

    def forward(self, state, program, lib):
        trace = []
        for step, sel in enumerate(program):
            emb = self.step_embedding(torch.tensor(step, device=state.device))
            state = lib.apply_soft(sel, state + emb)
            trace.append(sel.detach())
        return state, trace


# =============================================================================
# Generator
# =============================================================================

class Generator(nn.Module):
    """Maps transformed state to a vocabulary prediction."""

    def __init__(self, state_dim, pattern_dim, vocab_size):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(state_dim * 2 + pattern_dim, state_dim),
            nn.LayerNorm(state_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(state_dim, state_dim // 2),
            nn.LayerNorm(state_dim // 2), nn.GELU(),
            nn.Linear(state_dim // 2, vocab_size),
        )

    def forward(self, transformed, original, pattern):
        return self.head(torch.cat([transformed, original, pattern]))


# =============================================================================
# Neural Program Reasoner (Full Model)
# =============================================================================

class NeuralProgramReasoner(nn.Module):
    """
    Complete model: Perceiver -> Synthesizer -> Executor -> Generator
    with Program Memory and cyclic primitive compression.
    """

    def __init__(self, gpt2_model, vocab_size):
        super().__init__()
        cfg = CONFIG
        self.perceiver = Perceiver(gpt2_model, layer=cfg["perceiver_layer"])
        sd = self.perceiver.hidden_dim

        self.primitive_library = PrimitiveLibrary(sd)
        self.synthesizer = ProgramSynthesizer(
            sd, len(BASE_PRIMITIVE_NAMES), cfg["num_program_steps"], cfg["proj_dim"])
        self.executor = ProgramExecutor(sd, cfg["num_program_steps"])
        self.generator = Generator(sd, cfg["proj_dim"], vocab_size)
        self.test_adapter = nn.Sequential(
            nn.Linear(sd, sd), nn.LayerNorm(sd), nn.GELU(), nn.Linear(sd, sd))
        self.memory = ProgramMemory(capacity=cfg["memory_capacity"])

    def forward(self, task, temperature=0.8, use_memory=True):
        example_reprs = self.perceiver.encode_batch(task["example_ids"][0])
        test_repr = self.perceiver.encode_single(task["test_input_ids"])
        original = test_repr.clone()
        state = self.test_adapter(test_repr)
        np_ = self.primitive_library.num_primitives

        signature = self.synthesizer.compute_signature(example_reprs)

        from_memory = False
        program = None
        if use_memory and not self.training:
            cached = self.memory.lookup(signature)
            if cached:
                program = [F.one_hot(torch.tensor(i), np_).float()
                           for i in cached["program"]]
                from_memory = True

        if program is None:
            program, _ = self.synthesizer(example_reprs, temperature, np_)

        final, trace = self.executor(state, program, self.primitive_library)
        logits = self.generator(final, original, signature)
        return logits, trace, program, signature, from_memory

    def memorize(self, signature, program, relation, correct):
        self.memory.store(signature,
                          [s.argmax().item() for s in program], relation, correct)

    def compress(self):
        cfg = CONFIG
        freq = self.memory.get_frequent_pairs(cfg["compression_threshold"])
        if not freq:
            print("  [COMPRESS] No frequent sequences found.")
            return 0
        created = 0
        for (a, b), count in sorted(freq.items(), key=lambda x: -x[1]):
            if created >= cfg["max_new_primitives"]:
                break
            na = self.primitive_library.primitive_names[a]
            nb = self.primitive_library.primitive_names[b]
            name = f"{na}_{nb}"
            if name in self.primitive_library.primitive_names:
                continue
            print(f"  [COMPRESS] Frequent: {na} -> {nb} ({count}x)")
            self.primitive_library.add_primitive(name, a, b)
            created += 1
        if created > 0:
            self.synthesizer.rebuild_heads(self.primitive_library.num_primitives)
            print(f"  [COMPRESS] Library: {self.primitive_library.num_primitives} primitives")
            self.memory.clear()
        return created


# =============================================================================
# Loss Functions
# =============================================================================

def diversity_loss(program):
    """Penalize identical primitive selections across steps."""
    if len(program) < 2:
        return torch.tensor(0.0)
    loss, n = torch.tensor(0.0), len(program)
    for i in range(n):
        for j in range(i + 1, n):
            loss = loss + F.relu(
                F.cosine_similarity(program[i].unsqueeze(0),
                                    program[j].unsqueeze(0)) - 0.3)
    return loss / (n * (n - 1) / 2)


def usage_loss(program, num_primitives):
    """Penalize low entropy in the average primitive distribution."""
    avg = torch.stack(program).mean(dim=0)[:num_primitives]
    return math.log(num_primitives) - (-(avg * torch.log(avg + 1e-8)).sum())


# =============================================================================
# Training & Evaluation
# =============================================================================

def format_program(program, names):
    return " -> ".join(
        f"{names[s.argmax().item()] if s.argmax().item() < len(names) else '?'}"
        f"({s[s.argmax().item()].item():.0%})"
        for s in program
    )


def train_cycle(model, tokenizer, w2i, i2w, num_iters, cycle, lr):
    cfg = CONFIG
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s:
    s / 200 if s < 200 else 0.5 * (1 + math.cos(math.pi * (s - 200) / (num_iters - 200))))

    model.train()
    correct, total = 0, 0
    names = model.primitive_library.primitive_names

    for it in range(num_iters):
        task = generate_task(tokenizer, w2i, num_examples=cfg["num_examples"])
        np_ = model.primitive_library.num_primitives
        logits, trace, prog, sig, _ = model(task, cfg["temperature"], use_memory=False)

        tgt = torch.tensor(task["target_idx"])
        lce = F.cross_entropy(logits.unsqueeze(0), tgt.unsqueeze(0))
        loss = (lce + 0.1 * diversity_loss(prog) +
                0.05 * usage_loss(prog, np_)) / cfg["grad_accumulation"]
        loss.backward()

        pred = i2w[logits.argmax().item()]
        ok = pred == task["expected_word"].lower()
        correct += ok; total += 1

        with torch.no_grad():
            model.memorize(sig, prog, task["relation_name"], ok)

        if (it + 1) % cfg["grad_accumulation"] == 0:
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); opt.zero_grad(); sched.step()

        if it % 200 == 0:
            acc = 100 * correct / max(total, 1)
            with torch.no_grad():
                t = generate_task(tokenizer, w2i, num_examples=cfg["num_examples"])
                lg, _, pr, _, _ = model(t, cfg["temperature"], use_memory=False)
                pw = i2w[lg.argmax().item()]
                tw = t["expected_word"].lower()
                t3 = [i2w[i] for i in lg.topk(3).indices.tolist()]
                m = "\u2713" if pw == tw else "\u2717"
            print(f"  [C{cycle}] It {it:4d} | L:{lce.item():.4f} | Acc:{acc:5.1f}% | "
                  f"[{t['relation_name']:12s}] '{t['test_word']}' -> '{pw}' "
                  f"(want:'{tw}') {m}  top3:{t3}")
            print(f"         Prog: {format_program(pr, names)}")
            if it > 0 and it % 1000 == 0:
                correct, total = 0, 0


def evaluate(model, tokenizer, w2i, i2w):
    model.eval()
    names = model.primitive_library.primitive_names
    results = {}

    print("\n  --- Discovered Programs ---\n")
    with torch.no_grad():
        for rel in RELATIONS:
            t = generate_task(tokenizer, w2i, relation_name=rel)
            _, _, pr, _, _ = model(t, temperature=0.1, use_memory=False)
            print(f"    {rel:15s}: {format_program(pr, names)}")

    print("\n  --- Accuracy ---\n")
    with torch.no_grad():
        for rel in RELATIONS:
            ok, t3ok, tot, shown = 0, 0, 30, 0
            for _ in range(tot):
                t = generate_task(tokenizer, w2i, relation_name=rel)
                lg, _, pr, _, mem = model(t, temperature=0.1, use_memory=True)
                pw = i2w[lg.argmax().item()]
                tw = t["expected_word"].lower()
                top3 = [i2w[i] for i in lg.topk(3).indices.tolist()]
                ok += pw == tw; t3ok += tw in top3
                if shown < 3:
                    mk = "\u2713" if pw == tw else ("~" if tw in top3 else "\u2717")
                    tg = " [MEM]" if mem else ""
                    print(f"    [{rel:12s}] '{t['test_word']:12s}' -> "
                          f"'{pw:12s}' (want:'{tw:12s}') {mk}{tg}")
                    shown += 1
            a, t3 = 100 * ok / tot, 100 * t3ok / tot
            results[rel] = (a, t3)
            print(f"    -> {rel}: {ok}/{tot} ({a:.0f}%) top3:{t3ok}/{tot} ({t3:.0f}%)\n")

    print("  --- Summary ---")
    tc, tt3, tt = 0, 0, 0
    for r, (a, t3) in sorted(results.items(), key=lambda x: -x[1][0]):
        b = "\u2588" * int(a / 5) + "\u2591" * int((t3 - a) / 5)
        print(f"    {r:15s}: {a:5.1f}% [{b}] (top3:{t3:.0f}%)")
        tc += a * 30 / 100; tt3 += t3 * 30 / 100; tt += 30
    ov, ov3 = 100 * tc / tt, 100 * tt3 / tt
    print(f"\n    OVERALL: {ov:.1f}% (top3:{ov3:.1f}%)")
    return ov, ov3


def main():
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    print("Loading GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt2.eval()

    _, w2i, i2w = build_vocab()
    print(f"Vocabulary: {len(w2i)} words | Primitives: {BASE_PRIMITIVE_NAMES}")

    model = NeuralProgramReasoner(gpt2, len(w2i))
    tp = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {tp:,}\n")

    cfg = CONFIG
    for c in range(cfg["num_cycles"]):
        print(f"{'=' * 60}")
        print(f"CYCLE {c + 1}/{cfg['num_cycles']} | "
              f"Primitives: {model.primitive_library.num_primitives} | "
              f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        print(f"Library: {model.primitive_library.primitive_names}")
        print(f"{'=' * 60}\n")

        train_cycle(model, tokenizer, w2i, i2w,
                    cfg["iters_per_cycle"][c], c + 1, cfg["lr_per_cycle"][c])
        print(f"\n--- Eval Cycle {c + 1} ---")
        evaluate(model, tokenizer, w2i, i2w)
        print(f"\n  Memory: {model.memory.stats()}")

        if c < cfg["num_cycles"] - 1:
            print(f"\n--- Compression ---")
            n = model.compress()
            if n: print(f"  {n} new primitives.")

    print(f"\n{'=' * 60}\nFINAL TEST\n{'=' * 60}")
    print(f"Library: {model.primitive_library.primitive_names}")
    evaluate(model, tokenizer, w2i, i2w)

    print("\n  --- Primitive Usage ---")
    usage = Counter()
    model.eval()
    with torch.no_grad():
        for _ in range(200):
            t = generate_task(tokenizer, w2i)
            _, _, pr, _, _ = model(t, temperature=0.1, use_memory=False)
            for s in pr:
                idx = s.argmax().item()
                n = model.primitive_library.primitive_names
                usage[n[idx] if idx < len(n) else f"P{idx}"] += 1
    tu = sum(usage.values())
    for nm, cnt in sorted(usage.items(), key=lambda x: -x[1]):
        print(f"    {nm:20s}: {100 * cnt / tu:5.1f}% [{'█' * int(50 * cnt / tu)}]")
    print(f"\n  Memory: {model.memory.stats()}")


if __name__ == "__main__":
    main()
