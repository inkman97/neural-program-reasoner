"""
Neural Program Reasoner — JEPA World Model
Visual Perceiver + Latent Prediction (no vocabulary):
  - Realistic scene-based images (not generic colored shapes)
  - Each object drawn as recognizable item (cup, door, lamp...)
  - Property shown via clear visual cues (steam, ice, flames, open/closed)
  - Dual loss (cosine + 0.5*MSE)
  - Deeper LatentPredictor (4 layers, hidden sd*2)
  - Action loss weight 0.4
  - No projection head

Requirements: pip install torch transformers Pillow
"""

import math
import random
import time
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

torch.manual_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    "proj_dim": 256, "num_slots": 2, "max_program_steps": 6, "min_program_steps": 2,
    "temperature": 0.8, "grad_accumulation": 16, "memory_capacity": 300,
    "compression_threshold": 6, "max_new_primitives": 2, "num_cycles": 3,
    "iters_per_cycle": [12000, 8000, 4000], "lr_per_cycle": [5e-4, 3e-4, 1e-4],
    "eval_samples": 40, "max_chain_length": 3, "multistep_ratio": 0.35,
    "chain_effect_ratio": 0.1, "goal_loss_weight": 0.2,
    "precondition_loss_weight": 0.15, "vit_dim": 768, "img_size": 224,
}

# === WORLD DEFINITION ===
OBJECTS = {"water": {"temperature": True}, "soup": {"temperature": True}, "coffee": {"temperature": True},
           "tea": {"temperature": True}, "milk": {"temperature": True}, "food": {"temperature": True},
           "oil": {"temperature": True}, "pan": {"temperature": True}, "iron": {"temperature": True},
           "metal": {"temperature": True}, "door": {"openness": True}, "window": {"openness": True},
           "box": {"openness": True}, "jar": {"openness": True}, "gate": {"openness": True},
           "drawer": {"openness": True}, "lid": {"openness": True}, "bag": {"openness": True},
           "bottle": {"openness": True}, "cabinet": {"openness": True},
           "lamp": {"power": True, "surface": "table", "fragile": True, "integrity": True}, "light": {"power": True},
           "screen": {"power": True}, "tv": {"power": True}, "radio": {"power": True}, "computer": {"power": True},
           "fan": {"power": True}, "heater": {"power": True}, "oven": {"power": True}, "speaker": {"power": True},
           "cup": {"fullness": True, "surface": "shelf", "fragile": True, "integrity": True},
           "glass": {"fullness": True, "surface": "table", "fragile": True, "integrity": True},
           "bowl": {"fullness": True, "fragile": True, "integrity": True}, "pot": {"fullness": True},
           "tub": {"fullness": True}, "bucket": {"fullness": True}, "jug": {"fullness": True},
           "tank": {"fullness": True}, "pool": {"fullness": True},
           "plate": {"integrity": True, "surface": "counter", "fragile": True},
           "vase": {"integrity": True, "fragile": True, "surface": "shelf"}, "mirror": {"integrity": True},
           "ball": {"surface": "table"}, "book": {"surface": "desk"}, "phone": {"surface": "bed"},
           "toy": {"surface": "shelf"}, "pen": {"surface": "desk"}, "remote": {"surface": "couch"},
           "key": {"container": "pocket"}, "coin": {"container": "jar"}, "letter": {"container": "envelope"},
           "shirt": {"container": "closet"}, "tool": {"container": "shed"}, "food_c": {"container": "fridge"},
           "ball_c": {"container": "box"}, "toy_c": {"container": "bag"}, "pen_c": {"container": "case"},
           "book_c": {"container": "drawer"}}
ACTIONS = {"heat": {"property": "temperature", "from": "cold", "to": "hot"},
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
           "put in": {"property": "containment", "from": "outside", "to": "inside"}}


def generate_world_observations():
    observations = [];
    rule_groups = defaultdict(list)
    for obj_name, obj_props in OBJECTS.items():
        for act_name, act_def in ACTIONS.items():
            prop = act_def["property"]
            if prop == "temperature" and not obj_props.get("temperature"): continue
            if prop == "openness" and not obj_props.get("openness"): continue
            if prop == "power" and not obj_props.get("power"): continue
            if prop == "fullness" and not obj_props.get("fullness"): continue
            if prop == "integrity" and not obj_props.get("integrity"): continue
            if prop == "position" and "surface" not in obj_props: continue
            if prop == "containment" and "container" not in obj_props: continue
            display = obj_name.split("_")[0]
            if prop == "position":
                state = f"the {display} is on the {obj_props['surface']}"; result = f"the {display} is on the floor"
            elif prop == "containment":
                state = f"the {display} is outside the {obj_props['container']}"; result = f"the {display} is inside the {obj_props['container']}"
            else:
                state = f"the {display} is {act_def['from']}"; result = f"the {display} is {act_def['to']}"
            obs = {"state": state, "action": act_name, "result": result, "property": prop, "object": display,
                   "fragile": obj_props.get("fragile", False)}
            observations.append(obs);
            rule_groups[act_name].append(obs)
    return observations, rule_groups


ALL_OBSERVATIONS, RULE_GROUPS = generate_world_observations()
TRANSITION_RULES = {}
for act_name, obs_list in RULE_GROUPS.items():
    triples = [(o["state"], o["action"], o["result"]) for o in obs_list]
    if triples: TRANSITION_RULES[act_name] = {"triples": triples, "precondition": ACTIONS[act_name]["from"].split()[0],
                                              "property_changed": obs_list[0]["property"]}
CHAIN_EFFECTS = [{"trigger_action": "push", "chain_action": "drop", "condition": "fragile"}]
CHAIN_TRIPLES = []
for ce in CHAIN_EFFECTS:
    chain_by_obj = {o["object"]: o for o in RULE_GROUPS.get(ce["chain_action"], [])}
    for obs in RULE_GROUPS.get(ce["trigger_action"], []):
        if obs.get(ce["condition"], False) and obs["object"] in chain_by_obj:
            co = chain_by_obj[obs["object"]];
            CHAIN_TRIPLES.append(
                {"state": obs["state"], "action": obs["action"], "result": co["result"], "intermediate": obs["result"],
                 "rules": [ce["trigger_action"], ce["chain_action"]], "object": obs["object"]})

print(f"World: {len(ALL_OBSERVATIONS)} observations, {len(RULE_GROUPS)} actions, {len(CHAIN_TRIPLES)} chain effects")
ALL_ACTIONS_LIST = sorted(set(a for r in TRANSITION_RULES.values() for _, a, _ in r["triples"]))
ACT2IDX = {a: i for i, a in enumerate(ALL_ACTIONS_LIST)};
IDX2ACT = {i: a for a, i in ACT2IDX.items()};
NUM_ACTIONS = len(ALL_ACTIONS_LIST)
PROPERTY_TYPES = sorted(set(r["property_changed"] for r in TRANSITION_RULES.values()));
PROP2IDX = {p: i for i, p in enumerate(PROPERTY_TYPES)};
NUM_PROPERTIES = len(PROPERTY_TYPES)

# === HELD-OUT OBJECTS for generalization test ===
# For each property type, hold out 2 objects from training
HELD_OUT_OBJECTS = {
    "temperature": ["iron", "metal"],
    "openness": ["bottle", "cabinet"],
    "power": ["oven", "speaker"],
    "fullness": ["jug", "tank"],
    "integrity": ["mirror", "lamp"],
    "position": ["pen", "remote"],
    "containment": ["tool", "book_c"],
}
HELD_OUT_SET = set()
for objs in HELD_OUT_OBJECTS.values():
    HELD_OUT_SET.update(objs)

# Split TRANSITION_RULES into train and held-out test sets
TRAIN_RULES = {}
HELDOUT_RULES = {}
for act_name, rule in TRANSITION_RULES.items():
    train_triples = []
    held_triples = []
    for s, a, res in rule["triples"]:
        obj = s.split()[1]  # "the OBJ is ..."
        if obj in HELD_OUT_SET or f"{obj}_c" in HELD_OUT_SET:
            held_triples.append((s, a, res))
        else:
            train_triples.append((s, a, res))
    if train_triples:
        TRAIN_RULES[act_name] = {"triples": train_triples, "precondition": rule["precondition"],
                                 "property_changed": rule["property_changed"]}
    if held_triples:
        HELDOUT_RULES[act_name] = {"triples": held_triples, "precondition": rule["precondition"],
                                   "property_changed": rule["property_changed"]}

n_train = sum(len(r["triples"]) for r in TRAIN_RULES.values())
n_held = sum(len(r["triples"]) for r in HELDOUT_RULES.values())
print(f"Train/test split: {n_train} train triples, {n_held} held-out triples")
print(f"Held-out objects: {sorted(HELD_OUT_SET)}")


def build_vocab():
    ss = set()
    for r in TRANSITION_RULES.values():
        for s, a, res in r["triples"]: ss.add(s); ss.add(res)
    for ct in CHAIN_TRIPLES: ss.add(ct["state"]); ss.add(ct["result"]); ss.add(ct["intermediate"])
    sl = sorted(ss);
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
    print(f"Chains: d1:{len(chains[1])}, d2:{len(chains[2])}, d3:{len(chains[3])}");
    return chains


# =============================================================================
# SCENE-BASED IMAGE RENDERER
# Each object is drawn as a recognizable item with clear property indicators
# =============================================================================

def _hash(name): return sum(ord(c) * (i + 1) for i, c in enumerate(name)) % 1000


# Background colors for different property types
BG_COLORS = {
    "temperature": {"cold": (200, 220, 255), "warm": (255, 240, 200), "hot": (255, 210, 200),
                    "boiling": (240, 200, 240)},
    "openness": {"closed": (200, 200, 200), "open": (220, 255, 220)},
    "power": {"off": (180, 180, 180), "on": (255, 255, 220)},
    "fullness": {"empty": (240, 240, 240), "full": (200, 220, 255)},
    "integrity": {"intact": (220, 240, 220), "broken": (255, 200, 200)},
}


def _draw_container(draw, cx, cy, sz, obj_name):
    """Draw a cup/bowl/pot/jug/etc shape."""
    w, h = sz, int(sz * 1.2)
    # Trapezoid body
    draw.polygon([(cx - w // 2, cy - h // 2), (cx + w // 2, cy - h // 2),
                  (cx + w // 2 + 8, cy + h // 2), (cx - w // 2 - 8, cy + h // 2)],
                 fill=(180, 140, 100), outline=(80, 60, 40), width=2)
    # Handle for cups/mugs
    if obj_name in ["cup", "mug", "jug"]:
        draw.arc([cx + w // 2, cy - h // 4, cx + w // 2 + 20, cy + h // 4], 270, 90, fill=(80, 60, 40), width=3)


def _draw_door(draw, cx, cy, sz, is_open):
    """Draw a door shape."""
    w, h = int(sz * 0.8), int(sz * 1.5)
    if is_open:
        # Open door - perspective trapezoid
        draw.polygon([(cx - w // 2, cy - h // 2), (cx, cy - h // 2 + 10),
                      (cx, cy + h // 2 - 10), (cx - w // 2, cy + h // 2)],
                     fill=(160, 120, 80), outline=(80, 60, 40), width=2)
        # Gap showing through
        draw.rectangle([cx + 2, cy - h // 2 + 12, cx + w // 2, cy + h // 2 - 12], fill=(100, 140, 100))
    else:
        draw.rectangle([cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2],
                       fill=(160, 120, 80), outline=(80, 60, 40), width=2)
        # Door knob
        draw.ellipse([cx + w // 4, cy - 4, cx + w // 4 + 8, cy + 4], fill=(200, 180, 50))


def _draw_device(draw, cx, cy, sz, is_on, obj_name):
    """Draw electronic device."""
    w, h = sz, int(sz * 0.8)
    # Body
    draw.rectangle([cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2],
                   fill=(60, 60, 70), outline=(30, 30, 40), width=2)
    if is_on:
        # Glowing screen/indicator
        draw.rectangle([cx - w // 2 + 6, cy - h // 2 + 6, cx + w // 2 - 6, cy + h // 2 - 6],
                       fill=(100, 200, 255))
        # Light rays
        for angle in range(0, 360, 45):
            ex = cx + int((w // 2 + 15) * math.cos(math.radians(angle)))
            ey = cy + int((h // 2 + 15) * math.sin(math.radians(angle)))
            draw.line([(cx + int((w // 2 + 3) * math.cos(math.radians(angle))),
                        cy + int((h // 2 + 3) * math.sin(math.radians(angle)))),
                       (ex, ey)], fill=(255, 255, 100), width=1)
    else:
        # Dark screen
        draw.rectangle([cx - w // 2 + 6, cy - h // 2 + 6, cx + w // 2 - 6, cy + h // 2 - 6],
                       fill=(30, 30, 35))


def _draw_object_on_surface(draw, cx, cy, sz, surface_name):
    """Draw a surface (table/shelf/desk) with object above it."""
    w = sz + 40
    # Surface line
    sy = cy + sz // 2 + 5
    draw.rectangle([cx - w // 2, sy, cx + w // 2, sy + 8], fill=(139, 90, 43), outline=(80, 50, 20))
    # Legs
    draw.rectangle([cx - w // 2 + 5, sy + 8, cx - w // 2 + 10, sy + 35], fill=(120, 80, 40))
    draw.rectangle([cx + w // 2 - 10, sy + 8, cx + w // 2 - 5, sy + 35], fill=(120, 80, 40))


def _draw_broken_overlay(draw, cx, cy, sz):
    """Draw cracks and broken pieces."""
    # Jagged crack line across
    pts = [(cx - sz // 2, cy)]
    for i in range(6):
        pts.append((cx - sz // 2 + (i + 1) * sz // 6, cy + random.randint(-sz // 4, sz // 4)))
    for p1, p2 in zip(pts, pts[1:]):
        draw.line([p1, p2], fill=(0, 0, 0), width=3)
    # Scattered pieces
    for _ in range(4):
        px = cx + random.randint(-sz // 2 - 10, sz // 2 + 10)
        py = cy + random.randint(sz // 4, sz // 2 + 15)
        psz = random.randint(3, 8)
        draw.polygon([(px, py - psz), (px + psz, py), (px, py + psz), (px - psz, py)],
                     fill=(180, 150, 120), outline=(100, 80, 60))


def _add_temp_effects(draw, cx, cy, sz, temp_val):
    """Add temperature visual effects."""
    if temp_val == "cold":
        # Snowflakes / ice crystals
        for _ in range(6):
            sx = cx + random.randint(-sz // 2 - 10, sz // 2 + 10)
            sy = cy + random.randint(-sz // 2 - 15, sz // 2 + 15)
            for angle in range(0, 360, 60):
                ex = sx + int(6 * math.cos(math.radians(angle)))
                ey = sy + int(6 * math.sin(math.radians(angle)))
                draw.line([(sx, sy), (ex, ey)], fill=(150, 200, 255), width=1)
    elif temp_val == "warm":
        # Gentle wavy lines above
        for i in range(3):
            wy = cy - sz // 2 - 10 - i * 10
            pts = []
            for x in range(cx - 20, cx + 21, 2):
                pts.append((x, wy + int(3 * math.sin(x / 4))))
            if len(pts) > 1:
                draw.line(pts, fill=(255, 180, 100), width=2)
    elif temp_val == "hot":
        # Steam lines rising
        for i in range(4):
            sx = cx - 15 + i * 10
            for j in range(3):
                sy = cy - sz // 2 - 8 - j * 12
                draw.line([(sx, sy), (sx + 3, sy - 8)], fill=(255, 100, 50), width=2)
                draw.line([(sx + 3, sy - 8), (sx - 2, sy - 14)], fill=(255, 80, 30), width=2)
    elif temp_val == "boiling":
        # Bubbles + intense steam
        for _ in range(10):
            bx = cx + random.randint(-sz // 3, sz // 3)
            by = cy + random.randint(-sz // 3, sz // 4)
            br = random.randint(4, 10)
            draw.ellipse([bx - br, by - br, bx + br, by + br], fill=(255, 255, 255), outline=(200, 0, 200), width=2)
        # Purple steam
        for i in range(5):
            sx = cx - 20 + i * 10
            for j in range(4):
                sy = cy - sz // 2 - 5 - j * 10
                draw.line([(sx, sy), (sx + random.randint(-4, 4), sy - 8)], fill=(200, 0, 200), width=2)


def render_state_image(state_text, size=224):
    """Render a world state as a scene-based image."""
    img = Image.new('RGB', (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    parts = state_text.split()
    obj_name = parts[1] if len(parts) > 1 else "unknown"

    # Determine property value
    prop_val = ""
    prop_type = ""
    if "on the floor" in state_text:
        prop_val = "floor"; prop_type = "position"
    elif "on the" in state_text:
        prop_val = "surface"; prop_type = "position"
    elif "outside" in state_text:
        prop_val = "outside"; prop_type = "containment"
    elif "inside" in state_text:
        prop_val = "inside"; prop_type = "containment"
    else:
        for ptype, vals in BG_COLORS.items():
            for val in vals:
                if f"is {val}" in state_text:
                    prop_val = val;
                    prop_type = ptype;
                    break
            if prop_val: break

    # Background based on property type and value
    bg = (245, 245, 245)
    if prop_type in BG_COLORS and prop_val in BG_COLORS[prop_type]:
        bg = BG_COLORS[prop_type][prop_val]
    draw.rectangle([0, 0, size, size], fill=bg)

    # Floor line
    floor_y = int(size * 0.75)
    draw.rectangle([0, floor_y, size, size], fill=(200, 190, 170))
    draw.line([(0, floor_y), (size, floor_y)], fill=(150, 140, 120), width=2)

    cx, cy = size // 2, floor_y - 45
    obj_sz = 50

    # Object-specific position
    if prop_val == "floor":
        cy = floor_y - 15  # on the ground
    elif prop_val == "surface":
        cy = floor_y - 65  # elevated

    # Draw based on object type and property
    h = _hash(obj_name)

    # TEMPERATURE objects (liquids/metals)
    if prop_type == "temperature" or (
            obj_name in ["water", "soup", "coffee", "tea", "milk", "food", "oil", "pan", "iron", "metal"]):
        _draw_container(draw, cx, cy, obj_sz, obj_name)
        if prop_val in ["cold", "warm", "hot", "boiling"]:
            _add_temp_effects(draw, cx, cy, obj_sz, prop_val)
        # Liquid color inside
        lw = obj_sz - 10
        lh = obj_sz // 2
        if prop_val == "cold":
            lc = (100, 150, 255)
        elif prop_val == "warm":
            lc = (255, 200, 100)
        elif prop_val == "hot":
            lc = (255, 80, 50)
        elif prop_val == "boiling":
            lc = (200, 50, 200)
        else:
            lc = (150, 180, 200)
        draw.rectangle([cx - lw // 2, cy - lh // 4, cx + lw // 2, cy + lh], fill=lc)
        # Object-specific label color band at bottom to distinguish liquids
        label_c = ((h * 47) % 150 + 50, (h * 83) % 150 + 50, (h * 127) % 150 + 50)
        draw.rectangle([cx - lw // 2, cy + lh - 6, cx + lw // 2, cy + lh], fill=label_c)

    # OPENNESS objects
    elif prop_type == "openness" or obj_name in ["door", "window", "box", "jar", "gate", "drawer", "lid", "bag",
                                                 "bottle", "cabinet"]:
        is_open = prop_val == "open"
        if obj_name in ["door", "gate", "cabinet"]:
            _draw_door(draw, cx, cy, obj_sz, is_open)
        else:
            # Box/container with lid
            w, h_ = obj_sz, int(obj_sz * 0.8)
            # Object-specific body color
            body_c = ((h * 31) % 100 + 120, (h * 61) % 80 + 100, (h * 97) % 60 + 80)
            draw.rectangle([cx - w // 2, cy - h_ // 4, cx + w // 2, cy + h_ // 2], fill=body_c, outline=(100, 80, 60),
                           width=2)
            if is_open:
                lid_c = (body_c[0] - 20, body_c[1] - 20, body_c[2] - 20)
                draw.polygon([(cx - w // 2, cy - h_ // 4), (cx - w // 2 + 10, cy - h_ // 4 - 20),
                              (cx + w // 2 + 10, cy - h_ // 4 - 20), (cx + w // 2, cy - h_ // 4)],
                             fill=lid_c, outline=(100, 80, 60), width=2)
            else:
                lid_c = (body_c[0] - 20, body_c[1] - 20, body_c[2] - 20)
                draw.rectangle([cx - w // 2 - 3, cy - h_ // 4 - 5, cx + w // 2 + 3, cy - h_ // 4],
                               fill=lid_c, outline=(100, 80, 60), width=2)

    # POWER objects
    elif prop_type == "power" or obj_name in ["lamp", "light", "screen", "tv", "radio", "computer", "fan", "heater",
                                              "oven", "speaker"]:
        is_on = prop_val == "on"
        _draw_device(draw, cx, cy, obj_sz, is_on, obj_name)

    # FULLNESS objects
    elif prop_type == "fullness" or obj_name in ["cup", "glass", "bowl", "pot", "tub", "bucket", "jug", "tank", "pool"]:
        _draw_container(draw, cx, cy, obj_sz, obj_name)
        if prop_val == "full":
            lw = obj_sz - 10
            draw.rectangle([cx - lw // 2, cy - obj_sz // 4, cx + lw // 2, cy + obj_sz // 2], fill=(80, 150, 230))
            draw.line([(cx - lw // 2, cy - obj_sz // 4), (cx + lw // 2, cy - obj_sz // 4)], fill=(100, 180, 255),
                      width=2)

    # INTEGRITY objects — each object has a unique shape
    elif prop_type == "integrity":
        w = obj_sz
        # Object-specific shapes and colors
        obj_color = ((h * 41) % 120 + 100, (h * 73) % 100 + 80, (h * 109) % 80 + 80)
        outline_c = (obj_color[0] - 50, obj_color[1] - 50, obj_color[2] - 50)

        if obj_name in ["glass", "cup", "bowl"]:
            # Curved container shape
            _draw_container(draw, cx, cy, w, obj_name)
        elif obj_name in ["plate"]:
            # Wide flat ellipse
            draw.ellipse([cx - w // 2 - 10, cy - w // 6, cx + w // 2 + 10, cy + w // 6], fill=obj_color,
                         outline=outline_c, width=2)
        elif obj_name in ["vase"]:
            # Tall narrow with wide top
            draw.polygon([(cx - w // 4, cy + w // 2), (cx - w // 3, cy), (cx - w // 5, cy - w // 3),
                          (cx + w // 5, cy - w // 3), (cx + w // 3, cy), (cx + w // 4, cy + w // 2)],
                         fill=obj_color, outline=outline_c, width=2)
        elif obj_name in ["mirror"]:
            # Rectangle with frame
            draw.rectangle([cx - w // 2 + 2, cy - w // 3, cx + w // 2 - 2, cy + w // 3], fill=(180, 210, 230),
                           outline=(120, 100, 60), width=3)
            draw.rectangle([cx - w // 2 + 8, cy - w // 3 + 6, cx + w // 2 - 8, cy + w // 3 - 6], fill=(200, 230, 250),
                           outline=(160, 180, 200))
        elif obj_name in ["window"]:
            # Window panes
            draw.rectangle([cx - w // 2, cy - w // 3, cx + w // 2, cy + w // 3], fill=(180, 200, 220),
                           outline=(100, 80, 60), width=2)
            draw.line([(cx, cy - w // 3), (cx, cy + w // 3)], fill=(100, 80, 60), width=2)
            draw.line([(cx - w // 2, cy), (cx + w // 2, cy)], fill=(100, 80, 60), width=2)
        elif obj_name in ["bottle", "jar"]:
            # Bottle neck + body
            draw.rectangle([cx - w // 6, cy - w // 2, cx + w // 6, cy - w // 4], fill=obj_color, outline=outline_c,
                           width=2)
            draw.rectangle([cx - w // 3, cy - w // 4, cx + w // 3, cy + w // 3], fill=obj_color, outline=outline_c,
                           width=2)
        elif obj_name == "lamp":
            # Lamp shade triangle + base
            draw.polygon([(cx, cy - w // 3), (cx - w // 2, cy + w // 6), (cx + w // 2, cy + w // 6)],
                         fill=(255, 240, 200), outline=(180, 160, 100), width=2)
            draw.rectangle([cx - w // 8, cy + w // 6, cx + w // 8, cy + w // 2], fill=(120, 100, 80))
        else:
            # Default: unique colored ellipse
            draw.ellipse([cx - w // 2, cy - w // 4, cx + w // 2, cy + w // 4], fill=obj_color, outline=outline_c,
                         width=2)

        if prop_val == "broken":
            _draw_broken_overlay(draw, cx, cy, obj_sz)

    # POSITION objects — each object has distinct shape and color
    elif prop_type == "position":
        obj_color = ((h * 37) % 150 + 80, (h * 67) % 150 + 80, (h * 103) % 150 + 80)
        outline_c = (max(0, obj_color[0] - 60), max(0, obj_color[1] - 60), max(0, obj_color[2] - 60))

        if obj_name == "ball":
            draw.ellipse([cx - obj_sz // 3, cy - obj_sz // 3, cx + obj_sz // 3, cy + obj_sz // 3],
                         fill=(220, 60, 60), outline=(150, 30, 30), width=2)
            # Stripe on ball
            draw.arc([cx - obj_sz // 3, cy - obj_sz // 3, cx + obj_sz // 3, cy + obj_sz // 3], 60, 120,
                     fill=(255, 255, 255), width=3)
        elif obj_name == "book":
            draw.rectangle([cx - obj_sz // 2 + 5, cy - obj_sz // 4, cx + obj_sz // 2 - 5, cy + obj_sz // 4],
                           fill=(50, 80, 150), outline=(30, 50, 100), width=2)
            # Spine lines
            for i in range(3):
                y = cy - obj_sz // 4 + 8 + i * 10
                draw.line([(cx - obj_sz // 2 + 10, y), (cx + obj_sz // 2 - 10, y)], fill=(200, 200, 200), width=1)
        elif obj_name == "phone":
            draw.rectangle([cx - obj_sz // 4, cy - obj_sz // 3, cx + obj_sz // 4, cy + obj_sz // 3],
                           fill=(40, 40, 50), outline=(20, 20, 30), width=2)
            draw.rectangle([cx - obj_sz // 4 + 4, cy - obj_sz // 3 + 4, cx + obj_sz // 4 - 4, cy + obj_sz // 3 - 12],
                           fill=(80, 130, 200))  # screen
        elif obj_name == "toy":
            # Star shape
            pts = []
            for i in range(10):
                angle = math.radians(i * 36 - 90)
                r = obj_sz // 3 if i % 2 == 0 else obj_sz // 6
                pts.append((cx + int(r * math.cos(angle)), cy + int(r * math.sin(angle))))
            draw.polygon(pts, fill=(255, 200, 50), outline=(200, 150, 20), width=2)
        elif obj_name == "pen":
            # Long thin rectangle
            draw.rectangle([cx - obj_sz // 2, cy - 4, cx + obj_sz // 2, cy + 4], fill=(30, 30, 120),
                           outline=(10, 10, 80), width=1)
            draw.polygon([(cx + obj_sz // 2, cy - 4), (cx + obj_sz // 2 + 10, cy), (cx + obj_sz // 2, cy + 4)],
                         fill=(200, 180, 100))
        elif obj_name == "remote":
            draw.rectangle([cx - obj_sz // 5, cy - obj_sz // 3, cx + obj_sz // 5, cy + obj_sz // 3],
                           fill=(50, 50, 55), outline=(30, 30, 35), width=2)
            # Buttons
            for i in range(3):
                for j in range(2):
                    bx = cx - 6 + j * 12
                    by = cy - obj_sz // 4 + 8 + i * 12
                    draw.ellipse([bx - 3, by - 3, bx + 3, by + 3], fill=(200, 50, 50))
        elif obj_name in ["lamp", "cup", "glass", "plate", "vase"]:
            # These might appear here if they have surface property — use unique shape
            draw.ellipse([cx - obj_sz // 3, cy - obj_sz // 3, cx + obj_sz // 3, cy + obj_sz // 3],
                         fill=obj_color, outline=outline_c, width=2)
        else:
            draw.ellipse([cx - obj_sz // 3, cy - obj_sz // 3, cx + obj_sz // 3, cy + obj_sz // 3],
                         fill=obj_color, outline=outline_c, width=2)

        if prop_val == "surface":
            _draw_object_on_surface(draw, cx, cy - 10, obj_sz, "table")

    # CONTAINMENT objects — distinct object + distinct container
    elif prop_type == "containment":
        container_w = obj_sz + 20
        container_h = obj_sz
        # Object-specific color
        obj_color = ((h * 43) % 150 + 80, (h * 79) % 150 + 80, (h * 113) % 150 + 80)
        # Container-specific color based on container name
        container_name = ""
        if "outside" in state_text or "inside" in state_text:
            parts_c = state_text.split()
            container_name = parts_c[-1] if len(parts_c) > 3 else ""
        cont_h = _hash(container_name) if container_name else h + 100
        cont_color = ((cont_h * 31) % 100 + 120, (cont_h * 67) % 80 + 100, (cont_h * 97) % 60 + 90)

        # Object shape based on object name
        obj_shape = h % 4
        if prop_val == "outside":
            ox, oy = cx - 50, cy
            if obj_shape == 0:
                draw.ellipse([ox - 15, oy - 15, ox + 15, oy + 15], fill=obj_color, outline=(0, 0, 0), width=2)
            elif obj_shape == 1:
                draw.rectangle([ox - 12, oy - 12, ox + 12, oy + 12], fill=obj_color, outline=(0, 0, 0), width=2)
            elif obj_shape == 2:
                draw.polygon([(ox, oy - 15), (ox + 15, oy + 10), (ox - 15, oy + 10)], fill=obj_color, outline=(0, 0, 0),
                             width=2)
            else:
                draw.polygon([(ox, oy - 15), (ox + 12, oy), (ox, oy + 15), (ox - 12, oy)], fill=obj_color,
                             outline=(0, 0, 0), width=2)
            # Container
            draw.rectangle([cx + 10, cy - container_h // 2, cx + 10 + container_w, cy + container_h // 2],
                           fill=cont_color, outline=(80, 60, 40), width=2)
        else:
            draw.rectangle([cx - container_w // 2, cy - container_h // 2, cx + container_w // 2, cy + container_h // 2],
                           fill=cont_color, outline=(80, 60, 40), width=2)
            # Object visible inside with its unique shape
            if obj_shape == 0:
                draw.ellipse([cx - 12, cy - 12, cx + 12, cy + 12], fill=obj_color, outline=(0, 0, 0), width=2)
            elif obj_shape == 1:
                draw.rectangle([cx - 10, cy - 10, cx + 10, cy + 10], fill=obj_color, outline=(0, 0, 0), width=2)
            elif obj_shape == 2:
                draw.polygon([(cx, cy - 12), (cx + 12, cy + 8), (cx - 12, cy + 8)], fill=obj_color, outline=(0, 0, 0),
                             width=2)
            else:
                draw.polygon([(cx, cy - 12), (cx + 10, cy), (cx, cy + 12), (cx - 10, cy)], fill=obj_color,
                             outline=(0, 0, 0), width=2)
    else:
        # Fallback: simple colored circle
        color = (150 + h % 100, 100 + h % 80, 80 + h % 60)
        draw.ellipse([cx - obj_sz // 2, cy - obj_sz // 2, cx + obj_sz // 2, cy + obj_sz // 2],
                     fill=color, outline=(0, 0, 0), width=2)

    # Object identifier: unique pattern in corner based on object name hash
    h = _hash(obj_name)
    # Small unique marker in top-left
    mx, my = 15, 15
    marker_color = ((h * 37) % 200 + 55, (h * 73) % 200 + 55, (h * 113) % 200 + 55)
    marker_shape = h % 4
    if marker_shape == 0:
        draw.ellipse([mx - 8, my - 8, mx + 8, my + 8], fill=marker_color, outline=(0, 0, 0))
    elif marker_shape == 1:
        draw.rectangle([mx - 8, my - 8, mx + 8, my + 8], fill=marker_color, outline=(0, 0, 0))
    elif marker_shape == 2:
        draw.polygon([(mx, my - 8), (mx + 8, my + 8), (mx - 8, my + 8)], fill=marker_color, outline=(0, 0, 0))
    else:
        draw.polygon([(mx, my - 8), (mx + 8, my), (mx, my + 8), (mx - 8, my)], fill=marker_color, outline=(0, 0, 0))
    # Second marker with different color for more uniqueness
    mx2 = 35
    mc2 = ((h * 53) % 200 + 55, (h * 97) % 200 + 55, (h * 137) % 200 + 55)
    n_dots = (h % 3) + 1
    for i in range(n_dots):
        draw.ellipse([mx2 + i * 10 - 3, my - 3, mx2 + i * 10 + 3, my + 3], fill=mc2)

    return img


# === IMAGE EMBEDDING CACHE (with EMA target ViT support) ===
class ImageEmbeddingCache:
    def __init__(self, vit_online, vit_processor, vit_target=None):
        self.cache = {};
        self.token_cache = {}
        self.vit = vit_online  # online encoder (receives gradients)
        self.vit_target = vit_target  # EMA target encoder (no gradients)
        self.processor = vit_processor
        self.device = next(vit_online.parameters()).device
        self.pixel_cache = {}
        print("\nPre-computing visual representations...")
        t0 = time.time()
        phrases = set()
        for r in TRANSITION_RULES.values():
            for s, a, res in r["triples"]: phrases.add(s); phrases.add(res)
        for ct in CHAIN_TRIPLES: phrases.add(ct["state"]); phrases.add(ct["result"]); phrases.add(ct["intermediate"])
        for act in ALL_ACTIONS_LIST: phrases.add(f"__action__{act}")
        print(f"  {len(phrases)} images to render and encode...")
        # Initial encoding uses target ViT (stable representations)
        enc_vit = vit_target if vit_target is not None else vit_online
        enc_vit.eval()
        with torch.no_grad():
            for p in sorted(phrases):
                if p.startswith("__action__"):
                    img = self._render_action(p.replace("__action__", ""))
                else:
                    img = render_state_image(p)
                inputs = self.processor(images=img, return_tensors="pt").to(self.device)
                self.pixel_cache[p] = inputs["pixel_values"].clone()
                outputs = enc_vit(**inputs)
                self.cache[p] = outputs.last_hidden_state[:, 0, :].squeeze(0).clone()
                self.token_cache[p] = outputs.last_hidden_state[0, 1:, :].clone()
        print(f"  Done! {len(self.cache)} cached in {time.time() - t0:.1f}s")

    def refresh(self):
        """Re-encode all images through the TARGET ViT (stable, no collapse)."""
        enc_vit = self.vit_target if self.vit_target is not None else self.vit
        enc_vit.eval()
        with torch.no_grad():
            for p, pixels in self.pixel_cache.items():
                outputs = enc_vit(pixel_values=pixels)
                self.cache[p] = outputs.last_hidden_state[:, 0, :].squeeze(0).clone()
                self.token_cache[p] = outputs.last_hidden_state[0, 1:, :].clone()

    def encode_live(self, state_text):
        """Encode through ONLINE ViT WITH gradients (for training)."""
        if state_text in self.pixel_cache:
            pixels = self.pixel_cache[state_text]
        else:
            img = render_state_image(state_text)
            inputs = self.processor(images=img, return_tensors="pt").to(self.device)
            pixels = inputs["pixel_values"]
            self.pixel_cache[state_text] = pixels.clone()
        outputs = self.vit(pixel_values=pixels)  # online ViT
        return outputs.last_hidden_state[:, 0, :].squeeze(0), outputs.last_hidden_state[0, 1:, :]

    def encode_target(self, state_text):
        """Encode through TARGET ViT WITHOUT gradients (for target vectors)."""
        if state_text in self.pixel_cache:
            pixels = self.pixel_cache[state_text]
        else:
            img = render_state_image(state_text)
            inputs = self.processor(images=img, return_tensors="pt").to(self.device)
            pixels = inputs["pixel_values"]
            self.pixel_cache[state_text] = pixels.clone()
        enc_vit = self.vit_target if self.vit_target is not None else self.vit
        with torch.no_grad():
            outputs = enc_vit(pixel_values=pixels)
        return outputs.last_hidden_state[:, 0, :].squeeze(0), outputs.last_hidden_state[0, 1:, :]

    def _render_action(self, act_name, size=224):
        """Render action as a distinctive arrow/icon image."""
        img = Image.new('RGB', (size, size), (240, 240, 240));
        draw = ImageDraw.Draw(img)
        h = _hash(act_name)
        r, g, b = (h * 37) % 200 + 55, (h * 73) % 200 + 55, (h * 113) % 200 + 55
        cx, cy = size // 2, size // 2
        # Large directional arrow
        draw.polygon([(cx - 50, cy - 20), (cx + 30, cy - 20), (cx + 30, cy - 40), (cx + 70, cy), (cx + 30, cy + 40),
                      (cx + 30, cy + 20), (cx - 50, cy + 20)],
                     fill=(r, g, b), outline=(0, 0, 0), width=2)
        # Action-specific pattern
        pattern = h % 3
        if pattern == 0:  # dots
            for i in range(3):
                draw.ellipse([cx - 60 + i * 15 - 4, cy + 50 - 4, cx - 60 + i * 15 + 4, cy + 50 + 4], fill=(r, g, b))
        elif pattern == 1:  # bars
            for i in range(3):
                draw.rectangle([cx - 60 + i * 20, cy + 45, cx - 60 + i * 20 + 12, cy + 55], fill=(r, g, b))
        else:  # zigzag
            pts = [(cx - 60 + i * 15, cy + 45 + (i % 2) * 10) for i in range(6)]
            if len(pts) > 1: draw.line(pts, fill=(r, g, b), width=3)
        return img

    def get(self, state_text):
        if state_text in self.cache: return self.cache[state_text]
        img = render_state_image(state_text)
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad(): outputs = self.vit(**inputs); vec = outputs.last_hidden_state[:, 0, :].squeeze(0)
        self.cache[state_text] = vec.clone();
        return vec

    def get_action(self, action_name):
        return self.get(f"__action__{action_name}")

    def get_tokens(self, state_text):
        if state_text in self.token_cache: return self.token_cache[state_text]
        img = render_state_image(state_text)
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad(): outputs = self.vit(**inputs); tokens = outputs.last_hidden_state[0, 1:, :]
        self.token_cache[state_text] = tokens.clone();
        return tokens


# === MODEL COMPONENTS (v2 — no projection head) ===
class SlotAttention(nn.Module):
    def __init__(self, sd, num_slots=2, n_iters=2):
        super().__init__();
        self.num_slots = num_slots;
        self.n_iters = n_iters;
        self.sd = sd
        self.slot_mu = nn.Parameter(torch.randn(num_slots, sd) * 0.02);
        self.k_proj = nn.Linear(sd, sd);
        self.v_proj = nn.Linear(sd, sd);
        self.q_proj = nn.Linear(sd, sd)
        self.gru = nn.GRUCell(sd, sd);
        self.mlp = nn.Sequential(nn.Linear(sd, sd * 2), nn.GELU(), nn.Linear(sd * 2, sd));
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
            q = self.q_proj(slots);
            al = torch.matmul(q, k.T) / self.scale
            attn = F.softmax(al, dim=0);
            aw = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8);
            updates = torch.matmul(aw, v)
            slots = self.gru(updates.reshape(-1, self.sd), sp.reshape(-1, self.sd));
            slots = slots.reshape(self.num_slots, self.sd);
            slots = slots + self.mlp(self.norm2(slots))
        return slots, attn


class ObjectExtractor(nn.Module):
    def __init__(self, sd):
        super().__init__();
        self.query = nn.Parameter(torch.randn(sd) * 0.02);
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
        self.identity = nn.Sequential(nn.Linear(sd, sd), nn.Tanh());
        self.negate_v = nn.Parameter(torch.randn(sd) * 0.1)
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
        return x - 2 * (torch.dot(v, x) / (torch.dot(v, v) + 1e-8)) * v

    def _base(self, i, s):
        if i == 0: return s + torch.sigmoid(self.gates[0]) * self.identity(s)
        if i == 1: return self._householder(s)
        if i == 2: return s + torch.sigmoid(self.gates[2]) * self.morph(s)
        if i == 3:
            q, k, v = self.aq(s), self.ak(s), self.av(s);
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
        print(f"  [COMPRESS] Created: {name}");
        return self.n - 1


class SlotSelector(nn.Module):
    def __init__(self, ad, ns):
        super().__init__();
        self.net = nn.Sequential(nn.Linear(ad, 128), nn.GELU(), nn.Linear(128, ns))

    def forward(self, a): return F.softmax(self.net(a), dim=-1)


class PropertyUpdater(nn.Module):
    def __init__(self, sd, ad, ms):
        super().__init__();
        self.step_emb = nn.Embedding(ms, sd);
        self.action_film = nn.Sequential(nn.Linear(ad, sd * 2), nn.GELU(), nn.Linear(sd * 2, sd * 2));
        self.sd = sd

    def forward(self, pv, action, prog, lib):
        film = self.action_film(action);
        scale, shift = film[:self.sd], film[self.sd:]
        for i, sel in enumerate(prog):
            pv = pv + self.step_emb(torch.tensor(i, device=pv.device));
            pv = pv * (1 + 0.1 * torch.tanh(scale)) + 0.1 * torch.tanh(shift);
            pv = lib.apply_soft(sel, pv)
        return pv


class PreconditionChecker(nn.Module):
    def __init__(self, sd, ad):
        super().__init__();
        self.net = nn.Sequential(nn.Linear(sd + ad, sd // 2), nn.GELU(), nn.Linear(sd // 2, 1), nn.Sigmoid())

    def forward(self, s, a): return self.net(torch.cat([s, a]))


class RuleSynthesizer(nn.Module):
    def __init__(self, ed, sd, np_, ms, pd=256):
        super().__init__();
        self.pd = pd;
        self._np = np_
        self.proj = nn.Sequential(nn.Linear(ed, pd), nn.LayerNorm(pd), nn.GELU())
        self.a1 = nn.MultiheadAttention(pd, 4, batch_first=True);
        self.n1 = nn.LayerNorm(pd);
        self.n2 = nn.LayerNorm(pd)
        self.f1 = nn.Sequential(nn.Linear(pd, pd * 2), nn.GELU(), nn.Linear(pd * 2, pd))
        self.a2 = nn.MultiheadAttention(pd, 4, batch_first=True);
        self.n3 = nn.LayerNorm(pd);
        self.n4 = nn.LayerNorm(pd)
        self.f2 = nn.Sequential(nn.Linear(pd, pd * 2), nn.GELU(), nn.Linear(pd * 2, pd))
        self.heads = nn.ModuleList(
            [nn.Sequential(nn.Linear(pd, 128), nn.GELU(), nn.Linear(128, np_)) for _ in range(ms)])
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
            n = nn.Sequential(nn.Linear(self.pd, 128), nn.GELU(), nn.Linear(128, new_n)).to(next(h.parameters()).device)
            with torch.no_grad():
                n[0].weight.copy_(h[0].weight);
                n[0].bias.copy_(h[0].bias);
                n[2].weight[:old].copy_(h[2].weight);
                n[2].bias[:old].copy_(h[2].bias)
                aw = h[2].weight.mean(0, keepdim=True).expand(new_n - old, -1);
                n[2].weight[old:].copy_(aw + torch.randn_like(aw) * 0.05);
                n[2].bias[old:].fill_(h[2].bias[:old].mean().item())
            nh.append(n)
        self.heads = nh;
        self._np = new_n

    def forward(self, ex, state, temp=0.8, np_=None, min_steps=2):
        pat = self.signature(ex);
        np_ = np_ or self._np;
        prog = [];
        stop_probs = []
        for i, h in enumerate(self.heads):
            if i >= min_steps:
                sp = self.stop(torch.cat([pat, state]));
                stop_probs.append(sp)
                if not self.training and sp.item() > 0.6: break
            lg = h(pat)
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


class LatentPredictor(nn.Module):
    def __init__(self, sd, ad, pd):
        super().__init__()
        inp = sd + sd + ad + pd
        self.net = nn.Sequential(
            nn.Linear(inp, sd * 2), nn.LayerNorm(sd * 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(sd * 2, sd * 2), nn.LayerNorm(sd * 2), nn.GELU(),
            nn.Linear(sd * 2, sd), nn.LayerNorm(sd), nn.GELU(),
            nn.Linear(sd, sd))

    def forward(self, obj, prop, action, sig): return self.net(torch.cat([obj, prop, action, sig]))


# === COMPLETE JEPA MODEL (v2 — no projection head) ===
class JEPAWorldModel(nn.Module):
    def __init__(self, sd):
        super().__init__();
        cfg = CONFIG;
        ed = sd * 3;
        ns = cfg["num_slots"];
        self.sd = sd
        self.obj_ext = ObjectExtractor(sd).to(DEVICE);
        self.slot_attn = SlotAttention(sd, ns).to(DEVICE)
        self.lib = PrimitiveLibrary(sd).to(DEVICE);
        self.syn = RuleSynthesizer(ed, sd, len(BASE_NAMES), cfg["max_program_steps"], cfg["proj_dim"]).to(DEVICE)
        self.slot_selector = SlotSelector(sd, ns).to(DEVICE);
        self.prop_updater = PropertyUpdater(sd, sd, cfg["max_program_steps"]).to(DEVICE)
        self.precond = PreconditionChecker(sd, sd).to(DEVICE);
        self.latent_pred = LatentPredictor(sd, sd, cfg["proj_dim"]).to(DEVICE)
        self.goal_eval = GoalEvaluator(sd).to(DEVICE)
        self.mem = Memory(cfg["memory_capacity"])

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
                p = self.prop_updater(p, ad, sp, self.lib)
            transformed = p
        else:
            transformed = self.prop_updater(pv, ta, prog, self.lib)
        pred_vec = self.latent_pred(obj_vec, transformed, ta, sig)
        return pred_vec, prog, sig, from_mem, stop_probs, oa, sa, sw, obj_vec, pc

    def simulate_step(self, cache, current_state_vec, current_tokens, action_name):
        """Simulate one step through the real World Model.
        Takes current state vector + tokens, action name, returns predicted next state vector."""
        # Get examples for this action
        rule = TRANSITION_RULES.get(action_name)
        if rule is None: return current_state_vec
        triples = rule["triples"]
        n_ex = min(3, len(triples))
        examples = random.sample(triples, n_ex)
        ex_reprs = torch.stack([torch.cat([cache.get(s), cache.get_action(a), cache.get(res)])
                                for s, a, res in examples])
        action_vec = cache.get_action(action_name)

        # Run through the World Model
        obj_vec, _ = self.obj_ext(current_tokens)
        slots, _ = self.slot_attn(current_tokens)
        sw = self.slot_selector(action_vec)
        pv = (sw.unsqueeze(1) * slots).sum(0)

        np_ = self.lib.n
        prog, _, _ = self.syn(ex_reprs.to(DEVICE), pv, 0.1, np_)
        sig = self.syn.signature(ex_reprs.to(DEVICE))
        transformed = self.prop_updater(pv, action_vec, prog, self.lib)
        pred_vec = self.latent_pred(obj_vec, transformed, action_vec, sig)
        return pred_vec

    def plan(self, cache, init_state, goal_state, max_depth=3, beam_width=3):
        """Model-based beam search planning: maintain top-k paths through
        the World Model, pick the plan with highest final goal similarity."""
        self.eval()
        with torch.no_grad():
            sv = cache.get(init_state);
            gv = cache.get(goal_state)
            if self.goal_eval(sv, gv).item() > 0.9: return [], self.goal_eval(sv, gv).item()

            cur_tokens = cache.get_tokens(init_state)
            # Each beam: (score, actions_list, current_vec, current_tokens, prev_action_idx)
            beams = [(F.cosine_similarity(sv.unsqueeze(0), gv.unsqueeze(0)).item(), [], sv, cur_tokens, None)]
            completed = []

            for step in range(max_depth):
                candidates = []
                for score, actions, cur, tokens, prev_act in beams:
                    for act_name in ALL_ACTIONS_LIST:
                        act_idx = ACT2IDX[act_name]
                        # Skip repeating same action
                        if prev_act is not None and act_idx == prev_act: continue
                        pred = self.simulate_step(cache, cur, tokens, act_name)
                        sim = F.cosine_similarity(pred.unsqueeze(0), gv.unsqueeze(0)).item()
                        new_actions = actions + [act_name]
                        candidates.append((sim, new_actions, pred, tokens, act_idx))

                if not candidates: break
                # Keep top beam_width candidates
                candidates.sort(key=lambda x: -x[0])
                beams = candidates[:beam_width]

                # Check if any beam reached the goal
                for sim, actions, cur, tokens, prev_act in beams:
                    if self.goal_eval(cur, gv).item() > 0.85:
                        completed.append((sim, actions, cur))

                if completed: break

            # Pick best from completed or from current beams
            if completed:
                completed.sort(key=lambda x: -x[0])
                best = completed[0]
                return best[1], self.goal_eval(best[2], gv).item()
            else:
                best = beams[0]
                return best[1], self.goal_eval(best[2], gv).item()

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


# === TASK GENERATION ===
def generate_single_task(cache, s2i, num_examples=5, rule_name=None, rules_dict=None):
    rules_dict = rules_dict or TRANSITION_RULES
    rule_name = rule_name or random.choice(list(rules_dict.keys()));
    r = rules_dict[rule_name];
    ts = r["triples"]
    ne = min(num_examples, len(ts) - 1);
    chosen = random.sample(ts, ne + 1);
    examples, test = chosen[:ne], chosen[ne]
    return {"example_reprs": torch.stack(
        [torch.cat([cache.get(s), cache.get_action(a), cache.get(res)]) for s, a, res in examples]),
            "test_state": cache.get(test[0]), "test_action": cache.get_action(test[1]),
            "test_state_tokens": cache.get_tokens(test[0]),
            "target_vec": cache.get(test[2]), "rule_name": rule_name, "state": test[0], "action": test[1],
            "expected": test[2],
            "depth": 1, "task_type": "single", "result_state_vec": cache.get(test[2]), "action_idx": ACT2IDX[test[1]],
            "precondition": r["precondition"], "property_changed": r["property_changed"],
            "precondition_holds": r["precondition"] in test[0], "property_idx": PROP2IDX[r["property_changed"]]}


def generate_multistep_task(cache, s2i, chains, num_examples=3):
    avail = [d for d in [2, 3] if chains.get(d)]
    if not avail: return generate_single_task(cache, s2i)
    depth = random.choice(avail);
    pool = chains[depth];
    ne = min(num_examples, len(pool) - 1)
    chosen = random.sample(pool, ne + 1);
    examples, test = chosen[:ne], chosen[ne]
    er = [torch.cat([cache.get(c['states'][0]), torch.stack([cache.get_action(a) for a in c["actions"]]).mean(0),
                     cache.get(c['states'][-1])]) for c in examples]
    ta = [cache.get_action(a) for a in test["actions"]]
    pse = []
    for rn in test["rules"]:
        r = TRANSITION_RULES[rn];
        st = random.sample(r["triples"], min(3, len(r["triples"])))
        pse.append(torch.stack([torch.cat([cache.get(s), cache.get_action(a), cache.get(res)]) for s, a, res in st]))
    fr = TRANSITION_RULES[test["rules"][0]]
    return {"example_reprs": torch.stack(er), "test_state": cache.get(test['states'][0]),
            "test_action": torch.stack(ta).mean(0), "test_actions": ta,
            "test_state_tokens": cache.get_tokens(test['states'][0]), "target_vec": cache.get(test['states'][-1]),
            "rule_name": "+".join(test["rules"]), "state": test["states"][0], "action": "+".join(test["actions"]),
            "expected": test["states"][-1],
            "depth": depth, "task_type": "multistep", "result_state_vec": cache.get(test['states'][-1]),
            "action_idx": ACT2IDX[test["actions"][0]],
            "precondition": fr["precondition"], "property_changed": fr["property_changed"], "precondition_holds": True,
            "property_idx": PROP2IDX[fr["property_changed"]], "per_step_examples": pse, "step_rules": test["rules"]}


def generate_chain_effect_task(cache, s2i):
    if not CHAIN_TRIPLES: return generate_single_task(cache, s2i)
    ct = random.choice(CHAIN_TRIPLES);
    ne = min(3, len(CHAIN_TRIPLES) - 1)
    examples = random.sample([c for c in CHAIN_TRIPLES if c != ct], ne)
    er = [torch.cat([cache.get(c['state']), cache.get_action(c['action']), cache.get(c['result'])]) for c in examples]
    return {"example_reprs": torch.stack(er), "test_state": cache.get(ct['state']),
            "test_action": cache.get_action(ct['action']),
            "test_state_tokens": cache.get_tokens(ct['state']), "target_vec": cache.get(ct['result']),
            "rule_name": f"chain:{'+'.join(ct['rules'])}", "state": ct['state'], "action": ct['action'],
            "expected": ct['result'],
            "depth": 1, "task_type": "chain", "result_state_vec": cache.get(ct['result']),
            "action_idx": ACT2IDX[ct['action']],
            "precondition": "on", "property_changed": "position", "precondition_holds": True,
            "property_idx": PROP2IDX["position"]}


def generate_mixed_task(cache, s2i, chains, ne=5):
    r = random.random()
    if r < CONFIG["chain_effect_ratio"] and CHAIN_TRIPLES: return generate_chain_effect_task(cache, s2i)
    if r < CONFIG["chain_effect_ratio"] + CONFIG["multistep_ratio"] and any(
        chains.get(d) for d in [2, 3]): return generate_multistep_task(cache, s2i, chains, 3)
    return generate_single_task(cache, s2i, ne, rules_dict=TRAIN_RULES)


def generate_planning_task(cache, s2i, chains):
    avail = [d for d in [1, 2, 3] if chains.get(d)]
    if not avail: return None
    c = random.choice(chains[random.choice(avail)])
    return {"initial_state": c["states"][0], "goal_state": c["states"][-1], "correct_actions": c["actions"],
            "depth": len(c["actions"])}


# === LOSSES ===
def div_loss(prog):
    if len(prog) < 2: return torch.tensor(0.0, device=DEVICE)
    l = torch.tensor(0.0, device=DEVICE);
    n = len(prog)
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


# Build object-to-states index for contrastive loss
# Maps object name → list of all state texts for that object
OBJ_TO_STATES = defaultdict(list)
for r in TRANSITION_RULES.values():
    for s, a, res in r["triples"]:
        obj = s.split()[1]  # "the OBJ is ..."
        OBJ_TO_STATES[obj].append(s)
        res_obj = res.split()[1]
        OBJ_TO_STATES[res_obj].append(res)
# Deduplicate
for obj in OBJ_TO_STATES:
    OBJ_TO_STATES[obj] = list(set(OBJ_TO_STATES[obj]))


def object_contrastive_loss(pred_vec, target_state, cache, all_states):
    """Push pred_vec to be closer to same-object states than different-object states.
    This preserves object identity in the predicted vector."""
    obj_name = target_state.split()[1]  # "the OBJ is ..."
    same_obj_states = OBJ_TO_STATES.get(obj_name, [])
    if len(same_obj_states) < 1: return torch.tensor(0.0, device=DEVICE)

    # Positive: random state of the SAME object
    pos_state = random.choice(same_obj_states)
    pos_vec = cache.get(pos_state)
    pos_sim = F.cosine_similarity(pred_vec.unsqueeze(0), pos_vec.unsqueeze(0))

    # Negative: state of a DIFFERENT object with same property value
    # e.g., if target is "the mirror is broken", neg could be "the plate is broken"
    target_prop = target_state.split()[-1]  # last word: "broken", "hot", etc.
    diff_obj_states = [s for s in all_states if target_prop in s and s.split()[1] != obj_name]
    if not diff_obj_states:
        diff_obj_states = [s for s in all_states if s.split()[1] != obj_name]
    neg_state = random.choice(diff_obj_states)
    neg_vec = cache.get(neg_state)
    neg_sim = F.cosine_similarity(pred_vec.unsqueeze(0), neg_vec.unsqueeze(0))

    # Margin loss: pred should be closer to same-object than different-object
    margin = 0.05
    return F.relu(neg_sim - pos_sim + margin)


# === TRAINING & EVAL ===
def nearest_neighbor(pred_vec, all_states, cache):
    best_sim, best_state = -1, None
    for st in all_states:
        vec = cache.get(st);
        sim = F.cosine_similarity(pred_vec.unsqueeze(0), vec.unsqueeze(0)).item()
        if sim > best_sim: best_sim, best_state = sim, st
    return best_state, best_sim


def ema_update(online_model, target_model, decay=0.996):
    """Exponential Moving Average update: target = decay * target + (1-decay) * online"""
    with torch.no_grad():
        for o_param, t_param in zip(online_model.parameters(), target_model.parameters()):
            t_param.data.mul_(decay).add_(o_param.data, alpha=1 - decay)


def train_cycle(model, cache, s2i, i2s, chains, all_states, all_sv, n_iters, cycle, lr):
    cfg = CONFIG
    # Separate param groups: model at full LR, online ViT at 10x lower LR
    vit_params = [p for p in cache.vit.parameters() if p.requires_grad]
    model_params = [p for p in model.parameters() if p.requires_grad]
    param_groups = [{"params": model_params, "lr": lr}, {"params": vit_params, "lr": lr * 0.1}]
    opt = torch.optim.AdamW(param_groups, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: s / 200 if s < 200 else 0.5 * (
                1 + math.cos(math.pi * (s - 200) / max(n_iters - 200, 1))))
    model.train();
    cache.vit.train();
    ok, tot = 0, 0;
    t0 = time.time()
    for it in range(n_iters):
        task = generate_mixed_task(cache, s2i, chains);
        np_ = model.lib.n

        # ONLINE ViT encodes input state (with gradients)
        test_state_text = task["state"]
        live_vec, live_tokens = cache.encode_live(test_state_text)
        task["test_state"] = live_vec
        task["test_state_tokens"] = live_tokens

        # TARGET ViT encodes target state (no gradients, prevents collapse)
        target_text = task["expected"]
        target_vec_ema, _ = cache.encode_target(target_text)
        task["target_vec"] = target_vec_ema.detach()
        task["result_state_vec"] = target_vec_ema.detach()

        pred_vec, prog, sig, _, stop_probs, _, _, _, obj_vec, precond_sc = model(task, cfg["temperature"],
                                                                                 use_mem=False)
        target_vec = task["target_vec"].to(DEVICE)
        cosine_loss = 1.0 - F.cosine_similarity(pred_vec.unsqueeze(0), target_vec.unsqueeze(0))
        mse_loss = F.mse_loss(pred_vec, target_vec)
        latent_loss = cosine_loss + 0.5 * mse_loss
        stop_l = torch.tensor(0.0, device=DEVICE)
        if stop_probs:
            for sp in stop_probs: stop_l = stop_l + (1.0 - sp).squeeze()
            stop_l = stop_l / len(stop_probs)
        rv = task["result_state_vec"].to(DEVICE);
        gl = goal_contrastive_loss(model.goal_eval, rv, all_sv)
        pt = torch.tensor(1.0 if task["precondition_holds"] else 0.0, device=DEVICE);
        pl = F.binary_cross_entropy(precond_sc.view(-1)[0], pt)
        loss = (latent_loss + 0.1 * div_loss(prog) + 0.05 * use_loss(prog, np_) + 0.1 * novelty_loss(prog,
                                                                                                     np_) + 0.03 * stop_l + 0.02 * len(
            prog) + cfg["goal_loss_weight"] * gl + cfg["precondition_loss_weight"] * pl) / cfg["grad_accumulation"]
        loss.backward()
        with torch.no_grad():
            pred_state, sim = nearest_neighbor(pred_vec, all_states, cache);
            correct = pred_state == task["expected"]
        ok += correct;
        tot += 1
        with torch.no_grad():
            model.memorize(sig, prog, task["rule_name"], correct)
        if (it + 1) % cfg["grad_accumulation"] == 0:
            nn.utils.clip_grad_norm_(model_params + vit_params, 1.0);
            opt.step();
            opt.zero_grad();
            sch.step()
            # EMA update: target ViT slowly follows online ViT
            if cache.vit_target is not None:
                ema_update(cache.vit, cache.vit_target, decay=0.996)
        if it % 500 == 0:
            acc = 100 * ok / max(tot, 1);
            speed = (it + 1) / max(time.time() - t0, 0.01)
            ms = f" [D{task['depth']}]" if task.get("task_type") == "multistep" else ""
            print(
                f"  [C{cycle}] {it:5d}/{n_iters} | cos:{cosine_loss.item():.3f} mse:{mse_loss.item():.3f} | Acc:{acc:4.0f}% | {speed:.0f}it/s | [{task['rule_name'][:12]:12s}] {'V' if correct else 'X'} K={len(prog)}{ms}")
            if it > 0 and it % 3000 == 0: ok, tot = 0, 0
        # Refresh cache periodically using target ViT
        if (it + 1) % 2000 == 0:
            cache.refresh()
            all_sv[:] = [cache.get(s).to(DEVICE) for s in all_states]
    # Final refresh after training cycle
    cache.refresh()
    all_sv[:] = [cache.get(s).to(DEVICE) for s in all_states]


def evaluate(model, cache, s2i, i2s, chains, all_states):
    cfg = CONFIG;
    model.eval();
    ne = cfg["eval_samples"]
    print(f"\n  --- Single-Step NN (n={ne}) ---\n")
    results = {}
    with torch.no_grad():
        for rule in sorted(TRANSITION_RULES.keys()):
            ok, shown = 0, 0
            for _ in range(ne):
                t = generate_single_task(cache, s2i, rule_name=rule)
                pv, _, _, _, _, _, _, _, _, _ = model(t, 0.1, use_mem=True)
                ps, sim = nearest_neighbor(pv, all_states, cache);
                ok += ps == t["expected"]
                if shown < 1: print(
                    f"    [{rule:12s}] '{t['state'][:20]}' + '{t['action']:10s}' -> {'V' if ps == t['expected'] else 'X'} (sim:{sim:.3f})"); shown += 1
            a = 100 * ok / ne;
            results[rule] = a;
            print(f"      -> {a:.0f}%")
    print("\n  --- Summary ---")
    for r, a in sorted(results.items(), key=lambda x: -x[1]): print(
        f"    {r:12s}: {a:5.1f}% [{chr(9608) * int(a / 5)}]")
    ov = sum(results.values()) / len(results);
    print(f"\n    SINGLE-STEP (NN): {ov:.1f}%")
    print("\n  --- Latent Similarity ---")
    ts, ns_ = 0, 0
    with torch.no_grad():
        for rule in sorted(TRANSITION_RULES.keys()):
            sims = []
            for _ in range(10):
                t = generate_single_task(cache, s2i, rule_name=rule);
                pv, _, _, _, _, _, _, _, _, _ = model(t, 0.1, use_mem=False)
                tv = t["target_vec"].to(DEVICE);
                sims.append(F.cosine_similarity(pv.unsqueeze(0), tv.unsqueeze(0)).item())
            avg = sum(sims) / len(sims);
            ts += avg;
            ns_ += 1;
            print(f"    {rule:12s}: cos_sim={avg:.3f}")
    print(f"    AVERAGE: {ts / ns_:.3f}")
    for depth in [2, 3]:
        if not chains.get(depth): continue
        print(f"\n  --- Depth-{depth} ---");
        ok, tested = 0, 0
        with torch.no_grad():
            for _ in range(min(30, len(chains[depth]))):
                t = generate_multistep_task(cache, s2i, chains, 3)
                if t["depth"] != depth: continue
                pv, _, _, _, _, _, _, _, _, _ = model(t, 0.1, use_mem=False)
                ps, sim = nearest_neighbor(pv, all_states, cache);
                ok += ps == t["expected"];
                tested += 1
                if tested <= 2: print(
                    f"    '{t['state'][:20]}' +[{t['action'][:20]}] -> {'V' if ps == t['expected'] else 'X'} (sim:{sim:.3f})")
        if tested: print(f"    DEPTH-{depth}: {100 * ok / tested:.0f}% [{tested} tested]")
    print(f"\n  --- Planning ---")
    pl_1st, pl_full, pl_t = 0, 0, 0
    real_1st, real_full, real_t = 0, 0, 0
    for _ in range(50):
        task = generate_planning_task(cache, s2i, chains)
        if not task: continue
        plan, sc = model.plan(cache, task["initial_state"], task["goal_state"], max_depth=task["depth"])
        is_round_trip = task["initial_state"] == task["goal_state"]
        if plan:
            fc = plan[0] == task["correct_actions"][0]
            full_c = plan[:len(task["correct_actions"])] == task["correct_actions"]
        else:
            fc = is_round_trip;
            full_c = is_round_trip
        pl_1st += fc;
        pl_full += full_c;
        pl_t += 1
        if not is_round_trip: real_1st += fc; real_full += full_c; real_t += 1
        if pl_t <= 6:
            rt = " [ROUND]" if is_round_trip else ""
            print(f"    {task['initial_state'][:22]} -> {task['goal_state'][:22]}{rt}")
            print(
                f"      correct:{task['correct_actions']}  plan:{plan}  1st:{'V' if fc else 'X'} full:{'V' if full_c else 'X'} score:{sc:.3f}")
    if pl_t: print(f"\n    ALL:  1st:{100 * pl_1st / pl_t:.0f}% full:{100 * pl_full / pl_t:.0f}% ({pl_t} tested)")
    if real_t: print(
        f"    REAL: 1st:{100 * real_1st / real_t:.0f}% full:{100 * real_full / real_t:.0f}% ({real_t} tested)")
    return results


def evaluate_generalization(model, cache, s2i, i2s, all_states):
    """Test generalization to held-out objects, novel combinations, and counterfactual visuals."""
    model.eval()

    # =========================================================================
    # TEST 1: HELD-OUT OBJECTS
    # The model was trained WITHOUT these objects. It must apply learned rules
    # to objects it has never seen during training.
    # Examples from training objects are used, but the test object is held-out.
    # =========================================================================
    print(f"\n  {'=' * 55}")
    print(f"  GENERALIZATION TEST 1: Held-Out Objects")
    print(f"  (model never trained on these objects)")
    print(f"  {'=' * 55}")

    ho_ok, ho_total = 0, 0
    ho_results = {}
    with torch.no_grad():
        for act_name in sorted(HELDOUT_RULES.keys()):
            hr = HELDOUT_RULES[act_name]
            tr = TRAIN_RULES.get(act_name)
            if not tr or len(tr["triples"]) < 3: continue

            rule_ok = 0
            for s, a, res in hr["triples"]:
                # Examples come from TRAINING objects only
                train_examples = random.sample(tr["triples"], min(3, len(tr["triples"])))
                ex_reprs = torch.stack([torch.cat([cache.get(es), cache.get_action(ea), cache.get(eres)])
                                        for es, ea, eres in train_examples])
                task = {
                    "example_reprs": ex_reprs,
                    "test_state": cache.get(s), "test_action": cache.get_action(a),
                    "test_state_tokens": cache.get_tokens(s),
                    "target_vec": cache.get(res), "result_state_vec": cache.get(res),
                    "rule_name": act_name, "state": s, "action": a, "expected": res,
                    "depth": 1, "task_type": "single", "action_idx": ACT2IDX[a],
                    "precondition": hr["precondition"], "property_changed": hr["property_changed"],
                    "precondition_holds": True, "property_idx": PROP2IDX[hr["property_changed"]],
                }
                pv, _, _, _, _, _, _, _, _, _ = model(task, 0.1, use_mem=False)
                ps, sim = nearest_neighbor(pv, all_states, cache)
                correct = ps == res
                ho_ok += correct;
                ho_total += 1;
                rule_ok += correct

                if ho_total <= 8:
                    obj = s.split()[1]
                    print(
                        f"    [{act_name:12s}] {obj:8s}: '{s[:22]}' + '{a}' -> {'V' if correct else 'X'} (sim:{sim:.3f})")
                    if not correct:
                        print(f"      predicted: '{ps[:25]}' want: '{res[:25]}'")

            n_held = len(hr["triples"])
            pct = 100 * rule_ok / n_held if n_held > 0 else 0
            ho_results[act_name] = pct

    print(f"\n    Per-action held-out accuracy:")
    for act, pct in sorted(ho_results.items(), key=lambda x: -x[1]):
        print(f"      {act:12s}: {pct:5.1f}%")
    if ho_total:
        print(f"\n    HELD-OUT OBJECTS: {100 * ho_ok / ho_total:.1f}% ({ho_ok}/{ho_total})")

    # =========================================================================
    # TEST 2: COUNTERFACTUAL RENDERING
    # Render images with swapped visual cues and test if model still works.
    # E.g., "hot" rendered with cold visuals (blue, ice) to check if model
    # relies on visual shortcuts or learned the rule structure.
    # =========================================================================
    print(f"\n  {'=' * 55}")
    print(f"  GENERALIZATION TEST 2: Counterfactual Visuals")
    print(f"  (correct state but WRONG visual cues)")
    print(f"  {'=' * 55}")

    # Create counterfactual images: state says "hot" but render with "cold" visuals
    counterfactual_pairs = [
        ("the water is hot", "cold", "cool", "the water is cold"),  # hot rendered as cold visual
        ("the coffee is cold", "hot", "heat", "the coffee is hot"),  # cold rendered as hot visual
        ("the door is open", "closed", "close", "the door is closed"),  # open rendered as closed
        ("the box is closed", "open", "open", "the box is open"),  # closed rendered as open
        ("the lamp is on", "off", "switch off", "the lamp is off"),  # on rendered as off
        ("the fan is off", "on", "switch on", "the fan is on"),  # off rendered as on
    ]

    cf_ok, cf_total = 0, 0
    with torch.no_grad():
        for true_state, fake_visual, action, expected in counterfactual_pairs:
            # Render the image with WRONG visual cue
            parts = true_state.split()
            obj_name = parts[1]
            # Create fake state text for rendering only
            fake_state = true_state.replace(f"is {parts[-1]}", f"is {fake_visual}")

            # Encode the counterfactual image
            cf_img = render_state_image(fake_state)
            inputs = cache.processor(images=cf_img, return_tensors="pt").to(cache.device)
            with torch.no_grad():
                outputs = cache.vit(**inputs)
                cf_vec = outputs.last_hidden_state[:, 0, :].squeeze(0)
                cf_tokens = outputs.last_hidden_state[0, 1:, :]

            # Use training examples for this rule
            rule = TRAIN_RULES.get(action)
            if not rule: continue
            train_ex = random.sample(rule["triples"], min(3, len(rule["triples"])))
            ex_reprs = torch.stack([torch.cat([cache.get(s), cache.get_action(a), cache.get(r)])
                                    for s, a, r in train_ex])

            task = {
                "example_reprs": ex_reprs,
                "test_state": cf_vec, "test_action": cache.get_action(action),
                "test_state_tokens": cf_tokens,
                "target_vec": cache.get(expected), "result_state_vec": cache.get(expected),
                "rule_name": action, "state": f"{true_state} [CF:{fake_visual}]",
                "action": action, "expected": expected,
                "depth": 1, "task_type": "single", "action_idx": ACT2IDX[action],
                "precondition": rule["precondition"], "property_changed": rule["property_changed"],
                "precondition_holds": True, "property_idx": PROP2IDX[rule["property_changed"]],
            }
            pv, _, _, _, _, _, _, _, _, _ = model(task, 0.1, use_mem=False)
            ps, sim = nearest_neighbor(pv, all_states, cache)

            # For counterfactual: check if model predicts based on action (correct)
            # or based on visual shortcut (wrong)
            correct = ps == expected
            cf_ok += correct;
            cf_total += 1

            print(f"    '{true_state[:20]}' [visual={fake_visual:6s}] + '{action:10s}' -> "
                  f"'{ps[:22]}' {'V' if correct else 'X'} (sim:{sim:.3f})")
            if not correct:
                print(f"      want: '{expected[:25]}'")

    if cf_total:
        print(f"\n    COUNTERFACTUAL: {100 * cf_ok / cf_total:.1f}% ({cf_ok}/{cf_total})")
        if cf_ok / cf_total > 0.5:
            print(f"    -> Model relies on rule structure, not just visual shortcuts")
        else:
            print(f"    -> Model may rely on visual shortcuts (renderer bias)")

    # =========================================================================
    # TEST 3: CROSS-PROPERTY (novel object-action combinations)
    # Apply actions to objects that don't normally have that property.
    # E.g., "the door is cold + heat" — door is openness, not temperature.
    # The model has never seen "the door is cold" in training.
    # We test if the learned rule generalizes beyond the original object set.
    # =========================================================================
    print(f"\n  {'=' * 55}")
    print(f"  GENERALIZATION TEST 3: Cross-Property Transfer")
    print(f"  (apply actions to objects from different property types)")
    print(f"  {'=' * 55}")

    cross_tests = [
        # (novel_state, action, expected_result, description)
        ("the door is cold", "heat", "the door is hot", "openness obj + temperature action"),
        ("the lamp is cold", "heat", "the lamp is hot", "power obj + temperature action"),
        ("the water is closed", "open", "the water is open", "temperature obj + openness action"),
        ("the ball is empty", "fill", "the ball is full", "position obj + fullness action"),
        ("the key is intact", "drop", "the key is broken", "containment obj + integrity action"),
        ("the cup is off", "switch on", "the cup is on", "fullness obj + power action"),
    ]

    cp_ok, cp_total = 0, 0
    with torch.no_grad():
        for novel_state, action, expected, desc in cross_tests:
            # Encode the novel image (renderer will handle it generically)
            rule = TRAIN_RULES.get(action)
            if not rule: continue

            train_ex = random.sample(rule["triples"], min(3, len(rule["triples"])))
            ex_reprs = torch.stack([torch.cat([cache.get(s), cache.get_action(a), cache.get(r)])
                                    for s, a, r in train_ex])

            task = {
                "example_reprs": ex_reprs,
                "test_state": cache.get(novel_state), "test_action": cache.get_action(action),
                "test_state_tokens": cache.get_tokens(novel_state),
                "target_vec": cache.get(expected), "result_state_vec": cache.get(expected),
                "rule_name": action, "state": novel_state, "action": action, "expected": expected,
                "depth": 1, "task_type": "single", "action_idx": ACT2IDX[action],
                "precondition": rule["precondition"], "property_changed": rule["property_changed"],
                "precondition_holds": True, "property_idx": PROP2IDX[rule["property_changed"]],
            }
            pv, _, _, _, _, _, _, _, _, _ = model(task, 0.1, use_mem=False)
            ps, sim = nearest_neighbor(pv, all_states, cache)
            correct = ps == expected
            cp_ok += correct;
            cp_total += 1

            print(
                f"    [{desc[:30]:30s}] '{novel_state[:18]}' + '{action:10s}' -> {'V' if correct else 'X'} (sim:{sim:.3f})")
            if not correct:
                print(f"      got: '{ps[:25]}' want: '{expected[:25]}'")

    if cp_total:
        print(f"\n    CROSS-PROPERTY: {100 * cp_ok / cp_total:.1f}% ({cp_ok}/{cp_total})")

    print(f"\n  {'=' * 55}")
    print(f"  GENERALIZATION SUMMARY")
    print(f"  {'=' * 55}")
    if ho_total: print(f"    Held-out objects:     {100 * ho_ok / ho_total:5.1f}% ({ho_ok}/{ho_total})")
    if cf_total: print(f"    Counterfactual:       {100 * cf_ok / cf_total:5.1f}% ({cf_ok}/{cf_total})")
    if cp_total: print(f"    Cross-property:       {100 * cp_ok / cp_total:5.1f}% ({cp_ok}/{cp_total})")


def main():
    from transformers import ViTModel, ViTImageProcessor
    print(f"Device: {DEVICE}")
    print("Loading ViT (google/vit-base-patch16-224)...")
    vit_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224").to(DEVICE)
    # Freeze all layers first
    for p in vit_model.parameters(): p.requires_grad = False
    # Unfreeze last 2 encoder layers + layernorm for domain adaptation
    for name, p in vit_model.named_parameters():
        if "encoder.layer.10" in name or "encoder.layer.11" in name or "layernorm" in name:
            p.requires_grad = True
    vit_trainable = sum(p.numel() for p in vit_model.parameters() if p.requires_grad)
    print(f"ViT online: {vit_trainable:,} trainable params (last 2 layers + layernorm)")

    # Create EMA target ViT (copy of online, fully frozen, updated via EMA)
    import copy
    vit_target = copy.deepcopy(vit_model).to(DEVICE)
    for p in vit_target.parameters(): p.requires_grad = False
    print(f"ViT target: EMA copy (decay=0.996, no gradients)")

    sl, s2i, i2s = build_vocab()
    print(f"\nRules:{len(TRANSITION_RULES)} | States:{len(sl)} | Actions:{NUM_ACTIONS}")
    chains = find_chains(CONFIG["max_chain_length"])
    cache = ImageEmbeddingCache(vit_model, vit_processor, vit_target=vit_target)
    all_sv = [cache.get(s).to(DEVICE) for s in sl];
    all_states = sl
    model = JEPAWorldModel(CONFIG["vit_dim"])
    tp = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {tp:,}\n")
    cfg = CONFIG
    for c in range(cfg["num_cycles"]):
        print(
            f"{'=' * 65}\nCYCLE {c + 1}/{cfg['num_cycles']} | Prims:{model.lib.n}\nLibrary: {model.lib.names}\n{'=' * 65}\n")
        train_cycle(model, cache, s2i, i2s, chains, all_states, all_sv, cfg["iters_per_cycle"][c], c + 1,
                    cfg["lr_per_cycle"][c])
        print(f"\n--- Eval Cycle {c + 1} ---")
        evaluate(model, cache, s2i, i2s, chains, all_states)
        print(f"\n  Memory: {model.mem.stats()}")
        if c < cfg["num_cycles"] - 1:
            print(f"\n--- Compression ---");
            n = model.compress()
            if n: print(f"  {n} new primitives.")
    print(f"\n{'=' * 65}\nFINAL TEST\n{'=' * 65}\nLibrary: {model.lib.names}")
    evaluate(model, cache, s2i, i2s, chains, all_states)
    print(f"\n{'=' * 65}\nGENERALIZATION TESTS\n{'=' * 65}")
    evaluate_generalization(model, cache, s2i, i2s, all_states)
    print(f"\n  Memory: {model.mem.stats()}\nDone.")


if __name__ == "__main__":
    main()
