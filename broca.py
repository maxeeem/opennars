import subprocess
import threading
import time
import os
import sys
import signal
import re
import math
import random
import argparse
from collections import Counter
import hashlib
import json
from dataclasses import dataclass
from typing import Callable, Tuple, Dict, List, Optional
import shutil

try:
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
except Exception:
    pass

try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    print("[Error] Missing dependencies. Run: pip install -r requirements.txt")
    sys.exit(1)

# --- Configuration ---
NARS_JAR = "target/opennars-3.0.4-SNAPSHOT.jar"
GLOVE_FILE = "glove-embeddings/glove.txt"
EMBEDDING_FILE = "water_embeddings.txt"
IMAGE_DIR = "retina_cache"

# Seed used for deterministic random vectors / projection.
RNG_SEED = 1337

@dataclass
class DomainSpec:
    name: str
    labels: Tuple[str, str]
    train_stimuli: Tuple[str, str]
    test_stimuli: Tuple[str, str]
    stimulus_tokens: Dict[str, str]  
    # Terms to vectorize (labels + control signals)
    vector_terms: List[str]
    asset_generator: Callable[[str, bool, int, str], None] # (token, is_train, seed, path)

    @property
    def all_stimuli(self) -> List[str]:
        return list(self.stimulus_tokens.keys())
    
    @property
    def embedding_file_base(self) -> str:
        return f"{self.name}_embeddings.txt"

# --- Domain Generators ---

def generate_water_wind(token: str, is_train: bool, seed: int, path: str):
    from PIL import Image, ImageDraw # type: ignore
    w, h = 224, 224
    rnd = random.Random(seed)
    img = Image.new("RGB", (w, h), (0, 0, 0))
    px = img.load()
    
    # Map token to conceptual class based on prefix/structure (hardcoded for this domain)
    # W1, W2 -> Water; N1, N2 -> Wind
    is_water = "W" in token
    
    if is_water:
        # Blue vertical gradient + smooth sine bands (deterministic).
        phase1 = rnd.random() * 2.0 * math.pi
        phase2 = rnd.random() * 2.0 * math.pi
        f1 = 2.0 + rnd.random() * 2.0
        f2 = 5.0 + rnd.random() * 3.0
        for y in range(h):
            gy = y / (h - 1)
            band = 0.35 * math.sin((gy * f1 * 2.0 * math.pi) + phase1) + 0.20 * math.sin((gy * f2 * 2.0 * math.pi) + phase2)
            for x in range(w):
                gx = x / (w - 1)
                ripple = 0.12 * math.sin(((gx * 6.0 + gy * 1.5) * 2.0 * math.pi) + phase2)
                v = max(0.0, min(1.0, gy * 0.65 + 0.20 + band + ripple))
                r = int(20 + 25 * v)
                g = int(90 + 110 * v)
                b = int(150 + 95 * v)
                px[x, y] = (r, g, b)
    else:
        # Wind: Fire-like pattern (Red/Orange/Yellow) - High contrast to Water (Blue).
        for y in range(h):
            for x in range(w):
                # Vertical gradient from Yellow (top) to Red (bottom)
                gy = y / h
                r = 255
                g = int(200 * (1.0 - gy))
                b = 0
                # Add some noise
                noise = rnd.randint(-30, 30)
                r = max(0, min(255, r + noise))
                g = max(0, min(255, g + noise))
                b = max(0, min(255, b + noise))
                px[x, y] = (r, g, b)
        
        # Add some "flames"
        draw = ImageDraw.Draw(img)
        count = 15
        for _ in range(count):
            x = rnd.randint(0, w)
            y = h
            h_flame = rnd.randint(50, 150)
            w_flame = rnd.randint(10, 40)
            draw.polygon([(x, y), (x - w_flame, y - h_flame), (x + w_flame, y - h_flame)], fill=(255, 50, 0))

    img.save(path, format="PNG")

def generate_shape(token: str, is_train: bool, seed: int, path: str):
    from PIL import Image, ImageDraw
    w, h = 224, 224
    rnd = random.Random(seed)
    
    # White background
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    is_circle = "C" in token # C1, C2 -> Circle
    # S1, S2 -> Square
    
    # Variations based on token
    # Train: Centered, specific color
    # Test: Off-center, different color/size
    
    cx, cy = w // 2, h // 2
    size = 80
    color = (0, 0, 0)
    
    if is_train: # C1 or S1
        # Train: Centered, Blue/Red
        shift_x = rnd.randint(-10, 10)
        shift_y = rnd.randint(-10, 10)
        size = rnd.randint(70, 90)
        color = (0, 0, 200) if is_circle else (200, 0, 0)
    else: # C2 or S2 (Test)
        # Test: Shifted, Green/Orange
        shift_x = rnd.randint(-40, 40)
        shift_y = rnd.randint(-40, 40)
        size = rnd.randint(50, 110)
        color = (0, 150, 0) if is_circle else (200, 150, 0)
    
    cx += shift_x
    cy += shift_y
    
    if is_circle:
        draw.ellipse([cx - size//2, cy - size//2, cx + size//2, cy + size//2], fill=color, outline=None)
    else:
        draw.rectangle([cx - size//2, cy - size//2, cx + size//2, cy + size//2], fill=color, outline=None)
        
    img.save(path, format="PNG")


COMMON_VECTOR_TERMS = [
    "confirm", "need_label", "seen", "heard", "say", "babble", "SELF", "TEACHER"
]

DOMAINS = {
    "water_wind": DomainSpec(
        name="water_wind",
        labels=("water", "wind"),
        train_stimuli=("sensation_W1", "sensation_N1"),
        test_stimuli=("sensation_W2", "sensation_N2"),
        stimulus_tokens={
            "sensation_W1": "procedural:water_train",
            "sensation_N1": "procedural:wind_train",
            "sensation_W2": "procedural:water_test",
            "sensation_N2": "procedural:wind_test",
        },
        vector_terms=["water", "wind"] + COMMON_VECTOR_TERMS,
        asset_generator=generate_water_wind
    ),
    "shape": DomainSpec(
        name="shape",
        labels=("circle", "square"),
        train_stimuli=("sensation_C1", "sensation_S1"),
        test_stimuli=("sensation_C2", "sensation_S2"),
        stimulus_tokens={
            "sensation_C1": "procedural:circle_train",
            "sensation_S1": "procedural:square_train",
            "sensation_C2": "procedural:circle_test",
            "sensation_S2": "procedural:square_test",
        },
        vector_terms=["circle", "square"] + COMMON_VECTOR_TERMS,
        asset_generator=generate_shape
    )
}

# Map abstract sensations to deterministic local procedural images.
# These token names are what NARS sees (opaque signals), not the labels.
# Order matters: we want train vectors available before ablating test vectors.
# VISUAL_STIMULI and VECTOR_TERMS are now derived from the selected domain.




def _embedding_file_has_all_vocab(filename: str, vocab: list[str]) -> bool:
    if not os.path.exists(filename):
        return False
    try:
        present: set[str] = set()
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                token = line.split(" ", 1)[0]
                present.add(token)
        return set(vocab).issubset(present)
    except OSError:
        return False


def _unit_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]


def _get_glove_dim(glove_path: str) -> int:
    with open(glove_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # token + dims
            return max(0, len(parts) - 1)
    raise RuntimeError(f"Empty glove file: {glove_path}")


def _load_glove_vectors(glove_path: str, wanted: set[str]) -> dict[str, list[float]]:
    found: dict[str, list[float]] = {}
    if not wanted:
        return found
    with open(glove_path, "r") as f:
        for line in f:
            if len(found) == len(wanted):
                break
            line = line.strip()
            if not line:
                continue
            token, rest = line.split(" ", 1)
            if token not in wanted:
                continue
            parts = rest.split()
            try:
                vec = [float(x) for x in parts]
            except ValueError:
                continue
            found[token] = _unit_normalize(vec)
    return found


def _deterministic_random_unit_vector(dim: int, token: str) -> list[float]:
    # Stable across runs and machines.
    digest = hashlib.sha256(f"{RNG_SEED}:{token}".encode("utf-8")).digest()
    seed = int.from_bytes(digest[:4], "big", signed=False)
    rnd = random.Random(seed)
    vec = [rnd.gauss(0.0, 1.0) for _ in range(dim)]
    return _unit_normalize(vec)


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _load_embedding_vectors(filename: str, wanted: set[str]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            token, rest = line.split(" ", 1)
            if token not in wanted:
                continue
            parts = rest.split()
            try:
                vec = [float(x) for x in parts]
            except ValueError:
                continue
            out[token] = vec
            if len(out) == len(wanted):
                break
    return out


class Retina:
    def __init__(self, domain_spec: DomainSpec):
        print("[Retina] Loading CLIP (ViT-B/32) image encoder...")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.domain = domain_spec

        os.makedirs(IMAGE_DIR, exist_ok=True)

    def fetch_images(self) -> dict[str, str]:
        try:
            from PIL import Image, ImageDraw  # type: ignore
        except ImportError:
            print("[Error] Missing dependency 'Pillow'. Run: pip install -r requirements.txt")
            sys.exit(1)
            
        print("[Retina] Using deterministic procedural stimuli (no network)")
        local_paths: dict[str, str] = {}
        
        for token in self.domain.stimulus_tokens.keys():
            path = os.path.join(IMAGE_DIR, f"{token}.png")
            if not os.path.exists(path):
                print(f"   Synthesizing {token}...")
                is_train = token in self.domain.train_stimuli
                
                # Deterministic seed per token
                digest = hashlib.sha256(f"{RNG_SEED}:{token}".encode("utf-8")).digest()
                seed = int.from_bytes(digest[:4], "big", signed=False)
                
                self.domain.asset_generator(token, is_train, seed, path)
                
            local_paths[token] = path
        return local_paths

    def generate_embedding_file(self, filename: str, ablate_bridge: bool = False) -> None:
        try:
            from PIL import Image  # type: ignore
        except ImportError:
            print("[Error] Missing dependency 'Pillow'. Run: pip install -r requirements.txt")
            sys.exit(1)

        """Generate a mixed embedding file:
        - sensation_* vectors from CLIP image features (projected down to GloVe dimensionality)
        - word/control vectors from GloVe if present, otherwise deterministic random

        Critically: NO CLIP text vectors are generated (so label 'water' is GloVe-only).
        """
        print("[Retina] Encoding images into vectors...")
        image_paths = self.fetch_images()

        glove_dim = _get_glove_dim(GLOVE_FILE)
        if glove_dim <= 0:
            raise RuntimeError(f"Invalid glove dim from {GLOVE_FILE}: {glove_dim}")

        # Create a deterministic random projection from CLIP-dim -> glove-dim.
        proj_in_dim = 512  # CLIP ViT-B/32 feature size
        gen = torch.Generator(device="cpu")
        gen.manual_seed(RNG_SEED)
        proj = torch.randn(glove_dim, proj_in_dim, generator=gen)

        wanted_glove = {w.lower() for w in self.domain.vector_terms if w.islower()}
        glove = _load_glove_vectors(GLOVE_FILE, wanted_glove)

        required_vocab = self.domain.all_stimuli + self.domain.vector_terms
        with open(filename, "w") as f:
            # 1) Image embeddings (opaque sensations)
            # First pass: compute CLIP vectors for all non-ablated images
            clip_vectors: dict[str, torch.Tensor] = {}
            
            for token, path in image_paths.items():
                if ablate_bridge and token in self.domain.test_stimuli:
                    continue # Handled separately
                
                try:
                    image = Image.open(path).convert("RGB")
                    inputs = self.processor(images=image, return_tensors="pt")
                    with torch.no_grad():
                        outputs = self.model.get_image_features(**inputs)
                        outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
                    
                    v512 = outputs[0].to(torch.float32)
                    if v512.numel() != proj_in_dim:
                        raise RuntimeError(f"Unexpected CLIP dim for {token}: {v512.numel()}")
                    clip_vectors[token] = v512
                except Exception as e:
                    raise RuntimeError(f"Blind spot for {token} ({path}): {e}")

            # Center the CLIP vectors to maximize contrast
            if clip_vectors:
                all_v = torch.stack(list(clip_vectors.values()))
                mean_v = torch.mean(all_v, dim=0)
                for t in clip_vectors:
                    clip_vectors[t] = clip_vectors[t] - mean_v

            # Second pass: Project and write (handling ablation)
            # Track train vectors for ablation basis
            train_vecs: List[List[float]] = []

            for token in image_paths: # Preserve order
                if ablate_bridge and token in self.domain.test_stimuli:
                    # Ablation: replace TEST stimulus vectors with deterministic random
                    # unit vectors (still hasUserVector=true because they are in-file).
                    # We additionally remove any component along train vectors to
                    # reliably drop cosine similarity.
                    r = _deterministic_random_unit_vector(glove_dim, f"ABLATE_{token}")
                    basis = list(train_vecs) # Use all train vectors seen so far (or all train vectors generally?)
                    # Original logic used w1_vec and n1_vec which are train stimuli.
                    
                    v_ablate = r
                    for b in basis:
                        dot = sum(a * bb for a, bb in zip(v_ablate, b))
                        v_ablate = [a - dot * bb for a, bb in zip(v_ablate, b)]
                    vec = _unit_normalize(v_ablate)
                    vec_str = " ".join([f"{x:.6f}" for x in vec])
                    f.write(f"{token} {vec_str}\n")
                    continue
                
                # Normal projection
                v512 = clip_vectors[token]
                v = torch.matmul(proj, v512)
                v = v / v.norm(p=2)
                vec = v.tolist()
                
                if token in self.domain.train_stimuli:
                    train_vecs.append(vec)
                
                vec_str = " ".join([f"{x:.6f}" for x in vec])
                f.write(f"{token} {vec_str}\n")

            # 2) Word/control embeddings (GloVe if available; otherwise deterministic random)
            for term in self.domain.vector_terms:
                key = term.lower() if term.islower() else None
                if key is not None and key in glove:
                    vec = glove[key]
                else:
                    vec = _deterministic_random_unit_vector(glove_dim, f"TERM_{term}")
                vec_str = " ".join([f"{x:.6f}" for x in vec])
                f.write(f"{term} {vec_str}\n")

        if not _embedding_file_has_all_vocab(filename, required_vocab):
            raise RuntimeError(f"Embedding file missing required vocab entries: {filename}")

        print(f"[Retina] Synaptic weights saved to {filename}")


class Environment:
    """The Laws of Physics. It echoes sound regardless of source."""

    def __init__(self, nars):
        self.nars = nars
        self.test_mode = False

    def set_test_mode(self, enabled: bool) -> None:
        self.test_mode = enabled

    def emit_confirm(self, *, cycles: int = 20) -> None:
        self.nars.input("<confirm --> [felt]>. :|:", cycles=cycles)

    def sound_event(self, content):
        # Hearing is always active (TRAIN and TEST), but never echoes
        # the semantic content token. Instead, it emits an opaque,
        # deterministic auditory token suitable for auditing.
        content = (content or "").strip()
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        auditory_token = f"utterance_{digest}"
        self.nars.input(f"<{auditory_token} --> [heard]>. :|:")


class Teacher:
    """The Other Agent. It models behavior."""

    def __init__(self, nars, env):
        self.nars = nars
        self.env = env

    def guide_hand(
        self,
        visual_signal: str,
        label: str,
        *,
        confirm: bool = True,
        drive_need_label: bool = True,
    ) -> None:
        print(f"\n[Teacher/TRAIN] Hand-under-hand: {visual_signal} -> force ^say({label})")
        self.nars.input(f"<{visual_signal} --> [seen]>. :|:", cycles=20)
        # Training-only: ground the symbol label to the sensation token.
        self.nars.input(f"<{visual_signal} --> {label}>. :|:", cycles=20)
        # Training-only: pair the label-free drive with the correct action so
        # the TEST-time goal window can elicit a discriminative response.
        if drive_need_label:
            self.nars.input("<need_label --> [felt]>! :|:", cycles=20)
        # NOTE: This is training-only scaffolding, not a test-time prompt.
        self.nars.input(f"<(*, {{SELF}}, {label}) --> ^say>! :|:", cycles=20)
        if confirm:
            self.env.emit_confirm(cycles=20)


class NarsOrganism:
    def __init__(
        self,
        jar_path: str,
        embedding_file: str,
        *,
        config: str | None = None,
        run_id: str | None = None,
        nal: str | None = None,
        cycles: str | int | None = None,
    ):
        self.process = None
        self.jar_path = jar_path
        self.embedding_file = embedding_file
        self.config = config
        self.run_id = run_id
        self.nal = nal
        self.cycles = cycles
        self.listening = True
        self.env = None  # Will attach later
        self.on_say = None
        self.on_input = None

    def start(self):
        if not os.path.exists(self.jar_path):
            print("[Error] JAR not found.")
            return False

        # The jar's Main-Class is org.opennars.main.Shell which expects
        # 4 positional args:
        #   narOrConfigFileOrNull idOrNull nalFileOrNull cyclesToRunOrNull
        # Always pass them explicitly to avoid interactive-mode ambiguity.
        arg_config = self.config if self.config is not None else "null"
        arg_id = self.run_id if self.run_id is not None else "null"
        arg_nal = self.nal if self.nal is not None else "null"
        arg_cycles = str(self.cycles) if self.cycles is not None else "null"

        cmd = [
            "java",
            "-Dopennars.vector=true",
            "-jar",
            self.jar_path,
            arg_config,
            arg_id,
            arg_nal,
            arg_cycles,
            "--glove",
            self.embedding_file,
        ]
        print(f"[NarsOrganism] Opening eyes... ({' '.join(cmd)})")

        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        threading.Thread(target=self._listen_stdout, daemon=True).start()
        threading.Thread(target=self._listen_stderr, daemon=True).start()
        return True

    def attach_env(self, env):
        self.env = env

    def input(self, text, cycles=50):
        if self.on_input is not None:
            try:
                self.on_input(text)
            except Exception:
                pass
        if self.process:
            try:
                self.process.stdin.write(text + "\n")
                if cycles > 0:
                    self.process.stdin.write(str(cycles) + "\n")
                self.process.stdin.flush()
            except BrokenPipeError:
                pass

    def _emit_speech(self, content: str) -> None:
        content = content.strip()
        if not content:
            return

        if self.on_say is not None:
            try:
                self.on_say(content)
            except Exception:
                pass

        if self.env:
            self.env.sound_event(content)

    def _listen_stdout(self):
        say_self_re = re.compile(r"\bOUT:\s*\(\^say\s*,\s*\{SELF\}\s*,\s*([^\)]+)\)")
        while self.listening and self.process and self.process.poll() is None:
            try:
                line = self.process.stdout.readline()
                if not line:
                    continue
                clean_line = line.strip()

                if "[OUTPUT]" in clean_line:
                    content = clean_line.split("[OUTPUT]")[1].strip()
                    self._emit_speech(content)
                else:
                    m = say_self_re.search(clean_line)
                    if m:
                        content = m.group(1).strip().rstrip(". ")
                        self._emit_speech(content)
            except Exception:
                pass

    def _listen_stderr(self):
        while self.listening and self.process and self.process.poll() is None:
            try:
                line = self.process.stderr.readline()
                if not line:
                    continue
            except Exception:
                pass

    def kill(self):
        self.listening = False
        if self.process:
            self.process.terminate()


class ExperimentHarness:
    def __init__(self, env: Environment, *, log_path: str):
        self.env = env
        self.mode = "TRAIN"
        self.current_stimulus: str | None = None
        self.test_started_at: float | None = None
        self.utterances: list[dict] = []
        self.inputs: list[dict] = []
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._log_f = open(log_path, "a", encoding="utf-8")
        self._jsonl({"kind": "meta", "t": time.time(), "log_path": log_path})

    def close(self) -> None:
        try:
            self._log_f.close()
        except Exception:
            pass

    def _jsonl(self, obj: dict) -> None:
        try:
            self._log_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            self._log_f.flush()
        except Exception:
            pass

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.env.set_test_mode(mode == "TEST")

    def set_stimulus(self, stim: str) -> None:
        self.current_stimulus = stim

    def on_input(self, text: str) -> None:
        evt = {
            "kind": "input",
            "t": time.time(),
            "mode": self.mode,
            "stimulus": self.current_stimulus,
            "text": text,
        }
        self.inputs.append(evt)
        self._jsonl(evt)

    def on_say(self, content: str) -> None:
        now = time.time()
        print(f"[UTTERANCE] mode={self.mode} stimulus={self.current_stimulus} say={content}")
        evt = {
            "kind": "utterance",
            "t": now,
            "mode": self.mode,
            "stimulus": self.current_stimulus,
            "content": content,
        }
        self.utterances.append(evt)
        self._jsonl(evt)

    def assert_no_label_leakage_in_test(self, label: str) -> None:
        needle = label.lower()
        bad = [
            e
            for e in self.inputs
            if e["mode"] == "TEST" and needle in (e.get("text") or "").lower()
        ]
        if bad:
            example = bad[0]["text"]
            raise RuntimeError(f"TEST leakage: saw '{label}' injected into NARS input: {example}")

    def assert_no_confirm_injection_in_test(self) -> None:
        bad = [
            e
            for e in self.inputs
            if e["mode"] == "TEST" and "<confirm --> [felt]>" in (e.get("text") or "")
        ]
        if bad:
            example = bad[0]["text"]
            raise RuntimeError(
                "TEST reward injection: saw '<confirm --> [felt]>' injected into NARS input: "
                + example
            )


def _summarize_run(name: str, harness: ExperimentHarness) -> dict:
    uttered = [u for u in harness.utterances if u["mode"] == "TEST"]
    dist = Counter([u["content"].strip() for u in uttered])
    first_utt_t = uttered[0]["t"] if uttered else None
    t0 = harness.test_started_at
    time_to_first = (first_utt_t - t0) if (first_utt_t is not None and t0 is not None) else None

    print(f"\n=== {name} ===")
    print(f"mode: TEST stimulus: {harness.current_stimulus}")
    print(f"utterance distribution: {dict(dist)}")
    print(f"time_to_first_utterance: {time_to_first}")

    return {
        "name": name,
        "dist": dist,
        "time_to_first": time_to_first,
    }


def _wait_for_quiescence(harness: ExperimentHarness, *, timeout_s: float, idle_s: float) -> None:
    """Wait until no new utterances arrive for a short idle window."""
    deadline = time.time() + timeout_s
    last_n = len(harness.utterances)
    last_change = time.time()
    while time.time() < deadline:
        n = len(harness.utterances)
        if n != last_n:
            last_n = n
            last_change = time.time()
        else:
            if time.time() - last_change >= idle_s:
                return
        time.sleep(0.05)


def _score_trial(
    harness: ExperimentHarness,
    *,
    window_start_t: float,
    window_end_t: float,
    label_a: str,
    label_b: str,
) -> dict:
    """Score utterances within the goal window.

    Categories:
      - said_<label_a>, said_<label_b>, other, none
    """
    uttered = [
        u
        for u in harness.utterances
        if u["mode"] == "TEST" and window_start_t <= u["t"] <= window_end_t
    ]
    first_t = uttered[0]["t"] if uttered else None
    time_to_first = (first_t - window_start_t) if first_t is not None else None

    counts = Counter([(u.get("content") or "").strip().lower() for u in uttered])
    a = int(counts.get(label_a.lower(), 0))
    b = int(counts.get(label_b.lower(), 0))

    category = "none"
    pred_label = "none"
    
    if not uttered:
        category = "none"
        pred_label = "none"
    else:
        # First utterance determines the label (single-response policy).
        first_c = uttered[0]["content"].strip().lower()
        if first_c == label_a.lower():
            category = f"said_{label_a}"
            pred_label = label_a
        elif first_c == label_b.lower():
            category = f"said_{label_b}"
            pred_label = label_b
        else:
            category = "other"
            pred_label = "other"

    first_content = uttered[0]["content"].strip() if uttered else None

    return {
        "category": category,
        "pred_label": pred_label,
        "time_to_first_utterance": time_to_first,
        "first_utterance": first_content,
        "label_counts": {label_a: a, label_b: b},
        "utterance_count": len(uttered),
    }


def _run_single_condition(
    name: str,
    domain: DomainSpec,
    *,
    jar_path: str,
    train: bool,
    ablate_bridge: bool,
    run_condition: str,
    run_stamp: str,
    config: str | None,
    nal: str | None,
    shell_cycles: str | int | None,
    train_trials: int = 10,
    test_trials: int = 8,
    settle_cycles: int = 200,
    goal_cycles: int = 800,
    settle_wait_s: float = 0.5,
    goal_window_wait_s: float = 2.5,
) -> dict:
    # Use domain-specific embedding file name
    base_emb = domain.embedding_file_base
    embedding_file = (
        base_emb.replace(".txt", "_ablate.txt") if ablate_bridge else base_emb
    )
    
    # Generate embeddings if needed
    required_vocab = domain.all_stimuli + domain.vector_terms
    if not _embedding_file_has_all_vocab(embedding_file, required_vocab):
        # Pass domain to Retina
        Retina(domain).generate_embedding_file(embedding_file, ablate_bridge=ablate_bridge)

    nars = NarsOrganism(
        jar_path,
        embedding_file,
        config=config,
        nal=nal,
        cycles=shell_cycles,
    )
    if not nars.start():
        raise RuntimeError("Could not start NARS")

    env = Environment(nars)
    nars.attach_env(env)
    teacher = Teacher(nars, env)

    log_path = os.path.join("runs", f"{run_stamp}_{run_condition}.jsonl")
    harness = ExperimentHarness(env, log_path=log_path)
    nars.on_say = harness.on_say
    nars.on_input = harness.on_input

    time.sleep(2.5)
    
    # Helper for generic logging
    label1, label2 = domain.labels
    s1_train, s2_train = domain.train_stimuli
    s1_test, s2_test = domain.test_stimuli
    
    try:
        # Training: learn a discriminative mapping.
        harness.set_mode("TRAIN")
        if train:
            print(f"\n--- TRAIN: discriminative grounding ({s1_train}->{label1}, {s2_train}->{label2}) ---")
            for _ in range(train_trials):
                harness.set_stimulus(s1_train)
                teacher.guide_hand(s1_train, label1, confirm=True)
                time.sleep(0.15)

                harness.set_stimulus(s2_train)
                teacher.guide_hand(s2_train, label2, confirm=True)
                time.sleep(0.3)
        else:
            print("\n--- TRAIN: skipped (baseline) ---")

        # Cosine diagnostics (baseline/ablate) for audit.
        vectors = _load_embedding_vectors(
            embedding_file,
            set(domain.all_stimuli),
        )
        
        # Compute cosines between train and test stimuli
        # A=Train1, B=Train2, C=Test1, D=Test2
        vecs = {
            "Tr1": vectors.get(s1_train) or [],
            "Tr2": vectors.get(s2_train) or [],
            "Te1": vectors.get(s1_test) or [],
            "Te2": vectors.get(s2_test) or [],
        }
        
        cosines = {
            "Te1_Tr1": _cosine(vecs["Te1"], vecs["Tr1"]), # Same class 1
            "Te1_Tr2": _cosine(vecs["Te1"], vecs["Tr2"]), # Cross class
            "Te2_Tr2": _cosine(vecs["Te2"], vecs["Tr2"]), # Same class 2
            "Te2_Tr1": _cosine(vecs["Te2"], vecs["Tr1"]), # Cross class
        }
        
        print("\n--- COSINE DIAGNOSTICS (glove-space) ---")
        print(f"cos(Te1,Tr1)={cosines['Te1_Tr1']:.4f}  cos(Te1,Tr2)={cosines['Te1_Tr2']:.4f}")
        print(f"cos(Te2,Tr2)={cosines['Te2_Tr2']:.4f}  cos(Te2,Tr1)={cosines['Te2_Tr1']:.4f}")

        # Test: bounded per-trial evaluation window.
        harness.set_mode("TEST")
        
        trial_rows: list[dict] = []
        test_cases = [
            ("Te1", s1_test, label1),
            ("Te2", s2_test, label2),
        ]
        
        # Global trial counter for this run
        trial_idx = 0
        
        for case_name, stimulus, target_label in test_cases:
            print(f"\n--- TEST CASE: {case_name} stimulus={stimulus} target={target_label} ---")
            for i in range(test_trials):
                trial_idx += 1
                harness.set_stimulus(stimulus)

                # Present stimulus + settle.
                nars.input(f"<{stimulus} --> [seen]>. :|:", cycles=settle_cycles)
                time.sleep(settle_wait_s)
                _wait_for_quiescence(harness, timeout_s=2.0, idle_s=0.35)

                # Goal window (label-free drive).
                window_start = time.time()
                nars.input("<need_label --> [felt]>! :|:", cycles=goal_cycles)
                # Re-present perception inside the scoring window (still label-free).
                nars.input(f"<{stimulus} --> [seen]>. :|:", cycles=20)
                
                # Single-response policy: wait for label1 or label2, then cut short.
                deadline = window_start + goal_window_wait_s
                while time.time() < deadline:
                    relevant = [u for u in harness.utterances if u["t"] >= window_start and u["mode"] == "TEST"]
                    found_target = False
                    for u in relevant:
                        c = (u.get("content") or "").strip().lower()
                        if c in (label1, label2):
                            found_target = True
                            break
                    if found_target:
                        break
                    time.sleep(0.1)
                
                window_end = time.time()

                score = _score_trial(
                    harness,
                    window_start_t=window_start,
                    window_end_t=window_end,
                    label_a=label1,
                    label_b=label2,
                )
                
                # Add cosine diagnostics to row for easy analysis
                row = {
                    "trial_id": trial_idx,
                    "stimulus": case_name, # Te1 or Te2
                    "target_label": target_label,
                    "pred_label": score["pred_label"],
                    "time_to_first_utt": score["time_to_first_utterance"],
                }
                # Add cosines
                row.update(cosines)
                trial_rows.append(row)

        # Demonstration/audit aid: print every TEST input as recorded in JSONL.
        try:
            print("\n--- TEST INPUTS (from JSONL log) ---")
            with open(log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if rec.get("kind") == "input" and rec.get("mode") == "TEST":
                        print(rec.get("text"))
        except Exception:
            pass

        # Hard acceptance check: no label token injected during TEST.
        harness.assert_no_label_leakage_in_test(label1)
        harness.assert_no_label_leakage_in_test(label2)
        harness.assert_no_confirm_injection_in_test()
        # Return a richer audit summary.
        return {
            "name": name,
            "domain": domain.name,
            "trials": trial_rows,
            "cosines": cosines
        }
    finally:
        try:
            harness.close()
        except Exception:
            pass
        nars.kill()


def _ensure_jar_or_build(jar_path: str) -> None:
    if os.path.exists(jar_path):
        return
    print(f"[Runner] Jar missing at {jar_path}. Building with Maven...")
    subprocess.run(
        ["mvn", "-Dmaven.javadoc.skip=true", "package"],
        check=True,
    )
    if not os.path.exists(jar_path):
        raise RuntimeError(f"Jar still missing after build: {jar_path}")


def _write_summary(run_condition: str, summary: dict, *, run_stamp: str) -> str:
    os.makedirs("runs", exist_ok=True)
    path = os.path.join("runs", f"{run_stamp}_{run_condition}.summary.json")
    payload = dict(summary)
    payload["run_condition"] = run_condition
    payload["t"] = time.time()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def _find_latest_summary(run_condition: str) -> tuple[str, dict] | None:
    if not os.path.isdir("runs"):
        return None
    candidates = [
        os.path.join("runs", fn)
        for fn in os.listdir("runs")
        if fn.endswith(f"_{run_condition}.summary.json")
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    path = candidates[0]
    try:
        with open(path, "r", encoding="utf-8") as f:
            return path, json.load(f)
    except Exception:
        return None


def _print_side_by_side(a_name: str, a: dict, b_name: str, b: dict) -> None:
    print("\n=== BASELINE vs ABLATE_BRIDGE ===")
    for name, d in ((a_name, a), (b_name, b)):
        confusion = d.get("confusion") or {}
        cosines = d.get("cosines") or {}
        print(f"{name}: confusion={confusion}")
        if cosines:
            print(
                f"{name}: cos(W2,W1)={cosines.get('cos_W2_W1')}, cos(W2,N1)={cosines.get('cos_W2_N1')}, cos(N2,N1)={cosines.get('cos_N2_N1')}, cos(N2,W1)={cosines.get('cos_N2_W1')}"
            )


def run_experiment_reps(
    domain: DomainSpec,
    *,
    run_condition: str,
    jar_path: str,
    ablate_bridge: bool,
    config: str | None,
    nal: str | None,
    shell_cycles: str | int | None,
    reps: int = 1,
    seed: int = 0,
) -> None:
    global RNG_SEED
    _ensure_jar_or_build(jar_path)

    run_stamp = time.strftime("%Y%m%d_%H%M%S")

    print("\n==============================")
    print(f"Project Broca: {domain.name} transfer")
    print("==============================")
    print(f"condition={run_condition} ablate_bridge={ablate_bridge} reps={reps} seed={seed}")

    summaries = []

    for i in range(reps):
        current_seed = seed + i
        RNG_SEED = current_seed
        print(f"\n--- Rep {i+1}/{reps} (seed={current_seed}) ---")

        # Force regeneration of procedural stimuli and embeddings for each seed.
        base_emb = domain.embedding_file_base
        embedding_file = (
            base_emb.replace(".txt", "_ablate.txt") if ablate_bridge else base_emb
        )
        if os.path.exists(embedding_file):
            os.remove(embedding_file)
        
        # Stimuli are checked/regen inside Retina/fetch_images but we want to force distinct procedural generation per seed.
        # But wait, Retina.fetch_images handles checking existence. 
        # If I want randomness per rep, I should delete the cached images.
        for token in domain.stimulus_tokens.keys():
            path = os.path.join(IMAGE_DIR, f"{token}.png")
            if os.path.exists(path):
                os.remove(path)

        summary = _run_single_condition(
            f"train->test ({domain.name})",
            domain,
            jar_path=jar_path,
            train=True,
            ablate_bridge=ablate_bridge,
            run_condition=run_condition,
            run_stamp=f"{run_stamp}_{domain.name}_rep{i}",
            config=config,
            nal=nal,
            shell_cycles=shell_cycles,
        )
        summaries.append(summary)
        _write_summary(f"{domain.name}_{run_condition}_rep{i}", summary, run_stamp=run_stamp)

    # Aggregate results
    all_trials = []
    
    for i, s in enumerate(summaries):
        for t in s["trials"]:
            t["rep"] = i
            t["seed"] = seed + i
            t["condition"] = run_condition
            all_trials.append(t)

    # Write trials.jsonl
    trials_path = os.path.join("runs", f"{run_stamp}_{domain.name}_{run_condition}.trials.jsonl")
    with open(trials_path, "w", encoding="utf-8") as f:
        for t in all_trials:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"[Runner] Trials log saved: {trials_path}")

    # Compute Confusion Matrix
    label1, label2 = domain.labels
    te1, te2 = domain.test_stimuli # Stimulus names e.g. sensation_W2, sensation_N2. 
    # But wait, logic below uses Te1/Te2 from test_cases names?
    # In _run_single_condition I used "Te1" and "Te2" as case names.
    
    agg_confusion = {
        "Te1": {f"said_{label1}": 0, f"said_{label2}": 0, "other": 0, "none": 0},
        "Te2": {f"said_{label1}": 0, f"said_{label2}": 0, "other": 0, "none": 0},
    }
    
    for t in all_trials:
        stim = t["stimulus"] # Te1 or Te2
        pred = t["pred_label"]
        key = "other"
        if pred == label1:
            key = f"said_{label1}"
        elif pred == label2:
            key = f"said_{label2}"
        elif pred == "none":
            key = "none"
            
        if stim in agg_confusion:
            agg_confusion[stim][key] += 1

    # Compute Accuracy (excluding none)
    def calc_acc(stim, target):
        relevant = [t for t in all_trials if t["stimulus"] == stim and t["pred_label"] != "none"]
        if not relevant:
            return 0.0
        correct = len([t for t in relevant if t["pred_label"] == target])
        return correct / len(relevant)

    acc_te1 = calc_acc("Te1", label1)
    acc_te2 = calc_acc("Te2", label2)
    balanced_acc = (acc_te1 + acc_te2) / 2.0

    # Answered Rate
    total = len(all_trials)
    answered = len([t for t in all_trials if t["pred_label"] != "none"])
    answered_rate = answered / total if total > 0 else 0.0
    none_rate = 1.0 - answered_rate
    
    # Label Bias (P(label1) - P(label2))
    l1_count = len([t for t in all_trials if t["pred_label"] == label1])
    l2_count = len([t for t in all_trials if t["pred_label"] == label2])
    label_bias = (l1_count - l2_count) / total if total > 0 else 0.0
    
    # Entropy
    from math import log2
    counts = Counter([t["pred_label"] for t in all_trials])
    probs = [c/total for c in counts.values()]
    entropy = -sum(p * log2(p) for p in probs if p > 0)
    
    # Latency stats
    latencies = [t["time_to_first_utt"] for t in all_trials if t["time_to_first_utt"] is not None]
    lat_mean = sum(latencies)/len(latencies) if latencies else 0.0
    lat_median = sorted(latencies)[len(latencies)//2] if latencies else 0.0

    # Average Cosines
    avg_cosines = {}
    for k in ["Te1_Tr1", "Te1_Tr2", "Te2_Tr2", "Te2_Tr1"]:
        vals = [t[k] for t in all_trials if k in t]
        avg_cosines[k] = sum(vals) / len(vals) if vals else 0.0
        
    same_class = (avg_cosines.get("Te1_Tr1", 0) + avg_cosines.get("Te2_Tr2", 0)) / 2
    cross_class = (avg_cosines.get("Te1_Tr2", 0) + avg_cosines.get("Te2_Tr1", 0)) / 2
    difficulty = cross_class - same_class # Expected to be negative? cross < same usually.
    # Wait, user said: "difficulty = mean(cross_class_cos) - mean(same_class_cos)"
    # If cross is huge (confusion) and same is small, difficulty is high.
    # If same is high (1.0) and cross is low (0.0), difficulty is -1.0 (Very Easy).
    
    agg_summary = {
        "name": f"Aggregate {domain.name} {run_condition} ({reps} reps)",
        "confusion": agg_confusion,
        "metrics": {
            "accuracy_excl_none": {
                "Te1": acc_te1,
                "Te2": acc_te2,
                "balanced": balanced_acc
            },
            "answered_rate": answered_rate,
            "none_rate": none_rate,
            "label_bias": label_bias,
            "entropy": entropy,
            "latency_mean": lat_mean,
            "latency_median": lat_median,
            "difficulty_score": difficulty
        },
        "cosines": avg_cosines,
        "reps": reps,
        "seed_start": seed,
        "trials_path": trials_path
    }

    summary_path = _write_summary(f"{domain.name}_{run_condition}", agg_summary, run_stamp=run_stamp)
    print(f"[Runner] Aggregate summary saved: {summary_path}")

    # If the other condition has been run previously, print + save a comparison.
    other = "ablate_bridge" if run_condition == "baseline" else "baseline"
    other_loaded = _find_latest_summary(f"{domain.name}_{other}")
    if other_loaded is None:
        return

    other_path, other_summary = other_loaded
    this_summary = agg_summary
    this_path = summary_path
    
    if run_condition == "baseline":
        _print_side_by_side("baseline", this_summary, "ablate_bridge", other_summary)
    else:
        _print_side_by_side("baseline", other_summary, "ablate_bridge", this_summary)

    compare_path = os.path.join("runs", f"{run_stamp}_{domain.name}_compare_baseline_vs_ablate_bridge.json")
    with open(compare_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "t": time.time(),
                "domain": domain.name,
                "baseline": other_summary if run_condition != "baseline" else this_summary,
                "ablate_bridge": this_summary if run_condition != "baseline" else other_summary,
                "baseline_summary_path": other_path if run_condition != "baseline" else this_path,
                "ablate_bridge_summary_path": this_path if run_condition != "baseline" else other_path,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[Runner] Comparison saved: {compare_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Project Broca: audit-safe grounding + transfer (ICLR Package)"
    )
    parser.add_argument(
        "--run",
        choices=["baseline", "ablate_bridge"],
        required=True,
        help="Operational runner entrypoint (audit-safe).",
    )
    parser.add_argument(
        "--domain",
        default="all",
        help="Domain to run: water_wind, shape, or all (default: all).",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=1,
        help="Number of repetitions (default: 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed (default: 0).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional NARS config XML (default: null).",
    )
    parser.add_argument(
        "--nal",
        default=None,
        help="Optional NAL file for rules (default: null).",
    )
    parser.add_argument(
        "--cycles",
        default=None,
        help="Optional cycles limit (default: null).",
    )
    parser.add_argument(
        "--jar",
        default=NARS_JAR,
        help=f"Path to OpenNARS jar (default: {NARS_JAR}).",
    )

    args = parser.parse_args()
    
    selected_domains = []
    if args.domain == "all":
        selected_domains = list(DOMAINS.values())
    elif args.domain in DOMAINS:
        selected_domains = [DOMAINS[args.domain]]
    else:
        print(f"[Error] Unknown domain: {args.domain}. Choices: {list(DOMAINS.keys())} or all")
        sys.exit(1)

    for domain in selected_domains:
        run_experiment_reps(
            domain,
            run_condition=args.run,
            jar_path=args.jar,
            ablate_bridge=(args.run == "ablate_bridge"),
            config=args.config,
            nal=args.nal,
            shell_cycles=args.cycles,
            reps=args.reps,
            seed=args.seed,
        )

