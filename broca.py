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
import requests
from PIL import Image

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

# Map abstract sensations to real image URLs.
# These token names are what NARS sees (opaque signals), not the labels.
VISUAL_STIMULI = {
    # Training water-flow stimulus
    "sensation_W1": "https://commons.wikimedia.org/wiki/Special:FilePath/Water_tap_running.jpg?width=240",
    # Test water-flow stimulus (novel variant)
    "sensation_W2": "https://commons.wikimedia.org/wiki/Special:FilePath/Waterfall_in_Iceland.jpg?width=240",
    # Distractor non-water stimulus
    "sensation_N1": "https://commons.wikimedia.org/wiki/Special:FilePath/Electric_fan.jpg?width=240",
}

# Harness-side ground-truth (NEVER injected during TEST).
GROUND_TRUTH: dict[str, str | None] = {
    "sensation_W2": "water",
    "sensation_N1": None,
}

# Terms we want vectors for (labels + control signals). In this experiment:
# - sensation_* vectors come from CLIP images (projected to glove dims)
# - label vectors come from GloVe only (no CLIP text path)
VECTOR_TERMS = [
    "water",
    "confirm",
    "need_label",
    "seen",
    "heard",
    "say",
    "babble",
    "SELF",
    "TEACHER",
]


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


class Retina:
    def __init__(self):
        print("[Retina] Loading CLIP (ViT-B/32) image encoder...")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        os.makedirs(IMAGE_DIR, exist_ok=True)

    def fetch_images(self) -> dict[str, str]:
        print("[Retina] Gathering light from the web...")
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://upload.wikimedia.org/",
        }
        local_paths: dict[str, str] = {}
        for token, url in VISUAL_STIMULI.items():
            path = os.path.join(IMAGE_DIR, f"{token}.jpg")
            if not os.path.exists(path):
                print(f"   Downloading {token}...")
                try:
                    last_exc: Exception | None = None
                    for attempt in range(3):
                        try:
                            resp = requests.get(url, timeout=30, headers=headers)
                            resp.raise_for_status()
                            break
                        except Exception as e:
                            last_exc = e
                            time.sleep(1.5 * (attempt + 1))
                    if last_exc is not None and ("resp" not in locals() or not getattr(resp, "ok", False)):
                        raise last_exc
                    with open(path, "wb") as f:
                        f.write(resp.content)
                except Exception as e:
                    raise RuntimeError(f"Could not fetch {token} from {url}: {e}")
            local_paths[token] = path
        return local_paths

    def generate_embedding_file(self, filename: str, ablate_bridge: bool = False) -> None:
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

        wanted_glove = {w.lower() for w in VECTOR_TERMS if w.islower()}
        glove = _load_glove_vectors(GLOVE_FILE, wanted_glove)

        required_vocab = list(VISUAL_STIMULI.keys()) + VECTOR_TERMS
        with open(filename, "w") as f:
            # 1) Image embeddings (opaque sensations)
            for token, path in image_paths.items():
                try:
                    if ablate_bridge and token == "sensation_W2":
                        # Disable similarity transfer by using an unrelated vector.
                        vec = _deterministic_random_unit_vector(glove_dim, "ABLATE_W2")
                        vec_str = " ".join([f"{x:.6f}" for x in vec])
                        f.write(f"{token} {vec_str}\n")
                        continue

                    image = Image.open(path).convert("RGB")
                    inputs = self.processor(images=image, return_tensors="pt")
                    with torch.no_grad():
                        outputs = self.model.get_image_features(**inputs)
                        outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)

                    v512 = outputs[0].to(torch.float32)
                    if v512.numel() != proj_in_dim:
                        raise RuntimeError(f"Unexpected CLIP dim for {token}: {v512.numel()}")

                    v = torch.matmul(proj, v512)
                    v = v / v.norm(p=2)

                    vec = v.tolist()
                    vec_str = " ".join([f"{x:.6f}" for x in vec])
                    f.write(f"{token} {vec_str}\n")
                except Exception as e:
                    raise RuntimeError(f"Blind spot for {token} ({path}): {e}")

            # 2) Word/control embeddings (GloVe if available; otherwise deterministic random)
            for term in VECTOR_TERMS:
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

    def emit_confirm(self) -> None:
        self.nars.input("<confirm --> [felt]>. :|:")

    def sound_event(self, content):
        # Training: allow speech to be echoed into hearing.
        # Test: strictly disable echo to avoid label leakage.
        if self.test_mode:
            return
        self.nars.input(f"<{content} --> [heard]>. :|:")


class Teacher:
    """The Other Agent. It models behavior."""

    def __init__(self, nars, env):
        self.nars = nars
        self.env = env

    def guide_hand(self, visual_signal: str, label: str, confirm: bool = True) -> None:
        print(f"\n[Teacher/TRAIN] Hand-under-hand: {visual_signal} -> force ^say({label})")
        self.nars.input(f"<{visual_signal} --> [seen]>. :|:")
        # NOTE: This is training-only scaffolding, not a test-time prompt.
        self.nars.input(f"<(*, {{SELF}}, {label}) --> ^say>! :|:")
        if confirm:
            self.env.emit_confirm()


class NarsOrganism:
    def __init__(self, jar_path, embedding_file):
        self.process = None
        self.jar_path = jar_path
        self.embedding_file = embedding_file
        self.listening = True
        self.env = None  # Will attach later
        self.on_say = None
        self.on_input = None

    def start(self):
        if not os.path.exists(self.jar_path):
            print("[Error] JAR not found.")
            return False

        cmd = ["java", "-Dopennars.vector=true", "-jar", self.jar_path, "--glove", self.embedding_file]
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
    def __init__(self, env: Environment):
        self.env = env
        self.mode = "TRAIN"
        self.current_stimulus: str | None = None
        self.test_started_at: float | None = None
        self.confirmed_at: float | None = None
        self.utterances: list[dict] = []
        self.inputs: list[dict] = []

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.env.set_test_mode(mode == "TEST")

    def set_stimulus(self, stim: str) -> None:
        self.current_stimulus = stim

    def on_input(self, text: str) -> None:
        self.inputs.append(
            {
                "t": time.time(),
                "mode": self.mode,
                "stimulus": self.current_stimulus,
                "text": text,
            }
        )

    def on_say(self, content: str) -> None:
        now = time.time()
        print(f"[UTTERANCE] mode={self.mode} stimulus={self.current_stimulus} say={content}")
        self.utterances.append(
            {
                "t": now,
                "mode": self.mode,
                "stimulus": self.current_stimulus,
                "content": content,
            }
        )

        if self.mode != "TEST":
            return
        if self.confirmed_at is not None:
            return
        if self.current_stimulus is None:
            return

        expected = GROUND_TRUTH.get(self.current_stimulus)
        if expected is not None and content.strip() == expected:
            self.env.emit_confirm()
            self.confirmed_at = time.time()
            print(
                f"[CONFIRM] mode=TEST stimulus={self.current_stimulus} expected={expected}"
            )

    def assert_no_label_leakage_in_test(self, label: str) -> None:
        bad = [e for e in self.inputs if e["mode"] == "TEST" and label in e["text"]]
        if bad:
            example = bad[0]["text"]
            raise RuntimeError(f"TEST leakage: saw '{label}' injected into NARS input: {example}")


def _summarize_run(name: str, harness: ExperimentHarness) -> dict:
    uttered = [u for u in harness.utterances if u["mode"] == "TEST"]
    dist = Counter([u["content"].strip() for u in uttered])
    first_utt_t = uttered[0]["t"] if uttered else None
    t0 = harness.test_started_at
    time_to_first = (first_utt_t - t0) if (first_utt_t is not None and t0 is not None) else None
    time_to_confirm = (harness.confirmed_at - t0) if (harness.confirmed_at is not None and t0 is not None) else None

    print(f"\n=== {name} ===")
    print(f"mode: TEST stimulus: {harness.current_stimulus}")
    print(f"utterance distribution: {dict(dist)}")
    print(f"confirmed: {harness.confirmed_at is not None}")
    print(f"time_to_first_utterance: {time_to_first}")
    print(f"time_to_confirm: {time_to_confirm}")

    return {
        "name": name,
        "dist": dist,
        "confirmed": harness.confirmed_at is not None,
        "time_to_first": time_to_first,
        "time_to_confirm": time_to_confirm,
    }


def _run_single_condition(
    name: str,
    *,
    train: bool,
    test_stimulus: str,
    ablate_bridge: bool,
    train_trials: int = 8,
    test_wait_s: float = 8.0,
) -> dict:
    embedding_file = (
        EMBEDDING_FILE.replace(".txt", "_ablate.txt") if ablate_bridge else EMBEDDING_FILE
    )
    required_vocab = list(VISUAL_STIMULI.keys()) + VECTOR_TERMS
    if not _embedding_file_has_all_vocab(embedding_file, required_vocab):
        Retina().generate_embedding_file(embedding_file, ablate_bridge=ablate_bridge)

    nars = NarsOrganism(NARS_JAR, embedding_file)
    if not nars.start():
        raise RuntimeError("Could not start NARS")

    env = Environment(nars)
    nars.attach_env(env)
    teacher = Teacher(nars, env)

    harness = ExperimentHarness(env)
    nars.on_say = harness.on_say
    nars.on_input = harness.on_input

    time.sleep(2.5)
    try:
        # Training episode (hand-under-hand) on W1.
        harness.set_mode("TRAIN")
        harness.set_stimulus("sensation_W1")
        if train:
            print("\n--- TRAIN: WATER grounding (W1) ---")
            for _ in range(train_trials):
                nars.input("<sensation_W1 --> [seen]>. :|:", cycles=30)
                teacher.guide_hand("sensation_W1", "water", confirm=True)
                time.sleep(0.3)
        else:
            print("\n--- TRAIN: skipped (baseline) ---")

        # Test episode (label-free drive) on chosen stimulus.
        harness.set_mode("TEST")
        harness.set_stimulus(test_stimulus)
        harness.test_started_at = time.time()

        print(f"\n--- TEST: stimulus={test_stimulus}, label-free drive ---")
        nars.input(f"<{test_stimulus} --> [seen]>. :|:", cycles=50)
        nars.input("<need_label --> [felt]>! :|:", cycles=600)

        time.sleep(test_wait_s)

        # Hard acceptance check: no label token injected during TEST.
        harness.assert_no_label_leakage_in_test("water")
        return _summarize_run(name, harness)
    finally:
        nars.kill()


def run_water_protocol(ablate_bridge: bool = False) -> None:
    print("\n==============================")
    print("WATER protocol (Keller-style)")
    print("==============================")
    print(f"ablate_bridge={ablate_bridge}")

    results = []

    # A) Train → Test (main condition)
    results.append(
        _run_single_condition(
            "A) train(W1)->test(W2)",
            train=True,
            test_stimulus="sensation_W2",
            ablate_bridge=ablate_bridge,
        )
    )

    # B) No-training baseline
    results.append(
        _run_single_condition(
            "B) no-train->test(W2)",
            train=False,
            test_stimulus="sensation_W2",
            ablate_bridge=ablate_bridge,
        )
    )

    # C) Distractor baseline (after training)
    results.append(
        _run_single_condition(
            "C) train(W1)->test(N1)",
            train=True,
            test_stimulus="sensation_N1",
            ablate_bridge=ablate_bridge,
        )
    )

    print("\n=== SUMMARY ===")
    for r in results:
        water_count = int(r["dist"].get("water", 0))
        print(
            f"{r['name']}: water_utterances={water_count}, confirmed={r['confirmed']}, time_to_confirm={r['time_to_confirm']}"
        )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Keller-style WATER grounding + transfer demo")
    parser.add_argument(
        "--ablate-bridge",
        action="store_true",
        help="Disable perceptual similarity bridge by replacing W2 vector with an unrelated vector.",
    )
    args = parser.parse_args()
    run_water_protocol(ablate_bridge=bool(args.ablate_bridge))
