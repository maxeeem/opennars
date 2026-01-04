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
        try:
            import requests  # type: ignore
        except ImportError:
            print("[Error] Missing dependency 'requests'. Run: pip install -r requirements.txt")
            sys.exit(1)

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

    def guide_hand(self, visual_signal: str, label: str, confirm: bool = True) -> None:
        print(f"\n[Teacher/TRAIN] Hand-under-hand: {visual_signal} -> force ^say({label})")
        self.nars.input(f"<{visual_signal} --> [seen]>. :|:")
        # NOTE: This is training-only scaffolding, not a test-time prompt.
        self.nars.input(f"<(*, {{SELF}}, {label}) --> ^say>! :|:")
        if confirm:
            self.env.emit_confirm()


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


def _run_single_condition(
    name: str,
    *,
    jar_path: str,
    train: bool,
    test_stimulus: str,
    ablate_bridge: bool,
    run_condition: str,
    config: str | None,
    nal: str | None,
    shell_cycles: str | int | None,
    train_trials: int = 8,
    test_wait_s: float = 8.0,
) -> dict:
    embedding_file = (
        EMBEDDING_FILE.replace(".txt", "_ablate.txt") if ablate_bridge else EMBEDDING_FILE
    )
    required_vocab = list(VISUAL_STIMULI.keys()) + VECTOR_TERMS
    if not _embedding_file_has_all_vocab(embedding_file, required_vocab):
        Retina().generate_embedding_file(embedding_file, ablate_bridge=ablate_bridge)

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

    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join("runs", f"{ts}_{run_condition}.jsonl")
    harness = ExperimentHarness(env, log_path=log_path)
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
        harness.assert_no_confirm_injection_in_test()
        return _summarize_run(name, harness)
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


def _write_summary(run_condition: str, summary: dict) -> str:
    os.makedirs("runs", exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join("runs", f"{ts}_{run_condition}.summary.json")
    payload = dict(summary)
    payload["run_condition"] = run_condition
    payload["t"] = time.time()
    payload["dist"] = dict(payload.get("dist", {}))
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
    def _water_count(d: dict) -> int:
        dist = d.get("dist") or {}
        try:
            return int(dist.get("water", 0))
        except Exception:
            return 0

    print("\n=== BASELINE vs ABLATE_BRIDGE ===")
    print(
        f"{a_name}: water_utterances={_water_count(a)}, time_to_first_utterance={a.get('time_to_first')}"
    )
    print(
        f"{b_name}: water_utterances={_water_count(b)}, time_to_first_utterance={b.get('time_to_first')}"
    )


def run_single_water_protocol(
    *,
    run_condition: str,
    jar_path: str,
    ablate_bridge: bool,
    config: str | None,
    nal: str | None,
    shell_cycles: str | int | None,
) -> None:
    _ensure_jar_or_build(jar_path)

    print("\n==============================")
    print("Project Broca: WATER transfer")
    print("==============================")
    print(f"condition={run_condition} ablate_bridge={ablate_bridge}")

    summary = _run_single_condition(
        "train(W1)->test(W2)",
        jar_path=jar_path,
        train=True,
        test_stimulus="sensation_W2",
        ablate_bridge=ablate_bridge,
        run_condition=run_condition,
        config=config,
        nal=nal,
        shell_cycles=shell_cycles,
    )
    summary_path = _write_summary(run_condition, summary)
    print(f"[Runner] Summary saved: {summary_path}")
    print(f"[Runner] JSONL logs saved under: runs/*_{run_condition}.jsonl")

    # If the other condition has been run previously, print + save a comparison.
    other = "ablate_bridge" if run_condition == "baseline" else "baseline"
    other_loaded = _find_latest_summary(other)
    if other_loaded is None:
        return

    other_path, other_summary = other_loaded
    this_loaded = _find_latest_summary(run_condition)
    if this_loaded is None:
        return
    this_path, this_summary = this_loaded

    if run_condition == "baseline":
        _print_side_by_side("baseline", this_summary, "ablate_bridge", other_summary)
    else:
        _print_side_by_side("baseline", other_summary, "ablate_bridge", this_summary)

    ts = time.strftime("%Y%m%d_%H%M%S")
    compare_path = os.path.join("runs", f"{ts}_compare_baseline_vs_ablate_bridge.json")
    with open(compare_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "t": time.time(),
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
        "--run",
        choices=["baseline", "ablate_bridge"],
        help="Operational runner entrypoint (audit-safe).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional NARS config XML for Shell positional arg #1 (default: null).",
    )
    parser.add_argument(
        "--nal",
        default=None,
        help="Optional NAL file for Shell positional arg #3 (default: null).",
    )
    parser.add_argument(
        "--cycles",
        default=None,
        help="Optional cyclesToRunOrNull for Shell positional arg #4 (default: null).",
    )
    parser.add_argument(
        "--jar",
        default=NARS_JAR,
        help=f"Path to OpenNARS jar (default: {NARS_JAR}).",
    )
    # Back-compat: keep old flag for interactive experimentation.
    parser.add_argument(
        "--ablate-bridge",
        action="store_true",
        help="(Legacy) Disable bridge in the older multi-condition demo.",
    )

    args = parser.parse_args()
    if args.run is not None:
        run_single_water_protocol(
            run_condition=args.run,
            jar_path=args.jar,
            ablate_bridge=(args.run == "ablate_bridge"),
            config=args.config,
            nal=args.nal,
            shell_cycles=args.cycles,
        )
    else:
        run_water_protocol(ablate_bridge=bool(args.ablate_bridge))
