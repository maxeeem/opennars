import subprocess
import threading
import time
import os
import sys
import signal
import re
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
EMBEDDING_FILE = "clip_embeddings.txt"
IMAGE_DIR = "retina_cache"

# Map abstract sensations to real image URLs.
# These token names are what NARS sees (opaque signals), not the labels.
VISUAL_STIMULI = {
    "sensation_A": "https://commons.wikimedia.org/wiki/Special:FilePath/Red_Apple.jpg?width=240",
    "sensation_B": "https://commons.wikimedia.org/wiki/Special:FilePath/Hamburger_(black_bg).jpg?width=240",
    "sensation_C": "https://commons.wikimedia.org/wiki/Special:FilePath/Chair.jpg?width=240",
    "sensation_D": "https://commons.wikimedia.org/wiki/Special:FilePath/Eq_it-na_pizza-margherita_sep2005_sml.jpg?width=240",
}

# Non-visual concepts still need text embeddings.
TEXT_CONCEPTS = ["food", "babble", "seen", "heard", "say", "SELF", "TEACHER"]


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

    def generate_embedding_file(self, filename: str) -> None:
        print("[Retina] Encoding images into vectors...")
        image_paths = self.fetch_images()

        required_vocab = list(VISUAL_STIMULI.keys()) + TEXT_CONCEPTS
        with open(filename, "w") as f:
            # 1) Image embeddings (opaque sensations)
            for token, path in image_paths.items():
                try:
                    image = Image.open(path).convert("RGB")
                    inputs = self.processor(images=image, return_tensors="pt")
                    with torch.no_grad():
                        outputs = self.model.get_image_features(**inputs)
                        outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)

                    vec = outputs[0].tolist()
                    vec_str = " ".join([f"{v:.6f}" for v in vec])
                    f.write(f"{token} {vec_str}\n")
                except Exception as e:
                    raise RuntimeError(f"Blind spot for {token} ({path}): {e}")

            # 2) Text embeddings (labels, predicates, actors)
            inputs = self.processor(text=TEXT_CONCEPTS, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = self.model.get_text_features(**inputs)
                outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)

            for i, word in enumerate(TEXT_CONCEPTS):
                vec = outputs[i].tolist()
                vec_str = " ".join([f"{v:.6f}" for v in vec])
                f.write(f"{word} {vec_str}\n")

        if not _embedding_file_has_all_vocab(filename, required_vocab):
            raise RuntimeError(f"Embedding file missing required vocab entries: {filename}")

        print(f"[Retina] Synaptic weights saved to {filename}")


class Environment:
    """The Laws of Physics. It echoes sound regardless of source."""

    def __init__(self, nars):
        self.nars = nars

    def sound_event(self, content):
        # Physics: Sound travels through the air
        # Result: NARS hears it.
        self.nars.input(f"<{content} --> [heard]>. :|:")


class Teacher:
    """The Other Agent. It models behavior."""

    def __init__(self, nars, env):
        self.nars = nars
        self.env = env

    def model_behavior(self, visual_signal, label):
        print(f"\n[Teacher] Models: {visual_signal} -> '{label}'")
        self.nars.input(f"<{visual_signal} --> [seen]>.")
        self.nars.input(f"<(*, {{TEACHER}}, {label}) --> ^say>. :|:")
        self.env.sound_event(label)

    def guide_hand(self, visual_signal, label):
        print(f"\n[Teacher] Guides Hand: {visual_signal} -> Force NARS say '{label}'")
        self.nars.input(f"<{visual_signal} --> [seen]>.")
        self.nars.input(f"<(*, {{SELF}}, {label}) --> ^say>! :|:")


class NarsOrganism:
    def __init__(self, jar_path, embedding_file):
        self.process = None
        self.jar_path = jar_path
        self.embedding_file = embedding_file
        self.listening = True
        self.env = None  # Will attach later

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

        # Ignore non-grounded variables like "#1" or "$1".
        # For this curriculum we only treat lowercase word tokens as spoken content.
        if not re.fullmatch(r"[a-z][a-z0-9_]*", content):
            return

        print(f"📢 NARS SPEAKS: '{content}'")

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
                    elif "OUT:" in clean_line:
                        if "food" in clean_line or "TEACHER" in clean_line:
                            print(f"🧠 {clean_line}")
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


def run_curriculum():
    required_vocab = list(VISUAL_STIMULI.keys()) + TEXT_CONCEPTS
    if not _embedding_file_has_all_vocab(EMBEDDING_FILE, required_vocab):
        Retina().generate_embedding_file(EMBEDDING_FILE)

    # 1. Initialize Actors
    nars = NarsOrganism(NARS_JAR, EMBEDDING_FILE)
    if not nars.start():
        return

    env = Environment(nars)
    nars.attach_env(env)
    teacher = Teacher(nars, env)

    time.sleep(3)

    try:
        print("\n--- Phase 1: Agency ---")
        print("NARS discovers its own voice.")
        nars.input("<(*, {SELF}, babble) --> ^say>! :|:", cycles=20)
        time.sleep(1)

        print("\n--- Phase 2: Training (Apple & Burger) ---")
        print("Train on opaque signals only (no 'apple'/'burger' labels).")
        teacher.guide_hand("sensation_A", "food")
        time.sleep(0.5)
        teacher.guide_hand("sensation_B", "food")

        print("\n[Control] Untrained object (Chair).")
        nars.input("<sensation_C --> [seen]>.", cycles=10)

        print("\n--- Phase 3: The Retina Test (Pizza) ---")
        print("Stimulus: sensation_D (real pixels of a pizza).")
        print("Teacher is Silent.")
        print("Drive: Hunger (<food --> [heard]>!).")

        nars.input("<sensation_D --> [seen]>.", cycles=50)
        nars.input("<food --> [heard]>! :|:", cycles=500)
        time.sleep(6)

        print("\n[System] Interactive Mode. Type 'exit' to quit.")
        while True:
            try:
                user_input = input("> ")
                if user_input.lower() == "exit":
                    break
                nars.input(user_input, cycles=100)
            except EOFError:
                break

    except KeyboardInterrupt:
        print("\n[System] Shutting down.")
    finally:
        nars.kill()


if __name__ == "__main__":
    run_curriculum()
