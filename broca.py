import subprocess
import threading
import time
import os
import sys

# Try imports to warn user if environment is wrong
try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    print("[Error] Missing dependencies. Run: pip install -r requirements.txt")
    sys.exit(1)

# --- Configuration ---
# Matches the pom.xml version
NARS_JAR = "target/opennars-3.0.4-SNAPSHOT.jar"
EMBEDDING_FILE = "clip_embeddings.txt"

# Initial vocabulary to ground the vision system
VOCAB_LIST = [
    "cat", "dog", "wolf", "tiger",
    "furry", "bark", "meow", "roar",
    "danger", "safe", "food", "run", "pet",
    "red", "blue", "green", "object"
]


class VisualCortex:
    """The 'Right Brain'. Generates Multimodal Embeddings using CLIP."""

    def __init__(self):
        print("[VisualCortex] Loading CLIP (ViT-B/32)...")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        print("[VisualCortex] Vision System Online.")

    def generate_embedding_file(self, filename, words):
        """Creates a GloVe-formatted text file using CLIP vectors."""
        print(f"[VisualCortex] Dreaming of {len(words)} concepts...")

        with open(filename, 'w') as f:
            # CLIP ViT-B/32 output is 512 dimensions
            inputs = self.processor(text=words, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = self.model.get_text_features(**inputs)
                # Normalize vectors (crucial for cosine similarity)
                outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)

            # Write format: "word 0.123 0.456 ..."
            for i, word in enumerate(words):
                vec = outputs[i].tolist()
                vec_str = " ".join([f"{v:.6f}" for v in vec])
                f.write(f"{word} {vec_str}\n")

        print(f"[VisualCortex] Memories stored in {filename}")


class NarsOrganism:
    """The 'Left Brain'. Runs the Logic Engine."""

    def __init__(self, jar_path, embedding_file):
        self.process = None
        self.jar_path = jar_path
        self.embedding_file = embedding_file
        self.listening = True

    def start(self):
        if not os.path.exists(self.jar_path):
            print(f"[Error] JAR not found: {self.jar_path}")
            print("Did you run 'mvn package -Dmaven.javadoc.skip=true'?")
            return False

        # -Dopennars.vector=true : Enables the hybrid architecture
        cmd = [
            "java",
            "-Dopennars.vector=true",
            "-jar", self.jar_path,
            "--glove", self.embedding_file
        ]
        print(f"[NarsOrganism] Waking up... ({' '.join(cmd)})")

        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1  # Line buffered
        )

        # Start ear (listener thread)
        threading.Thread(target=self._listen, daemon=True).start()
        return True

    def _listen(self):
        """Monitors NARS stdout for [OUTPUT] speech, errors, and internal logs."""
        while self.listening and self.process and self.process.poll() is None:
            try:
                line = self.process.stdout.readline()
                if not line:
                    continue

                clean_line = line.strip()

                # Filtering disabled: show internal logs too
                if "[OUTPUT]" in clean_line:
                    content = clean_line.split("[OUTPUT]")[1].strip()
                    print(f"\n[NARS] SAYS: '{content}'")
                elif "Exception" in clean_line:
                    print(f"\n[NARS] ERROR: {clean_line}")
                else:
                    print(f"[NARS] {clean_line}")
            except Exception as e:
                print(f"[Listener Error] {e}")

    def speak(self, text, cycles=100):
        """Sends input AND drives the clock (cycles)."""
        if self.process:
            print(f"Teacher: {text}")
            try:
                # 1) Send the thought
                self.process.stdin.write(text + "\n")
                # 2) Drive the clock so shell mode actually processes it
                if cycles and cycles > 0:
                    self.process.stdin.write(str(cycles) + "\n")
                self.process.stdin.flush()
            except BrokenPipeError:
                print("[Error] NARS process died.")

    def kill(self):
        self.listening = False
        if self.process:
            self.process.terminate()


# --- The Curriculum ---
def run_curriculum():
    # 1. Initialize Vision (if memories don't exist)
    if not os.path.exists(EMBEDDING_FILE):
        print("Initializing Visual Cortex...")
        try:
            vision = VisualCortex()
            vision.generate_embedding_file(EMBEDDING_FILE, VOCAB_LIST)
        except Exception as e:
            print(f"[Error] Failed to load Vision: {e}")
            return

    # 2. Wake up NARS
    nars = NarsOrganism(NARS_JAR, EMBEDDING_FILE)
    if not nars.start():
        return

    # Give it a moment to load vectors
    time.sleep(3)

    try:
        # 3. Lesson 1: Self-Awareness (Testing ^say)
        print("\n--- Lesson 1: The Voice ---")
        nars.speak("<(*, {SELF}, hello_world) --> ^say>! :|:", cycles=100)
        time.sleep(2)

        # 4. Lesson 2: Visual Grounding (The CLIP Test)
        print("\n--- Lesson 2: Associations ---")

        # We tell it a Wolf is aggressive.
        nars.speak("<wolf --> [aggressive]>.", cycles=10)

        # We ask if a Dog is aggressive.
        # NARS must use CLIP similarity (Wolf~Dog) to answer.
        nars.speak("<dog --> [aggressive]>?", cycles=500)

        print("\n[System] Interactive Mode (Type 'exit' to quit):")
        while True:
            try:
                user_input = input("> ")
                if user_input.lower() == "exit":
                    break
                nars.speak(user_input, cycles=100)
            except EOFError:
                break

    except KeyboardInterrupt:
        print("\n[System] Shutting down.")
    finally:
        nars.kill()


if __name__ == "__main__":
    run_curriculum()
