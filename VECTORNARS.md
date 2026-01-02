Here is a summary document tailored for an OpenNARS Pull Request description, formatted as a mini technical report.

---

# PR: VectorNARS – Neuro-Symbolic Integration with Vector Symbolic Architectures (VSA)

## 1. Overview

This Pull Request introduces **VectorNARS**, a hybrid architectural enhancement that integrates high-dimensional Vector Symbolic Architectures (VSA/Hypercomputing) directly into the OpenNARS reasoning loop.

By grounding symbolic concepts in continuous semantic vector spaces (e.g., GloVe embeddings), the system gains "subconscious" capabilities:

* **Semantic Association:** Concepts are now related by their vector proximity, not just by explicit Narsese logic links.
* **Zero-Shot Analogies:** The system can infer properties about new terms if they are semantically similar to known terms (e.g., inferring `<kitten --> furry>` from `<cat --> furry>` without explicit definitions).
* **Attention Guidance:** The inference control loop now prioritizes concepts that are semantically relevant to the current context, even if their activation levels (priority) are lower.

## 2. Key Architectural Changes

### A. The Hypervector Engine (`Hypervector.java`, `ProjectionMatrix.java`)

* **Representation:** Implements 1024-bit binary hypervectors packed into `long` arrays for high-performance bitwise operations.
* **Projection:** Includes a `ProjectionMatrix` using sparse ternary projections (Achlioptas distribution) to deterministically project dense float vectors (like GloVe 300d) into the binary hypervector space.
* **Operations:** Supports Hamming distance similarity calculations and Hebbian "nudging" (moving vectors closer based on co-occurrence/context).

### B. High-Performance Embedding Loader (`GloVeLoader.java`)

* **Parallel Processing:** Utilizes `Stream.parallel()` to parse and project large GloVe text files across all CPU cores.
* **Binary Caching:** Automatically serializes projected hypervectors to a local `.bin` file. Subsequent loads are near-instantaneous (milliseconds vs. seconds/minutes).
* **CLI Integration:** Added `--glove <path>` argument to `Shell.java` to trigger vector loading at startup.

### C. Vector-Guided Inference (`VectorInference.java`, `GeneralInferenceControl.java`)

* **Concept Selection:** Modified `Bag.takeWithContext` to score candidates based on a weighted mix of **Priority** (Urgency) and **Vector Similarity** (Relevance to the current thought).
* **Bridge Injection:** If the system encounters two concepts with high vector similarity (threshold `> 0.8`), it automatically injects a `Sim` statement (e.g., `<cat <-> kitten>`) into the inference loop. This allows standard NAL rules to perform the reasoning transfer.
* **Integrity Checks:** Added logic to prevent "echo chambers" (re-injecting static vector similarities as new evidence) and ensure thread safety during bridge injection.

### D. Testing & Verification

* **`VectorNarsTest.java`:** Unit tests verifying that vector similarity can override standard priority-based selection when context is active.
* **`SynonymTest.java` & `RealSynonymTest.java`:** Integration tests demonstrating the full loop: loading vectors, teaching a fact about one term, and successfully querying a synonym term.

## 3. Configuration & Usage

The feature is controlled via system properties and CLI args:

* **Enable:** `java -jar opennars.jar --glove /path/to/glove.6B.50d.txt`
* **Properties:**
* `opennars.vector`: Master switch.
* `opennars.vectorContext`: Enables vector-biased concept selection.
* `opennars.vectorBridgeInjection`: Enables automatic injection of similarity links.



## 4. Current Limitations & Next Steps

### Missing Pieces

* **Contrastive Learning:** Currently, the system only "nudges" vectors closer (Hebbian attraction). It lacks a "repel" mechanism for negative evidence, which is necessary to prevent "semantic collapse" (where all vectors eventually drift towards a single average) in long-running sessions.
* **Persistence:** Vector updates (learning) happen in RAM but are not saved back to disk on shutdown. The system currently has "amnesia" regarding semantic shifts between runs.
* **Advanced Composition:** We are treating terms as atomic vectors. We do not yet fully exploit VSA binding (XOR) to represent complex Narsese compound terms (e.g., `(*, cat, fish)`) structurally in the vector space.

### Next Steps

1. Implement **Persistence Hooks** to save modified vectors to `glove.learned.bin`.
2. Add **Negative Feedback** loops to push vectors apart when predictions fail.
3. Explore **Holographic Reduced Representations (HRR)** for mapping Narsese compound term structures directly to vector operations.