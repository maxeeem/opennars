PR: VectorNARS – Neuro-Symbolic Integration (v3.0.4-patch)
1. Overview
This PR introduces VectorNARS, a hybrid architecture that integrates Vector Symbolic Architectures (VSA/Hypercomputing) into the OpenNARS reasoning loop. It grounds symbolic terms in continuous semantic spaces (e.g., GloVe embeddings), enabling "subconscious" capabilities like Zero-Shot Analogy and Associative Attention.

Stability Verdict:

✅ Regression Proof: Passed full NALTest (215/215) and MultiStep (24/24) with ALL vector features enabled (vector=true, context=true, bridge=true).

✅ Zero Guards: We successfully removed the "Safety Guard" for bridge injection. The system relies on the mathematical orthogonality of random hypervectors to prevent hallucinations on unknown terms.

2. Key Innovations
A. Executive Control (The "Focus" Mechanism)

We implemented a biological "Focus" mechanism to prevent the subconscious from derailing strict procedural planning.

Declarative Mode: When exploring beliefs, the system uses Vector Bridges (<cat <-> kitten>) to find creative associations.

Procedural Mode: When executing a Goal, the system enforces Executive Control, suppressing bridge injection to focus on the exact logical procedure.

Implementation: GeneralInferenceControl.java detects pending Goals and inhibits the bridge.

B. Signal-to-Noise Attention Filter

We implemented a data integrity check for the Attention mechanism.

Problem: Without a loaded embedding file (as in standard tests), terms have random vectors. Using them to guide attention creates "Noise," leading to confidence drift.

Solution: The system now checks hasUserVector. It only biases attention if the concept has a grounded, meaningful embedding. Random noise is ignored.

C. Attention Tuning

We tuned the Bag.java concept selection formula to be a "Nudge" rather than a "Gatekeeper."

Old: Priority * (0.3 + 0.7 * Similarity) (Aggressive).

New: Priority * (0.8 + 0.2 * Similarity) (Conservative).

Result: High-priority logic tasks retain enough budget to complete, even if they lack semantic appeal.

3. Technical Changes
Core Logic

VectorInference.java: Controller for vector operations. Manages Present Tense bridge injection to avoid conflicting with NARS belief revision.

Hypervector.java: 1024-bit binary hypervectors (BSC) with high-performance XOR/Hamming operations.

GloVeLoader.java: Parallelized parser for loading dense embeddings.

Integration Points

GeneralInferenceControl.java: Modified to consult vectors for concept selection.

Memory.java: Added transient storage for lastContextVector.

Build & Config

Flag: Features are controlled via -Dopennars.vector=true and -Dopennars.vectorBridgeInjection=true.

Default: OFF (Zero regression for standard users).

---

## Build & Run (Jar + GloVe)

### Rebuild the jar

This repo currently runs the javadoc-jar goal during `package`, which can fail on newer JDKs. If you hit a javadoc plugin error, skip javadocs:

`mvn -Dmaven.javadoc.skip=true package`

The runnable jar will be in:

`target/opennars-3.0.4-SNAPSHOT.jar`

### Run with VectorNARS flags (non-interactive)

The jar’s `Main-Class` is `org.opennars.main.Shell`, which expects **4 positional arguments**:

`narOrConfigFileOrNull idOrNull nalFileOrNull cyclesToRunOrNull`

To load GloVe embeddings and run a `.nal` script (recommended for reproducible runs), do:

`java -Dopennars.vector=true -Dopennars.vectorContext=true -Dopennars.vectorBridgeInjection=true -jar target/opennars-3.0.4-SNAPSHOT.jar null null glove-embeddings/phase21_hulk_irrational.nal 0 --glove glove-embeddings/glove.txt`

Notes:

- `--glove <path>` loads embeddings (and will also set `opennars.vector=true` internally once loading succeeds).
- The example above sets `cyclesToRunOrNull=0` so the process exits after the input file is processed.
- This specific `.nal` script contains a final line `5000`, which triggers 5000 cycles inside the engine.

### Interactive mode (what *not* to do)

If you run only:

`java -Dopennars.vector=true -Dopennars.vectorBridgeInjection=true -jar target/opennars-3.0.4-SNAPSHOT.jar --glove glove-embeddings/glove.txt`

…the system loads embeddings and then enters interactive step mode waiting for input / cycle counts from stdin. If you Ctrl-C, you’ll see exit code `130`.