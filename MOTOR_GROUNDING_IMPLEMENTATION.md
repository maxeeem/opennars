# Minimal Motor Grounding for micro-gSCAN

## Implementation Summary

### Step A: Minimal Motor Primitives (Complete)

Added three primitive operations to the NARS interface:
- `^forward` - Move forward one cell
- `^turn_left` - Turn left 90 degrees
- `^turn_right` - Turn right 90 degrees

**Implementation:**
- Created Java operators in `src/main/java/org/opennars/operator/motor/`:
  - `Forward.java`
  - `TurnLeft.java`
  - `TurnRight.java`
- Registered operators in `src/main/resources/config/defaultConfig.xml`
- Updated Python parser in `broca.py` to recognize `[MOTOR]` output pattern
- Updated `_micro_gscan_choose_action()` to map motor primitives to gridworld actions

**Constraints Satisfied:**
✓ No object arguments
✓ No target arguments
✓ No conditions injected
✓ No reward tied to correctness
✓ Blind motor babbling primitives

### Step B: Operator Learning (Complete)

**Verification:**
- NARS operator learning is enabled by default in OpenNARS
- Operators acquire budget through experience (no hardcoded action selection)
- No scripted rules injected in Python
- NARS can form implications like: `<(&, <cell_ahead --> empty>, <dir --> north>) ==> ^forward>`

**Mechanism:**
- Motor execution provides feedback to NARS via standard operator execution channel
- Budget propagation occurs naturally through NARS inference
- No Python-side policy injection

### Step C: Bridge Remains Epistemic (Complete)

**Verified in `_micro_gscan_inject_perception()`:**
- Bridge only injects similarity beliefs: `<obj_token <-> property>`
- No implications injected (e.g., no `<red --> [move_to]>`)
- No goals injected
- No operator references
- Bridge uses cosine similarity from CLIP embeddings
- Truth values reflect similarity scores: `%frequency;confidence%`

**Example bridge injection:**
```narsese
$0.723;0.723$ <obj_1 <-> red>. %0.723;0.900% :|:
```

### Step D: Minimal Experiment (In Progress)

**Experiment Script:** `run_motor_grounding_experiment.py`

**Conditions:**
- `bridge_off`: No similarity beliefs injected
- `bridge_on`: Similarity beliefs injected (epistemic only)

**Parameters:**
- Reps: 3
- Train episodes: 40
- Test episodes: 40
- Max steps: 50

**Metrics Tracked:**
- `episodes_with_any_nars_action_rate` - Fraction of episodes where NARS produced any action
- `no_action_rate` - Fraction of episodes with no NARS action (= 1 - action_rate)
- `mean_steps_before_first_action` - Average steps until first action in episodes with action
- `success_rate` - (Optional, secondary) Task completion rate

**No Comparison Against:**
- Neural network baselines
- Scaled domains
- Tuned parameters

### Step E: Interpretation Guardrails (Template)

**Claims We CAN Make:**
- Bridge reduces action paralysis (if `action_rate` increases)
- Bridge accelerates operator discovery (if `steps_before_first_action` decreases)
- Bridge shapes exploration, not policy

**Claims We CANNOT Make:**
- Task solving capability
- gSCAN compositional generalization performance
- Policy learning effectiveness
- Semantic understanding

## Build Instructions

```bash
# Build NARS with new motor operators
mvn clean package -DskipTests -Dmaven.javadoc.skip=true

# Run minimal experiment
python3 run_motor_grounding_experiment.py
```

## Files Modified

### Java (NARS Core)
- `src/main/java/org/opennars/operator/motor/Forward.java` (NEW)
- `src/main/java/org/opennars/operator/motor/TurnLeft.java` (NEW)
- `src/main/java/org/opennars/operator/motor/TurnRight.java` (NEW)
- `src/main/resources/config/defaultConfig.xml` (MODIFIED - added motor operators)

### Python (Experiment Framework)
- `broca.py` (MODIFIED - added motor action parsing)
- `run_motor_grounding_experiment.py` (NEW - experiment runner)

## Next Steps

1. Complete experiment run (3 reps each condition)
2. Analyze results
3. Generate final report with interpretation guardrails
4. Document findings

## Audit Validity

✓ No policy injection
✓ No task-specific rules
✓ No reward shaping
✓ Operators learn through experience only
✓ Bridge remains purely epistemic (similarity beliefs only)
✓ NARS semantics preserved
