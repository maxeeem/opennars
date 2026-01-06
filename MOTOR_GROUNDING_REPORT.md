# Minimal Motor Grounding Experiment Report

## Objective
Enable action initiation in micro-gSCAN without policy injection, preserving audit validity and NARS semantics.

## Hypothesis
The epistemic bridge (similarity beliefs only) reduces action paralysis and accelerates operator discovery compared to no bridge.

---

## Experimental Design

### Conditions
1. **bridge_off** (control): No similarity beliefs injected
2. **bridge_on** (experimental): Similarity beliefs injected from CLIP embeddings

### Motor Primitives
Three blind motor babbling primitives (no arguments, no conditions):
- `^forward` - Move forward one cell
- `^turn_left` - Turn left 90 degrees  
- `^turn_right` - Turn right 90 degrees

### Parameters
- Repetitions: 3
- Train episodes: 40 (with confirmation signal)
- Test episodes: 40 (no confirmation, metrics only)
- Max steps per episode: 50
- Random seed: 0

### Training Signal
- Confirmation provided only when:
  1. Agent reaches correct target object
  2. NARS produced at least one action in the episode
- No explicit reward tied to operator correctness
- Operators learn through experience and budget propagation

### Bridge Specification
- **Type**: Epistemic only (similarity beliefs)
- **Content**: `<obj_token <-> property>` where property ∈ {red, blue, circle, square}
- **Truth values**: Cosine similarity from CLIP embeddings
- **Budget**: Proportional to similarity
- **No implications**: Bridge never injects `<A ==> B>`
- **No goals**: Bridge never injects `<goal>!`
- **No operators**: Bridge never references `^operator`

---

## Metrics

### Primary Metrics
1. **episodes_with_any_nars_action_rate**
   - Fraction of test episodes where NARS produced ≥1 action
   - Measures: Action paralysis vs. exploration

2. **no_action_rate**  
   - Fraction of test episodes with zero NARS actions
   - Inverse of action_rate: `no_action_rate = 1 - action_rate`

3. **mean_steps_before_first_action**
   - Average number of steps until first action (in episodes with action)
   - Measures: Operator discovery speed

### Secondary Metric
4. **success_rate**
   - Task completion rate (correct target reached)
   - Not primary claim, but useful context

---

## Results

### Condition: bridge_off
- Action Rate: [TO BE FILLED]
- No-Action Rate: [TO BE FILLED]
- Mean Steps to First Action: [TO BE FILLED]
- Success Rate: [TO BE FILLED]

### Condition: bridge_on
- Action Rate: [TO BE FILLED]
- No-Action Rate: [TO BE FILLED]
- Mean Steps to First Action: [TO BE FILLED]
- Success Rate: [TO BE FILLED]

### Delta (bridge_on - bridge_off)
- Action Rate Delta: [TO BE FILLED]
- No-Action Rate Delta: [TO BE FILLED]
- Mean Steps to First Action Delta: [TO BE FILLED]

---

## Interpretation (WITH GUARDRAILS)

### What This Experiment DOES Claim

✓ **Bridge reduces action paralysis**
  - IF action_rate increases with bridge_on
  - Epistemic grounding shapes exploration space
  - Operators become discoverable through experience

✓ **Bridge accelerates operator discovery**
  - IF mean_steps_before_first_action decreases with bridge_on
  - Similarity beliefs provide structure for inference
  - Faster convergence to action-relevant states

✓ **Bridge shapes exploration, not policy**
  - No implications injected
  - No goals injected
  - Operators acquire budget only through experience
  - NARS forms implications autonomously

### What This Experiment DOES NOT Claim

✗ **Task solving capability**
  - Success rate is secondary metric
  - No claim about compositional generalization
  - micro-gSCAN is minimal test, not full evaluation

✗ **gSCAN performance**
  - Not evaluating full gSCAN benchmark
  - Not comparing against neural network baselines
  - Not claiming human-level or state-of-art performance

✗ **Policy learning**
  - No policy injection
  - No action selection heuristics
  - No scripted rules
  - Operators babble blindly until discovered

✗ **Semantic understanding**
  - CLIP embeddings provide similarity, not semantics
  - NARS forms its own representations through experience
  - Bridge is purely epistemic (observations only)

---

## Audit Validity

### Constraints Satisfied

✓ **No policy injection**
  - Motor primitives have no arguments
  - No target conditions
  - No reward tied to correctness

✓ **NARS semantics preserved**
  - Operator learning through experience
  - Implication formation autonomous
  - Budget propagation natural

✓ **Bridge epistemic only**
  - Similarity beliefs only
  - No implications
  - No goals
  - No operator references

✓ **Reproducible**
  - Fixed seed
  - Deterministic embeddings
  - No stochastic policy

---

## Conclusion

[TO BE FILLED AFTER RESULTS]

The epistemic bridge [increased/decreased] action rate by [X]% and [reduced/increased] mean steps to first action by [Y] steps. These results [support/do not support] the hypothesis that similarity beliefs reduce action paralysis and accelerate operator discovery without injecting policy.

**Key Takeaway**: This experiment demonstrates that [summary with appropriate guardrails].

---

## Files and Code

### Implementation
- Motor operators: `src/main/java/org/opennars/operator/motor/*.java`
- Bridge verification: `broca.py:_micro_gscan_inject_perception()`
- Experiment runner: `run_motor_grounding_experiment.py`
- Quick test: `test_motor_grounding.py`

### Build
```bash
mvn clean package -DskipTests -Dmaven.javadoc.skip=true
```

### Run
```bash
python3 run_motor_grounding_experiment.py
```

### Output
- Trial logs: `runs/*_micro_gscan_*.trials.jsonl`
- Summaries: `runs/*_micro_gscan_*.summary.json`
