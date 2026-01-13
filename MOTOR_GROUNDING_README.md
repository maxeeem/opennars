# Minimal Motor Grounding for micro-gSCAN

## Quick Start

### Build
```bash
mvn clean package -DskipTests -Dmaven.javadoc.skip=true
```

### Test (Quick)
```bash
python3 test_motor_grounding.py
```

### Run Full Experiment
```bash
python3 run_motor_grounding_experiment.py
```

## What Was Implemented

### Step A: Minimal Motor Primitives ✓
- `^forward` - blind forward movement
- `^turn_left` - blind left turn
- `^turn_right` - blind right turn
- No arguments, no conditions, no rewards

### Step B: Operator Learning ✓
- NARS forms implications through experience
- No scripted rules in Python
- Operators acquire budget naturally
- Example: `<(&, <cell_ahead --> empty>, <dir --> north>) ==> ^forward>`

### Step C: Bridge Remains Epistemic ✓
- Only injects similarity beliefs: `<obj <-> property>`
- No implications
- No goals
- No operator references
- Verified in `broca.py:_micro_gscan_inject_perception()`

### Step D: Minimal Experiment ✓
- Conditions: `bridge_off`, `bridge_on`
- Reps: 3
- Metrics:
  - `episodes_with_any_nars_action_rate`
  - `no_action_rate`
  - `mean_steps_before_first_action`
  - `success_rate` (secondary)

### Step E: Interpretation Guardrails ✓
**Claims:**
- Bridge reduces action paralysis
- Bridge accelerates operator discovery
- Bridge shapes exploration, not policy

**Does NOT claim:**
- Task solving
- gSCAN performance
- Policy learning

## Files

### New Files
- `src/main/java/org/opennars/operator/motor/Forward.java`
- `src/main/java/org/opennars/operator/motor/TurnLeft.java`
- `src/main/java/org/opennars/operator/motor/TurnRight.java`
- `run_motor_grounding_experiment.py`
- `test_motor_grounding.py`
- `MOTOR_GROUNDING_IMPLEMENTATION.md`
- `MOTOR_GROUNDING_REPORT.md`
- `MOTOR_GROUNDING_README.md` (this file)

### Modified Files
- `src/main/resources/config/defaultConfig.xml` (registered motor operators)
- `broca.py` (added motor action parsing)

## Documentation

- **Implementation Details**: See [MOTOR_GROUNDING_IMPLEMENTATION.md](MOTOR_GROUNDING_IMPLEMENTATION.md)
- **Experiment Report**: See [MOTOR_GROUNDING_REPORT.md](MOTOR_GROUNDING_REPORT.md)

## Audit Validity

✓ No policy injection  
✓ No task-specific rules  
✓ No reward shaping  
✓ Operators learn through experience only  
✓ Bridge remains purely epistemic  
✓ NARS semantics preserved
