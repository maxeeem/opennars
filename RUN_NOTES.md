# Micro-gSCAN Validity Hardening — Run Notes

**Date**: January 6, 2026  
**Commit**: Validity fixes for micro-gSCAN experiment

---

## What Changed

### Critical Validity Fixes

#### 1. **Removed Random Action Fallback (Task A)**
**Problem**: Previously, when NARS failed to output an action, the harness would inject a random action (`left`, `right`, or `forward`) and continue the episode. If the random action led to success, the system would emit `confirm`, attributing success to behavior NARS didn't cause.

**Fix**: 
- Replaced random fallback with explicit **no-action handling**
- When `act is None`, the step is now recorded as a **no-op** (no movement occurs)
- In TRAIN: `confirm` is **only** emitted when the agent reaches target **AND** at least one NARS action occurred in the episode
- In TEST: Success is only recorded if NARS acted
- Added new failure mode: **`no_action`** to track episodes where NARS never produced an action

**Impact**: Training credit is no longer contaminated by random wandering. Test success now requires genuine NARS-driven behavior.

---

#### 2. **Added Spatial Observations (Task B)**
**Problem**: The original implementation only injected `<obj_i --> [seen]>` for all visible objects with no spatial structure. NARS had full observability but no information about:
- Agent position or orientation
- Object locations relative to agent
- Which cell the agent is facing
- Distance or direction to objects

This made systematic navigation impossible — success was essentially chance-based wandering.

**Fix**: Added **egocentric spatial observations** while preserving compositional generalization:
- `<cell_ahead --> obj_?>. :|:` when object is in front of agent (or `empty` if none)
- `<cell_here --> obj_?>. :|:` for object at current position (or `empty`)
- `<dir --> north>. :|:` (or `east`, `south`, `west`) for current orientation
- `<pos_x --> x2>. :|:` and `<pos_y --> y4>. :|:` for absolute grid position

**Critical design choice**: Objects remain **opaque tokens** (`obj_1`, `obj_2`, etc.) with no direct color/shape facts. Only the bridge mechanism (when enabled) provides similarity between objects and properties. This preserves the compositional generalization requirement.

**Impact**: NARS can now learn structured policies like:
- "If target property holds for object ahead, move forward"
- "If wrong object ahead, turn"
- "Navigate toward target based on spatial cues"

The environment is now **solvable by rule-like behavior** rather than pure chance.

---

#### 3. **TRAIN Supervision Guards (Task C)**
**Problem**: `confirm` was emitted whenever the agent reached the target, even in episodes dominated by random fallback actions.

**Fix**:
- Added `episode_had_nars_action` flag tracking whether NARS produced at least one action during the episode
- `confirm` is now only emitted when both:
  1. Agent reaches correct target, AND
  2. `episode_had_nars_action == True`
- This flag is logged in all trial records

**Impact**: Training signal is now **legitimate** — only NARS-driven successes receive reinforcement.

---

#### 4. **Clarified Bridge Mechanisms (Task D)**
**Problem**: The code had bridge logic scattered without clear separation. It was unclear whether the Python-side similarity injection or Java vector bridge was active.

**Fix**:
- Added explicit `bridge_mechanism` field to all summary JSONs: `"python"`, `"off"`, or `"java"` (future)
- Clarified conditions:
  - `bridge_off`: No similarity injection (pure symbolic)
  - `bridge_on`: Python-side similarity injection with real embeddings
  - `randomized_embeddings`: Python-side injection with randomized object vectors (control)
  - `ablate_bridge`: Alias for `bridge_off` (legacy compatibility)
- Java bridge remains forcibly disabled for this benchmark (set via `config/bridge_off.xml`)

**Impact**: Experimental conditions are now unambiguous and traceable in logs.

---

## New Metrics

### Success Metrics
- **`test_success_rate`**: Proportion of test episodes where agent reached correct target **with at least one NARS action**
- **`test_time_to_success_mean`**: Average steps to success (for successful episodes only)

### Failure Breakdown
- **`wrong_object`**: Reached distractor instead of target
- **`timeout`**: Max steps reached without contacting any object
- **`no_action`**: NARS never produced an action in the episode (new)

### Action Attribution
- **`episodes_with_any_nars_action_rate`**: Proportion of test episodes where NARS produced at least one action
- **`episode_had_nars_action`**: Per-trial boolean flag (in JSONL logs)
- **`no_action_steps`**: Count of steps where NARS failed to act (per trial)

---

## Why This Removes "Engineering Cheats"

### Before Fixes
1. **Random fallback masked NARS incompetence**: An agent that never learned could still "succeed" by chance wandering
2. **Training on noise**: `confirm` could reinforce episodes where NARS contributed nothing
3. **Unsolvable task**: Without spatial structure, optimal behavior was undefined
4. **Opaque metrics**: Success rate included random-driven outcomes

### After Fixes
1. **No credit without action**: Success requires NARS to actually choose moves
2. **Clean training signal**: Reinforcement only for NARS-driven successes
3. **Learnable structure**: Spatial observations enable systematic policy learning
4. **Traceable failures**: `no_action` mode explicitly shows when NARS fails to act

---

## Expected Results Post-Fix

### Bridge-Off (Pure Symbolic)
- **Expected**: Very low success rate initially
- **High `no_action` rate**: NARS has no grounding between abstract properties and spatial actions
- **Metric integrity**: 0% success is now **valid and informative** (not contaminated by random fallback)

### Bridge-On (Python Similarity)
- **Expected**: Gradual learning if spatial structure enables policy formation
- **Lower `no_action` rate**: Similarity facts should eventually trigger action rules
- **Success depends on**: Whether NARS can integrate similarity facts with spatial navigation

### Randomized Embeddings
- **Expected**: Similar to bridge-off (control for semantic content)
- **Purpose**: Verify that success in bridge-on is due to meaningful embeddings, not just any numerical similarity injection

---

## Running Comparisons

### Smoke Test (Verification)
```bash
python broca.py --domain micro_gscan --condition bridge_off --reps 1 --seed 0
```
✓ Confirms `no_action` failures appear  
✓ No random fallback in logs  
✓ `episode_had_nars_action` and `no_action_steps` tracked

### Minimal Comparison (10 Reps)
```bash
python broca.py --domain micro_gscan --condition bridge_off --reps 10 --seed 100 --stamp 20260106_compare
python broca.py --domain micro_gscan --condition bridge_on --reps 10 --seed 100 --stamp 20260106_compare
python broca.py --domain micro_gscan --condition randomized_embeddings --reps 10 --seed 100 --stamp 20260106_compare
```

### Output Artifacts
- `runs/20260106_compare_micro_gscan_<condition>.summary.json`: Aggregate stats per condition
- `runs/20260106_compare_micro_gscan_<condition>.trials.jsonl`: Per-episode details
- Comparison JSON (to be generated): Side-by-side metrics across conditions

---

## Assertions for Validity

1. ✓ **No random actions**: Grep logs for "random" fallback → should find none
2. ✓ **No-action tracked**: All conditions show `no_action` failures (expected in baseline)
3. ✓ **Spatial facts injected**: Logs contain `<cell_ahead --> ...>`, `<dir --> ...>`, etc.
4. ✓ **Episode action tracking**: All trials have `episode_had_nars_action` field
5. ✓ **Bridge mechanism labeled**: All summaries include `bridge_mechanism` field

---

## Next Steps (Not Yet Implemented)

### Task E: Neural Baseline (Future Work)
To complete the experimental package, add a comparable baseline:
- **Option 1**: Tiny seq2seq trained on TRAIN combos, evaluated on TEST (compositional split)
- **Option 2**: Hand-coded "bag-of-attributes" controller using similarity but no NARS memory

Baseline must use the **same observation space** (spatial facts) and be evaluated with the same success criteria.

---

## Summary

These fixes transform micro-gSCAN from a **broken benchmark** (where numbers were meaningless due to random fallback contamination and unsolvable task structure) into a **valid experimental environment** where:

- Success attributions are correct
- Training signals are clean
- The task is learnable
- Metrics are interpretable

**Before trusting any micro-gSCAN numbers, these fixes were mandatory.**
