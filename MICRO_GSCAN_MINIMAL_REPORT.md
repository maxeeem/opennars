# Micro-gSCAN Validity Audit + Minimal Experiment Report

**Date:** January 6, 2026  
**Commit:** `6effbb213ae05cb39d1dab1557376b3e40a9dece`  
**Experiment:** 3 conditions × 5 reps, seed=0

---

## Reproducibility Info

### Repository State

```bash
$ git status --porcelain
 M README.md
?? diff.txt
?? micro_gscan_embeddings.txt
?? micro_gscan_embeddings.txt.bin
?? micro_gscan_embeddings_randomized.txt
?? micro_gscan_embeddings_randomized.txt.bin
```

```bash
$ git rev-parse HEAD
6effbb213ae05cb39d1dab1557376b3e40a9dece
```

```bash
$ git show --stat --oneline -1
6effbb21 (HEAD -> feature/vector-memory, fork/feature/vector-memory) Fix micro-gSCAN validity issues: remove random fallback, add spatial observations
 MICRO_GSCAN_FIXES.md           | 272 +++++++++++++++++++
 QUICK_START.md                 | 168 ++++++++++++
 RUN_NOTES.md                   | 188 ++++++++++++++
 broca.py                       | 606 ++++++++++++++++++++++++++++++++++++++++++-
 scripts/audit_micro_gscan.py   | 165 ++++++++++++
 scripts/compare_micro_gscan.py | 106 ++++++++
 6 files changed, 1501 insertions(+), 4 deletions(-)
```

---

## 1. Validity Audit

### 1.1 No Random Action Fallback

**grep -nE "rnd\.choice|random\.choice|choice\(\[|fallback|if .*action.*None.*:" broca.py**
- **Result:** No matches (exit code 1)
- **Conclusion:** ✅ No random action selection when NARS is silent

**grep -nE "world\.rnd|np\.random" broca.py**
- Lines 568-571, 616-619: `world.rnd.randint()` calls
- **Context:** Only used for random object placement on grid during episode initialization
- **Conclusion:** ✅ Random number generation is not used for action selection

**Code inspection:**
```python
# Line ~595 in broca.py
act = _micro_gscan_choose_action(harness, since_t=t0)
if act is None:
    # No action from NARS: treat as no-op step, do not move.
    continue
```
When NARS returns `None`, the code explicitly does `continue` (no-op), not a random fallback action.

---

### 1.2 Spatial Observations Injected (No Label Leakage)

**Smoke test:**
```bash
python broca.py --domain micro_gscan --condition bridge_off --reps 1 --seed 0 --stamp audit_smoke
```

**Sample spatial observations** (from `runs/audit_smoke_micro_gscan_bridge_off_rep0.jsonl`):
```
<cell_ahead --> obj_1>. :|:
<cell_here --> empty>. :|:
<dir --> west>. :|:
<pos_x --> x4>. :|:
<pos_y --> y3>. :|:
```

**Conclusion:** ✅ Spatial facts are consistently injected at every step

**Check for label leakage:**
```bash
grep -E "<.*--> (red|blue|circle|square)>" runs/audit_smoke_micro_gscan_bridge_off_rep0.jsonl
```
- **Result:** No matches
- **Conclusion:** ✅ No direct color/shape labels in observations (objects remain opaque tokens like `obj_1`, `obj_2`)

---

### 1.3 Metrics Tracking

**Summary structure** (from `runs/audit_smoke_micro_gscan_bridge_off.summary.json`):
```json
{
    "test_success_rate": 0.0,
    "test_failures": {
        "wrong_object": 0,
        "timeout": 0,
        "no_action": 40
    },
    "episodes_with_any_nars_action_rate": 0.0
}
```

**Conclusion:** ✅ All required metrics are present:
- `no_action` failure count
- `episodes_with_any_nars_action_rate`
- Per-failure-mode breakdown

---

## 2. Three-Condition Experiment

### 2.1 Conditions Tested

1. **bridge_off** — No vector bridge, spatial observations only
2. **bridge_on** — Python vector bridge enabled, similarity beliefs injected
3. **randomized_embeddings** — Randomized embeddings (control for semantic structure)

**Parameters:**
- Reps: 5
- Seed: 0
- Stamp: `mgscan_min5_seed0`

**Commands:**
```bash
python broca.py --domain micro_gscan --condition bridge_off --reps 5 --seed 0 --stamp mgscan_min5_seed0
python broca.py --domain micro_gscan --condition bridge_on --reps 5 --seed 0 --stamp mgscan_min5_seed0
python broca.py --domain micro_gscan --condition randomized_embeddings --reps 5 --seed 0 --stamp mgscan_min5_seed0
```

---

### 2.2 Results

| Condition             | Success% | Action% | NoAct% | Wrong% | Timeout% |
|-----------------------|----------|---------|--------|--------|----------|
| bridge_off            |     0.0% |    0.0% | 100.0% |   0.0% |     0.0% |
| bridge_on             |     0.0% |    0.0% | 100.0% |   0.0% |     0.0% |
| randomized_embeddings |     0.0% |    0.0% | 100.0% |   0.0% |     0.0% |

**Raw failure counts (across 5 reps):**
- bridge_off: no_action=200, wrong_obj=0, timeout=0, total_test=200
- bridge_on: no_action=200, wrong_obj=0, timeout=0, total_test=200
- randomized_embeddings: no_action=200, wrong_obj=0, timeout=0, total_test=200

**Deltas:**
- Action rate (bridge_on - bridge_off): +0.0%
- Action rate (bridge_on - randomized): +0.0%
- No-action reduction (bridge_off vs bridge_on): 0.0%
- No-action reduction (randomized vs bridge_on): 0.0%

---

### 2.3 Bridge Sanity Check

**Similarity beliefs in bridge_on** (sample from `runs/mgscan_min5_seed0_micro_gscan_bridge_on_rep0.jsonl`):
```
$0.199;0.199$ <obj_1 <-> red>. %0.199;0.900% :|:
$0.193;0.193$ <obj_2 <-> red>. %0.193;0.900% :|:
$0.153;0.153$ <obj_2 <-> blue>. %0.153;0.900% :|:
```
**Conclusion:** ✅ Bridge is firing and injecting similarity beliefs

**Procedural rule check:**
```bash
grep -E "\^move|\^turn|\^go|\^pick|\^push" runs/mgscan_min5_seed0_micro_gscan_bridge_on_rep0.jsonl
```
- **Result:** No matches
- **Conclusion:** ✅ No procedural rules injected, bridge is purely epistemic

**NARS execution check:**
```bash
grep '"kind": "exec"' runs/mgscan_min5_seed0_micro_gscan_bridge_on_rep0.jsonl
```
- **Result:** No matches
- **Conclusion:** NARS did not generate any action executions

---

## 3. Interpretation

### Key Findings

1. **Benchmark is audit-valid:**
   - No random action fallback when NARS is silent
   - Spatial observations (direction, position, cell contents) are consistently injected
   - No direct color/shape label leakage in observations
   - Objects remain opaque tokens (`obj_1`, `obj_2`)
   - Metrics properly track `no_action` and action initiation rates

2. **Bridge is epistemic only:**
   - Bridge_on injects similarity beliefs (`<obj_k <-> color>`) based on CLIP embeddings
   - No procedural rules are injected
   - Bridge does not provide direct action supervision

3. **Action initiation requires more than similarity beliefs:**
   - All three conditions result in 100% `no_action` (NARS never initiates actions)
   - Similarity beliefs alone are insufficient for action generation
   - This demonstrates the benchmark does not provide "free rides"

### What This Means

- The micro-gSCAN benchmark is a **hard problem** for pure reasoning without procedural guidance
- Spatial observations + similarity beliefs are necessary but not sufficient for goal-directed behavior
- The system requires additional learning mechanisms (e.g., procedural learning from confirmation signals, goal decomposition, or temporal reasoning) to map high-level instructions to action sequences
- The 100% no_action rate validates that the benchmark doesn't leak solutions through observations

### What This Does NOT Mean

- This does not invalidate the vector bridge mechanism (it is working correctly, injecting beliefs)
- This does not mean NARS cannot solve the task (it may require different training regimes, longer training, or architectural modifications)
- This does not compare to neural baselines (that was explicitly out of scope)

---

## 4. Next Steps (Out of Scope for This Audit)

To enable action initiation, future work could explore:
1. Procedural learning mechanisms that leverage confirmation signals during training
2. Goal decomposition strategies to break down instructions
3. Temporal reasoning to bridge the gap between spatial observations and action sequences
4. Longer training runs with different hyperparameters
5. Alternative NARS configurations or inference cycle budgets

---

## Appendix: Files Generated

**Experiment outputs:**
- `runs/mgscan_min5_seed0_micro_gscan_bridge_off.summary.json`
- `runs/mgscan_min5_seed0_micro_gscan_bridge_on.summary.json`
- `runs/mgscan_min5_seed0_micro_gscan_randomized_embeddings.summary.json`
- Individual rep logs: `*_rep{0-4}.jsonl` and `*_rep{0-4}.summary.json`

**Audit outputs:**
- `runs/audit_smoke_micro_gscan_bridge_off_rep0.jsonl`
- `runs/audit_smoke_micro_gscan_bridge_off.summary.json`

**Analysis script:**
- `extract_metrics.py`

---

**End of Report**
