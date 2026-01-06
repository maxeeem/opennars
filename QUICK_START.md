# Quick Start: Micro-gSCAN After Validity Fixes

## TL;DR — What Changed

**CRITICAL FIXES APPLIED** ✅

Before these fixes, all micro-gSCAN results were **invalid** due to:
1. ❌ Random action fallback contaminating training
2. ❌ No spatial information (task unsolvable)
3. ❌ Training credit for random behavior

**All fixed. Numbers are now meaningful.**

---

## Quick Commands

### Verify Fixes Work
```bash
# Syntax check
python -m py_compile broca.py

# Smoke test (fast)
python broca.py --domain micro_gscan --condition bridge_off --reps 1 --seed 0

# Audit results
python scripts/audit_micro_gscan.py runs/<stamp>_micro_gscan_bridge_off.trials.jsonl
python scripts/audit_micro_gscan.py runs/<stamp>_micro_gscan_bridge_off.summary.json
```

### Run Experiments
```bash
# Full comparison (10 reps each, ~2400 episodes total, takes time)
python broca.py --domain micro_gscan --condition bridge_off --reps 10 --seed 100 --stamp myrun
python broca.py --domain micro_gscan --condition bridge_on --reps 10 --seed 100 --stamp myrun
python broca.py --domain micro_gscan --condition randomized_embeddings --reps 10 --seed 100 --stamp myrun

# Generate comparison
python scripts/compare_micro_gscan.py myrun
```

---

## What to Look For

### Expected Patterns

**Bridge-Off (No grounding)**:
- Success rate: ~0% (NARS can't connect abstract properties to actions)
- No-action failures: Very high (NARS doesn't learn action rules)
- **This is valid and expected** — not contaminated by random fallback

**Bridge-On (With similarity)**:
- Success rate: Should be > bridge-off (if spatial structure enables learning)
- No-action failures: Should decrease (similarity facts trigger action rules)
- Time to success: Should decrease if NARS learns efficient policies

**Randomized Embeddings (Control)**:
- Success rate: Similar to bridge-off (no semantic content)
- **Purpose**: Prove bridge-on success is due to meaningful embeddings

---

## Key Metrics

### Success Attribution
- `test_success_rate` — Only counts NARS-driven successes
- `episodes_with_any_nars_action_rate` — Did NARS even try to act?

### Failure Modes
- `no_action` — NARS never produced an action (new)
- `wrong_object` — Reached distractor instead of target
- `timeout` — Didn't reach any object

### Per-Trial Details (JSONL)
- `episode_had_nars_action` — Boolean flag
- `no_action_steps` — How many steps NARS was silent

---

## Files

### Code
- **`broca.py`** — Main experiment harness (fixed)

### Documentation
- **`MICRO_GSCAN_FIXES.md`** — Detailed implementation summary
- **`RUN_NOTES.md`** — Comprehensive validity fixes documentation
- **`QUICK_START.md`** — This file

### Scripts
- **`scripts/compare_micro_gscan.py`** — Generate comparison JSONs
- **`scripts/audit_micro_gscan.py`** — Verify fixes are working

---

## Quick Audit Checklist

Before trusting results, verify:

✅ No random fallback in code:
```bash
grep "world.rnd.choice" broca.py | grep -v "#"  # Should be empty
```

✅ Spatial observations injected:
```bash
grep "cell_ahead" broca.py  # Should find injection code
```

✅ New metrics in summary:
```bash
python scripts/audit_micro_gscan.py runs/<file>.summary.json
```

✅ Trial tracking works:
```bash
python scripts/audit_micro_gscan.py runs/<file>.trials.jsonl
```

---

## Interpretation Guide

### "Success rate is 0%"
**Before fixes**: ❌ Bad — means experiment is broken  
**After fixes**: ✅ Valid — NARS genuinely hasn't learned

### "All failures are no_action"
**Before fixes**: ❌ Would be hidden by random fallback  
**After fixes**: ✅ Informative — NARS isn't producing actions

### "Bridge-on same as bridge-off"
**Before fixes**: ❌ Can't tell (contaminated)  
**After fixes**: ✅ Valid finding — either:
- Bridge isn't helping
- Training is insufficient
- Task still too hard

### "Success without no_action failures"
**Before fixes**: ❌ Could be random wandering  
**After fixes**: ✅ Genuine learning — NARS is acting and succeeding

---

## What's Still Missing (Future)

### Not Implemented Yet
- Task E: Neural baseline for comparison
  - Seq2seq on TRAIN combos only
  - Hand-coded bag-of-attributes controller
  - Must use same observation space

### Recommendations
1. Run full 10+ rep experiments for statistical power
2. Add baseline for context (user requirement, not yet done)
3. Consider longer training if no-action rate is high
4. Analyze TRAIN phase logs to see if learning is occurring

---

## Bottom Line

**All micro-gSCAN results before these fixes should be discarded.**

**After these fixes, the benchmark is valid.** Low numbers are now informative rather than artifacts of engineering cheats.

Run experiments, trust the numbers, iterate on NARS configuration.
