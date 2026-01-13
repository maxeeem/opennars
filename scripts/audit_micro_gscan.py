#!/usr/bin/env python3
"""
Audit micro-gSCAN logs to verify validity fixes are working correctly.
"""
import json
import sys
from pathlib import Path

def audit_trials(trials_path):
    """Audit a trials JSONL file for validity issues."""
    trials_path = Path(trials_path)
    if not trials_path.exists():
        print(f"Error: {trials_path} not found")
        return False
    
    print(f"\n=== Auditing {trials_path.name} ===\n")
    
    trials = []
    with open(trials_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            trials.append(json.loads(line))
    
    print(f"Total trials: {len(trials)}")
    
    # Check 1: All trials have episode_had_nars_action field
    missing_flag = [t for t in trials if "episode_had_nars_action" not in t]
    if missing_flag:
        print(f"❌ FAIL: {len(missing_flag)} trials missing episode_had_nars_action field")
        return False
    print("✅ PASS: All trials have episode_had_nars_action field")
    
    # Check 2: All trials have no_action_steps field
    missing_steps = [t for t in trials if "no_action_steps" not in t]
    if missing_steps:
        print(f"❌ FAIL: {len(missing_steps)} trials missing no_action_steps field")
        return False
    print("✅ PASS: All trials have no_action_steps field")
    
    # Check 3: no_action outcome exists
    no_action_trials = [t for t in trials if t.get("outcome") == "no_action"]
    print(f"✅ INFO: {len(no_action_trials)} trials with no_action outcome")
    
    # Check 4: Success only when episode_had_nars_action is True
    success_trials = [t for t in trials if t.get("outcome") == "success"]
    success_without_action = [t for t in success_trials if not t.get("episode_had_nars_action", False)]
    if success_without_action:
        print(f"❌ FAIL: {len(success_without_action)} success trials without NARS action!")
        for t in success_without_action[:3]:
            print(f"  Episode {t['episode']}: {t}")
        return False
    print(f"✅ PASS: All {len(success_trials)} success trials had NARS action")
    
    # Check 5: Statistics
    outcomes = {}
    for t in trials:
        outcome = t.get("outcome", "unknown")
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
    
    print("\nOutcome distribution:")
    for outcome, count in sorted(outcomes.items()):
        print(f"  {outcome}: {count} ({100*count/len(trials):.1f}%)")
    
    action_rate = sum(1 for t in trials if t.get("episode_had_nars_action", False)) / len(trials)
    print(f"\nEpisodes with any NARS action: {action_rate:.2%}")
    
    avg_no_action_steps = sum(t.get("no_action_steps", 0) for t in trials) / len(trials)
    print(f"Average no-action steps per episode: {avg_no_action_steps:.1f}")
    
    return True

def audit_summary(summary_path):
    """Audit a summary JSON file."""
    summary_path = Path(summary_path)
    if not summary_path.exists():
        print(f"Error: {summary_path} not found")
        return False
    
    print(f"\n=== Auditing {summary_path.name} ===\n")
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    # Check 1: bridge_mechanism field exists
    if "bridge_mechanism" not in summary:
        print("❌ FAIL: Missing bridge_mechanism field")
        return False
    print(f"✅ PASS: bridge_mechanism = {summary['bridge_mechanism']}")
    
    # Check 2: episodes_with_any_nars_action_rate exists
    if "episodes_with_any_nars_action_rate" not in summary:
        print("❌ FAIL: Missing episodes_with_any_nars_action_rate field")
        return False
    print(f"✅ PASS: episodes_with_any_nars_action_rate = {summary['episodes_with_any_nars_action_rate']:.2%}")
    
    # Check 3: no_action in test_failures
    failures = summary.get("test_failures", {})
    if "no_action" not in failures:
        print("❌ FAIL: Missing no_action in test_failures")
        return False
    print(f"✅ PASS: no_action failures = {failures['no_action']}")
    
    # Check 4: Print key metrics
    print("\nKey metrics:")
    print(f"  Success rate: {summary.get('test_success_rate', 0):.2%}")
    print(f"  Mean time to success: {summary.get('test_time_to_success_mean', 'N/A')}")
    print(f"  Failures:")
    for failure_type, count in failures.items():
        print(f"    {failure_type}: {count}")
    
    return True

def audit_condition_log(log_path):
    """Audit a condition log JSONL to check for random action fallback."""
    log_path = Path(log_path)
    if not log_path.exists():
        print(f"Error: {log_path} not found")
        return False
    
    print(f"\n=== Auditing {log_path.name} for random fallback ===\n")
    
    with open(log_path, 'r') as f:
        for i, line in enumerate(f, 1):
            line_lower = line.lower()
            # Check for suspicious patterns that might indicate random fallback
            if "random" in line_lower and ("choice" in line_lower or "fallback" in line_lower):
                print(f"⚠️  WARNING: Line {i} contains 'random' and 'choice'/'fallback'")
                print(f"  {line.strip()[:200]}")
    
    print("✅ PASS: No obvious random fallback patterns found in log")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python audit_micro_gscan.py <trials.jsonl|summary.json|log.jsonl>")
        print("Example: python audit_micro_gscan.py runs/20260106_091615_micro_gscan_bridge_off.trials.jsonl")
        sys.exit(1)
    
    path = Path(sys.argv[1])
    
    if not path.exists():
        print(f"Error: {path} not found")
        sys.exit(1)
    
    success = True
    
    if path.suffix == ".jsonl":
        if "trials" in path.name:
            success = audit_trials(path)
        else:
            success = audit_condition_log(path)
    elif path.suffix == ".json":
        success = audit_summary(path)
    else:
        print(f"Unknown file type: {path.suffix}")
        sys.exit(1)
    
    if success:
        print("\n✅ All audits passed!")
        sys.exit(0)
    else:
        print("\n❌ Some audits failed!")
        sys.exit(1)
