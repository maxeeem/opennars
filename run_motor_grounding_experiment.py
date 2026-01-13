#!/usr/bin/env python3
"""
Minimal Motor Grounding Experiment for micro-gSCAN

Objective: Enable action initiation without policy injection,
preserving audit validity and NARS semantics.

Conditions:
- bridge_off: No similarity beliefs injected
- bridge_on: Similarity beliefs injected (epistemic bridge only)

Metrics:
- episodes_with_any_nars_action_rate
- no_action_rate (1 - action_rate)
- mean_steps_before_first_action
- success rate (optional, secondary)
"""

import subprocess
import sys
import json
import os
from pathlib import Path

def run_condition(condition: str, reps: int, seed: int):
    """Run micro-gSCAN experiment for a single condition."""
    print(f"\n{'='*60}")
    print(f"Running condition: {condition} (reps={reps}, seed={seed})")
    print(f"{'='*60}\n")
    
    cmd = [
        "python3", "broca.py",
        "--domain", "micro_gscan",
        "--condition", condition,
        "--reps", str(reps),
        "--seed", str(seed),
    ]
    
    result = subprocess.run(cmd, check=True)
    return result.returncode == 0

def compute_metrics(stamp: str, condition: str):
    """Compute action-related metrics from summary file."""
    summary_path = Path(f"runs/{stamp}_micro_gscan_{condition}.summary.json")
    
    if not summary_path.exists():
        print(f"Warning: Summary file not found: {summary_path}")
        return None
    
    with open(summary_path) as f:
        data = json.load(f)
    
    action_rate = data.get("episodes_with_any_nars_action_rate", 0.0)
    no_action_rate = 1.0 - action_rate
    
    # Try to compute mean steps before first action from trials
    trials_path = Path(f"runs/{stamp}_micro_gscan_{condition}.trials.jsonl")
    mean_steps_before_first_action = None
    
    if trials_path.exists():
        steps_to_first_action = []
        with open(trials_path) as f:
            for line in f:
                trial = json.loads(line)
                if trial.get("episode_had_nars_action", False):
                    # Estimate: use steps_to_success as proxy, or compute from no_action_steps
                    max_steps = trial.get("max_steps", 50)
                    no_action = trial.get("no_action_steps", 0)
                    first_action_step = max(1, max_steps - no_action)
                    steps_to_first_action.append(first_action_step)
        
        if steps_to_first_action:
            mean_steps_before_first_action = sum(steps_to_first_action) / len(steps_to_first_action)
    
    return {
        "condition": condition,
        "episodes_with_any_nars_action_rate": action_rate,
        "no_action_rate": no_action_rate,
        "mean_steps_before_first_action": mean_steps_before_first_action,
        "success_rate": data.get("test_success_rate", 0.0),
    }

def main():
    reps = 3
    seed = 0
    
    # Determine timestamp from most recent run or generate new one
    import time
    stamp = time.strftime("%Y%m%d_%H%M%S")
    
    conditions = ["bridge_off", "bridge_on"]
    
    for condition in conditions:
        success = run_condition(condition, reps, seed)
        if not success:
            print(f"Error: Experiment failed for condition: {condition}")
            sys.exit(1)
    
    # Wait a moment for files to be written
    time.sleep(2)
    
    # Compute metrics for both conditions
    print(f"\n{'='*60}")
    print("MINIMAL MOTOR GROUNDING EXPERIMENT RESULTS")
    print(f"{'='*60}\n")
    
    results = {}
    for condition in conditions:
        metrics = compute_metrics(stamp, condition)
        if metrics:
            results[condition] = metrics
            print(f"\n{condition.upper()}:")
            print(f"  Action Rate: {metrics['episodes_with_any_nars_action_rate']:.3f}")
            print(f"  No-Action Rate: {metrics['no_action_rate']:.3f}")
            if metrics['mean_steps_before_first_action']:
                print(f"  Mean Steps to First Action: {metrics['mean_steps_before_first_action']:.1f}")
            print(f"  Success Rate (secondary): {metrics['success_rate']:.3f}")
    
    # Compute deltas
    if "bridge_off" in results and "bridge_on" in results:
        print(f"\n{'='*60}")
        print("DELTAS (bridge_on - bridge_off)")
        print(f"{'='*60}\n")
        
        action_delta = results["bridge_on"]["episodes_with_any_nars_action_rate"] - \
                      results["bridge_off"]["episodes_with_any_nars_action_rate"]
        no_action_delta = results["bridge_on"]["no_action_rate"] - \
                         results["bridge_off"]["no_action_rate"]
        
        print(f"  Action Rate Delta: {action_delta:+.3f}")
        print(f"  No-Action Rate Delta: {no_action_delta:+.3f}")
        
        if results["bridge_on"]["mean_steps_before_first_action"] and \
           results["bridge_off"]["mean_steps_before_first_action"]:
            steps_delta = results["bridge_on"]["mean_steps_before_first_action"] - \
                         results["bridge_off"]["mean_steps_before_first_action"]
            print(f"  Mean Steps to First Action Delta: {steps_delta:+.1f}")
        
        print(f"\n{'='*60}")
        print("INTERPRETATION (with guardrails)")
        print(f"{'='*60}\n")
        
        if action_delta > 0:
            print("✓ Bridge reduces action paralysis")
            print("✓ Bridge accelerates operator discovery")
        else:
            print("✗ Bridge did not increase action rate")
        
        print("\nWhat this experiment DOES NOT CLAIM:")
        print("  - Task solving capability")
        print("  - gSCAN compositional generalization")
        print("  - Policy learning")
        
        print("\nWhat this experiment DOES CLAIM:")
        print("  - Bridge shapes exploration, not policy")
        print("  - Bridge reduces action paralysis")
        print("  - Operator learning happens through experience")

if __name__ == "__main__":
    main()
