#!/usr/bin/env python3
"""
Generate comparison JSON from multiple micro-gSCAN condition summaries.
"""
import json
import sys
import os
from pathlib import Path

def load_summary(path):
    """Load a summary JSON file."""
    with open(path, 'r') as f:
        return json.load(f)

def generate_comparison(stamp, conditions):
    """Generate comparison JSON for given stamp and conditions."""
    runs_dir = Path("runs")
    summaries = {}
    
    for condition in conditions:
        summary_path = runs_dir / f"{stamp}_micro_gscan_{condition}.summary.json"
        if not summary_path.exists():
            print(f"Warning: {summary_path} not found", file=sys.stderr)
            continue
        summaries[condition] = load_summary(summary_path)
    
    if not summaries:
        print("Error: No summaries found", file=sys.stderr)
        sys.exit(1)
    
    # Build comparison structure
    comparison = {
        "timestamp": stamp,
        "domain": "micro_gscan",
        "conditions": list(summaries.keys()),
        "metrics": {},
    }
    
    # Extract key metrics for each condition
    for condition, summary in summaries.items():
        comparison["metrics"][condition] = {
            "bridge_mechanism": summary.get("bridge_mechanism", "unknown"),
            "test_success_rate": summary.get("test_success_rate", 0.0),
            "test_time_to_success_mean": summary.get("test_time_to_success_mean"),
            "episodes_with_any_nars_action_rate": summary.get("episodes_with_any_nars_action_rate", 0.0),
            "test_failures": summary.get("test_failures", {}),
            "reps": summary.get("reps", 0),
            "seed": summary.get("seed", 0),
        }
    
    # Calculate deltas (bridge_on vs bridge_off as baseline)
    if "bridge_off" in summaries and "bridge_on" in summaries:
        baseline = summaries["bridge_off"]
        treatment = summaries["bridge_on"]
        
        comparison["deltas"] = {
            "bridge_on_vs_bridge_off": {
                "success_rate_delta": treatment.get("test_success_rate", 0) - baseline.get("test_success_rate", 0),
                "action_rate_delta": treatment.get("episodes_with_any_nars_action_rate", 0) - baseline.get("episodes_with_any_nars_action_rate", 0),
            }
        }
    
    # Save comparison
    output_path = runs_dir / f"{stamp}_micro_gscan_compare.json"
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    
    print(f"Comparison saved: {output_path}")
    
    # Print summary table
    print("\n=== Micro-gSCAN Comparison ===")
    print(f"Timestamp: {stamp}\n")
    
    print(f"{'Condition':<25} {'Bridge':<10} {'Success':<10} {'NARS Action':<15} {'No-Action':<12} {'Wrong':<8} {'Timeout':<8}")
    print("-" * 100)
    
    for condition in ["bridge_off", "bridge_on", "randomized_embeddings"]:
        if condition not in summaries:
            continue
        s = summaries[condition]
        bridge = s.get("bridge_mechanism", "?")
        success = f"{s.get('test_success_rate', 0):.2%}"
        action_rate = f"{s.get('episodes_with_any_nars_action_rate', 0):.2%}"
        failures = s.get("test_failures", {})
        no_action = failures.get("no_action", 0)
        wrong = failures.get("wrong_object", 0)
        timeout = failures.get("timeout", 0)
        
        print(f"{condition:<25} {bridge:<10} {success:<10} {action_rate:<15} {no_action:<12} {wrong:<8} {timeout:<8}")
    
    if "deltas" in comparison:
        print("\n=== Deltas (bridge_on vs bridge_off) ===")
        d = comparison["deltas"]["bridge_on_vs_bridge_off"]
        print(f"Success rate delta: {d['success_rate_delta']:+.2%}")
        print(f"Action rate delta:  {d['action_rate_delta']:+.2%}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compare_micro_gscan.py <stamp> [condition1 condition2 ...]")
        print("Example: python compare_micro_gscan.py 20260106_compare bridge_off bridge_on randomized_embeddings")
        sys.exit(1)
    
    stamp = sys.argv[1]
    conditions = sys.argv[2:] if len(sys.argv) > 2 else ["bridge_off", "bridge_on", "randomized_embeddings"]
    
    generate_comparison(stamp, conditions)
