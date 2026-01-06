#!/usr/bin/env python3
"""Extract key metrics from 3-condition micro-gSCAN experiment."""

import json
import sys

def load_summary(path):
    with open(path) as f:
        return json.load(f)

def extract_metrics(summary):
    """Extract key metrics from a summary JSON."""
    failures = summary.get("test_failures", {})
    total_test = failures.get("wrong_object", 0) + failures.get("timeout", 0) + failures.get("no_action", 0)
    
    return {
        "success_rate": summary.get("test_success_rate", 0.0),
        "action_rate": summary.get("episodes_with_any_nars_action_rate", 0.0),
        "no_action_rate": failures.get("no_action", 0) / total_test if total_test > 0 else 0.0,
        "wrong_object_rate": failures.get("wrong_object", 0) / total_test if total_test > 0 else 0.0,
        "timeout_rate": failures.get("timeout", 0) / total_test if total_test > 0 else 0.0,
        "mean_steps": summary.get("test_time_to_success_mean", None),
        "no_action_count": failures.get("no_action", 0),
        "wrong_object_count": failures.get("wrong_object", 0),
        "timeout_count": failures.get("timeout", 0),
        "total_test_episodes": total_test
    }

def main():
    stamp = "mgscan_min5_seed0"
    conditions = {
        "bridge_off": f"runs/{stamp}_micro_gscan_bridge_off.summary.json",
        "bridge_on": f"runs/{stamp}_micro_gscan_bridge_on.summary.json",
        "randomized_embeddings": f"runs/{stamp}_micro_gscan_randomized_embeddings.summary.json"
    }
    
    results = {}
    for name, path in conditions.items():
        summary = load_summary(path)
        results[name] = extract_metrics(summary)
    
    # Print table header
    print("\n" + "="*80)
    print("MICRO-GSCAN MINIMAL EXPERIMENT RESULTS (5 reps, seed=0)")
    print("="*80)
    print(f"\n{'Condition':<25} {'Success%':<10} {'Action%':<10} {'NoAct%':<10} {'Wrong%':<10} {'Timeout%':<10}")
    print("-" * 80)
    
    # Print each condition
    for name in ["bridge_off", "bridge_on", "randomized_embeddings"]:
        m = results[name]
        print(f"{name:<25} {m['success_rate']*100:>8.1f}% {m['action_rate']*100:>8.1f}% {m['no_action_rate']*100:>8.1f}% "
              f"{m['wrong_object_rate']*100:>8.1f}% {m['timeout_rate']*100:>8.1f}%")
    
    # Print counts
    print("\n" + "-" * 80)
    print("RAW FAILURE COUNTS:")
    print("-" * 80)
    for name in ["bridge_off", "bridge_on", "randomized_embeddings"]:
        m = results[name]
        print(f"{name:<25} no_action={m['no_action_count']:<4} wrong_obj={m['wrong_object_count']:<4} "
              f"timeout={m['timeout_count']:<4} total_test={m['total_test_episodes']}")
    
    # Print deltas
    print("\n" + "="*80)
    print("DELTAS (comparing bridge_on vs baselines)")
    print("="*80)
    
    delta_off = results["bridge_on"]["action_rate"] - results["bridge_off"]["action_rate"]
    delta_rand = results["bridge_on"]["action_rate"] - results["randomized_embeddings"]["action_rate"]
    
    no_action_delta_off = results["bridge_off"]["no_action_rate"] - results["bridge_on"]["no_action_rate"]
    no_action_delta_rand = results["randomized_embeddings"]["no_action_rate"] - results["bridge_on"]["no_action_rate"]
    
    print(f"\nAction rate deltas:")
    print(f"  bridge_on - bridge_off:          {delta_off:+.4f} ({delta_off*100:+.1f}%)")
    print(f"  bridge_on - randomized:          {delta_rand:+.4f} ({delta_rand*100:+.1f}%)")
    
    print(f"\nNo-action reduction (baseline - bridge_on):")
    print(f"  bridge_off vs bridge_on:         {no_action_delta_off:+.4f} ({no_action_delta_off*100:+.1f}%)")
    print(f"  randomized vs bridge_on:         {no_action_delta_rand:+.4f} ({no_action_delta_rand*100:+.1f}%)")
    
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
