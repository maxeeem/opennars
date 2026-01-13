import os
import json
import glob
from typing import Dict, Any

def load_latest_compare(domain: str) -> Dict[str, Any] | None:
    # Pattern: *_{domain}_compare_baseline_vs_ablate_bridge.json
    pattern = os.path.join("runs", f"*_{domain}_compare_baseline_vs_ablate_bridge.json")
    files = glob.glob(pattern)
    if not files:
        return None
    # Sort by time (filename usually starts with timestamp)
    files.sort(reverse=True)
    with open(files[0], 'r') as f:
        return json.load(f)

def print_summary(domain: str):
    data = load_latest_compare(domain)
    if not data:
        print(f"No data for {domain}")
        return

    print(f"\n=== SUMMARY FOR {domain.upper()} ===")
    
    base = data["baseline"]
    ablate = data["ablate_bridge"]
    
    # Extract metrics
    # We want to compare Accuracy (Balanced), None Rate, Difficulty
    
    def get_metrics(d):
        m = d.get("metrics", {})
        acc = m.get("accuracy_excl_none", {}).get("balanced", 0.0)
        none_rate = m.get("none_rate", 0.0)
        diff = m.get("difficulty_score", 0.0)
        return acc, none_rate, diff

    b_acc, b_none, b_diff = get_metrics(base)
    a_acc, a_none, a_diff = get_metrics(ablate)
    
    print(f"{'Metric':<20} | {'Baseline':<10} | {'Ablation':<10} | {'Delta':<10}")
    print("-" * 60)
    print(f"{'Balanced Acc':<20} | {b_acc:.4f}     | {a_acc:.4f}     | {b_acc-a_acc:+.4f}")
    print(f"{'None Rate':<20} | {b_none:.4f}     | {a_none:.4f}     | {b_none-a_none:+.4f}")
    print(f"{'Difficulty':<20} | {b_diff:.4f}     | {a_diff:.4f}     | {b_diff-a_diff:+.4f}")
    
    # Cosines
    print("\n[Geometry / Cosines]")
    # We can take avg same-class vs avg cross-class
    b_cos = base.get("cosines", {})
    a_cos = ablate.get("cosines", {})
    
    def get_avg_cos(c):
        same = (c.get("Te1_Tr1", 0) + c.get("Te2_Tr2", 0)) / 2
        cross = (c.get("Te1_Tr2", 0) + c.get("Te2_Tr1", 0)) / 2
        return same, cross
    
    b_same, b_cross = get_avg_cos(b_cos)
    a_same, a_cross = get_avg_cos(a_cos)
    
    print(f"Baseline: Same={b_same:.3f}, Cross={b_cross:.3f} -> Diff={b_cross-b_same:.3f}")
    print(f"Ablation: Same={a_same:.3f}, Cross={a_cross:.3f} -> Diff={a_cross-a_same:.3f}")

if __name__ == "__main__":
    for domain in ["water_wind", "shape"]:
        print_summary(domain)
