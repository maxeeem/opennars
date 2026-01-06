import os
import json
import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

OUTPUT_DIR = "docs/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_latest_compare(domain):
    pattern = os.path.join("runs", f"*_{domain}_compare_baseline_vs_ablate_bridge.json")
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(reverse=True)
    with open(files[0], 'r') as f:
        return json.load(f)

def make_bar_chart(data, metric_key, title, filename, ylabel):
    domains = list(data.keys())
    baseline_vals = [data[d]["baseline"] for d in domains]
    ablate_vals = [data[d]["ablate"] for d in domains]
    
    x = np.arange(len(domains))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(6, 4))
    rects1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline', color='skyblue')
    rects2 = ax.bar(x + width/2, ablate_vals, width, label='Ablation', color='salmon')
    
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(domains)
    ax.legend()
    
    ax.set_ylim(0, 1.1)
    
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"Saved {filename}")

def main():
    domains = ["water_wind", "shape"]
    results = {}
    
    for d in domains:
        res = load_latest_compare(d)
        if not res:
            print(f"Skipping {d} (no data)")
            continue
        results[d] = res
        
    if not results:
        return

    # Accuracy
    acc_data = {}
    for d, res in results.items():
        acc_data[d] = {
            "baseline": res["baseline"]["metrics"]["accuracy_excl_none"]["balanced"],
            "ablate": res["ablate_bridge"]["metrics"]["accuracy_excl_none"]["balanced"]
        }
    make_bar_chart(acc_data, "acc", "Balanced Accuracy (Transfer)", "fig2_accuracy.png", "Accuracy")
    
    # None Rate
    none_data = {}
    for d, res in results.items():
        none_data[d] = {
            "baseline": res["baseline"]["metrics"]["none_rate"],
            "ablate": res["ablate_bridge"]["metrics"]["none_rate"]
        }
    make_bar_chart(none_data, "none", "None Rate (Silence)", "fig3_none_rate.png", "Rate")
    
    # Cosine Matrices (Heatmaps)
    # We want 4 subplots: 2 domains * (Baseline, Ablation)
    # Actually, simpler: one figure per domain
    
    for d, res in results.items():
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        conditions = ["baseline", "ablate_bridge"]
        titles = ["Baseline", "Ablation"]
        
        for idx, cond in enumerate(conditions):
            ax = axes[idx]
            cosines = res[cond]["cosines"]
            # Matrix 2x2: Test1, Test2 vs Train1, Train2
            # Rows: Te1, Te2
            # Cols: Tr1, Tr2
            
            matrix = np.array([
                [cosines.get("Te1_Tr1", 0), cosines.get("Te1_Tr2", 0)],
                [cosines.get("Te2_Tr1", 0), cosines.get("Te2_Tr2", 0)]
            ])
            
            im = ax.imshow(matrix, cmap="coolwarm", vmin=-0.2, vmax=1.0)
            ax.set_title(f"{d.upper()} - {titles[idx]}")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Tr1", "Tr2"])
            ax.set_yticklabels(["Te1", "Te2"])
            
            # Annotate
            for i in range(2):
                for j in range(2):
                    text = ax.text(j, i, f"{matrix[i, j]:.2f}",
                                   ha="center", va="center", color="black")
                                   
        plt.tight_layout()
        plt.colorbar(im, ax=axes.ravel().tolist())
        filename = f"fig1_cosines_{d}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, filename))
        plt.close()
        print(f"Saved {filename}")

if __name__ == "__main__":
    main()
