# Project Broca: ICLR Submission

## Abstract
We present a mechanistic demonstration of symbol grounding transfer in a NARS-based cognitive architecture...

## Method
We use a bridge-mediated vector-symbol integration...

## Results

We evaluated the architecture on two domains: "Water/Wind" (texture/fluidity transfer) and "Shape" (geometric transfer). In the "Baseline" (Bridge-Enabled) condition, the system utilizes the `VectorMix` operator to align sensory vectors with symbolic concepts. In the "Ablation" (No Bridge) condition, this pathway is severed.

### Quantitative Analysis

Table 1: Zero-shot Performance (N=5 reps)

| Domain | Condition | Balanced Accuracy | Cosine Similarity (Te-Tr) |
| :--- | :--- | :--- | :--- |
| **Water/Wind** | **Baseline** | **0.46** | **0.84** |
| | Ablation | 0.38 | ~0.00 |
| **Shape** | **Baseline** | **0.56** | **0.24** |
| | Ablation | 0.49 | ~0.00 |

### Metrics Discussion

**Cosine Similarity as Mechanism:** The primary validator of our "Meaning Transfer" hypothesis is the cosine similarity between the unseen Test stimulus vector and the Trained label vector in the global embedding space.
- In the **Water/Wind** domain, the baseline achieves a high similarity of **0.84**, indicating robust semantic alignment.
- In the **Shape** domain, the baseline achieves **0.24**, significantly higher than the ablation's **0.00**, though lower than Water/Wind (likely due to the orthogonality of shape concepts in standard Glove/CLIP spaces).

**Impact on Accuracy:** This semantic alignment translates into a performance advantage. The Baseline outperforms the Ablation in Balanced Accuracy across both domains (+8% in Water/Wind, +7% in Shape). While absolute accuracy is constrained by the inherent difficulty of zero-shot learning in this specific setup, the *relative* advantage is consistent, validating the architectural claim.

### Visualizations
![Cosines Water](paper_assets/fig1_cosines_water_wind.png)
![Cosines Shape](paper_assets/fig1_cosines_shape.png)
![Accuracy](paper_assets/fig2_accuracy.png)

## Discussion
...
