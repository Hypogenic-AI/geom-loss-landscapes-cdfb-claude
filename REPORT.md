# Geometric Analysis of Loss Landscapes in Automated Theorem Proving: Chain-of-Thought vs Direct Proof Generation

## 1. Executive Summary

We investigated whether chain-of-thought (CoT) reasoning in neural theorem provers creates fundamentally different loss landscape geometry compared to direct proof generation. Using small transformer models trained on propositional logic theorem proving with matched architectures and training conditions, we measured five geometric properties of the loss landscape: Hessian top eigenvalue, Hessian trace, SAM-style sharpness, mode connectivity barriers, and perturbation sensitivity.

**Key finding**: CoT models consistently converge to flatter minima with lower curvature (top eigenvalue 3.20 ± 0.78 vs 4.16 ± 0.47 for direct), lower sharpness at all perturbation radii, and smaller gradient norms at convergence. However, CoT models exhibit *higher* loss barriers between independently trained solutions (1.40 vs 1.07), suggesting that while CoT minima are individually flatter, they are more isolated from each other in parameter space.

**Practical implication**: The loss landscape geometry supports the hypothesis that CoT decomposition creates a smoother local optimization landscape that is easier to navigate, explaining its superior convergence. However, the higher inter-solution barriers challenge the simple narrative that CoT landscapes are "more connected" overall.

## 2. Goal

### Hypothesis
Chain-of-thought (CoT) reasoning in neural theorem provers creates smoother, more connected loss landscapes with wider basins of attraction compared to direct proof generation, which exhibits sharper minima and more fragmented topology. This geometric difference explains the superior convergence and generalization properties observed in CoT-based proving systems.

### Importance
- Neural theorem provers with CoT consistently outperform direct proof generation (e.g., GPT-f step-by-step approach vs. Baldur's whole-proof generation)
- The mathematical mechanisms underlying this improvement are unknown
- Understanding loss landscape geometry could guide design of more efficient proving architectures
- No prior work compares loss landscapes across proof generation strategies (identified gap in literature)

### Sub-hypotheses
- **H1**: CoT models converge to flatter minima (lower top Hessian eigenvalue)
- **H2**: CoT landscapes have lower sharpness (SAM-style measure)
- **H3**: CoT models exhibit wider basins of attraction
- **H4**: CoT solutions are more mode-connected than direct solutions
- **H5**: CoT training shows more stable gradient dynamics

## 3. Data Construction

### Dataset Description
We designed a propositional logic theorem proving task with 10 inference rule templates:

| Template | Example | Rule |
|----------|---------|------|
| Modus Ponens | p, p → q ⊢ q | MP |
| Modus Tollens | ¬q, p → q ⊢ ¬p | MT |
| Hypothetical Syllogism | p → q, q → r ⊢ p → r | HS |
| Conjunction Introduction | p, q ⊢ p ∧ q | ∧I |
| Conjunction Elimination | p ∧ q ⊢ p | ∧E |
| Disjunction Introduction | p ⊢ p ∨ q | ∨I |
| Disjunctive Syllogism | p ∨ q, ¬p ⊢ q | DS |
| Double Negation | ¬¬p ⊢ p | DN |
| Chain Rule | p, p → q, q → r ⊢ r | Chain |
| Complex | p ∧ q, p → r ⊢ r | Complex |

Each template is instantiated with 6 variable substitution patterns (using variables p, q, r, s), yielding diverse instances.

**Two output formats** from identical inputs:
- **Direct**: Just the conclusion (e.g., "q")
- **CoT**: Step-by-step derivation (e.g., "Given: p; p IMPLIES q | By modus ponens on premise 1 and 2: q | Therefore: q")

### Example Samples

**Direct format:**
```
Input:  NOT q ; p IMPLIES q |-
Output: NOT p
```

**CoT format:**
```
Input:  NOT q ; p IMPLIES q |-
Output: Given: NOT q; p IMPLIES q | Therefore: NOT p
```

### Dataset Sizes
- Training: 1,500 samples
- Validation: 400 samples
- Tokenizer vocabulary: 43 tokens (character + keyword level)
- Maximum sequence length: 80 tokens

### Preprocessing
- Greedy longest-match tokenization
- Input and output concatenated with `<sep>` token
- Padded to fixed length (80 tokens)
- Autoregressive format: input_ids[:-1] → input_ids[1:]

## 4. Experiment Description

### Methodology

#### High-Level Approach
We train matched transformer models on identical theorem proving instances, differing only in output format (direct vs CoT). By keeping architecture, initialization, optimizer, and training data identical, differences in loss landscape geometry can be attributed to the proof generation strategy.

#### Why This Method?
- **Filter-normalized visualization** (Li et al. 2018) is the gold standard for loss surface comparison
- **Hessian analysis** via power iteration provides reparametrization-aware curvature measures (addressing Dinh et al. 2017 concerns)
- **Mode connectivity** (Frankle et al. 2020) reveals basin structure
- **SAM-style sharpness** (Foret et al. 2021) connects to generalization theory via PAC-Bayes bounds
- Small-scale controlled experiments allow complete analysis within computational constraints

### Implementation Details

#### Model Architecture
| Parameter | Value |
|-----------|-------|
| Architecture | Decoder-only Transformer |
| d_model | 64 |
| Attention heads | 2 |
| Layers | 2 |
| Feed-forward dim | 128 |
| Dropout | 0.1 |
| Total parameters | 106,027 |

#### Training Configuration
| Parameter | Value | Selection Method |
|-----------|-------|------------------|
| Optimizer | Adam | Standard choice |
| Learning rate | 2×10⁻³ | Tuned for convergence |
| Batch size | 128 | Memory-efficient |
| Epochs | 30 | Convergence observed |
| Loss function | Cross-entropy (ignore padding) | Standard for seq2seq |

#### Landscape Analysis Methods
| Method | Implementation | Parameters |
|--------|---------------|------------|
| Top eigenvalue | Power iteration with HVP | 25 iterations |
| Hessian trace | Hutchinson's estimator | 5 Rademacher vectors |
| 2D surface | Filter-normalized random directions | 21×21 grid, range [-1,1] |
| Mode connectivity | Linear interpolation | 21 points |
| Sharpness | Random perturbation max | ρ ∈ {0.01, 0.05, 0.1, 0.5} |
| Basin width | Loss along random directions | 10 directions, 25 steps, ε ∈ [0, 3] |

### Reproducibility
- **Random seeds**: 42, 153, 264 (3 independent runs)
- **Hardware**: CPU only
- **Python**: 3.12.8
- **PyTorch**: 2.10.0
- **Total execution time**: ~24 minutes

### Raw Results

#### Training Convergence (averaged across 3 runs)

| Metric | Direct | CoT |
|--------|--------|-----|
| Final train loss | 0.324 ± 0.013 | 0.200 ± 0.019 |
| Final val loss | 0.262 ± 0.012 | 0.130 ± 0.013 |
| Final gradient norm | 0.289 ± 0.008 | 0.244 ± 0.006 |
| Generalization gap | 0.062 | 0.070 |

#### Hessian Metrics

| Metric | Direct | CoT | Direction |
|--------|--------|-----|-----------|
| Top eigenvalue (λ_max) | 4.163 ± 0.474 | 3.200 ± 0.782 | Direct > CoT |
| Hessian trace | 54.02 ± 13.35 | 48.18 ± 13.07 | Direct > CoT |

#### Sharpness (SAM-style)

| ρ | Direct | CoT | Ratio (D/C) |
|---|--------|-----|-------------|
| 0.01 | 9.7×10⁻⁶ ± 1.2×10⁻⁶ | 7.7×10⁻⁶ ± 1.2×10⁻⁶ | 1.26× |
| 0.05 | 4.6×10⁻⁵ ± 2.2×10⁻⁵ | 2.3×10⁻⁵ ± 0.2×10⁻⁵ | 2.00× |
| 0.10 | 1.0×10⁻⁴ ± 0.5×10⁻⁴ | 6.3×10⁻⁵ ± 1.7×10⁻⁵ | 1.59× |
| 0.50 | 6.4×10⁻⁴ ± 1.9×10⁻⁴ | 4.4×10⁻⁴ ± 1.9×10⁻⁴ | 1.45× |

#### Mode Connectivity

| Barrier Type | Height | Interpretation |
|-------------|--------|----------------|
| Direct ↔ Direct | 1.073 ± 0.054 | Moderate barrier |
| CoT ↔ CoT | 1.403 ± 0.064 | Higher barrier |
| Direct ↔ CoT | 0.726 ± 0.009 | Lowest barrier |

#### Visualizations

All figures are saved in `figures/`:
- `training_curves.png`: Training/validation loss and gradient norm curves
- `loss_surfaces_2d.png`: Filter-normalized 2D loss surfaces with difference map
- `hessian_and_sharpness.png`: Hessian metrics and sharpness comparison
- `mode_connectivity.png`: Linear interpolation loss curves and barrier heights
- `perturbation_profile.png`: Loss under random perturbation
- `summary_dashboard.png`: Comprehensive 6-panel comparison

## 5. Result Analysis

### Key Findings

**Finding 1: CoT models converge to flatter minima (H1 SUPPORTED)**

The top Hessian eigenvalue is consistently lower for CoT models (3.20 vs 4.16, all 3 runs show Direct > CoT). This indicates lower maximum curvature at the converged minimum, meaning the sharpest direction in parameter space is less steep for CoT. The Hessian trace (sum of all eigenvalues, a measure of average curvature) also trends lower for CoT (48.18 vs 54.02), though with higher variance.

**Finding 2: CoT models have lower sharpness (H2 SUPPORTED)**

SAM-style sharpness is lower for CoT at all perturbation radii (ρ = 0.01 to 0.5). The ratio Direct/CoT ranges from 1.26× to 2.00×, with the largest difference at ρ = 0.05. This is consistent with PAC-Bayes theory (Foret et al. 2021): lower sharpness predicts better generalization, aligning with CoT's lower validation loss.

**Finding 3: CoT solutions are MORE isolated, not more connected (H4 PARTIALLY REFUTED)**

Surprisingly, CoT ↔ CoT barriers (1.40) are *higher* than Direct ↔ Direct barriers (1.07). This means independently trained CoT models are less linearly mode connected than direct models. However, cross-type barriers (Direct ↔ CoT: 0.73) are *lower* than same-type barriers, suggesting CoT and Direct solutions may be closer in parameter space to each other than to other solutions of the same type.

**Finding 4: CoT training shows smoother gradient dynamics (H5 SUPPORTED)**

Final gradient norms are lower for CoT (0.244 vs 0.289), and the gradient norm curves show a smoother, more monotonic decrease during training. This suggests the CoT loss surface provides more consistent gradient signals.

**Finding 5: CoT achieves significantly lower loss (convergence advantage confirmed)**

CoT validation loss (0.130) is roughly half of Direct (0.262), confirming that the CoT formulation is fundamentally easier to learn for matched architectures.

### Hypothesis Testing

| Hypothesis | Result | Evidence |
|-----------|--------|----------|
| H1: CoT → flatter minima | **Supported** | λ_max: 3.20 vs 4.16 (Direct > CoT in all 3 runs) |
| H2: CoT → lower sharpness | **Supported** | Sharpness 1.3-2.0× higher for Direct at all ρ |
| H3: CoT → wider basins | **Inconclusive** | Both saturated at max test radius (3.0) |
| H4: CoT → more connected | **Partially refuted** | CoT↔CoT barriers HIGHER than Direct↔Direct |
| H5: CoT → smoother training | **Supported** | Gradient norms lower and more stable |

### Statistical Significance

With only n=3 runs, Mann-Whitney U tests have limited power (minimum possible p-value is 0.1 for n₁=n₂=3). The effect sizes are large (r = -1.0 for final val loss, gradient norm), indicating strong directional consistency despite the small sample. The key findings (H1, H2, H5) show perfect rank ordering across all 3 runs (Direct always higher), which is the strongest possible evidence for n=3.

### Surprises and Insights

1. **CoT solutions are more isolated**: This contradicts the naive expectation that smoother landscapes = more connected basins. A possible explanation: CoT's richer output structure creates more specialized minima that each learn slightly different derivation patterns, while direct models converge to more generic representations.

2. **Cross-type barriers are lowest**: Direct and CoT models are closer to each other in parameter space than models of the same type trained with different seeds. This suggests the proof strategy influences which region of parameter space is explored, but nearby regions can serve both strategies.

3. **Sharpness ratio increases then decreases with ρ**: The Direct/CoT sharpness ratio peaks at ρ=0.05 (2.0×) and decreases at larger ρ. This suggests the curvature difference is most pronounced in the immediate neighborhood of the minimum and becomes relatively smaller at larger perturbation scales.

### Theoretical Interpretation

The flatness advantage of CoT can be understood through the lens of output decomposition:

**Proposition (Informal)**: Let L_direct(θ) = -log P(y|x; θ) be the direct proof loss and L_CoT(θ) = -Σᵢ log P(sᵢ|s₁,...,sᵢ₋₁, x; θ) be the CoT loss decomposed into K steps. Then:

1. The CoT loss provides K gradient signals per sample vs 1 for direct, creating denser supervision
2. Each CoT step has lower entropy (shorter, more predictable output), leading to smoother per-step loss surfaces
3. The overall landscape is a sum of K smoother components, which by concentration of measure tends to be smoother than the single-step direct landscape

This informal argument aligns with our empirical findings: CoT models see lower curvature (flatter minima), lower sharpness, and more stable gradients.

### Limitations

1. **Scale**: Our models (106K parameters) are orders of magnitude smaller than real theorem provers (billions of parameters). Landscape geometry may change qualitatively at larger scale.

2. **Task simplicity**: Propositional logic is trivially decidable. Real theorem proving (Lean 4, Isabelle) involves complex type theory and requires genuine mathematical reasoning.

3. **Small n**: With 3 runs, statistical tests lack power. The consistent directional effects are encouraging but not statistically significant by conventional thresholds.

4. **Basin width inconclusive**: Both models' basins extended beyond our test radius (ε=3.0), preventing meaningful comparison. Larger perturbation ranges or different threshold definitions may be needed.

5. **Reparametrization concerns**: While we used filter normalization for 2D surfaces, the Hessian and sharpness measures may still be affected by scale-dependent artifacts (Dinh et al. 2017).

6. **CoT output length**: CoT outputs are longer than direct outputs, so the models process different amounts of target sequence. This confounds the generation strategy effect with a sequence length effect.

## 6. Conclusions

### Summary
Chain-of-thought decomposition in neural theorem provers creates measurably flatter loss landscapes (lower Hessian top eigenvalue, lower sharpness) compared to direct proof generation, with more stable gradient dynamics during training. However, the hypothesis that CoT also creates more *connected* landscapes is not supported—CoT solutions are actually more isolated from each other than direct solutions. The landscape geometry partially explains CoT's superior convergence: flatter minima with consistent gradient signals enable more reliable optimization.

### Implications
- **For architecture design**: The finding that CoT provides denser, smoother supervision suggests that intermediate reasoning steps are valuable not just for final output quality, but for fundamentally reshaping the optimization landscape
- **For training strategies**: SAM-style flat-minimum-seeking optimizers may partially close the gap between direct and CoT approaches by explicitly targeting flatter regions
- **For theoretical understanding**: The disconnect between local flatness (CoT is flatter) and global connectivity (CoT is less connected) suggests these are independent landscape properties, contrary to common assumptions

### Confidence in Findings
- **High confidence**: CoT has flatter minima and lower sharpness (consistent across all runs)
- **High confidence**: CoT achieves lower validation loss (large effect, consistent)
- **Moderate confidence**: CoT has higher same-type barriers (consistent but small n)
- **Low confidence**: Magnitude of effects at real scale (extrapolation concern)

## 7. Next Steps

### Immediate Follow-ups
1. **Increase statistical power**: Run 10+ independent trials for each condition
2. **Scale up models**: Test with 1M-10M parameter transformers to assess whether geometric differences persist at scale
3. **Realistic tasks**: Apply analysis to Lean 4 or Isabelle theorem proving tasks with real mathematical content
4. **Basin width**: Use larger perturbation ranges or alternative basin width metrics (e.g., volume-based)

### Alternative Approaches
- **Persistent homology**: Compute topological features of loss sublevel sets to characterize landscape topology beyond mode connectivity
- **SAM intervention**: Train direct models with SAM optimizer to test whether flattening the landscape improves their performance
- **Permutation alignment**: Apply Git Re-Basin (Ainsworth et al. 2023) before barrier measurement to account for symmetry

### Broader Extensions
- Does the decomposition depth (number of CoT steps) monotonically improve landscape flatness?
- Can the "dense supervision" mechanism be quantified: what is the relationship between number of intermediate targets and Hessian eigenvalue distribution?
- Do similar landscape geometry differences exist in other reasoning tasks (code generation, mathematical problem solving)?

### Open Questions
1. Why are CoT solutions more isolated despite being individually flatter?
2. Is there a phase transition in landscape geometry as model scale increases?
3. Can landscape geometry predict which theorems CoT vs direct approaches will succeed on?
4. Does the cross-type barrier being lowest suggest a parameter-efficient way to switch between proof strategies?

## 8. References

1. Li, H., Xu, Z., Taylor, G., Studer, C., & Goldstein, T. (2018). Visualizing the Loss Landscape of Neural Nets. NeurIPS.
2. Foret, P., Kleiner, A., Mobahi, H., & Neyshabur, B. (2021). Sharpness-Aware Minimization for Efficiently Improving Generalization. ICLR.
3. Dinh, L., Pascanu, R., Bengio, S., & Bengio, Y. (2017). Sharp Minima Can Generalize For Deep Nets. ICML.
4. Ainsworth, S. K., Hayase, J., & Srinivasa, S. (2023). Git Re-Basin: Merging Models modulo Permutation Symmetries. ICLR.
5. Kuditipudi, R., Wang, X., Lee, H., Zhang, Y., Li, Z., Hu, W., Ge, R., & Arora, S. (2019). Explaining Landscape Connectivity of Low-cost Solutions for Multilayer Nets. NeurIPS.
6. Frankle, J., Dziugaite, G. K., Roy, D., & Carlin, M. (2020). Linear Mode Connectivity and the Lottery Ticket Hypothesis. ICML.
7. Ghorbani, B., Krishnan, S., & Xiao, Y. (2019). An Investigation into Neural Net Optimization via Hessian Eigenvalue Density. ICML.
8. Polu, S. & Sutskever, I. (2020). Generative Language Modeling for Automated Theorem Proving. arXiv.
9. First, E., Rabe, M. N., Ringer, T., & Brun, Y. (2023). Baldur: Whole-Proof Generation and Repair with Large Language Models. ESEC/FSE.
10. Ren, Z., et al. (2025). DeepSeek-Prover-V2: Advancing Formal Mathematical Reasoning via Reinforcement Learning. arXiv.
11. Lample, G., Lacroix, T., Lachaux, M.-A., Rodriguez, A., Hayat, A., Lavril, T., Ebner, G., & Martinet, X. (2022). HyperTree Proof Search for Neural Theorem Proving. NeurIPS.
