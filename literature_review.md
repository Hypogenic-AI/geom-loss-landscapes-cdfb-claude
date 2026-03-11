# Literature Review: Geometric Analysis of Loss Landscapes in Automated Theorem Proving

## Research Area Overview

This research investigates whether chain-of-thought (CoT) reasoning in neural theorem provers creates fundamentally different loss landscape geometry compared to direct proof generation. The hypothesis is that CoT produces smoother, more connected landscapes with wider basins of attraction, while direct proof generation exhibits sharper minima and more fragmented topology. This work sits at the intersection of three areas: (1) loss landscape geometry and optimization theory, (2) neural theorem proving, and (3) mode connectivity and landscape topology.

---

## Key Definitions

### Loss Landscape Geometry

**Definition (Loss Function).** For a neural network with parameters θ trained on dataset {(xᵢ, yᵢ)}ᵢ₌₁ᵐ: L(θ) = (1/m) Σᵢ ℓ(xᵢ, yᵢ; θ).

**Definition (ε-sharpness, Keskar et al. 2017, modified by Dinh et al. 2017).** For a non-negative loss L at minimum θ*, the ε-sharpness is: S(θ*, ε) = max_{θ' ∈ B₂(ε, θ*)} [L(θ') - L(θ*)] / [1 + L(θ*)]. Via second-order Taylor expansion at a critical point: S(θ*, ε) ≈ ‖H(θ*)‖₂ · ε² / [2(1 + L(θ*))].

**Definition (Volume ε-flatness, Dinh et al. 2017).** Given ε > 0 and minimum θ, C(L, θ, ε) is the largest connected set containing θ such that L(θ') < L(θ) + ε for all θ' ∈ C. The ε-flatness is vol(C(L, θ, ε)).

**Definition (SAM Objective, Foret et al. 2021).** L^SAM_S(w) = max_{‖ε‖_p ≤ ρ} L_S(w + ε). The SAM optimization problem seeks parameters with uniformly low loss in a neighborhood of radius ρ.

**Definition (Filter-wise Normalization, Li et al. 2018).** For direction vector d compatible with parameters θ, normalize each filter: dᵢ,ⱼ ← (dᵢ,ⱼ / ‖dᵢ,ⱼ‖) · ‖θᵢ,ⱼ‖, where dᵢ,ⱼ is the j-th filter of the i-th layer. This removes scale invariance artifacts.

### Mode Connectivity

**Definition (Linear Mode Connectivity, Frankle et al. 2020).** Two parameter configurations Θ_A, Θ_B are linearly mode connected if the loss along the linear interpolation (1-λ)Θ_A + λΘ_B remains low for all λ ∈ [0,1].

**Definition (Loss Barrier, Frankle et al. 2020).** Given Θ_A, Θ_B with L(Θ_A) ≈ L(Θ_B): barrier = max_{λ∈[0,1]} L((1-λ)Θ_A + λΘ_B) - ½(L(Θ_A) + L(Θ_B)). Zero barrier indicates convex interpolation.

**Definition (ε-connectivity, Kuditipudi et al. 2019).** Parameters θ^A and θ^B are ε-connected if there exists a continuous path π(t) with π(0) = θ^A, π(1) = θ^B, and L(f_{π(t)}) ≤ max{L(f_{θ^A}), L(f_{θ^B})} + ε for all t.

**Definition (ε-dropout stability, Kuditipudi et al. 2019).** A solution θ is ε-dropout stable if zeroing out up to half the hidden units per layer (with appropriate rescaling) increases the loss by at most ε.

**Definition (Permutation Symmetry).** For an MLP, applying permutation P to layer ℓ's outputs and P⁻¹ to layer ℓ+1's inputs produces a functionally equivalent network: W'_ℓ = PW_ℓ, b'_ℓ = Pb_ℓ, W'_{ℓ+1} = W_{ℓ+1}P⊤ (Ainsworth et al. 2023).

### Hessian Spectrum

**Definition (Layer Cushion, Kuditipudi et al. 2019).** μᵢ = min_{x∈S} ‖Aᵢφ(x^{i-1})‖ / (‖Aᵢ‖_F ‖φ(x^{i-1})‖). Measures how well each layer preserves signal magnitude.

**Definition (Noise Stability, Kuditipudi et al. 2019).** A network θ is ε-noise stable for ε = β · c · d^{3/2} · max_{x}(‖f_θ(x)‖) / (h_{min}^{1/2} · min_i(μᵢ · μ_{i→})). Smaller ε means more robust.

---

## Key Papers

### Paper 1: Visualizing the Loss Landscape of Neural Nets (Li et al., 2018)
- **Main Results**: (1) Filter normalization enables meaningful cross-architecture landscape comparison. (2) Skip connections prevent transition from convex to chaotic landscapes as depth increases. (3) Wider networks produce flatter minima. (4) Sharpness under filter normalization correlates with generalization error. (5) SGD trajectories lie in extremely low-dimensional subspaces (2D captures 40-90% of variation).
- **Proof Techniques**: Filter-normalized random direction visualization; Hessian eigenvalue computation; PCA-based trajectory analysis.
- **Relevance**: Provides the primary visualization methodology applicable to comparing CoT vs. direct proof generation landscapes. The finding that architectural choices (skip connections, width) fundamentally alter landscape geometry suggests that proof generation strategy (sequential CoT vs. one-shot) may similarly reshape the landscape.

### Paper 2: Sharpness-Aware Minimization (Foret et al., 2021)
- **Main Results**: PAC-Bayes bound: L_D(w) ≤ max_{‖ε‖₂≤ρ} L_S(w+ε) + h(‖w‖²₂/ρ²). SAM improves generalization by seeking flat minima. Introduces m-sharpness (per-example sharpness) as a stronger predictor of generalization.
- **Relevance**: SAM's formulation provides a concrete, trainable objective for controlling landscape geometry. Could be used to deliberately flatten the landscape for direct proof generation to test whether this closes the gap with CoT.

### Paper 3: Sharp Minima Can Generalize (Dinh et al., 2017)
- **Main Results**: (1) Volume ε-flatness is infinite for all ReLU network minima (Theorem 2). (2) Hessian spectral norm can be made arbitrarily large via α-scale transformations without changing the function (Theorem 4). (3) Multiple Hessian eigenvalues can be simultaneously inflated (Theorem 5).
- **Proof Techniques**: Exploits non-negative homogeneity of ReLU; uses Jacobian determinant analysis; Horn's inequalities for singular values.
- **Relevance**: CRITICAL methodological warning — naive sharpness comparisons between CoT and direct proof models may be misleading. Must use reparametrization-invariant measures or filter normalization.

### Paper 4: Git Re-Basin (Ainsworth et al., 2023)
- **Main Results**: (1) Neural network loss landscapes contain nearly a single basin after accounting for permutation symmetries. (2) Three algorithms for aligning models via permutation (activation matching, weight matching, straight-through estimation). (3) First demonstration of zero-barrier linear mode connectivity between independently trained ResNets. (4) Linear mode connectivity is an emergent property of SGD, not of architectures.
- **Relevance**: Provides methodology for testing whether CoT and direct proof models end up in the "same basin" modulo permutations, or in genuinely different landscape regions. The single-basin hypothesis, if it holds for theorem proving models, would suggest differences are more subtle than basin-level separation.

### Paper 5: Explaining Landscape Connectivity (Kuditipudi et al., 2019)
- **Main Results**: (1) Dropout-stable optima are connected via piecewise-linear paths with at most O(d) segments (Theorem 1). (2) Noise-stable networks are dropout-stable, hence connected. (3) Not all optima are connected — connectivity is a property of SGD-found solutions.
- **Proof Techniques**: Constructive path building via alternating top-layer interpolation and input weight modification; leverages convexity of loss in final layer weights.
- **Relevance**: Provides testable conditions (dropout stability, noise stability) that predict whether CoT and direct proof solutions will be mode-connected. If both are dropout-stable but direct proof solutions are less noise-stable, this would support the hypothesis of more fragmented topology.

### Paper 6: GPT-f (Polu & Sutskever, 2020)
- **Main Results**: (1) Transformer LMs can effectively generate individual proof steps for Metamath. (2) Pre-training on mathematical text (arXiv) significantly improves performance. (3) Iterative value function training improves proof search. (4) 56.22% proof rate on Metamath test set.
- **Training**: Conditional language modeling: GOAL → PROOFSTEP. Loss computed only on proofstep tokens. Best-first search with 32 sampled tactics per expansion.
- **Relevance**: Represents the step-by-step (sequential reasoning) paradigm for theorem proving. The per-step loss function creates dense training signal, potentially leading to smoother optimization landscapes.

### Paper 7: Baldur (First et al., 2023)
- **Main Results**: (1) LLMs can generate entire formal proofs in one shot without interactive proof search. (2) Proof repair model improves success rate. (3) 47.9% proof rate on Isabelle/HOL with 62B model. (4) Complementary to search-based methods (combined 65.7%).
- **Training**: Standard autoregressive loss on full proof text. Overfitting observed at 50-70K steps.
- **Relevance**: KEY paper for direct proof generation. The early overfitting suggests a sharper loss landscape with narrower basins. The complementarity with search-based methods suggests different landscape regions are explored.

### Paper 8: DeepSeek-Prover-V2 (Ren et al., 2025)
- **Main Results**: (1) Hybrid CoT + formal proof approach via subgoal decomposition. (2) Cold-start with synthetic CoT data + GRPO reinforcement learning. (3) Consistency reward prevents structural divergence between CoT and formal proof. (4) 88.9% on MiniF2F-test.
- **Training**: Two stages: supervised cold-start, then GRPO with binary + consistency rewards. The consistency reward penalizes misalignment between CoT decomposition and formal proof structure.
- **Relevance**: KEY paper for CoT approach. The consistency reward can be interpreted as a landscape-shaping mechanism that encourages smoother loss surfaces. Emergent specialization in smaller models (7B vs 671B) suggests capacity-dependent basin structure.

### Paper 9: HyperTree Proof Search (Lample et al., 2022)
- **Main Results**: (1) MCTS-inspired search for hypertree-structured proofs. (2) Online training creates non-stationary optimization where data distribution evolves with model capability. (3) 82.6% on Metamath with online training.
- **Relevance**: The online training paradigm fundamentally alters the loss landscape over time, creating a sequence of increasingly complex landscapes. This is relevant to understanding how CoT step-by-step learning dynamics differ from one-shot direct generation.

### Paper 10: Hessian Eigenvalue Density (Ghorbani et al., 2019)
- **Main Results**: Hessian spectrum has two components: (1) bulk near zero (continuous density) and (2) isolated outlier eigenvalues. Outliers correspond to per-class gradient directions. Number of outliers ≈ number of classes.
- **Relevance**: For theorem proving, "classes" could be proof strategies or tactic types. CoT (many intermediate steps) vs. direct (one output) may produce different outlier structures in the Hessian.

---

## Known Results (Prerequisite Theorems)

| Result | Source | Statement Summary | Used For |
|--------|--------|-------------------|----------|
| PAC-Bayes generalization bound | Foret et al. 2021 | L_D(w) ≤ max L_S(w+ε) + h(‖w‖²/ρ²) | Connecting sharpness to generalization |
| Dropout-stable optima connectivity | Kuditipudi et al. 2019 | ε-dropout stable solutions connected via O(d)-segment paths | Testing basin connectivity |
| Volume flatness uninformative | Dinh et al. 2017 | All ReLU network minima have infinite volume ε-flatness | Ruling out naive flatness measures |
| Hessian manipulability | Dinh et al. 2017 | ‖H(T_α(θ))‖₂ can be made arbitrarily large for any minimum | Requiring invariant measures |
| Permutation invariance conjecture | Entezari et al. 2021 | Most SGD solutions are LMC modulo permutation symmetries | Single-basin hypothesis |
| Single basin modulo symmetries | Ainsworth et al. 2023 | Empirically verified for ResNets on CIFAR-10 | Landscape topology analysis |
| Loss simplexes | Benton et al. 2021 | SGD solutions form connected low-loss volumes, not just paths | Richer topology than paths |
| Hessian bulk+outlier structure | Ghorbani et al. 2019 | Spectrum = continuous bulk near 0 + isolated outliers | Curvature analysis methodology |
| Depth-chaos transition | Li et al. 2018 | Deep networks without skip connections have chaotic landscapes | Architecture-landscape relationship |

---

## Proof Techniques in the Literature

### Loss Landscape Analysis Techniques
1. **Filter-normalized visualization** (Li et al. 2018): Plot f(α,β) = L(θ* + αδ + βη) with filter-normalized random directions δ,η. Gold standard for visual comparison.
2. **Hessian eigenvalue analysis** (Sagun 2017, Ghorbani 2019): Compute top-k eigenvalues via Lanczos iteration; full spectrum via stochastic trace estimation.
3. **Linear interpolation** with loss barrier measurement (Frankle 2020, Ainsworth 2023): Evaluate L((1-λ)Θ_A + λΘ_B) for λ ∈ [0,1].
4. **Permutation alignment** (Ainsworth 2023): Match units across models via activation matching, weight matching, or straight-through estimation before interpolation.
5. **PCA trajectory analysis** (Li et al. 2018): Project optimization trajectory onto top-2 PCA components to visualize low-dimensional training dynamics.

### Theorem Proving Training Techniques
1. **Step-by-step tactic prediction**: Train on GOAL → PROOFSTEP pairs; search via best-first or MCTS (GPT-f, HTPS).
2. **Whole-proof generation**: Train on THEOREM → PROOF pairs; sample & check without search (Baldur).
3. **CoT subgoal decomposition**: Generate natural language proof sketch, then formalize each subgoal (DeepSeek-Prover-V2).
4. **Online/iterative training**: Prove theorems, add successful proofs to training data, retrain (HTPS).
5. **RL with shaped rewards**: Binary verification reward + consistency reward for CoT-proof alignment (DeepSeek-Prover-V2).

---

## Related Open Problems

1. **Reparametrization-invariant sharpness measures**: Dinh et al. (2017) showed standard measures are manipulable. What is the right measure? (Wen et al. 2023 provides partial answers.)

2. **Does the single-basin hypothesis hold for autoregressive theorem provers?** Ainsworth et al. (2023) showed it for image classifiers. Does the sequential structure of proof generation create genuinely separate basins?

3. **How does output sequence length affect landscape geometry?** CoT produces longer sequences than direct generation. Is there a formal relationship between sequence length and landscape curvature?

4. **Non-stationary landscapes in online training**: HTPS's iterative training changes the data distribution. How does the landscape evolve? Are there phase transitions?

5. **Consistency rewards as landscape shapers**: DeepSeek-Prover-V2's consistency reward explicitly couples CoT structure to formal proof. How does this alter the loss landscape topology?

---

## Gaps and Opportunities

1. **No prior work directly compares loss landscapes of CoT vs. direct proof generation**. This is a completely open question. The closest work compares landscapes across architectures (Li et al. 2018) or training regimes (Keskar et al. 2017 on batch size), but not across output generation strategies for reasoning tasks.

2. **Theorem proving loss landscapes are unstudied**. All existing landscape analysis is on classification or generation tasks. Theorem proving has unique structure: verification is possible (binary correct/incorrect), output is structured (proof trees), and there are multiple valid proofs per theorem.

3. **The connection between reasoning depth and landscape geometry is unexplored**. CoT adds "depth" through the output sequence rather than through network layers. Whether this creates analogous landscape effects as architectural depth (Li et al. 2018's depth-chaos transition) is unknown.

4. **Complementarity between approaches is unexplained**. Baldur + Thor proves 65.7% vs. 47.9% and 57% individually. This suggests different approaches find solutions in different landscape regions, but this has not been verified through landscape analysis.

---

## Recommendations for Proof Strategy

### Recommended Experimental Approach
1. **Train matched models**: Train transformer models on the same theorem proving dataset with (a) step-by-step tactic prediction (CoT-like), (b) whole-proof generation (direct), and (c) CoT subgoal decomposition + proof generation.

2. **Measure landscape geometry**:
   - Filter-normalized 2D loss surface visualization around each trained model
   - Hessian top eigenvalue spectrum (bulk + outlier structure)
   - Loss barrier measurements between models (with and without permutation alignment)
   - Training trajectory PCA analysis

3. **Test connectivity**: Use Git Re-Basin permutation alignment to test whether CoT and direct models lie in the same basin or separate basins.

4. **Analyze training dynamics**: Track sharpness measures (SAM-style) throughout training for both approaches. Look for phase transitions or qualitative differences in convergence behavior.

### Key Lemmas to Establish
- Whether filter normalization is appropriate for autoregressive models (not just classifiers)
- Whether permutation symmetry arguments extend to decoder-only transformers
- Relationship between proof sequence length and effective landscape dimensionality

### Potential Obstacles
- Computational cost of Hessian analysis for large language models
- Dinh et al.'s reparametrization critique may undermine some sharpness comparisons
- Need to control for confounders: CoT and direct models may differ in many ways beyond landscape geometry
- Proof-checking provides binary signal; converting this to continuous landscape measures requires care
