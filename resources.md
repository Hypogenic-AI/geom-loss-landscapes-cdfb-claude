# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project: **Geometric Analysis of Loss Landscapes in Automated Theorem Proving: Chain-of-Thought vs Direct Proof Generation**.

Resources span three interconnected areas: loss landscape geometry, neural theorem proving, and mode connectivity/topology.

## Papers
Total papers downloaded: 28

| # | Title | Authors | Year | File | Key Results |
|---|-------|---------|------|------|-------------|
| 1 | Visualizing the Loss Landscape of Neural Nets | Li et al. | 2018 | papers/li2018_visualizing_loss_landscape.pdf | Filter normalization; skip connections → flat landscapes |
| 2 | Sharpness-Aware Minimization | Foret et al. | 2021 | papers/foret2020_sharpness_aware_minimization.pdf | SAM objective; PAC-Bayes bound for sharpness |
| 3 | Sharp Minima Can Generalize | Dinh et al. | 2017 | papers/dinh2017_sharp_minima_can_generalize.pdf | Sharpness measures are reparametrization-sensitive |
| 4 | Hessian Eigenvalue Density | Ghorbani et al. | 2019 | papers/ghorbani2019_investigation_hessian.pdf | Bulk + outlier Hessian spectrum structure |
| 5 | Empirical Analysis of Hessian | Sagun et al. | 2017 | papers/sagun2017_empirical_hessian.pdf | Most Hessian eigenvalues cluster near zero |
| 6 | Emergent Local Geometry | Fort & Ganguli | 2019 | papers/fort2019_emergent_geometry.pdf | Local curvature properties vs architecture |
| 7 | Git Re-Basin | Ainsworth et al. | 2023 | papers/ainsworth2022_git_rebasin.pdf | Single basin modulo permutation symmetries |
| 8 | Permutation Invariance in LMC | Entezari et al. | 2021 | papers/entezari2021_permutation_invariance.pdf | Permutation invariance conjecture |
| 9 | Bridging Mode Connectivity | Zhao et al. | 2020 | papers/zhao2020_bridging_mode_connectivity.pdf | Mode connectivity ↔ adversarial robustness |
| 10 | Explaining Landscape Connectivity | Kuditipudi et al. | 2019 | papers/kuditipudi2019_explaining_connectivity.pdf | Dropout-stable optima are connected |
| 11 | Deep Ensembles Loss Landscape | Fort et al. | 2019 | papers/fort2019_deep_ensembles_landscape.pdf | Ensemble diversity from different basins |
| 12 | Loss Surface Simplexes | Benton et al. | 2021 | papers/benton2021_loss_surface_simplexes.pdf | Connected low-loss volumes |
| 13 | Taxonomizing Local vs Global | Simsek et al. | 2021 | papers/simsek2021_taxonomizing_local_global.pdf | Taxonomy of local/global landscape properties |
| 14 | Mode Connectivity Analysis | Gotmare et al. | 2018 | papers/gotmare2018_mode_connectivity_analysis.pdf | Mode connectivity for training analysis |
| 15 | GPT-f | Polu & Sutskever | 2020 | papers/polu2020_gpt_theorem_proving.pdf | Transformer LMs for step-by-step proving |
| 16 | HyperTree Proof Search | Lample et al. | 2022 | papers/lample2022_hypertree.pdf | MCTS for proof search; online training |
| 17 | Proof Artifact Co-training | Han et al. | 2021 | papers/han2021_proof_artifact_cotraining.pdf | PACT training methodology |
| 18 | Baldur | First et al. | 2023 | papers/first2023_baldur.pdf | Whole-proof generation (direct approach) |
| 19 | DeepSeek-Prover | Xin et al. | 2024 | papers/xin2024_deepseek_prover.pdf | Synthetic data for neural theorem proving |
| 20 | DeepSeek-Prover-V2 | Ren et al. | 2025 | papers/ren2025_deepseek_prover_v2.pdf | CoT subgoal decomposition + RL |
| 21 | LEGO-Prover | Xin et al. | 2023 | papers/xin2023_lego_prover.pdf | Growing skill library for proving |
| 22 | NaturalProver | Welleck et al. | 2022 | papers/jiang2022_naturalprover.pdf | Natural language proof generation |
| 23 | Understanding SAM | Andriushchenko & Flammarion | 2022 | papers/andriushchenko2022_understanding_sam.pdf | Why SAM works theoretically |
| 24 | Asymmetric Valleys | He et al. | 2019 | papers/he2019_asymmetric_valleys.pdf | Valleys are asymmetric, not just sharp/flat |
| 25 | Modern Sharpness & Generalization | Andriushchenko et al. | 2023 | papers/wen2023_modern_sharpness_generalization.pdf | Re-evaluation of sharpness-generalization |
| 26 | AlphaGeometry | Trinh et al. | 2024 | papers/trinh2024_alphageometry.pdf | Neuro-symbolic geometry proving |
| 27 | Survey: DL for Theorem Proving | Li et al. | 2024 | papers/lu2024_survey_deep_learning_theorem.pdf | Comprehensive survey |
| 28 | Formal Mathematical Reasoning | Yang et al. | 2024 | papers/lu2024_formal_mathematical_reasoning.pdf | Formal reasoning survey |

See papers/README.md for detailed descriptions.

## Prior Results Catalog

Key theorems and lemmas available for our analysis:

| Result | Source | Statement Summary | Used For |
|--------|--------|-------------------|----------|
| PAC-Bayes sharpness bound | Foret et al. 2021 | Population loss bounded by neighborhood-worst-case training loss + norm regularizer | Theoretical justification for sharpness-generalization link |
| Volume flatness is trivial | Dinh et al. 2017 | All ReLU minima have infinite volume ε-flatness | Ruling out naive flatness |
| Hessian norm is manipulable | Dinh et al. 2017 | Any minimum's Hessian norm can be made arbitrarily large via scale transforms | Requiring invariant measures |
| Dropout-stable connectivity | Kuditipudi et al. 2019 | ε-dropout-stable solutions connected via O(d)-segment piecewise-linear paths | Testing connectivity hypothesis |
| Noise stability → dropout stability | Kuditipudi et al. 2019 | Noise-stable ⇒ dropout-stable ⇒ mode-connected | Measurable condition for connectivity |
| Not all optima connected | Kuditipudi et al. 2019 | Counterexample: some optima are disconnected; connectivity is property of SGD solutions | Landscape can have disconnected regions |
| Single basin mod permutations | Ainsworth et al. 2023 | After permutation alignment, independently trained models are LMC | Methodology for basin analysis |
| LMC is emergent, not architectural | Ainsworth et al. 2023 | Counterexample where same architecture lacks LMC under different training | Training method determines connectivity |
| Filter norm correlates with gen. error | Li et al. 2018 | Under filter normalization, visual sharpness predicts test error | Visualization methodology |
| Depth without shortcuts → chaos | Li et al. 2018 | Deep networks without skip connections have chaotic loss surfaces | Architecture-landscape relationship |
| Hessian = bulk + outliers | Ghorbani et al. 2019 | Spectrum has continuous near-zero bulk + isolated large eigenvalues | Curvature analysis template |
| SGD trajectories are low-dimensional | Li et al. 2018 | 2 PCA components capture 40-90% of SGD trajectory variation | Trajectory analysis methodology |

## Computational Tools

| Tool | Purpose | Location | Notes |
|------|---------|----------|-------|
| PyTorch | Neural network training & Hessian computation | .venv (pip) | torch 2.10.0; core framework for training provers and computing landscape metrics |
| SymPy | Symbolic math for theoretical analysis | .venv (pip) | sympy 1.14.0; useful for algebraic manipulation of bounds |
| NumPy/SciPy | Numerical computation, eigenvalue analysis | .venv (pip) | For Hessian eigenvalue computation, interpolation experiments |
| Matplotlib | Loss landscape visualization | .venv (pip) | For 2D/3D loss surface plots, contour plots |
| NetworkX | Proof tree structure analysis | .venv (pip) | For analyzing proof tree topology differences between CoT and direct |
| loss-landscape (GitHub) | Li et al.'s visualization code | https://github.com/tomgoldstein/loss-landscape | Reference implementation of filter-normalized visualization |
| SAM (GitHub) | Foret et al.'s SAM optimizer | https://github.com/google-research/sam | Reference for sharpness-aware training |

## Resource Gathering Notes

### Search Strategy
1. Used paper-finder service with 6 targeted queries covering all three research pillars
2. Searched across Semantic Scholar with relevance scoring
3. Total unique papers found: 369; papers with relevance ≥ 2: 269
4. Selected top 28 papers spanning foundational theory, methodology, and application areas

### Selection Criteria
- Foundational papers establishing key definitions and theorems (Dinh, Foret, Li, Kuditipudi)
- Methodological papers providing analysis tools (Ainsworth, Ghorbani, Sagun)
- Neural theorem proving papers representing both CoT and direct approaches (GPT-f, Baldur, DeepSeek-Prover-V2, HTPS)
- Survey papers for context (Lu et al. 2024)
- Prioritized highly-cited papers and recent SOTA results

### Challenges Encountered
- The DeepSeek-Prover-V2 paper (2501.14333) was initially the wrong paper; re-downloaded with correct arXiv ID (2504.21801)
- No prior work directly compares loss landscapes between CoT and direct proof generation — this is genuinely novel territory
- The intersection of loss landscape theory and theorem proving is essentially unexplored

## Recommendations for Proof Construction

### 1. Experimental Design
- **Train matched pairs**: Same architecture, same data, different generation strategy (step-by-step vs. whole-proof vs. CoT+subgoal)
- **Control variables**: Model size, training data, number of training steps, optimizer
- **Use Lean 4 or Isabelle/HOL** as the formal verification backend (best tooling available)

### 2. Measurement Protocol
- **Sharpness**: Use filter-normalized visualization (Li et al.) AND SAM-style neighborhood loss (Foret et al.) — avoid relying on raw Hessian norm alone (Dinh et al. warning)
- **Connectivity**: Use Git Re-Basin permutation alignment (Ainsworth et al.) to test whether CoT and direct models lie in the same or different basins
- **Hessian spectrum**: Compute top-k eigenvalues + bulk density to characterize curvature structure
- **Training dynamics**: Track sharpness throughout training; compare convergence curves

### 3. Key Hypotheses to Test (derived from literature)
- H1: CoT models converge to flatter minima (wider basins) than direct proof models
- H2: CoT and direct models are NOT linearly mode connected (even after permutation alignment), indicating genuinely different landscape regions
- H3: CoT training trajectories explore a lower-dimensional subspace than direct proof training
- H4: The Hessian spectrum of CoT models has fewer/smaller outlier eigenvalues than direct models
- H5: Applying SAM to direct proof training closes the generalization gap with CoT

### 4. Potential Difficulties
- Hessian computation is expensive for large models; may need to use smaller models or approximate methods
- Reparametrization invariance concerns (Dinh et al.) require careful experimental design
- Need to isolate the effect of generation strategy from other confounders (data, architecture, training recipe)
- Binary proof verification signal makes continuous landscape analysis challenging
