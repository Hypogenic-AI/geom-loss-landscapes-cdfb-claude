# Research Plan: Geometric Analysis of Loss Landscapes in ATP

## Motivation & Novelty Assessment

### Why This Research Matters
Neural theorem provers with chain-of-thought (CoT) decomposition consistently outperform direct proof generation, but the mathematical mechanisms are unknown. Understanding loss landscape geometry could explain this gap and guide architecture design for formal reasoning systems.

### Gap in Existing Work
From the literature review: "No prior work directly compares loss landscapes of CoT vs. direct proof generation" and "Theorem proving loss landscapes are unstudied." All existing landscape analysis covers classification/generation tasks. The intersection of loss landscape theory and theorem proving is essentially unexplored.

### Our Novel Contribution
We provide the first systematic comparison of loss landscape geometry between CoT and direct proof generation strategies, using:
1. A controlled experimental framework with matched models on simplified theorem proving
2. Comprehensive geometric analysis (Hessian spectra, mode connectivity, basin width)
3. A theoretical framework connecting proof decomposition depth to landscape smoothness

### Experiment Justification
- **Exp 1 (Landscape Visualization)**: Filter-normalized 2D loss surfaces reveal qualitative landscape structure differences
- **Exp 2 (Hessian Spectrum)**: Eigenvalue distributions quantify curvature — key to understanding basin geometry
- **Exp 3 (Mode Connectivity)**: Loss barrier measurements test whether approaches find solutions in the same or different landscape regions
- **Exp 4 (Basin Width)**: Random perturbation analysis measures robustness of minima
- **Exp 5 (Training Dynamics)**: Gradient norm and loss trajectory analysis reveals convergence properties

## Research Question
Does chain-of-thought decomposition in neural theorem provers create fundamentally different loss landscape geometry compared to direct proof generation, and can these geometric differences explain the observed performance gap?

## Hypothesis Decomposition
- **H1**: CoT models converge to flatter minima (larger top Hessian eigenvalue ratio) than direct models
- **H2**: CoT and direct models have different mode connectivity properties (higher loss barriers between direct model solutions)
- **H3**: CoT models exhibit wider basins of attraction (loss increases more slowly under random perturbation)
- **H4**: The Hessian spectrum of CoT models has fewer/smaller outlier eigenvalues
- **H5**: CoT training trajectories show more stable gradient norms

## Methodology

### Task Design
Propositional logic theorem proving with controllable complexity:
- Variables: p, q, r, s (4 propositional variables)
- Connectives: AND, OR, NOT, IMPLIES
- Task: Given premises, derive a conclusion
- Ground truth available via truth tables

### Two Training Modes
1. **Direct**: Input = premises → Output = conclusion (single step)
2. **CoT**: Input = premises → Output = step1; step2; ...; conclusion (multi-step derivation)

### Model Architecture
- Small transformer (2-4 layers, 64-128 dim, 2-4 heads)
- Same architecture for both modes
- Differences only in target sequence format

### Landscape Analysis Pipeline
1. Train both models to convergence
2. Compute filter-normalized 2D loss surfaces (Li et al. 2018)
3. Compute top-k Hessian eigenvalues via power iteration
4. Measure loss barriers along linear interpolation paths
5. Estimate basin width via random perturbation
6. Track training dynamics (loss, gradient norm, per-step sharpness)

### Evaluation Metrics
- Hessian top eigenvalue and eigenvalue ratio (top/median)
- Loss barrier height between independently trained models
- Basin width at ε = {0.1, 0.5, 1.0} above minimum
- Filter-normalized sharpness (Li et al. 2018)
- Training loss convergence rate

### Statistical Analysis
- 5 independent runs per condition (different random seeds)
- Mann-Whitney U test for comparing distributions (non-parametric, small n)
- Report medians, IQR, and effect sizes

## Timeline (within 1 hour)
- Phase 1 Planning: 5 min (this document)
- Phase 2 Implementation: 25 min (data generation, model, training, metrics)
- Phase 3 Experiments: 15 min (run all experiments)
- Phase 4 Analysis: 5 min
- Phase 5-6 Documentation: 10 min

## Potential Challenges
- Small model/task may not capture all phenomena of large-scale provers
- CPU-only constraint limits model size and number of runs
- Binary proof verification vs. continuous loss landscape tension
- Need to ensure CoT vs direct differences aren't artifacts of sequence length
