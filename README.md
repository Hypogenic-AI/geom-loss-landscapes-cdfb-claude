# Geometric Analysis of Loss Landscapes: CoT vs Direct Proof Generation

## Overview
This project investigates how chain-of-thought (CoT) reasoning affects the loss landscape geometry of neural theorem provers compared to direct proof generation. We train matched transformer models on propositional logic tasks and measure landscape curvature, sharpness, mode connectivity, and training dynamics.

## Key Results
- **CoT minima are flatter**: Top Hessian eigenvalue 3.20 ± 0.78 (CoT) vs 4.16 ± 0.47 (Direct)
- **CoT has lower sharpness**: SAM-style sharpness 1.3-2.0x lower for CoT at all perturbation radii
- **CoT solutions are more isolated**: CoT-CoT loss barriers (1.40) are higher than Direct-Direct (1.07)
- **CoT training is smoother**: Lower gradient norms and more monotonic convergence
- **CoT achieves much lower loss**: 0.130 vs 0.262 final validation loss

## Reproducing Results

```bash
# Setup
source .venv/bin/activate

# Run experiment (~25 minutes on CPU)
USER=researcher python src/run_experiment.py

# Generate visualizations and statistics
python src/visualize.py
```

## Project Structure
```
REPORT.md              # Full research report
planning.md            # Research plan and methodology
src/
  data_generation.py   # Propositional logic theorem proving data
  model.py             # Transformer architecture
  run_experiment.py    # Main experiment (training + metrics)
  visualize.py         # Figure generation and statistics
results/
  experiment_results.json  # All raw experimental data
  statistics.json          # Statistical test results
figures/               # Generated plots
  training_curves.png
  loss_surfaces_2d.png
  hessian_and_sharpness.png
  mode_connectivity.png
  perturbation_profile.png
  summary_dashboard.png
literature_review.md   # Pre-gathered literature review
resources.md           # Resource catalog
papers/                # Downloaded research papers
```

## See Also
- [REPORT.md](REPORT.md) for full analysis and findings
