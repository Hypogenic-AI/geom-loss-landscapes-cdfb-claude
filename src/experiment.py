"""
Main experiment: Train CoT and Direct proof models, compute landscape metrics.
"""

import os
import sys
import json
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from data_generation import generate_dataset, SimpleTokenizer
from model import create_model, ProofTransformer

# ============================================================
# Configuration
# ============================================================

SEED = 42
N_TRAIN = 2000
N_VAL = 500
N_RUNS = 5  # Independent runs for statistics
EPOCHS = 50
BATCH_SIZE = 64
LR = 1e-3
MAX_SEQ_LEN = 128
DEVICE = 'cpu'

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# Dataset
# ============================================================

class ProofDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=MAX_SEQ_LEN):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []

        for inp_str, out_str in data:
            inp_ids = tokenizer.encode(inp_str)
            out_ids = tokenizer.encode(out_str)
            # Concatenate: input <sep> output
            full_ids = inp_ids[:-1] + [tokenizer.sep_id] + out_ids[1:]  # remove duplicate bos/eos
            full_ids = tokenizer.pad_sequence(full_ids, max_len)
            # Target: shift by 1 for autoregressive
            self.samples.append({
                'input_ids': torch.tensor(full_ids, dtype=torch.long),
                'sep_pos': len(inp_ids) - 1,  # position of <sep>
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ============================================================
# Training
# ============================================================

def train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR):
    """Train model and return training history."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {'train_loss': [], 'val_loss': [], 'grad_norm': []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_grad_norm = 0
        n_batches = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(DEVICE)

            logits = model(input_ids[:, :-1])
            targets = input_ids[:, 1:]

            # Compute loss only on non-padding tokens
            mask = (targets != 0).float()  # pad_id = 0
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   targets.reshape(-1), ignore_index=0)

            optimizer.zero_grad()
            loss.backward()

            # Track gradient norm
            grad_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5

            optimizer.step()

            total_loss += loss.item()
            total_grad_norm += grad_norm
            n_batches += 1

        avg_train_loss = total_loss / n_batches
        avg_grad_norm = total_grad_norm / n_batches

        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                logits = model(input_ids[:, :-1])
                targets = input_ids[:, 1:]
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                       targets.reshape(-1), ignore_index=0)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['grad_norm'].append(avg_grad_norm)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: train_loss={avg_train_loss:.4f}, "
                  f"val_loss={avg_val_loss:.4f}, grad_norm={avg_grad_norm:.4f}")

    return history


# ============================================================
# Landscape Metrics
# ============================================================

def compute_loss(model, data_loader):
    """Compute average loss."""
    model.eval()
    total_loss = 0
    n_batches = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            logits = model(input_ids[:, :-1])
            targets = input_ids[:, 1:]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   targets.reshape(-1), ignore_index=0)
            total_loss += loss.item()
            n_batches += 1
    return total_loss / n_batches


def compute_hessian_top_eigenvalues(model, data_loader, n_eigenvalues=10, n_iter=50):
    """
    Compute top eigenvalues of the Hessian using power iteration.
    Uses Hessian-vector products (no explicit Hessian construction).
    """
    model.eval()

    def hvp(v):
        """Compute Hessian-vector product."""
        model.zero_grad()
        # Compute loss - disable flash attention for double backward
        total_loss = 0
        n = 0
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            for batch in data_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                logits = model(input_ids[:, :-1])
                targets = input_ids[:, 1:]
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                       targets.reshape(-1), ignore_index=0)
                total_loss += loss
                n += 1
                if n >= 5:  # Use subset for efficiency
                    break
            avg_loss = total_loss / n

            # First backward pass to get gradients
            grads = torch.autograd.grad(avg_loss, model.parameters(), create_graph=True)
            flat_grad = torch.cat([g.reshape(-1) for g in grads])

            # Compute Hv = gradient of (grad . v)
            grad_v = torch.sum(flat_grad * v)
            hvp_result = torch.autograd.grad(grad_v, model.parameters())
        return torch.cat([h.reshape(-1) for h in hvp_result]).detach()

    n_params = sum(p.numel() for p in model.parameters())
    eigenvalues = []

    # Deflation-based power iteration for top eigenvalues
    found_vectors = []

    for k in range(min(n_eigenvalues, 20)):
        # Random initial vector
        v = torch.randn(n_params)
        v = v / v.norm()

        for _ in range(n_iter):
            # Hessian-vector product
            hv = hvp(v)

            # Deflate: remove components along already-found eigenvectors
            for ev, evec in zip(eigenvalues, found_vectors):
                hv -= ev * torch.dot(hv, evec) * evec

            eigenvalue = torch.dot(v, hv).item()
            v = hv / (hv.norm() + 1e-10)

        eigenvalues.append(eigenvalue)
        found_vectors.append(v.clone())

    return sorted(eigenvalues, reverse=True)


def compute_filter_normalized_directions(model):
    """
    Generate filter-normalized random directions (Li et al. 2018).
    For each parameter tensor, generate random direction and normalize
    to match the parameter's norm.
    """
    direction = []
    for p in model.parameters():
        d = torch.randn_like(p.data)
        # Filter normalization: normalize each "filter" to match param norm
        if p.dim() >= 2:
            for i in range(p.size(0)):
                d_norm = d[i].norm()
                p_norm = p.data[i].norm()
                if d_norm > 1e-10:
                    d[i] = d[i] * (p_norm / d_norm)
        else:
            d_norm = d.norm()
            p_norm = p.data.norm()
            if d_norm > 1e-10:
                d = d * (p_norm / d_norm)
        direction.append(d)
    return direction


def compute_2d_loss_surface(model, data_loader, n_points=21, range_val=1.0):
    """
    Compute 2D loss surface using filter-normalized random directions.
    """
    # Save original parameters
    orig_params = [p.data.clone() for p in model.parameters()]

    # Generate two random directions
    dir1 = compute_filter_normalized_directions(model)
    dir2 = compute_filter_normalized_directions(model)

    alphas = np.linspace(-range_val, range_val, n_points)
    betas = np.linspace(-range_val, range_val, n_points)
    surface = np.zeros((n_points, n_points))

    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            # Set parameters to theta* + alpha*d1 + beta*d2
            for p, orig, d1, d2 in zip(model.parameters(), orig_params, dir1, dir2):
                p.data.copy_(orig + alpha * d1 + beta * d2)

            surface[i, j] = compute_loss(model, data_loader)

    # Restore original parameters
    for p, orig in zip(model.parameters(), orig_params):
        p.data.copy_(orig)

    return alphas, betas, surface


def compute_loss_barrier(model1, model2, data_loader, n_points=21):
    """
    Compute loss along linear interpolation between two models.
    Returns barrier height.
    """
    params1 = model1.get_flat_params()
    params2 = model2.get_flat_params()

    lambdas = np.linspace(0, 1, n_points)
    losses = []

    for lam in lambdas:
        interpolated = (1 - lam) * params1 + lam * params2
        model1.set_flat_params(interpolated)
        loss = compute_loss(model1, data_loader)
        losses.append(loss)

    # Restore model1
    model1.set_flat_params(params1)

    endpoint_avg = (losses[0] + losses[-1]) / 2
    barrier = max(losses) - endpoint_avg

    return lambdas, losses, barrier


def compute_basin_width(model, data_loader, n_directions=20, n_steps=50, max_eps=2.0):
    """
    Estimate basin width by measuring loss increase along random directions.
    Returns: average epsilon at which loss doubles.
    """
    orig_params = model.get_flat_params()
    base_loss = compute_loss(model, data_loader)

    basin_widths = []
    epsilons = np.linspace(0, max_eps, n_steps)

    loss_profiles = []

    for d in range(n_directions):
        # Random normalized direction
        direction = torch.randn_like(orig_params)
        direction = direction / direction.norm()

        losses = []
        for eps in epsilons:
            model.set_flat_params(orig_params + eps * direction)
            loss = compute_loss(model, data_loader)
            losses.append(loss)

        loss_profiles.append(losses)

        # Find epsilon where loss exceeds threshold
        threshold = base_loss + 0.5 * base_loss  # 50% increase
        width = max_eps  # default if never exceeded
        for k, loss in enumerate(losses):
            if loss > threshold:
                width = epsilons[k]
                break
        basin_widths.append(width)

    # Restore
    model.set_flat_params(orig_params)

    return {
        'mean_basin_width': np.mean(basin_widths),
        'std_basin_width': np.std(basin_widths),
        'epsilons': epsilons.tolist(),
        'avg_loss_profile': np.mean(loss_profiles, axis=0).tolist(),
        'base_loss': base_loss,
    }


def compute_sharpness(model, data_loader, rho=0.05, n_samples=20):
    """
    Compute SAM-style sharpness: max_{||eps||<=rho} L(w+eps) - L(w).
    Approximated by sampling random perturbations.
    """
    orig_params = model.get_flat_params()
    base_loss = compute_loss(model, data_loader)

    max_perturbed_loss = base_loss
    for _ in range(n_samples):
        # Random perturbation within ball of radius rho
        eps = torch.randn_like(orig_params)
        eps = eps / eps.norm() * rho

        model.set_flat_params(orig_params + eps)
        perturbed_loss = compute_loss(model, data_loader)
        max_perturbed_loss = max(max_perturbed_loss, perturbed_loss)

    model.set_flat_params(orig_params)

    return max_perturbed_loss - base_loss


# ============================================================
# Main Experiment
# ============================================================

def run_single_experiment(seed, tokenizer, direct_train, direct_val, cot_train, cot_val):
    """Run one experiment with given seed, return all metrics."""
    set_seed(seed)
    print(f"\n{'='*60}")
    print(f"Run with seed={seed}")
    print(f"{'='*60}")

    # Create datasets
    direct_train_ds = ProofDataset(direct_train, tokenizer)
    direct_val_ds = ProofDataset(direct_val, tokenizer)
    cot_train_ds = ProofDataset(cot_train, tokenizer)
    cot_val_ds = ProofDataset(cot_val, tokenizer)

    direct_train_loader = DataLoader(direct_train_ds, batch_size=BATCH_SIZE, shuffle=True)
    direct_val_loader = DataLoader(direct_val_ds, batch_size=BATCH_SIZE)
    cot_train_loader = DataLoader(cot_train_ds, batch_size=BATCH_SIZE, shuffle=True)
    cot_val_loader = DataLoader(cot_val_ds, batch_size=BATCH_SIZE)

    # Create models
    model_config = {'d_model': 64, 'nhead': 2, 'num_layers': 2,
                    'dim_feedforward': 128, 'dropout': 0.1}
    direct_model = create_model(tokenizer.vocab_size, model_config).to(DEVICE)
    cot_model = create_model(tokenizer.vocab_size, model_config).to(DEVICE)

    print(f"Parameters per model: {direct_model.count_params()}")

    # Train Direct model
    print("\n--- Training Direct Model ---")
    set_seed(seed)  # Reset for reproducibility
    direct_history = train_model(direct_model, direct_train_loader, direct_val_loader)

    # Train CoT model
    print("\n--- Training CoT Model ---")
    set_seed(seed)
    cot_history = train_model(cot_model, cot_train_loader, cot_val_loader)

    # ---- Compute Landscape Metrics ----
    results = {
        'seed': seed,
        'direct_history': direct_history,
        'cot_history': cot_history,
    }

    # 1. Hessian top eigenvalues
    print("\nComputing Hessian eigenvalues (Direct)...")
    direct_eigenvalues = compute_hessian_top_eigenvalues(
        direct_model, direct_train_loader, n_eigenvalues=10, n_iter=30)
    print(f"  Direct top eigenvalues: {[f'{e:.4f}' for e in direct_eigenvalues[:5]]}")

    print("Computing Hessian eigenvalues (CoT)...")
    cot_eigenvalues = compute_hessian_top_eigenvalues(
        cot_model, cot_train_loader, n_eigenvalues=10, n_iter=30)
    print(f"  CoT top eigenvalues: {[f'{e:.4f}' for e in cot_eigenvalues[:5]]}")

    results['direct_eigenvalues'] = direct_eigenvalues
    results['cot_eigenvalues'] = cot_eigenvalues

    # 2. Basin width
    print("\nComputing basin width (Direct)...")
    direct_basin = compute_basin_width(direct_model, direct_val_loader, n_directions=15, n_steps=30)
    print(f"  Direct basin width: {direct_basin['mean_basin_width']:.4f} ± {direct_basin['std_basin_width']:.4f}")

    print("Computing basin width (CoT)...")
    cot_basin = compute_basin_width(cot_model, cot_val_loader, n_directions=15, n_steps=30)
    print(f"  CoT basin width: {cot_basin['mean_basin_width']:.4f} ± {cot_basin['std_basin_width']:.4f}")

    results['direct_basin'] = direct_basin
    results['cot_basin'] = cot_basin

    # 3. SAM-style sharpness
    print("\nComputing sharpness...")
    for rho in [0.01, 0.05, 0.1]:
        direct_sharp = compute_sharpness(direct_model, direct_val_loader, rho=rho)
        cot_sharp = compute_sharpness(cot_model, cot_val_loader, rho=rho)
        results[f'direct_sharpness_rho{rho}'] = direct_sharp
        results[f'cot_sharpness_rho{rho}'] = cot_sharp
        print(f"  rho={rho}: Direct={direct_sharp:.6f}, CoT={cot_sharp:.6f}")

    # 4. Loss barrier between independently trained models of same type
    # We'll need a second model for this - train with different seed
    print("\nTraining second Direct model for mode connectivity...")
    set_seed(seed + 1000)
    direct_model2 = create_model(tokenizer.vocab_size, model_config).to(DEVICE)
    _ = train_model(direct_model2, direct_train_loader, direct_val_loader)

    set_seed(seed + 1000)
    cot_model2 = create_model(tokenizer.vocab_size, model_config).to(DEVICE)
    _ = train_model(cot_model2, cot_train_loader, cot_val_loader)

    print("\nComputing loss barriers...")
    _, direct_interp_losses, direct_barrier = compute_loss_barrier(
        direct_model, direct_model2, direct_val_loader)
    _, cot_interp_losses, cot_barrier = compute_loss_barrier(
        cot_model, cot_model2, cot_val_loader)
    print(f"  Direct barrier: {direct_barrier:.6f}")
    print(f"  CoT barrier: {cot_barrier:.6f}")

    results['direct_barrier'] = direct_barrier
    results['cot_barrier'] = cot_barrier
    results['direct_interp_losses'] = direct_interp_losses
    results['cot_interp_losses'] = cot_interp_losses

    # 5. Cross-type loss barrier (interpolation between CoT and Direct)
    print("\nComputing cross-type loss barrier...")
    # Need to use same evaluation data - use direct val since both models process the same inputs differently
    # Actually, the models are trained on different target distributions, so cross-interpolation
    # tests whether they're in the same basin of the PARAMETER space
    _, cross_interp_losses, cross_barrier = compute_loss_barrier(
        direct_model, cot_model, direct_val_loader)
    print(f"  Cross-type barrier: {cross_barrier:.6f}")
    results['cross_barrier'] = cross_barrier
    results['cross_interp_losses'] = cross_interp_losses

    return results


def run_2d_surface_experiment(seed, tokenizer, direct_train, direct_val, cot_train, cot_val):
    """Compute 2D loss surfaces (expensive, run once)."""
    set_seed(seed)

    direct_train_ds = ProofDataset(direct_train, tokenizer)
    direct_val_ds = ProofDataset(direct_val, tokenizer)
    cot_train_ds = ProofDataset(cot_train, tokenizer)
    cot_val_ds = ProofDataset(cot_val, tokenizer)

    direct_train_loader = DataLoader(direct_train_ds, batch_size=BATCH_SIZE, shuffle=True)
    direct_val_loader = DataLoader(direct_val_ds, batch_size=BATCH_SIZE)
    cot_train_loader = DataLoader(cot_train_ds, batch_size=BATCH_SIZE, shuffle=True)
    cot_val_loader = DataLoader(cot_val_ds, batch_size=BATCH_SIZE)

    model_config = {'d_model': 64, 'nhead': 2, 'num_layers': 2,
                    'dim_feedforward': 128, 'dropout': 0.1}

    # Train models
    direct_model = create_model(tokenizer.vocab_size, model_config).to(DEVICE)
    cot_model = create_model(tokenizer.vocab_size, model_config).to(DEVICE)

    set_seed(seed)
    print("Training Direct model for surface...")
    train_model(direct_model, direct_train_loader, direct_val_loader)

    set_seed(seed)
    print("Training CoT model for surface...")
    train_model(cot_model, cot_train_loader, cot_val_loader)

    # Compute 2D surfaces
    print("Computing 2D surface (Direct)...")
    d_alphas, d_betas, d_surface = compute_2d_loss_surface(
        direct_model, direct_val_loader, n_points=21, range_val=1.0)

    print("Computing 2D surface (CoT)...")
    c_alphas, c_betas, c_surface = compute_2d_loss_surface(
        cot_model, cot_val_loader, n_points=21, range_val=1.0)

    return {
        'alphas': d_alphas.tolist(),
        'betas': d_betas.tolist(),
        'direct_surface': d_surface.tolist(),
        'cot_surface': c_surface.tolist(),
    }


def main():
    start_time = time.time()

    # Generate data
    print("Generating datasets...")
    tokenizer = SimpleTokenizer()
    direct_train, cot_train = generate_dataset(n_samples=N_TRAIN, seed=SEED)
    direct_val, cot_val = generate_dataset(n_samples=N_VAL, seed=SEED + 100)

    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Training samples: {len(direct_train)}")
    print(f"Validation samples: {len(direct_val)}")

    # Run multiple experiments
    all_results = []
    for run_idx in range(N_RUNS):
        seed = SEED + run_idx * 111
        results = run_single_experiment(
            seed, tokenizer, direct_train, direct_val, cot_train, cot_val)
        all_results.append(results)

        elapsed = time.time() - start_time
        print(f"\nElapsed: {elapsed:.1f}s")

    # Compute 2D loss surfaces (run once)
    print("\n" + "="*60)
    print("Computing 2D loss surfaces...")
    print("="*60)
    surface_data = run_2d_surface_experiment(
        SEED, tokenizer, direct_train, direct_val, cot_train, cot_val)

    # Save all results
    # Convert to JSON-serializable format
    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    results_path = os.path.join(RESULTS_DIR, 'experiment_results.json')
    with open(results_path, 'w') as f:
        json.dump(make_serializable({
            'config': {
                'n_train': N_TRAIN, 'n_val': N_VAL, 'n_runs': N_RUNS,
                'epochs': EPOCHS, 'batch_size': BATCH_SIZE, 'lr': LR,
                'model_config': {'d_model': 64, 'nhead': 2, 'num_layers': 2,
                                  'dim_feedforward': 128},
            },
            'runs': all_results,
            'surface': surface_data,
        }), f, indent=2)

    print(f"\nResults saved to {results_path}")
    print(f"Total time: {time.time() - start_time:.1f}s")

    # Print summary
    print_summary(all_results)


def print_summary(all_results):
    """Print summary statistics across all runs."""
    print("\n" + "="*60)
    print("SUMMARY OF RESULTS")
    print("="*60)

    # Collect metrics across runs
    metrics = {
        'direct_top_eigenvalue': [],
        'cot_top_eigenvalue': [],
        'direct_eigenvalue_ratio': [],
        'cot_eigenvalue_ratio': [],
        'direct_basin_width': [],
        'cot_basin_width': [],
        'direct_barrier': [],
        'cot_barrier': [],
        'cross_barrier': [],
        'direct_final_train_loss': [],
        'cot_final_train_loss': [],
        'direct_final_val_loss': [],
        'cot_final_val_loss': [],
    }

    for r in all_results:
        d_eig = r['direct_eigenvalues']
        c_eig = r['cot_eigenvalues']
        metrics['direct_top_eigenvalue'].append(d_eig[0])
        metrics['cot_top_eigenvalue'].append(c_eig[0])
        # Ratio of top to 5th eigenvalue
        if len(d_eig) >= 5 and abs(d_eig[4]) > 1e-10:
            metrics['direct_eigenvalue_ratio'].append(abs(d_eig[0] / d_eig[4]))
        if len(c_eig) >= 5 and abs(c_eig[4]) > 1e-10:
            metrics['cot_eigenvalue_ratio'].append(abs(c_eig[0] / c_eig[4]))

        metrics['direct_basin_width'].append(r['direct_basin']['mean_basin_width'])
        metrics['cot_basin_width'].append(r['cot_basin']['mean_basin_width'])
        metrics['direct_barrier'].append(r['direct_barrier'])
        metrics['cot_barrier'].append(r['cot_barrier'])
        metrics['cross_barrier'].append(r['cross_barrier'])
        metrics['direct_final_train_loss'].append(r['direct_history']['train_loss'][-1])
        metrics['cot_final_train_loss'].append(r['cot_history']['train_loss'][-1])
        metrics['direct_final_val_loss'].append(r['direct_history']['val_loss'][-1])
        metrics['cot_final_val_loss'].append(r['cot_history']['val_loss'][-1])

    for name, values in metrics.items():
        if values:
            print(f"  {name}: {np.mean(values):.6f} ± {np.std(values):.6f} "
                  f"(median={np.median(values):.6f})")


if __name__ == '__main__':
    main()
