"""
Fast experiment: Train CoT and Direct proof models, compute all landscape metrics.
Optimized for CPU execution within time constraints.
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
from model import create_model

# ============================================================
# Configuration - optimized for speed
# ============================================================

SEED = 42
N_TRAIN = 1500
N_VAL = 400
N_RUNS = 5
EPOCHS = 30
BATCH_SIZE = 128
LR = 2e-3
MAX_SEQ_LEN = 80
DEVICE = 'cpu'

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ProofDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=MAX_SEQ_LEN):
        self.samples = []
        for inp_str, out_str in data:
            inp_ids = tokenizer.encode(inp_str)
            out_ids = tokenizer.encode(out_str)
            full_ids = inp_ids[:-1] + [tokenizer.sep_id] + out_ids[1:]
            full_ids = tokenizer.pad_sequence(full_ids, max_len)
            self.samples.append(torch.tensor(full_ids, dtype=torch.long))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def compute_loss(model, data_loader):
    model.eval()
    total_loss = 0
    n = 0
    with torch.no_grad():
        for batch in data_loader:
            logits = model(batch[:, :-1])
            targets = batch[:, 1:]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   targets.reshape(-1), ignore_index=0)
            total_loss += loss.item()
            n += 1
    return total_loss / max(n, 1)


def train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {'train_loss': [], 'val_loss': [], 'grad_norm': []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_grad_norm = 0
        n_batches = 0

        for batch in train_loader:
            logits = model(batch[:, :-1])
            targets = batch[:, 1:]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   targets.reshape(-1), ignore_index=0)
            optimizer.zero_grad()
            loss.backward()

            grad_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5

            optimizer.step()
            total_loss += loss.item()
            total_grad_norm += grad_norm
            n_batches += 1

        avg_loss = total_loss / n_batches
        avg_gn = total_grad_norm / n_batches
        val_loss = compute_loss(model, val_loader)

        history['train_loss'].append(avg_loss)
        history['val_loss'].append(val_loss)
        history['grad_norm'].append(avg_gn)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: train={avg_loss:.4f}, val={val_loss:.4f}, gn={avg_gn:.4f}")

    return history


def compute_hessian_top_eigenvalues(model, data_loader, n_eigenvalues=5, n_iter=20):
    """Top eigenvalues via power iteration with Hessian-vector products."""
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())

    def hvp(v):
        model.zero_grad()
        total_loss = 0
        n = 0
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            for batch in data_loader:
                logits = model(batch[:, :-1])
                targets = batch[:, 1:]
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                       targets.reshape(-1), ignore_index=0)
                total_loss += loss
                n += 1
                if n >= 3:
                    break
            avg_loss = total_loss / n
            grads = torch.autograd.grad(avg_loss, model.parameters(), create_graph=True)
            flat_grad = torch.cat([g.reshape(-1) for g in grads])
            grad_v = torch.sum(flat_grad * v)
            hvp_result = torch.autograd.grad(grad_v, model.parameters())
        return torch.cat([h.reshape(-1) for h in hvp_result]).detach()

    eigenvalues = []
    found_vectors = []

    for k in range(n_eigenvalues):
        v = torch.randn(n_params)
        v = v / v.norm()

        eigenvalue = 0
        for _ in range(n_iter):
            hv = hvp(v)
            for ev, evec in zip(eigenvalues, found_vectors):
                hv -= ev * torch.dot(hv, evec) * evec
            eigenvalue = torch.dot(v, hv).item()
            v_new = hv / (hv.norm() + 1e-10)
            v = v_new

        eigenvalues.append(eigenvalue)
        found_vectors.append(v.clone())

    return sorted(eigenvalues, reverse=True)


def compute_filter_normalized_directions(model):
    direction = []
    for p in model.parameters():
        d = torch.randn_like(p.data)
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
    orig_params = [p.data.clone() for p in model.parameters()]
    set_seed(999)  # Fixed seed for comparable directions
    dir1 = compute_filter_normalized_directions(model)
    dir2 = compute_filter_normalized_directions(model)

    alphas = np.linspace(-range_val, range_val, n_points)
    betas = np.linspace(-range_val, range_val, n_points)
    surface = np.zeros((n_points, n_points))

    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            for p, orig, d1, d2 in zip(model.parameters(), orig_params, dir1, dir2):
                p.data.copy_(orig + alpha * d1 + beta * d2)
            surface[i, j] = compute_loss(model, data_loader)

    for p, orig in zip(model.parameters(), orig_params):
        p.data.copy_(orig)

    return alphas, betas, surface


def compute_loss_barrier(model1, model2, data_loader, n_points=21):
    params1 = model1.get_flat_params()
    params2 = model2.get_flat_params()

    lambdas = np.linspace(0, 1, n_points)
    losses = []

    for lam in lambdas:
        model1.set_flat_params((1 - lam) * params1 + lam * params2)
        losses.append(compute_loss(model1, data_loader))

    model1.set_flat_params(params1)

    endpoint_avg = (losses[0] + losses[-1]) / 2
    barrier = max(losses) - endpoint_avg
    return lambdas.tolist(), losses, barrier


def compute_basin_width(model, data_loader, n_directions=10, n_steps=20, max_eps=2.0):
    orig_params = model.get_flat_params()
    base_loss = compute_loss(model, data_loader)
    basin_widths = []
    epsilons = np.linspace(0, max_eps, n_steps)
    all_profiles = []

    for d in range(n_directions):
        direction = torch.randn_like(orig_params)
        direction = direction / direction.norm()
        losses = []
        for eps in epsilons:
            model.set_flat_params(orig_params + eps * direction)
            losses.append(compute_loss(model, data_loader))
        all_profiles.append(losses)

        threshold = base_loss * 1.5
        width = max_eps
        for k, loss in enumerate(losses):
            if loss > threshold:
                width = epsilons[k]
                break
        basin_widths.append(width)

    model.set_flat_params(orig_params)
    return {
        'mean_basin_width': float(np.mean(basin_widths)),
        'std_basin_width': float(np.std(basin_widths)),
        'epsilons': epsilons.tolist(),
        'avg_loss_profile': np.mean(all_profiles, axis=0).tolist(),
        'base_loss': float(base_loss),
    }


def compute_sharpness(model, data_loader, rho=0.05, n_samples=10):
    orig_params = model.get_flat_params()
    base_loss = compute_loss(model, data_loader)
    max_loss = base_loss
    for _ in range(n_samples):
        eps = torch.randn_like(orig_params)
        eps = eps / eps.norm() * rho
        model.set_flat_params(orig_params + eps)
        max_loss = max(max_loss, compute_loss(model, data_loader))
    model.set_flat_params(orig_params)
    return float(max_loss - base_loss)


def run_single_experiment(seed, tokenizer, direct_train, direct_val, cot_train, cot_val):
    set_seed(seed)
    print(f"\n{'='*50}\nRun seed={seed}\n{'='*50}")

    direct_train_ds = ProofDataset(direct_train, tokenizer)
    direct_val_ds = ProofDataset(direct_val, tokenizer)
    cot_train_ds = ProofDataset(cot_train, tokenizer)
    cot_val_ds = ProofDataset(cot_val, tokenizer)

    direct_train_loader = DataLoader(direct_train_ds, batch_size=BATCH_SIZE, shuffle=True)
    direct_val_loader = DataLoader(direct_val_ds, batch_size=BATCH_SIZE)
    cot_train_loader = DataLoader(cot_train_ds, batch_size=BATCH_SIZE, shuffle=True)
    cot_val_loader = DataLoader(cot_val_ds, batch_size=BATCH_SIZE)

    cfg = {'d_model': 64, 'nhead': 2, 'num_layers': 2, 'dim_feedforward': 128, 'dropout': 0.1}

    # Train both models
    print("Training Direct model...")
    set_seed(seed)
    direct_model = create_model(tokenizer.vocab_size, cfg)
    direct_history = train_model(direct_model, direct_train_loader, direct_val_loader)

    print("Training CoT model...")
    set_seed(seed)
    cot_model = create_model(tokenizer.vocab_size, cfg)
    cot_history = train_model(cot_model, cot_train_loader, cot_val_loader)

    results = {
        'seed': seed,
        'direct_history': direct_history,
        'cot_history': cot_history,
    }

    # Hessian eigenvalues
    print("Computing Hessian eigenvalues...")
    t0 = time.time()
    results['direct_eigenvalues'] = compute_hessian_top_eigenvalues(
        direct_model, direct_train_loader, n_eigenvalues=5, n_iter=15)
    results['cot_eigenvalues'] = compute_hessian_top_eigenvalues(
        cot_model, cot_train_loader, n_eigenvalues=5, n_iter=15)
    print(f"  Direct eigs: {[f'{e:.3f}' for e in results['direct_eigenvalues']]}")
    print(f"  CoT eigs:    {[f'{e:.3f}' for e in results['cot_eigenvalues']]}")
    print(f"  Hessian time: {time.time()-t0:.1f}s")

    # Basin width
    print("Computing basin widths...")
    t0 = time.time()
    results['direct_basin'] = compute_basin_width(direct_model, direct_val_loader)
    results['cot_basin'] = compute_basin_width(cot_model, cot_val_loader)
    print(f"  Direct basin: {results['direct_basin']['mean_basin_width']:.3f}")
    print(f"  CoT basin:    {results['cot_basin']['mean_basin_width']:.3f}")
    print(f"  Basin time: {time.time()-t0:.1f}s")

    # Sharpness
    print("Computing sharpness...")
    for rho in [0.01, 0.05, 0.1]:
        results[f'direct_sharpness_rho{rho}'] = compute_sharpness(
            direct_model, direct_val_loader, rho=rho)
        results[f'cot_sharpness_rho{rho}'] = compute_sharpness(
            cot_model, cot_val_loader, rho=rho)

    # Mode connectivity - train second models
    print("Training second models for connectivity...")
    set_seed(seed + 1000)
    direct_model2 = create_model(tokenizer.vocab_size, cfg)
    train_model(direct_model2, direct_train_loader, direct_val_loader)

    set_seed(seed + 1000)
    cot_model2 = create_model(tokenizer.vocab_size, cfg)
    train_model(cot_model2, cot_train_loader, cot_val_loader)

    print("Computing loss barriers...")
    _, results['direct_interp_losses'], results['direct_barrier'] = \
        compute_loss_barrier(direct_model, direct_model2, direct_val_loader)
    _, results['cot_interp_losses'], results['cot_barrier'] = \
        compute_loss_barrier(cot_model, cot_model2, cot_val_loader)
    _, results['cross_interp_losses'], results['cross_barrier'] = \
        compute_loss_barrier(direct_model, cot_model, direct_val_loader)

    print(f"  Direct barrier: {results['direct_barrier']:.4f}")
    print(f"  CoT barrier: {results['cot_barrier']:.4f}")
    print(f"  Cross barrier: {results['cross_barrier']:.4f}")

    return results


def main():
    start_time = time.time()
    print("="*60)
    print("LOSS LANDSCAPE EXPERIMENT: CoT vs Direct Proof Generation")
    print("="*60)

    tokenizer = SimpleTokenizer()
    direct_train, cot_train = generate_dataset(n_samples=N_TRAIN, seed=SEED)
    direct_val, cot_val = generate_dataset(n_samples=N_VAL, seed=SEED + 100)
    print(f"Vocab: {tokenizer.vocab_size}, Train: {len(direct_train)}, Val: {len(direct_val)}")

    all_results = []
    for run_idx in range(N_RUNS):
        seed = SEED + run_idx * 111
        results = run_single_experiment(seed, tokenizer, direct_train, direct_val, cot_train, cot_val)
        all_results.append(results)
        elapsed = time.time() - start_time
        print(f"\n--- Run {run_idx+1}/{N_RUNS} done. Elapsed: {elapsed:.1f}s ---")

    # 2D loss surfaces (once)
    print("\n" + "="*50)
    print("Computing 2D loss surfaces...")
    set_seed(SEED)

    direct_train_ds = ProofDataset(direct_train, tokenizer)
    direct_val_ds = ProofDataset(direct_val, tokenizer)
    cot_train_ds = ProofDataset(cot_train, tokenizer)
    cot_val_ds = ProofDataset(cot_val, tokenizer)

    direct_val_loader = DataLoader(direct_val_ds, batch_size=BATCH_SIZE)
    cot_val_loader = DataLoader(cot_val_ds, batch_size=BATCH_SIZE)
    direct_train_loader = DataLoader(direct_train_ds, batch_size=BATCH_SIZE, shuffle=True)
    cot_train_loader = DataLoader(cot_train_ds, batch_size=BATCH_SIZE, shuffle=True)

    cfg = {'d_model': 64, 'nhead': 2, 'num_layers': 2, 'dim_feedforward': 128, 'dropout': 0.1}

    set_seed(SEED)
    direct_model = create_model(tokenizer.vocab_size, cfg)
    train_model(direct_model, direct_train_loader, direct_val_loader)

    set_seed(SEED)
    cot_model = create_model(tokenizer.vocab_size, cfg)
    train_model(cot_model, cot_train_loader, cot_val_loader)

    d_a, d_b, d_surf = compute_2d_loss_surface(direct_model, direct_val_loader, n_points=21)
    c_a, c_b, c_surf = compute_2d_loss_surface(cot_model, cot_val_loader, n_points=21)

    surface_data = {
        'alphas': d_a.tolist(), 'betas': d_b.tolist(),
        'direct_surface': d_surf.tolist(), 'cot_surface': c_surf.tolist(),
    }

    # Save everything
    def jsonify(obj):
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64, np.float16)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: jsonify(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [jsonify(v) for v in obj]
        return obj

    output = jsonify({
        'config': {
            'n_train': N_TRAIN, 'n_val': N_VAL, 'n_runs': N_RUNS,
            'epochs': EPOCHS, 'batch_size': BATCH_SIZE, 'lr': LR,
            'max_seq_len': MAX_SEQ_LEN,
            'model': {'d_model': 64, 'nhead': 2, 'num_layers': 2, 'dim_ff': 128},
            'n_params': direct_model.count_params(),
        },
        'runs': all_results,
        'surface': surface_data,
    })

    results_path = os.path.join(RESULTS_DIR, 'experiment_results.json')
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)

    total_time = time.time() - start_time
    print(f"\nResults saved to {results_path}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    metrics_to_compare = [
        ('Top Eigenvalue', 'direct_eigenvalues', 'cot_eigenvalues', lambda r, k: r[k][0]),
        ('Final Val Loss', 'direct_history', 'cot_history', lambda r, k: r[k]['val_loss'][-1]),
        ('Final Grad Norm', 'direct_history', 'cot_history', lambda r, k: r[k]['grad_norm'][-1]),
        ('Basin Width', 'direct_basin', 'cot_basin', lambda r, k: r[k]['mean_basin_width']),
        ('Same-type Barrier', 'direct_barrier', 'cot_barrier', lambda r, k: r[k]),
        ('Cross Barrier', 'cross_barrier', 'cross_barrier', lambda r, k: r[k]),
        ('Sharpness (rho=0.05)', 'direct_sharpness_rho0.05', 'cot_sharpness_rho0.05', lambda r, k: r[k]),
    ]

    for name, dk, ck, extract in metrics_to_compare:
        d_vals = [extract(r, dk) for r in all_results]
        c_vals = [extract(r, ck) for r in all_results]
        print(f"\n{name}:")
        print(f"  Direct: {np.mean(d_vals):.6f} ± {np.std(d_vals):.6f}")
        print(f"  CoT:    {np.mean(c_vals):.6f} ± {np.std(c_vals):.6f}")


if __name__ == '__main__':
    main()
