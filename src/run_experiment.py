"""
Final experiment: CoT vs Direct proof generation loss landscape analysis.
Fixed Hessian computation, optimized for CPU.
"""

import os, sys, json, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from data_generation import generate_dataset, SimpleTokenizer
from model import create_model

SEED = 42
N_TRAIN = 1500
N_VAL = 400
N_RUNS = 3
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
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

def compute_loss(model, loader):
    model.eval()
    total, n = 0, 0
    with torch.no_grad():
        for batch in loader:
            logits = model(batch[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   batch[:, 1:].reshape(-1), ignore_index=0)
            total += loss.item(); n += 1
    return total / max(n, 1)

def train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR, verbose=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {'train_loss': [], 'val_loss': [], 'grad_norm': []}
    for epoch in range(epochs):
        model.train()
        tot_loss, tot_gn, nb = 0, 0, 0
        for batch in train_loader:
            logits = model(batch[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   batch[:, 1:].reshape(-1), ignore_index=0)
            optimizer.zero_grad()
            loss.backward()
            gn = sum(p.grad.data.norm(2).item()**2 for p in model.parameters() if p.grad is not None)**0.5
            optimizer.step()
            tot_loss += loss.item(); tot_gn += gn; nb += 1
        vl = compute_loss(model, val_loader)
        history['train_loss'].append(tot_loss/nb)
        history['val_loss'].append(vl)
        history['grad_norm'].append(tot_gn/nb)
        if verbose and (epoch+1) % 10 == 0:
            print(f"  Epoch {epoch+1}: train={tot_loss/nb:.4f}, val={vl:.4f}, gn={tot_gn/nb:.4f}")
    return history


def compute_top_eigenvalue(model, loader, n_iter=30):
    """Compute ONLY the top eigenvalue via power iteration (stable)."""
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    v = torch.randn(n_params); v = v / v.norm()

    for _ in range(n_iter):
        model.zero_grad()
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            total_loss, n = 0, 0
            for batch in loader:
                logits = model(batch[:, :-1])
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                       batch[:, 1:].reshape(-1), ignore_index=0)
                total_loss += loss; n += 1
                if n >= 3: break
            avg_loss = total_loss / n
            grads = torch.autograd.grad(avg_loss, model.parameters(), create_graph=True)
            flat_grad = torch.cat([g.reshape(-1) for g in grads])
            grad_v = torch.sum(flat_grad * v)
            hvp_result = torch.autograd.grad(grad_v, model.parameters())
        hv = torch.cat([h.reshape(-1) for h in hvp_result]).detach()
        eigenvalue = torch.dot(v, hv).item()
        v = hv / (hv.norm() + 1e-10)

    return eigenvalue


def compute_hessian_trace(model, loader, n_samples=10):
    """Estimate Hessian trace via Hutchinson's estimator."""
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    trace_est = 0.0

    for _ in range(n_samples):
        # Random Rademacher vector
        v = torch.sign(torch.randn(n_params))
        model.zero_grad()
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            total_loss, n = 0, 0
            for batch in loader:
                logits = model(batch[:, :-1])
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                       batch[:, 1:].reshape(-1), ignore_index=0)
                total_loss += loss; n += 1
                if n >= 3: break
            avg_loss = total_loss / n
            grads = torch.autograd.grad(avg_loss, model.parameters(), create_graph=True)
            flat_grad = torch.cat([g.reshape(-1) for g in grads])
            grad_v = torch.sum(flat_grad * v)
            hvp_result = torch.autograd.grad(grad_v, model.parameters())
        hv = torch.cat([h.reshape(-1) for h in hvp_result]).detach()
        trace_est += torch.dot(v, hv).item()

    return trace_est / n_samples


def compute_filter_normalized_directions(model, seed=None):
    if seed is not None: set_seed(seed)
    direction = []
    for p in model.parameters():
        d = torch.randn_like(p.data)
        if p.dim() >= 2:
            for i in range(p.size(0)):
                dn = d[i].norm()
                pn = p.data[i].norm()
                if dn > 1e-10: d[i] = d[i] * (pn / dn)
        else:
            dn = d.norm(); pn = p.data.norm()
            if dn > 1e-10: d = d * (pn / dn)
        direction.append(d)
    return direction


def compute_2d_loss_surface(model, loader, n_points=21, range_val=1.0):
    orig = [p.data.clone() for p in model.parameters()]
    d1 = compute_filter_normalized_directions(model, seed=999)
    d2 = compute_filter_normalized_directions(model, seed=1000)
    alphas = np.linspace(-range_val, range_val, n_points)
    betas = np.linspace(-range_val, range_val, n_points)
    surface = np.zeros((n_points, n_points))
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            for p, o, dd1, dd2 in zip(model.parameters(), orig, d1, d2):
                p.data.copy_(o + a*dd1 + b*dd2)
            surface[i, j] = compute_loss(model, loader)
    for p, o in zip(model.parameters(), orig): p.data.copy_(o)
    return alphas, betas, surface


def compute_loss_barrier(m1, m2, loader, n_points=21):
    p1, p2 = m1.get_flat_params(), m2.get_flat_params()
    lambdas = np.linspace(0, 1, n_points)
    losses = []
    for lam in lambdas:
        m1.set_flat_params((1-lam)*p1 + lam*p2)
        losses.append(compute_loss(m1, loader))
    m1.set_flat_params(p1)
    barrier = max(losses) - (losses[0] + losses[-1])/2
    return lambdas.tolist(), losses, barrier


def compute_perturbation_sensitivity(model, loader, epsilons, n_dirs=10):
    """Measure loss increase along random perturbation directions."""
    orig = model.get_flat_params()
    base_loss = compute_loss(model, loader)
    profiles = []
    for _ in range(n_dirs):
        d = torch.randn_like(orig); d = d / d.norm()
        ls = []
        for eps in epsilons:
            model.set_flat_params(orig + eps * d)
            ls.append(compute_loss(model, loader))
        profiles.append(ls)
    model.set_flat_params(orig)
    avg_profile = np.mean(profiles, axis=0)
    # Basin width: epsilon where avg loss exceeds base + 0.5*base
    threshold = base_loss * 1.5
    width = float(epsilons[-1])
    for k, l in enumerate(avg_profile):
        if l > threshold:
            width = float(epsilons[k])
            break
    return {
        'base_loss': float(base_loss),
        'avg_profile': avg_profile.tolist(),
        'basin_width': width,
        'profiles': [p for p in profiles],  # all individual profiles
    }


def compute_sharpness(model, loader, rho=0.05, n_samples=15):
    orig = model.get_flat_params()
    base_loss = compute_loss(model, loader)
    max_loss = base_loss
    for _ in range(n_samples):
        eps = torch.randn_like(orig); eps = eps / eps.norm() * rho
        model.set_flat_params(orig + eps)
        max_loss = max(max_loss, compute_loss(model, loader))
    model.set_flat_params(orig)
    return float(max_loss - base_loss)


def run_experiment(seed, tokenizer, direct_train, direct_val, cot_train, cot_val):
    set_seed(seed)
    print(f"\n{'='*50}\nRun seed={seed}\n{'='*50}")

    dtrain = DataLoader(ProofDataset(direct_train, tokenizer), batch_size=BATCH_SIZE, shuffle=True)
    dval = DataLoader(ProofDataset(direct_val, tokenizer), batch_size=BATCH_SIZE)
    ctrain = DataLoader(ProofDataset(cot_train, tokenizer), batch_size=BATCH_SIZE, shuffle=True)
    cval = DataLoader(ProofDataset(cot_val, tokenizer), batch_size=BATCH_SIZE)

    cfg = {'d_model': 64, 'nhead': 2, 'num_layers': 2, 'dim_feedforward': 128, 'dropout': 0.1}

    print("Training Direct model...")
    set_seed(seed)
    dm = create_model(tokenizer.vocab_size, cfg)
    dh = train_model(dm, dtrain, dval)

    print("Training CoT model...")
    set_seed(seed)
    cm = create_model(tokenizer.vocab_size, cfg)
    ch = train_model(cm, ctrain, cval)

    r = {'seed': seed, 'direct_history': dh, 'cot_history': ch}

    # Top Hessian eigenvalue
    print("Computing Hessian metrics...")
    t0 = time.time()
    r['direct_top_eig'] = compute_top_eigenvalue(dm, dtrain, n_iter=25)
    r['cot_top_eig'] = compute_top_eigenvalue(cm, ctrain, n_iter=25)
    print(f"  Direct top eig: {r['direct_top_eig']:.4f}")
    print(f"  CoT top eig: {r['cot_top_eig']:.4f}")

    # Hessian trace
    r['direct_trace'] = compute_hessian_trace(dm, dtrain, n_samples=5)
    r['cot_trace'] = compute_hessian_trace(cm, ctrain, n_samples=5)
    print(f"  Direct trace: {r['direct_trace']:.4f}")
    print(f"  CoT trace: {r['cot_trace']:.4f}")
    print(f"  Hessian time: {time.time()-t0:.1f}s")

    # Perturbation sensitivity
    print("Computing perturbation sensitivity...")
    epsilons = np.linspace(0, 3.0, 25)
    r['direct_perturb'] = compute_perturbation_sensitivity(dm, dval, epsilons)
    r['cot_perturb'] = compute_perturbation_sensitivity(cm, cval, epsilons)
    print(f"  Direct basin width: {r['direct_perturb']['basin_width']:.3f}")
    print(f"  CoT basin width: {r['cot_perturb']['basin_width']:.3f}")

    # Sharpness
    print("Computing sharpness...")
    for rho in [0.01, 0.05, 0.1, 0.5]:
        r[f'direct_sharp_{rho}'] = compute_sharpness(dm, dval, rho=rho)
        r[f'cot_sharp_{rho}'] = compute_sharpness(cm, cval, rho=rho)
        print(f"  rho={rho}: Direct={r[f'direct_sharp_{rho}']:.6f}, CoT={r[f'cot_sharp_{rho}']:.6f}")

    # Mode connectivity
    print("Training 2nd models for connectivity...")
    set_seed(seed + 5000)
    dm2 = create_model(tokenizer.vocab_size, cfg)
    train_model(dm2, dtrain, dval, verbose=False)
    set_seed(seed + 5000)
    cm2 = create_model(tokenizer.vocab_size, cfg)
    train_model(cm2, ctrain, cval, verbose=False)

    print("Computing barriers...")
    _, r['direct_interp'], r['direct_barrier'] = compute_loss_barrier(dm, dm2, dval)
    _, r['cot_interp'], r['cot_barrier'] = compute_loss_barrier(cm, cm2, cval)
    # Cross-type: use combined loader (direct val since architecture is same)
    _, r['cross_interp'], r['cross_barrier'] = compute_loss_barrier(dm, cm, dval)
    print(f"  Direct↔Direct barrier: {r['direct_barrier']:.4f}")
    print(f"  CoT↔CoT barrier: {r['cot_barrier']:.4f}")
    print(f"  Direct↔CoT barrier: {r['cross_barrier']:.4f}")

    return r, dm, cm, dval, cval


def main():
    start = time.time()
    print("="*60)
    print("LOSS LANDSCAPE: CoT vs Direct Proof Generation")
    print("="*60)

    tokenizer = SimpleTokenizer()
    dt, ct = generate_dataset(n_samples=N_TRAIN, seed=SEED)
    dv, cv = generate_dataset(n_samples=N_VAL, seed=SEED+100)
    print(f"Vocab={tokenizer.vocab_size}, Train={N_TRAIN}, Val={N_VAL}")

    all_results = []
    last_dm = last_cm = last_dval = last_cval = None
    for i in range(N_RUNS):
        seed = SEED + i*111
        r, dm, cm, dvl, cvl = run_experiment(seed, tokenizer, dt, dv, ct, cv)
        all_results.append(r)
        last_dm, last_cm, last_dval, last_cval = dm, cm, dvl, cvl
        print(f"\n--- Run {i+1}/{N_RUNS} done. Elapsed: {time.time()-start:.1f}s ---")

    # 2D surfaces using last trained models
    print("\nComputing 2D loss surfaces...")
    da, db, ds = compute_2d_loss_surface(last_dm, last_dval, n_points=21)
    ca, cb, cs = compute_2d_loss_surface(last_cm, last_cval, n_points=21)

    surf = {'alphas': da.tolist(), 'betas': db.tolist(),
            'direct_surface': ds.tolist(), 'cot_surface': cs.tolist()}

    # Save
    def jsonify(o):
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, dict): return {k: jsonify(v) for k, v in o.items()}
        if isinstance(o, list): return [jsonify(v) for v in o]
        if isinstance(o, (float,)): return o
        return o

    output = jsonify({
        'config': {'n_train': N_TRAIN, 'n_val': N_VAL, 'n_runs': N_RUNS,
                    'epochs': EPOCHS, 'batch_size': BATCH_SIZE, 'lr': LR,
                    'model': {'d_model': 64, 'nhead': 2, 'layers': 2, 'ff': 128},
                    'n_params': last_dm.count_params()},
        'runs': all_results,
        'surface': surf,
    })

    path = os.path.join(RESULTS_DIR, 'experiment_results.json')
    with open(path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {path}")
    print(f"Total: {time.time()-start:.0f}s ({(time.time()-start)/60:.1f} min)")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    keys = [
        ('Final Val Loss', lambda r: r['direct_history']['val_loss'][-1],
         lambda r: r['cot_history']['val_loss'][-1]),
        ('Top Eigenvalue', lambda r: r['direct_top_eig'], lambda r: r['cot_top_eig']),
        ('Hessian Trace', lambda r: r['direct_trace'], lambda r: r['cot_trace']),
        ('Basin Width', lambda r: r['direct_perturb']['basin_width'],
         lambda r: r['cot_perturb']['basin_width']),
        ('Sharpness (ρ=0.05)', lambda r: r['direct_sharp_0.05'], lambda r: r['cot_sharp_0.05']),
        ('Sharpness (ρ=0.5)', lambda r: r['direct_sharp_0.5'], lambda r: r['cot_sharp_0.5']),
        ('Same-type Barrier', lambda r: r['direct_barrier'], lambda r: r['cot_barrier']),
        ('Cross Barrier', lambda r: r['cross_barrier'], lambda r: r['cross_barrier']),
        ('Final Grad Norm', lambda r: r['direct_history']['grad_norm'][-1],
         lambda r: r['cot_history']['grad_norm'][-1]),
    ]
    for name, fd, fc in keys:
        dv = [fd(r) for r in all_results]
        cv = [fc(r) for r in all_results]
        print(f"\n{name}:")
        print(f"  Direct: {np.mean(dv):.6f} ± {np.std(dv):.6f}")
        print(f"  CoT:    {np.mean(cv):.6f} ± {np.std(cv):.6f}")

if __name__ == '__main__':
    main()
