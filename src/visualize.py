"""
Visualization script for loss landscape analysis results.
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

def load_results():
    with open(os.path.join(RESULTS_DIR, 'experiment_results.json')) as f:
        return json.load(f)


def plot_training_curves(data):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    runs = data['runs']

    # Aggregate across runs
    for ax_idx, (key, ylabel, title) in enumerate([
        ('train_loss', 'Loss', 'Training Loss'),
        ('val_loss', 'Loss', 'Validation Loss'),
        ('grad_norm', 'Gradient Norm', 'Gradient Norm'),
    ]):
        ax = axes[ax_idx]
        for r in runs:
            epochs = range(1, len(r['direct_history'][key]) + 1)
            ax.plot(epochs, r['direct_history'][key], 'b-', alpha=0.3, linewidth=1)
            ax.plot(epochs, r['cot_history'][key], 'r-', alpha=0.3, linewidth=1)

        # Plot average
        d_avg = np.mean([r['direct_history'][key] for r in runs], axis=0)
        c_avg = np.mean([r['cot_history'][key] for r in runs], axis=0)
        epochs = range(1, len(d_avg) + 1)
        ax.plot(epochs, d_avg, 'b-', linewidth=2.5, label='Direct (avg)')
        ax.plot(epochs, c_avg, 'r-', linewidth=2.5, label='CoT (avg)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved training_curves.png")


def plot_2d_surfaces(data):
    surf = data['surface']
    alphas = np.array(surf['alphas'])
    betas = np.array(surf['betas'])
    d_surf = np.array(surf['direct_surface'])
    c_surf = np.array(surf['cot_surface'])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    A, B = np.meshgrid(alphas, betas)

    # Common scale
    vmin = min(d_surf.min(), c_surf.min())
    vmax = min(max(d_surf.max(), c_surf.max()), vmin + 4)

    for ax, surf_data, title in [
        (axes[0], d_surf, 'Direct Model'),
        (axes[1], c_surf, 'CoT Model'),
    ]:
        c = ax.contourf(A, B, surf_data.T, levels=30, cmap='viridis', vmin=vmin, vmax=vmax)
        ax.contour(A, B, surf_data.T, levels=15, colors='white', alpha=0.3, linewidths=0.5)
        ax.set_xlabel('Direction 1')
        ax.set_ylabel('Direction 2')
        ax.set_title(f'{title} Loss Surface')
        ax.plot(0, 0, 'r*', markersize=15)
        plt.colorbar(c, ax=ax, label='Loss')

    # Difference surface
    ax = axes[2]
    diff = d_surf - c_surf
    c = ax.contourf(A, B, diff.T, levels=30, cmap='RdBu_r')
    ax.set_xlabel('Direction 1')
    ax.set_ylabel('Direction 2')
    ax.set_title('Direct - CoT Loss Difference')
    ax.plot(0, 0, 'r*', markersize=15)
    plt.colorbar(c, ax=ax, label='Loss Difference')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'loss_surfaces_2d.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved loss_surfaces_2d.png")


def plot_hessian_metrics(data):
    runs = data['runs']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Top eigenvalue
    ax = axes[0]
    d_eig = [r['direct_top_eig'] for r in runs]
    c_eig = [r['cot_top_eig'] for r in runs]
    bp = ax.boxplot([d_eig, c_eig], labels=['Direct', 'CoT'], patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightsalmon')
    ax.set_ylabel('λ_max (Top Eigenvalue)')
    ax.set_title('Hessian Top Eigenvalue')
    ax.grid(True, alpha=0.3)
    if len(d_eig) >= 3:
        _, p = stats.mannwhitneyu(d_eig, c_eig, alternative='two-sided')
        ax.text(0.5, 0.95, f'p={p:.4f}', transform=ax.transAxes, ha='center', va='top', fontsize=9)

    # Hessian trace
    ax = axes[1]
    d_tr = [r['direct_trace'] for r in runs]
    c_tr = [r['cot_trace'] for r in runs]
    bp = ax.boxplot([d_tr, c_tr], labels=['Direct', 'CoT'], patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightsalmon')
    ax.set_ylabel('Trace(H)')
    ax.set_title('Hessian Trace')
    ax.grid(True, alpha=0.3)

    # Sharpness across rho values
    ax = axes[2]
    rhos = [0.01, 0.05, 0.1, 0.5]
    d_means = [np.mean([r[f'direct_sharp_{rho}'] for r in runs]) for rho in rhos]
    d_stds = [np.std([r[f'direct_sharp_{rho}'] for r in runs]) for rho in rhos]
    c_means = [np.mean([r[f'cot_sharp_{rho}'] for r in runs]) for rho in rhos]
    c_stds = [np.std([r[f'cot_sharp_{rho}'] for r in runs]) for rho in rhos]

    x = np.arange(len(rhos))
    w = 0.35
    ax.bar(x - w/2, d_means, w, yerr=d_stds, label='Direct', color='lightblue', edgecolor='blue', capsize=4)
    ax.bar(x + w/2, c_means, w, yerr=c_stds, label='CoT', color='lightsalmon', edgecolor='red', capsize=4)
    ax.set_xlabel('Perturbation Radius (ρ)')
    ax.set_ylabel('Sharpness')
    ax.set_title('SAM-Style Sharpness')
    ax.set_xticks(x)
    ax.set_xticklabels([str(rho) for rho in rhos])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'hessian_and_sharpness.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved hessian_and_sharpness.png")


def plot_mode_connectivity(data):
    runs = data['runs']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Interpolation curves (first run)
    ax = axes[0]
    r = runs[0]
    lam = np.linspace(0, 1, len(r['direct_interp']))
    ax.plot(lam, r['direct_interp'], 'b-', label='Direct↔Direct', linewidth=2)
    ax.plot(lam, r['cot_interp'], 'r-', label='CoT↔CoT', linewidth=2)
    ax.plot(lam, r['cross_interp'], 'g--', label='Direct↔CoT', linewidth=2)
    ax.set_xlabel('Interpolation λ')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Along Linear Interpolation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Barrier heights
    ax = axes[1]
    d_bar = [r['direct_barrier'] for r in runs]
    c_bar = [r['cot_barrier'] for r in runs]
    x_bar = [r['cross_barrier'] for r in runs]
    bp = ax.boxplot([d_bar, c_bar, x_bar],
                    labels=['Direct↔Direct', 'CoT↔CoT', 'Direct↔CoT'],
                    patch_artist=True, widths=0.5)
    for box, color in zip(bp['boxes'], ['lightblue', 'lightsalmon', 'lightgreen']):
        box.set_facecolor(color)
    ax.set_ylabel('Loss Barrier Height')
    ax.set_title('Mode Connectivity Barriers')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'mode_connectivity.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved mode_connectivity.png")


def plot_perturbation_profiles(data):
    fig, ax = plt.subplots(figsize=(8, 5))
    r = data['runs'][0]

    n_pts = len(r['direct_perturb']['avg_profile'])
    eps_d = np.linspace(0, 3.0, n_pts)
    eps_c = np.linspace(0, 3.0, len(r['cot_perturb']['avg_profile']))
    ax.plot(eps_d, r['direct_perturb']['avg_profile'], 'b-', label='Direct', linewidth=2)
    ax.plot(eps_c, r['cot_perturb']['avg_profile'], 'r-', label='CoT', linewidth=2)

    # Add base loss lines
    ax.axhline(y=r['direct_perturb']['base_loss'], color='b', linestyle=':', alpha=0.5, label='Direct min')
    ax.axhline(y=r['cot_perturb']['base_loss'], color='r', linestyle=':', alpha=0.5, label='CoT min')

    ax.set_xlabel('Perturbation Magnitude (ε)')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Under Random Perturbation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'perturbation_profile.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved perturbation_profile.png")


def plot_summary_dashboard(data):
    runs = data['runs']
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Loss Landscape Geometry: CoT vs Direct Proof Generation\n(Across 3 Independent Runs)',
                 fontsize=14, fontweight='bold')

    metrics = [
        ('Final Validation Loss', [r['direct_history']['val_loss'][-1] for r in runs],
         [r['cot_history']['val_loss'][-1] for r in runs]),
        ('Top Hessian Eigenvalue (λ_max)', [r['direct_top_eig'] for r in runs],
         [r['cot_top_eig'] for r in runs]),
        ('Hessian Trace', [r['direct_trace'] for r in runs],
         [r['cot_trace'] for r in runs]),
        ('Sharpness (ρ=0.05)', [r['direct_sharp_0.05'] for r in runs],
         [r['cot_sharp_0.05'] for r in runs]),
        ('Sharpness (ρ=0.5)', [r['direct_sharp_0.5'] for r in runs],
         [r['cot_sharp_0.5'] for r in runs]),
        ('Final Gradient Norm', [r['direct_history']['grad_norm'][-1] for r in runs],
         [r['cot_history']['grad_norm'][-1] for r in runs]),
    ]

    for idx, (name, d_vals, c_vals) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        bp = ax.boxplot([d_vals, c_vals], labels=['Direct', 'CoT'], patch_artist=True, widths=0.5)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightsalmon')
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.grid(True, alpha=0.3)

        # Add individual points
        for i, vals in enumerate([d_vals, c_vals], 1):
            ax.scatter([i]*len(vals), vals, color='black', alpha=0.5, s=20, zorder=5)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'summary_dashboard.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved summary_dashboard.png")


def compute_statistics(data):
    runs = data['runs']
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)

    tests = {
        'Final Val Loss': ([r['direct_history']['val_loss'][-1] for r in runs],
                           [r['cot_history']['val_loss'][-1] for r in runs]),
        'Top Eigenvalue': ([r['direct_top_eig'] for r in runs], [r['cot_top_eig'] for r in runs]),
        'Hessian Trace': ([r['direct_trace'] for r in runs], [r['cot_trace'] for r in runs]),
        'Sharpness (ρ=0.05)': ([r['direct_sharp_0.05'] for r in runs], [r['cot_sharp_0.05'] for r in runs]),
        'Sharpness (ρ=0.5)': ([r['direct_sharp_0.5'] for r in runs], [r['cot_sharp_0.5'] for r in runs]),
        'Same-type Barrier': ([r['direct_barrier'] for r in runs], [r['cot_barrier'] for r in runs]),
        'Final Grad Norm': ([r['direct_history']['grad_norm'][-1] for r in runs],
                             [r['cot_history']['grad_norm'][-1] for r in runs]),
    }

    results = {}
    for name, (dv, cv) in tests.items():
        dm, ds = np.mean(dv), np.std(dv)
        cm, cs = np.mean(cv), np.std(cv)
        try:
            stat, p = stats.mannwhitneyu(dv, cv, alternative='two-sided')
        except:
            stat, p = 0, 1.0
        n1, n2 = len(dv), len(cv)
        effect = 1 - (2*stat)/(n1*n2) if n1*n2 > 0 else 0

        print(f"\n{name}:")
        print(f"  Direct: {dm:.6f} ± {ds:.6f}")
        print(f"  CoT:    {cm:.6f} ± {cs:.6f}")
        print(f"  U={stat:.1f}, p={p:.4f}, effect_r={effect:.3f}")
        print(f"  {'Direct > CoT' if dm > cm else 'CoT > Direct'}")

        results[name] = {
            'direct_mean': dm, 'direct_std': ds,
            'cot_mean': cm, 'cot_std': cs,
            'p_value': p, 'effect_size': effect,
        }

    with open(os.path.join(RESULTS_DIR, 'statistics.json'), 'w') as f:
        json.dump(results, f, indent=2)
    return results


def main():
    data = load_results()
    print("Generating visualizations...")
    plot_training_curves(data)
    plot_2d_surfaces(data)
    plot_hessian_metrics(data)
    plot_mode_connectivity(data)
    plot_perturbation_profiles(data)
    plot_summary_dashboard(data)
    compute_statistics(data)
    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == '__main__':
    main()
