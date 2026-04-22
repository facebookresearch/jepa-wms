"""
Profile CEM planning loop for different ToMe r values.
Generates publication-quality figures.

Run:
    conda run -n jepa-wms python scripts/profile_planning.py
"""
import sys
sys.path.insert(0, '/home/aarav/WorldModels/jepa-wms')

import torch
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from functools import partial

from app.plan_common.models.AdaLN_vit import vit_predictor_AdaLN

# ── Hardware ─────────────────────────────────────────────────────────────────
DEVICE = 'cuda'

# ── Model dims (match actual config) ─────────────────────────────────────────
D_IN     = 384
PRED_DIM = 384
GRID_H = GRID_W = 16   # 256 tokens
T_CTX  = 2             # ctxt_window

# ── CEM config (cem.yaml) ────────────────────────────────────────────────────
CEM_ITER    = 30
CEM_B       = 300   # num_samples
HORIZON     = 6

# ── Benchmarking ──────────────────────────────────────────────────────────────
WARMUP = 10
REPS   = 30

R_VALUES = [0, 8, 64]

# ─────────────────────────────────────────────────────────────────────────────
def make_model(tome_r: int):
    model = vit_predictor_AdaLN(
        img_size=(224, 224),
        patch_size=14,
        num_frames=T_CTX,
        tubelet_size=1,
        embed_dim=D_IN,
        predictor_embed_dim=PRED_DIM,
        depth=6,
        num_heads=16,
        use_rope=True,
        use_activation_checkpointing=False,
        action_dim=2,
        use_proprio=False,
        action_encoder_inpred=True,
        tome_r=tome_r,
        local_window=(T_CTX, -1, -1),
    )
    return model.eval().to(DEVICE)


def cuda_time_ms(fn, warmup=WARMUP, reps=REPS):
    """Mean ± std wall time in ms using CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(reps):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return float(np.mean(times)), float(np.std(times))


def measure(model, r: int) -> dict:
    N = GRID_H * GRID_W  # 256
    x = torch.randn(CEM_B, T_CTX, 1, GRID_H, GRID_W, D_IN, device=DEVICE)
    a = torch.randn(CEM_B, T_CTX, 2, device=DEVICE)

    with torch.no_grad():
        fwd_mean, fwd_std = cuda_time_ms(lambda: model(x, a))

    # Objective: L2 over predicted tokens
    pred = torch.randn(CEM_B, T_CTX, N, D_IN, device=DEVICE)
    goal = torch.randn(1,     T_CTX, N, D_IN, device=DEVICE).expand_as(pred)
    with torch.no_grad():
        obj_mean, obj_std = cuda_time_ms(
            lambda: ((pred - goal) ** 2).mean(dim=(-1, -2))
        )

    return dict(fwd=(fwd_mean, fwd_std), obj=(obj_mean, obj_std))


def estimate_planning_ms(t: dict) -> dict:
    """CEM total = CEM_ITER × HORIZON × fwd(B=CEM_B) + CEM_ITER × obj."""
    return {
        'Predictor forwards': CEM_ITER * HORIZON * t['fwd'][0],
        'Objective (L2)':     CEM_ITER * t['obj'][0],
    }


# ─────────────────────────────────────────────────────────────────────────────
def run_all():
    results_raw  = {}
    results_plan = {}

    for r in R_VALUES:
        print(f"\n{'─'*50}")
        print(f"  tome_r = {r}")
        print('─'*50)
        model = make_model(r)
        t = measure(model, r)
        results_raw[r]  = t
        results_plan[r] = estimate_planning_ms(t)
        del model
        torch.cuda.empty_cache()

        total = sum(v for v in results_plan[r].values())
        print(f"  fwd (B={CEM_B}):  {t['fwd'][0]:.1f} ± {t['fwd'][1]:.1f} ms")
        print(f"  objective:        {t['obj'][0]:.2f} ± {t['obj'][1]:.2f} ms")
        print(f"  ── Estimated CEM planning step: {total/1000:.2f} s")
        for k, v in results_plan[r].items():
            print(f"     {k:<22s}: {v/1000:.2f} s  ({100*v/total:.1f}%)")

    return results_raw, results_plan


# ─────────────────────────────────────────────────────────────────────────────
PALETTE = {
    'Predictor forwards': '#4C72B0',
    'Objective (L2)':     '#C44E52',
}

def make_figures(results_raw, results_plan, out_prefix='scripts/planning_profile'):
    r_labels   = [f'r = {r}' for r in R_VALUES]
    components = list(PALETTE.keys())
    colors     = list(PALETTE.values())
    x          = np.arange(len(R_VALUES))

    totals = np.array([sum(results_plan[r].values()) / 1000 for r in R_VALUES])
    base   = totals[0]

    # ── Figure 1: Stacked bar – planning time breakdown ───────────────────────
    fig1, ax1 = plt.subplots(figsize=(6, 4.5))

    bottoms = np.zeros(len(R_VALUES))
    for c, col in zip(components, colors):
        vals = np.array([results_plan[r][c] / 1000 for r in R_VALUES])
        ax1.bar(x, vals, bottom=bottoms, label=c, color=col,
                width=0.5, edgecolor='white', linewidth=0.8)
        for xi, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 0.5:
                ax1.text(xi, b + v / 2, f'{v:.1f} s',
                         ha='center', va='center',
                         fontsize=9.5, color='white', fontweight='bold')
        bottoms += vals

    for xi, (tot, su) in enumerate(zip(totals, base / totals)):
        ax1.text(xi, tot + base * 0.02,
                 f'{su:.2f}×\n({tot:.1f} s)',
                 ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    ax1.set_xticks(x)
    ax1.set_xticklabels(r_labels, fontsize=12)
    ax1.set_ylabel('Estimated time per planning step (s)', fontsize=11)
    ax1.set_title(
        'CEM Planning Step Time Breakdown\n'
        f'({CEM_ITER} iterations × {CEM_B} samples × H = {HORIZON})',
        fontsize=11,
    )
    ax1.legend(fontsize=10, framealpha=0.9)
    ax1.set_ylim(0, base * 1.25)
    ax1.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    ax1.spines[['top', 'right']].set_visible(False)
    fig1.tight_layout()
    p1 = f'{out_prefix}_cem_breakdown.pdf'
    fig1.savefig(p1, dpi=200, bbox_inches='tight')
    print(f'\nSaved: {p1}')

    # ── Figure 2: Per-call forward time with speedup annotations ─────────────
    fig2, ax2 = plt.subplots(figsize=(5, 4.5))

    means = np.array([results_raw[r]['fwd'][0] for r in R_VALUES])
    stds  = np.array([results_raw[r]['fwd'][1] for r in R_VALUES])
    bar_colors = ['#4C72B0', '#55A868', '#C44E52']

    bars = ax2.bar(x, means, yerr=stds, capsize=6,
                   color=bar_colors, width=0.5,
                   edgecolor='white', linewidth=0.8,
                   error_kw=dict(elinewidth=1.5, ecolor='#444'))

    for xi, (m, s, b) in enumerate(zip(means, stds, bars)):
        su = means[0] / m
        ax2.text(b.get_x() + b.get_width() / 2,
                 m + s + means[0] * 0.025,
                 f'{su:.2f}×',
                 ha='center', va='bottom', fontsize=10.5, fontweight='bold')

    ax2.set_xticks(x)
    ax2.set_xticklabels(r_labels, fontsize=12)
    ax2.set_ylabel('Time per predictor call (ms)', fontsize=11)
    ax2.set_title(
        f'Predictor Forward Pass (B = {CEM_B})\nvs. ToMe Merge Rate r',
        fontsize=11,
    )
    ax2.set_ylim(0, means[0] * 1.30)
    ax2.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax2.grid(axis='y', linestyle='--', alpha=0.4)
    ax2.spines[['top', 'right']].set_visible(False)
    fig2.tight_layout()
    p2 = f'{out_prefix}_cem_fwd.pdf'
    fig2.savefig(p2, dpi=200, bbox_inches='tight')
    print(f'Saved: {p2}')


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Warmup={WARMUP}, Reps={REPS}')
    results_raw, results_plan = run_all()
    make_figures(results_raw, results_plan)
    print('\nDone.')
