"""
Profile the AdaLN predictor forward pass to see where time goes with/without ToMe.
Run with:
    conda run -n jepa-wms python scripts/profile_predictor.py
"""
import sys
sys.path.insert(0, '/home/aarav/WorldModels/jepa-wms')

import torch
import time
from functools import partial
from torch.profiler import profile, record_function, ProfilerActivity

from app.plan_common.models.AdaLN_vit import vit_predictor_AdaLN

# ── Match the actual config params ──────────────────────────────────────────
DEVICE = 'cuda'
B      = 100   # CEM num_samples
T      = 2     # ctxt_window
N      = 256   # 16x16 spatial tokens
D_IN   = 384   # encoder embed_dim
WARMUP = 5
REPS   = 20

def make_model(tome_r):
    model = vit_predictor_AdaLN(
        img_size=(224, 224),
        patch_size=14,
        num_frames=T,
        tubelet_size=1,
        embed_dim=D_IN,
        predictor_embed_dim=384,
        depth=6,
        num_heads=16,
        mlp_ratio=4.0,
        use_rope=True,
        use_activation_checkpointing=False,
        action_dim=2,
        use_proprio=False,
        action_encoder_inpred=True,
        tome_r=tome_r,
        local_window=(2, -1, -1),  # ctxt_window=2 → causal mask
    )
    return model.eval().to(DEVICE)


def make_inputs(model):
    # x: [B, T, 1, H, W, D_IN]  (V=1 view)
    x = torch.randn(B, T, 1, 16, 16, D_IN, device=DEVICE)
    # actions: [B, T, A]
    actions = torch.randn(B, T, 2, device=DEVICE)
    return x, actions


def bench(model, x, actions, label):
    # warmup
    with torch.no_grad():
        for _ in range(WARMUP):
            model(x, actions)
    torch.cuda.synchronize()

    # timed
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(REPS):
            model(x, actions)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / REPS * 1000
    print(f"[{label}] avg forward pass: {elapsed:.2f} ms")
    return elapsed


def prof(model, x, actions, label, tome_r):
    with torch.no_grad():
        for _ in range(3):
            model(x, actions)
    torch.cuda.synchronize()

    trace_path = f"/tmp/predictor_trace_r{tome_r}"
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_path),
    ) as prof_ctx:
        with torch.no_grad():
            for step in range(5):
                model(x, actions)
                prof_ctx.step()

    # Print top ops by CUDA time
    print(f"\n=== {label} — Top 20 ops by CUDA time ===")
    print(prof_ctx.key_averages().table(
        sort_by="cuda_time_total", row_limit=20
    ))
    print(f"(Full trace saved to {trace_path})")


if __name__ == "__main__":
    for r in [0, 8, 64]:
        print(f"\n{'='*60}")
        print(f"tome_r = {r}")
        print('='*60)
        model = make_model(r)
        x, actions = make_inputs(model)

        # Quick timing first
        bench(model, x, actions, f"r={r}")

        # Detailed profile
        prof(model, x, actions, f"r={r}", r)
