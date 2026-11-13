"""Chunk 2: round-trip two synthetic chunks through released FAST+ on CPU.
Synthetic results are debugging evidence only."""
import json
import numpy as np
from huggingface_hub import HfApi
from transformers import AutoProcessor

REPO = "physical-intelligence/fast"
H, D, GRIP = 50, 14, (6, 13)

rev = HfApi().model_info(REPO).sha
proc = AutoProcessor.from_pretrained(REPO, trust_remote_code=True, revision=rev)

t = np.linspace(0, 1, H)
smooth = np.stack([0.5 * np.sin(2 * np.pi * (0.5 + 0.1 * j) * t + j) for j in range(D)], 1)
ramp = smooth.copy()
for g in GRIP:  # sharp gripper close over 3 frames
    ramp[:, g] = np.clip((t - 0.5) / 0.06, 0, 1) * 1.6 - 0.8

report = {"fast_revision": rev, "scale": getattr(proc, "scale", None),
          "vocab_size": getattr(proc, "vocab_size", None)}
for name, x in [("smooth", smooth), ("gripper_ramp", ramp)]:
    toks = proc(x[None].astype(np.float32))[0]
    rec = np.asarray(proc.decode([toks], time_horizon=H, action_dim=D))[0]
    assert rec.shape == (H, D) and np.isfinite(rec).all()
    err = np.abs(rec - x)
    arm = [j for j in range(D) if j not in GRIP]
    report[name] = {"n_tokens": len(toks), "mae_all": float(err.mean()),
                    "mae_arm": float(err[:, arm].mean()), "mae_grip": float(err[:, list(GRIP)].mean()),
                    "max_err": float(err.max()), "all_zero_decode": bool(np.allclose(rec, 0))}
json.dump(report, open("results/sanity_fast.json", "w"), indent=2)
print(json.dumps(report, indent=2))
