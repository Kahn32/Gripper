"""Chunks 5-7: fit normalization + event thresholds on train; report stats; make inspection plots."""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.data import load_episodes, load_split, action_names, CFG, ROOT
from src.preprocess import fit_norm, normalize, chunk_episode, common_frame_mask
from src import events as ev

(ROOT / "results" / "figures").mkdir(parents=True, exist_ok=True)
eps, _ = load_episodes(); split = load_split(); names = action_names()
train = {e: eps[e] for e in split["train"]}
norm = fit_norm(train); json.dump(norm, open(CFG / "norm.json", "w"), indent=2)
Z = {e: normalize(a, norm) for e, a in eps.items()}

rep = {"normalization": {}, "events": {}}
for part in ("train", "val", "test"):
    z = np.concatenate([Z[e] for e in split[part]])
    out = np.abs(z) > 1
    rep["normalization"][part] = {"frac_out_of_range": float(out.mean()),
                                  "per_channel": out.mean(0).round(4).tolist(), "max_abs": float(np.abs(z).max())}

ecfg = ev.fit_thresholds([Z[e] for e in split["train"]])
json.dump(ecfg, open(CFG / "events.json", "w"), indent=2)
for part in ("train", "val", "test"):
    n_ev, n_bearing, cov, oor = 0, 0, [], []
    for e in split[part]:
        d = ev.detect(Z[e], ecfg); m = ev.event_mask(Z[e], ecfg)
        n_ev += len(d); n_bearing += bool(d); cov.append(m.mean())
        if m.any(): oor.append((np.abs(Z[e][m]) > 1).mean())
    rep["events"][part] = {"n_events": n_ev, "event_bearing_episodes": n_bearing,
                           "n_episodes": len(split[part]), "frac_frames_in_E": float(np.mean(cov)),
                           "frac_out_of_range_inside_E": float(np.mean(oor)) if oor else None}

# chunking checks
T = len(eps[0])
rep["chunking"] = {"T": T, "chunks_per_episode": {o: len(chunk_episode(Z[0], 50, o)[1]) for o in (0, 10, 20, 30, 40)},
                   "common_frames": int(common_frame_mask(T).sum())}
json.dump(rep, open(ROOT / "results" / "prep_report.json", "w"), indent=2)
print(json.dumps({k: rep[k] for k in ("events", "chunking")}, indent=1))
print("norm out-of-range:", {p: round(v["frac_out_of_range"], 4) for p, v in rep["normalization"].items()},
      "floored:", norm["floored_channels"])

# figures: 5 episodes, and 10 training events
fig, ax = plt.subplots(5, 1, figsize=(10, 11), sharex=True)
for a, e in zip(ax, split["train"][:5]):
    a.plot(Z[e][:, [j for j in range(14) if j not in (6, 13)]], lw=.6, color="0.6")
    a.plot(Z[e][:, 6], label="left_gripper"); a.plot(Z[e][:, 13], label="right_gripper")
    m = ev.event_mask(Z[e], ecfg)
    a.fill_between(range(T), -1.5, 1.5, where=m, color="C3", alpha=.15, step="mid")
    for s in range(0, T, 50): a.axvline(s, color="k", lw=.3, ls=":")
    a.set_ylabel(f"ep {e}")
ax[0].legend(loc="upper right", fontsize=7); ax[-1].set_xlabel("frame (50 Hz)")
fig.tight_layout(); fig.savefig(ROOT / "results" / "figures" / "data_episodes_events.png", dpi=110)

evs = [(e, d) for e in split["train"] for d in ev.detect(Z[e], ecfg)][:: max(1, 1)][:10]
fig, ax = plt.subplots(2, 5, figsize=(14, 5))
for a, (e, (j, s, t)) in zip(ax.flat, evs):
    lo, hi = max(0, s - 15), min(T, t + 16)
    a.plot(range(lo, hi), Z[e][lo:hi, j], ".-")
    a.axvspan(s - 3, t + 3, color="C3", alpha=.2); a.set_title(f"ep{e} ch{j} [{s},{t}]", fontsize=8)
fig.tight_layout(); fig.savefig(ROOT / "results" / "figures" / "data_ten_events.png", dpi=110)
