"""Render the ~55 s demo video (results/demo.mp4) from real data and saved results. Needs ffmpeg."""
import json
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from src.data import load_episodes, load_split, CFG, ROOT
from src.preprocess import normalize, chunk_episode
from src import events as ev
from src.codecs import load_released, fit_local
from src.evaluate import reconstruct_episode
from src.plots import COL, INK, MUTED

FPS = 20
RES = ROOT / "results"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})

eps, _ = load_episodes(); split = load_split(); norm = json.load(open(CFG / "norm.json"))
ecfg = json.load(open(CFG / "events.json")); frozen = json.load(open(CFG / "frozen.json"))
Z = {e: normalize(a, norm) for e, a in eps.items()}
tr = list(np.concatenate([chunk_episode(Z[e])[0] for e in split["train"]]))
parse = lambda n: (int(n.split("_")[1][1:]), float(n.split("_")[2][1:]))
codecs = {"control": fit_local(tr, *parse(frozen["control"]), frozen["vocab_size"]),
          "weighted": fit_local(tr, *parse(frozen["weighted"]), frozen["vocab_size"])}
E = split["test"][0]; z = Z[E]; mask = ev.event_mask(z, ecfg)
rec = {k: reconstruct_episode(c, z)["recon"] for k, c in codecs.items()}
evt = [d for d in ev.detect(z, ecfg) if d[0] == 13][0]; J = 13
lo, hi = evt[1] - 22, evt[2] + 22
boot = json.load(open(RES / "test_bootstrap.json"))["weighted_vs_control"]
val = pd.read_csv(RES / "val_sweep.csv"); test = pd.read_csv(RES / "test_summary.csv")
ins = json.load(open(RES / "insertion_results.json"))["weighted_vs_control"]["within_task"]["event_mae"]

SCENES = [("title", 7), ("episode", 12), ("zoom", 14), ("frontier", 11), ("result", 11)]
starts = np.cumsum([0] + [s[1] * FPS for s in SCENES]); TOTAL = starts[-1]
fig = plt.figure(figsize=(12.8, 7.2), dpi=100); fig.patch.set_facecolor("white")


def scene(i):
    k = np.searchsorted(starts, i, side="right") - 1
    return SCENES[k][0], (i - starts[k]) / (SCENES[k][1] * FPS)


def text(x, y, s, size=20, **kw):
    fig.text(x, y, s, fontsize=size, color=kw.pop("color", INK), ha=kw.pop("ha", "center"), va="center", **kw)


def draw(i):
    fig.clear(); name, p = scene(i)
    if name == "title":
        text(.5, .70, "Preserving critical movements", 34, weight="bold")
        text(.5, .62, "in compressed robot actions", 34, weight="bold")
        if p > .25: text(.5, .47, "Robot AI models read motion as tokens. Compression loses detail.", 18, color=MUTED)
        if p > .5: text(.5, .39, "Does the FAST tokenizer lose more detail exactly when the gripper snaps shut —", 17)
        if p > .5: text(.5, .34, "and can giving the gripper extra precision fix it cheaply?", 17)
        if p > .8: text(.5, .18, "Real data: ALOHA sim cube transfer · 50 demos · offline reconstruction only", 13, color=MUTED)
    elif name == "episode":
        ax = fig.add_axes([.07, .14, .88, .66]); T = int(np.clip(p * 1.15, 0, 1) * len(z)) or 1
        for j in range(14):
            if j not in (6, 13): ax.plot(z[:T, j], color="#cbd2d9", lw=.8)
        ax.plot(z[:T, 6], color="#2f6fdf", lw=2.2, label="left gripper"); ax.plot(z[:T, 13], color=COL["weighted"], lw=2.2, label="right gripper")
        ax.fill_between(range(T), -1.6, 1.6, where=mask[:T], color=COL["event"], alpha=.9, step="mid", lw=0)
        ax.set_xlim(0, 400); ax.set_ylim(-1.6, 1.6); ax.set_xlabel("frame (50 per second)"); ax.legend(loc="upper left", frameon=False)
        text(.5, .9, f"One held-out demonstration (8 s): 12 arm joints in grey, two grippers in colour", 18, weight="bold")
        text(.5, .85, "Shaded = 'events': rapid gripper commands (±60 ms), detected from training data only", 13, color=MUTED)
    elif name == "zoom":
        ax = fig.add_axes([.08, .14, .55, .66]); t = np.arange(lo, hi)
        ax.axvspan(evt[1] - 3, evt[2] + 3, color=COL["event"], lw=0)
        ax.plot(t, z[lo:hi, J], color=INK, lw=3, label="original")
        stage = min(2, int(p * 3))
        if stage >= 1: ax.plot(t, rec["control"][lo:hi, J], "--", color=COL["control"], lw=2.2, label="FAST, γ=5 (control)")
        if stage >= 2: ax.plot(t, rec["weighted"][lo:hi, J], "--", color=COL["weighted"], lw=2.2, label="FAST, γ=5, gripper ×2 precision")
        ax.legend(loc="upper left", frameon=False, fontsize=11); ax.set_xlabel("frame"); ax.set_ylabel("right gripper (normalized)")
        text(.5, .9, "Compress → decode → compare, zoomed on one event", 20, weight="bold")
        lines = ["FAST = DCT (cosine waves)", "+ rounding (the lossy step)", "+ BPE packing into tokens", "",
                 "Our one change: multiply the", "gripper channels by λ = 2 before", "rounding, divide after decoding.",
                 "→ gripper gets a 2× finer step."]
        for n, s in enumerate(lines): text(.67, .74 - n * .05, s, 12.5, ha="left", color=INK if n > 3 else MUTED)
    elif name == "frontier":
        ax = fig.add_axes([.09, .14, .62, .66]); n = int(np.clip(p * 1.3, 0, 1) * 10)
        loc = val[val.gamma.notna()]; pts = [r for _, r in loc.iterrows()]
        for r in pts[:n]:
            c = COL["control"] if r["lambda"] == 1 else COL["weighted"]
            ax.scatter(r.tokens_per_s, r.event_mae, s=110, color=c, zorder=3)
            ax.annotate(f"γ{int(r.gamma)} λ{int(r['lambda'])}", (r.tokens_per_s, r.event_mae), xytext=(6, 4),
                        textcoords="offset points", fontsize=9, color=MUTED)
        rel = val[val.gamma.isna()].iloc[0]
        if n >= 9: ax.scatter(rel.tokens_per_s, rel.event_mae, marker="*", s=380, color=COL["released"], zorder=4)
        ax.set_yscale("log"); ax.set_xlim(30, 110); ax.set_ylim(.0018, .025)
        ax.set_xlabel("tokens per second (cost →)"); ax.set_ylabel("gripper error near events (↓ better)")
        text(.5, .9, "Validation sweep: 9 local tokenizers + released FAST+", 20, weight="bold")
        for k, (s, c) in enumerate([("blue = λ 1 (no weighting)", COL["control"]), ("orange = gripper weighted", COL["weighted"]),
                                    ("★ = released FAST+", COL["released"]), ("", INK),
                                    ("Selection rule fixed in advance:", INK), ("token rates within 5%,", MUTED),
                                    ("arm error ≤ 5% worse.", MUTED), ("Chosen: γ5 λ2 vs γ5 λ1", INK)]):
            text(.75, .74 - k * .05, s, 14, ha="left", color=c)
    else:
        b = boot["event_mae"]; base = b["mean_diff"] / b["rel_diff"]
        text(.5, .86, "Held-out test (10 unseen demos), weighted vs identical control", 20, weight="bold")
        text(.5, .70, f"{100 * b['rel_diff']:+.0f}% gripper error at events", 44, weight="bold", color=COL["weighted"])
        text(.5, .61, f"95% CI {100 * b['ci_hi'] / base:+.0f}% to {100 * b['ci_lo'] / base:+.0f}%  ·  better in {b['n_better']}/10 demos  ·  "
                      f"{100 * boot['tokens_per_s']['rel_diff']:+.1f}% tokens  ·  no arm cost", 16)
        if p > .2: text(.5, .50, f"Holds at all 5 chunk offsets · replicates on a second task ({100 * ins['rel_diff']:+.0f}%) when refit", 15, color=MUTED)
        if p > .4: text(.5, .41, "Not a free win over released FAST+: same gripper accuracy, 24% fewer tokens, 75% more arm error", 14, color=MUTED)
        if p > .6: text(.5, .32, "Also found: the official fit() can silently corrupt held-out chunks with small vocabularies", 15, color=MUTED)
        if p > .8: text(.5, .17, "Offline reconstruction only — no policy training, no physical contact measured.", 13, color=MUTED, style="italic")


anim = FuncAnimation(fig, draw, frames=int(TOTAL))
anim.save(RES / "demo.mp4", writer=FFMpegWriter(fps=FPS, bitrate=2500))
for s, n in (("episode", 11), ("zoom", 13), ("result", 10.5)):  # stills for the README
    draw(int(starts[[x[0] for x in SCENES].index(s)] + n * FPS)); fig.savefig(RES / "figures" / f"demo_{s}.png", dpi=80)
import subprocess  # small looping GIF for the README (GitHub won't inline a relative MP4)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(RES / "demo.mp4"), "-vf",
                "fps=8,scale=760:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64[p];[b][p]paletteuse=dither=none",
                str(RES / "demo.gif")], check=True)
print("demo ->", RES / "demo.mp4", f"{TOTAL / FPS:.0f}s")
