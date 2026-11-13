"""All README figures. Reads results/*.csv|json; refits the two frozen local codecs for overlays."""
import json
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.fft import dct
from src.data import load_episodes, load_split, CFG, ROOT
from src.preprocess import normalize, denormalize, chunk_episode
from src import events as ev
from src.codecs import load_released, fit_local, ref_quantize, channel_weights
from src.evaluate import reconstruct_episode

RES, FIG = ROOT / "results", ROOT / "results" / "figures"
INK, MUTED, GRID = "#1f2933", "#7b8794", "#e4e7eb"
COL = {"released": "#7b8794", "control": "#2f6fdf", "weighted": "#e8590c", "orig": "#1f2933", "event": "#f8d7c4"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": .8, "axes.titleweight": "bold",
                     "axes.titlesize": 11, "legend.frameon": False, "savefig.bbox": "tight", "savefig.dpi": 140})


def short(name):
    if name.startswith("released"): return "FAST+ (released)"
    g, l = name.split("_")[1][1:], name.split("_")[2][1:]
    return f"γ={g}, λ={l}"


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(11, 2.6)); ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(0, 2.6)
    boxes = [("Actions\n50 × 14", "1 s of commands"),
             ("Normalize\nto [-1, 1]", "joints on one scale"),
             ("× weight\ngrippers × λ", "our one change"),
             ("DCT +\nround(γ·c)", "the lossy step"),
             ("BPE", "lossless packing\n→ tokens"),
             ("Decode,\n÷ weight", "compare to original")]
    for i, (t, s) in enumerate(boxes):
        x = .15 + i * 1.8; hl = i == 2
        ax.add_patch(FancyBboxPatch((x, .9), 1.5, 1.2, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc="#fff4ec" if hl else "#f5f7fa", ec=COL["weighted"] if hl else MUTED, lw=1.4))
        ax.text(x + .75, 1.5, t, ha="center", va="center", fontsize=9, color=INK, weight="bold")
        ax.text(x + .75, .55, s, ha="center", va="center", fontsize=8.5, color=MUTED, style="italic")
        if i < len(boxes) - 1:
            ax.annotate("", (x + 1.78, 1.5), (x + 1.52, 1.5), arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.4))
    fig.savefig(FIG / "pipeline.png"); plt.close(fig)


def fig_how_dct_works(Z, split):
    """One real gripper close: DCT coefficients and reconstruction at coarse vs fine precision."""
    e = split["train"][0]; z = chunk_episode(Z[e])[0]
    k = int(np.argmax([np.abs(np.diff(c[:, 13])).max() for c in z])); c = z[k:k + 1]
    rec1, q1 = ref_quantize(c, 5); rec4, q4 = ref_quantize(c, 5, channel_weights(4))
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    t = np.arange(50)
    ax[0].plot(t, c[0, :, 13], color=COL["orig"], lw=2); ax[0].set_title("① A gripper command (1 s)")
    ax[0].set_xlabel("frame"); ax[0].set_ylabel("normalized position")
    co = dct(c[0, :, 13], norm="ortho")
    ax[1].bar(range(50), co, color=COL["control"], width=.8); ax[1].set_title("② Its DCT: mostly a few low frequencies")
    ax[1].set_xlabel("frequency index"); ax[1].set_ylabel("coefficient")
    ax[2].plot(t, c[0, :, 13], color=COL["orig"], lw=2, label="original")
    ax[2].plot(t, rec1[0, :, 13], color=COL["control"], lw=1.6, ls="--", label=f"coarse (λ=1)  {int((q1[0,:,13]!=0).sum())} nonzero coeffs")
    ax[2].plot(t, rec4[0, :, 13], color=COL["weighted"], lw=1.6, ls="--", label=f"fine (λ=4)  {int((q4[0,:,13]!=0).sum())} nonzero coeffs")
    ax[2].set_title("③ Round → decode (γ = 5)"); ax[2].legend(fontsize=8, loc="best"); ax[2].set_xlabel("frame")
    fig.tight_layout(); fig.savefig(FIG / "how_fast_works.png"); plt.close(fig)


def fig_frontier(frozen):
    v = pd.read_csv(RES / "val_sweep.csv"); t = pd.read_csv(RES / "test_summary.csv")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for a, col, title in ((ax[0], "event_mae", "Gripper error near events  (lower = better)"),
                          (ax[1], "arm_mae", "Arm error, all frames  (lower = better)")):
        for df, alpha, lab in ((v, 1, "validation"), (t, .35, "test")):
            loc = df[df.gamma.notna()]
            for l, mk in ((1, "o"), (2, "s"), (4, "D")):
                d = loc[loc["lambda"] == l].sort_values("tokens_per_s")
                c = COL["control"] if l == 1 else COL["weighted"]
                a.plot(d.tokens_per_s, d[col], marker=mk, color=c, alpha=alpha, lw=1.2, ms=7,
                       mfc=c if l != 2 else "white", label=f"λ={l} ({lab})" if lab == "validation" else None)
                if lab == "validation":
                    for _, r in d.iterrows():
                        a.annotate(f"γ={int(r.gamma)}", (r.tokens_per_s, r[col]), textcoords="offset points",
                                   xytext=(6, 4), fontsize=7.5, color=MUTED)
            rel = df[df.gamma.isna()]
            a.scatter(rel.tokens_per_s, rel[col], marker="*", s=220, color=COL["released"], alpha=alpha, zorder=5,
                      label="released FAST+" if lab == "validation" else None)
        for key, c in (("weighted", COL["weighted"]), ("control", COL["control"])):
            r = v[v.method == frozen[key]].iloc[0]
            a.scatter(r.tokens_per_s, r[col], s=260, facecolors="none", edgecolors=c, lw=2.2, zorder=6)
        a.set_yscale("log"); a.set_xlabel("tokens per second  (cost →)"); a.set_title(title)
    ax[0].set_ylabel("mean absolute error (normalized units)")
    ax[0].legend(fontsize=8, loc="upper right")
    fig.text(.5, -.03, "Solid = validation, faded = test. Circled = frozen comparison pair. "
             "Each line connects γ = 5, 10, 20 for one λ.", ha="center", color=MUTED, fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "fig1_rate_vs_error.png"); plt.close(fig)


def fig_overlays(Z, eps, norm, split, EM, codecs):
    e = split["test"][0]; recs = {k: reconstruct_episode(c, Z[e])["recon"] for k, c in codecs.items()}
    evs = ev.detect(Z[e], json.load(open(CFG / "events.json")))
    picks = [d for d in evs if d[0] == 6][:1] + [d for d in evs if d[0] == 13][:2]
    fig, ax = plt.subplots(1, len(picks) + 1, figsize=(15, 3.8))
    for a, (j, s, u) in zip(ax, picks):
        lo, hi = max(0, s - 20), min(len(Z[e]), u + 21); t = np.arange(lo, hi)
        a.axvspan(s - 3, u + 3, color=COL["event"], alpha=.8, lw=0, label="scored event window")
        a.plot(t, eps[e][lo:hi, j], color=COL["orig"], lw=2.4, label="original")
        for k, c in codecs.items():
            a.plot(t, denormalize(recs[k], norm)[lo:hi, j], color=COL[k], lw=1.5, ls="--", label=short(c.name))
        a.set_title(f"{'left' if j == 6 else 'right'} gripper, test ep {e}"); a.set_xlabel("frame")
    ax[0].set_ylabel("command (original units)")
    j = 1; lo, hi = 100, 300; t = np.arange(lo, hi); a = ax[-1]
    a.plot(t, eps[e][lo:hi, j], color=COL["orig"], lw=2.4)
    for k in codecs:
        a.plot(t, denormalize(recs[k], norm)[lo:hi, j], color=COL[k], lw=1.3, ls="--")
    a.set_title("arm: left shoulder (context)"); a.set_xlabel("frame")
    ax[0].legend(fontsize=7.5, loc="best"); fig.tight_layout(); fig.savefig(FIG / "fig2_event_overlays.png"); plt.close(fig)


def fig_phase():
    p = pd.read_csv(RES / "phase_summary.csv"); pe = pd.read_csv(RES / "phase_episodes.csv")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for lab, key in (("released_fast+", "released"), ("control", "control"), ("weighted", "weighted")):
        d = p[p.label == lab].sort_values("offset")
        for a, col in ((ax[0], "event_mae"), (ax[1], "arm_mae")):
            ee = pe[pe.label == lab].groupby("offset")[col]
            se = ee.std() / np.sqrt(ee.count())
            a.errorbar(d.offset, d[col], yerr=se.values, marker="o", color=COL[key], lw=1.8, capsize=3,
                       label={"released_fast+": "released FAST+", "control": "local control", "weighted": "weighted (γ5 λ2)"}[lab])
    ax[0].set_title("Gripper error near events vs chunk start")
    ax[0].text(.02, .04, "released FAST+ is hidden under the orange line:\nsame gripper step (γ·λ = 10·1 = 5·2)",
               transform=ax[0].transAxes, fontsize=8, color=MUTED); ax[1].set_title("Arm error vs chunk start")
    for a in ax: a.set_xlabel("chunk start offset (frames)"); a.set_xticks([0, 10, 20, 30, 40]); a.set_yscale("log")
    ax[0].set_ylabel("MAE (normalized, common 320 frames)"); ax[0].legend(fontsize=8)
    fig.text(.5, -.03, "Test episodes; error bars = ±1 s.e. across 10 episodes. Local codecs refit per offset.",
             ha="center", color=MUTED, fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "fig3_chunk_phase.png"); plt.close(fig)


def fig_h1(frozen):
    t = pd.read_csv(RES / "test_episodes.csv")
    ms = ["released_fast+", frozen["control"], frozen["weighted"]]
    fig, ax = plt.subplots(figsize=(7.5, 3.8)); x = np.arange(len(ms)); w = .36
    for i, (col, c, lab) in enumerate((("nonevent_grip_mae", "#c3cad3", "gripper, calm frames"),
                                       ("event_mae", COL["weighted"], "gripper, event frames"))):
        g = [t[t.method == m][col] for m in ms]
        ax.bar(x + (i - .5) * w, [v.mean() for v in g], w, color=c, label=lab,
               yerr=[v.std() / np.sqrt(len(v)) for v in g], capsize=3, ecolor=MUTED)
    ax.set_xticks(x, [short(m) for m in ms]); ax.set_ylabel("MAE (normalized)")
    ax.set_title("H1 — is error concentrated around rapid gripper commands?"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig_h1_concentration.png"); plt.close(fig)


def fig_forest():
    b = json.load(open(RES / "test_bootstrap.json"))["weighted_vs_control"]
    labs = {"event_mae": "event MAE", "event_p95": "event 95th pct", "event_diff_err": "event speed error",
            "arm_mae": "arm MAE", "tokens_per_s": "tokens / s"}
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    for i, (k, lab) in enumerate(labs.items()):
        r = b[k]; base = r["mean_diff"] / r["rel_diff"] if r["rel_diff"] else 1
        lo, hi, m = 100 * r["ci_lo"] / base, 100 * r["ci_hi"] / base, 100 * r["rel_diff"]
        good = (hi < 0) if k != "tokens_per_s" else None
        c = COL["control"] if k == "tokens_per_s" else (COL["weighted"] if good else MUTED)
        ax.plot([lo, hi], [i, i], color=c, lw=3, solid_capstyle="round"); ax.plot(m, i, "o", color=c, ms=8)
        ax.annotate(f"{m:+.0f}%", (m, i), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5, color=INK)
    ax.set_ylim(len(labs) - .4, -.8); ax.axvline(0, color=INK, lw=1); ax.set_yticks(range(len(labs)), list(labs.values()))
    ax.set_xlabel("weighted − control, % of control  (95% episode-bootstrap CI)")
    ax.set_title("Held-out test: what the weighting bought, and what it cost", pad=12)
    fig.tight_layout(); fig.savefig(FIG / "fig4_test_effects.png"); plt.close(fig)


def fig_insertion():
    f = RES / "insertion_results.json"
    if not f.exists():
        return
    r = json.load(open(f))
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), sharey=False)
    for a, setting, title in ((ax[0], "within_task", "Insertion task — refit on insertion data"),
                              (ax[1], "transfer", "Insertion task — cube setup frozen (transfer)")):
        s = r["summary"][setting]; roles = ["released", "control", "weighted"]
        x = np.arange(3); w = .38
        a.bar(x - w / 2, [s[k]["event_mae"] for k in roles], w, color=[COL[k] for k in roles], label="gripper, events")
        a.bar(x + w / 2, [s[k]["arm_mae"] for k in roles], w, color=[COL[k] for k in roles], alpha=.35, hatch="//",
              edgecolor="white", label="arm, all frames")
        for i, k in enumerate(roles):
            a.text(i, max(s[k]["event_mae"], s[k]["arm_mae"]) * 1.04, f"{s[k]['tokens_per_s']:.0f} tok/s",
                   ha="center", fontsize=8, color=MUTED)
        a.set_ylim(0, max(max(s[k]["event_mae"], s[k]["arm_mae"]) for k in roles) * 1.18)
        b = r["weighted_vs_control"][setting]["event_mae"]
        a.set_title(f"{title}\nweighted vs control event MAE: {100 * b['rel_diff']:+.0f}% "
                    f"({b['n_better']}/{b['n_episodes']} episodes better)", fontsize=10)
        a.set_xticks(x, ["released FAST+", "control γ5 λ1", "weighted γ5 λ2"]); a.set_ylabel("MAE (normalized)")
    ax[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "fig5_second_task.png"); plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    eps, _ = load_episodes(); split = load_split(); norm = json.load(open(CFG / "norm.json"))
    frozen = json.load(open(CFG / "frozen.json")); ecfg = json.load(open(CFG / "events.json"))
    Z = {e: normalize(a, norm) for e, a in eps.items()}; EM = {e: ev.event_mask(Z[e], ecfg) for e in Z}
    tr = list(np.concatenate([chunk_episode(Z[e])[0] for e in split["train"]]))
    parse = lambda n: (int(n.split("_")[1][1:]), float(n.split("_")[2][1:]))
    codecs = {"released": load_released(),
              "control": fit_local(tr, *parse(frozen["control"]), frozen["vocab_size"]),
              "weighted": fit_local(tr, *parse(frozen["weighted"]), frozen["vocab_size"])}
    fig_pipeline(); fig_how_dct_works(Z, split); fig_frontier(frozen)
    fig_overlays(Z, eps, norm, split, EM, codecs); fig_phase(); fig_h1(frozen); fig_forest(); fig_insertion()
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
