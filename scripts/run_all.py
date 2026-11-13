"""One command reproduces everything:  python -m scripts.run_all

Stages: data -> preprocessing/events -> validation sweep -> freeze -> held-out test (once)
-> paired bootstrap -> chunk-phase audit -> error audit -> timing -> figures."""
import json, subprocess, sys
import numpy as np
from tokenizers.pre_tokenizers import ByteLevel
import pandas as pd
from src.data import load_episodes, load_split, CFG, ROOT
from src.preprocess import normalize, chunk_episode, common_frame_mask
from src import events as ev
from src.codecs import load_released, fit_local, FAST_REVISION
from src.evaluate import reconstruct_episode, episode_metrics, bootstrap_paired, time_codec

RES = ROOT / "results"
GAMMAS, LAMBDAS, OFFSETS = (5, 10, 20), (1, 2, 4), (0, 10, 20, 30, 40)
VOCAB_PREFERRED, VOCAB_SHARED = 1024, 2048
SELECTION_RULE = (
    "Written before any validation result was seen. Pair every weighted config (lambda>1) with every "
    "local control (lambda=1) whose mean validation token rate is within 5%. A pair is eligible if the "
    "weighted arm MAE is at most 5% above the control's. Choose the eligible pair with the largest relative "
    "event-MAE reduction. If no pair is eligible, choose the weighted config with the lowest validation "
    "event MAE, pair it with the control of nearest token rate, and mark the comparison as NOT rate-matched. "
    "Practical target (reported, not required): >=10% event-MAE reduction.")


def stage(cmd):
    print(f"\n=== {cmd}", flush=True)
    subprocess.run([sys.executable, "-m", cmd], check=True, cwd=ROOT)


def evaluate(codec, Z, episodes, emasks, offset=0, score_fn=None):
    rows = []
    for e in episodes:
        r = reconstruct_episode(codec, Z[e], offset)
        score = score_fn(len(Z[e])) if score_fn else ~np.isnan(r["recon"]).any(1)
        m = episode_metrics(Z[e], r["recon"], emasks[e], score)
        sec = r["n_chunks"] * 50 / 50
        rows.append({"method": codec.name, "episode": e, "offset": offset, "tokens_per_s": r["n_tokens"] / sec,
                     "n_failed_chunks": r["n_failed"], **m})
    return rows


def train_chunks(Z, split, offset):
    return list(np.concatenate([chunk_episode(Z[e], 50, offset)[0] for e in split["train"]]))


def main():
    import time, resource, platform
    t_start = time.time()
    # always run: downloads the pinned revision if data is missing; never regenerates an existing split
    stage("src.data")
    stage("scripts.prep_report")

    eps, _ = load_episodes(); split = load_split()
    norm = json.load(open(CFG / "norm.json")); ecfg = json.load(open(CFG / "events.json"))
    Z = {e: normalize(a, norm) for e, a in eps.items()}
    EM = {e: ev.event_mask(Z[e], ecfg) for e in Z}
    tr0 = train_chunks(Z, split, 0)
    released = load_released()

    # ---------- chunk 10: feasibility at the preferred vocab, then fit all at the shared vocab ----------
    feas = []
    for g in GAMMAS:
        for l in LAMBDAS:
            try:
                fit_local(tr0, g, l, VOCAB_PREFERRED); ok, msg = True, ""
            except AssertionError as err:
                ok, msg = False, str(err)
            feas.append({"gamma": g, "lambda": l, "vocab": VOCAB_PREFERRED, "feasible": ok, "message": msg})
    json.dump({"preferred_vocab": feas, "decision": f"{VOCAB_PREFERRED} rejected for at least one config -> "
               f"all local configs use shared vocab {VOCAB_SHARED} (same as released FAST+)"
               if not all(f["feasible"] for f in feas) else f"all feasible at {VOCAB_PREFERRED}"},
              open(RES / "vocab_feasibility.json", "w"), indent=2)
    vocab = VOCAB_SHARED if not all(f["feasible"] for f in feas) else VOCAB_PREFERRED
    codecs = {(g, l): fit_local(tr0, g, l, vocab) for g in GAMMAS for l in LAMBDAS}

    # ---------- chunk 13: validation sweep ----------
    rows = evaluate(released, Z, split["val"], EM)
    for (g, l), c in codecs.items():
        for r in evaluate(c, Z, split["val"], EM):
            rows.append({**r, "gamma": g, "lambda": l})
    val = pd.DataFrame(rows); val.to_csv(RES / "val_episodes.csv", index=False)
    vagg = val.groupby("method", sort=False).mean(numeric_only=True).reset_index()
    vagg.to_csv(RES / "val_sweep.csv", index=False)

    # ---------- chunk 14: freeze (never re-select once frozen) ----------
    fz_path = CFG / "frozen.json"
    if fz_path.exists():
        frozen = json.load(open(fz_path))
    else:
        loc = vagg[vagg["gamma"].notna()].set_index("method")
        ctrl, wtd = loc[loc["lambda"] == 1], loc[loc["lambda"] > 1]
        cands = []
        for wn, w in wtd.iterrows():
            for cn, c in ctrl.iterrows():
                rate_gap = abs(w.tokens_per_s - c.tokens_per_s) / c.tokens_per_s
                arm_inc = (w.arm_mae - c.arm_mae) / c.arm_mae
                red = (c.event_mae - w.event_mae) / c.event_mae
                cands.append(dict(weighted=wn, control=cn, rate_gap=rate_gap, arm_increase=arm_inc,
                                  event_reduction=red, eligible=bool(rate_gap <= .05 and arm_inc <= .05)))
        cdf = pd.DataFrame(cands); cdf.to_csv(RES / "val_pair_candidates.csv", index=False)
        el = cdf[cdf.eligible]
        if len(el):
            best = el.sort_values("event_reduction", ascending=False).iloc[0]; matched = True
        else:
            wn = wtd["event_mae"].idxmin()
            sub = cdf[cdf.weighted == wn]; best = sub.loc[sub.rate_gap.idxmin()]; matched = False
        frozen = {"weighted": best.weighted, "control": best.control, "rate_matched": matched,
                  "val_rate_gap": float(best.rate_gap), "val_arm_increase": float(best.arm_increase),
                  "val_event_reduction": float(best.event_reduction),
                  "meets_10pct_target": bool(matched and best.event_reduction >= .10),
                  "vocab_size": vocab, "selection_rule": SELECTION_RULE}
        json.dump(frozen, open(fz_path, "w"), indent=2)
    name2key = {c.name: k for k, c in codecs.items()}
    W, C = codecs[name2key[frozen["weighted"]]], codecs[name2key[frozen["control"]]]
    print("frozen:", frozen)

    # ---------- chunk 15: held-out test, run once over every config (full table) ----------
    rows = evaluate(released, Z, split["test"], EM)
    for (g, l), c in codecs.items():
        for r in evaluate(c, Z, split["test"], EM):
            rows.append({**r, "gamma": g, "lambda": l})
    test = pd.DataFrame(rows); test.to_csv(RES / "test_episodes.csv", index=False)
    tagg = test.groupby("method", sort=False).mean(numeric_only=True).reset_index()
    tagg.to_csv(RES / "test_summary.csv", index=False)

    # ---------- chunk 16: paired episode bootstrap ----------
    def ep(method, col): return test[test.method == method].sort_values("episode")[col].to_numpy()
    boot = {}
    for a, b, tag in [(W.name, C.name, "weighted_vs_control"), (W.name, released.name, "weighted_vs_released"),
                      (C.name, released.name, "control_vs_released")]:
        boot[tag] = {col: bootstrap_paired(ep(a, col), ep(b, col)) for col in
                     ("event_mae", "event_p95", "event_diff_err", "arm_mae", "tokens_per_s")}
    # same-gamma pure precision effect (lambda>1 vs lambda=1 at identical gamma)
    for g in GAMMAS:
        for l in (2, 4):
            boot[f"same_gamma_g{g}_l{l}_vs_l1"] = {col: bootstrap_paired(ep(codecs[(g, l)].name, col),
                                                                           ep(codecs[(g, 1)].name, col))
                                                   for col in ("event_mae", "arm_mae", "tokens_per_s")}
    # H1: event vs non-event gripper error, per method
    boot["H1_event_vs_nonevent"] = {m: bootstrap_paired(ep(m, "event_mae"), ep(m, "nonevent_grip_mae"))
                                    for m in (released.name, C.name, W.name)}
    json.dump(boot, open(RES / "test_bootstrap.json", "w"), indent=2)

    # ---------- chunk 17: chunk-phase audit (refit locals per offset, score common frames) ----------
    prow = []
    for off in OFFSETS:
        tro = train_chunks(Z, split, off)
        Co = fit_local(tro, C.scale, C.w[6], vocab); Wo = fit_local(tro, W.scale, W.w[6], vocab)
        for codec, label in ((released, "released_fast+"), (Co, "control"), (Wo, "weighted")):
            for r in evaluate(codec, Z, split["test"], EM, off, common_frame_mask):
                prow.append({**r, "label": label})
    phase = pd.DataFrame(prow); phase.to_csv(RES / "phase_episodes.csv", index=False)
    pagg = phase.groupby(["label", "offset"]).mean(numeric_only=True).reset_index()
    pagg.to_csv(RES / "phase_summary.csv", index=False)
    spread = {l: {"event_mae_spread": float(g.event_mae.max() - g.event_mae.min()),
                  "event_mae_rel_spread": float((g.event_mae.max() - g.event_mae.min()) / g.event_mae.mean()),
                  "best_offset": int(g.loc[g.event_mae.idxmin(), "offset"]),
                  "worst_offset": int(g.loc[g.event_mae.idxmax(), "offset"])}
              for l, g in pagg.groupby("label")}
    # does weighted beat control at every offset? (per-offset paired bootstrap)
    for off in OFFSETS:
        p = phase[phase.offset == off]
        a = p[p.label == "weighted"].sort_values("episode").event_mae.to_numpy()
        b = p[p.label == "control"].sort_values("episode").event_mae.to_numpy()
        spread[f"weighted_vs_control_offset{off}"] = bootstrap_paired(a, b)
    json.dump(spread, open(RES / "phase_spread.json", "w"), indent=2)

    # ---------- chunk 18: error audit + timing ----------
    audit = {"decode_failures_test": test.groupby("method").n_failed_chunks.sum().to_dict(),
             "codec_stats": {c.name: c.stats for c in [released, C, W]}}
    worst = []
    for codec in (released, C, W):
        for e in split["test"]:
            rec = reconstruct_episode(codec, Z[e])["recon"]
            err = np.abs(Z[e] - rec)
            for j in (6, 13):
                for t in np.flatnonzero(EM[e]):
                    worst.append({"method": codec.name, "episode": e, "frame": int(t), "channel": j,
                                  "abs_err": float(err[t, j]), "z": float(Z[e][t, j]),
                                  "out_of_range": bool(abs(Z[e][t, j]) > 1)})
    wdf = pd.DataFrame(worst)
    audit["top10_event_errors"] = wdf.sort_values("abs_err", ascending=False).head(10).to_dict("records")
    audit["event_err_by_out_of_range"] = wdf.groupby(["method", "out_of_range"]).abs_err.mean().unstack().to_dict()
    audit["worst_episode_by_method"] = {m: int(g.loc[g.event_mae.idxmax(), "episode"])
                                        for m, g in test.groupby("method")}
    # byte-coverage diagnostic: how often would the UNMODIFIED official fit corrupt held-out chunks?
    bug = {}
    for (g, l) in codecs:
        off = fit_local(tr0, g, l, vocab, official_fit=True)
        held = np.concatenate([chunk_episode(Z[e])[0] for e in split["val"] + split["test"]])
        off.roundtrip(held)
        bug[off.name] = {"heldout_chunks": len(held), "corrupted_or_failed": off.stats["lossy_bpe"] + off.stats["decode_failures"],
                         "missing_byte_symbols": len(set(ByteLevel.alphabet()) - set(off.proc.bpe_tokenizer.get_vocab()))}
    audit["official_fit_byte_coverage_bug"] = bug
    vchunks = np.concatenate([chunk_episode(Z[e])[0] for e in split["val"]])
    import platform
    audit["timing"] = {c.name: time_codec(c, vchunks) for c in (released, C, W)}
    audit["timing_hardware"] = {"machine": platform.machine(), "platform": platform.platform(),
                                "processor": platform.processor()}
    json.dump(audit, open(RES / "audit.json", "w"), indent=2, default=str)

    stage("scripts.transfer")
    stage("src.plots")

    # ---------- run manifest: wall time, peak memory, revisions ----------
    to_mb = (lambda b: b / 2**20) if sys.platform == "darwin" else (lambda kb: kb / 2**10)  # ru_maxrss units
    manifest = {"wall_time_s": round(time.time() - t_start, 1),
                "peak_rss_mb_main": round(to_mb(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss), 1),
                "peak_rss_mb_largest_child": round(to_mb(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss), 1),
                "hardware": {"machine": platform.machine(), "platform": platform.platform(),
                             "cpu_count": __import__("os").cpu_count()},
                "python": sys.version.split()[0],
                "dataset": json.load(open(CFG / "dataset.json")),
                "dataset_insertion": json.load(open(CFG / "dataset_insertion.json")),
                "fast_revision": FAST_REVISION,
                "frozen": {k: v for k, v in frozen.items() if k != "selection_rule"}}
    json.dump(manifest, open(RES / "run_manifest.json", "w"), indent=2)
    print(json.dumps(manifest, indent=1))
    print("\nDone. Results in results/, figures in results/figures/")


if __name__ == "__main__":
    main()
