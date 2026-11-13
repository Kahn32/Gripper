"""Optional second task: ALOHA insertion. Run AFTER run_all (needs configs/frozen.json).

Two separately labelled evaluations on the insertion test split (independent 30/10/10, seed 0):
  transfer     -- cube-task normalization, event thresholds and tokenizers, all frozen
  within_task  -- normalization, events and tokenizers refit on insertion training episodes
                  (same frozen gamma/lambda choice; nothing re-selected)"""
import json
import numpy as np
import pandas as pd
from src.data import (download, load_episodes, load_info, validate, make_split, load_split,
                      dataset_path, CFG, ROOT)
from src.preprocess import fit_norm, normalize, chunk_episode
from src import events as ev
from src.codecs import load_released, fit_local
from src.evaluate import bootstrap_paired
from scripts.run_all import evaluate

RES = ROOT / "results"
REPO, CFGN = "lerobot/aloha_sim_insertion_human", "dataset_insertion.json"


def parse(n):
    return int(n.split("_")[1][1:]), float(n.split("_")[2][1:])


def run_block(label, Z, split_ids, EM, codecs):
    rows = []
    for name, c in codecs.items():
        for r in evaluate(c, Z, split_ids, EM):
            rows.append({**r, "setting": label, "role": name})
    return rows


def main():
    path, rev = download(REPO, cfg_name=CFGN)
    eps, df = load_episodes(path)
    schema = validate(eps, df, load_info(path))
    sp_path = CFG / "split_insertion.json"
    if not sp_path.exists():
        json.dump(make_split(eps.keys(), seed=0), open(sp_path, "w"), indent=2)
    sp = json.load(open(sp_path))
    frozen = json.load(open(CFG / "frozen.json")); vocab = frozen["vocab_size"]
    gC, lC = parse(frozen["control"]); gW, lW = parse(frozen["weighted"])
    released = load_released()

    # ---- transfer: everything from the cube task, frozen ----
    cube_norm = json.load(open(CFG / "norm.json")); cube_ev = json.load(open(CFG / "events.json"))
    cube_eps = load_episodes()[0]; cube_sp = load_split()
    Zc = {e: normalize(a, cube_norm) for e, a in cube_eps.items()}
    cube_tr = list(np.concatenate([chunk_episode(Zc[e])[0] for e in cube_sp["train"]]))
    Zt = {e: normalize(a, cube_norm) for e, a in eps.items()}
    EMt = {e: ev.event_mask(Zt[e], cube_ev) for e in Zt}
    rows = run_block("transfer", Zt, sp["test"], EMt, {
        "released": released, "control": fit_local(cube_tr, gC, lC, vocab), "weighted": fit_local(cube_tr, gW, lW, vocab)})

    # ---- within-task: refit preprocessing + tokenizers on insertion train ----
    norm = fit_norm({e: eps[e] for e in sp["train"]})
    Zw = {e: normalize(a, norm) for e, a in eps.items()}
    ecfg = ev.fit_thresholds([Zw[e] for e in sp["train"]])
    EMw = {e: ev.event_mask(Zw[e], ecfg) for e in Zw}
    tr = list(np.concatenate([chunk_episode(Zw[e])[0] for e in sp["train"]]))
    rows += run_block("within_task", Zw, sp["test"], EMw, {
        "released": load_released(), "control": fit_local(tr, gC, lC, vocab), "weighted": fit_local(tr, gW, lW, vocab)})

    df_ = pd.DataFrame(rows); df_.to_csv(RES / "insertion_test_episodes.csv", index=False)
    out = {"dataset": REPO, "revision": rev, "split": sp, "episode_length": schema["episode_lengths"],
           "summary": {}, "weighted_vs_control": {}, "event_stats": {}}
    for setting, (Z, EM) in {"transfer": (Zt, EMt), "within_task": (Zw, EMw)}.items():
        d = df_[df_.setting == setting]
        out["summary"][setting] = d.groupby("role")[["tokens_per_s", "event_mae", "event_p95", "arm_mae",
                                                     "nonevent_grip_mae", "n_failed_chunks"]].mean().to_dict("index")
        ep = lambda role, col: d[d.role == role].sort_values("episode")[col].to_numpy()
        out["weighted_vs_control"][setting] = {c: bootstrap_paired(ep("weighted", c), ep("control", c))
                                               for c in ("event_mae", "event_p95", "arm_mae", "tokens_per_s")}
        out["event_stats"][setting] = {
            "events_per_test_episode": float(np.mean([len(ev.detect(Z[e], cube_ev if setting == "transfer" else ecfg))
                                                      for e in sp["test"]])),
            "frac_frames_in_E": float(np.mean([EM[e].mean() for e in sp["test"]])),
            "frac_out_of_range_test": float(np.mean([(np.abs(Z[e]) > 1).mean() for e in sp["test"]]))}
    json.dump(out, open(RES / "insertion_results.json", "w"), indent=2)
    print(json.dumps({k: out[k] for k in ("summary", "event_stats")}, indent=1))
    for s, v in out["weighted_vs_control"].items():
        print(s, {c: (round(r["rel_diff"], 3), round(r["ci_lo"], 5), round(r["ci_hi"], 5), r["n_better"]) for c, r in v.items()})


if __name__ == "__main__":
    main()
