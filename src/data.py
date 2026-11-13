"""Download action-only files at a pinned revision, validate schema, make splits, load episodes."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data_cache"
CFG = ROOT / "configs"
REPO = "lerobot/aloha_sim_transfer_cube_human"


def download(repo=REPO, revision=None, cfg_name="dataset.json"):
    """Fetch only meta/** and data/** at a pinned revision; record it in configs/<cfg_name>."""
    if revision is None and (CFG / cfg_name).exists():
        revision = json.load(open(CFG / cfg_name))["revision"]  # keep the pinned revision
    revision = revision or HfApi().dataset_info(repo).sha
    path = snapshot_download(repo, repo_type="dataset", revision=revision,
                             allow_patterns=["meta/**", "data/**"],
                             local_dir=DATA_DIR / repo.split("/")[1])
    json.dump({"repo": repo, "revision": revision}, open(CFG / cfg_name, "w"), indent=2)
    return Path(path), revision


def dataset_path(cfg_name="dataset.json"):
    return DATA_DIR / json.load(open(CFG / cfg_name))["repo"].split("/")[1]


def load_info(path=None):
    return json.load(open((path or dataset_path()) / "meta" / "info.json"))


def action_names(info=None):
    info = info or load_info()
    names = info["features"]["action"]["names"]
    if isinstance(names, dict):  # e.g. {"motors": [...]}
        names = next(iter(names.values()))
    return list(names)


def gripper_indices(names):
    return [i for i, n in enumerate(names) if "gripper" in n.lower()]


def load_episodes(path=None):
    """Return {episode_index: (T,14) float64 action array}, sorted by frame_index."""
    files = sorted(((path or dataset_path()) / "data").rglob("*.parquet"))
    df = pd.concat([pd.read_parquet(f, columns=["episode_index", "frame_index", "timestamp", "action"])
                    for f in files])
    eps = {}
    for e, g in df.groupby("episode_index"):
        g = g.sort_values("frame_index")
        assert (np.diff(g["frame_index"].to_numpy()) == 1).all(), f"gap in episode {e}"
        eps[int(e)] = np.stack(g["action"].to_numpy()).astype(np.float64)
    return eps, df


def validate(eps, df, info):
    names = action_names(info)
    g = gripper_indices(names)
    rep = {"n_episodes": len(eps), "n_frames": int(sum(len(a) for a in eps.values())),
           "fps": info["fps"], "action_dim": len(names), "action_names": names, "gripper_idx": g,
           "episode_lengths": sorted({len(a) for a in eps.values()})}
    dt = df.groupby("episode_index")["timestamp"].diff().dropna()
    rep["timestamp_dt_range"] = [float(dt.min()), float(dt.max())]
    assert rep["action_dim"] == 14 and g == [6, 13], rep
    assert all(a.shape[1] == 14 for a in eps.values())
    assert np.allclose(dt, 1 / info["fps"], atol=1e-4)
    return rep


def make_split(episode_ids, seed=0, n=(30, 10, 10)):
    ids = np.array(sorted(episode_ids))
    assert len(ids) == sum(n)
    perm = np.random.default_rng(seed).permutation(ids)
    s = {"seed": seed, "train": sorted(perm[:n[0]].tolist()),
         "val": sorted(perm[n[0]:n[0] + n[1]].tolist()), "test": sorted(perm[n[0] + n[1]:].tolist())}
    assert not (set(s["train"]) & set(s["val"]) or set(s["train"]) & set(s["test"]) or set(s["val"]) & set(s["test"]))
    return s


def load_split():
    return json.load(open(CFG / "split.json"))


if __name__ == "__main__":
    _, rev = download()
    eps, df = load_episodes()
    rep = validate(eps, df, load_info())
    json.dump(rep, open(ROOT / "results" / "schema_report.json", "w"), indent=2)
    if not (CFG / "split.json").exists():  # lock once; never regenerate
        json.dump(make_split(eps.keys()), open(CFG / "split.json", "w"), indent=2)
    print("revision", rev); print(json.dumps({k: v for k, v in rep.items() if k != "action_names"}))
    print(rep["action_names"]); print(load_split())
