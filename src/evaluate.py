"""Episode-level metrics, codec application, and paired episode bootstrap."""
import time
import numpy as np
from src.preprocess import chunk_episode
from src.codecs import GRIP

ARM = [j for j in range(14) if j not in GRIP]
FPS, H = 50, 50


def reconstruct_episode(codec, z, offset=0):
    """Encode/decode every full chunk of one episode. Uncovered frames stay NaN."""
    chunks, starts = chunk_episode(z, H, offset)
    r = codec.roundtrip(chunks)
    rec = np.full_like(z, np.nan)
    for c, s in zip(r["recon"], starts):
        rec[s:s + H] = c
    return {"recon": rec, "n_tokens": int(r["n_tokens"].sum()), "n_chunks": len(starts),
            "n_failed": int(r["failed"].sum()), "encode_s": r["encode_s"], "decode_s": r["decode_s"]}


def episode_metrics(z, rec, emask, score):
    """z, rec: (T,14) normalized. emask: event frames. score: frames to score (bool)."""
    err = np.abs(z - rec)
    E = emask & score
    G = list(GRIP)
    nonE = score & ~emask
    out = {
        "event_mae": float(np.nanmean(err[E][:, G])) if E.any() else np.nan,
        "event_p95": float(np.nanpercentile(err[E][:, G], 95)) if E.any() else np.nan,
        "nonevent_grip_mae": float(np.nanmean(err[nonE][:, G])),
        "grip_mae": float(np.nanmean(err[score][:, G])),
        "arm_mae": float(np.nanmean(err[score][:, ARM])),
        "n_event_frames": int(E.sum()),
    }
    # first-difference error on event frames whose predecessor is also scored
    dz, dr = np.diff(z, axis=0), np.diff(rec, axis=0)
    ok = E[1:] & score[:-1]
    out["event_diff_err"] = float(np.nanmean(np.abs(dz - dr)[ok][:, G])) if ok.any() else np.nan
    return out


def bootstrap_paired(a, b, n=2000, seed=0):
    """Paired episode bootstrap of mean(a - b). a, b: per-episode arrays (NaNs dropped pairwise)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    keep = ~(np.isnan(a) | np.isnan(b)); d = a[keep] - b[keep]
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n, len(d)))].mean(1)
    rel = (a[keep].mean() - b[keep].mean()) / b[keep].mean()
    return {"mean_diff": float(d.mean()), "ci_lo": float(np.percentile(boots, 2.5)),
            "ci_hi": float(np.percentile(boots, 97.5)), "rel_diff": float(rel),
            "n_episodes": int(len(d)), "n_better": int((d < 0).sum())}


def time_codec(codec, chunks, batch=32, reps=3):
    """CPU encode/decode ms per chunk after warm-up."""
    b = chunks[:batch]
    codec.roundtrip(b)  # warm-up
    enc, dec = [], []
    for _ in range(reps):
        t0 = time.perf_counter(); toks = codec.encode(b); t1 = time.perf_counter()
        codec.decode(toks); t2 = time.perf_counter()
        enc.append(t1 - t0); dec.append(t2 - t1)
    return {"encode_ms_per_chunk": 1e3 * min(enc) / len(b), "decode_ms_per_chunk": 1e3 * min(dec) / len(b),
            "batch": len(b)}
