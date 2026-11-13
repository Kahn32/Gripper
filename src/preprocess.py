"""Training-only quantile normalization and episode-local chunking."""
import numpy as np

DEN_FLOOR = 1e-3  # documented floor for (q99 - q01) on near-constant channels


def fit_norm(train_eps, lo=0.01, hi=0.99):
    a = np.concatenate([train_eps[e] for e in sorted(train_eps)])
    q01, q99 = np.quantile(a, lo, axis=0), np.quantile(a, hi, axis=0)
    den = q99 - q01
    floored = (den < DEN_FLOOR).tolist()
    return {"q01": q01.tolist(), "q99": q99.tolist(), "den": np.maximum(den, DEN_FLOOR).tolist(),
            "floored_channels": [j for j, f in enumerate(floored) if f], "den_floor": DEN_FLOOR}


def normalize(a, norm):
    return 2 * (a - np.asarray(norm["q01"])) / np.asarray(norm["den"]) - 1


def denormalize(z, norm):
    return (z + 1) / 2 * np.asarray(norm["den"]) + np.asarray(norm["q01"])


def chunk_starts(T, H=50, offset=0):
    """Starts of non-overlapping full chunks within one episode."""
    return list(range(offset, T - H + 1, H))


def chunk_episode(z, H=50, offset=0):
    """(n_chunks, H, D) array plus starts; never crosses the episode boundary."""
    s = chunk_starts(len(z), H, offset)
    return np.stack([z[i:i + H] for i in s]) if s else np.zeros((0, H, z.shape[1])), s


def covered_mask(T, H=50, offset=0):
    m = np.zeros(T, bool)
    for s in chunk_starts(T, H, offset):
        m[s:s + H] = True
    return m


def common_frame_mask(T, H=50, offsets=(0, 10, 20, 30, 40)):
    """Frames covered by a chunk under every offset (scored set for the phase audit)."""
    return np.logical_and.reduce([covered_mask(T, H, o) for o in offsets])
