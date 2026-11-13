"""Gripper event detection: thresholds fit on training episodes only, frozen thereafter."""
import numpy as np

NOISE_FLOOR = 1e-4  # normalized units; changes at/below this are treated as no motion
WINDOW = 3          # ±3 frames = ±60 ms at 50 Hz
MERGE_GAP = 3       # revision (train-only inspection): detections ≤3 frames apart are one event


def fit_thresholds(train_z, grip=(6, 13), pct=90):
    thr = {}
    for j in grip:
        r = np.concatenate([np.abs(np.diff(z[:, j])) for z in train_z])
        thr[j] = float(max(np.percentile(r[r > NOISE_FLOOR], pct), NOISE_FLOOR))
    return {"thresholds": {str(k): v for k, v in thr.items()}, "pct": pct,
            "noise_floor": NOISE_FLOOR, "window": WINDOW, "merge_gap": MERGE_GAP}


def detect(z, cfg):
    """Return list of events (channel, start, end inclusive) after merging consecutive detections."""
    events = []
    for j, thr in cfg["thresholds"].items():
        j = int(j)
        hit = np.zeros(len(z), bool)
        hit[1:] = np.abs(np.diff(z[:, j])) > thr
        t = 0
        while t < len(z):
            if hit[t]:
                u = t
                while True:
                    nxt = np.flatnonzero(hit[u + 1:u + 2 + MERGE_GAP])
                    if not len(nxt):
                        break
                    u = u + 1 + nxt[-1]
                events.append((j, t, u))
                t = u + 1
            else:
                t += 1
    return events


def event_mask(z, cfg):
    """Union of ±window frames around every event (frames counted once)."""
    m = np.zeros(len(z), bool)
    w = cfg["window"]
    for _, s, e in detect(z, cfg):
        m[max(0, s - w):min(len(z), e + w + 1)] = True
    return m
