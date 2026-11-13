import numpy as np
import pytest
from src.data import make_split
from src.preprocess import chunk_episode, chunk_starts, common_frame_mask, fit_norm, normalize, denormalize
from src import events as ev
from src.codecs import ref_quantize, channel_weights

rng = np.random.default_rng(0)


def test_split_disjoint_and_complete():
    s = make_split(range(50), seed=0)
    assert len(s["train"]) == 30 and len(s["val"]) == 10 and len(s["test"]) == 10
    assert set(s["train"]) | set(s["val"]) | set(s["test"]) == set(range(50))
    assert make_split(range(50), seed=0) == s  # deterministic


def test_chunks_never_cross_boundary():
    for off in (0, 10, 20, 30, 40):
        st = chunk_starts(400, 50, off)
        assert all(s >= 0 and s + 50 <= 400 for s in st)
        z = rng.normal(size=(400, 14))
        c, _ = chunk_episode(z, 50, off)
        assert np.array_equal(c[0], z[off:off + 50])
    assert common_frame_mask(400).sum() == 320


def test_norm_roundtrip():
    a = {0: rng.normal(size=(100, 14))}
    n = fit_norm(a)
    assert np.allclose(denormalize(normalize(a[0], n), n), a[0])


def test_event_mask_window_and_merge():
    z = np.zeros((60, 14)); z[30:, 6] = 1.0  # single step at t=30
    cfg = {"thresholds": {"6": 0.5, "13": 0.5}, "window": 3}
    d = ev.detect(z, cfg)
    assert d == [(6, 30, 30)]
    m = ev.event_mask(z, cfg)
    assert np.flatnonzero(m).tolist() == list(range(27, 34))
    z[32:, 6] = 2.0  # second step 2 frames later -> merged
    assert len(ev.detect(z, cfg)) == 1


def test_weighting_reversible_before_quantization():
    z = rng.normal(size=(4, 50, 14)); w = channel_weights(4.0)
    assert np.allclose((z * w) / w, z)
    # with huge scale quantization is ~lossless => weighted path recovers z
    rec, _ = ref_quantize(z, 1e6, w)
    assert np.abs(rec - z).max() < 1e-5


def test_weighting_refines_gripper_quantization():
    z = rng.normal(size=(64, 50, 14)) * 0.3
    e1 = np.abs(ref_quantize(z, 10)[0] - z)
    e4 = np.abs(ref_quantize(z, 10, channel_weights(4.0))[0] - z)
    assert e4[..., [6, 13]].mean() < 0.5 * e1[..., [6, 13]].mean()
    assert np.allclose(e4[..., :6], e1[..., :6])  # arm untouched by the weighting itself


def test_paired_bootstrap_sign():
    from src.evaluate import bootstrap_paired
    a = np.array([1., 2, 3, 4]); r = bootstrap_paired(a - 0.5, a)
    assert r["mean_diff"] == -0.5 and r["ci_lo"] == r["ci_hi"] == -0.5 and r["n_better"] == 4
