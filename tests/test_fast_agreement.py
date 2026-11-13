"""Chunk 9: official FAST+ round-trip must agree with the reference DCT quantizer."""
import numpy as np
import pytest
from src.codecs import load_released, ref_quantize


@pytest.fixture(scope="module")
def codec():
    return load_released()


def _chunks():
    t = np.linspace(0, 1, 50)
    smooth = np.stack([0.5 * np.sin(2 * np.pi * (0.5 + 0.1 * j) * t + j) for j in range(14)], 1)
    ramp = smooth.copy(); ramp[:, [6, 13]] = (np.clip((t - .5) / .06, 0, 1) * 1.6 - .8)[:, None]
    return np.stack([smooth, ramp])


def test_official_matches_reference(codec):
    z = _chunks()
    r = codec.roundtrip(z)
    ref, _ = ref_quantize(z, codec.scale)
    assert not r["failed"].any() and np.isfinite(r["recon"]).all()
    assert codec.stats["lossy_bpe"] == 0 and codec.stats["clipped_coeffs"] == 0
    assert np.abs(r["recon"] - ref).max() < 1e-9


def test_decode_failure_is_flagged_not_zero(codec):
    toks = codec.encode(_chunks())
    bad = [toks[0][:-3], toks[1]]  # truncated sequence -> wrong coefficient count
    rec, failed = codec.decode(bad)
    assert failed.tolist() == [True, False]
    assert np.isnan(rec[0]).all() and np.isfinite(rec[1]).all()


def test_weighted_local_codec_matches_reference():
    """Weight -> official fit/encode/decode -> unweight equals round(γ w c)/(γ w) exactly."""
    from src.codecs import fit_local, channel_weights
    rng = np.random.default_rng(1)
    t = np.linspace(0, 1, 50)[None, :, None]
    train = list(np.tanh(rng.normal(size=(64, 1, 14)) + np.sin(6 * t * rng.uniform(.5, 2, (64, 1, 14)))))
    z = np.stack(train[:8])
    for lam in (1.0, 4.0):
        c = fit_local(train, gamma=10, lam=lam, vocab_size=2048)
        r = c.roundtrip(z)
        ref, _ = ref_quantize(z, 10, channel_weights(lam))
        assert c.stats["lossy_bpe"] == 0 and not r["failed"].any()
        assert np.abs(r["recon"] - ref).max() < 1e-9


def test_full_alphabet_fit_is_lossless_beyond_training_range():
    """Regression test for the byte-coverage bug: a held-out coefficient just outside the
    training range must round-trip exactly with our fit, and is corrupted by the official fit."""
    from src.codecs import fit_local
    from tokenizers.pre_tokenizers import ByteLevel
    rng = np.random.default_rng(2)
    train = list(rng.uniform(-.5, .5, size=(40, 50, 14)))
    z = np.zeros((1, 50, 14)); z[0, :, 6] = 2.0  # DC coefficient far above training range
    ours = fit_local(train, gamma=5, vocab_size=2048)
    assert set(ByteLevel.alphabet()) <= set(ours.proc.bpe_tokenizer.get_vocab())
    r = ours.roundtrip(z)
    ref, _ = ref_quantize(z, 5)
    assert not r["failed"].any() and np.abs(r["recon"] - ref).max() < 1e-9
    off = fit_local(train, gamma=5, vocab_size=2048, official_fit=True)
    assert off.roundtrip(z)["failed"].any()  # detected and flagged, not silently used
