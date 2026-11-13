"""Codecs: reference DCT quantizer, released FAST+, local FAST control, weighted local FAST.

All codecs take normalized chunks z of shape (N, H, D) and return reconstructions plus
a per-chunk failure flag. Failed decodes are NaN, never silent zeros."""
import time
import numpy as np
from scipy.fft import dct, idct

FAST_REPO = "physical-intelligence/fast"
FAST_REVISION = "ec4d7aa71691cac0b8bed6942be45684db2110f4"
GRIP = (6, 13)


# ---------- reference implementation (chunk 9) ----------
def ref_quantize(z, scale, w=None):
    """DCT over time, round(scale*w*c)/(scale*w), inverse DCT. z: (N,H,D)."""
    w = np.ones(z.shape[-1]) if w is None else np.asarray(w, float)
    q = np.around(dct(z * w, axis=1, norm="ortho") * scale)
    return idct(q / scale, axis=1, norm="ortho") / w, q


# ---------- FAST-family wrapper ----------
class FastCodec:
    """Wraps an official UniversalActionProcessor with explicit metadata, weighting and failure accounting."""

    def __init__(self, proc, name, weights=None, H=50, D=14):
        self.proc, self.name, self.H, self.D = proc, name, H, D
        self.w = np.ones(D) if weights is None else np.asarray(weights, float)
        self.stats = {"clipped_coeffs": 0, "n_coeffs": 0, "decode_failures": 0, "lossy_bpe": 0}

    @property
    def scale(self):
        return self.proc.scale

    def encode(self, z):
        u = z * self.w
        q = np.around(dct(u, axis=1, norm="ortho") * self.scale)
        self.stats["clipped_coeffs"] += int((q - self.proc.min_token < 0).sum())
        self.stats["n_coeffs"] += q.size
        self._last_q = q
        return self.proc(u.astype(np.float64))

    def decode(self, tokens, expected_q=None):
        """Decode explicitly with (H, D); return (recon (N,H,D) with NaN on failure, failed bool (N,))."""
        out = np.full((len(tokens), self.H, self.D), np.nan)
        failed = np.zeros(len(tokens), bool)
        mt = self.proc.min_token
        for i, tok in enumerate(tokens):
            try:
                s = self.proc.bpe_tokenizer.decode(tok, clean_up_tokenization_spaces=False)
                q = (np.array([ord(c) for c in s]) + mt).reshape(self.H, self.D)
            except Exception:
                failed[i] = True
                continue
            if expected_q is not None and not np.array_equal(q, np.maximum(expected_q[i], mt)):
                self.stats["lossy_bpe"] += 1  # corrupted decode: treat as a failure, never as motion
                failed[i] = True
                continue
            out[i] = idct(q / self.scale, axis=0, norm="ortho") / self.w
        self.stats["decode_failures"] += int(failed.sum())
        return out, failed

    def roundtrip(self, z):
        t0 = time.perf_counter(); toks = self.encode(z); t1 = time.perf_counter()
        rec, failed = self.decode(toks, expected_q=self._last_q); t2 = time.perf_counter()
        return {"recon": rec, "failed": failed, "n_tokens": np.array([len(t) for t in toks]),
                "encode_s": t1 - t0, "decode_s": t2 - t1}


def load_released():
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(FAST_REPO, trust_remote_code=True, revision=FAST_REVISION)
    return FastCodec(proc, "released_fast+")


def channel_weights(lam, D=14, grip=GRIP):
    w = np.ones(D); w[list(grip)] = lam
    return w


_RELEASED_CLS = None


def _fit_full_alphabet(cls, action_data, scale, vocab_size, H, D):
    """Copy of the official UniversalActionProcessor.fit with ONE change: the BPE initial alphabet
    is all 256 ByteLevel symbols. The official version seeds chr(0..range), but ByteLevel splits
    chars >=128 into two UTF-8 byte symbols, so bytes unseen in (small) training data are missing
    from the vocab and are silently dropped at encode time -> U+FFFD -> garbage on held-out data."""
    from tokenizers import ByteLevelBPETokenizer
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer
    from transformers import PreTrainedTokenizerFast
    coeffs = [np.around(dct(a, axis=0, norm="ortho").flatten() * scale) for a in action_data]
    allq = np.concatenate(coeffs)
    min_token, max_token = int(allq.min()), int(allq.max())
    assert max_token - min_token <= vocab_size, \
        f"Vocab size {vocab_size} is too small for the range of tokens {max_token - min_token}"
    bpe = ByteLevelBPETokenizer()
    trainer = BpeTrainer(vocab_size=vocab_size, min_frequency=2, show_progress=False, special_tokens=[],
                         initial_alphabet=ByteLevel.alphabet(), max_token_length=10000)
    bpe._tokenizer.train_from_iterator(("".join(map(chr, (q - min_token).astype(int))) for q in coeffs),
                                       trainer=trainer)
    return cls(PreTrainedTokenizerFast(tokenizer_object=bpe, clean_up_tokenization_spaces=False),
               scale=scale, vocab_size=vocab_size, min_token=min_token, time_horizon=H, action_dim=D)


def fit_local(train_chunks, gamma, lam=1.0, vocab_size=1024, H=50, D=14, official_fit=False):
    """Fit a FAST-style BPE vocab on weighted training chunks.
    Raises AssertionError if the coefficient range exceeds vocab_size (record, don't hide).
    official_fit=True uses the unmodified official routine (kept to reproduce the byte-coverage bug)."""
    global _RELEASED_CLS
    _RELEASED_CLS = _RELEASED_CLS or type(load_released().proc)
    w = channel_weights(lam, D)
    data = [c * w for c in train_chunks]
    if official_fit:
        proc = _RELEASED_CLS.fit(data, scale=gamma, vocab_size=vocab_size, time_horizon=H, action_dim=D)
    else:
        proc = _fit_full_alphabet(_RELEASED_CLS, data, gamma, vocab_size, H, D)
    name = f"local_g{gamma}_l{lam:g}_v{vocab_size}"
    return FastCodec(proc, name, weights=w, H=H, D=D)
