"""Inspect original WAV samples without altering the recording."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import welch


def inspect_wav(path):
    """Return signal checks; musical quality still requires listening."""
    path = Path(path)
    x, sr = sf.read(path, dtype="float64", always_2d=True)
    finite = bool(np.isfinite(x).all())
    if not finite:
        return {"path": str(path), "finite": False, "status": "FAIL"}
    if len(x) == 0:
        return {"path": str(path), "finite": True, "status": "FAIL", "reason": "empty WAV"}
    peak = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(x * x)))
    mono = x.mean(axis=1)
    freq, power = welch(mono, sr, nperseg=min(8192, len(mono)))
    power = power + 1e-30
    flatness = float(np.exp(np.mean(np.log(power))) / np.mean(power))
    blocks = [np.sqrt(np.mean(b * b)) for b in np.array_split(x, max(1, int(len(x) / sr * 10)))]
    result = {
        "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "sample_rate": sr, "channels": x.shape[1], "frames": len(x),
        "duration_seconds": len(x) / sr, "subtype": sf.info(path).subtype,
        "finite": finite, "peak": peak, "rms": rms,
        "dc_offset": x.mean(axis=0).tolist(),
        "samples_at_or_above_full_scale_fraction": float(np.mean(np.abs(x) >= 1)),
        "near_silent_blocks_fraction": float(np.mean(np.array(blocks) < 1e-5)),
        "spectral_flatness": flatness,
        "high_frequency_power_fraction": float(power[freq > 15000].sum() / power.sum()),
        "status": "FAIL" if rms < 1e-5 or peak == 0 else "SIGNAL_CHECKS_ONLY",
        "listening_review": "Pending; signal statistics cannot establish musical fidelity or exclude all static-like artifacts."
    }
    return result

