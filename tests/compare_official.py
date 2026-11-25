"""Compare the native CLI against Meta's HTDemucs and write level-matched A/B clips.

Example: python tests/compare_official.py mix.wav 955717e8-8726e21a.th results --native native-stems
Use a stereo music excerpt longer than 7.8 seconds to exercise a window boundary.
Reference dependencies: demucs==4.0.1, torch, soundfile, numpy, scipy.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from demucs.states import load_model
from scipy.signal import correlate, correlation_lags


CHECKPOINT = "8726e21a993978c7ba086d3872e7608d7d5bfca646ca4aca459ffda844faa8b4"
RATE = 44100
WINDOW = 343980
SOURCES = ("drums", "bass", "other", "vocals")


def load_official(path):
    # This historical upstream checkpoint contains Python objects. Only deserialize
    # the exact official artifact, including when run with modern PyTorch.
    with path.open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != CHECKPOINT:
            raise ValueError("Unexpected official checkpoint digest")
    model = load_model(torch.load(path, map_location="cpu", weights_only=False)).eval()
    if tuple(model.sources) != SOURCES or model.samplerate != RATE:
        raise ValueError("Unexpected model sources or sample rate")
    model.crosstransformer.sin_random_shift = 0
    return model


@torch.inference_mode()
def reference(mix, model):
    overlap = WINDOW // 4
    step = WINDOW - overlap
    weights = np.ones(WINDOW, dtype=np.float32)
    fade = np.linspace(0, 1, overlap, dtype=np.float32)
    weights[:overlap], weights[-overlap:] = fade, fade[::-1]
    frames = len(mix)
    result = np.zeros((len(SOURCES), frames, 2), dtype=np.float32)
    total = np.zeros(frames, dtype=np.float32)
    for start in range(0, frames, step):
        count = min(WINDOW, frames - start)
        chunk = torch.zeros((1, 2, WINDOW))
        chunk[0, :, :count] = torch.from_numpy(mix[start:start + count].T.copy())
        prediction = model(chunk)[0, :, :, :count].numpy().transpose(0, 2, 1)
        window = weights[:count].copy()
        if start == 0:
            # The first window has no predecessor to fade in from.
            window[:overlap] = 1
        result[:, start:start + count] += prediction * window[:, None]
        total[start:start + count] += window
    return result / np.maximum(total[:, None], 1e-8)


def compare(native, reference_audio):
    delta = native.astype(np.float64) - reference_audio
    rms = np.sqrt(np.mean(delta ** 2, axis=0))
    energy = np.sqrt(np.mean(reference_audio.astype(np.float64) ** 2, axis=0))
    snr = 20 * np.log10(np.maximum(energy, 1e-12) / np.maximum(rms, 1e-12))
    delays = []
    for channel in range(2):
        correlation = correlate(native[:, channel], reference_audio[:, channel], method="fft")
        delays.append(int(correlation_lags(len(native), len(reference_audio))[correlation.argmax()]))
    return {"snr_db_lr": snr.tolist(), "rms_error_lr": rms.tolist(),
            "lag_samples_lr": delays, "peak_error": float(abs(delta).max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mix", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--native", type=Path)
    args = parser.parse_args()
    mix, rate = sf.read(args.mix, dtype="float32", always_2d=True)
    if rate != RATE or mix.shape[1] != 2 or not len(mix):
        parser.error("Provide a nonempty 44.1 kHz stereo WAV")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    expected = reference(mix, load_official(args.checkpoint))
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = {}
    passed = True
    for name, audio in zip(SOURCES, expected):
        sf.write(args.output / f"{name}.wav", audio, rate, subtype="FLOAT")
        if args.native is None:
            continue
        candidate, stem_rate = sf.read(args.native / f"{name}.wav", dtype="float32", always_2d=True)
        if candidate.shape != audio.shape or stem_rate != rate:
            raise ValueError(f"{name}: expected stereo, {len(audio)} frames at {rate} Hz")
        # The CLI writes PCM16, so match its available dynamic range.
        bounded = audio.clip(-1.0, 32767 / 32768)
        metrics[name] = m = compare(candidate, bounded)
        # For near-silent stems even rounding a perfect prediction to PCM16
        # can dominate relative SNR. Allow one PCM16 LSB RMS in that case.
        close = all(snr >= 30.0 or rms <= 1 / 32768
                    for snr, rms in zip(m["snr_db_lr"], m["rms_error_lr"]))
        passed &= close and m["lag_samples_lr"] == [0, 0]
        pause = np.zeros((rate // 3, 2), dtype=np.float32)
        pair = np.concatenate([bounded, pause, candidate])
        sf.write(args.output / f"{name}-AB.wav", pair, rate, subtype="PCM_16")
    if metrics:
        report = {"frames": len(mix), "sample_rate": rate, "stems": metrics, "passed": passed}
        (args.output / "comparison.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        if not passed:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
