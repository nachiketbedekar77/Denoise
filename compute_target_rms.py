"""
Computes the RMS level the model actually saw during training and saves it
to configs/audio_stats.json. realtime_demo_v2.py loads this value and uses
it as the single, fixed calibration target for the mic pre-scaler.

Why this matters: the model's masks are only meaningful for input levels
similar to what it was trained on. If your mic's raw float32 level is much
quieter than the training mixes, log1p(mag) is small everywhere and the
model has no basis to distinguish speech from noise -> it tends to leave
the mask wide open (the "Input Blindness" bug).

Run this once per dataset (or whenever you regenerate the dataset):
    python compute_target_rms.py
"""
import json
from pathlib import Path

import librosa
import numpy as np

NOISY_AUDIO_DIR = Path("dataset/noisy_audio")   # produced by generate_dataset.py
OUT_PATH = Path("configs/audio_stats.json")
SR = 16000


def main():
    files = sorted(NOISY_AUDIO_DIR.glob("*.wav"))
    if not files:
        raise SystemExit(
            f"No audio found in {NOISY_AUDIO_DIR}. Run generate_dataset.py first, "
            f"or point NOISY_AUDIO_DIR at wherever your training-time noisy mixes live."
        )

    rms_values = []
    for f in files:
        audio, _ = librosa.load(f, sr=SR)
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2) + 1e-12))
        if rms > 1e-6:  # skip any accidental silent files
            rms_values.append(rms)

    rms_values = np.array(rms_values)
    target_rms = float(np.median(rms_values))  # median: robust to a few loud/quiet outliers

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "target_rms": target_rms,
        "mean_rms": float(rms_values.mean()),
        "std_rms": float(rms_values.std()),
        "n_files": int(len(rms_values)),
        "source_dir": str(NOISY_AUDIO_DIR),
    }, indent=2))

    print(f"Scanned {len(rms_values)} files.")
    print(f"RMS distribution: median={target_rms:.5f} mean={rms_values.mean():.5f} "
          f"std={rms_values.std():.5f}")
    print(f"Saved target_rms={target_rms:.5f} -> {OUT_PATH}")


if __name__ == "__main__":
    main()
