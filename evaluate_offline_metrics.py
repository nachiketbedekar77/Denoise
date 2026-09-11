"""
Real, measured SNR / STOI / PESQ evaluation - replaces the hardcoded numbers
currently shown in app.py's "Tactical Performance Metrics" panel (those are
UI placeholders, not computed from your model's actual output).

IMPORTANT - held-out test set:
    Point --clean_dir / --noisy_dir at audio that was NOT used to train the
    model. Evaluating on training files will give inflated numbers that
    don't reflect real performance. generate_dataset.py currently mixes and
    uses every clean file for training with no split - reserve ~10-15% of
    Raw-data/clean before running it, mix those separately (same process_file
    logic / SNR range), and use that reserved set here.

Usage:
    python evaluate_offline_metrics.py --clean_dir test_data/clean --noisy_dir test_data/noisy

Requires: pip install pystoi pesq
"""
import argparse
from pathlib import Path

import librosa
import numpy as np
from pystoi import stoi as stoi_fn
from pesq import pesq as pesq_fn

from inference import denoise_audio

SR = 16000

# Thresholds mirrored from app.py's dashboard, now checked against real numbers.
THRESHOLDS = {
    "delta_snr_db": 15.0,
    "stoi": 0.85,
    "pesq": 2.5,
}


def snr_db(clean, degraded):
    n = min(len(clean), len(degraded))
    clean, degraded = clean[:n], degraded[:n]
    noise = clean - degraded
    num = np.sum(clean.astype(np.float64) ** 2)
    den = np.sum(noise.astype(np.float64) ** 2) + 1e-12
    return 10.0 * np.log10(num / den)


def safe_pesq(clean, degraded, sr):
    n = min(len(clean), len(degraded))
    try:
        return pesq_fn(sr, clean[:n], degraded[:n], "wb" if sr == 16000 else "nb")
    except Exception as e:
        return float("nan")


def safe_stoi(clean, degraded, sr):
    n = min(len(clean), len(degraded))
    try:
        return stoi_fn(clean[:n], degraded[:n], sr, extended=False)
    except Exception:
        return float("nan")


def evaluate_pair(clean_path, noisy_path, tmp_out_path):
    clean, _ = librosa.load(clean_path, sr=SR)
    noisy, _ = librosa.load(noisy_path, sr=SR)

    denoise_audio(str(noisy_path), str(tmp_out_path))
    processed, _ = librosa.load(tmp_out_path, sr=SR)

    return {
        "file": clean_path.name,
        "snr_in": snr_db(clean, noisy),
        "snr_out": snr_db(clean, processed),
        "stoi_in": safe_stoi(clean, noisy, SR),
        "stoi_out": safe_stoi(clean, processed, SR),
        "pesq_in": safe_pesq(clean, noisy, SR),
        "pesq_out": safe_pesq(clean, processed, SR),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_dir", default="test_data/clean")
    ap.add_argument("--noisy_dir", default="test_data/noisy")
    ap.add_argument("--out_csv", default="eval_results.csv")
    ap.add_argument("--limit", type=int, default=None, help="evaluate at most N pairs (debugging)")
    args = ap.parse_args()

    clean_dir, noisy_dir = Path(args.clean_dir), Path(args.noisy_dir)
    clean_files = {f.stem: f for f in clean_dir.glob("*.wav")}
    noisy_files = {f.stem: f for f in noisy_dir.glob("*.wav")}

    # match by shared stem prefix (handles "<id>" vs "<id>_noisy" naming)
    pairs = []
    for stem, cpath in clean_files.items():
        candidates = [npath for nstem, npath in noisy_files.items() if nstem.startswith(stem)]
        if candidates:
            pairs.append((cpath, candidates[0]))

    if not pairs:
        raise SystemExit(
            f"No matching clean/noisy pairs found between {clean_dir} and {noisy_dir}. "
            f"Filenames must share a common stem, e.g. speaker1_001.wav / speaker1_001_noisy.wav"
        )
    if args.limit:
        pairs = pairs[: args.limit]

    print(f"Evaluating {len(pairs)} held-out pairs...\n")

    tmp_out = Path("eval_tmp_output.wav")
    rows = []
    for cpath, npath in pairs:
        try:
            row = evaluate_pair(cpath, npath, tmp_out)
            rows.append(row)
            print(f"  {row['file']:30s} SNR {row['snr_in']:6.2f} -> {row['snr_out']:6.2f} dB | "
                  f"STOI {row['stoi_in']:.3f} -> {row['stoi_out']:.3f} | "
                  f"PESQ {row['pesq_in']:.2f} -> {row['pesq_out']:.2f}")
        except Exception as e:
            print(f"  [skip] {cpath.name}: {e}")

    if tmp_out.exists():
        tmp_out.unlink()

    if not rows:
        raise SystemExit("No pairs evaluated successfully.")

    import csv
    with open(args.out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    def agg(key):
        vals = np.array([r[key] for r in rows if np.isfinite(r[key])])
        return vals.mean(), vals.std()

    snr_in_m, _ = agg("snr_in")
    snr_out_m, snr_out_s = agg("snr_out")
    delta_snr = snr_out_m - snr_in_m
    stoi_out_m, stoi_out_s = agg("stoi_out")
    pesq_out_m, pesq_out_s = agg("pesq_out")

    print("\n" + "=" * 60)
    print(f"RESULTS  (n={len(rows)} held-out files, saved to {args.out_csv})")
    print("=" * 60)
    print(f"Delta SNR : {delta_snr:6.2f} dB   "
          f"{'PASS' if delta_snr >= THRESHOLDS['delta_snr_db'] else 'FAIL'} "
          f"(target >= {THRESHOLDS['delta_snr_db']} dB)")
    print(f"STOI      : {stoi_out_m:.3f} +/- {stoi_out_s:.3f}   "
          f"{'PASS' if stoi_out_m >= THRESHOLDS['stoi'] else 'FAIL'} "
          f"(target >= {THRESHOLDS['stoi']})")
    print(f"PESQ      : {pesq_out_m:.2f} +/- {pesq_out_s:.2f}   "
          f"{'PASS' if pesq_out_m >= THRESHOLDS['pesq'] else 'FAIL'} "
          f"(target >= {THRESHOLDS['pesq']})")
    print("=" * 60)
    print("Note: latency is NOT measured here (this is an offline, non-streaming "
          "evaluation of inference.py). Run measure_latency.py for that.")


if __name__ == "__main__":
    main()
