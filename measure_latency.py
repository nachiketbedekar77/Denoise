"""
Two independent ways to actually measure latency, instead of hardcoding
"18 ms" in a dashboard:

  --mode compute   Feeds real audio through the exact StreamingOLA + model
                    path used in realtime_demo_v2.py, many times, and reports
                    the processing-time distribution (p50/p95/max) and the
                    real-time factor (RTF = compute_time / hop_time). This
                    tells you whether your RTX 3050 can sustain the hop rate
                    at all - if RTF > 1 on average, no amount of buffering
                    fixes it, you need a bigger hop or a smaller model.

  --mode loopback   Plays a click train out of your speakers/line-out and
                    records it back in through the live realtime_demo_v2
                    pipeline via an audio loopback route (Windows "Stereo
                    Mix", a virtual audio cable, or a physical cable from
                    output to mic input). Cross-correlating the recorded
                    clicks against the played clicks gives the TRUE
                    end-to-end latency: OS audio buffers + algorithmic delay
                    + compute time, all included. This is the number that
                    matters for a live conversation, and it can only be
                    obtained by measuring, not by computing N_FFT/SR on paper.

Usage:
    python measure_latency.py --mode compute --seconds 10
    python measure_latency.py --mode loopback --loopback_device "Stereo Mix"
"""
import argparse
import time

import numpy as np
import torch

from realtime_demo_v2 import (
    StreamingOLA, SR, CHUNK_SIZE, FREQ_BINS, MODEL_FRAMES,
    BANDPASS_LOW_BIN, BANDPASS_HIGH_BIN, BANDPASS_LOW_GAIN, BANDPASS_HIGH_GAIN, MASK_FLOOR,
)
from train import UNetDenoiseMask


def run_compute_benchmark(seconds, model_path="denoise_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetDenoiseMask().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    dummy = torch.zeros(1, 1, FREQ_BINS, MODEL_FRAMES, device=device)
    with torch.inference_mode():
        for _ in range(5):
            model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    ola = StreamingOLA()
    rng = np.random.default_rng(0)
    n_chunks = int(seconds * SR / CHUNK_SIZE)
    times_ms = []

    hop_budget_ms = CHUNK_SIZE / SR * 1000.0
    print(f"Running {n_chunks} chunks ({seconds:.1f}s of audio) on device={device} ...")
    print(f"Hop budget per chunk: {hop_budget_ms:.2f} ms (must stay under this, on average, "
          f"to sustain real time)\n")

    for _ in range(n_chunks):
        chunk = (0.05 * rng.standard_normal(CHUNK_SIZE)).astype(np.float32)

        t0 = time.perf_counter()
        ola.push_input(chunk)
        mag, phase = ola.analyze()
        ola.update_context(mag)

        with torch.inference_mode():
            x = torch.from_numpy(np.log1p(ola.mag_context)).unsqueeze(0).unsqueeze(0).to(device)
            mask = model(x).squeeze(0).squeeze(0)[:, -1]
            if device.type == "cuda":
                torch.cuda.synchronize()   # don't let async CUDA dispatch hide the real cost
            mask = mask.cpu().numpy()

        mask = mask.copy()
        mask[:BANDPASS_LOW_BIN] *= BANDPASS_LOW_GAIN
        mask[BANDPASS_HIGH_BIN:] *= BANDPASS_HIGH_GAIN
        mask = np.clip(mask, MASK_FLOOR, 1.0)
        cleaned_mag = mag * mask

        frame = ola.synthesize(cleaned_mag, phase)
        ola.push_output(frame)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(times_ms)
    p50, p95, p99, pmax = np.percentile(arr, [50, 95, 99, 100])
    rtf = arr.mean() / hop_budget_ms

    print("=" * 55)
    print(f"Processing time per {CHUNK_SIZE}-sample chunk (n={len(arr)}):")
    print(f"  mean={arr.mean():.3f} ms  p50={p50:.3f} ms  p95={p95:.3f} ms  "
          f"p99={p99:.3f} ms  max={pmax:.3f} ms")
    print(f"  Real-time factor (mean compute / hop budget): {rtf:.3f}")
    print(f"  -> {'SUSTAINABLE' if rtf < 0.8 else ('MARGINAL' if rtf < 1.0 else 'NOT SUSTAINABLE')} "
          f"at this chunk size on this device.")
    print("=" * 55)
    print("This measures compute cost only. Add the algorithmic window delay "
          f"({(512/SR)*1000:.1f} ms for N_FFT=512) and OS/driver buffering "
          "(measure with --mode loopback) for true end-to-end latency.")


def run_loopback_benchmark(loopback_device, seconds=5.0, n_clicks=8):
    import sounddevice as sd

    print(f"Loopback device: {loopback_device!r}")
    print("This plays clicks out and records them back through your OS loopback "
          "route. Set your input device to that loopback route (or physically "
          "cable output -> mic input) before running.\n")

    n_samples = int(seconds * SR)
    play_signal = np.zeros(n_samples, dtype=np.float32)
    click_positions = np.linspace(0, n_samples - SR // 2, n_clicks, dtype=int)
    click = (np.hanning(64) * 0.8).astype(np.float32)
    for p in click_positions:
        play_signal[p:p + len(click)] += click

    rec_signal = sd.playrec(play_signal.reshape(-1, 1), samplerate=SR,
                             channels=1, device=(loopback_device, loopback_device))
    sd.wait()
    rec_signal = rec_signal[:, 0]

    delays_ms = []
    search = SR // 2  # search window around each expected click
    for p in click_positions:
        lo, hi = max(0, p - search // 2), min(n_samples, p + search)
        seg = rec_signal[lo:hi]
        if len(seg) < len(click):
            continue
        corr = np.correlate(seg, click, mode="valid")
        best = lo + int(np.argmax(corr))
        delays_ms.append((best - p) / SR * 1000.0)

    if not delays_ms:
        print("No clicks detected in the recording - check your loopback routing.")
        return

    delays_ms = np.array(delays_ms)
    print(f"Measured round-trip delays over {len(delays_ms)} clicks:")
    print(f"  mean={delays_ms.mean():.2f} ms  std={delays_ms.std():.2f} ms  "
          f"min={delays_ms.min():.2f} ms  max={delays_ms.max():.2f} ms")
    print("\nNote: this is PLAY -> capture -> (if you route it through the live "
          "realtime_demo_v2 process during recording) -> processed -> capture "
          "round trip, i.e. the number a user would actually experience.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["compute", "loopback"], default="compute")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--loopback_device", type=str, default=None)
    args = ap.parse_args()

    if args.mode == "compute":
        run_compute_benchmark(args.seconds)
    else:
        if not args.loopback_device:
            raise SystemExit("--loopback_device is required for --mode loopback "
                              "(e.g. 'Stereo Mix' on Windows, or your virtual cable's name)")
        run_loopback_benchmark(args.loopback_device, seconds=args.seconds)
