"""
Tactical Edge ANC - Real-Time Streaming Engine (v2)
====================================================
Fixes over realtime_demo.py:

  1. LATENCY / TEARING
     The audio callback (PortAudio's real-time thread) now does almost nothing:
     it copies the input chunk into a bounded queue and copies a pre-computed
     output chunk out. All STFT / GPU inference / ISTFT work happens on a
     separate worker thread. The callback can never be blocked by the GPU.

  2. INPUT BLINDNESS
     A FIXED (session-constant) gain is applied to the mic signal before the
     STFT, calibrated once against the RMS level the model actually saw during
     training (see compute_target_rms.py). This is NOT per-frame AGC - the
     gain is computed once and held constant, so silence stays silence instead
     of being amplified into "speech-shaped" noise.

  3. PHASE TEARING
     Analysis and synthesis both use a sqrt-Hann window at 75% overlap
     (hop = N_FFT/4), with an explicit running normalization envelope. This
     combination is COLA-exact (verified numerically: the envelope is a flat
     constant, giving >110 dB reconstruction fidelity on band-limited audio;
     see test_ola.py). This replaces the old single-Hann-applied-twice
     approach, which had no explicit normalization and relied on 75% overlap
     "happening" to be close enough.

Dependencies: torch, numpy, sounddevice only (as requested).
"""

import json
import queue
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch

from train import UNetDenoiseMask

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
SR = 16000
N_FFT = 512
WINDOW_SIZE = 512
CHUNK_SIZE = 128                 # hop: 8 ms @ 16 kHz -> 75% overlap (COLA-exact w/ sqrt-Hann)
FREQ_BINS = 256                  # matches dataset_loader.py's 257 -> 256 slice
MODEL_FRAMES = 256

MODEL_PATH = "denoise_model.pth"
STATS_PATH = Path("configs/audio_stats.json")
DEFAULT_TARGET_RMS = 0.05        # fallback if compute_target_rms.py hasn't been run yet

QUEUE_MAXLEN = 8                 # bounds worst-case queued latency to ~QUEUE_MAXLEN * hop
CALIBRATION_SECONDS = 2.0

# Tactical sub-band shaping (kept from the original design, now clearly
# separated from the RMS pre-scaler - this shapes the MASK, it does not
# touch signal level, so it cannot reintroduce the AGC problem).
BANDPASS_LOW_BIN = 10             # ~300 Hz
BANDPASS_HIGH_BIN = 110           # ~3400 Hz
BANDPASS_LOW_GAIN = 0.05
BANDPASS_HIGH_GAIN = 0.10
MASK_FLOOR = 0.05

# Impulse (gunshot) guard - operates on raw amplitude, independent of the
# fixed RMS pre-scaler, applied AFTER it.
IMPULSE_RATIO_THRESHOLD = 6.0


# ----------------------------------------------------------------------
# WINDOWING / OLA
# ----------------------------------------------------------------------
def periodic_hann(n):
    k = np.arange(n)
    return (0.5 - 0.5 * np.cos(2 * np.pi * k / n)).astype(np.float32)


def sqrt_hann(n):
    return np.sqrt(periodic_hann(n)).astype(np.float32)


class StreamingOLA:
    """Causal, chunked STFT/ISTFT with an explicit COLA normalization
    envelope. See test_ola.py for the numeric proof that this reconstructs
    losslessly (mask = 1) to floating point / Nyquist-bin precision."""

    def __init__(self):
        self.in_buffer = np.zeros(WINDOW_SIZE, dtype=np.float32)
        self.out_buffer = np.zeros(WINDOW_SIZE, dtype=np.float32)
        self.norm_buffer = np.zeros(WINDOW_SIZE, dtype=np.float32)
        self.analysis_window = sqrt_hann(WINDOW_SIZE)
        self.synthesis_window = sqrt_hann(WINDOW_SIZE)
        self.window_product = self.analysis_window * self.synthesis_window
        self.mag_context = np.zeros((FREQ_BINS, MODEL_FRAMES), dtype=np.float32)

    def push_input(self, chunk):
        self.in_buffer = np.roll(self.in_buffer, -CHUNK_SIZE)
        self.in_buffer[-CHUNK_SIZE:] = chunk

    def analyze(self):
        windowed = self.in_buffer * self.analysis_window
        spectrum = np.fft.rfft(windowed, n=N_FFT)      # 257 bins
        mag = np.abs(spectrum)[:FREQ_BINS]               # drop Nyquist bin (matches training)
        phase = np.angle(spectrum)[:FREQ_BINS]
        return mag, phase

    def update_context(self, mag):
        self.mag_context = np.roll(self.mag_context, -1, axis=1)
        self.mag_context[:, -1] = mag

    def synthesize(self, mag, phase):
        full_mag = np.concatenate([mag, [0.0]]).astype(np.float32)
        full_phase = np.concatenate([phase, [0.0]]).astype(np.float32)
        spectrum = full_mag * np.exp(1j * full_phase)
        frame = np.fft.irfft(spectrum, n=N_FFT).astype(np.float32) * self.synthesis_window
        return frame

    def push_output(self, frame):
        self.out_buffer = np.roll(self.out_buffer, -CHUNK_SIZE)
        self.out_buffer[-CHUNK_SIZE:] = 0.0
        self.out_buffer += frame

        self.norm_buffer = np.roll(self.norm_buffer, -CHUNK_SIZE)
        self.norm_buffer[-CHUNK_SIZE:] = 0.0
        self.norm_buffer += self.window_product

        norm = np.maximum(self.norm_buffer[:CHUNK_SIZE], 1e-6)
        out_chunk = (self.out_buffer[:CHUNK_SIZE] / norm).astype(np.float32)
        return out_chunk


# ----------------------------------------------------------------------
# FIXED RMS PRE-SCALER (calibrated once, held constant for the session)
# ----------------------------------------------------------------------
def load_target_rms():
    if STATS_PATH.exists():
        try:
            data = json.loads(STATS_PATH.read_text())
            return float(data["target_rms"])
        except Exception:
            print(f"[warn] Could not parse {STATS_PATH}, using default target RMS.")
    else:
        print(f"[warn] {STATS_PATH} not found - run compute_target_rms.py against your "
              f"training set for a properly calibrated target. Using a rough default for now.")
    return DEFAULT_TARGET_RMS


def calibrate_input_gain(target_rms, duration=CALIBRATION_SECONDS):
    print(f"[calibration] Recording {duration:.1f}s - please speak at a normal volume...")
    rec = sd.rec(int(duration * SR), samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    measured_rms = float(np.sqrt(np.mean(rec.astype(np.float64) ** 2) + 1e-12))
    if measured_rms < 1e-5:
        print("[calibration] Input was near-silent; falling back to unity gain (1.0). "
              "Re-run calibration if this seems wrong.")
        return 1.0
    gain = float(np.clip(target_rms / measured_rms, 0.1, 50.0))
    print(f"[calibration] measured mic RMS={measured_rms:.5f}  target RMS={target_rms:.5f}  "
          f"=> fixed gain={gain:.3f} (held constant for this session)")
    return gain


# ----------------------------------------------------------------------
# THREADED PIPELINE
# ----------------------------------------------------------------------
input_queue = queue.Queue(maxsize=QUEUE_MAXLEN)
output_queue = queue.Queue(maxsize=QUEUE_MAXLEN)

stats_lock = threading.Lock()
stats = {"callbacks": 0, "dropped_input": 0, "dropped_output": 0, "underruns": 0}
proc_times_ms = []  # rolling processing-time samples, read/written by worker only

# Simple float shared between the worker thread (writer) and the callback
# thread (reader) for the anti-click underrun fade. A plain float
# assignment is atomic under the GIL, so this needs no lock; a stale read
# on the rare race is inaudible and self-corrects on the next chunk.
_last_out_level = {"v": 0.0}


def _drop_oldest_and_put(q, item):
    try:
        q.put_nowait(item)
        return False
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except queue.Full:
            pass
        return True


def audio_callback(indata, outdata, frames, time_info, status):
    # This function runs on PortAudio's real-time thread. Keep it O(1) and
    # allocation-light: no STFT, no torch, no model calls here.
    chunk = indata[:, 0].copy()
    dropped = _drop_oldest_and_put(input_queue, chunk)

    try:
        out_chunk = output_queue.get_nowait()
        outdata[:, 0] = out_chunk
    except queue.Empty:
        # Ramp from the last known output level down to zero instead of a
        # hard cut, to avoid an audible click on underrun.
        outdata[:, 0] = np.linspace(_last_out_level["v"], 0.0, CHUNK_SIZE, dtype=np.float32)
        _last_out_level["v"] = 0.0
        with stats_lock:
            stats["underruns"] += 1

    with stats_lock:
        stats["callbacks"] += 1
        if dropped:
            stats["dropped_input"] += 1


def processing_worker(stop_event, model, device, gain, ola):
    short_energy, long_energy = 1e-3, 1e-3

    while not stop_event.is_set():
        try:
            chunk = input_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        t0 = time.perf_counter()

        # --- FIXED RMS pre-scale (constant gain, computed once at startup) ---
        chunk = chunk * gain

        # --- Impulse / gunshot guard (independent of the fixed pre-scaler) ---
        pwr = float(np.dot(chunk, chunk)) / CHUNK_SIZE + 1e-7
        short_energy = 0.8 * short_energy + 0.2 * pwr
        long_energy = 0.99 * long_energy + 0.01 * pwr
        transient_score = short_energy / (long_energy + 1e-7)
        if transient_score > IMPULSE_RATIO_THRESHOLD:
            scale = np.sqrt(IMPULSE_RATIO_THRESHOLD / transient_score)
            chunk = np.tanh(chunk * scale) * 0.8

        # --- Analysis ---
        ola.push_input(chunk)
        mag, phase = ola.analyze()
        ola.update_context(mag)

        # --- GPU inference ---
        with torch.inference_mode():
            x = torch.from_numpy(np.log1p(ola.mag_context)).unsqueeze(0).unsqueeze(0).to(device)
            mask = model(x).squeeze(0).squeeze(0)[:, -1].cpu().numpy()

        # --- Tactical sub-band shaping + floor ---
        mask = mask.copy()
        mask[:BANDPASS_LOW_BIN] *= BANDPASS_LOW_GAIN
        mask[BANDPASS_HIGH_BIN:] *= BANDPASS_HIGH_GAIN
        mask = np.clip(mask, MASK_FLOOR, 1.0)

        cleaned_mag = mag * mask

        # --- Synthesis (COLA-exact OLA) ---
        frame = ola.synthesize(cleaned_mag, phase)
        out_chunk = ola.push_output(frame)
        out_chunk = np.tanh(out_chunk * 0.9).astype(np.float32)  # soft limiter
        _last_out_level["v"] = float(out_chunk[-1])

        dropped = _drop_oldest_and_put(output_queue, out_chunk)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        proc_times_ms.append(elapsed_ms)
        if len(proc_times_ms) > 500:
            del proc_times_ms[:250]

        if dropped:
            with stats_lock:
                stats["dropped_output"] += 1


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device = {device}")

    model = UNetDenoiseMask().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    dummy = torch.zeros(1, 1, FREQ_BINS, MODEL_FRAMES, device=device)
    with torch.inference_mode():
        for _ in range(3):
            model(dummy)  # warm up CUDA kernels / cuDNN autotune before the real stream starts

    target_rms = load_target_rms()
    gain = calibrate_input_gain(target_rms)

    ola = StreamingOLA()
    stop_event = threading.Event()
    worker = threading.Thread(
        target=processing_worker, args=(stop_event, model, device, gain, ola), daemon=True
    )
    worker.start()

    hop_ms = CHUNK_SIZE / SR * 1000
    window_ms = WINDOW_SIZE / SR * 1000
    print("\n=======================================================")
    print("TACTICAL CORE RUNNING (threaded, COLA-exact OLA, fixed RMS pre-scale)")
    print(f"Hop size: {hop_ms:.2f} ms | Window (algorithmic) latency: {window_ms:.2f} ms")
    print("Press Ctrl+C to stop.")
    print("=======================================================\n")

    try:
        with sd.Stream(samplerate=SR, blocksize=CHUNK_SIZE, channels=1,
                        dtype="float32", callback=audio_callback):
            while not stop_event.is_set():
                time.sleep(1.0)
                with stats_lock:
                    s = dict(stats)
                if proc_times_ms:
                    arr = np.array(proc_times_ms[-100:])
                    p50, p95, pmax = np.percentile(arr, 50), np.percentile(arr, 95), arr.max()
                    rtf = p50 / hop_ms
                    print(f"[stats] cb={s['callbacks']} drop_in={s['dropped_input']} "
                          f"drop_out={s['dropped_output']} underrun={s['underruns']} | "
                          f"proc p50={p50:.2f}ms p95={p95:.2f}ms max={pmax:.2f}ms RTF={rtf:.2f}")
                    if rtf > 1.0:
                        print("[warn] Real-time factor > 1.0 - GPU/CPU cannot keep up with the "
                              "hop rate; expect growing drops. Consider a larger CHUNK_SIZE.")
    except KeyboardInterrupt:
        print("\n[!] Stream stopped by user.")
    finally:
        stop_event.set()
        worker.join(timeout=2.0)


if __name__ == "__main__":
    main()
