import time
import threading
import sounddevice as sd
import numpy as np
import torch
import warnings
from train import UNetDenoiseMask

warnings.filterwarnings('ignore')

# --- 1. STABLE LOW-LATENCY PARAMETERS ---
SR = 16000
CHUNK_SIZE = 128       # 8.0 ms hop (Perfect for 75% overlap)
WINDOW_SIZE = 512      # Matches N_FFT to prevent phase tearing/pulsing
N_FFT = 512            # Required for 257 bins
MODEL_FRAMES = 256     # Tensor width dimension expected by U-Net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNetDenoiseMask().to(device)
model.load_state_dict(torch.load("denoise_model.pth", map_location=device, weights_only=True))
model.eval()

# GPU Engine Warmup
dummy_input = torch.zeros(1, 1, 256, MODEL_FRAMES, device=device)
with torch.no_grad():
    for _ in range(3):
        _ = model(dummy_input)

# --- 2. FAST BUFFERS & FILTERS ---
in_buffer = np.zeros(WINDOW_SIZE, dtype=np.float32)
out_buffer = np.zeros(WINDOW_SIZE, dtype=np.float32)
mag_context = np.zeros((256, MODEL_FRAMES), dtype=np.float32)
window = np.hanning(WINDOW_SIZE).astype(np.float32)

# Smoothed Trackers
short_energy = 0.001
long_energy = 0.001
prev_mask = np.zeros(256, dtype=np.float32)

# Sub-band Filter (300Hz - 3400Hz)
tactical_band = np.ones(256, dtype=np.float32)
tactical_band[:10] = 0.02
tactical_band[110:] = 0.05

stop_event = threading.Event()

# --- 3. HARDWARE STREAM CALLBACK ---
def audio_callback(indata, outdata, frames, time_info, status):
    global in_buffer, out_buffer, mag_context, short_energy, long_energy, prev_mask
    
    chunk = indata[:, 0].copy()

    # --- STAGE 1: IMPULSE SUPPRESSOR (Relaxed for Speech) ---
    pwr = np.dot(chunk, chunk) / CHUNK_SIZE + 1e-7
    short_energy = 0.8 * short_energy + 0.2 * pwr
    long_energy = 0.99 * long_energy + 0.01 * pwr
    transient_score = short_energy / (long_energy + 1e-7)

    # Threshold set to 6.0 so loud human voice survives easily
    if transient_score > 6.0:
        scale = np.sqrt(6.0 / transient_score)
        chunk = np.tanh(chunk * scale) * 0.8

    # --- STAGE 2: OLA INPUT ---
    in_buffer = np.roll(in_buffer, -CHUNK_SIZE)
    in_buffer[-CHUNK_SIZE:] = chunk
    
    windowed = in_buffer * window
    stft_res = np.fft.rfft(windowed, n=N_FFT)
    mag = np.abs(stft_res)[:256]
    phase = np.angle(stft_res)[:256]

    # --- STAGE 3: INFERENCE ---
    mag_context = np.roll(mag_context, -1, axis=1)
    mag_context[:, -1] = mag

    with torch.no_grad():
        x = torch.from_numpy(np.log1p(mag_context)).float().unsqueeze(0).unsqueeze(0).to(device)
        mask = model(x).squeeze(0).squeeze(0)[:, -1].cpu().numpy()

    # --- STAGE 4: REFINED DUAL GATING ---
    mask = mask * tactical_band

    # Smoother temporal transition to kill the pulse effect
    alpha = 0.5 
    mask = (alpha * mask) + ((1.0 - alpha) * prev_mask)
    prev_mask = mask.copy()

    # Impulse Clamp for gunshots
    if transient_score > 6.0:
        mask *= 0.4

    # Lowered squelch threshold (0.08) so quiet human words don't get choked
    speech_energy = np.percentile(mask, 90)
    if speech_energy < 0.08:
        cleaned_mag = mag * 0.02 
    else:
        smooth_mask = np.clip(mask, 0.05, 1.0)
        # Boost gain to 2.0 for louder, clear output
        cleaned_mag = mag * ((smooth_mask ** 1.5) * 2.0)

    # --- STAGE 5: PERFECT OVERLAP-ADD SYNTHESIS ---
    stft_synth = np.zeros(257, dtype=np.complex64)
    stft_synth[:256] = cleaned_mag * np.exp(1j * phase)
    
    # Do NOT slice this! Keep all 512 points for proper overlap
    reconstructed = np.fft.irfft(stft_synth, n=N_FFT) * window

    out_buffer = np.roll(out_buffer, -CHUNK_SIZE)
    out_buffer[-CHUNK_SIZE:] = 0.0
    out_buffer += reconstructed

    # Analog Soft-Clipper via hyperbolic tangent
    outdata[:, 0] = np.tanh(out_buffer[:CHUNK_SIZE] * 0.85)

# --- 4. START RUNTIME ---
print("\n=======================================================")
print("⚡ TACTICAL CORE RUNNING (STABLE OLA PHASE)")
print(f"⏱️ Algorithmic Latency: {(CHUNK_SIZE/SR)*1000:.2f} ms")
print("🔥 RTX 3050 Tensor Acceleration: Active")
print("🔴 Press Ctrl+C to terminate safely.")
print("=======================================================\n")

try:
    with sd.Stream(samplerate=SR, blocksize=CHUNK_SIZE, channels=1, callback=audio_callback):
        # Non-blocking loop so Ctrl+C works instantly
        while not stop_event.is_set():
            time.sleep(0.1)
except KeyboardInterrupt:
    print("\n[!] Edge stream safely terminated by user.")
finally:
    stop_event.set()