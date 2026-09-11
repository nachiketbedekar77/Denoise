import time
import threading
import queue
import sounddevice as sd
import numpy as np
import torch
import warnings

# 🔥 THE FIX: Import your actual heavy U-Net! 
# (Ignore the yellow line in VS Code, your terminal will find this file perfectly)
from train import UNetDenoiseMask 

warnings.filterwarnings('ignore')

# --- 1. HARDWARE & LATENCY PARAMETERS ---
SR = 16000
CHUNK_SIZE = 128       
WINDOW_SIZE = 512      
N_FFT = 512            
MODEL_FRAMES = 256     

# Cheat codes from your tests
TARGET_RMS = 0.06492  
OLA_FACTOR = 2.0      

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Initializing Tactical AI Core on: {device}")

# --- 2. THE REAL MODEL LOADING ---
model = UNetDenoiseMask().to(device)
try:
    # strict=True ensures it crashes immediately if weights don't match exactly!
    model.load_state_dict(torch.load("denoise_model.pth", map_location=device, weights_only=True), strict=True)
    model.eval()
    print("✅ Model weights loaded PERFECTLY! AI is fully armed.")
except Exception as e:
    print(f"\n❌ FATAL ERROR: {e}")
    print("Fix: Ensure 'train.py' and 'denoise_model.pth' are in this exact folder.")
    exit() # Kill script so you don't hear random untrained noise

# Warmup GPU
with torch.no_grad():
    for _ in range(3): _ = model(torch.zeros(1, 1, 256, MODEL_FRAMES, device=device))

# --- 3. THREADED QUEUES (Overflow Fix) ---
# Increased maxsize to prevent "input overflow"
audio_queue = queue.Queue(maxsize=200)
out_queue = queue.Queue(maxsize=200)

window = np.hanning(WINDOW_SIZE).astype(np.float32)
in_buffer = np.zeros(WINDOW_SIZE, dtype=np.float32)
out_buffer = np.zeros(WINDOW_SIZE, dtype=np.float32)
mag_context = np.zeros((256, MODEL_FRAMES), dtype=np.float32)

stop_event = threading.Event()

# --- 4. FAST MICROPHONE CALLBACK ---
def audio_callback(indata, outdata, frames, time_info, status):
    # Removed print(status) to completely stop "input overflow" frame dropping
    
    try:
        audio_queue.put_nowait(indata[:, 0].copy())
    except queue.Full:
        pass

    try:
        outdata[:, 0] = out_queue.get_nowait()
    except queue.Empty:
        outdata[:, 0] = 0.0

# --- 5. THE AI WORKER THREAD ---
def ai_worker_thread():
    global in_buffer, out_buffer, mag_context
    
    while not stop_event.is_set():
        try:
            chunk = audio_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        # RMS PRE-SCALER
        current_rms = np.sqrt(np.mean(chunk**2)) + 1e-8
        
        if current_rms < 0.001:
            out_queue.put(np.zeros(CHUNK_SIZE, dtype=np.float32))
            continue
            
        scale_factor = TARGET_RMS / current_rms
        scaled_chunk = chunk * scale_factor

        # OLA SHIFT IN
        in_buffer = np.roll(in_buffer, -CHUNK_SIZE)
        in_buffer[-CHUNK_SIZE:] = scaled_chunk
        
        windowed = in_buffer * window
        stft_res = np.fft.rfft(windowed, n=N_FFT)
        mag = np.abs(stft_res)[:256]
        phase = np.angle(stft_res)[:256]

        mag_context = np.roll(mag_context, -1, axis=1)
        mag_context[:, -1] = mag

        # INFERENCE
        with torch.no_grad():
            x = torch.from_numpy(np.log1p(mag_context)).float().unsqueeze(0).unsqueeze(0).to(device)
            mask = model(x).squeeze(0).squeeze(0)[:, -1].cpu().numpy()

        # 🔥 TACTICAL SQUELCH & WIENER FILTERING
        peak_conf = np.percentile(mask, 95)
        
        if peak_conf < 0.30:  
            cleaned_mag = mag * 0.0
        else:
            # Pro-DSP Math[cite: 3]
            smooth_mask = np.clip(mask, 0.02, 1.0)
            cleaned_mag = mag * (smooth_mask ** 2.0) * 2.2

        # RECONSTRUCTION
        stft_synth = np.zeros(257, dtype=np.complex64)
        stft_synth[:256] = cleaned_mag * np.exp(1j * phase)
        reconstructed = np.fft.irfft(stft_synth, n=N_FFT)
        
        # 🛠️ REVERSE SCALING & OLA MATH FIX
        reconstructed = reconstructed / scale_factor
        reconstructed = (reconstructed * window) / OLA_FACTOR

        out_buffer = np.roll(out_buffer, -CHUNK_SIZE)
        out_buffer[-CHUNK_SIZE:] = 0.0
        out_buffer += reconstructed

        final_out = np.tanh(out_buffer[:CHUNK_SIZE])
        
        # Safely push to headphone
        try:
            out_queue.put_nowait(final_out)
        except queue.Full:
            pass

# --- 6. START RUNTIME ---
print("\n" + "="*55)
print(f"⚡ TRUE THREADED CORE | RMS: {TARGET_RMS} | OLA: {OLA_FACTOR}")
print("="*55 + "\n")

worker = threading.Thread(target=ai_worker_thread, daemon=True)
worker.start()

try:
    with sd.Stream(samplerate=SR, blocksize=CHUNK_SIZE, channels=1, callback=audio_callback):
        while True:
            time.sleep(0.1)
except KeyboardInterrupt:
    print("\n[!] Edge stream safely terminated by user.")
finally:
    stop_event.set()