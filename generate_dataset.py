import os
import random
import librosa
import numpy as np
import soundfile as sf
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# --- Configuration ---
CLEAN_DIR = Path("Raw-data/clean")
NOISE_DIR = Path("Raw-data/noise") # Agar noise folder ka naam bhi kuch aur hai toh yahan change kar lena
OUT_AUDIO_DIR = Path("dataset/noisy_audio")
OUT_SPEC_NOISY = Path("dataset/spectrograms_noisy")
OUT_SPEC_CLEAN = Path("dataset/spectrograms_clean")

# Ensure output directories exist
for d in [OUT_AUDIO_DIR, OUT_SPEC_NOISY, OUT_SPEC_CLEAN]:
    d.mkdir(parents=True, exist_ok=True)

# Audio parameters
SR = 16000
N_FFT = 512
HOP_LENGTH = 256

def compute_spectrogram(audio):
    """Computes the Short-Time Fourier Transform (STFT) magnitude."""
    stft = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
    magnitude = np.abs(stft)
    return magnitude

def process_file(clean_path, noise_paths):
    try:
        # 1. Load Clean Speech
        clean_audio, _ = librosa.load(clean_path, sr=SR)
        
        # 2. Load and trim/loop Noise
        noise_path = random.choice(noise_paths)
        noise_audio, _ = librosa.load(noise_path, sr=SR)
        
        if len(noise_audio) < len(clean_audio):
            repeats = int(np.ceil(len(clean_audio) / len(noise_audio)))
            noise_audio = np.tile(noise_audio, repeats)
            
        start_idx = random.randint(0, len(noise_audio) - len(clean_audio))
        noise_audio = noise_audio[start_idx : start_idx + len(clean_audio)]
        
        # 3. Mix Audio (Random SNR between 0dB and 15dB)
        target_snr = random.uniform(0, 15)
        rms_clean = np.sqrt(np.mean(clean_audio**2))
        rms_noise = np.sqrt(np.mean(noise_audio**2))
        
        if rms_noise == 0:
            gain = 0
        else:
            target_noise_rms = rms_clean / (10 ** (target_snr / 20))
            gain = target_noise_rms / rms_noise
            
        noise_scaled = noise_audio * gain
        noisy_audio = clean_audio + noise_scaled
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(noisy_audio))
        if max_val > 1.0:
            noisy_audio = noisy_audio / max_val
            clean_audio = clean_audio / max_val # Keep scale consistent
            
        # 4. Save mixed audio
        base_name = clean_path.stem
        audio_out_path = OUT_AUDIO_DIR / f"{base_name}_noisy.wav"
        sf.write(audio_out_path, noisy_audio, SR)
        
        # 5. Generate and Save Spectrograms (.npy format)
        spec_noisy = compute_spectrogram(noisy_audio)
        spec_clean = compute_spectrogram(clean_audio)
        
        np.save(OUT_SPEC_NOISY / f"{base_name}_noisy.npy", spec_noisy)
        np.save(OUT_SPEC_CLEAN / f"{base_name}_clean.npy", spec_clean)
        
        return {"file_id": base_name, "noise_used": noise_path.name, "snr": round(target_snr, 2)}
        
    except Exception as e:
        return None

if __name__ == '__main__':
    # Gather file paths
    clean_files = list(CLEAN_DIR.rglob("*.flac"))
    noise_files = list(NOISE_DIR.rglob("*.wav"))
    
    print(f"Found {len(clean_files)} clean files and {len(noise_files)} noise files.")
    
    # Run multi-processing
    results = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_file, cf, noise_files) for cf in clean_files]
        
        for future in tqdm(futures, total=len(clean_files), desc="Processing Dataset"):
            res = future.result()
            if res:
                results.append(res)
                
    # Save Metadata CSV
    df = pd.DataFrame(results)
    df.to_csv("dataset/metadata.csv", index=False)
    print("Dataset generation complete!")