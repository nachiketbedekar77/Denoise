import torch
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from train import UNetDenoiseMask

def denoise_audio(noisy_wav_path, output_wav_path="denoised_output.wav"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load trained U-Net Masking model
    model = UNetDenoiseMask().to(device)
    model.load_state_dict(torch.load("denoise_model.pth", map_location=device, weights_only=True))
    model.eval()

    # 2. Load noisy audio at 16kHz
    SR = 16000
    audio, _ = librosa.load(noisy_wav_path, sr=SR)
    
    # 3. Compute STFT
    N_FFT = 512
    HOP_LENGTH = 256
    stft = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mag = np.abs(stft)
    phase = np.angle(stft)

    # Slice 257 frequency bins to 256
    mag_256 = mag[:256, :]
    total_frames = mag_256.shape[1]
    
    chunk_size = 256
    stride = 128  
    
    cleaned_mag = np.zeros_like(mag_256)
    weight_matrix = np.zeros_like(mag_256)

    print(f"Applying Tactical Military Bandpass + AI DSP... (Frames: {total_frames})")

    with torch.no_grad():
        for start in range(0, total_frames, stride):
            end = min(start + chunk_size, total_frames)
            chunk = mag_256[:, start:end]
            actual_len = chunk.shape[1]
            
            if actual_len < chunk_size:
                pad_w = chunk_size - actual_len
                chunk_padded = np.pad(chunk, ((0, 0), (0, pad_w)), mode='constant')
            else:
                chunk_padded = chunk

            # AI Prediction
            tensor_input = torch.from_numpy(np.log1p(chunk_padded)).float().unsqueeze(0).unsqueeze(0).to(device)
            predicted_mask = model(tensor_input).squeeze().cpu().numpy()[:, :actual_len]
            
            # =======================================================
            # 🛡️ DEFENSE-GRADE DSP LOGIC (The Secret Sauce) 🛡️
            # =======================================================
            
            # 1. MILITARY BANDPASS FILTER (300Hz - 3400Hz)
            # (16000Hz / 512 FFT = ~31.25 Hz per bin)
            # Bin 0-9 (< 312 Hz): Kills engine rumble, wind, and explosions
            # Bin 110-256 (> 3437 Hz): Kills static hiss and sharp siren noises
            bandpass = np.ones_like(predicted_mask)
            bandpass[:10, :] = 0.05   # Massive cut to low-end rumble
            bandpass[110:, :] = 0.10  # Massive cut to high-end hiss
            
            # Apply hardware-style filter to AI mask
            tactical_mask = predicted_mask * bandpass
            
            # 2. ADAPTIVE SQUELCH (Noise Gate)
            if np.max(tactical_mask) < 0.15:
                cleaned_chunk = chunk * 0.0  # Zero transmission on silence
            else:
                # 3. ANTI-ROBOTIC WIENER MASK
                # Floor of 0.03 leaves a tiny natural background so voice doesn't sound "underwater"
                final_mask = np.clip(tactical_mask, 0.03, 1.0) ** 2.0
                
                # 4. VOCAL PRESENCE ENHANCEMENT
                # Boost human voice frequencies back up heavily
                cleaned_chunk = chunk * final_mask * 2.8
            
            # =======================================================

            cleaned_mag[:, start:start + actual_len] += cleaned_chunk
            weight_matrix[:, start:start + actual_len] += 1.0

    weight_matrix[weight_matrix == 0] = 1.0
    cleaned_mag /= weight_matrix
    
    full_cleaned_mag = np.vstack([cleaned_mag, np.zeros((1, total_frames))])
    cleaned_stft = full_cleaned_mag * np.exp(1j * phase)
    cleaned_audio = librosa.istft(cleaned_stft, hop_length=HOP_LENGTH, length=len(audio))

    # TACTICAL LIMITER: Prevents audio from cracking on loud shouts
    cleaned_audio = np.tanh(cleaned_audio)

    sf.write(output_wav_path, cleaned_audio, SR)
    print(f"✅ Tactical Grade Audio saved successfully to: {output_wav_path}")

if __name__ == "__main__":
    target_file = "audio.wav"
    
    if Path(target_file).exists():
        denoise_audio(target_file, "denoised_output.wav")
    else:
        print(f"Error: '{target_file}' not found in project folder.")