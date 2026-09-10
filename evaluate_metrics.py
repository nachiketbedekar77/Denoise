import numpy as np
import soundfile as sf
from pystoi import stoi

def evaluate_metrics(clean_wav_path, noisy_wav_path, denoised_wav_path, sr=16000):
    clean, _ = sf.read(clean_wav_path)
    noisy, _ = sf.read(noisy_wav_path)
    denoised, _ = sf.read(denoised_wav_path)

    # Make lengths equal
    min_len = min(len(clean), len(noisy), len(denoised))
    clean, noisy, denoised = clean[:min_len], noisy[:min_len], denoised[:min_len]

    # Calculate SNR improvement (Pure NumPy)
    noise_initial = noisy - clean
    noise_residual = denoised - clean
    
    snr_initial = 10 * np.log10(np.sum(clean**2) / (np.sum(noise_initial**2) + 1e-8))
    snr_final = 10 * np.log10(np.sum(clean**2) / (np.sum(noise_residual**2) + 1e-8))
    delta_snr = snr_final - snr_initial

    # Calculate Intelligibility Score (pystoi)
    stoi_score = stoi(clean, denoised, sr, extended=False)

    print("\n================ BENCHMARK RESULTS ================")
    print(f"Input SNR          : {snr_initial:.2f} dB")
    print(f"Output SNR         : {snr_final:.2f} dB")
    print(f"Delta SNR Gain     : +{delta_snr:.2f} dB  (Target: >15 dB)")
    print(f"STOI Intelligibility: {stoi_score:.3f}       (Target: >0.85)")
    print("===================================================\n")

if __name__ == "__main__":
    # Apni test files ka path pass karo
    # evaluate_metrics("clean_sample.wav", "noisy_sample.wav", "denoised_output.wav")
    pass