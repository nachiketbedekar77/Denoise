import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_comparison(noisy_path, denoised_path, output_img="spectrogram_proof.png"):
    if not os.path.exists(noisy_path) or not os.path.exists(denoised_path):
        print("Error: Audio files not found! Pehle inference.py run karke audio generate kar.")
        return

    # Audio load kar rahe hain (16kHz par)
    y_noisy, sr = librosa.load(noisy_path, sr=16000)
    y_denoised, _ = librosa.load(denoised_path, sr=16000)

    # STFT nikal kar usko Decibels (dB) me convert kar rahe hain visual plot ke liye
    d_noisy = librosa.amplitude_to_db(np.abs(librosa.stft(y_noisy, n_fft=512, hop_length=256)), ref=np.max)
    d_denoised = librosa.amplitude_to_db(np.abs(librosa.stft(y_denoised, n_fft=512, hop_length=256)), ref=np.max)

    # Figure setup (2 graphs upar-neeche)
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Pehla Graph: Noisy Audio
    img1 = librosa.display.specshow(d_noisy, sr=sr, hop_length=256, x_axis='time', y_axis='hz', ax=ax[0], cmap='magma')
    ax[0].set_title("Input: Battlefield Noisy Audio (Gunfire / Sirens / Hiss)")
    fig.colorbar(img1, ax=ax[0], format="%+2.0f dB")

    # Doosra Graph: Denoised (Clean) Audio
    img2 = librosa.display.specshow(d_denoised, sr=sr, hop_length=256, x_axis='time', y_axis='hz', ax=ax[1], cmap='magma')
    ax[1].set_title("Output: Real-Time U-Net Masked Audio (Clean Speech)")
    fig.colorbar(img2, ax=ax[1], format="%+2.0f dB")

    plt.tight_layout()
    plt.savefig(output_img, dpi=300)
    print(f"Visual proof generated successfully: {output_img}")

if __name__ == "__main__":
    # Ensure tere folder me ye dono files is naam se available hon inference ke baad
    plot_comparison("audio.wav", "denoised_output.wav")