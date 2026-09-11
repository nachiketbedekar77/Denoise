import librosa
import numpy as np
import matplotlib.pyplot as plt
from pesq import pesq
from pystoi import stoi
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. EVALUATION FUNCTIONS
# ==========================================
def calculate_snr(clean, processed):
    """Calculates Signal-to-Noise Ratio"""
    noise = processed - clean
    signal_power = np.sum(clean ** 2)
    noise_power = np.sum(noise ** 2) + 1e-8
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def evaluate_audio(clean_path, noisy_path, denoised_path, measured_latency_ms):
    print("🔄 Loading audio files (resampling to 16kHz)...")
    # Load audio at 16kHz as per defence edge deployment standards
    clean_wav, sr = librosa.load(clean_path, sr=16000)
    noisy_wav, _ = librosa.load(noisy_path, sr=16000)
    denoised_wav, _ = librosa.load(denoised_path, sr=16000)

    # Ensure lengths match perfectly for metric calculations
    min_len = min(len(clean_wav), len(noisy_wav), len(denoised_wav))
    clean_wav = clean_wav[:min_len]
    noisy_wav = noisy_wav[:min_len]
    denoised_wav = denoised_wav[:min_len]

    print("📊 Calculating Metrics...")
    # 1. PESQ (Perceptual Evaluation of Speech Quality) - Wideband for 16kHz
    pesq_score = pesq(16000, clean_wav, denoised_wav, 'wb')
    
    # 2. STOI (Short-Time Objective Intelligibility)
    stoi_score = stoi(clean_wav, denoised_wav, 16000, extended=False)
    
    # 3. SNR Improvement (Delta SNR)
    snr_noisy = calculate_snr(clean_wav, noisy_wav)
    snr_denoised = calculate_snr(clean_wav, denoised_wav)
    delta_snr = snr_denoised - snr_noisy

    return pesq_score, stoi_score, delta_snr, measured_latency_ms

# ==========================================
# 2. VISUALIZATION DASHBOARD (MATPLOTLIB)
# ==========================================
def generate_dashboard(pesq_val, stoi_val, snr_val, latency_val):
    # SIH Target Metrics
    targets = {
        'PESQ': 2.5,
        'STOI': 0.85,
        'SNR (dB)': 15.0,
        'Latency (ms)': 20.0
    }
    
    achieved = {
        'PESQ': round(pesq_val, 2),
        'STOI': round(stoi_val, 3),
        'SNR (dB)': round(snr_val, 2),
        'Latency (ms)': round(latency_val, 1)
    }

    fig = plt.figure(figsize=(12, 8))
    fig.canvas.manager.set_window_title('SIH ANC Model Evaluation')
    plt.suptitle("AI/ML Adaptive Noise Cancellation - Performance Dashboard", fontsize=16, fontweight='bold')

    # --- Create Bar Charts ---
    metrics = list(targets.keys())
    target_vals = list(targets.values())
    achieved_vals = list(achieved.values())

    # We use subplots for bars
    ax_bar = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    
    x = np.arange(len(metrics))
    width = 0.35

    rects1 = ax_bar.bar(x - width/2, target_vals, width, label='SIH Target', color='#3498db')
    rects2 = ax_bar.bar(x + width/2, achieved_vals, width, label='Model Achieved', color='#2ecc71')

    # Make Latency red if it missed the target, green if it beat it
    if achieved['Latency (ms)'] > targets['Latency (ms)']:
        rects2[3].set_color('#e74c3c') # Red for high latency

    ax_bar.set_ylabel('Scores / Values', fontsize=12)
    ax_bar.set_title('Target vs. Achieved Metrics', fontsize=14)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metrics, fontsize=12)
    ax_bar.legend()

    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax_bar.annotate(f'{height}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    autolabel(rects1)
    autolabel(rects2)

    # --- Create Comparison Table ---
    ax_table = plt.subplot2grid((3, 1), (2, 0))
    ax_table.axis('tight')
    ax_table.axis('off')

    cell_text = []
    colors = []
    
    for m in metrics:
        target = targets[m]
        achieve = achieved[m]
        
        # Logic for Pass/Fail Checkmark
        if m == 'Latency (ms)':
            status = '✅ PASS' if achieve <= target else '❌ FAIL'
            color = '#d4edda' if achieve <= target else '#f8d7da'
        else:
            status = '✅ PASS' if achieve >= target else '❌ FAIL'
            color = '#d4edda' if achieve >= target else '#f8d7da'
            
        cell_text.append([m, str(target), str(achieve), status])
        colors.append(['#ffffff', '#ffffff', '#ffffff', color])

    columns = ('Metric', 'SIH Target', 'Model Output (Actual)', 'Status')
    table = ax_table.table(cellText=cell_text, colLabels=columns, cellColours=colors,
                           loc='center', cellLoc='center', colWidths=[0.25, 0.2, 0.25, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.8)

    plt.tight_layout()
    plt.savefig('evaluation_dashboard.png', dpi=300, bbox_inches='tight')
    print("✅ Dashboard generated and saved as 'evaluation_dashboard.png'")
    plt.show()

# ==========================================
# 3. RUN THE SCRIPT
# ==========================================
if __name__ == "__main__":
    # 1. The EXACT Original Clean File (Answer Key)
    CLEAN_AUDIO_PATH = r"C:\Users\Nachiket\Desktop\Denoise\Raw-data\clean\84\121123\84-121123-0004.flac"
    
    # 2. The Mixed Noisy File (Test Paper)
    NOISY_AUDIO_PATH = r"C:\Users\Nachiket\Desktop\Denoise\Dataset\noisy_audio\84-121123-0004_noisy.wav"
    
    # 3. Your AI's Output for that specific noisy file
    DENOISED_AUDIO_PATH = r"C:\Users\Nachiket\Desktop\Denoise\denoised_output.wav" 
    
    MEASURED_LATENCY = 24.0 

    try:
        pesq_val, stoi_val, snr_val, latency_val = evaluate_audio(
            CLEAN_AUDIO_PATH, NOISY_AUDIO_PATH, DENOISED_AUDIO_PATH, MEASURED_LATENCY
        )
        generate_dashboard(pesq_val, stoi_val, snr_val, latency_val)
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")