import numpy as np

SR = 16000
N_FFT = 512
WINDOW_SIZE = 512
CHUNK_SIZE = 128
FREQ_BINS = 256


def periodic_hann(n):
    k = np.arange(n)
    return (0.5 - 0.5 * np.cos(2 * np.pi * k / n)).astype(np.float64)


def sqrt_hann(n):
    return np.sqrt(periodic_hann(n))


class StreamingOLA:
    def __init__(self):
        self.in_buffer = np.zeros(WINDOW_SIZE, dtype=np.float64)
        self.out_buffer = np.zeros(WINDOW_SIZE, dtype=np.float64)
        self.norm_buffer = np.zeros(WINDOW_SIZE, dtype=np.float64)
        self.analysis_window = sqrt_hann(WINDOW_SIZE)
        self.synthesis_window = sqrt_hann(WINDOW_SIZE)
        self.window_product = self.analysis_window * self.synthesis_window

    def push_input(self, chunk):
        self.in_buffer = np.roll(self.in_buffer, -CHUNK_SIZE)
        self.in_buffer[-CHUNK_SIZE:] = chunk

    def analyze(self):
        windowed = self.in_buffer * self.analysis_window
        spectrum = np.fft.rfft(windowed, n=N_FFT)          # 257 bins
        mag = np.abs(spectrum)[:FREQ_BINS]                  # drop Nyquist bin -> 256
        phase = np.angle(spectrum)[:FREQ_BINS]
        return mag, phase

    def synthesize(self, mag, phase):
        full_mag = np.concatenate([mag, [0.0]])              # re-pad Nyquist bin = 0
        full_phase = np.concatenate([phase, [0.0]])
        spectrum = full_mag * np.exp(1j * full_phase)
        frame = np.fft.irfft(spectrum, n=N_FFT) * self.synthesis_window
        return frame

    def push_output(self, frame):
        self.out_buffer = np.roll(self.out_buffer, -CHUNK_SIZE)
        self.out_buffer[-CHUNK_SIZE:] = 0.0
        self.out_buffer += frame

        self.norm_buffer = np.roll(self.norm_buffer, -CHUNK_SIZE)
        self.norm_buffer[-CHUNK_SIZE:] = 0.0
        self.norm_buffer += self.window_product

        norm = np.maximum(self.norm_buffer[:CHUNK_SIZE], 1e-8)
        return (self.out_buffer[:CHUNK_SIZE] / norm).astype(np.float64)


# ---- 1. Check the COLA normalization envelope reaches a flat constant ----
ola = StreamingOLA()
n_hops = WINDOW_SIZE // CHUNK_SIZE + 4
envelope = []
dummy_frame = np.zeros(WINDOW_SIZE)
for i in range(n_hops):
    ola.out_buffer = np.roll(ola.out_buffer, -CHUNK_SIZE)
    ola.out_buffer[-CHUNK_SIZE:] = 0.0
    ola.norm_buffer = np.roll(ola.norm_buffer, -CHUNK_SIZE)
    ola.norm_buffer[-CHUNK_SIZE:] = 0.0
    ola.norm_buffer += ola.window_product
    envelope.append(ola.norm_buffer[:CHUNK_SIZE].copy())

steady = np.concatenate(envelope[4:])
print("Normalization envelope steady-state: min=%.6f max=%.6f (flat => COLA holds)" %
      (steady.min(), steady.max()))

# ---- 2. End-to-end identity reconstruction test (mask = 1 everywhere) ----
rng = np.random.default_rng(0)
duration_s = 1.5
n_samples = int(SR * duration_s)
t = np.arange(n_samples) / SR
test_signal = (
    0.6 * np.sin(2 * np.pi * 220 * t) +
    0.3 * np.sin(2 * np.pi * 1200 * t) +
    0.05 * rng.standard_normal(n_samples)
)

ola = StreamingOLA()
output = np.zeros(n_samples)
n_chunks = n_samples // CHUNK_SIZE
for i in range(n_chunks):
    chunk = test_signal[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE]
    ola.push_input(chunk)
    mag, phase = ola.analyze()
    frame = ola.synthesize(mag, phase)   # mask = 1 (identity) -> should reproduce input
    out_chunk = ola.push_output(frame)
    output[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE] = out_chunk

# find the pipeline delay empirically via cross-correlation, then align + trim transients
corr = np.correlate(output[:4000], test_signal[:4000], mode='full')
lag = corr.argmax() - (4000 - 1)
print(f"Measured pipeline delay: {lag} samples = {lag / SR * 1000:.2f} ms "
      f"(theoretical (N-H)/SR = {(WINDOW_SIZE - CHUNK_SIZE) / SR * 1000:.2f} ms)")

# trim first/last window worth of samples (startup/shutdown transient) and align by lag
trim = WINDOW_SIZE * 2
a = test_signal[trim: n_samples - trim]
b = output[trim + lag: n_samples - trim + lag]
m = min(len(a), len(b))
a, b = a[:m], b[:m]

err = a - b
recon_snr_db = 10 * np.log10(np.sum(a ** 2) / np.sum(err ** 2))
print(f"Reconstruction fidelity (mask=1, steady-state, aligned): {recon_snr_db:.2f} dB SNR vs original")
print(f"Max abs sample error: {np.max(np.abs(err)):.6f} (signal peak ~{np.max(np.abs(a)):.3f})")
