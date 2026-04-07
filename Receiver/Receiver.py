# AFRLS filter based receiver for bionic covert signalling

import numpy as np
from scipy.signal import resample, butter, sosfilt
import matplotlib.pyplot as plt

# AFRLS (Adaptive Fast Recursive Least Squares) Filter
class AFRLS1:
    def __init__(self, L=128, delta=10.0, a=0.98, b=2):
        self.L = L
        self.h = np.zeros(L)
        self.P = (1/delta)*np.eye(L)
        self.a = a
        self.b = b

    def lam1(self, e):
        return self.a + (2/np.pi)*(1-self.a)*np.arctan(1/(2*self.b*e+1e-8))

    def run1(self, x, d):
        N = len(x)
        e_out = np.zeros(N)

        for n in range(self.L, N):
            x_vec = x[n-self.L:n][::-1]
            y = np.dot(self.h, x_vec)
            e = d[n] - y
            lam = self.lam1(e)
            Px = self.P @ x_vec
            K = Px / (lam + x_vec.T @ Px)
            self.h += K * e
            self.P = (self.P - np.outer(K, x_vec.T @ self.P)) / lam
            e_out[n] = e

        return e_out

def generate_hfm1(fs, f1, f2, T, N):
    t = np.linspace(0, T, N)
    k = (f1 - f2)/(f2*T)
    phase = (2*np.pi*f1/k)*np.log(1+k*t)
    return np.cos(phase)

# Decoder
def decode1(signal, codebook, num_symbols):
    samples_per_symbol = len(signal)//num_symbols
    detected = []

    for i in range(num_symbols):
        seg = signal[i*samples_per_symbol:(i+1)*samples_per_symbol]

        scores = []
        for sym, code in codebook.items():
            code_r = resample(code, len(seg))
            score = np.sum(seg * code_r) / (np.linalg.norm(seg)+1e-8)
            scores.append((sym, score))
            
        detected.append(max(scores, key=lambda x:x[1])[0])
    return detected

def symbols_to_bits1(symbols, m_ary=2):
    bits = []
    for s in symbols:
        bits += [int(x) for x in format(s, f'0{m_ary}b')]
    return bits

def run_receiver1(received, fs, f1, f2, T, codebook, tx_bits):

    received = received / np.max(np.abs(received))
    ref = generate_hfm1(fs, f1, f2, T, len(received))
    ref = ref / np.max(np.abs(ref))

    af = AFRLS1(L=64)
    extracted = af.run1(received, ref)

    sos = butter(6, [2000, 10000], btype='bandpass', fs=fs, output='sos')
    filtered = sosfilt(sos, extracted)

    fc = (f1 + f2)/2
    t = np.arange(len(filtered))/fs
    demod = filtered * np.cos(2*np.pi*fc*t)

    num_symbols = int(np.ceil(len(tx_bits)/2))
    symbols = decode1(demod, codebook, num_symbols)
    rx_bits = symbols_to_bits1(symbols)

    return rx_bits, extracted
    
# WCC(Waveform Correlation Coefficient) - similarity between the original call and data embedded signal
def compute_wcc1(x, y):
    x = x / np.max(np.abs(x))
    y = y / np.max(np.abs(y))
    return np.corrcoef(x, y)[0,1]
    
# MelD (Mel-scale Cepstral Distance) - shows how well the data is hidden
def compute_meld1(x, y, fs):
    mel_x = librosa.feature.melspectrogram(y=x, sr=fs)
    mel_y = librosa.feature.melspectrogram(y=y, sr=fs)
    mel_x_db = librosa.power_to_db(mel_x)
    mel_y_db = librosa.power_to_db(mel_y)

    return np.mean(np.abs(mel_x_db - mel_y_db))
    
# BER (Bit Error Rate) - Rate of difference between original and decoded bits
def ber1(tx, rx):
    tx = np.array(tx)
    rx = np.array(rx[:len(tx)])
    return np.mean(tx != rx)    

# BER vs SNR curve 
def ber_plot1(tx1, carrier, bits, codebook, fs, f1, f2, T):

    snr_range = np.arange(-15, 26, 5)
    ber_values = []

    for snr in snr_range:

        data_wave, _ = tx1.mgss_modulation(bits, len(carrier))
        stego = tx1.data_embedding(carrier, data_wave, snr_target=-30)

        received = tx1.ocean_environment(stego, snr=snr)

        rx_bits, _ = run_receiver1(
            received,
            fs,
            f1,
            f2,
            T,
            codebook,
            bits
        )

        ber_val = ber1(bits, rx_bits)
        ber_values.append(ber_val)

    plt.figure()
    plt.plot(snr_range, ber_values, marker='o')
    plt.title("BER vs SNR Curve")
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.grid()
    plt.show()

    return snr_range, ber_values