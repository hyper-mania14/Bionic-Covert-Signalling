#Bionic Embedded Signalling for Sparrow calls (similar representation in air)

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample, welch, windows # <--- Changed this
from IPython.display import Audio, display

class AerialBionicTransmitter:
    def __init__(self, fs=44100):
        self.fs = fs
        self.chip_length = 127 

    def generate_bird_chirp(self, f_start, f_end, T):
        t = np.linspace(0, T, int(self.fs * T))
        k = (f_start - f_end) / (f_end * T)
        phase = (2 * np.pi * f_start / k) * np.log(1 + k * t)
        chirp = np.cos(phase)
        
        # Use the windows submodule for Tukey
        # alpha=0.1 means 10% of the signal is spent fading in/out
        win = windows.tukey(len(chirp), alpha=0.1) 
        return chirp * win, t

    def hide_data_air(self, carrier, bits, snr_db=-15):
        np.random.seed(7)
        num_symbols = len(bits)
        spread_factor = len(carrier) // num_symbols
        
        spread_signal = np.array([])
        for bit in bits:
            chip = np.random.choice([-1, 1], spread_factor)
            spread_signal = np.append(spread_signal, chip * (1 if bit==1 else -1))
        
        spread_signal = resample(spread_signal, len(carrier))
        
        p_c = np.mean(carrier**2)
        p_d_target = p_c / (10**(abs(snr_db)/10))
        scalar = np.sqrt(p_d_target / np.mean(spread_signal**2))
        
        return carrier + (spread_signal * scalar)

# execution
tx_air = AerialBionicTransmitter()

# Sparrow: 150ms, sweeping 12kHz down to 4kHz
T_note = 0.15 
f1, f2 = 12000, 4000 
message = [1, 0, 1, 1, 0] 

bird_note, t_air = tx_air.generate_bird_chirp(f1, f2, T_note)
stego_bird = tx_air.hide_data_air(bird_note, message)

# plots
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(t_air, stego_bird, color='goldenrod', lw=0.5)
plt.title("Aerial Waveform : Tukey Window")

plt.subplot(1, 2, 2)
plt.specgram(stego_bird, Fs=tx_air.fs, NFFT=512, noverlap=256)
plt.title("Spectrogram : Bird Chirp")
plt.ylim(0, 18000)
plt.show()

print("Sparrow Stealth Signal:")
display(Audio(stego_bird, rate=tx_air.fs))

