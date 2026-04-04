#Bionic Embedded Signalling for Dolphin calls

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample, windows
from IPython.display import Audio, display

class DolphinBionicTransmitter:
    def __init__(self, fs=96000): # Dolphins = higher sample rates (96kHz)
        self.fs = fs

    def generate_dolphin_whistle(self, f_start, f_end, T):
        t = np.linspace(0, T, int(self.fs * T))
        # Upward HFM Sweep
        k = (f_end - f_start) / (f_start * T)
        phase = (2 * np.pi * f_start / k) * np.log(1 + k * t)
        whistle = np.cos(phase)
        
        #hanning window for smooth edges (simulate a real dolphin whistle)
        win = windows.hann(len(whistle))
        return whistle * win, t

# working
tx_dolphin = DolphinBionicTransmitter()

# real dolphin whistle is about 0.8 seconds
# Sweeping from 8kHz to 18kHz
T_dolph = 0.8
f1, f2 = 8000, 18000 

dolph_whistle, t_d = tx_dolphin.generate_dolphin_whistle(f1, f2, T_dolph)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(t_d, dolph_whistle, color='green')
plt.title("Dolphin Call")

plt.subplot(1, 2, 2)
plt.specgram(dolph_whistle, Fs=tx_dolphin.fs, NFFT=512, noverlap=256, cmap='winter')
plt.title("Dolphin HFM Spectrogram")
plt.ylim(0, 25000)
plt.show()

print("Stealth Dolphin Whistle:")
display(Audio(dolph_whistle, rate=tx_dolphin.fs))