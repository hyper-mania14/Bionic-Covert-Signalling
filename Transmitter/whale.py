#Bionic Embedded Signalling for Sperm Whale calls (with multiple secret message datasets)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec #to allow plotting
from scipy.signal import resample, welch, resample_poly, hilbert #resampling, Power Spectral Density, hilbert transform
from IPython.display import Audio, display

class wave_transmitter:
    def __init__(src, fs=44100):
        src.fs = fs
        src.m_ary = 4  # 2 bits per symbol : 00, 01, 10, 11
        src.chip_length = 255  #number of bits used to represent secret data

    def whale_whistle(src, f1, f2, T):
        #Hyperbolic Frequency Modulation whistle. Varies with frequency to accomodate for Doppler effect for bodies moving in the ocean
        t = np.linspace(0, T, int(src.fs * T))
        k = (f1 - f2) / (f2 * T) # modulation / sweep constant
        phase = (2 * np.pi * f1 / k) * np.log(1 + k * t) #logarthmic phase
        return np.cos(phase), t #whistle signal

    def whale_clicks(src, T, num_clicks=5):
        # Use sperm whale clicks to make it realistic & hide high amplitude data
        total_samples = int(src.fs * T)
        click_signal = np.zeros(total_samples)
        click_positions = np.linspace(0.1, T-0.1, num_clicks)
        
        pulse_len = int(src.fs * 0.01) # 10ms pulse
        pulse_t = np.linspace(-0.005, 0.005, pulse_len)
        pulse = np.exp(- (pulse_t**2) / (2 * 0.001**2)) * np.cos(2*np.pi*5000*pulse_t) #gaussian window (smoothening) with 5kHz cosine carrier
        
        for pos in click_positions:
            idx = int(pos * src.fs) #repeats clicks at fixed positions given by indexes
            start = idx - pulse_len // 2
            end = start + pulse_len
            if start >= 0 and end <= total_samples:
                click_signal[start:end] += pulse #cumulative signal
        return click_signal

    def mgss_modulation(src, bits, target_len):
        #M-ary encoding to allow 2 bits of data per symbol (reduces amt of bits to send)
        
        np.random.seed(42) #fixed to produce same sequence
        #grouping bits (M=4) bits in pairs
        if len(bits) % 2 != 0:
            bits = bits + [0]
        #00=0, 01=1, 10=2, 11=3    
        symbols = [int("".join(map(str, bits[i:i+2])), 2) for i in range(0, len(bits), 2)]
        
        #orthogonal PN Library (key of 1 symbol is noise to other)
        #Creating unique keys for each symbol (strings of -1 and 1 of chip_length) 
        lib = {i: np.random.choice([-1, 1], src.chip_length) for i in range(4)}
        
        #spreading : symbols replaced with pulses with PN values
        spread = np.concatenate([lib[s] for s in symbols])
        
        #resample to the carrier length
        return resample(spread, target_len), lib

    def data_embedding(src, carrier, data, snr_target=-20):

        p_carrier = np.mean(carrier**2) #mean square for avg power
        p_req_data = p_carrier / (10**(abs(snr_target) / 10)) #power of signal for its SNR to be 20dB lesser than whistle
        curr_p_data = np.mean(data**2)
        adj_factor = np.sqrt(p_req_data / curr_p_data) # amplitude factor to tone SNR of hidden data to be 0.01 of whale whistle power
        
        secret_signal = carrier + (adj_factor * data)
        return secret_signal / np.max(np.abs(secret_signal)) #fixes range of amplitude [-1.0 ,1.0]

    def ocean_environment(src, signal, snr, doppler_fact=1.001):
        #Creating a channel like an ocean environment (doppler effect + multiple echos + pink gaussian noise)
        
        # doppler effect simulation (like transmitter moving)
        y = resample_poly(signal, int(1000 * doppler_fact), 1000)
        y = y[:len(signal)] if len(y) > len(signal) else np.pad(y, (0, len(signal) - len(y)))
        
        # multipath echo (each with a 15ms delay) (simulating reflections off seabed)
        delay = int(src.fs * 0.015)
        echo = np.zeros_like(y)
        echo[delay:] = y[:-delay] * 0.4 #echo at 40% original volume (adjustable)
        y = y + echo #orginal signal + echos
        
        # pink noise (simulating deep sea low frequency static)
        white_noise = np.random.randn(len(y))
        pink_noise = np.cumsum(white_noise) #cumulative sum of the gaussian white noise = deeper sound
        pink_noise -= np.mean(pink_noise)
        pink_noise /= np.max(np.abs(pink_noise))
        
        sig_pwr = np.mean(y**2)
        noise_pwr = sig_pwr / (10**(snr / 10)) #ratio of noise power to add to original signal
        return y + pink_noise * np.sqrt(noise_pwr)


#working
tx = wave_transmitter()
T = 1.0
whale_w, t = tx.whale_whistle(8000, 2000, T)
clicks = tx.whale_clicks(T)
carrier = (whale_w + clicks) / 2 # carrier

datasets = [
    [1, 0, 1, 1],                    # secret data 1
    [0, 1, 0, 0, 1, 1],              # secret data 2
    [1, 1, 1, 1, 0, 0, 1, 0]         # secret data 3
]

#representing secret dataset in time domain
plt.figure(figsize=(18, 10))

for i, bits in enumerate(datasets):
    #same as carrier length
    data_wave, _ = tx.mgss_modulation(bits, len(carrier))
    plt.subplot(3, 1, i+1)
    plt.plot(t, data_wave, color='blue', lw=0.8, alpha=0.8)
    plt.title(f"Secret Data {i+1} time domain : (spread bits: {bits})", fontsize=14)
    plt.ylabel("Normalized Amplitude")
    if i == len(datasets) - 1:
        plt.xlabel("Time (s)")
    plt.grid(alpha=0.3)
    plt.ylim(-1.5, 1.5) # PN sequences are bipolar (-1, 1)

plt.tight_layout()
plt.show()

#Zoomed time domain representation for each dataset
plt.figure(figsize=(18, 12)) 
zoom_samples = int(tx.fs * 0.02) # 20ms window
for i, data in enumerate(datasets, 1):
    data_wave, _ = tx.mgss_modulation(data, len(carrier))
    plt.subplot(3, 1, i)
    plt.step(t[:zoom_samples], data_wave[:zoom_samples], where='post', color='purple', lw=1.5)
    plt.title(f"20ms of Secret Data {i}", fontsize=14, fontweight='bold')
    plt.ylabel("Chip Value")
    plt.grid(alpha=0.4, linestyle='--')
    if i == 3:
        plt.xlabel("Time (s)")
plt.tight_layout()
plt.show()


for i, bits in enumerate(datasets):
    print(f"\nSecret Data {i+1}:")
    
    # 1.modulating and embedding secret message
    data_wave, code_book = tx.mgss_modulation(bits, len(carrier))
    stego = tx.data_embedding(carrier, data_wave, snr_target=-20) #stego = steganography (hiding messages in signals)
    
    # 2.simulate the ocean (transmission channel here)
    received = tx.ocean_environment(stego, snr=18)

    plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(2, 3)

    # 1: waveform (3ms window)
    plt.subplot(gs[0, 0])
    num_samples = int(tx.fs* 0.006 ) 
    t_zoom = t[:num_samples]
    plt.plot(t_zoom, stego[:num_samples], color='red', alpha=0.3, label='Stealth Signal')
    plt.plot(t_zoom, carrier[:num_samples], color='green', alpha=0.9, label='Whale Whistle')
    plt.title(f"Waveform (Secret data {i+1})")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)

    # 2: Spectrogram (relation between frequency, time and intensity)
    plt.subplot(gs[0, 1])
    plt.specgram(received, Fs=tx.fs, NFFT=1024, noverlap=512, cmap='viridis')
    plt.title(f"Ocean Spectrogram (Received Signal)")
    plt.ylabel("Frequency (Hz)")
    plt.ylim(0, 10000)

    # 3: Power Spectral Density Comparison (to check if it has low probability of detection)
    plt.subplot(gs[0, 2])
    #Welch method to estimate power of 1024 samples and then average it
    #1024 for higher resolution of spectrogram
    f_c, psd_c = welch(carrier, tx.fs, nperseg=1024) 
    f_s, psd_s = welch(stego, tx.fs, nperseg=1024)
    #logarithmic scale for plotting
    plt.semilogy(f_c, psd_c, label='Pure Whale Whistle') 
    plt.semilogy(f_s, psd_s, label='Stealth Signal', alpha=0.7) #alpha sets opacity
    plt.title(f"PSD Comparison (Secret data {i+1})")
    plt.grid(alpha=0.3)
    plt.legend()

    #4: Time-Frequency Masking
    plt.subplot(gs[1, :])
    plt.specgram(stego, Fs=tx.fs, NFFT=1024, noverlap=512, cmap='plasma')
    envelope = np.abs(hilbert(stego)) #uses hilbert transform for smooth envelope instead of raw data
    # overlaying the envelope to show data sync with whale call
    plt.plot(t, (envelope * 4000) + 1000, color='cyan', alpha=0.6, label='Hidden data envelope')
    plt.title(f"Time-Frequency Masking of Hidden Data ")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Audio output
    print(f" Secret data {i+1} - Stealth Audio Output:")
    display(Audio(stego, rate=tx.fs))
    print(f"Secret data {i+1} - Received Signal via Ocean :")
    display(Audio(received, rate=tx.fs))