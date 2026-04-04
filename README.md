# Bionic Covert Signalling
#### A Python Implementation of Bio-Acoustic Steganography

This repository provides a Python-based framework for hiding digital data within biological soundscapes. By mimicking the acoustic signatures of Marine and Terrestrial wildlife, we achieve Low Probability of Detection (LPD) for covert data transmission.

**Reference Paper**: This implementation is inspired by research from the paper cited below:
 J. Du, Y. Zhang and G. Chen, "Bionic Covert Hydroacoustic Spread Spectrum Communication based on Improved Adaptive Filtering Algorithm," 2024 4th International Conference on Neural Networks, Information and Communication Engineering (NNICE), Guangzhou, China, 2024 

---

 ### Implementation

 #### PART A : The Transmitter and Encoder
 The transmitter's goal is to produce an audio file that is indistinguishable from natural biological noise but carries a hidden digital payload.

 ##### 1. Bionic Carrier Generation
 To ensure the signal blends into the environment, the implementation is tested in three distinct biological carriers:
 * **Sperm Whale:** High-energy HFM (Hyperbolic Frequency Modulation) whistles and *Codas* (clicks) for deep-sea stealth.
 * **Dolphin:** High-frequency whistles for coastal and shallow water environments.
 * **Sparrow:** Complex, frequency-modulated chirps for terrestrial and forest-based environments.

 ##### 2. MGSS Modulation (Multi-Group Spread Spectrum)
 Standard digital bits are structured and can easily be detected by interceptors. This implementation uses **MGSS** to split and redistribute the data:
 * **M-ary Encoding:** Groups bits into pairs (e.g., `[1,0]`), mapping them to one of 4 unique symbols to increase data density.
 * **Orthogonal PN Library:** Each symbol is replaced by a unique Pseudo-Noise (PN) sequence (255 *chips*). This transforms the data into a signal that resembles white noise.

 ##### 3. Environment (Channel) Simulation
 This would not be required in an actual transmission setting, but to get as close as possible, to a real received input, we used channel estimation via noise addition to simulate a real environment.
 
 * **Whale** : pink noise (decrease in intensity with increase in frequency) to simulate the deep sea environment (kind of like a bass hum), Mulitple echos (reflections off the sea floor) and Doppler effect 
 * **Dolphin** : Multiple echos (reflections of seabed), HPF white noise (ocean surface / waves) and wider Doppler spread (since dolphins move faster than whales)
 * **Sparrow** : LPF (limit range of air travel), possible echos (forests/ mountainous terrain) and white/pink noise (for windy and forset environments)

---

 #### PART B : The Receiver and Decoder
 The receiver's challenge is to extract a hidden signal that is 100x quieter than the carrier out of a noisy, distorted environment.

 ##### 1. Envelope Detection & Sync
 The receiver uses the Hilbert Transform to extract the *envelope* (the volume outline). It uses the animal's natural pulses (like whale clicks) as time-sync anchors to determine exactly where the data sequence begins.

 ##### 2. Adaptive Filtering & Despreading
 Based on the referenced research, the receiver uses an Adaptive Filtering Algorithm to suppress the carrier (the loud whale sound) and isolate the payload (the quieter data).

 * **Cross-Correlation**: The receiver slides its local copy of the PN keys against the filtered audio.

 * **Correlation Peaks**: When a match is found, a sharp peak appears in the detector, allowing the receiver to see and extract the bits through the noise.

---

### Plots

##### Data signal (sent and extracted)
##### Spectogram
##### Power Spectral Density 

---

### Where it could be used (i.e applications)

#### Real-World Use Cases
* **Submarine Stealth**: Sending emergency status updates without revealing a position.

* **Environmental Monitoring**: Collecting data from sensors in protected marine areas without disturbing wildlife.

* **Covert Terrestrial Comms**: Using bird chirps for border security sensors in restricted radio zones.

#### Possible Future Developments

* **AI-Adaptive Masking**: Using neural networks to mimic the exact individual animal currently nearby in the environment.

* **High-Order M-ary**: Increasing data rates (e.g., 16-ary) while maintaining a low biological footprint.

* **Dynamic Channel Estimation**: Real-time correction for water temperature and salinity changes.

---

### Files included

* Python codes (.py files)
* Jupyter notebook files (.ipynb files)
* Waveforms and plots
* Readme 

--- 