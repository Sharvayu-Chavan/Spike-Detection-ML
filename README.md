# Spike Detection ML

An interactive, Python-based desktop application for detecting spikes in EEG and Fiber Photometry data. This application implements a machine learning approach combining feature extraction based on the **Dingle Model** (Adam et al.) with **Particle Swarm Optimization (PSO)** to tune classification thresholds through a supervised labeling interface.

## Features
- **Interactive GUI**: Built with CustomTkinter for a modern, responsive user experience.
- **Data Support**: Load OpenEphys (`.dat`, `.npy`) and Photometry (`.csv`, `.ppd` stubs) files.
- **Supervised Labeling**: Drag-and-drop or load data, extract candidate peaks, and manually label 500ms snippets as "True Spike", "Artifact", or "Unsure".
- **Visual Indicators**: Displays calculated leading ($S_1$) and trailing ($S_2$) slopes directly on the waveform.
- **Automated Parameter Tuning**: Uses Particle Swarm Optimization (PSO) to learn the optimal threshold parameters based on your manual labels.
- **Export Framework**: Export the tuned parameters as a `.json` configuration file and the final candidate evaluations as a `.csv`.

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Sharvayu-Chavan/Spike-Detection-ML.git
   cd Spike-Detection-ML
   ```

2. **Install the required dependencies**:
   Ensure you have Python 3.8+ installed. Then install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
   *Dependencies include: `numpy`, `scipy`, `matplotlib`, and `customtkinter`.*

3. **Run the application**:
   ```bash
   python app.py
   ```

---

## How to Use the Application

1. **Load Data**: Click the "Load Data" button in the sidebar and select your data file.
2. **Extract Candidates**: Once data is loaded, click "Extract Candidates". The system will perform a basic threshold pass to find potential peaks (candidates) in the signal.
3. **Label Data**:
   - The main window will display a 500ms snippet centered around a candidate peak.
   - You will see the features ($S_1, S_2, R, D$) calculated for that peak.
   - Click **True Spike** (Green) if it's a valid biological spike.
   - Click **Artifact** (Red) if it's noise or a movement artifact.
   - Click **Unsure** (Gray) to skip the snippet without affecting the training set.
4. **Run Optimization**: After labeling at least a few candidate peaks, click "Run PSO Optimization". The algorithm will find the optimal mathematical thresholds to separate your spikes from the artifacts.
5. **Export Data**: Once optimization is complete, click "Export Settings & CSV" to save your tuned thresholds (`.json`) and the fully evaluated dataset (`.csv`).

---

## The Machine Learning Approach

Detecting spikes in continuous, noisy electrophysiological data (like EEG or Photometry) is challenging because movement artifacts closely resemble true neural spikes. This project applies a **Rule-Based Machine Learning** approach using feature extraction and swarm intelligence to solve this.

### 1. Feature Extraction (The Dingle Model)
Instead of feeding raw signal windows into a complex neural network, we extract biologically relevant features mathematically, inspired by the "Dingle Model" proposed by Adam et al.

For every candidate peak, we calculate:
*   **$S_1$ (Leading Slope)**: The maximum positive slope immediately preceding the peak.
*   **$S_2$ (Trailing Slope)**: The maximum negative slope immediately following the peak.
*   **$R$ (Slope Ratio)**: Calculated as $S_1 / |S_2|$. True neural spikes typically have a symmetrical 'V' or inverted 'V' shape, meaning $R \approx 1.0$. Artifacts are often highly asymmetrical.
*   **$D$ (Sharpness / Slope Sum)**: Calculated as $S_1 + |S_2|$. This represents the steepness and distinctness of the peak against the background noise.

### 2. Rule-Based Classification
Using the extracted features, a peak is classified as a True Spike if it passes three specific threshold gates:
1.  **Sharpness Threshold ($T_a$)**: The peak must be distinct enough ($D > T_a$).
2.  **Symmetry / Ratio Threshold ($T_s$)**: The rising and falling slopes must be relatively symmetrical ($|1.0 - R| < T_s$).
3.  **Steepness Threshold ($T_w$)**: The initial rise must be fast enough ($S_1 > T_w$).

*If a candidate peak passes all three rules, it is classified as a `1` (Spike); otherwise, it is a `0` (Artifact).*

### 3. Particle Swarm Optimization (PSO)
The challenge with rule-based models is determining the exact values for $T_a$, $T_s$, and $T_w$. Because different datasets, subjects, and recording setups produce distinct noise profiles and amplitude scales, hardcoded thresholds will fail.

To solve this, we use **Particle Swarm Optimization**, an ML technique inspired by the foraging behavior of birds or fish:
*   **Initialization**: We spawn a "swarm" of particles. Each particle represents a random set of thresholds $[T_a, T_s, T_w]$.
*   **Evaluation**: Every particle attempts to classify the peaks you manually labeled. Particles get a "score" based on their accuracy against your human labels.
*   **Updating**: Particles move through the mathematical parameter space, adjusting their threshold values by moving towards their own personal best historic score, as well as the global best score found by the entire swarm.
*   **Convergence**: After several iterations, the swarm converges on the optimal combination of $T_a, T_s$, and $T_w$ that perfectly (or maximally) separates your labeled Spikes from Artifacts.

### Why this approach?
Compared to Deep Learning (like CNNs or LSTMs), the Dingle + PSO approach is highly interpretable, requires drastically less compute power, and can be accurately tuned with a very small number of human labels (few-shot learning), making it ideal for rapid desktop processing.

---

## Project Structure

*   `app.py`: The main GUI application (CustomTkinter) handling the Views and Controllers.
*   `engine.py`: Contains the `DataLoader` for file parsing and the `DingleModel` for mathematical feature extraction ($S_1, S_2, R, D$).
*   `optimizer.py`: Contains the `PSOOptimizer` class that tunes the classification thresholds based on labeled data.
*   `requirements.txt`: Python package dependencies.
