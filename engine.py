import numpy as np
import scipy.io
import json
import os

class DataLoader:
    @staticmethod
    def load_openephys(file_path):
        """
        Loads OpenEphys data.
        Assumes file_path is the path to a continuous.dat file from an OpenEphys recording,
        or a generic binary dump if directly passed.
        For full .oebin support, use the open-ephys-python-tools.
        """
        if file_path.endswith('.dat'):
            # OpenEphys continuous.dat usually stores int16
            data = np.fromfile(file_path, dtype=np.int16)
            # Typically need to scale by bitVolts from structure.oebin
            # Here we just return the raw or basic scaled array.
            return data.astype(np.float32)
        elif file_path.endswith('.npy'):
            return np.load(file_path)
        else:
            raise NotImplementedError("Only .dat and .npy are natively supported for OpenEphys stubs without `open-ephys-python-tools`.")

    @staticmethod
    def load_photometry(file_path):
        """
        Loads Photometry data (.ppd or .csv).
        .ppd is typically Doric's format. Without Doric's h5py/bespoke reader, 
        we will provide a stub that loads standard numpy arrays or CSVs.
        """
        if file_path.endswith('.csv'):
            return np.loadtxt(file_path, delimiter=',')
        elif file_path.endswith('.npy'):
            return np.load(file_path)
        else:
            # Stub for .ppd if it's a raw binary of floats
            try:
                return np.fromfile(file_path, dtype=np.float64)
            except Exception as e:
                raise NotImplementedError(f"Failed to parse .ppd file automatically. Please provide a CSV or install specific Doric readers: {e}")

class DingleModel:
    @staticmethod
    def calculate_slopes(snippet, sampling_rate=1000.0):
        """
        Calculates S1 (leading slope) and S2 (trailing slope) for a peak in the center of the snippet.
        snippet: 1D numpy array of the signal window.
        """
        n = len(snippet)
        if n < 3:
            return 0.0, 0.0
        
        peak_idx = np.argmax(snippet)
        
        # If the peak is at the very edge, we can't properly calculate slopes
        if peak_idx == 0 or peak_idx == n - 1:
            return 0.0, 0.0

        # Calculate derivative (differences)
        diffs = np.diff(snippet) * sampling_rate # scale to units per second
        
        # Leading slope (S1) is the maximum positive slope before the peak
        # Trailing slope (S2) is the maximum negative slope after the peak
        s1 = np.max(diffs[:peak_idx]) if len(diffs[:peak_idx]) > 0 else 0.0
        s2 = np.min(diffs[peak_idx:]) if len(diffs[peak_idx:]) > 0 else 0.0
        
        return s1, s2

    @staticmethod
    def extract_features(snippet, sampling_rate=1000.0):
        """
        Extracts S1, S2, R, and D from a snippet.
        R = S1 / |S2|
        D = S1 + |S2|
        """
        s1, s2 = DingleModel.calculate_slopes(snippet, sampling_rate)
        
        s2_abs = abs(s2)
        
        # Calculate R
        if s2_abs > 0:
            r = s1 / s2_abs
        else:
            r = 0.0 # Or np.inf if we wanted to be strictly mathematical, but 0 is safer for ML
            
        # Calculate D
        d = s1 + s2_abs
        
        return {
            'S1': s1,
            'S2': s2,
            'R': r,
            'D': d
        }

    @staticmethod
    def classify(features, thresholds):
        """
        Classifies a peak as a True Spike (1) or Artifact (0) based on thresholds.
        thresholds = {'Ta': val, 'Ts': val, 'Tw': val} for checking specific rules.
        A generic threshold rule: D > Ta and R > Ts and S1 > Tw (as an example mapping)
        We will formulate based on standard Dingle logic:
        e.g., S1 & S2 must be steep enough, Ratio must be balanced, Sharpness high.
        """
        ta = thresholds.get('Ta', 0.0)
        ts = thresholds.get('Ts', 0.0)
        tw = thresholds.get('Tw', 0.0)
        
        s1 = features['S1']
        r = features['R']
        d = features['D']
        
        # Dingle example logic:
        # Sharpness > Ta
        # Ratio roughly around 1.0, bounded by Ts
        # S1 steepness > Tw
        is_spike = (d > ta) and (abs(1.0 - r) < ts) and (s1 > tw)
        return int(is_spike)

def get_candidate_peaks(signal, threshold, min_distance=10):
    """
    Simple peak detection for candidate selection.
    min_distance defines the minimum indices between peaks.
    """
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(signal, height=threshold, distance=min_distance)
    return peaks
