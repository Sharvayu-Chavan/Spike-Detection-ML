import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
import csv
import os

from engine import DataLoader, DingleModel, get_candidate_peaks
from optimizer import PSOOptimizer

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class SpikeLabelingApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Spike Detection ML")
        self.geometry("1200x800")
        
        # State variables
        self.signal = None
        self.sampling_rate = 1000.0
        self.candidate_peaks = []
        self.current_idx = 0
        self.labels = [] # List of tuples (index, label_int, features_dict)
                         # label_int: 1 (Spike), 0 (Artifact), -1 (Unsure)
        
        # GUI Layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(0, weight=1)
        
        self.setup_sidebar()
        self.setup_main_area()
        
    def setup_sidebar(self):
        self.sidebar_frame = ctk.CTkFrame(self, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(8, weight=1)
        
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Spike ML", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        self.btn_load = ctk.CTkButton(self.sidebar_frame, text="Load Data", command=self.load_data)
        self.btn_load.grid(row=1, column=0, padx=20, pady=10)
        
        self.btn_extract = ctk.CTkButton(self.sidebar_frame, text="Extract Candidates", command=self.extract_candidates, state="disabled")
        self.btn_extract.grid(row=2, column=0, padx=20, pady=10)
        
        self.lbl_stats = ctk.CTkLabel(self.sidebar_frame, text="Candidates: 0\nLabeled: 0")
        self.lbl_stats.grid(row=3, column=0, padx=20, pady=10)
        
        self.btn_train = ctk.CTkButton(self.sidebar_frame, text="Run PSO Optimization", command=self.run_pso, state="disabled")
        self.btn_train.grid(row=4, column=0, padx=20, pady=10)
        
        self.btn_export = ctk.CTkButton(self.sidebar_frame, text="Export Settings & CSV", command=self.export_data, state="disabled")
        self.btn_export.grid(row=5, column=0, padx=20, pady=10)
        
    def setup_main_area(self):
        self.main_frame = ctk.CTkFrame(self, corner_radius=0)
        self.main_frame.grid(row=0, column=1, sticky="nsew")
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=5)
        self.main_frame.grid_rowconfigure(2, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        
        # Plot
        self.figure = plt.Figure(figsize=(6,4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, self.main_frame)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=20, pady=20)
        
        # Labels and controls
        self.lbl_features = ctk.CTkLabel(self.main_frame, text="Features: S1=0, S2=0, R=0, D=0", font=ctk.CTkFont(size=14))
        self.lbl_features.grid(row=0, column=0, padx=20, pady=10)
        
        # Buttons Frame
        self.btn_frame = ctk.CTkFrame(self.main_frame)
        self.btn_frame.grid(row=2, column=0, sticky="nsew", padx=20, pady=20)
        
        self.btn_spike = ctk.CTkButton(self.btn_frame, text="True Spike", fg_color="green", command=lambda: self.label_peak(1))
        self.btn_spike.grid(row=0, column=0, padx=10, pady=10)
        
        self.btn_artifact = ctk.CTkButton(self.btn_frame, text="Artifact", fg_color="red", command=lambda: self.label_peak(0))
        self.btn_artifact.grid(row=0, column=1, padx=10, pady=10)
        
        self.btn_unsure = ctk.CTkButton(self.btn_frame, text="Unsure", fg_color="gray", command=lambda: self.label_peak(-1))
        self.btn_unsure.grid(row=0, column=2, padx=10, pady=10)
        
        self.btn_prev = ctk.CTkButton(self.btn_frame, text="< Prev", command=self.prev_peak)
        self.btn_prev.grid(row=0, column=3, padx=10, pady=10)
        
        self.btn_next = ctk.CTkButton(self.btn_frame, text="Next >", command=self.next_peak)
        self.btn_next.grid(row=0, column=4, padx=10, pady=10)
        
        for btn in [self.btn_spike, self.btn_artifact, self.btn_unsure, self.btn_prev, self.btn_next]:
            btn.configure(state="disabled")
            
    def load_data(self):
        file_path = filedialog.askopenfilename(title="Select OpenEphys or Photometry Data", filetypes=[("Data Files", "*.dat *.npy *.csv *.ppd")])
        if file_path:
            try:
                # Basic inference of type by extension 
                if file_path.endswith('.dat'):
                    self.signal = DataLoader.load_openephys(file_path)
                elif file_path.endswith('.csv') or file_path.endswith('.npy') or file_path.endswith('.ppd'):
                    self.signal = DataLoader.load_photometry(file_path)
                    
                messagebox.showinfo("Success", f"Loaded data with length {len(self.signal)}")
                self.btn_extract.configure(state="normal")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load data: {e}")
                
    def extract_candidates(self):
        # We need a generic threshold to find some initial peaks
        threshold = np.mean(self.signal) + np.std(self.signal) # crude but effective starting point
        self.candidate_peaks = get_candidate_peaks(self.signal, threshold)
        
        self.labels = [] # Reset labels
        self.current_idx = 0
        
        self.lbl_stats.configure(text=f"Candidates: {len(self.candidate_peaks)}\nLabeled: 0")
        
        if len(self.candidate_peaks) > 0:
            for btn in [self.btn_spike, self.btn_artifact, self.btn_unsure, self.btn_prev, self.btn_next]:
                btn.configure(state="normal")
            self.btn_train.configure(state="normal")
            self.show_current_peak()
        else:
            messagebox.showinfo("Info", "No candidates found with current basic threshold.")

    def get_snippet(self, peak_idx, duration_ms=500):
        # Samples for 500ms segment = 500ms * (sampling_rate/1000)
        half_window = int((duration_ms / 2.0) * (self.sampling_rate / 1000.0))
        
        start = max(0, peak_idx - half_window)
        end = min(len(self.signal), peak_idx + half_window)
        
        # Center index relative to the snippet
        center_rel = peak_idx - start
        
        return self.signal[start:end], center_rel

    def show_current_peak(self):
        if self.current_idx < 0 or self.current_idx >= len(self.candidate_peaks):
            return
            
        peak_idx = self.candidate_peaks[self.current_idx]
        snippet, center_rel = self.get_snippet(peak_idx)
        
        features = DingleModel.extract_features(snippet, self.sampling_rate)
        
        self.lbl_features.configure(text=f"Peak {self.current_idx+1}/{len(self.candidate_peaks)} " + \
                                         f"- Features: S1={features['S1']:.2f}, S2={features['S2']:.2f}, R={features['R']:.2f}, D={features['D']:.2f}")
        
        self.ax.clear()
        self.ax.plot(snippet, label="Signal", color='blue')
        self.ax.plot(center_rel, snippet[center_rel], 'ro', label="Peak")
        self.ax.axvline(center_rel, color='r', linestyle='--')
        
        # Visual indicators for slopes
        self.ax.plot([center_rel-10, center_rel], [snippet[center_rel-10], snippet[center_rel]], 'g--', label="S1 Slope")
        self.ax.plot([center_rel, center_rel+10], [snippet[center_rel], snippet[center_rel+10]], 'm--', label="S2 Slope")
        
        self.ax.legend()
        self.canvas.draw()
        
    def label_peak(self, label_val):
        peak_idx = self.candidate_peaks[self.current_idx]
        snippet, _ = self.get_snippet(peak_idx)
        features = DingleModel.extract_features(snippet, self.sampling_rate)
        
        # Update labels list
        existing = [i for i, v in enumerate(self.labels) if v[0] == peak_idx]
        if existing:
            self.labels[existing[0]] = (peak_idx, label_val, features)
        else:
            self.labels.append((peak_idx, label_val, features))
            
        labeled_count = len([x for x in self.labels if x[1] in [0, 1]])
        self.lbl_stats.configure(text=f"Candidates: {len(self.candidate_peaks)}\nLabeled: {labeled_count}")
        
        self.next_peak()
        
    def next_peak(self):
        if self.current_idx < len(self.candidate_peaks) - 1:
            self.current_idx += 1
            self.show_current_peak()
            
    def prev_peak(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.show_current_peak()
            
    def run_pso(self):
        labeled_spikes = [item for item in self.labels if item[1] in [0, 1]]
        if len(labeled_spikes) < 2: # Reduce to 2 for easier testing
            messagebox.showwarning("Warning", "Label at least 2 peaks before training.")
            return
            
        features_list = [item[2] for item in labeled_spikes]
        y_labels = [item[1] for item in labeled_spikes]
        
        opt = PSOOptimizer(num_particles=20, num_iterations=30)
        self.best_params = opt.optimize(features_list, y_labels)
        
        messagebox.showinfo("PSO Completed", f"Tuned Thresholds:\nTa: {self.best_params['Ta']:.2f}\nTs: {self.best_params['Ts']:.2f}\nTw: {self.best_params['Tw']:.2f}")
        self.btn_export.configure(state="normal")
        
    def export_data(self):
        if not hasattr(self, 'best_params'):
            return
            
        file_path = filedialog.asksaveasfilename(title="Save CSV", defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if file_path:
            json_file = file_path.replace('.csv', '.json')
            
            with open(json_file, 'w') as f:
                json.dump(self.best_params, f, indent=4)
                
            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['PeakIndex', 'S1', 'S2', 'R', 'D', 'IsSpike'])
                
                # Evaluate ALL candidates based on tuned parameters
                for peak_idx in self.candidate_peaks:
                    snippet, _ = self.get_snippet(peak_idx)
                    fdict = DingleModel.extract_features(snippet, self.sampling_rate)
                    is_spike = DingleModel.classify(fdict, self.best_params)
                    writer.writerow([peak_idx, fdict['S1'], fdict['S2'], fdict['R'], fdict['D'], is_spike])
                    
            messagebox.showinfo("Export Successful", f"Saved config to {json_file}\nSaved candidates to {file_path}")

if __name__ == "__main__":
    app = SpikeLabelingApp()
    app.mainloop()
