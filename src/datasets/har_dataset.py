import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.preprocessing import StandardScaler
import os

class RealHARDataset(Dataset):
    """
    Ingests real CSV/TXT wearable data (WISDM), handles missing channels, 
    applies tri-axial normalization, and performs sliding window segmentation.
    """
    def __init__(self, csv_file_path, original_hz=20, target_hz=20, window_size_sec=6, overlap_pct=0.5):
        print(f"Loading real data from: {csv_file_path}")
        
        # 1. Parse the WISDM text file
        # Format: user, activity, timestamp, x, y, z;
        columns = ['user', 'activity', 'timestamp', 'acc_x', 'acc_y', 'acc_z']
        
        # We use on_bad_lines='skip' because the WISDM dataset has a few corrupted rows
        df = pd.read_csv(csv_file_path, header=None, names=columns, on_bad_lines='skip')
        
        # Clean up the trailing semicolon in the Z column and convert to float
        df['acc_z'] = df['acc_z'].astype(str).str.replace(';', '').astype(float)
        
        # Drop any rows with missing/NaN values
        df = df.dropna()
        
        # Extract the continuous sensor arrays
        raw_acc_data = df[['acc_x', 'acc_y', 'acc_z']].values
        
        # Pad with 3 columns of zeros to simulate Gyroscope (matches our 6-channel architecture)
        zero_pad = np.zeros_like(raw_acc_data)
        self.raw_data = np.concatenate((raw_acc_data, zero_pad), axis=1) # Shape: [N, 6]
        
        # Extract numerical labels for the 6 activities
        activity_mapping = {'Walking': 0, 'Jogging': 1, 'Stairs': 2, 'Sitting': 3, 'Standing': 4, 'LyingDown': 5}
        self.raw_labels = df['activity'].map(activity_mapping).fillna(0).values 
        
        # 2. Resample (WISDM is natively 20Hz, so this usually skips, but keeps it safe for HARTH later)
        if original_hz != target_hz:
            num_samples = int(len(self.raw_data) * (target_hz / original_hz))
            self.data = signal.resample(self.raw_data, num_samples)
            indices = np.linspace(0, len(self.raw_labels)-1, num_samples).astype(int)
            self.labels = self.raw_labels[indices]
        else:
            self.data = self.raw_data
            self.labels = self.raw_labels
            
        # 3. Tri-axial Normalization (Z-score scaling)
        self.scaler = StandardScaler()
        self.data = self.scaler.fit_transform(self.data)
        
        # 4. Sliding Window Segmentation
        self.window_length = int(target_hz * window_size_sec)
        self.step_size = int(self.window_length * (1 - overlap_pct))
        
        self.windows, self.window_labels = self._create_windows()
        print(f"Successfully generated {len(self.windows)} overlapping windows for training.")

    def _create_windows(self):
        windows = []
        labels = []
        for start in range(0, len(self.data) - self.window_length + 1, self.step_size):
            end = start + self.window_length
            windows.append(self.data[start:end])
            
            # Label the window based on the most frequent activity in that timeframe
            window_label = int(np.bincount(self.labels[start:end].astype(int)).argmax())
            labels.append(window_label)
            
        return np.array(windows), np.array(labels)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window_tensor = torch.tensor(self.windows[idx], dtype=torch.float32)
        label_tensor = torch.tensor(self.window_labels[idx], dtype=torch.long)
        return window_tensor, label_tensor # <-- Now returning both data and label

if __name__ == "__main__":
    # Point directly to the file shown in your VS Code sidebar
    test_file = "data/raw/wisdm-dataset/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt"
    if os.path.exists(test_file):
        dataset = RealHARDataset(csv_file_path=test_file)
        sample_x = dataset[0]
        print(f"Real Tensor Shape: {sample_x.shape} (Expected: [120, 6])")
    else:
        print("File not found! Check your folder paths.")