import os
import glob
import scipy.io as sio
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import torchvision.transforms as T # Updated to T for shorter syntax

class UTDMHADDataset(Dataset):
    """
    Pairs UTD-MHAD Wearable Sensor data (.mat) with corresponding Video data (.avi).
    Uses recursive Deep Search to find files and robust string splitting to pair them.
    """
    # ---> NEW: Added is_training flag <---
    def __init__(self, root_dir, seq_len=120, img_size=224, is_training=False):
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.img_size = img_size
        self.is_training = is_training
        
        # ---> NEW: Vision Scrambler (Random Erasing & Color Jitter) <---
        if self.is_training:
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize((img_size, img_size)),
                # Aggressively change lighting to simulate different environments
                T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
                T.RandomGrayscale(p=0.2), 
                T.ToTensor(),
                # Black out random chunks of the video so it can't memorize the wall
                T.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0), 
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            print("🛡️ [Domain-Blind Mode ACTIVATED] Vision Scrambler is online.")
        else:
            # Standard Evaluation/Zero-Shot Transform (No Augmentation)
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        print(f"Scanning deep inside: {self.root_dir}")
        self.samples = self._pair_files()
        print(f"Successfully paired {len(self.samples)} multimodal trials!")

    def _pair_files(self):
        paired_data = []
        
        # DEEP SEARCH: Hunt through all subdirectories for .avi and .mat files
        all_videos = glob.glob(os.path.join(self.root_dir, "**", "*.avi"), recursive=True)
        all_inertial = glob.glob(os.path.join(self.root_dir, "**", "*.mat"), recursive=True)
        
        print(f"  -> Found {len(all_videos)} raw Video files.")
        print(f"  -> Found {len(all_inertial)} raw Inertial files.")

        # Build a dictionary of videos based ONLY on the aX_sY_tZ prefix
        video_dict = {}
        for v_path in all_videos:
            filename = os.path.basename(v_path).lower()
            base_name = "_".join(filename.split('_')[:3]) 
            video_dict[base_name] = v_path

        # Scan inertial files and match them
        for i_path in all_inertial:
            filename = os.path.basename(i_path).lower()
            base_name = "_".join(filename.split('_')[:3])
            
            if base_name in video_dict:
                # Extract the action class (a1 -> label 0)
                action_str = base_name.split('_')[0].replace('a', '')
                label = int(action_str) - 1 
                
                paired_data.append({
                    "inertial_path": i_path,
                    "video_path": video_dict[base_name],
                    "label": label
                })
                
        return paired_data

    # ---> NEW: The IMU Scrambler <---
    def augment_imu(self, sensor_data):
        """ Inject physical noise to simulate different smartwatch hardware """
        # 1. Inject Gaussian Noise (simulates sensor drift/static)
        noise = np.random.normal(0, 0.05, sensor_data.shape)
        sensor_data = sensor_data + noise
        
        # 2. Random Scaling (simulates different user arm lengths / force profiles)
        scale = np.random.normal(1.0, 0.1)
        sensor_data = sensor_data * scale
        
        return sensor_data

    def _process_inertial(self, path):
        # UTD-MHAD .mat files store data under the 'd_iner' key
        mat_data = sio.loadmat(path)
        imu_data = mat_data['d_iner']
        
        # Fixed sequence length (120) for our Transformer
        if len(imu_data) > self.seq_len:
            imu_data = imu_data[:self.seq_len, :]
        elif len(imu_data) < self.seq_len:
            pad_len = self.seq_len - len(imu_data)
            imu_data = np.pad(imu_data, ((0, pad_len), (0, 0)), mode='constant')
            
        # ---> NEW: Apply IMU Augmentation during training <---
        if self.is_training:
            imu_data = self.augment_imu(imu_data)
            
        return torch.tensor(imu_data, dtype=torch.float32)

    def _process_video(self, path):
        # Extract the middle frame of the video
        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames // 2))
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            frame = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            
        # Convert BGR (OpenCV) to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.transform(frame) # This now dynamically uses the Scrambler if is_training=True

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        imu_tensor = self._process_inertial(sample["inertial_path"])
        rgb_tensor = self._process_video(sample["video_path"])
        label = torch.tensor(sample["label"], dtype=torch.long)
        
        return imu_tensor, rgb_tensor, label