import torch
import cv2
import numpy as np
import os
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt
from torchvision import transforms

# Import your Tiny Student Model
from train_multimodal_distillation import TinyStudentNetwork 

def apply_lowpass_filter(data, cutoff=15, fs=50, order=3):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)

def load_mmact_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // 16, 1)
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    for i in range(16):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(transform(frame))
        else:
            frames.append(torch.zeros(3, 224, 224))
    cap.release()
    return torch.stack(frames).mean(dim=0).unsqueeze(0) 

def load_mmact_imu_split(acc_path, gyro_path):
    try:
        df_acc = pd.read_csv(acc_path).select_dtypes(include=[np.number])
        df_gyro = pd.read_csv(gyro_path).select_dtypes(include=[np.number])
        min_len = min(len(df_acc), len(df_gyro))
        sensor_data = np.hstack((df_acc.iloc[:min_len, :3].values, df_gyro.iloc[:min_len, :3].values))
    except Exception:
        return torch.zeros(1, 120, 6)
        
    if len(sensor_data) > 5:
        sensor_data = apply_lowpass_filter(sensor_data)
        time_old = np.linspace(0, 1, len(sensor_data))
        time_new = np.linspace(0, 1, 120)
        resampled_data = np.zeros((120, 6))
        for channel in range(6):
            resampled_data[:, channel] = np.interp(time_new, time_old, sensor_data[:, channel])
        sensor_data = resampled_data
    else:
        sensor_data = np.zeros((120, 6))
        
    tensor_imu = torch.tensor(sensor_data, dtype=torch.float32).unsqueeze(0)
    mean = tensor_imu.mean(dim=1, keepdim=True)
    std = tensor_imu.std(dim=1, keepdim=True) + 1e-8
    return (tensor_imu - mean) / std

def discover_mmact_pairs(root_dir):
    root_path = Path(root_dir)
    videos = list((root_path / "video").rglob("*.mp4"))
    acc_dir = root_path / "acc2_clip"
    gyro_dir = root_path / "gyro_clip"
    
    pairs = []
    for vid_path in videos:
        parts = vid_path.stem.split('.')
        if len(parts) >= 3:
            subject_id, action_name = parts[1], parts[2]
            acc_matches = list(acc_dir.rglob(f"*.{subject_id}.{action_name}.csv"))
            gyro_matches = list(gyro_dir.rglob(f"*.{subject_id}.{action_name}.csv"))
            if acc_matches and gyro_matches:
                pairs.append({'video': str(vid_path), 'acc': str(acc_matches[0]), 'gyro': str(gyro_matches[0])})
    return pairs

def evaluate_ablation():
    print("--- Phase 15: Student Network Diagnostic Ablation Study ---")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    raw_dir = Path("data/raw")
    acc_folders = list(raw_dir.rglob("acc2_clip"))
    if not acc_folders:
        print("❌ Could not find dataset.")
        return
    MMACT_ROOT = acc_folders[0].parent
    
    paired_files = discover_mmact_pairs(MMACT_ROOT)
    
    model = TinyStudentNetwork(num_classes=27).to(device)
    model.load_state_dict(torch.load("saved_models/tiny_student_model.pth", map_location=device))
    model.eval()
    
    print("\nRunning Diagnostics on the first 15 files...")
    print(f"{'File':<40} | {'Normal Fusion':<15} | {'IMU Only':<15} | {'Video Only':<15}")
    print("-" * 95)
    
    for data_dict in paired_files[:15]:
        file_name = os.path.basename(data_dict['video'])
        
        # 1. Load normal data
        rgb_tensor = load_mmact_video(data_dict['video']).to(device)
        imu_tensor = load_mmact_imu_split(data_dict['acc'], data_dict['gyro']).to(device)
        
        # 2. Create BLANK data to isolate modalities
        blank_rgb = torch.zeros_like(rgb_tensor)
        blank_imu = torch.zeros_like(imu_tensor)
        
        with torch.no_grad():
            # Test 1: Both modalities
            out_fusion = model(imu_tensor, rgb_tensor)
            # Test 2: IMU Only (Vision is blinded)
            out_imu = model(imu_tensor, blank_rgb)
            # Test 3: Vision Only (IMU is silenced)
            out_rgb = model(blank_imu, rgb_tensor)
            
        pred_fusion = out_fusion.argmax(dim=1).item()
        pred_imu = out_imu.argmax(dim=1).item()
        pred_rgb = out_rgb.argmax(dim=1).item()
        
        # Format for terminal output
        name_short = file_name[:38]
        print(f"{name_short:<40} | Class {pred_fusion:<9} | Class {pred_imu:<9} | Class {pred_rgb:<9}")

if __name__ == "__main__":
    evaluate_ablation()