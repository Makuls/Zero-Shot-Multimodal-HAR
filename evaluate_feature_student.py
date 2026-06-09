import torch
import cv2
import numpy as np
import os
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt
from torchvision import transforms

# ---> IMPORT THE NEW FEATURE STUDENT ARCHITECTURE <---
from train_feature_distillation import FeatureStudentNetwork

CLASS_NAMES = {
    4: "Throwing", 5: "Cross Arms", 21: "Jogging", 22: "Walking", 26: "Jumping"
}

def apply_lowpass_filter(data, cutoff=15, fs=50, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)

def load_mmact_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // 16, 1)
    transform = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((224, 224)),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    for i in range(16):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if ret: frames.append(transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        else: frames.append(torch.zeros(3, 224, 224))
    cap.release()
    return torch.stack(frames).mean(dim=0).unsqueeze(0) 

def load_mmact_imu_split(acc_path, gyro_path):
    try:
        df_acc = pd.read_csv(acc_path).select_dtypes(include=[np.number])
        df_gyro = pd.read_csv(gyro_path).select_dtypes(include=[np.number])
        min_len = min(len(df_acc), len(df_gyro))
        sensor_data = np.hstack((df_acc.iloc[:min_len, :3].values, df_gyro.iloc[:min_len, :3].values))
        
        if len(sensor_data) > 5:
            sensor_data = apply_lowpass_filter(sensor_data)
            time_old = np.linspace(0, 1, len(sensor_data))
            time_new = np.linspace(0, 1, 120)
            resampled = np.zeros((120, 6))
            for c in range(6): resampled[:, c] = np.interp(time_new, time_old, sensor_data[:, c])
            sensor_data = resampled
        else:
            sensor_data = np.zeros((120, 6))
    except Exception:
        sensor_data = np.zeros((120, 6))
        
    tensor_imu = torch.tensor(sensor_data, dtype=torch.float32).unsqueeze(0)
    mean = tensor_imu.mean(dim=1, keepdim=True)
    std = tensor_imu.std(dim=1, keepdim=True) + 1e-8
    return (tensor_imu - mean) / std

def discover_mmact_pairs(root_dir):
    root_path = Path(root_dir)
    vid_dir = root_path / "video"
    acc_dir = root_path / "acc2_clip"
    gyro_dir = root_path / "gyro_clip"
    
    videos = list(vid_dir.rglob("*.mp4")) + list(vid_dir.rglob("*.avi"))
    pairs = []
    
    for vid_path in videos:
        normalized_stem = vid_path.stem.replace('_', '.').replace('-', '.')
        parts = normalized_stem.split('.')
        subject_id, action_name = None, None
        
        raw_parts = vid_path.stem.split('.')
        if len(raw_parts) >= 3:
            subject_id, action_name = raw_parts[1], raw_parts[2]

        if not subject_id or not action_name: continue

        acc = list(acc_dir.rglob(f"*.{subject_id}.{action_name}.csv"))
        gyro = list(gyro_dir.rglob(f"*.{subject_id}.{action_name}.csv"))
        
        if acc and gyro:
            pairs.append({'video': str(vid_path), 'acc': str(acc[0]), 'gyro': str(gyro[0])})
    return pairs

def evaluate_feature_zero_shot():
    print("--- Phase 19: Feature-Distilled Zero-Shot Evaluation ---")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    raw_dir = Path("data/raw")
    acc_folders = list(raw_dir.rglob("acc2_clip"))
    if not acc_folders: return
    MMACT_ROOT = acc_folders[0].parent
    
    paired_files = discover_mmact_pairs(MMACT_ROOT)
    print(f"✅ Executing forward pass on {len(paired_files)} trials...\n")
    
    print("Loading the Feature-Matched Edge Model...")
    # Make sure teacher_feature_dim matches what we trained with (512)
    model = FeatureStudentNetwork(num_classes=27, teacher_feature_dim=512).to(device)
    model.load_state_dict(torch.load("saved_models/feature_student_model.pth", map_location=device))
    model.eval()
    
    results_log = []
    
    for i, data_dict in enumerate(paired_files):
        file_name = os.path.basename(data_dict['video'])
        true_action = file_name.split('.')[2] 
        
        # print(f"[{i+1}/{len(paired_files)}] Testing {file_name}...")
        
        rgb_tensor = load_mmact_video(data_dict['video']).to(device)
        imu_tensor = load_mmact_imu_split(data_dict['acc'], data_dict['gyro']).to(device)
        
        with torch.no_grad():
            # We don't need return_features=True here because we only care about the final prediction
            output = model(imu_tensor, rgb_tensor)
            
        predicted_idx = output.argmax(dim=1).item()
        predicted_name = CLASS_NAMES.get(predicted_idx, f"Class {predicted_idx}")
        
        results_log.append({
            "File": file_name,
            "True_MMAct_Action": true_action,
            "Predicted_UTD_Action": predicted_name
        })
        
    df_results = pd.DataFrame(results_log)
    df_results.to_csv("mmact_feature_results.csv", index=False)
    print("\n🎉 FEATURE EVALUATION COMPLETE!")
    print("Results saved to 'mmact_feature_results.csv'.")

if __name__ == "__main__":
    evaluate_feature_zero_shot()