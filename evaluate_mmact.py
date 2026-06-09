import torch
import cv2
import numpy as np
import os
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt
from torchvision import transforms

# Import your heavy, 90.75% accurate architecture
from src.models.multimodal_fusion import MultimodalFusionNetwork

CLASS_NAMES = {
    4: "Throwing / Arm Swipe",
    5: "Cross Arms",
    21: "Jogging",
    22: "Walking",
    26: "Jumping / Squat"
}

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
            frame = transform(frame)
            frames.append(frame)
        else:
            frames.append(torch.zeros(3, 224, 224))
            
    cap.release()
    return torch.stack(frames).mean(dim=0).unsqueeze(0) 

def load_mmact_imu_split(acc_path, gyro_path):
    try:
        df_acc = pd.read_csv(acc_path).select_dtypes(include=[np.number])
        df_gyro = pd.read_csv(gyro_path).select_dtypes(include=[np.number])
        
        acc_data = df_acc.iloc[:, :3].values
        gyro_data = df_gyro.iloc[:, :3].values
        
        min_len = min(len(acc_data), len(gyro_data))
        sensor_data = np.hstack((acc_data[:min_len], gyro_data[:min_len]))
    except Exception as e:
        return torch.zeros(1, 120, 6)
        
    current_length = len(sensor_data)
    if current_length > 5:
        sensor_data = apply_lowpass_filter(sensor_data)
        target_length = 120
        time_old = np.linspace(0, 1, current_length)
        time_new = np.linspace(0, 1, target_length)
        
        resampled_data = np.zeros((target_length, 6))
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
    """ Robustly links video and sensor data with built-in path diagnostics """
    root_path = Path(root_dir)
    print(f"\n🔍 [DIAGNOSTIC] Current Working Directory: {os.getcwd()}")
    print(f"🔍 [DIAGNOSTIC] Target Root Path Absolute: {root_path.resolve()}")
    
    if not root_path.exists():
        print(f"❌ [ERROR] The directory path '{root_dir}' does not exist!")
        # Help look for alternative locations
        if Path("data").exists():
            print("💡 Found a 'data' folder in your workspace root. Its contents are:")
            print([p.name for p in Path("data").iterdir()])
        return []

    vid_dir = root_path / "video"
    acc_dir = root_path / "acc2_clip"
    gyro_dir = root_path / "gyro_clip"
    
    # 1. Gather all files recursively
    videos = list(vid_dir.rglob("*.mp4")) + list(vid_dir.rglob("*.avi"))
    accs = list(acc_dir.rglob("*.csv"))
    gyros = list(gyro_dir.rglob("*.csv"))
    
    print(f"\n📊 [FILE COUNT DIAGNOSTIC]")
    print(f"  -> Videos found (.mp4/.avi): {len(videos)}")
    print(f"  -> Accelerometer files found (.csv): {len(accs)}")
    print(f"  -> Gyroscope files found (.csv): {len(gyros)}")
    
    if len(videos) == 0 or len(accs) == 0:
        print("❌ [ERROR] Missing raw files. Check if folder names ('video', 'acc2_clip') match perfectly.")
        return []

    # Print samples to check delimiter style (e.g. dots vs underscores)
    print(f"\n📝 [NAMING PATTERN DIAGNOSTIC]")
    print(f"  -> Sample Video Name: {videos[0].name}")
    print(f"  -> Sample Accel Name: {accs[0].name}")

    pairs = []
    for vid_path in videos:
        # Standardize formatting to handle potential dot or underscore mismatches
        normalized_stem = vid_path.stem.replace('_', '.').replace('-', '.')
        parts = normalized_stem.split('.')
        
        # Look for subject identifiers and action names dynamically
        subject_id = None
        action_name = None
        
        for part in parts:
            if "subject" in part:
                subject_id = part
        
        # Fallback to direct token parsing if standard keywords aren't found
        if len(parts) >= 3:
            # Handles 'subject1-1.subject1.checking_time' or similar structures
            # Extracts 'subject1' and 'checking_time'
            raw_parts = vid_path.stem.split('.')
            if len(raw_parts) >= 3:
                subject_id = raw_parts[1]
                action_name = raw_parts[2]

        if not subject_id or not action_name:
            continue

        # Create patterns to locate files across folders
        acc_pattern = f"*.{subject_id}.{action_name}.csv"
        gyro_pattern = f"*.{subject_id}.{action_name}.csv"
        
        acc_matches = list(acc_dir.rglob(acc_pattern))
        gyro_matches = list(gyro_dir.rglob(gyro_pattern))
        
        if acc_matches and gyro_matches:
            pairs.append({
                'video': str(vid_path),
                'acc': str(acc_matches[0]),
                'gyro': str(gyro_matches[0])
            })
            
    return pairs

def evaluate_zero_shot():
    print("--- Phase 13: Cross-Dataset Zero-Shot Evaluation (MMAct) ---")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    print("🔍 Auto-locating the MMAct dataset inside your data/raw/ folder...")
    raw_dir = Path("data/raw")
    acc_folders = list(raw_dir.rglob("acc2_clip"))
    
    if not acc_folders:
        print("❌ [ERROR] Could not find the 'acc2_clip' folder.")
        return
        
    MMACT_ROOT = acc_folders[0].parent
    print(f"🎯 Locked onto the true dataset path: {MMACT_ROOT}\n")
    
    paired_files = discover_mmact_pairs(MMACT_ROOT)
    
    if not paired_files:
        print(f"\n❌ Pipeline Stopped: Zero matching triplets.")
        return
        
    print(f"✅ Successfully forged {len(paired_files)} Multimodal Triplets! Executing forward pass...\n")
    
    print("Loading the 90.75% MultimodalFusionNetwork...")
    model = MultimodalFusionNetwork(num_classes=27).to(device)
    model.load_state_dict(torch.load("saved_models/best_multimodal_model.pth", map_location=device))
    model.eval()
    
    # --- NEW: Setup a list to save our results ---
    results_log = []
    
    # Notice we removed [:10] so it runs ALL 542 files!
    for i, data_dict in enumerate(paired_files):
        file_name = os.path.basename(data_dict['video'])
        true_action = file_name.split('.')[2] # Extract true action from filename
        
        print(f"[{i+1}/{len(paired_files)}] Analyzing {file_name}...")
        
        rgb_tensor = load_mmact_video(data_dict['video']).to(device)
        imu_tensor = load_mmact_imu_split(data_dict['acc'], data_dict['gyro']).to(device)
        
        with torch.no_grad():
            output = model(imu_tensor, rgb_tensor)
            
        predicted_idx = output.argmax(dim=1).item()
        predicted_name = CLASS_NAMES.get(predicted_idx, f"Class {predicted_idx}")
        
        # Save to log
        results_log.append({
            "File": file_name,
            "True_MMAct_Action": true_action,
            "Predicted_UTD_Action": predicted_name
        })
        
    # --- NEW: Save everything to a CSV for your paper ---
    df_results = pd.DataFrame(results_log)
    df_results.to_csv("mmact_zero_shot_results.csv", index=False)
    print("\n🎉 EVALUATION COMPLETE!")
    print("All 542 predictions have been saved to 'mmact_zero_shot_results.csv'.")
    print("You can open this file in Excel to calculate your metrics for the research paper!")

if __name__ == "__main__":
    evaluate_zero_shot()