import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
import torchvision.models as models
import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt
from torchvision import transforms

# Import your UTD-MHAD Dataset and Teacher Network
from src.datasets.multimodal_dataset import UTDMHADDataset
from src.models.multimodal_fusion import MultimodalFusionNetwork

# ---------------------------------------------------------
# 1. THE MATHEMATICAL CORE: GRADIENT REVERSAL LAYER
# ---------------------------------------------------------
class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

# ---------------------------------------------------------
# 2. THE UPGRADED STUDENT (With Domain Discriminator)
# ---------------------------------------------------------
class UDATinyStudentNetwork(nn.Module):
    def __init__(self, num_classes=27):
        super().__init__()
        
        # 1D-CNN IMU Branch
        self.imu_branch = nn.Sequential(
            nn.Conv1d(6, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        # MobileNetV2 Vision Branch
        mobilenet = models.mobilenet_v2(weights=None)
        self.vision_branch = nn.Sequential(
            mobilenet.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1280, 64)
        )
        
        # Action Classifier Head (Predicts 27 Classes)
        self.fusion_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
        # ---> NEW: Domain Discriminator Head (Predicts Source vs Target)
        self.domain_discriminator = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2) # 0 = UTD-MHAD, 1 = MMAct
        )

    def forward(self, imu, rgb, alpha=None):
        imu = imu.transpose(1, 2)
        imu_features = self.imu_branch(imu)
        rgb_features = self.vision_branch(rgb)
        
        # 128-Dimensional Latent Space
        fused = torch.cat((imu_features, rgb_features), dim=1)
        
        # Standard Action Prediction
        class_logits = self.fusion_head(fused)
        
        # If alpha is provided, we are in UDA Training Mode
        if alpha is not None:
            reversed_features = GradientReversalFn.apply(fused, alpha)
            domain_logits = self.domain_discriminator(reversed_features)
            return class_logits, domain_logits
            
        return class_logits

# ---------------------------------------------------------
# 3. UNLABELED MMACT DATA LOADER (No Labels Used!)
# ---------------------------------------------------------
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
    return torch.stack(frames).mean(dim=0) 

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
        
    tensor_imu = torch.tensor(sensor_data, dtype=torch.float32)
    mean = tensor_imu.mean(dim=0, keepdim=True)
    std = tensor_imu.std(dim=0, keepdim=True) + 1e-8
    return (tensor_imu - mean) / std

class UnlabeledMMActDataset(Dataset):
    def __init__(self, root_dir):
        root_path = Path(root_dir)
        
        # 1. Robustly auto-locate the MMAct folder using recursive search
        acc_folders = list(root_path.rglob("acc2_clip"))
        if not acc_folders:
            print("❌ [ERROR] Could not find the 'acc2_clip' folder.")
            self.pairs = []
            return
            
        self.MMACT_ROOT = acc_folders[0].parent
        vid_dir = self.MMACT_ROOT / "video"
        acc_dir = self.MMACT_ROOT / "acc2_clip"
        gyro_dir = self.MMACT_ROOT / "gyro_clip"
        
        videos = list(vid_dir.rglob("*.mp4")) + list(vid_dir.rglob("*.avi"))
        self.pairs = []
        
        # 2. Robust filename matching (handling dots and underscores)
        for vid_path in videos:
            normalized_stem = vid_path.stem.replace('_', '.').replace('-', '.')
            parts = normalized_stem.split('.')
            
            subject_id = None
            action_name = None
            
            # Fallback to direct token parsing 
            raw_parts = vid_path.stem.split('.')
            if len(raw_parts) >= 3:
                subject_id = raw_parts[1]
                action_name = raw_parts[2]

            if not subject_id or not action_name:
                continue

            acc_pattern = f"*.{subject_id}.{action_name}.csv"
            gyro_pattern = f"*.{subject_id}.{action_name}.csv"
            
            acc_matches = list(acc_dir.rglob(acc_pattern))
            gyro_matches = list(gyro_dir.rglob(gyro_pattern))
            
            if acc_matches and gyro_matches:
                self.pairs.append({
                    'video': str(vid_path),
                    'acc': str(acc_matches[0]),
                    'gyro': str(gyro_matches[0])
                })
        print(f"🎯 MMAct Domain Target Locked! Found {len(self.pairs)} unlabeled pairs for Adaptation.")
                    
    def __len__(self): return len(self.pairs)
    
    def __getitem__(self, idx):
        data = self.pairs[idx]
        return load_mmact_imu_split(data['acc'], data['gyro']), load_mmact_video(data['video'])

# ---------------------------------------------------------
# 4. THE UDA DISTILLATION TRAINING LOOP
# ---------------------------------------------------------
def distillation_loss(student_logits, teacher_logits, labels, temperature=3.0, alpha_kd=0.5):
    hard_loss = F.cross_entropy(student_logits, labels)
    soft_targets = F.softmax(teacher_logits / temperature, dim=1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    soft_loss = F.kl_div(student_log_probs, soft_targets, reduction='batchmean') * (temperature ** 2)
    return (alpha_kd * hard_loss) + ((1.0 - alpha_kd) * soft_loss)

def train_uda_distillation():
    print("--- Phase 16: Unsupervised Domain Adaptation (UDA) Distillation ---")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Source Dataloader (Labeled UTD-MHAD)
    source_dataset = UTDMHADDataset(root_dir="data/raw/UTD-MHAD/UTD-MHAD/")
    source_loader = DataLoader(source_dataset, batch_size=16, shuffle=True, drop_last=True)
    
    # 2. Target Dataloader (Unlabeled MMAct)
    target_dataset = UnlabeledMMActDataset("data/raw")
    target_loader = DataLoader(target_dataset, batch_size=16, shuffle=True, drop_last=True)
    
    if len(target_dataset) == 0:
        print("❌ Could not find MMAct dataset for adaptation.")
        return
        
    print(f"Loaded {len(source_dataset)} Source samples and {len(target_dataset)} Target samples.")
    
    # 3. Load Frozen Teacher
    teacher = MultimodalFusionNetwork(num_classes=27).to(device)
    teacher.load_state_dict(torch.load("saved_models/best_multimodal_model.pth", map_location=device))
    teacher.eval()
    for param in teacher.parameters(): param.requires_grad = False
        
    # 4. Initialize UDA Student
    student = UDATinyStudentNetwork(num_classes=27).to(device)
    optimizer = optim.AdamW(student.parameters(), lr=1e-3)
    domain_criterion = nn.CrossEntropyLoss()
    
    epochs = 10
    total_batches = min(len(source_loader), len(target_loader))
    
    print("Initiating Adversarial Domain Alignment...")
    for epoch in range(epochs):
        student.train()
        epoch_loss = 0.0
        target_iter = iter(target_loader)
        
        for batch_idx, (s_imu, s_rgb, s_labels) in enumerate(source_loader):
            if batch_idx >= total_batches: break
            
            # Get Target Data
            t_imu, t_rgb = next(target_iter)
            
            # Send to Device
            s_imu, s_rgb, s_labels = s_imu.to(device), s_rgb.to(device), s_labels.to(device)
            t_imu, t_rgb = t_imu.to(device), t_rgb.to(device)
            
            # Calculate GRL Alpha (Gradually increases adversarial pressure)
            p = float(batch_idx + epoch * total_batches) / (epochs * total_batches)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            
            optimizer.zero_grad()
            
            # --- SOURCE PASS (Knowledge Distillation) ---
            with torch.no_grad():
                t_logits = teacher(s_imu, s_rgb)
            s_class_logits, s_domain_logits = student(s_imu, s_rgb, alpha=alpha)
            
            kd_loss = distillation_loss(s_class_logits, t_logits, s_labels)
            
            # Source Domain Label = 0
            domain_label_source = torch.zeros(s_imu.size(0), dtype=torch.long).to(device)
            source_domain_loss = domain_criterion(s_domain_logits, domain_label_source)
            
            # --- TARGET PASS (Unsupervised Adaptation) ---
            _, t_domain_logits = student(t_imu, t_rgb, alpha=alpha)
            
            # Target Domain Label = 1
            domain_label_target = torch.ones(t_imu.size(0), dtype=torch.long).to(device)
            target_domain_loss = domain_criterion(t_domain_logits, domain_label_target)
            
            # Total Adversarial Loss (GRL flips the gradients implicitly!)
            loss = kd_loss + 0.5 * (source_domain_loss + target_domain_loss)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {epoch_loss/total_batches:.4f} | Adaptation Alpha: {alpha:.3f}")
        
    os.makedirs("saved_models", exist_ok=True)
    # We save this as the final, hardened edge model
    torch.save(student.state_dict(), "saved_models/uda_student_model.pth")
    print("✅ Domain-Adapted Student Model Saved!")

if __name__ == "__main__":
    train_uda_distillation()