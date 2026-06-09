import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.models as models
import os

# Import your heavy, 90.75% accurate architecture and dataset
from src.models.multimodal_fusion import MultimodalFusionNetwork
from src.datasets.multimodal_dataset import UTDMHADDataset

# --- 1. Define the Tiny Student Network (For Smartwatches / Edge Devices) ---
class TinyStudentNetwork(nn.Module):
    def __init__(self, num_classes=27):
        super().__init__()
        # Tiny IMU Branch (Lightweight 1D CNN instead of a deep Transformer)
        self.imu_branch = nn.Sequential(
            nn.Conv1d(6, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
# --- 1. Define the Tiny Student Network (For Smartwatches / Edge Devices) ---
class TinyStudentNetwork(nn.Module):
    def __init__(self, num_classes=27):
        super().__init__()
        # Tiny IMU Branch (Lightweight 1D CNN instead of a deep Transformer)
        self.imu_branch = nn.Sequential(
            nn.Conv1d(6, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        # Tiny Vision Branch (MobileNetV2 instead of heavy ResNet50)
        mobilenet = models.mobilenet_v2(weights=None)
        self.vision_branch = nn.Sequential(
            mobilenet.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1280, 64)
        )
        
        # Tiny Fusion Head
        self.fusion_head = nn.Sequential(
            nn.Linear(64 + 64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, imu, rgb):
        # ---> THE FIX: Flip the shape from (Batch, Time, Channels) to (Batch, Channels, Time)
        imu = imu.transpose(1, 2)
        
        imu_features = self.imu_branch(imu)
        rgb_features = self.vision_branch(rgb)
        fused = torch.cat((imu_features, rgb_features), dim=1)
        return self.fusion_head(fused)
# --- 2. The Distillation Loss Function ---
def distillation_loss(student_logits, teacher_logits, labels, temperature=3.0, alpha=0.5):
    # Hard Loss: How well does the student match the actual ground truth?
    hard_loss = F.cross_entropy(student_logits, labels)
    
    # Soft Loss (KL Divergence): How well does the student mimic the Teacher's brain?
    soft_targets = F.softmax(teacher_logits / temperature, dim=1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    soft_loss = F.kl_div(student_log_probs, soft_targets, reduction='batchmean') * (temperature ** 2)
    
    return (alpha * hard_loss) + ((1.0 - alpha) * soft_loss)

def train_distillation():
    print("--- Phase 11: Multimodal Knowledge Distillation ---")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Compute Device: {device}")
    
    # 1. Dataset & Leakage-Proof Splitting
    torch.manual_seed(42)
    dataset = UTDMHADDataset(root_dir="data/raw/UTD-MHAD/UTD-MHAD/", is_training=True)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # 2. Load the Master Teacher
    print("Loading the 90.75% Teacher Network...")
    teacher = MultimodalFusionNetwork(num_classes=27).to(device)
    teacher.load_state_dict(torch.load("saved_models/best_multimodal_model.pth", map_location=device))
    teacher.eval() # Teacher NEVER trains here, it only tutors!
    
    # Freeze Teacher Weights to save Mac memory
    for param in teacher.parameters():
        param.requires_grad = False
    
    # 3. Initialize the Tiny Student
    print("Initializing the Tiny Student Network...")
    student = TinyStudentNetwork(num_classes=27).to(device)
    optimizer = optim.AdamW(student.parameters(), lr=1e-3)
    
    # Compare Parameter Counts for the console output
    t_params = sum(p.numel() for p in teacher.parameters())
    s_params = sum(p.numel() for p in student.parameters())
    print(f"\n[Compression Stats]")
    print(f"Teacher Parameters: {t_params:,}")
    print(f"Student Parameters: {s_params:,}")
    print(f"Compression Ratio:  {t_params/s_params:.1f}x smaller!\n")
    
    # 4. Training Loop
    epochs = 10
    best_val_acc = 0.0
    
    print("Starting Knowledge Transfer...")
    for epoch in range(epochs):
        student.train()
        total_loss = 0.0
        
        for imu, rgb, labels in train_loader:
            imu, rgb, labels = imu.to(device), rgb.to(device), labels.to(device)
            
            # CRITICAL: IMU Normalization
            mean = imu.mean(dim=1, keepdim=True)
            std = imu.std(dim=1, keepdim=True) + 1e-6
            imu = (imu - mean) / std
            
            optimizer.zero_grad()
            
            # Teacher predicts (No gradients needed)
            with torch.no_grad():
                teacher_logits = teacher(imu, rgb)
                
            # Student predicts
            student_logits = student(imu, rgb)
            
            # Calculate KD Loss and Backpropagate on Student ONLY
            loss = distillation_loss(student_logits, teacher_logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Validation Phase (Testing the Student)
        student.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imu, rgb, labels in val_loader:
                imu, rgb, labels = imu.to(device), rgb.to(device), labels.to(device)
                
                # Match validation normalization
                mean = imu.mean(dim=1, keepdim=True)
                std = imu.std(dim=1, keepdim=True) + 1e-6
                imu = (imu - mean) / std
                
                outputs = student(imu, rgb)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        val_acc = 100.0 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] | Distillation Loss: {total_loss/len(train_loader):.4f} | Student Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("saved_models", exist_ok=True)
            torch.save(student.state_dict(), "saved_models/augmented_student_model.pth")
            print(f"  -> Saved Tiny Student! ({val_acc:.2f}%)")

if __name__ == "__main__":
    train_distillation()