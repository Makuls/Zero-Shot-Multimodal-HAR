import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.models as models
import os
import numpy as np

# Import your Dataset and Teacher
from src.datasets.multimodal_dataset import UTDMHADDataset
from src.models.multimodal_fusion import MultimodalFusionNetwork

class FeatureStudentNetwork(nn.Module):
    def __init__(self, num_classes=27, teacher_feature_dim=256): # Adjust teacher_feature_dim if needed
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
        
        # The Student's internal feature representation
        self.student_feature_dim = 128 # 64 (IMU) + 64 (Vision)
        
        # ---> NEW: The Feature Projection Layer <---
        # This mathematically translates the Student's brain waves to match the Teacher's brain waves
        self.feature_projector = nn.Linear(self.student_feature_dim, teacher_feature_dim)
        
        # Final Classifier
        self.fusion_head = nn.Sequential(
            nn.Linear(self.student_feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, imu, rgb, return_features=False):
        imu = imu.transpose(1, 2)
        imu_features = self.imu_branch(imu)
        rgb_features = self.vision_branch(rgb)
        
        fused = torch.cat((imu_features, rgb_features), dim=1)
        logits = self.fusion_head(fused)
        
        if return_features:
            projected_features = self.feature_projector(fused)
            return logits, projected_features
        return logits

def feature_distillation_loss(s_logits, t_logits, s_features, t_features, labels, temp=3.0, alpha=0.5, beta=10.0):
    # 1. Hard Loss (Ground Truth)
    hard_loss = F.cross_entropy(s_logits, labels)
    
    # 2. Soft Loss (Logit Matching)
    soft_targets = F.softmax(t_logits / temp, dim=1)
    s_log_probs = F.log_softmax(s_logits / temp, dim=1)
    soft_loss = F.kl_div(s_log_probs, soft_targets, reduction='batchmean') * (temp ** 2)
    
    # 3. ---> NEW: Feature MSE Loss (Brain Matching) <---
    feature_loss = F.mse_loss(s_features, t_features)
    
    # Total Loss = (Logits) + (Features)
    total_loss = (alpha * hard_loss) + ((1.0 - alpha) * soft_loss) + (beta * feature_loss)
    return total_loss, feature_loss

def train_feature_distillation():
    print("--- Phase 18: Intermediate Feature-Based Distillation ---")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Load UTD-MHAD Training Data
    dataset = UTDMHADDataset(root_dir="data/raw/UTD-MHAD/UTD-MHAD/")
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    
    # 2. Load the Master Teacher (Read-Only)
    print("Loading the Frozen Teacher Network...")
    teacher = MultimodalFusionNetwork(num_classes=27).to(device)
    teacher.load_state_dict(torch.load("saved_models/best_multimodal_model.pth", map_location=device))
    teacher.eval() 
    for param in teacher.parameters():
        param.requires_grad = False # <--- 100% Safe. No weights will change.
        
    # 3. Initialize the Feature Student
    # Note: If your Teacher outputs a different feature dimension (like 512), change it here!
    student = FeatureStudentNetwork(num_classes=27, teacher_feature_dim=512).to(device)
    optimizer = optim.AdamW(student.parameters(), lr=1e-3)
    
    epochs = 10
    print("Initiating Feature Matching...")
    
    for epoch in range(epochs):
        student.train()
        epoch_total_loss = 0.0
        epoch_feat_loss = 0.0
        
        for imu, rgb, labels in train_loader:
            imu, rgb, labels = imu.to(device), rgb.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Extract Teacher Features (No gradients)
            with torch.no_grad():
                t_logits, t_features = teacher(imu, rgb, return_features=True)
                
            # Extract Student Features
            s_logits, s_features = student(imu, rgb, return_features=True)
            
            # Calculate Combined Loss
            loss, f_loss = feature_distillation_loss(s_logits, t_logits, s_features, t_features, labels)
            
            loss.backward()
            optimizer.step()
            
            epoch_total_loss += loss.item()
            epoch_feat_loss += f_loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}] | Total KD Loss: {epoch_total_loss/len(train_loader):.4f} | Feature MSE Loss: {epoch_feat_loss/len(train_loader):.4f}")
        
    os.makedirs("saved_models", exist_ok=True)
    # Saved as a completely separate file!
    torch.save(student.state_dict(), "saved_models/feature_student_model.pth")
    print("\n✅ Feature-Distilled Student Model Saved Successfully!")

if __name__ == "__main__":
    train_feature_distillation()