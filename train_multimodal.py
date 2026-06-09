import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np

# Import our architecture and the real dataset
from src.models.multimodal_fusion import MultimodalFusionNetwork
from src.datasets.multimodal_dataset import UTDMHADDataset

def train_multimodal_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for imu, rgb, labels in dataloader:
        imu, rgb, labels = imu.to(device), rgb.to(device), labels.to(device)
        
        # ---> CRITICAL: Apply IMU Normalization (Matches SSL Phase) <---
        mean = imu.mean(dim=1, keepdim=True)
        std = imu.std(dim=1, keepdim=True) + 1e-6
        imu = (imu - mean) / std
        
        optimizer.zero_grad()
        activity_logits = model(imu, rgb)
        
        loss = criterion(activity_logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = activity_logits.max(dim=1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    return total_loss / len(dataloader), 100.0 * correct / total

def validate_multimodal(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imu, rgb, labels in dataloader:
            imu, rgb, labels = imu.to(device), rgb.to(device), labels.to(device)
            
            # ---> CRITICAL: Apply IMU Normalization (Matches SSL Phase) <---
            mean = imu.mean(dim=1, keepdim=True)
            std = imu.std(dim=1, keepdim=True) + 1e-6
            imu = (imu - mean) / std
            
            activity_logits = model(imu, rgb)
            loss = criterion(activity_logits, labels)
            
            total_loss += loss.item()
            _, predicted = activity_logits.max(dim=1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return total_loss / len(dataloader), 100.0 * correct / total

if __name__ == "__main__":
    print("--- Phase 10: Real Multimodal Fusion (Sensor + Vision) ---")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Compute Device: {device}")
    
    # 1. Load the Real UTD-MHAD Dataset
    print("Loading UTD-MHAD dataset...")
    dataset = UTDMHADDataset(root_dir="data/raw/UTD-MHAD/UTD-MHAD/")
    
    # 2. Split into Train/Val
    # ---> THE DATA LEAKAGE FIX: Lock the random seed here too! <---
    torch.manual_seed(42)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # 3. Initialize Model (UTD-MHAD has 27 action classes)
    model = MultimodalFusionNetwork(num_classes=27).to(device)
    
    # ---> NEW: INJECT THE SSL FOUNDATION INTO THE SENSOR BACKBONE <---
    print("Loading pre-trained SSL Foundation into Sensor Backbone...")
    try:
        # Load the SSL weights you just trained
        foundation_weights = torch.load("saved_models/ssl_foundation.pth", map_location=device, weights_only=True)
        # Inject specifically into the sensor_backbone, leaving the vision backbone untouched
        model.sensor_backbone.load_state_dict(foundation_weights, strict=False)
        print("✅ SSL Weights loaded successfully! The IMU branch is now pre-trained.")
    except Exception as e:
        print(f"⚠️ Warning: Could not load SSL weights. Error: {e}")
    # -----------------------------------------------------------------
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # 4. Training Loop
    epochs = 10
    print(f"\nStarting {epochs}-Epoch Multimodal Training...")
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        train_loss, train_acc = train_multimodal_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate_multimodal(model, val_loader, criterion, device)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        # Added a checkpointing save block so you always keep the highest performing model!
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "saved_models/best_multimodal_model.pth")
            print(f"  -> New Best Multimodal Model Saved! ({val_acc:.2f}%)")
        
    print("\nPhase 10 Complete: You have successfully fused Vision and Inertial representations!")