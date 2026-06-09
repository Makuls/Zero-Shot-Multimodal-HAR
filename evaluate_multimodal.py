import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, random_split
import os

from src.models.multimodal_fusion import MultimodalFusionNetwork
from src.datasets.multimodal_dataset import UTDMHADDataset

def evaluate_model():
    print("--- Phase 12: Final Model Evaluation & Confusion Matrix ---")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Ensure the exact same Train/Val split as the training script
    torch.manual_seed(42)
    dataset = UTDMHADDataset(root_dir="data/raw/UTD-MHAD/UTD-MHAD/")
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_dataset = random_split(dataset, [train_size, val_size])
    
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # 2. Load the trained weights
    model = MultimodalFusionNetwork(num_classes=27).to(device)
    model_path = "saved_models/best_multimodal_model.pth"
    
    if not os.path.exists(model_path):
        print(f"Error: Could not find {model_path}. Did the training script save it?")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Successfully loaded optimal weights!")

    # 3. Run Inference on Unseen Data
    all_preds = []
    all_labels = []
    
    print(f"Running inference on {val_size} unseen validation samples...")
    with torch.no_grad():
        for imu, rgb, labels in val_loader:
            imu, rgb = imu.to(device), rgb.to(device)
            
            # ---> CRITICAL FIX: Apply IMU Normalization (Matches Training) <---
            mean = imu.mean(dim=1, keepdim=True)
            std = imu.std(dim=1, keepdim=True) + 1e-6
            imu = (imu - mean) / std
            # ------------------------------------------------------------------
            
            outputs = model(imu, rgb)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    # 4. Generate the Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Multimodal Validation Confusion Matrix", fontsize=16, pad=20)
    plt.ylabel("Actual Human Activity (Class ID)", fontsize=12)
    plt.xlabel("Model's Prediction (Class ID)", fontsize=12)
    
    # Save the Master Graphic
    os.makedirs("data/processed/visualizations", exist_ok=True)
    save_path = "data/processed/visualizations/confusion_matrix.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Calculate Final Accuracy
    accuracy = 100.0 * np.sum(np.diag(cm)) / np.sum(cm)
    print(f"\nFinal Unseen Validation Accuracy: {accuracy:.2f}%")
    print(f"Confusion Matrix saved to: {save_path}")

if __name__ == "__main__":
    evaluate_model()