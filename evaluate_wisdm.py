import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, random_split
import os

from src.datasets.har_dataset import RealHARDataset
from train_real_classifier import RealIMUClassifier

def evaluate_wisdm():
    print("--- Phase 12B: WISDM Sensor-Only Evaluation & Confusion Matrix ---")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Ensure the exact same Train/Val split as the training script
    torch.manual_seed(42) # Keeps the split consistent
    file_path = "data/raw/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt"
    dataset = RealHARDataset(csv_file_path=file_path, original_hz=20, target_hz=20, window_size_sec=6)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_dataset = random_split(dataset, [train_size, val_size])
    
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    # 2. Load the trained WISDM weights
    model = RealIMUClassifier(num_classes=6).to(device)
    model_path = "saved_models/best_imu_classifier.pth"
    
    if not os.path.exists(model_path):
        print(f"Error: Could not find {model_path}. Run train_real_classifier.py first!")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("✅ Successfully loaded optimal WISDM weights!")

    # 3. Run Inference on Unseen Data
    all_preds = []
    all_labels = []
    
    print(f"Running inference on {val_size} unseen WISDM validation samples...")
    with torch.no_grad():
        for data, labels in val_loader:
            data, labels = data.to(device), labels.to(device)
            
            # ---> CRITICAL: Apply the same normalization used in training <---
            mean = data.mean(dim=1, keepdim=True)
            std = data.std(dim=1, keepdim=True) + 1e-6
            data = (data - mean) / std
            
            outputs = model(data)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # 4. Generate the Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # WISDM typical classes (adjust if your parser maps them differently)
    class_names = ["Walking", "Jogging", "Stairs", "Sitting", "Standing", "Typing"]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", cbar=False, 
                xticklabels=class_names, yticklabels=class_names)
    plt.title("WISDM Validation Confusion Matrix (IMU Only)", fontsize=16, pad=20)
    plt.ylabel("Actual Human Activity", fontsize=12)
    plt.xlabel("Model's Prediction", fontsize=12)
    
    # Save the Master Graphic
    os.makedirs("data/processed/visualizations", exist_ok=True)
    save_path = "data/processed/visualizations/wisdm_confusion_matrix.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Calculate Final Accuracy
    accuracy = 100.0 * np.sum(np.diag(cm)) / np.sum(cm)
    print(f"\nFinal Unseen Validation Accuracy: {accuracy:.2f}%")
    print(f"Confusion Matrix saved to: {save_path}")

if __name__ == "__main__":
    evaluate_wisdm()