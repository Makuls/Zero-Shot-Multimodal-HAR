import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, random_split

from src.datasets.har_dataset import RealHARDataset
from train_real_classifier import RealIMUClassifier

def generate_wisdm_saliency():
    print("--- Phase 12B: WISDM Interpretability (Saliency Maps) ---")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Load the dataset (Seeded to prevent leakage)
    torch.manual_seed(42)
    file_path = "data/raw/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt"
    dataset = RealHARDataset(csv_file_path=file_path, original_hz=20, target_hz=20, window_size_sec=6)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Batch size 1 so we can examine a single specific movement
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=0)
    
    # 2. Load the optimal WISDM model (97.49%)
    model = RealIMUClassifier(num_classes=6).to(device)
    model.load_state_dict(torch.load("saved_models/best_imu_classifier.pth", map_location=device))
    model.eval()
    print("✅ Model loaded. Extracting gradient attention...")

    # 3. Grab a single sample
    data, label = next(iter(val_loader))
    data = data.to(device)
    
    # ---> CRITICAL: Apply the exact IMU normalization <---
    mean = data.mean(dim=1, keepdim=True)
    std = data.std(dim=1, keepdim=True) + 1e-6
    data = (data - mean) / std
    
    # 4. Enable gradient tracking (Saliency)
    data.requires_grad_()
    
    # Forward Pass
    outputs = model(data)
    predicted_class = outputs.argmax().item()
    
    # Backward Pass (Force the model to trace its decision backwards)
    outputs[0, predicted_class].backward()
    
    # 5. Process Gradients into a 1D Heatmap line
    # Average the gradient importance across the 6 channels (which is dim=1, not dim=0)
    saliency = data.grad.abs().squeeze().mean(dim=1).cpu().numpy()
    
    # 6. Plot the Master Graphic
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Map the IDs to human-readable names
    class_names = ["Walking", "Jogging", "Stairs", "Sitting", "Standing", "Typing"]
    true_name = class_names[label.item()]
    pred_name = class_names[predicted_class]
    
    ax.set_title(f"WISDM Smartphone Sensor Attention Map\nTrue: {true_name} | Predicted: {pred_name}", fontsize=18)
    
    # Extract the X, Y, Z lines for plotting. Shape is (120, 6)
    sensor_lines = data.detach().squeeze().cpu().numpy()
    
    # ---> THE FIX: We plot all rows (:) for columns 0, 1, and 2 <---
    ax.plot(sensor_lines[:, 0], label="Accelerometer X", alpha=0.7, color='#ff4757')
    ax.plot(sensor_lines[:, 1], label="Accelerometer Y", alpha=0.7, color='#2ed573')
    ax.plot(sensor_lines[:, 2], label="Accelerometer Z", alpha=0.7, color='#1e90ff')
    ax.set_xlabel("Time Step (Window)")
    ax.set_ylabel("Normalized Acceleration")
    ax.legend(loc="upper right")
    
    # Overlay the red Saliency heatmap in the background
    # ---> THE FIX: Set extent to sensor_lines.shape[0] (which is 120) <---
    ax.imshow(saliency[np.newaxis, :], cmap="Reds", aspect="auto", 
              extent=[0, sensor_lines.shape[0], ax.get_ylim()[0], ax.get_ylim()[1]], alpha=0.5)
    
    os.makedirs("data/processed/visualizations", exist_ok=True)
    save_path = "data/processed/visualizations/wisdm_saliency_map.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"🔥 WISDM Saliency Map saved to: {save_path}")

if __name__ == "__main__":
    generate_wisdm_saliency()