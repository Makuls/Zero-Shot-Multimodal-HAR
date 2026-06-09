import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, random_split
from scipy.ndimage import gaussian_filter # ---> NEW: The smoothing filter

# Import your bulletproof architecture and dataset
from src.models.multimodal_fusion import MultimodalFusionNetwork
from src.datasets.multimodal_dataset import UTDMHADDataset

def generate_heatmaps():
    print("--- Phase 12C: Multimodal Interpretability (Saliency Maps) ---")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Load the dataset with the exact same seed to prevent data leakage
    torch.manual_seed(42)
    dataset = UTDMHADDataset(root_dir="data/raw/UTD-MHAD/UTD-MHAD/")
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Batch size of 1 so we can look at a single, specific human action
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=0)
    
    # 2. Load the optimal 90.75% model
    model = MultimodalFusionNetwork(num_classes=27).to(device)
    model.load_state_dict(torch.load("saved_models/best_multimodal_model.pth", map_location=device))
    model.eval()
    print("✅ Model loaded. Extracting gradient attention...")

    # 3. Grab a single sample
    imu, rgb, label = next(iter(val_loader))
    imu, rgb = imu.to(device), rgb.to(device)
    
    # Apply the exact IMU normalization
    mean = imu.mean(dim=1, keepdim=True)
    std = imu.std(dim=1, keepdim=True) + 1e-6
    imu = (imu - mean) / std
    
    # 4. Enable gradient tracking
    imu.requires_grad_()
    rgb.requires_grad_()
    
    # Forward & Backward Pass
    outputs = model(imu, rgb)
    predicted_class = outputs.argmax().item()
    outputs[0, predicted_class].backward()
    
    # 5. Process the Gradients
    imu_saliency = imu.grad.abs().squeeze().mean(dim=1).cpu().numpy()
    rgb_saliency = rgb.grad.abs().squeeze().max(dim=0).values.cpu().numpy()
    
    # ---> THE GAUSSIAN BLUR FIX <---
    # This smooths out the static noise and creates clean blobs of attention
    rgb_saliency = gaussian_filter(rgb_saliency, sigma=4)
    
    # Normalize the saliency map to a 0-1 scale
    saliency_norm = (rgb_saliency - rgb_saliency.min()) / (rgb_saliency.max() - rgb_saliency.min() + 1e-8)
    
    # Get the jet colormap and convert our 2D array into a 3D RGBA array
    cmap = plt.get_cmap('jet')
    saliency_rgba = cmap(saliency_norm)
    
    # Explicitly edit the Alpha channel
    # We lowered the threshold to 70% because the blur spreads the heat out
    threshold = np.percentile(rgb_saliency, 70)
    saliency_rgba[rgb_saliency < threshold, 3] = 0.0  # Force background to invisible
    saliency_rgba[rgb_saliency >= threshold, 3] = 0.5 # Force heatmap to 50% visible
    
    # Prepare the original RGB image for plotting
    rgb_image = rgb.detach().squeeze().cpu().permute(1, 2, 0).numpy()
    rgb_image = np.clip((rgb_image * 0.229) + 0.485, 0, 1)
    
    # 6. Plotting the Master Graphic
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Multimodal Attention Map | True Class: {label.item()} | Predicted: {predicted_class}", fontsize=18)
    
    # Plot 1: IMU Attention
    axes[0].plot(imu.detach().squeeze().cpu().numpy(), alpha=0.3)
    axes[0].set_title("Inertial Sensor (IMU) Attention", fontsize=14)
    axes[0].set_xlabel("Time Step (Window)")
    axes[0].set_ylabel("Sensor Amplitude")
    axes[0].imshow(imu_saliency[np.newaxis, :], cmap="Reds", aspect="auto", 
                   extent=[0, 120, axes[0].get_ylim()[0], axes[0].get_ylim()[1]], alpha=0.5)
                   
    # Plot 2: Vision Attention
    axes[1].imshow(rgb_image)
    # Plot the smoothed RGBA array
    axes[1].imshow(saliency_rgba) 
    axes[1].set_title("Spatial Vision (ResNet) Attention", fontsize=14)
    axes[1].axis('off')
    
    # Save it
    os.makedirs("data/processed/visualizations", exist_ok=True)
    save_path = "data/processed/visualizations/multimodal_attention_heatmap.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"🔥 Saliency Heatmaps generated and saved to: {save_path}")

if __name__ == "__main__":
    generate_heatmaps()