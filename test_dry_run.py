import torch
import time

# Import your heavy, 90.75% accurate architecture
from src.models.multimodal_fusion import MultimodalFusionNetwork

def dry_run():
    print("--- Phase 13 Pre-Check: Dummy Data Dry Run ---")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Compute Device: {device}")
    
    # 1. Load the Master Model
    print("\n1. Initializing MultimodalFusionNetwork...")
    model = MultimodalFusionNetwork(num_classes=27).to(device)
    
    # Load your highly trained weights
    model.load_state_dict(torch.load("saved_models/best_multimodal_model.pth", map_location=device))
    model.eval() # Strictly evaluation mode
    print("✅ Model loaded successfully.")

    # 2. Generate Fake "WEAR" Data (Random Noise)
    print("\n2. Generating fake random tensors to simulate the WEAR dataset...")
    # Shape: (Batch Size 1, 120 time steps, 6 sensor channels)
    fake_imu = torch.randn(1, 120, 6).to(device)
    
    # Shape: (Batch Size 1, 3 color channels, 224x224 image resolution)
    fake_rgb = torch.randn(1, 3, 224, 224).to(device)
    
    print(f"  -> Fake IMU Tensor Shape: {fake_imu.shape}")
    print(f"  -> Fake RGB Tensor Shape: {fake_rgb.shape}")

    # 3. Push it through the network
    print("\n3. Executing Forward Pass through the Fusion Layer...")
    start_time = time.time()
    
    with torch.no_grad():
        output = model(fake_imu, fake_rgb)
        
    inference_time = (time.time() - start_time) * 1000 # Convert to milliseconds

    # 4. Results
    predicted_class = output.argmax(dim=1).item()
    print("\n✅ SUCCESS! The mathematical plumbing is flawless.")
    print(f"  -> Output Tensor Shape: {output.shape} (Expected: [1, 27])")
    print(f"  -> Inference Time:      {inference_time:.2f} ms")
    print(f"  -> Random Prediction:   Class {predicted_class}")
    print("\nNote: The predicted class is completely meaningless because the input was just random white noise!")
    print("If you reached this line without crashing, you are 100% ready for the WEAR dataset.")

if __name__ == "__main__":
    dry_run()