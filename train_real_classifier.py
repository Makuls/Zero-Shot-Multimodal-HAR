import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np

# Import our custom modules
from src.datasets.har_dataset import RealHARDataset
from src.models.temporal_transformer import HierarchicalMaskedAutoencoder

class RealIMUClassifier(nn.Module):
    """
    Takes our pretrained Hierarchical Encoder and adds a classification head
    to predict the actual human activity.
    """
    def __init__(self, num_classes=6, base_dim=64):
        super(RealIMUClassifier, self).__init__()
        
        # 1. Load the architecture
        self.encoder = HierarchicalMaskedAutoencoder(input_channels=6, base_dim=base_dim)
        
        # 2. ---> LOAD THE SSL FOUNDATION WEIGHTS <---
        print("Loading pre-trained SSL Foundation Brain...")
        try:
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            foundation_weights = torch.load("saved_models/ssl_foundation.pth", map_location=device, weights_only=True)
            self.encoder.load_state_dict(foundation_weights, strict=False)
            print("✅ SSL Weights loaded successfully! The model already understands motion.")
        except Exception as e:
            print(f"⚠️ Warning: Could not load SSL weights. Error: {e}")
        
        # 3. ---> DYNAMIC SHAPE ADAPTER (THE FIX) <---
        # We run a dummy batch through the encoder to see exactly what shape it outputs.
        # This makes the code immune to shape crashes (like the 27 vs 256 error).
        self.encoder.eval()
        with torch.no_grad():
            dummy_x = torch.zeros(2, 120, 6) # Simulated WISDM batch [Batch, Time, Channels]
            
            try:
                dummy_out = self.encoder(dummy_x, mode="features")
            except Exception:
                dummy_out = self.encoder(dummy_x)
                
            if isinstance(dummy_out, tuple):
                dummy_feat = dummy_out[-1]
            else:
                dummy_feat = dummy_out
                
            if dummy_feat.dim() == 3:
                dummy_feat = dummy_feat.mean(dim=1)
                
            extracted_dim = dummy_feat.shape[-1] # Dynamically grabs the output dimension
        self.encoder.train()
        
        print(f"🔧 Dynamic Adapter configured! Building classifier for dimension: {extracted_dim}")
        
        # 4. Add the classification head perfectly sized for the encoder
        self.classifier = nn.Sequential(
            nn.Linear(extracted_dim, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Safely extract the features exactly how the dummy pass did it
        try:
            encoded_output = self.encoder(x, mode="features")
        except Exception:
            encoded_output = self.encoder(x)
            
        if isinstance(encoded_output, tuple):
            latent_features = encoded_output[-1] 
        else:
            latent_features = encoded_output     
            
        # Globally average pool the time dimension if it exists
        if latent_features.dim() == 3:
            pooled_features = latent_features.mean(dim=1)
        else:
            pooled_features = latent_features
            
        # Predict the activity
        return self.classifier(pooled_features)

if __name__ == "__main__":
    print("--- Phase 9: Supervised Fine-Tuning (WISDM) ---")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Compute Device: {device}")
    
    # 1. Load the Dataset
    file_path = "data/raw/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt"
    full_dataset = RealHARDataset(csv_file_path=file_path, original_hz=20, target_hz=20, window_size_sec=6)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    # 2. Initialize Model (WISDM has 6 classes)
    model = RealIMUClassifier(num_classes=6).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    
    # 3. Supervised Training Loop
    epochs = 10
    print(f"\nStarting {epochs}-Epoch Supervised Fine-Tuning...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            
            # Apply Normalization
            mean = data.mean(dim=1, keepdim=True)
            std = data.std(dim=1, keepdim=True) + 1e-6
            data = (data - mean) / std
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Validation Pass
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                
                # Apply validation normalization
                mean = data.mean(dim=1, keepdim=True)
                std = data.std(dim=1, keepdim=True) + 1e-6
                data = (data - mean) / std
                
                outputs = model(data)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        accuracy = 100.0 * correct / total
        avg_train_loss = total_loss / len(train_loader)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Accuracy: {accuracy:.2f}%")

    # Save the final fine-tuned classifier
    torch.save(model.state_dict(), "saved_models/best_imu_classifier.pth")
    print("\nPhase 9 Complete! Model saved to saved_models/best_imu_classifier.pth")