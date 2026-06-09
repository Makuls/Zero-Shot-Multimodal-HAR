import torch
import torch.nn as nn
import torchvision.models as models
from src.models.temporal_transformer import HierarchicalMaskedAutoencoder

class VisualBackbone(nn.Module):
    """
    Extracts spatial/posture embeddings from RGB frames.
    Utilizes a pretrained ResNet50, stripped of its final classification head, 
    to output raw feature vectors.
    """
    def __init__(self, embed_dim=256):
        super(VisualBackbone, self).__init__()
        # Load a pretrained ResNet50 for robust visual feature extraction
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Remove the final fully connected layer to get the raw 2048-dim feature map
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # Project the 2048-dim visual feature down to our desired embedding dimension
        self.projection = nn.Linear(2048, embed_dim)

    def forward(self, x):
        # x shape: [Batch, Channels (3), Height, Width]
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1) # Flatten: [Batch, 2048]
        visual_embedding = self.projection(features)   # [Batch, embed_dim]
        return visual_embedding

class MultimodalFusionNetwork(nn.Module):
    """
    Phase 4 Late Fusion Architecture.
    Combines temporal sensor embeddings with spatial visual embeddings.
    """
    def __init__(self, num_classes=6, sensor_embed_dim=256, visual_embed_dim=256):
        super(MultimodalFusionNetwork, self).__init__()
        
        # 1. Initialize our Hierarchical Sensor Model
        self.sensor_backbone = HierarchicalMaskedAutoencoder(input_channels=6, base_dim=64)
        
        # ---> DYNAMIC SHAPE ADAPTER <---
        # Run a dummy batch through the sensor backbone to find its exact output dimension
        self.sensor_backbone.eval()
        with torch.no_grad():
            dummy_x = torch.zeros(2, 120, 6) # Simulated IMU batch
            
            try:
                dummy_out = self.sensor_backbone(dummy_x, mode="features")
            except Exception:
                dummy_out = self.sensor_backbone(dummy_x)
                
            if isinstance(dummy_out, tuple):
                dummy_feat = dummy_out[-1]
            else:
                dummy_feat = dummy_out
                
            if dummy_feat.dim() == 3:
                dummy_feat = dummy_feat.mean(dim=1)
                
            actual_sensor_dim = dummy_feat.shape[-1] # Dynamically grabs the output dimension (e.g., 27)
        self.sensor_backbone.train()
        
        print(f"🔧 Fusion Adapter configured! Sensor dim: {actual_sensor_dim}, Visual dim: {visual_embed_dim}")
        
        # 2. Initialize the Visual Backbone
        self.visual_backbone = VisualBackbone(embed_dim=visual_embed_dim)
        
        # 3. The Fusion MLP 
        # Dynamically calculate the fusion dimension (e.g., 27 + 256 = 283)
        self.fusion_dim = actual_sensor_dim + visual_embed_dim
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(self.fusion_dim // 2, num_classes)
        )

    # ---> NEW: Added return_features=False <---
    def forward(self, sensor_data, visual_data, return_features=False):
        # --- Modality 1: Motion (IMU) ---
        # --- BULLETPROOF SENSOR EXTRACTION ---
        try:
            sensor_out = self.sensor_backbone(sensor_data, mode="features")
        except Exception:
            sensor_out = self.sensor_backbone(sensor_data)
            
        if isinstance(sensor_out, tuple):
            sensor_embedding = sensor_out[-1] 
        else:
            sensor_embedding = sensor_out     
            
        if sensor_embedding.dim() == 3:
            sensor_embedding = sensor_embedding.mean(dim=1)
        # -------------------------------------
        
        # --- Modality 2: Posture (RGB) ---
        visual_embedding = self.visual_backbone(visual_data) # Shape: [Batch, visual_embed_dim]
        
        # --- Late Fusion ---
        # Concatenate features along the dimension axis
        fused_features = torch.cat((sensor_embedding, visual_embedding), dim=1) 
        
        # Predict the Activity Class
        activity_logits = self.fusion_mlp(fused_features)
        
        # ---> NEW: Safe Feature Extraction Hatch <---
        if return_features:
            return activity_logits, fused_features
            
        return activity_logits

if __name__ == "__main__":
    # Local verification block
    simulated_imu = torch.randn(16, 120, 6)
    simulated_rgb = torch.randn(16, 3, 224, 224)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = MultimodalFusionNetwork(num_classes=6).to(device)
    
    simulated_imu = simulated_imu.to(device)
    simulated_rgb = simulated_rgb.to(device)
    
    # Test standard output
    predictions = model(simulated_imu, simulated_rgb)
    
    # Test feature extraction output
    preds, features = model(simulated_imu, simulated_rgb, return_features=True)
    
    print("Phase 4 Fusion Architecture: SUCCESS")
    print(f"Feature Dimension for Student to mimic: {features.shape[1]}")