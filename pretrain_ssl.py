import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.nn.utils.rnn import pad_sequence

# --- YOUR EXACT FILE IMPORTS ---
from src.datasets.har_dataset import RealHARDataset
from src.datasets.multimodal_dataset import UTDMHADDataset
from src.models.temporal_transformer import HierarchicalMaskedAutoencoder
from src.utils.masking import generate_mask, apply_mask

def pad_collate(batch):
    """
    Custom collator to handle different sequence lengths and return formats 
    between WISDM and UTD-MHAD.
    """
    # 1. Extract the data tensor from the batch items
    # Some datasets return (data, label), others just data. We only want data for SSL.
    data_list = []
    for item in batch:
        if isinstance(item, (list, tuple)):
            data_list.append(item[0]) # Grab the sensor data, ignore label
        else:
            data_list.append(item)
            
    # 2. Ensure everything is a Float Tensor
    tensor_list = []
    for d in data_list:
        if not isinstance(d, torch.Tensor):
            tensor_list.append(torch.tensor(d, dtype=torch.float32).clone().detach())
        else:
            tensor_list.append(d.clone().detach().float())
    
    # 3. Pad the sequences to match the longest sequence in this specific batch
    # pad_sequence expects a list of tensors of shape [L, C] and returns [B, max_L, C]
    padded_batch = pad_sequence(tensor_list, batch_first=True, padding_value=0.0)
    
    return padded_batch

def pretrain_ssl():
    # Use your M4 Mac's GPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Initialize Datasets
    print("Loading WISDM dataset...")
    d1 = RealHARDataset(
        csv_file_path="data/raw/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt", 
        original_hz=20, target_hz=20, window_size_sec=6
    )
    
    print("Loading UTD-MHAD dataset...")
    d2 = UTDMHADDataset(root_dir="data/raw/UTD-MHAD/UTD-MHAD/")
    
    # 2. Combine and pass to DataLoader with our custom collator
    dataset = ConcatDataset([d1, d2])
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=pad_collate)
    
    # 3. Initialize Model (Targeting 6 channels)
    model = HierarchicalMaskedAutoencoder(input_channels=6, base_dim=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    model.train()
    print(f"\n--- Starting SSL Foundation Training on {len(dataset)} samples ---")
    
    for epoch in range(5):
        total_loss = 0
        for batch_idx, batch in enumerate(loader):
            # Move batch to M4 GPU
            batch = batch.to(device)
            
            # --- CHANNEL PADDING LOGIC ---
            # If one dataset has 3 channels (e.g., only Accelerometer) and the model expects 6:
            if batch.shape[-1] == 3:
                padding = torch.zeros(batch.shape[0], batch.shape[1], 3, device=device)
                batch = torch.cat([batch, padding], dim=-1)
                
            # ---> NEW: ON-THE-FLY DATA NORMALIZATION <---
            # This fixes the exploding loss by scaling all data to standard ranges
            # before the model sees it.
            mean = batch.mean(dim=1, keepdim=True)
            std = batch.std(dim=1, keepdim=True) + 1e-6
            batch = (batch - mean) / std
            # ---------------------------------------------
                
            B, T, C = batch.shape
            
            # --- MASKING & FORWARD PASS ---
            mask = generate_mask(B, T, mask_ratio=0.3).to(device)
            masked_batch = apply_mask(batch, mask)
            
            optimizer.zero_grad()
            reconstructed = model(masked_batch, mode="ssl")
            
            # --- LOSS CALCULATION ---
            # We calculate loss ONLY on the parts that were hidden (where mask == 0)
            inverse_mask = (1 - mask).unsqueeze(-1).expand(-1, -1, C)
            
            # Safety check to prevent empty loss if everything happened to be kept
            if inverse_mask.sum() > 0:
                loss = criterion(reconstructed * inverse_mask, batch * inverse_mask)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # Print progress every 100 batches so you know it hasn't frozen!
            if batch_idx % 100 == 0:
                print(f"  Batch [{batch_idx}/{len(loader)}] - Running Loss: {loss.item():.4f}")
            
        avg_loss = total_loss / len(loader)
        print(f"-> Epoch [{epoch+1}/5] Completed | Average MSE Loss: {avg_loss:.4f}\n")

    # 4. Save the pre-trained weights
    torch.save(model.state_dict(), "saved_models/ssl_foundation.pth")
    print("Foundation model successfully saved to: saved_models/ssl_foundation.pth")

if __name__ == "__main__":
    pretrain_ssl()