import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

class Downsample1D(nn.Module):
    def __init__(self, in_dim, out_dim, scale_factor=2):
        super().__init__()
        self.proj = nn.Conv1d(in_dim, out_dim, kernel_size=scale_factor, stride=scale_factor)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.proj(x)
        x = x.transpose(1, 2)
        return x

class Upsample1D(nn.Module):
    def __init__(self, in_dim, out_dim, scale_factor=2):
        super().__init__()
        self.proj = nn.ConvTranspose1d(in_dim, out_dim, kernel_size=scale_factor, stride=scale_factor)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.proj(x)
        x = x.transpose(1, 2)
        return x

class HierarchicalMaskedAutoencoder(nn.Module):
    def __init__(self, input_channels=6, base_dim=64, nhead=4, num_classes=27):
        super(HierarchicalMaskedAutoencoder, self).__init__()
        
        # 1. Projection
        self.input_projection = nn.Linear(input_channels, base_dim)
        self.pos_encoder_1 = PositionalEncoding(base_dim)
        
        # 2. Encoder Layers
        self.encoder_layer_1 = nn.TransformerEncoderLayer(d_model=base_dim, nhead=nhead, batch_first=True)
        self.downsample_1 = Downsample1D(base_dim, base_dim * 2, scale_factor=2)
        self.encoder_layer_2 = nn.TransformerEncoderLayer(d_model=base_dim * 2, nhead=nhead*2, batch_first=True)
        self.downsample_2 = Downsample1D(base_dim * 2, base_dim * 4, scale_factor=2)
        self.encoder_layer_3 = nn.TransformerEncoderLayer(d_model=base_dim * 4, nhead=nhead*4, batch_first=True)
        
        # 3. Decoder Layers (For Reconstruction)
        self.upsample_1 = Upsample1D(base_dim * 4, base_dim * 2, scale_factor=2)
        self.decoder_layer_1 = nn.TransformerEncoderLayer(d_model=base_dim * 2, nhead=nhead*2, batch_first=True)
        self.upsample_2 = Upsample1D(base_dim * 2, base_dim, scale_factor=2)
        self.decoder_layer_2 = nn.TransformerEncoderLayer(d_model=base_dim, nhead=nhead, batch_first=True)
        self.reconstruction_head = nn.Linear(base_dim, input_channels)
        
        # 4. Classification Head
        self.classifier_head = nn.Linear(base_dim * 4, num_classes)

    def forward(self, x, mode="classification"):
        # --- ENCODER ---
        x = self.input_projection(x)
        x = self.pos_encoder_1(x)
        x1 = self.encoder_layer_1(x)
        x2 = self.downsample_1(x1)
        x2 = self.encoder_layer_2(x2)
        latent_global = self.downsample_2(x2)
        latent_global = self.encoder_layer_3(latent_global)
        
        # --- MODE SWITCH ---
        if mode == "ssl":
            # --- DECODER ---
            d1 = self.upsample_1(latent_global)
            d1 = self.decoder_layer_1(d1)
            d2 = self.upsample_2(d1)
            d2 = self.decoder_layer_2(d2)
            return self.reconstruction_head(d2)
        elif mode == "features":
            # ---> THE FIX: Return the rich 256-dim embeddings! <---
            return latent_global
        else:
            # --- CLASSIFICATION ---
            return self.classifier_head(latent_global.mean(dim=1))