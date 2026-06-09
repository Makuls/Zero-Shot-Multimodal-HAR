import torch

def generate_mask(batch_size, seq_len, mask_ratio=0.3):
    """
    Creates a random binary mask for a batch of sequences.
    Returns: [batch_size, seq_len] where 1 = Keep, 0 = Mask
    """
    # Generate random values and create a binary mask based on the ratio
    mask = torch.rand(batch_size, seq_len) > mask_ratio
    return mask.float() 

def apply_mask(x, mask):
    """
    x: [batch_size, seq_len, input_dim]
    mask: [batch_size, seq_len]
    """
    # Reshape mask to [batch_size, seq_len, 1] to broadcast across channels
    mask_expanded = mask.unsqueeze(-1).to(x.device)
    return x * mask_expanded