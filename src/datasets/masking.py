import numpy as np
import torch

class IMUSpanMasker:
    """
    Implements the geometric Span Masking mechanism from LIMU-BERT/LIMU-BERT-X.
    Instead of masking random isolated time steps, it masks continuous blocks (spans)
    of data to force the transformer to learn true temporal movement dynamics.
    """
    def __init__(self, p_success=0.2, max_span_len=10, mask_ratio=0.15, mask_prob=0.8):
        self.p_success = p_success          # p in Geometric distribution Geo(p)
        self.max_span_len = max_span_len    # l_max clipping value
        self.mask_ratio = mask_ratio        # Total percentage of sequence to mask (p_r)
        self.mask_prob = mask_prob          # Probability of replacing with 0 vs keeping unchanged (P_m)

    def generate_mask(self, seq_len):
        """
        Executes the explicit span selection loop based on the paper's core algorithm.
        Returns a boolean mask where True indicates a masked position.
        """
        max_masked_elements = int(seq_len * self.mask_ratio)
        masked_indices = set()
        masked_count = 0
        
        # Sample p_m globally for the sequence to determine the masking style rule
        p_m = np.random.uniform(0, 1)
        should_replace_with_zero = p_m < self.mask_prob

        while masked_count < max_masked_elements:
            # Sample a random starting position 's' from U[0, L)
            start_pos = np.random.randint(0, seq_len)
            
            if start_pos not in masked_indices:
                # Sample length 'l' from a Geometric distribution
                span_len = np.random.geometric(p=self.p_success)
                
                # Clip length to avoid exceeding boundaries or quotas
                span_len = min(span_len, self.max_span_len)
                span_len = min(span_len, max_masked_elements - masked_count)
                
                end_pos = min(start_pos + span_len, seq_len)
                
                # Add the selected continuous window indices to our mask tracking group
                for idx in range(start_pos, end_pos):
                    if idx not in masked_indices:
                        masked_indices.add(idx)
                        masked_count += 1
                        
        # Construct the final mask tensor
        mask = torch.zeros(seq_len, dtype=torch.bool)
        for idx in masked_indices:
            mask[idx] = True
            
        return mask, should_replace_with_zero

    def mask_sequence(self, sequence):
        """
        Args:
            sequence (Tensor): Input IMU tensor of shape (seq_len, channels)
        Returns:
            masked_seq (Tensor): The sequence with continuous spans blacked out or perturbed
            mask (Tensor): Boolean mask tracking which indices were selected
        """
        seq_len, channels = sequence.shape
        masked_seq = sequence.clone()
        
        mask, should_replace_with_zero = self.generate_mask(seq_len)
        
        if should_replace_with_zero:
            # Replaces the geometric chunks entirely with 0 to challenge the encoder
            masked_seq[mask] = 0.0
            
        return masked_seq, mask

# Quick verification block to test locally on your MacBook Air M4
if __name__ == "__main__":
    # Simulate an IMU data window: 6 seconds at 20Hz = 120 timestamps, 6 sensor axes (Acc XYZ, Gyro XYZ)
    test_tensor = torch.randn(120, 6)
    
    masker = IMUSpanMasker()
    masked_data, mask_indices = masker.mask_sequence(test_tensor)
    
    total_masked = mask_indices.sum().item()
    print("--- Span Masking Test Output ---")
    print(f"Original Sequence Shape: {test_tensor.shape}")
    print(f"Total timestamps masked: {total_masked} out of 120 ({(total_masked/120)*100:.1f}%)")
    print(f"Mask array sample (first 20 frames): {mask_indices[:20].int().tolist()}")