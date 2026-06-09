import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class AttentionVisualizer:
    """
    Phase 5: Mechanistic Interpretability.
    Extracts and plots the attention matrix to explain WHAT the model is looking at.
    """
    def __init__(self, output_dir="data/processed/visualizations"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def plot_attention_heatmap(self, attention_matrix, layer_name="Layer_1", head_idx=0):
        """
        Plots a heatmap of the temporal attention weights.
        Args:
            attention_matrix (Tensor or ndarray): Shape [Seq_Len, Seq_Len]
            layer_name (str): Name of the transformer block
            head_idx (int): Which attention head we are visualizing
        """
        if torch.is_tensor(attention_matrix):
            # Move to CPU and convert to numpy for plotting
            attention_matrix = attention_matrix.detach().cpu().numpy()

        plt.figure(figsize=(10, 8))
        
        # Plot using a high-contrast colormap (viridis)
        sns.heatmap(attention_matrix, cmap="viridis", 
                    xticklabels=False, yticklabels=False, 
                    cbar_kws={'label': 'Attention Weight'})
        
        plt.title(f"Temporal Attention Map: {layer_name} | Head {head_idx}")
        plt.xlabel("Target Temporal Sequence")
        plt.ylabel("Source Temporal Sequence")
        
        # Save the figure
        filename = f"{self.output_dir}/attention_heatmap_{layer_name}_head{head_idx}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[*] Interpretability Heatmap saved to: {filename}")

# Verification block to test the plotting logic
if __name__ == "__main__":
    print("--- Phase 5: Testing Attention Visualizer ---")
    
    # Simulate a 30x30 attention matrix from our compressed Global layer
    # We use a diagonal bias to simulate the model paying attention to local temporal continuity
    simulated_attention = np.random.rand(30, 30) * 0.2
    np.fill_diagonal(simulated_attention, 1.0) 
    
    visualizer = AttentionVisualizer()
    visualizer.plot_attention_heatmap(simulated_attention, layer_name="Global_Encoder", head_idx=1)
    
    print("Phase 5 Utility: SUCCESS. Check the data/processed/visualizations folder for the PNG!")