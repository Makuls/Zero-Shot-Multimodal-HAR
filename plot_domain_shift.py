import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_heatmap():
    print("Loading zero-shot results...")
    # Load your generated CSV for the Feature-Distilled Model
    df = pd.read_csv("mmact_feature_results.csv")
    
    # Generate a cross-tabulation matrix (The Domain Shift Confusion Matrix)
    # This counts exactly how many times a True MMAct action was predicted as a specific UTD action
    shift_matrix = pd.crosstab(df['True_MMAct_Action'], df['Predicted_UTD_Action'])
    
    print("Generating publication-ready heatmap...")
    # Setup the matplotlib figure size to accommodate all the classes
    plt.figure(figsize=(18, 12))
    
    # Create a beautiful seaborn heatmap
    # annot=True puts the numbers in the boxes, fmt='d' ensures they are whole integers
    sns.heatmap(shift_matrix, annot=True, fmt='d', cmap='Blues', cbar=True, linewidths=.5)
    
    # Format the labels for academic presentation
    plt.title('Cross-Dataset Domain Shift: MMAct Actions vs. UTD-MHAD Predictions', fontsize=18, pad=20)
    plt.xlabel('Predicted UTD-MHAD Action (Model Output)', fontsize=14, labelpad=15)
    plt.ylabel('True MMAct Action (Unseen Ground Truth)', fontsize=14, labelpad=15)
    
    # Rotate the X-axis labels so they are readable
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    
    plt.tight_layout()
    
    # ---> UPDATED: Save as a distinct PNG for the Feature Distillation results
    output_filename = "domain_shift_feature_heatmap.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    
    print(f"✅ Success! Graphic saved to your project folder as '{output_filename}'")
    
    # Display the plot on your screen
    plt.show()

if __name__ == "__main__":
    # If you don't have seaborn installed, run: pip install seaborn matplotlib pandas
    generate_heatmap()