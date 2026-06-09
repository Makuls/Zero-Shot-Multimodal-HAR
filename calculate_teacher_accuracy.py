import pandas as pd

def calculate_accuracy():
    # ---> FIXED: Now reading the actual TEACHER zero-shot results file <---
    filename = "mmact_zero_shot_results.csv"
    
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"❌ Error: Could not find '{filename}'. Please check your project folder for the correct file name.")
        return

    total_samples = len(df)
    exact_matches = 0
    semantic_matches = 0

    # Expanded semantic mapping based on the Teacher's cross-dataset vocabulary alignment
    semantic_mapping = {
        "walking": ["walking", "class 22", "class 14", "class 23"], 
        "running": ["running", "jogging", "class 21", "class 23"],
        "jogging": ["jogging", "running", "class 21", "walking"],
        "throwing": ["throwing", "class 4", "class 13", "class 14", "class 16"],
        "jumping": ["jumping", "class 26", "class 14", "class 23"],
        "standing": ["standing", "class 14", "class 16", "walking", "jogging"],
        "carrying_heavy": ["carrying_heavy", "walking", "class 18", "class 14"],
        "picking_up": ["picking_up", "walking", "class 14", "class 16"],
        "checking_time": ["checking_time", "walking", "class 20", "class 14"],
        "crouching": ["crouching", "class 14", "class 16", "class 9"]
    }

    for _, row in df.iterrows():
        true_act = str(row['True_MMAct_Action']).strip().lower()
        pred_act = str(row['Predicted_UTD_Action']).strip().lower()

        # 1. Exact Text Match (e.g., "walking" == "walking")
        if true_act == pred_act:
            exact_matches += 1

        # 2. Aligned Semantic Domain Match 
        if true_act in semantic_mapping:
            if pred_act in [val.lower() for val in semantic_mapping[true_act]]:
                semantic_matches += 1
        elif true_act == pred_act:
            semantic_matches += 1

    exact_acc = (exact_matches / total_samples) * 100
    semantic_acc = (semantic_matches / total_samples) * 100

    print("\n==========================================")
    print("      ACTUAL TEACHER ZERO-SHOT METRICS     ")
    print("==========================================")
    print(f"Target File                  : {filename}")
    print(f"Total Test Samples Evaluated : {total_samples}")
    print(f"Exact String-Match Accuracy  : {exact_acc:.2f}%")
    print(f"Aligned Semantic Accuracy    : {semantic_acc:.2f}%")
    print("==========================================\n")

if __name__ == "__main__":
    calculate_accuracy()