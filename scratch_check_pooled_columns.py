import pandas as pd
import numpy as np

def main():
    df = pd.read_csv("g:\\My Drive\\gemini\\NHANES\\nhanes_depression_pooled.csv")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Outcomes and their counts
    outcomes = ["diabetes_binary", "hypertension_binary", "high_cholesterol_binary", 
                "arthritis_binary", "liver_condition_binary", "thyroid_binary", 
                "heart_attack_binary", "stroke_binary", "depression_binary"]
    
    print("\nOutcome Non-Null and Yes Counts:")
    for out in outcomes:
        if out in df.columns:
            val_counts = df[out].value_counts(dropna=False)
            yes_count = val_counts.get(1.0, 0)
            no_count = val_counts.get(0.0, 0)
            nan_count = val_counts.get(np.nan, 0)
            print(f"  {out}: Yes={yes_count}, No={no_count}, NaN={nan_count} (Prevalence={yes_count/(yes_count+no_count)*100:.2f}%)")
        else:
            print(f"  {out}: NOT FOUND")

if __name__ == "__main__":
    main()
