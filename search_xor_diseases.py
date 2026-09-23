import pandas as pd
import numpy as np
import statsmodels.api as sm
from itertools import combinations
import os

# Friendly names mapping for markers
MARKER_NAMES = {
    "systolic_bp": "Systolic BP (mmHg)",
    "diastolic_bp": "Diastolic BP (mmHg)",
    "BMXBMI": "Body Mass Index (kg/m2)",
    "BMXWAIST": "Waist Circumference (cm)",
    "LBXGH": "Glycohemoglobin (HbA1c, %)",
    "LBXGLU": "Fasting Glucose (mg/dL)",
    "LBXIN": "Insulin (uIU/mL)",
    "LBXTC": "Total Cholesterol (mg/dL)",
    "LBDHDD": "HDL Cholesterol (mg/dL)",
    "LBDLDL": "LDL Cholesterol (mg/dL)",
    "LBXTR": "Triglycerides (mg/dL)",
    "LBXSATSI": "ALT Liver Enzyme (U/L)",
    "LBXSASSI": "AST Liver Enzyme (U/L)",
    "LBXSAPSI": "Alkaline Phosphatase (U/L)",
    "LBXSCR": "Creatinine Kidney (mg/dL)",
    "LBXSBU": "BUN Kidney (mg/dL)",
    "LBXSUA": "Uric Acid (mg/dL)",
    "LBXSNASI": "Sodium (mmol/L)",
    "LBXSKSI": "Potassium (mmol/L)",
    "LBXSCA": "Calcium (mg/dL)",
    "LBXSAL": "Albumin (g/dL)",
    "LBXHSCRP": "hs-CRP Inflammation (mg/L)",
    "LBXHGB": "Hemoglobin (g/dL)",
    "LBXWBCSI": "WBC Count (10^3/uL)",
    "LBXPLTSI": "Platelet Count (10^3/uL)"
}

# Friendly names mapping for outcomes
OUTCOME_NAMES = {
    "diabetes_binary": "Diabetes",
    "hypertension_binary": "Hypertension",
    "high_cholesterol_binary": "High Cholesterol",
    "arthritis_binary": "Arthritis",
    "liver_condition_binary": "Liver Condition",
    "thyroid_binary": "Thyroid Problem",
    "heart_attack_binary": "Heart Attack",
    "stroke_binary": "Stroke",
    "depression_binary": "Depression"
}

def analyze_pair(df, outcome, m1, m2, anomaly_type):
    # 1. Clean subset (remove NaNs in outcome and markers)
    clean_cols = [outcome, m1, m2]
    df_clean = df[clean_cols].dropna().copy()
    
    n_samples = len(df_clean)
    n_cases = df_clean[outcome].sum()
    
    # Require minimum samples and cases
    if n_samples < 500 or n_cases < 30:
        return None
        
    # 2. Compute 10th and 90th percentiles for thresholds on this subset
    p10_1, p90_1 = df_clean[m1].quantile(0.1), df_clean[m1].quantile(0.9)
    p10_2, p90_2 = df_clean[m2].quantile(0.1), df_clean[m2].quantile(0.9)
    
    # 3. Create binary anomaly flags
    if anomaly_type == "High-High":
        df_clean["X_A"] = (df_clean[m1] > p90_1).astype(float)
        df_clean["X_B"] = (df_clean[m2] > p90_2).astype(float)
        lbl_A, lbl_B = "High", "High"
    elif anomaly_type == "Low-Low":
        df_clean["X_A"] = (df_clean[m1] < p10_1).astype(float)
        df_clean["X_B"] = (df_clean[m2] < p10_2).astype(float)
        lbl_A, lbl_B = "Low", "Low"
    elif anomaly_type == "High-Low":
        df_clean["X_A"] = (df_clean[m1] > p90_1).astype(float)
        df_clean["X_B"] = (df_clean[m2] < p10_2).astype(float)
        lbl_A, lbl_B = "High", "Low"
    elif anomaly_type == "Low-High":
        df_clean["X_A"] = (df_clean[m1] < p10_1).astype(float)
        df_clean["X_B"] = (df_clean[m2] > p90_2).astype(float)
        lbl_A, lbl_B = "Low", "High"
    elif anomaly_type == "TwoSided-TwoSided":
        df_clean["X_A"] = ((df_clean[m1] < p10_1) | (df_clean[m1] > p90_1)).astype(float)
        df_clean["X_B"] = ((df_clean[m2] < p10_2) | (df_clean[m2] > p90_2)).astype(float)
        lbl_A, lbl_B = "Out-of-Bounds", "Out-of-Bounds"
    else:
        return None
        
    # 4. Count cell sizes
    # Cells: (X_A, X_B) -> (0,0), (1,0), (0,1), (1,1)
    c00 = df_clean[(df_clean["X_A"] == 0) & (df_clean["X_B"] == 0)]
    c10 = df_clean[(df_clean["X_A"] == 1) & (df_clean["X_B"] == 0)]
    c01 = df_clean[(df_clean["X_A"] == 0) & (df_clean["X_B"] == 1)]
    c11 = df_clean[(df_clean["X_A"] == 1) & (df_clean["X_B"] == 1)]
    
    n00, y00 = len(c00), c00[outcome].sum()
    n10, y10 = len(c10), c10[outcome].sum()
    n01, y01 = len(c01), c01[outcome].sum()
    n11, y11 = len(c11), c11[outcome].sum()
    
    # Require cell sizes to be sufficient
    # n11 is the co-occurrence, which will be small, but must exist.
    if n00 < 50 or n10 < 10 or n01 < 10 or n11 < 5:
        return None
        
    # 5. Compute raw probabilities
    p00 = y00 / n00 if n00 > 0 else 0
    p10 = y10 / n10 if n10 > 0 else 0
    p01 = y01 / n01 if n01 > 0 else 0
    p11 = y11 / n11 if n11 > 0 else 0
    
    # 6. Haldane-Anscombe smoothed Odds Ratios (add 0.5 to avoid division by zero)
    def smoothed_or(y_target, n_target, y_base, n_base):
        t_pos = y_target + 0.5
        t_neg = (n_target - y_target) + 0.5
        b_pos = y_base + 0.5
        b_neg = (n_base - y_base) + 0.5
        return (t_pos / t_neg) / (b_pos / b_neg)
        
    or10 = smoothed_or(y10, n10, y00, n00)
    or01 = smoothed_or(y01, n01, y00, n00)
    or11 = smoothed_or(y11, n11, y00, n00)
    
    # XOR groups: (1,0) and (0,1) vs Non-XOR groups: (0,0) and (1,1)
    nXOR = n10 + n01
    yXOR = y10 + y01
    nNonXOR = n00 + n11
    yNonXOR = y00 + y11
    orXOR = smoothed_or(yXOR, nXOR, yNonXOR, nNonXOR)
    
    # 7. Fit Logistic Regression using Statsmodels
    try:
        # Construct features
        df_clean["const"] = 1.0
        df_clean["interaction"] = df_clean["X_A"] * df_clean["X_B"]
        
        X = df_clean[["const", "X_A", "X_B", "interaction"]]
        y = df_clean[outcome]
        
        # Fit Logit model
        res = sm.Logit(y, X).fit(disp=0)
        
        beta0 = res.params["const"]
        beta1 = res.params["X_A"]
        beta2 = res.params["X_B"]
        beta3 = res.params["interaction"]
        
        p0 = res.pvalues["const"]
        p1 = res.pvalues["X_A"]
        p2 = res.pvalues["X_B"]
        p3 = res.pvalues["interaction"]
        
    except Exception as e:
        # If statsmodels fails, skip or return None (usually due to perfect separation)
        return None
        
    # 8. Record result
    result = {
        "outcome_var": outcome,
        "outcome_name": OUTCOME_NAMES[outcome],
        "marker1_var": m1,
        "marker1_name": MARKER_NAMES[m1],
        "marker2_var": m2,
        "marker2_name": MARKER_NAMES[m2],
        "anomaly_type": anomaly_type,
        "lbl_A": lbl_A,
        "lbl_B": lbl_B,
        "n_samples": n_samples,
        "n_cases": n_cases,
        # Cell sizes
        "n00": n00, "y00": y00, "p00": p00,
        "n10": n10, "y10": y10, "p10": p10,
        "n01": n01, "y01": y01, "p01": p01,
        "n11": n11, "y11": y11, "p11": p11,
        # Odds Ratios
        "or10": or10,
        "or01": or01,
        "or11": or11,
        "orXOR": orXOR,
        # Logistic Regression Params
        "beta0": beta0, "p_const": p0,
        "beta1": beta1, "p1_A": p1,
        "beta2": beta2, "p2_B": p2,
        "beta3": beta3, "p3_interaction": p3,
        # XOR score
        "beta1_plus_beta2_plus_beta3": beta1 + beta2 + beta3,
        "xor_synergy": -beta3 - abs(beta1 + beta2 + beta3)
    }
    return result

def main():
    if not os.path.exists("nhanes_merged_2017_2018.csv"):
        print("Error: nhanes_merged_2017_2018.csv not found! Run download_merge.py first.")
        return
        
    print("Loading merged dataset...")
    df = pd.read_csv("nhanes_merged_2017_2018.csv")
    print(f"Shape: {df.shape}")
    
    outcomes = list(OUTCOME_NAMES.keys())
    markers = list(MARKER_NAMES.keys())
    
    # We want to test all unordered pairs of markers
    marker_pairs = list(combinations(markers, 2))
    anomaly_types = ["High-High", "Low-Low", "High-Low", "Low-High", "TwoSided-TwoSided"]
    
    print(f"Starting grid search: {len(outcomes)} outcomes, {len(marker_pairs)} marker pairs, {len(anomaly_types)} anomaly definitions.")
    print(f"Total combinations to test: {len(outcomes) * len(marker_pairs) * len(anomaly_types)}")
    
    all_results = []
    count = 0
    
    for outcome in outcomes:
        print(f"Screening outcome: {OUTCOME_NAMES[outcome]}...")
        for m1, m2 in marker_pairs:
            for anomaly in anomaly_types:
                res = analyze_pair(df, outcome, m1, m2, anomaly)
                if res is not None:
                    all_results.append(res)
                count += 1
                if count % 1000 == 0:
                    print(f" - Completed {count} tests...")
                    
    print(f"\nGrid search completed. Valid tests: {len(all_results)}")
    
    results_df = pd.DataFrame(all_results)
    
    # Save all valid results
    results_df.to_csv("xor_all_results.csv", index=False)
    print("Saved all valid results to xor_all_results.csv")
    
    # Now filter for true XOR candidates!
    # Criteria:
    # 1. Main effects individually increase risk: beta1 > 0 and beta2 > 0
    # 2. Significant negative interaction (canceling effect): beta3 < 0 and p3_interaction < 0.05
    # 3. Main effects are significant or at least moderately so (p1 < 0.05 and p2 < 0.05)
    # We will do a strict filtering first, and if too few, a slightly relaxed one.
    
    strict_mask = (
        (results_df["beta1"] > 0) & 
        (results_df["beta2"] > 0) & 
        (results_df["beta3"] < 0) & 
        (results_df["p3_interaction"] < 0.05) & 
        (results_df["p1_A"] < 0.05) & 
        (results_df["p2_B"] < 0.05)
    )
    
    strict_candidates = results_df[strict_mask].copy()
    print(f"Strict XOR candidates found: {len(strict_candidates)}")
    
    if len(strict_candidates) == 0:
        # Relaxed criteria: at least one main effect is p < 0.05 and the other is p < 0.15, interaction is p < 0.05
        relaxed_mask = (
            (results_df["beta1"] > 0) & 
            (results_df["beta2"] > 0) & 
            (results_df["beta3"] < 0) & 
            (results_df["p3_interaction"] < 0.05) & 
            (
                ((results_df["p1_A"] < 0.05) & (results_df["p2_B"] < 0.15)) |
                ((results_df["p1_A"] < 0.15) & (results_df["p2_B"] < 0.05))
            )
        )
        candidates = results_df[relaxed_mask].copy()
        print(f"Relaxed XOR candidates found: {len(candidates)}")
    else:
        candidates = strict_candidates
        
    # Sort candidates by xor_synergy descending (how large the canceling interaction is, balanced by a near-zero combined risk)
    # Let's also sort by p3_interaction ascending
    candidates = candidates.sort_values(by="p3_interaction", ascending=True)
    
    candidates.to_csv("xor_candidates.csv", index=False)
    print("Saved sorted XOR candidates to xor_candidates.csv")
    
    # Print top 15 candidates
    print("\nTop 15 XOR Candidates:")
    if not candidates.empty:
        print(candidates[["outcome_name", "marker1_name", "lbl_A", "marker2_name", "lbl_B", "or10", "or01", "or11", "p3_interaction"]].head(15).to_string())
    else:
        print("No candidates matched the XOR criteria.")

if __name__ == "__main__":
    main()
