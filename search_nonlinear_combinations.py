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

def Z_score(series):
    return (series - series.mean()) / (series.std() + 1e-9)

def run_logical_gates_screen(df, outcomes, markers):
    print("\n==================================================")
    print("STARTING SCREEN 1: LOGICAL GATES (AND, OR, XOR, XNOR)")
    print("==================================================")
    
    marker_pairs = list(combinations(markers, 2))
    anomaly_types = ["High-High", "Low-Low", "High-Low", "Low-High", "TwoSided-TwoSided"]
    
    results = []
    
    for outcome in outcomes:
        for m1, m2 in marker_pairs:
            for anomaly_type in anomaly_types:
                # Clean subset
                df_clean = df[[outcome, m1, m2]].dropna().copy()
                n_samples = len(df_clean)
                n_cases = df_clean[outcome].sum()
                
                if n_samples < 500 or n_cases < 30:
                    continue
                    
                # Compute thresholds
                p10_1, p90_1 = df_clean[m1].quantile(0.1), df_clean[m1].quantile(0.9)
                p10_2, p90_2 = df_clean[m2].quantile(0.1), df_clean[m2].quantile(0.9)
                
                # Define anomaly flags
                if anomaly_type == "High-High":
                    df_clean["X_A"] = (df_clean[m1] > p90_1).astype(float)
                    df_clean["X_B"] = (df_clean[m2] > p90_2).astype(float)
                elif anomaly_type == "Low-Low":
                    df_clean["X_A"] = (df_clean[m1] < p10_1).astype(float)
                    df_clean["X_B"] = (df_clean[m2] < p10_2).astype(float)
                elif anomaly_type == "High-Low":
                    df_clean["X_A"] = (df_clean[m1] > p90_1).astype(float)
                    df_clean["X_B"] = (df_clean[m2] < p10_2).astype(float)
                elif anomaly_type == "Low-High":
                    df_clean["X_A"] = (df_clean[m1] < p10_1).astype(float)
                    df_clean["X_B"] = (df_clean[m2] > p90_2).astype(float)
                elif anomaly_type == "TwoSided-TwoSided":
                    df_clean["X_A"] = ((df_clean[m1] < p10_1) | (df_clean[m1] > p90_1)).astype(float)
                    df_clean["X_B"] = ((df_clean[m2] < p10_2) | (df_clean[m2] > p90_2)).astype(float)
                    
                # Count cell sizes
                c00 = df_clean[(df_clean["X_A"] == 0) & (df_clean["X_B"] == 0)]
                c10 = df_clean[(df_clean["X_A"] == 1) & (df_clean["X_B"] == 0)]
                c01 = df_clean[(df_clean["X_A"] == 0) & (df_clean["X_B"] == 1)]
                c11 = df_clean[(df_clean["X_A"] == 1) & (df_clean["X_B"] == 1)]
                
                n00, y00 = len(c00), c00[outcome].sum()
                n10, y10 = len(c10), c10[outcome].sum()
                n01, y01 = len(c01), c01[outcome].sum()
                n11, y11 = len(c11), c11[outcome].sum()
                
                if n00 < 50 or n10 < 10 or n01 < 10 or n11 < 5:
                    continue
                    
                p00 = y00 / n00
                p10 = y10 / n10
                p01 = y01 / n01
                p11 = y11 / n11
                
                # Haldane-Anscombe odds ratios
                def smoothed_or(y_t, n_t, y_b, n_b):
                    return ((y_t + 0.5) / (n_t - y_t + 0.5)) / ((y_b + 0.5) / (n_b - y_b + 0.5))
                    
                or10 = smoothed_or(y10, n10, y00, n00)
                or01 = smoothed_or(y01, n01, y00, n00)
                or11 = smoothed_or(y11, n11, y00, n00)
                
                try:
                    df_clean["const"] = 1.0
                    df_clean["interaction"] = df_clean["X_A"] * df_clean["X_B"]
                    X = df_clean[["const", "X_A", "X_B", "interaction"]]
                    y = df_clean[outcome]
                    
                    res = sm.Logit(y, X).fit(disp=0)
                    beta1, beta2, beta3 = res.params["X_A"], res.params["X_B"], res.params["interaction"]
                    p1, p2, p3 = res.pvalues["X_A"], res.pvalues["X_B"], res.pvalues["interaction"]
                except:
                    continue
                    
                # Classify logical gate
                gate = "None"
                
                # XOR Criteria: Main effects positive, interaction negative, joint returns to baseline
                if beta1 > 0 and beta2 > 0 and beta3 < 0 and p3 < 0.05 and p1 < 0.05 and p2 < 0.05 and or11 < 1.3:
                    gate = "XOR"
                # OR Criteria: Main effects positive, interaction negative, joint plateaus
                elif beta1 > 0 and beta2 > 0 and beta3 < 0 and p3 < 0.05 and p1 < 0.05 and p2 < 0.05 and or11 >= 1.3 and or11 <= 1.25 * max(or10, or01):
                    gate = "OR"
                # AND Criteria: Joint elevated, individual main effects not strongly elevated, positive interaction
                elif beta3 > 0 and p3 < 0.05 and or11 >= 1.8 and or10 < 1.5 and or01 < 1.5:
                    gate = "AND"
                # XNOR Criteria: Main effects negative, interaction positive, joint returns to baseline
                elif beta1 < 0 and beta2 < 0 and beta3 > 0 and p3 < 0.05 and p1 < 0.1 and p2 < 0.1 and p11 >= p00:
                    gate = "XNOR"
                    
                if gate != "None":
                    results.append({
                        "outcome_var": outcome,
                        "outcome_name": OUTCOME_NAMES[outcome],
                        "marker1_var": m1,
                        "marker1_name": MARKER_NAMES[m1],
                        "marker2_var": m2,
                        "marker2_name": MARKER_NAMES[m2],
                        "anomaly_type": anomaly_type,
                        "gate": gate,
                        "n_samples": n_samples,
                        "n_cases": n_cases,
                        "p00": p00, "p10": p10, "p01": p01, "p11": p11,
                        "or10": or10, "or01": or01, "or11": or11,
                        "beta1": beta1, "p1": p1,
                        "beta2": beta2, "p2": p2,
                        "beta3": beta3, "p3": p3
                    })
                    
    results_df = pd.DataFrame(results)
    results_df.to_csv("logical_gates_candidates.csv", index=False)
    print(f"Screen 1 completed. Saved {len(results_df)} candidates to logical_gates_candidates.csv")
    if not results_df.empty:
        print(results_df.groupby("gate").size())
    return results_df

def run_ratios_differences_screen(df, outcomes, markers):
    print("\n==================================================")
    print("STARTING SCREEN 2: ALGEBRAIC RATIOS & DIFFERENCES")
    print("==================================================")
    
    results = []
    
    # We test ordered pairs because R = A/B is asymmetric
    for outcome in outcomes:
        for m1 in markers:
            for m2 in markers:
                if m1 == m2:
                    continue
                    
                # Clean subset
                # Drop rows with division-by-zero or non-finite values in ratio
                df_clean = df[[outcome, m1, m2]].dropna().copy()
                df_clean = df_clean[df_clean[m2] != 0]
                df_clean = df_clean[np.isfinite(df_clean[m1] / df_clean[m2])]
                
                n_samples = len(df_clean)
                n_cases = df_clean[outcome].sum()
                
                if n_samples < 500 or n_cases < 30:
                    continue
                    
                # Compute continuous ratio and difference
                df_clean["ratio"] = df_clean[m1] / df_clean[m2]
                df_clean["diff"] = df_clean[m1] - df_clean[m2]
                
                # Z-score variables
                df_clean["Z_A"] = Z_score(df_clean[m1])
                df_clean["Z_B"] = Z_score(df_clean[m2])
                df_clean["Z_R"] = Z_score(df_clean["ratio"])
                df_clean["Z_D"] = Z_score(df_clean["diff"])
                
                df_clean["const"] = 1.0
                y = df_clean[outcome]
                
                # Fit 4 independent models
                try:
                    # Model A
                    res_A = sm.Logit(y, df_clean[["const", "Z_A"]]).fit(disp=0)
                    p_A = res_A.pvalues["Z_A"]
                    aic_A = res_A.aic
                    beta_A = res_A.params["Z_A"]
                    
                    # Model B
                    res_B = sm.Logit(y, df_clean[["const", "Z_B"]]).fit(disp=0)
                    p_B = res_B.pvalues["Z_B"]
                    aic_B = res_B.aic
                    beta_B = res_B.params["Z_B"]
                    
                    # Model Ratio
                    res_R = sm.Logit(y, df_clean[["const", "Z_R"]]).fit(disp=0)
                    p_R = res_R.pvalues["Z_R"]
                    aic_R = res_R.aic
                    beta_R = res_R.params["Z_R"]
                    
                    # Model Difference
                    res_D = sm.Logit(y, df_clean[["const", "Z_D"]]).fit(disp=0)
                    p_D = res_D.pvalues["Z_D"]
                    aic_D = res_D.aic
                    beta_D = res_D.params["Z_D"]
                except:
                    continue
                    
                # Check if Ratio is superior: p_R < 0.05 and p_R is more significant than both p_A, p_B, and AIC is lower
                if p_R < 0.05 and p_R < p_A and p_R < p_B and aic_R < aic_A and aic_R < aic_B:
                    results.append({
                        "outcome_var": outcome,
                        "outcome_name": OUTCOME_NAMES[outcome],
                        "marker1_var": m1, "marker1_name": MARKER_NAMES[m1],
                        "marker2_var": m2, "marker2_name": MARKER_NAMES[m2],
                        "combination_type": "Ratio",
                        "n_samples": n_samples,
                        "n_cases": n_cases,
                        "beta_comp": beta_R, "p_comp": p_R, "aic_comp": aic_R,
                        "beta_A": beta_A, "p_A": p_A, "aic_A": aic_A,
                        "beta_B": beta_B, "p_B": p_B, "aic_B": aic_B,
                        "p_gain": min(p_A, p_B) / (p_R + 1e-300),
                        "aic_gain": min(aic_A, aic_B) - aic_R
                    })
                    
                # Check if Difference is superior
                if p_D < 0.05 and p_D < p_A and p_D < p_B and aic_D < aic_A and aic_D < aic_B:
                    results.append({
                        "outcome_var": outcome,
                        "outcome_name": OUTCOME_NAMES[outcome],
                        "marker1_var": m1, "marker1_name": MARKER_NAMES[m1],
                        "marker2_var": m2, "marker2_name": MARKER_NAMES[m2],
                        "combination_type": "Difference",
                        "n_samples": n_samples,
                        "n_cases": n_cases,
                        "beta_comp": beta_D, "p_comp": p_D, "aic_comp": aic_D,
                        "beta_A": beta_A, "p_A": p_A, "aic_A": aic_A,
                        "beta_B": beta_B, "p_B": p_B, "aic_B": aic_B,
                        "p_gain": min(p_A, p_B) / (p_D + 1e-300),
                        "aic_gain": min(aic_A, aic_B) - aic_D
                    })
                    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="aic_gain", ascending=False)
    results_df.to_csv("ratio_difference_candidates.csv", index=False)
    print(f"Screen 2 completed. Saved {len(results_df)} superior ratios/differences to ratio_difference_candidates.csv")
    if not results_df.empty:
        print(results_df.groupby("combination_type").size())
    return results_df

def run_discordance_screen(df, outcomes, markers):
    print("\n==================================================")
    print("STARTING SCREEN 3: DISCORDANCE RESIDUAL DEVIATION")
    print("==================================================")
    
    # 1. Compute correlation matrix of markers
    corr_matrix = df[markers].corr()
    
    # 2. Extract highly correlated pairs (|r| >= 0.40)
    correlated_pairs = []
    for i in range(len(markers)):
        for j in range(i+1, len(markers)):
            m1, m2 = markers[i], markers[j]
            r = corr_matrix.loc[m1, m2]
            if abs(r) >= 0.40:
                correlated_pairs.append((m1, m2, r))
                
    print(f"Found {len(correlated_pairs)} highly correlated pairs (|r| >= 0.40)")
    
    results = []
    
    # 3. For each pair and outcome, calculate OLS residuals and fit logistic regression
    for outcome in outcomes:
        for m1, m2, r in correlated_pairs:
            # Clean subset
            df_clean = df[[outcome, m1, m2]].dropna().copy()
            n_samples = len(df_clean)
            n_cases = df_clean[outcome].sum()
            
            if n_samples < 500 or n_cases < 30:
                continue
                
            # Fit OLS: B = theta0 + theta1 * A + e
            try:
                # Add constant for OLS
                df_clean["const"] = 1.0
                X_ols = df_clean[["const", m1]]
                y_ols = df_clean[m2]
                
                ols_res = sm.OLS(y_ols, X_ols).fit()
                df_clean["residual"] = ols_res.resid
                
                # Z-score the residual
                df_clean["Z_e"] = Z_score(df_clean["residual"])
                df_clean["abs_Z_e"] = df_clean["Z_e"].abs()
                
                y = df_clean[outcome]
                
                # Fit directional discordance: logit(Y) = beta0 + beta1 * Z_e
                res_dir = sm.Logit(y, df_clean[["const", "Z_e"]]).fit(disp=0)
                beta_dir = res_dir.params["Z_e"]
                p_dir = res_dir.pvalues["Z_e"]
                aic_dir = res_dir.aic
                
                # Fit absolute discordance: logit(Y) = beta0 + beta2 * |Z_e|
                res_abs = sm.Logit(y, df_clean[["const", "abs_Z_e"]]).fit(disp=0)
                beta_abs = res_abs.params["abs_Z_e"]
                p_abs = res_abs.pvalues["abs_Z_e"]
                aic_abs = res_abs.aic
                
            except:
                continue
                
            if p_dir < 0.05 or p_abs < 0.05:
                results.append({
                    "outcome_var": outcome,
                    "outcome_name": OUTCOME_NAMES[outcome],
                    "marker1_var": m1, "marker1_name": MARKER_NAMES[m1],
                    "marker2_var": m2, "marker2_name": MARKER_NAMES[m2],
                    "correlation_r": r,
                    "n_samples": n_samples,
                    "n_cases": n_cases,
                    "beta_dir": beta_dir, "p_dir": p_dir, "aic_dir": aic_dir,
                    "beta_abs": beta_abs, "p_abs": p_abs, "aic_abs": aic_abs,
                    "best_p": min(p_dir, p_abs),
                    "is_directional": 1 if p_dir < p_abs else 0
                })
                
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="best_p", ascending=True)
    results_df.to_csv("discordance_candidates.csv", index=False)
    print(f"Screen 3 completed. Saved {len(results_df)} discordance pathologies to discordance_candidates.csv")
    return results_df

def main():
    if not os.path.exists("nhanes_merged_2017_2018.csv"):
        print("Error: nhanes_merged_2017_2018.csv not found! Run download_merge.py first.")
        return
        
    print("Loading merged dataset...")
    df = pd.read_csv("nhanes_merged_2017_2018.csv")
    print(f"Shape: {df.shape}")
    
    outcomes = list(OUTCOME_NAMES.keys())
    markers = list(MARKER_NAMES.keys())
    
    # Run all three Phase 2 advanced screens!
    run_logical_gates_screen(df, outcomes, markers)
    run_ratios_differences_screen(df, outcomes, markers)
    run_discordance_screen(df, outcomes, markers)
    
    print("\nAll advanced Phase 2 screening pipelines completed successfully!")

if __name__ == "__main__":
    main()
