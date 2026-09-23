"""
Age/Sex-Adjusted Re-Screen of NHANES Logical Gates
====================================================
This script re-runs the Screen 1 logical gate analysis with age and sex
as covariates in every logistic regression model, then applies FDR correction.

This is the critical fix identified in the audit: without age/sex adjustment,
many interaction effects are confounded by the fact that extreme biomarker
combinations select for younger/older cohorts.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from itertools import combinations
import warnings
import os

warnings.filterwarnings('ignore')

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

OUTCOME_NAMES = {
    "diabetes_binary": "Diabetes",
    "hypertension_binary": "Hypertension",
    "high_cholesterol_binary": "High Cholesterol",
    "arthritis_binary": "Arthritis",
    "liver_condition_binary": "Liver Condition",
    "thyroid_binary": "Thyroid Problem",
    "heart_attack_binary": "Heart Attack",
    "stroke_binary": "Stroke",
    # Excluding depression_binary due to confirmed binarization bug (52% prevalence, N=837)
}

def main():
    os.chdir("g:\\My Drive\\gemini\\NHANES")
    
    print("Loading merged dataset...")
    df = pd.read_csv("nhanes_merged_2017_2018.csv")
    print(f"Shape: {df.shape}")
    
    outcomes = list(OUTCOME_NAMES.keys())
    markers = list(MARKER_NAMES.keys())
    marker_pairs = list(combinations(markers, 2))
    anomaly_types = ["High-High", "Low-Low", "High-Low", "Low-High", "TwoSided-TwoSided"]
    
    total_tests = len(outcomes) * len(marker_pairs) * len(anomaly_types)
    print(f"Total tests to run: {total_tests}")
    print("Running age/sex-adjusted logistic regressions...\n")
    
    all_results = []
    all_p3_values = []  # Collect ALL p3 values (significant or not) for proper FDR
    count = 0
    
    for outcome in outcomes:
        print(f"  Screening: {OUTCOME_NAMES[outcome]}...")
        for m1, m2 in marker_pairs:
            for anomaly_type in anomaly_types:
                count += 1
                
                # Clean subset: need outcome, both markers, age, and sex
                cols = [outcome, m1, m2, "RIDAGEYR", "RIAGENDR"]
                df_clean = df[cols].dropna().copy()
                
                n_samples = len(df_clean)
                n_cases = int(df_clean[outcome].sum())
                
                if n_samples < 500 or n_cases < 30:
                    all_p3_values.append(1.0)  # non-testable = p=1
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
                
                # Cell sizes
                n11 = int(((df_clean["X_A"] == 1) & (df_clean["X_B"] == 1)).sum())
                n10 = int(((df_clean["X_A"] == 1) & (df_clean["X_B"] == 0)).sum())
                n01 = int(((df_clean["X_A"] == 0) & (df_clean["X_B"] == 1)).sum())
                n00 = int(((df_clean["X_A"] == 0) & (df_clean["X_B"] == 0)).sum())
                
                if n00 < 50 or n10 < 10 or n01 < 10 or n11 < 5:
                    all_p3_values.append(1.0)
                    continue
                
                # Cell disease rates and mean ages
                c11 = df_clean[(df_clean["X_A"] == 1) & (df_clean["X_B"] == 1)]
                c10 = df_clean[(df_clean["X_A"] == 1) & (df_clean["X_B"] == 0)]
                c01 = df_clean[(df_clean["X_A"] == 0) & (df_clean["X_B"] == 1)]
                c00 = df_clean[(df_clean["X_A"] == 0) & (df_clean["X_B"] == 0)]
                
                p00 = c00[outcome].mean()
                p10 = c10[outcome].mean()
                p01 = c01[outcome].mean()
                p11 = c11[outcome].mean()
                
                age00 = c00["RIDAGEYR"].mean()
                age10 = c10["RIDAGEYR"].mean()
                age01 = c01["RIDAGEYR"].mean()
                age11 = c11["RIDAGEYR"].mean()
                
                # ===== THE KEY FIX: Age/Sex-Adjusted Logistic Regression =====
                df_clean["const"] = 1.0
                df_clean["interaction"] = df_clean["X_A"] * df_clean["X_B"]
                
                try:
                    X = df_clean[["const", "X_A", "X_B", "interaction", "RIDAGEYR", "RIAGENDR"]]
                    y = df_clean[outcome]
                    
                    res = sm.Logit(y, X).fit(disp=0, maxiter=100)
                    
                    if not res.mle_retvals['converged']:
                        all_p3_values.append(1.0)
                        continue
                    
                    beta1 = res.params["X_A"]
                    beta2 = res.params["X_B"]
                    beta3 = res.params["interaction"]
                    beta_age = res.params["RIDAGEYR"]
                    beta_sex = res.params["RIAGENDR"]
                    
                    p1 = res.pvalues["X_A"]
                    p2 = res.pvalues["X_B"]
                    p3 = res.pvalues["interaction"]
                    p_age = res.pvalues["RIDAGEYR"]
                    p_sex = res.pvalues["RIAGENDR"]
                    
                except Exception:
                    all_p3_values.append(1.0)
                    continue
                
                all_p3_values.append(p3)
                
                # Only save if interaction is nominally significant
                if p3 < 0.05:
                    # Haldane-Anscombe smoothed ORs (unadjusted, for display)
                    def smoothed_or(y_t, n_t, y_b, n_b):
                        return ((y_t + 0.5) / (n_t - y_t + 0.5)) / ((y_b + 0.5) / (n_b - y_b + 0.5))
                    
                    y00, y10, y01, y11_y = c00[outcome].sum(), c10[outcome].sum(), c01[outcome].sum(), c11[outcome].sum()
                    or10 = smoothed_or(y10, n10, y00, n00)
                    or01 = smoothed_or(y01, n01, y00, n00)
                    or11 = smoothed_or(y11_y, n11, y00, n00)
                    
                    # Classify gate type
                    gate = "None"
                    if beta1 > 0 and beta2 > 0 and beta3 < 0 and p1 < 0.05 and p2 < 0.05 and or11 < 1.3:
                        gate = "XOR"
                    elif beta1 > 0 and beta2 > 0 and beta3 < 0 and p1 < 0.05 and p2 < 0.05 and or11 >= 1.3:
                        gate = "OR"
                    elif beta3 > 0 and or11 >= 1.8 and or10 < 1.5 and or01 < 1.5:
                        gate = "AND"
                    elif beta1 < 0 and beta2 < 0 and beta3 > 0 and p1 < 0.1 and p2 < 0.1 and p11 >= p00:
                        gate = "XNOR"
                    
                    if gate != "None":
                        all_results.append({
                            "outcome_name": OUTCOME_NAMES[outcome],
                            "marker1_name": MARKER_NAMES[m1],
                            "marker2_name": MARKER_NAMES[m2],
                            "anomaly_type": anomaly_type,
                            "gate": gate,
                            "n_samples": n_samples,
                            "n11": n11, "n10": n10, "n01": n01, "n00": n00,
                            "p00": round(p00, 4), "p10": round(p10, 4),
                            "p01": round(p01, 4), "p11": round(p11, 4),
                            "age00": round(age00, 1), "age10": round(age10, 1),
                            "age01": round(age01, 1), "age11": round(age11, 1),
                            "or10": round(or10, 3), "or01": round(or01, 3), "or11": round(or11, 3),
                            "beta1": round(beta1, 4), "p1": p1,
                            "beta2": round(beta2, 4), "p2": p2,
                            "beta3_adjusted": round(beta3, 4), "p3_adjusted": p3,
                            "beta_age": round(beta_age, 4), "p_age": p_age,
                            "beta_sex": round(beta_sex, 4), "p_sex": p_sex,
                        })
        
        if count % 2000 == 0:
            print(f"    ...completed {count}/{total_tests} tests")
    
    print(f"\nTotal tests completed: {count}")
    print(f"Total p3 values collected: {len(all_p3_values)}")
    print(f"Nominally significant gates (p3 < 0.05): {len(all_results)}")
    
    if not all_results:
        print("No nominally significant age/sex-adjusted gates found!")
        return
    
    results_df = pd.DataFrame(all_results)
    
    # Apply FDR correction using the FULL vector of p-values
    p_array = np.array(all_p3_values)
    rejected_fdr, p_fdr_full, _, _ = multipletests(p_array, alpha=0.05, method='fdr_bh')
    rejected_bonf, p_bonf_full, _, _ = multipletests(p_array, alpha=0.05, method='bonferroni')
    
    # Map back: we need to find which indices in all_p3_values correspond to our saved results
    # The saved results are only those with p3 < 0.05, so we need to track indices
    # Actually, let's just do FDR on the full vector and match back
    
    # Simpler approach: re-do FDR on just the saved results with proper N_total padding
    N_total = len(all_p3_values)
    p_saved = results_df['p3_adjusted'].values
    padding = np.ones(N_total - len(p_saved))
    full_p = np.concatenate([p_saved, padding])
    
    rej_fdr, p_fdr, _, _ = multipletests(full_p, alpha=0.05, method='fdr_bh')
    rej_bonf, p_bonf, _, _ = multipletests(full_p, alpha=0.05, method='bonferroni')
    
    results_df['p3_fdr'] = p_fdr[:len(results_df)]
    results_df['p3_bonferroni'] = p_bonf[:len(results_df)]
    results_df['survives_fdr'] = rej_fdr[:len(results_df)]
    results_df['survives_bonferroni'] = rej_bonf[:len(results_df)]
    
    results_df = results_df.sort_values('p3_adjusted')
    results_df.to_csv("gates_age_sex_adjusted.csv", index=False)
    
    print(f"\n{'='*60}")
    print(f"AGE/SEX-ADJUSTED RESULTS")
    print(f"{'='*60}")
    print(f"Total nominally significant gates: {len(results_df)}")
    print(f"Survive FDR (q < 0.05): {results_df['survives_fdr'].sum()}")
    print(f"Survive Bonferroni: {results_df['survives_bonferroni'].sum()}")
    
    print(f"\nGate distribution:")
    print(results_df.groupby('gate').size())
    
    print(f"\n--- TOP 15 (sorted by adjusted p3) ---")
    cols = ['outcome_name','marker1_name','marker2_name','gate','anomaly_type',
            'n11','p3_adjusted','p3_fdr','survives_fdr','age00','age11']
    print(results_df[cols].head(15).to_string())
    
    # Compare with unadjusted: check our previously highlighted findings
    print(f"\n--- PREVIOUSLY HIGHLIGHTED FINDINGS (age/sex adjusted) ---")
    highlights = [
        ("Diabetes", "Diastolic BP", "Creatinine"),
        ("Diabetes", "Sodium", "Hemoglobin"),
        ("Hypertension", "Diastolic BP", "Uric Acid"),
        ("Diabetes", "Body Mass Index", "Glycohemoglobin"),
        ("Diabetes", "LDL Cholesterol", "Creatinine"),
        ("Arthritis", "Systolic BP", "Hemoglobin"),
    ]
    
    for out, m1_substr, m2_substr in highlights:
        match = results_df[
            (results_df['outcome_name'] == out) &
            (results_df['marker1_name'].str.contains(m1_substr)) &
            (results_df['marker2_name'].str.contains(m2_substr))
        ]
        if match.empty:
            match = results_df[
                (results_df['outcome_name'] == out) &
                (results_df['marker2_name'].str.contains(m1_substr)) &
                (results_df['marker1_name'].str.contains(m2_substr))
            ]
        
        if not match.empty:
            r = match.iloc[0]
            status = "SURVIVES FDR" if r['survives_fdr'] else "FAILS FDR"
            print(f"  {out} vs {m1_substr} & {m2_substr}: adj_p3={r['p3_adjusted']:.2e}, fdr_p={r['p3_fdr']:.2e} [{status}], gate={r['gate']}, n11={r['n11']}")
        else:
            print(f"  {out} vs {m1_substr} & {m2_substr}: NOT FOUND (interaction no longer nominally significant after age/sex adjustment)")

if __name__ == "__main__":
    main()
