import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from itertools import combinations
import warnings

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
}

def Z_score(series):
    return (series - series.mean()) / (series.std() + 1e-9)

def main():
    os.chdir("g:\\My Drive\\gemini\\NHANES")
    df = pd.read_csv("nhanes_merged_2017_2018.csv")
    
    outcomes = list(OUTCOME_NAMES.keys())
    markers = list(MARKER_NAMES.keys())
    
    results = []
    total_tests = 0
    
    print("Starting age/sex-adjusted continuous re-screen...")
    for outcome in outcomes:
        print(f"  Screening: {OUTCOME_NAMES[outcome]}...")
        for m1 in markers:
            for m2 in markers:
                if m1 == m2:
                    continue
                
                # Clean subset
                df_clean = df[[outcome, m1, m2, 'RIDAGEYR', 'RIAGENDR']].dropna().copy()
                df_clean = df_clean[df_clean[m2] != 0]
                df_clean = df_clean[np.isfinite(df_clean[m1] / df_clean[m2])]
                
                n_samples = len(df_clean)
                n_cases = df_clean[outcome].sum()
                
                if n_samples < 500 or n_cases < 30:
                    continue
                
                # Compute ratio and difference
                df_clean["ratio"] = df_clean[m1] / df_clean[m2]
                df_clean["diff"] = df_clean[m1] - df_clean[m2]
                
                # Z-score variables
                df_clean["Z_A"] = Z_score(df_clean[m1])
                df_clean["Z_B"] = Z_score(df_clean[m2])
                df_clean["Z_R"] = Z_score(df_clean["ratio"])
                df_clean["Z_D"] = Z_score(df_clean["diff"])
                df_clean["const"] = 1.0
                
                y = df_clean[outcome]
                covs = df_clean[['const', 'RIDAGEYR', 'RIAGENDR']]
                
                # Models
                try:
                    # Model A (Constituent 1)
                    X_A = pd.concat([df_clean['Z_A'], covs], axis=1)
                    res_A = sm.Logit(y, X_A).fit(disp=0)
                    p_A = res_A.pvalues['Z_A']
                    aic_A = res_A.aic
                    beta_A = res_A.params['Z_A']
                    
                    # Model B (Constituent 2)
                    X_B = pd.concat([df_clean['Z_B'], covs], axis=1)
                    res_B = sm.Logit(y, X_B).fit(disp=0)
                    p_B = res_B.pvalues['Z_B']
                    aic_B = res_B.aic
                    beta_B = res_B.params['Z_B']
                    
                    # Model Ratio
                    X_R = pd.concat([df_clean['Z_R'], covs], axis=1)
                    res_R = sm.Logit(y, X_R).fit(disp=0)
                    p_R = res_R.pvalues['Z_R']
                    aic_R = res_R.aic
                    beta_R = res_R.params['Z_R']
                    
                    # Model Difference
                    X_D = pd.concat([df_clean['Z_D'], covs], axis=1)
                    res_D = sm.Logit(y, X_D).fit(disp=0)
                    p_D = res_D.pvalues['Z_D']
                    aic_D = res_D.aic
                    beta_D = res_D.params['Z_D']
                    
                except:
                    continue
                
                total_tests += 2 # one for ratio, one for difference
                
                # Check Ratio Superiority
                if p_R < 0.05 and p_R < p_A and p_R < p_B and aic_R < aic_A and aic_R < aic_B:
                    results.append({
                        "outcome_name": OUTCOME_NAMES[outcome],
                        "marker1_name": MARKER_NAMES[m1],
                        "marker2_name": MARKER_NAMES[m2],
                        "combination_type": "Ratio",
                        "beta_comp": beta_R, "p_comp": p_R, "aic_comp": aic_R,
                        "beta_A": beta_A, "p_A": p_A, "aic_A": aic_A,
                        "beta_B": beta_B, "p_B": p_B, "aic_B": aic_B,
                        "aic_gain": min(aic_A, aic_B) - aic_R
                    })
                
                # Check Difference Superiority
                if p_D < 0.05 and p_D < p_A and p_D < p_B and aic_D < aic_A and aic_D < aic_B:
                    results.append({
                        "outcome_name": OUTCOME_NAMES[outcome],
                        "marker1_name": MARKER_NAMES[m1],
                        "marker2_name": MARKER_NAMES[m2],
                        "combination_type": "Difference",
                        "beta_comp": beta_D, "p_comp": p_D, "aic_comp": aic_D,
                        "beta_A": beta_A, "p_A": p_A, "aic_A": aic_A,
                        "beta_B": beta_B, "p_B": p_B, "aic_B": aic_B,
                        "aic_gain": min(aic_A, aic_B) - aic_D
                    })
                    
    print(f"\nCompleted re-screening. Total test channels: {total_tests}")
    print(f"Nominally superior combinations: {len(results)}")
    
    if not results:
        print("No nominally superior combinations found after age/sex adjustment!")
        return
        
    results_df = pd.DataFrame(results)
    
    # FDR and Bonferroni correction
    p_saved = results_df['p_comp'].values
    padding = np.ones(total_tests - len(p_saved))
    full_p = np.concatenate([p_saved, padding])
    
    rej_fdr, p_fdr, _, _ = multipletests(full_p, alpha=0.05, method='fdr_bh')
    rej_bonf, p_bonf, _, _ = multipletests(full_p, alpha=0.05, method='bonferroni')
    
    results_df['p_comp_fdr'] = p_fdr[:len(results_df)]
    results_df['p_comp_bonferroni'] = p_bonf[:len(results_df)]
    results_df['survives_fdr'] = rej_fdr[:len(results_df)]
    results_df['survives_bonferroni'] = rej_bonf[:len(results_df)]
    
    results_df = results_df.sort_values(by="aic_gain", ascending=False)
    results_df.to_csv("ratios_differences_adjusted.csv", index=False)
    
    print(f"\nAdjusted screen results:")
    print(f"  Survive FDR (q < 0.05): {results_df['survives_fdr'].sum()}")
    print(f"  Survive Bonferroni: {results_df['survives_bonferroni'].sum()}")
    
    print("\nTop 15 Adjusted Superior Combinations:")
    cols = ['outcome_name', 'marker1_name', 'marker2_name', 'combination_type', 'beta_comp', 'p_comp', 'p_comp_fdr', 'aic_gain']
    print(results_df[cols].head(15).to_string())

if __name__ == "__main__":
    main()
