import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
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
    
    # 1. Pearson correlations
    corr_matrix = df[markers].corr()
    correlated_pairs = []
    for i in range(len(markers)):
        for j in range(i+1, len(markers)):
            m1, m2 = markers[i], markers[j]
            r = corr_matrix.loc[m1, m2]
            if abs(r) >= 0.40:
                correlated_pairs.append((m1, m2, r))
                
    results = []
    total_tests = 0
    
    print(f"Starting age/sex-adjusted residual re-screen on {len(correlated_pairs)} pairs...")
    
    for outcome in outcomes:
        for m1, m2, r in correlated_pairs:
            # Clean subset
            df_clean = df[[outcome, m1, m2, 'RIDAGEYR', 'RIAGENDR']].dropna().copy()
            n_samples = len(df_clean)
            n_cases = df_clean[outcome].sum()
            
            if n_samples < 500 or n_cases < 30:
                continue
            
            # ===== KEY ADJUSTMENT FIX: Fit OLS adjusted for Age and Sex =====
            # Waist size (or B) predicted by BMI (or A) + Age + Sex
            df_clean["const"] = 1.0
            X_ols = df_clean[["const", m1, "RIDAGEYR", "RIAGENDR"]]
            y_ols = df_clean[m2]
            
            try:
                ols_res = sm.OLS(y_ols, X_ols).fit()
                df_clean["residual"] = ols_res.resid
                
                df_clean["Z_e"] = Z_score(df_clean["residual"])
                df_clean["abs_Z_e"] = df_clean["Z_e"].abs()
                
                y = df_clean[outcome]
                covs = df_clean[['const', 'RIDAGEYR', 'RIAGENDR']]
                
                # Fit directional residual model
                X_dir = pd.concat([df_clean['Z_e'], covs], axis=1)
                res_dir = sm.Logit(y, X_dir).fit(disp=0)
                beta_dir = res_dir.params['Z_e']
                p_dir = res_dir.pvalues['Z_e']
                
                # Fit absolute residual model
                X_abs = pd.concat([df_clean['abs_Z_e'], covs], axis=1)
                res_abs = sm.Logit(y, X_abs).fit(disp=0)
                beta_abs = res_abs.params['abs_Z_e']
                p_abs = res_abs.pvalues['abs_Z_e']
                
            except:
                continue
                
            total_tests += 2 # one for directional, one for absolute
            
            if p_dir < 0.05 or p_abs < 0.05:
                results.append({
                    "outcome_name": OUTCOME_NAMES[outcome],
                    "marker1_name": MARKER_NAMES[m1],
                    "marker2_name": MARKER_NAMES[m2],
                    "correlation_r": r,
                    "beta_dir": beta_dir, "p_dir": p_dir,
                    "beta_abs": beta_abs, "p_abs": p_abs,
                    "best_p": min(p_dir, p_abs),
                    "is_directional": 1 if p_dir < p_abs else 0
                })
                
    print(f"\nCompleted re-screening. Total test channels: {total_tests}")
    print(f"Nominally significant residual discordances: {len(results)}")
    
    if not results:
        print("No nominally significant discordances found after age/sex adjustment!")
        return
        
    results_df = pd.DataFrame(results)
    
    # FDR and Bonferroni correction
    p_saved = results_df['best_p'].values
    padding = np.ones(total_tests - len(p_saved))
    full_p = np.concatenate([p_saved, padding])
    
    rej_fdr, p_fdr, _, _ = multipletests(full_p, alpha=0.05, method='fdr_bh')
    rej_bonf, p_bonf, _, _ = multipletests(full_p, alpha=0.05, method='bonferroni')
    
    results_df['p_fdr'] = p_fdr[:len(results_df)]
    results_df['p_bonferroni'] = p_bonf[:len(results_df)]
    results_df['survives_fdr'] = rej_fdr[:len(results_df)]
    results_df['survives_bonferroni'] = rej_bonf[:len(results_df)]
    
    results_df = results_df.sort_values(by="best_p")
    results_df.to_csv("discordance_adjusted.csv", index=False)
    
    print(f"\nAdjusted residual results:")
    print(f"  Survive FDR (q < 0.05): {results_df['survives_fdr'].sum()}")
    print(f"  Survive Bonferroni: {results_df['survives_bonferroni'].sum()}")
    
    print("\nTop 15 Adjusted Residual Discordances:")
    cols = ['outcome_name', 'marker1_name', 'marker2_name', 'correlation_r', 'is_directional', 'best_p', 'p_fdr', 'survives_bonferroni']
    print(results_df[cols].head(15).to_string())

if __name__ == "__main__":
    main()
