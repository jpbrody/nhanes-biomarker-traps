import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import pearsonr
import os

def Z_score(series):
    return (series - series.mean()) / (series.std() + 1e-9)

def fdr_correction(p_values):
    # Benjamini-Hochberg FDR correction
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]
    
    q_values = np.zeros(n)
    min_q = 1.0
    for i in range(n - 1, -1, -1):
        q = sorted_p[i] * n / (i + 1)
        min_q = min(min_q, q)
        q_values[sorted_indices[i]] = min_q
    return q_values

def main():
    os.chdir("g:\\My Drive\\gemini\\NHANES")
    
    if not os.path.exists("nhanes_depression_pooled.csv"):
        print("Error: nhanes_depression_pooled.csv not found! Run download_merge_pooled.py first.")
        return
        
    df = pd.read_csv("nhanes_depression_pooled.csv")
    print(f"Loaded pooled depression cohort. Shape: {df.shape}")
    print(f"Completed surveys: {len(df)}, Depressed cases: {(df['depression_binary'] == 1.0).sum()} ({ (df['depression_binary'] == 1.0).mean()*100:.2f}%)")
    
    biomarkers = [
        "BMXBMI", "BMXWAIST", "systolic_bp", "diastolic_bp", "LBXGH", "LBXGLU",
        "LBXIN", "LBXTC", "LBDHDD", "LBDLDL", "LBXTR", "LBXSATSI", "LBXSASSI",
        "LBXSAPSI", "LBXSCR", "LBXSBU", "LBXSUA", "LBXSNASI", "LBXSKSI", "LBXSCA",
        "LBXSAL", "LBXHSCRP", "LBXHGB", "LBXWBCSI", "LBXPLTSI"
    ]
    
    # Standardize biomarkers
    for b in biomarkers:
        df[f"Z_{b}"] = Z_score(df[b])
        
    # Baseline covariates
    df["const"] = 1.0
    covs = ["const", "RIDAGEYR", "RIAGENDR", "cycle"]
    
    # =========================================================================
    # SCREEN 1: Discrete Anomaly Gates
    # =========================================================================
    print("\nStarting Screen 1: Discrete Anomaly Gates (FDR adjusted)...")
    gate_results = []
    
    pairs = []
    for i in range(len(biomarkers)):
        for j in range(i + 1, len(biomarkers)):
            pairs.append((biomarkers[i], biomarkers[j]))
            
    # Evaluated anomaly configurations:
    # 1: Low-Low, 2: High-High, 3: Low-High, 4: High-Low, 5: TwoSided-TwoSided
    for b1, b2 in pairs:
        z1 = df[f"Z_{b1}"]
        z2 = df[f"Z_{b2}"]
        
        gates = {
            "Low-Low": ((z1 < -1.28), (z2 < -1.28)),
            "High-High": ((z1 > 1.28), (z2 > 1.28)),
            "Low-High": ((z1 < -1.28), (z2 > 1.28)),
            "High-Low": ((z1 > 1.28), (z2 < -1.28)),
            "TwoSided": ((z1.abs() > 1.28), (z2.abs() > 1.28))
        }
        
        for g_name, (x1, x2) in gates.items():
            df_g = df[['depression_binary', 'RIDAGEYR', 'RIAGENDR', 'cycle', 'const']].copy()
            df_g["X1"] = x1.astype(float)
            df_g["X2"] = x2.astype(float)
            df_g["X1_X2"] = (df_g["X1"] * df_g["X2"])
            
            df_g = df_g.dropna()
            if len(df_g) < 100:
                continue
                
            n_both = df_g["X1_X2"].sum()
            # We filter out tiny cells (require at least 15 subjects in the interaction cell)
            if n_both < 15:
                continue
                
            try:
                model = sm.Logit(df_g['depression_binary'], df_g[['const', 'X1', 'X2', 'X1_X2', 'RIDAGEYR', 'RIAGENDR', 'cycle']])
                res = model.fit(disp=0)
                
                # Check convergence
                if not np.isfinite(res.params['X1_X2']) or not np.isfinite(res.bse['X1_X2']):
                    continue
                    
                gate_results.append({
                    "Biomarker_A": b1,
                    "Biomarker_B": b2,
                    "Gate_Type": g_name,
                    "N_Both": int(n_both),
                    "beta_A": res.params['X1'],
                    "beta_B": res.params['X2'],
                    "beta_int": res.params['X1_X2'],
                    "z_int": res.tvalues['X1_X2'],
                    "p_int": res.pvalues['X1_X2'],
                    "dep_rate_both": df_g.loc[df_g["X1_X2"] == 1.0, "depression_binary"].mean(),
                    "dep_rate_control": df_g.loc[df_g["X1_X2"] == 0.0, "depression_binary"].mean()
                })
            except Exception:
                continue
                
    # FDR correction
    if gate_results:
        p_vals = [r["p_int"] for r in gate_results]
        q_vals = fdr_correction(p_vals)
        for idx, q in enumerate(q_vals):
            gate_results[idx]["FDR_q"] = q
            
        df_gates = pd.DataFrame(gate_results).sort_values("p_int")
        df_gates.to_csv("depression_gates.csv", index=False)
        print(f"Screen 1 Finished. Candidates evaluated: {len(gate_results)}")
        print(f"Top 5 Gates (Uncorrected p):")
        print(df_gates[['Biomarker_A', 'Biomarker_B', 'Gate_Type', 'N_Both', 'beta_int', 'p_int', 'FDR_q']].head(5).to_string(index=False))
        sig_gates = df_gates[df_gates["FDR_q"] < 0.05]
        print(f"Surviving FDR (q < 0.05): {len(sig_gates)}")
    else:
        print("Screen 1 Finished: 0 candidates completed.")

    # =========================================================================
    # SCREEN 2: Continuous Ratios & Differences
    # =========================================================================
    print("\nStarting Screen 2: Continuous Ratios & Differences...")
    ratio_results = []
    
    ordered_pairs = []
    for b1 in biomarkers:
        for b2 in biomarkers:
            if b1 != b2:
                ordered_pairs.append((b1, b2))
                
    for b1, b2 in ordered_pairs:
        # Create non-zero values for safe ratio
        offset1 = 1e-3 if (df[b1] == 0).any() else 0
        offset2 = 1e-3 if (df[b2] == 0).any() else 0
        
        ratio_raw = (df[b1] + offset1) / (df[b2] + offset2)
        diff_raw = df[b1] - df[b2]
        
        combinations = {
            "Ratio": ratio_raw,
            "Difference": diff_raw
        }
        
        for c_type, combo_series in combinations.items():
            df_c = df[['depression_binary', b1, b2, 'RIDAGEYR', 'RIAGENDR', 'cycle', 'const']].copy()
            df_c["Z_A"] = Z_score(df_c[b1])
            df_c["Z_B"] = Z_score(df_c[b2])
            df_c["Z_combo"] = Z_score(combo_series)
            
            df_c = df_c.dropna()
            if len(df_c) < 500:
                continue
                
            try:
                # Fit Individual and Combo Models
                res_A = sm.Logit(df_c['depression_binary'], df_c[['const', 'Z_A', 'RIDAGEYR', 'RIAGENDR', 'cycle']]).fit(disp=0)
                res_B = sm.Logit(df_c['depression_binary'], df_c[['const', 'Z_B', 'RIDAGEYR', 'RIAGENDR', 'cycle']]).fit(disp=0)
                res_Combo = sm.Logit(df_c['depression_binary'], df_c[['const', 'Z_combo', 'RIDAGEYR', 'RIAGENDR', 'cycle']]).fit(disp=0)
                
                p_A = res_A.pvalues['Z_A']
                p_B = res_B.pvalues['Z_B']
                p_Combo = res_Combo.pvalues['Z_combo']
                
                aic_A = res_A.aic
                aic_B = res_B.aic
                aic_Combo = res_Combo.aic
                
                # Check for superiority
                # Combo must be highly significant, outperform both components in p-value and lower AIC by >= 2.0
                if p_Combo < 0.05 and p_Combo < min(p_A, p_B) and aic_Combo < min(aic_A, aic_B) - 2.0:
                    ratio_results.append({
                        "Biomarker_A": b1,
                        "Biomarker_B": b2,
                        "Combo_Type": c_type,
                        "beta_A": res_A.params['Z_A'],
                        "p_A": p_A,
                        "beta_B": res_B.params['Z_B'],
                        "p_B": p_B,
                        "beta_combo": res_Combo.params['Z_combo'],
                        "p_combo": p_Combo,
                        "aic_gain": min(aic_A, aic_B) - aic_Combo
                    })
            except Exception:
                continue
                
    if ratio_results:
        p_vals_r = [r["p_combo"] for r in ratio_results]
        q_vals_r = fdr_correction(p_vals_r)
        for idx, q in enumerate(q_vals_r):
            ratio_results[idx]["FDR_q"] = q
            
        df_ratios = pd.DataFrame(ratio_results).sort_values("p_combo")
        df_ratios.to_csv("depression_ratios.csv", index=False)
        print(f"Screen 2 Finished. Superior candidates discovered: {len(ratio_results)}")
        print(f"Top 5 Ratios/Differences (Uncorrected p):")
        print(df_ratios[['Biomarker_A', 'Biomarker_B', 'Combo_Type', 'beta_combo', 'p_combo', 'FDR_q', 'aic_gain']].head(5).to_string(index=False))
        sig_ratios = df_ratios[df_ratios["FDR_q"] < 0.05]
        print(f"Surviving FDR (q < 0.05): {len(sig_ratios)}")
    else:
        print("Screen 2 Finished: 0 superior combinations discovered.")

    # =========================================================================
    # SCREEN 3: Discordance Residual Deviations
    # =========================================================================
    print("\nStarting Screen 3: Discordance Residual Deviations...")
    residual_results = []
    
    # Find strongly correlated pairs (|r| >= 0.30)
    correlated_pairs = []
    for i in range(len(biomarkers)):
        for j in range(i + 1, len(biomarkers)):
            b1 = biomarkers[i]
            b2 = biomarkers[j]
            df_sub = df[[b1, b2]].dropna()
            if len(df_sub) > 500:
                r_val, _ = pearsonr(df_sub[b1], df_sub[b2])
                if abs(r_val) >= 0.30:
                    correlated_pairs.append((b1, b2, r_val))
                    correlated_pairs.append((b2, b1, r_val)) # check both directions
                    
    print(f"Found {len(correlated_pairs)} strongly correlated biomarker directions (|r| >= 0.30).")
    
    for b1, b2, r_val in correlated_pairs:
        df_res = df[['depression_binary', b1, b2, 'RIDAGEYR', 'RIAGENDR', 'cycle', 'const']].dropna().copy()
        if len(df_res) < 500:
            continue
            
        try:
            # Fit structural baseline OLS
            ols_model = sm.OLS(df_res[b2], df_res[['const', b1, 'RIDAGEYR', 'RIAGENDR', 'cycle']])
            ols_res = ols_model.fit()
            
            df_res["residual"] = ols_res.resid
            df_res["Z_e"] = Z_score(df_res["residual"])
            df_res["abs_Z_e"] = df_res["Z_e"].abs()
            
            # Fit Directional Logit
            res_dir = sm.Logit(df_res['depression_binary'], df_res[['const', 'Z_e', 'RIDAGEYR', 'RIAGENDR', 'cycle']]).fit(disp=0)
            residual_results.append({
                "Biomarker_A": b1,
                "Biomarker_B": b2,
                "r_correlation": r_val,
                "Residual_Type": "Directional",
                "beta_e": res_dir.params['Z_e'],
                "z_stat": res_dir.tvalues['Z_e'],
                "p_value": res_dir.pvalues['Z_e'],
                "ols_slope_A": ols_res.params[b1]
            })
            
            # Fit Absolute Logit
            res_abs = sm.Logit(df_res['depression_binary'], df_res[['const', 'abs_Z_e', 'RIDAGEYR', 'RIAGENDR', 'cycle']]).fit(disp=0)
            residual_results.append({
                "Biomarker_A": b1,
                "Biomarker_B": b2,
                "r_correlation": r_val,
                "Residual_Type": "Absolute (Magnitude)",
                "beta_e": res_abs.params['abs_Z_e'],
                "z_stat": res_abs.tvalues['abs_Z_e'],
                "p_value": res_abs.pvalues['abs_Z_e'],
                "ols_slope_A": ols_res.params[b1]
            })
        except Exception:
            continue
            
    if residual_results:
        p_vals_res = [r["p_value"] for r in residual_results]
        q_vals_res = fdr_correction(p_vals_res)
        for idx, q in enumerate(q_vals_res):
            residual_results[idx]["FDR_q"] = q
            
        df_residuals = pd.DataFrame(residual_results).sort_values("p_value")
        df_residuals.to_csv("depression_residuals.csv", index=False)
        print(f"Screen 3 Finished. Residual models evaluated: {len(residual_results)}")
        print(f"Top 5 Residual Models (Uncorrected p):")
        print(df_residuals[['Biomarker_A', 'Biomarker_B', 'r_correlation', 'Residual_Type', 'beta_e', 'p_value', 'FDR_q']].head(5).to_string(index=False))
        sig_residuals = df_residuals[df_residuals["FDR_q"] < 0.05]
        print(f"Surviving FDR (q < 0.05): {len(sig_residuals)}")
    else:
        print("Screen 3 Finished: 0 residual models evaluated.")

if __name__ == "__main__":
    main()
