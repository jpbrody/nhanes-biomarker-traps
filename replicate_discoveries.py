import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

def Z_score(series):
    return (series - series.mean()) / (series.std() + 1e-9)

def main():
    os.chdir("g:\\My Drive\\gemini\\NHANES")
    
    if not os.path.exists("nhanes_merged_2015_2016.csv"):
        print("Error: nhanes_merged_2015_2016.csv not found! Run download_merge_2015_2016.py first.")
        return
        
    df = pd.read_csv("nhanes_merged_2015_2016.csv")
    print(f"Loaded replication cohort (2015-2016). Shape: {df.shape}")
    
    print("\n==============================================================")
    print("REPLICATION TEST 1: Sodium / HbA1c Biophysical Ratio (Diabetes)")
    print("==============================================================")
    df_r = df[['diabetes_binary', 'LBXSNASI', 'LBXGH', 'RIDAGEYR', 'RIAGENDR']].dropna().copy()
    df_r = df_r[df_r['LBXGH'] > 0]
    df_r["ratio"] = df_r['LBXSNASI'] / df_r['LBXGH']
    df_r["Z_A"] = Z_score(df_r['LBXSNASI'])
    df_r["Z_B"] = Z_score(df_r['LBXGH'])
    df_r["Z_R"] = Z_score(df_r['ratio'])
    df_r["const"] = 1.0
    
    y = df_r['diabetes_binary']
    covs = df_r[['const', 'RIDAGEYR', 'RIAGENDR']]
    
    res_A = sm.Logit(y, pd.concat([df_r['Z_A'], covs], axis=1)).fit(disp=0)
    res_B = sm.Logit(y, pd.concat([df_r['Z_B'], covs], axis=1)).fit(disp=0)
    res_R = sm.Logit(y, pd.concat([df_r['Z_R'], covs], axis=1)).fit(disp=0)
    
    aic_gain = min(res_A.aic, res_B.aic) - res_R.aic
    replicates_1 = res_R.pvalues['Z_R'] < 0.05 and res_R.pvalues['Z_R'] < min(res_A.pvalues['Z_A'], res_B.pvalues['Z_B']) and res_R.aic < min(res_A.aic, res_B.aic)
    
    print(f"Sodium model:  beta = {res_A.params['Z_A']:.4f}, p = {res_A.pvalues['Z_A']:.2e}, AIC = {res_A.aic:.2f}")
    print(f"HbA1c model:   beta = {res_B.params['Z_B']:.4f}, p = {res_B.pvalues['Z_B']:.2e}, AIC = {res_B.aic:.2f}")
    print(f"Ratio model:   beta = {res_R.params['Z_R']:.4f}, p = {res_R.pvalues['Z_R']:.2e}, AIC = {res_R.aic:.2f}")
    print(f"AIC Gain:      {aic_gain:.2f}")
    print(f"Replicated?    {replicates_1} (Ratio outperforms both individual components)")

    print("\n==============================================================")
    print("REPLICATION TEST 2: Fully Adjusted Waist-BMI Residual (Diabetes)")
    print("==============================================================")
    df_w = df[['diabetes_binary', 'BMXBMI', 'BMXWAIST', 'RIDAGEYR', 'RIAGENDR']].dropna().copy()
    df_w["const"] = 1.0
    
    # Fit structural OLS adjusted for Age & Sex
    ols_res = sm.OLS(df_w['BMXWAIST'], df_w[['const', 'BMXBMI', 'RIDAGEYR', 'RIAGENDR']]).fit()
    df_w["residual"] = ols_res.resid
    df_w["Z_e_adj"] = Z_score(df_w["residual"])
    
    res_logit = sm.Logit(df_w['diabetes_binary'], df_w[['const', 'Z_e_adj', 'RIDAGEYR', 'RIAGENDR']]).fit(disp=0)
    replicates_2 = res_logit.pvalues['Z_e_adj'] < 0.05
    
    print(f"OLS Formula:   Waist = {ols_res.params['const']:.2f} + {ols_res.params['BMXBMI']:.3f}*BMI + {ols_res.params['RIDAGEYR']:.3f}*Age + {ols_res.params['RIAGENDR']:.3f}*Sex")
    print(f"Logit Result:  beta = {res_logit.params['Z_e_adj']:.4f}, z-stat = {res_logit.tvalues['Z_e_adj']:.2f}, p-value = {res_logit.pvalues['Z_e_adj']:.2e}")
    print(f"Replicated?    {replicates_2}")

    print("\n==============================================================")
    print("REPLICATION TEST 3: Albumin-Corrected Calcium (Stroke)")
    print("==============================================================")
    df_c = df[['stroke_binary', 'LBXSAL', 'LBXSCA', 'RIDAGEYR', 'RIAGENDR']].dropna().copy()
    df_c["const"] = 1.0
    
    # Fit structural OLS adjusted for Age & Sex
    ols_res_c = sm.OLS(df_c['LBXSCA'], df_c[['const', 'LBXSAL', 'RIDAGEYR', 'RIAGENDR']]).fit()
    df_c["residual"] = ols_res_c.resid
    df_c["Z_e_adj"] = Z_score(df_c["residual"])
    
    res_logit_c = sm.Logit(df_c['stroke_binary'], df_c[['const', 'Z_e_adj', 'RIDAGEYR', 'RIAGENDR']]).fit(disp=0)
    replicates_3 = res_logit_c.pvalues['Z_e_adj'] < 0.05
    
    print(f"OLS Slope:     {ols_res_c.params['LBXSAL']:.4f} (Expected close to clinical correction factor 0.8)")
    print(f"Logit Result:  beta = {res_logit_c.params['Z_e_adj']:.4f}, z-stat = {res_logit_c.tvalues['Z_e_adj']:.2f}, p-value = {res_logit_c.pvalues['Z_e_adj']:.2e}")
    print(f"Replicated?    {replicates_3}")

    print("\n==============================================================")
    print("REPLICATION TEST 4: Glycemic Homeostatic Uncoupling (Diabetes)")
    print("==============================================================")
    df_g = df[['diabetes_binary', 'LBXGH', 'LBXGLU', 'RIDAGEYR', 'RIAGENDR']].dropna().copy()
    df_g["const"] = 1.0
    
    # Fit structural OLS adjusted for Age & Sex
    ols_res_g = sm.OLS(df_g['LBXGLU'], df_g[['const', 'LBXGH', 'RIDAGEYR', 'RIAGENDR']]).fit()
    df_g["residual"] = ols_res_g.resid
    df_g["Z_e_adj"] = Z_score(df_g["residual"])
    df_g["abs_Z_e"] = df_g["Z_e_adj"].abs()
    
    res_logit_g = sm.Logit(df_g['diabetes_binary'], df_g[['const', 'abs_Z_e', 'RIDAGEYR', 'RIAGENDR']]).fit(disp=0)
    replicates_4 = res_logit_g.pvalues['abs_Z_e'] < 0.05
    
    print(f"Logit Result:  beta = {res_logit_g.params['abs_Z_e']:.4f}, z-stat = {res_logit_g.tvalues['abs_Z_e']:.2f}, p-value = {res_logit_g.pvalues['abs_Z_e']:.2e}")
    print(f"Replicated?    {replicates_4}")

    print("\n==============================================================")
    print("REPLICATION STUDY SUMMARY")
    print("==============================================================")
    print(f"1. Sodium / HbA1c Ratio:         {'REPLICATED [SUCCESS]' if replicates_1 else 'FAILED'}")
    print(f"2. Waist-BMI visceral residual:  {'REPLICATED [SUCCESS]' if replicates_2 else 'FAILED'}")
    print(f"3. Corrected Calcium stroke:     {'REPLICATED [SUCCESS]' if replicates_3 else 'FAILED'}")
    print(f"4. Glycemic Uncoupling diabetes: {'REPLICATED [SUCCESS]' if replicates_4 else 'FAILED'}")

if __name__ == "__main__":
    main()
