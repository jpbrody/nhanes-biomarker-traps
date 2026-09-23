import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from pandas_nhanes import get_dataset
import os

def Z_score(series):
    return (series - series.mean()) / (series.std() + 1e-9)

def get_discovery_models():
    print("Loading Discovery Cohort (2015-2018) for model fitting...")
    df_disc = pd.read_csv("nhanes_depression_pooled.csv")
    df_disc["const"] = 1.0
    df_disc_clean = df_disc[['BMXBMI', 'BMXWAIST', 'LBXHSCRP', 'LBDHDD', 'RIDAGEYR', 'RIAGENDR', 'const']].dropna().copy()
    
    # Fit OLS Waist
    ols_visc = sm.OLS(df_disc_clean['BMXWAIST'], df_disc_clean[['const', 'BMXBMI', 'RIDAGEYR', 'RIAGENDR']]).fit()
    # Fit OLS CRP
    df_disc_clean["log_CRP"] = np.log(df_disc_clean["LBXHSCRP"] + 1e-3)
    ols_crp = sm.OLS(df_disc_clean['log_CRP'], df_disc_clean[['const', 'BMXBMI', 'RIDAGEYR', 'RIAGENDR']]).fit()
    
    # Discovery Standardizing Constants
    df_disc_clean['e_visc'] = df_disc_clean['BMXWAIST'] - ols_visc.predict(df_disc_clean[['const', 'BMXBMI', 'RIDAGEYR', 'RIAGENDR']])
    df_disc_clean['e_crp'] = df_disc_clean['log_CRP'] - ols_crp.predict(df_disc_clean[['const', 'BMXBMI', 'RIDAGEYR', 'RIAGENDR']])
    df_disc_clean['ratio'] = df_disc_clean['BMXBMI'] / df_disc_clean['LBDHDD']
    
    ref_params = {
        'ratio_mean': df_disc_clean['ratio'].mean(),
        'ratio_std': df_disc_clean['ratio'].std(),
        'visc_mean': df_disc_clean['e_visc'].mean(),
        'visc_std': df_disc_clean['e_visc'].std(),
        'crp_mean': df_disc_clean['e_crp'].mean(),
        'crp_std': df_disc_clean['e_crp'].std()
    }
    
    return ols_visc, ols_crp, ref_params

def clean_phq9(df):
    depression_cols = [f"DPQ{i:03d}" for i in range(10, 100, 10)]
    dep_clean = df[depression_cols].copy()
    for col in depression_cols:
        dep_clean[col] = dep_clean[col].apply(lambda x: 0.0 if not pd.isna(x) and x < 1e-5 else (x if x in [1.0, 2.0, 3.0] else np.nan))
    dep_clean = dep_clean.dropna()
    df_clean = df.loc[dep_clean.index].copy()
    df_clean["depression_score"] = dep_clean.sum(axis=1)
    df_clean["depression_binary"] = (df_clean["depression_score"] >= 10).astype(float)
    return df_clean

def download_and_clean_val_a():
    print("Ingesting Historical Validation Cohort A (2009-2010)...")
    target_variables = {
        "RIDAGEYR": "DEMO_F", "RIAGENDR": "DEMO_F",
        "BMXBMI": "BMX_F", "BMXWAIST": "BMX_F",
        "LBDHDD": "HDL_F", "LBXCRP": "CRP_F",
        "DPQ010": "DPQ_F", "DPQ020": "DPQ_F", "DPQ030": "DPQ_F", "DPQ040": "DPQ_F",
        "DPQ050": "DPQ_F", "DPQ060": "DPQ_F", "DPQ070": "DPQ_F", "DPQ080": "DPQ_F", "DPQ090": "DPQ_F"
    }
    
    ds_to_vars = {}
    for var, ds in target_variables.items():
        ds_to_vars.setdefault(ds, []).append(var)
        
    merged_df = None
    for ds_name, vars_in_ds in ds_to_vars.items():
        ds_df = get_dataset(ds_name)
        cols_in_ds = {c.upper(): c for c in ds_df.columns}
        cols_to_keep = ["SEQN"]
        for v in vars_in_ds:
            if v.upper() in cols_in_ds:
                cols_to_keep.append(cols_in_ds[v.upper()])
        ds_subset = ds_df[cols_to_keep].copy()
        if merged_df is None:
            merged_df = ds_subset
        else:
            merged_df = pd.merge(merged_df, ds_subset, on="SEQN", how="outer")
            
    merged_df = clean_phq9(merged_df)
    merged_df["CRP_mgL"] = merged_df["LBXCRP"] * 10.0 # mg/dL to mg/L
    merged_df["const"] = 1.0
    return merged_df

def download_and_clean_val_b():
    print("Ingesting Modern Validation Cohort B (2021-2023)...")
    target_variables = {
        "RIDAGEYR": "DEMO_L", "RIAGENDR": "DEMO_L",
        "BMXBMI": "BMX_L", "BMXWAIST": "BMX_L",
        "LBDHDD": "HDL_L", "LBXHSCRP": "HSCRP_L",
        "DPQ010": "DPQ_L", "DPQ020": "DPQ_L", "DPQ030": "DPQ_L", "DPQ040": "DPQ_L",
        "DPQ050": "DPQ_L", "DPQ060": "DPQ_L", "DPQ070": "DPQ_L", "DPQ080": "DPQ_L", "DPQ090": "DPQ_L"
    }
    
    ds_to_vars = {}
    for var, ds in target_variables.items():
        ds_to_vars.setdefault(ds, []).append(var)
        
    merged_df = None
    for ds_name, vars_in_ds in ds_to_vars.items():
        ds_df = get_dataset(ds_name)
        cols_in_ds = {c.upper(): c for c in ds_df.columns}
        cols_to_keep = ["SEQN"]
        for v in vars_in_ds:
            if v.upper() in cols_in_ds:
                cols_to_keep.append(cols_in_ds[v.upper()])
        ds_subset = ds_df[cols_to_keep].copy()
        if merged_df is None:
            merged_df = ds_subset
        else:
            merged_df = pd.merge(merged_df, ds_subset, on="SEQN", how="outer")
            
    merged_df = clean_phq9(merged_df)
    merged_df["const"] = 1.0
    return merged_df

def main():
    os.chdir("g:\\My Drive\\gemini\\NHANES")
    
    # 1. Get models
    ols_visc, ols_crp, ref = get_discovery_models()
    
    # 2. Get and project Cohort A (2009-2010)
    df_val_a = download_and_clean_val_a()
    df_val_a_clean = df_val_a[['depression_score', 'depression_binary', 'BMXBMI', 'BMXWAIST', 'LBDHDD', 'CRP_mgL', 'RIDAGEYR', 'RIAGENDR', 'const']].dropna().copy()
    
    # Residuals
    e_visc_a = df_val_a_clean['BMXWAIST'] - ols_visc.predict(df_val_a_clean[['const', 'BMXBMI', 'RIDAGEYR', 'RIAGENDR']])
    e_crp_a = np.log(df_val_a_clean['CRP_mgL'] + 1e-3) - ols_crp.predict(df_val_a_clean[['const', 'BMXBMI', 'RIDAGEYR', 'RIAGENDR']])
    ratio_a = df_val_a_clean['BMXBMI'] / df_val_a_clean['LBDHDD']
    
    # Z-scores standardizing relative to reference
    Z_ratio_a = (ratio_a - ref['ratio_mean']) / ref['ratio_std']
    Z_visc_a = (e_visc_a - ref['visc_mean']) / ref['visc_std']
    Z_crp_a = (e_crp_a - ref['crp_mean']) / ref['crp_std']
    
    # Master Score
    df_val_a_clean['Z_master'] = (0.2136 * Z_ratio_a + 0.1591 * Z_visc_a + 0.1005 * Z_crp_a) / 0.3071588
    
    # 3. Get and project Cohort B (2021-2023)
    df_val_b = download_and_clean_val_b()
    df_val_b_clean = df_val_b[['depression_score', 'depression_binary', 'BMXBMI', 'BMXWAIST', 'LBDHDD', 'LBXHSCRP', 'RIDAGEYR', 'RIAGENDR', 'const']].dropna().copy()
    
    # Residuals
    e_visc_b = df_val_b_clean['BMXWAIST'] - ols_visc.predict(df_val_b_clean[['const', 'BMXBMI', 'RIDAGEYR', 'RIAGENDR']])
    e_crp_b = np.log(df_val_b_clean['LBXHSCRP'] + 1e-3) - ols_crp.predict(df_val_b_clean[['const', 'BMXBMI', 'RIDAGEYR', 'RIAGENDR']])
    ratio_b = df_val_b_clean['BMXBMI'] / df_val_b_clean['LBDHDD']
    
    # Z-scores standardizing relative to reference
    Z_ratio_b = (ratio_b - ref['ratio_mean']) / ref['ratio_std']
    Z_visc_b = (e_visc_b - ref['visc_mean']) / ref['visc_std']
    Z_crp_b = (e_crp_b - ref['crp_mean']) / ref['crp_std']
    
    # Master Score
    df_val_b_clean['Z_master'] = (0.2136 * Z_ratio_b + 0.1591 * Z_visc_b + 0.1005 * Z_crp_b) / 0.3071588
    
    # 4. Generate Plot
    sns.set_theme(style="darkgrid")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'figure.titlesize': 20,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.facecolor': '#111216',
        'axes.facecolor': '#181a20',
        'text.color': '#e0e2e8',
        'axes.labelcolor': '#e0e2e8',
        'xtick.color': '#a0a4b0',
        'ytick.color': '#a0a4b0',
        'grid.color': '#2a2d37',
        'grid.linestyle': '--',
        'grid.linewidth': 0.5
    })
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
    
    # Jitter function
    def add_jitter(series, scale=0.15):
        return series + np.random.normal(0, scale, size=len(series))
    
    # Plot Val A (Left)
    ax_a = axes[0]
    # Split controls and cases for glowing color assignment
    ctrls_a = df_val_a_clean[df_val_a_clean['depression_binary'] == 0]
    cases_a = df_val_a_clean[df_val_a_clean['depression_binary'] == 1]
    
    ax_a.scatter(add_jitter(ctrls_a['Z_master']), add_jitter(ctrls_a['depression_score']),
                 color='#00e5ff', alpha=0.12, s=8, label='Healthy Controls (PHQ-9 < 10)')
    ax_a.scatter(add_jitter(cases_a['Z_master']), add_jitter(cases_a['depression_score']),
                 color='#ff5252', alpha=0.25, s=10, label='Clinical Cases (PHQ-9 $\geq$ 10)')
    
    # Superimpose regression trend
    sns.regplot(x='Z_master', y='depression_score', data=df_val_a_clean, ax=ax_a,
                scatter=False, color='#ffffff', line_kws={'linewidth': 3.5, 'label': 'OLS Continuous TrendLine'})
    
    ax_a.set_title("Panel A: Historical Decadal Cohort (2009-2010)\n$N = 5,181$", pad=15)
    ax_a.set_xlabel("Master Neuro-Metabolic Score Z-Score")
    ax_a.set_ylabel("Depression Severity Score (PHQ-9)")
    ax_a.set_xlim(-3.5, 4.5)
    ax_a.set_ylim(-0.8, 27.8)
    ax_a.legend(facecolor='#181a20', edgecolor='#2a2d37', loc='upper left')
    
    # Plot Val B (Right)
    ax_b = axes[1]
    ctrls_b = df_val_b_clean[df_val_b_clean['depression_binary'] == 0]
    cases_b = df_val_b_clean[df_val_b_clean['depression_binary'] == 1]
    
    ax_b.scatter(add_jitter(ctrls_b['Z_master']), add_jitter(ctrls_b['depression_score']),
                 color='#00e5ff', alpha=0.12, s=8, label='Healthy Controls (PHQ-9 < 10)')
    ax_b.scatter(add_jitter(cases_b['Z_master']), add_jitter(cases_b['depression_score']),
                 color='#ff5252', alpha=0.25, s=10, label='Clinical Cases (PHQ-9 $\geq$ 10)')
    
    # Superimpose regression trend
    sns.regplot(x='Z_master', y='depression_score', data=df_val_b_clean, ax=ax_b,
                scatter=False, color='#ffffff', line_kws={'linewidth': 3.5, 'label': 'OLS Continuous TrendLine'})
    
    ax_b.set_title("Panel B: Post-Pandemic Modern Cohort (2021-2023)\n$N = 4,817$", pad=15)
    ax_b.set_xlabel("Master Neuro-Metabolic Score Z-Score")
    ax_b.set_xlim(-3.5, 4.5)
    ax_b.legend(facecolor='#181a20', edgecolor='#2a2d37', loc='upper left')
    
    plt.suptitle("Clinical Depression Severity vs. Master Neuro-Metabolic Z-Score\n(Side-by-Side Multi-Era External Validations)", y=0.98)
    plt.tight_layout()
    
    brain_dir = r"C:\Users\jpbro\.gemini\antigravity\brain\1663cb7c-e3b5-41f5-a60c-8dde1ab89813"
    plt.savefig(os.path.join(brain_dir, "depression_validation_scatter.png"), dpi=300, facecolor='#111216')
    plt.close()
    
    print("\nStunning dual-panel scatter plot generated successfully at depression_validation_scatter.png!")
    print("==================================================")

if __name__ == "__main__":
    main()
