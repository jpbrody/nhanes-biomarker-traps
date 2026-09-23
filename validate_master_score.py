import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from pandas_nhanes import get_dataset
import os

def Z_score(series):
    return (series - series.mean()) / (series.std() + 1e-9)

def download_and_clean_validation():
    print("\n==================================================")
    print("Downloading and cleaning 2009-2010 Validation Cohort")
    print("==================================================")
    
    target_variables = {
        "RIDAGEYR": "DEMO_F",
        "RIAGENDR": "DEMO_F",
        "BMXBMI": "BMX_F",
        "BMXWAIST": "BMX_F",
        "LBDHDD": "HDL_F",
        "LBXCRP": "CRP_F", # CRP measured as LBXCRP in mg/dL
        "DPQ010": "DPQ_F",
        "DPQ020": "DPQ_F",
        "DPQ030": "DPQ_F",
        "DPQ040": "DPQ_F",
        "DPQ050": "DPQ_F",
        "DPQ060": "DPQ_F",
        "DPQ070": "DPQ_F",
        "DPQ080": "DPQ_F",
        "DPQ090": "DPQ_F",
    }
    
    dataset_to_vars = {}
    for var, ds in target_variables.items():
        if ds not in dataset_to_vars:
            dataset_to_vars[ds] = []
        dataset_to_vars[ds].append(var)
        
    merged_df = None
    for ds_name, vars_in_ds in dataset_to_vars.items():
        print(f"Downloading dataset {ds_name}...")
        try:
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
        except Exception as e:
            print(f"Error downloading {ds_name}: {e}")
            
    # Clean PHQ-9 items
    depression_cols = [f"DPQ{i:03d}" for i in range(10, 100, 10)]
    existing_dep = [c for c in depression_cols if c in merged_df.columns]
    
    if len(existing_dep) == 9:
        print("Cleaning PHQ-9 items...")
        dep_clean = merged_df[existing_dep].copy()
        for col in existing_dep:
            dep_clean[col] = dep_clean[col].apply(lambda x: 0.0 if not pd.isna(x) and x < 1e-5 else (x if x in [1.0, 2.0, 3.0] else np.nan))
        dep_clean = dep_clean.dropna()
        merged_df = merged_df.loc[dep_clean.index].copy()
        merged_df["depression_score"] = dep_clean.sum(axis=1)
        merged_df["depression_binary"] = (merged_df["depression_score"] >= 10).astype(float)
        
    # Standardize CRP: multiply mg/dL by 10 to get mg/L (to match LBXHSCRP!)
    merged_df["CRP_mgL"] = merged_df["LBXCRP"] * 10.0
    
    print(f"Validation cohort clean shape: {merged_df.shape}")
    print(f"Completed surveys: {len(merged_df)}, Depressed cases: {(merged_df['depression_binary'] == 1.0).sum()} ({ (merged_df['depression_binary'] == 1.0).mean()*100:.2f}%)")
    return merged_df

def main():
    os.chdir("g:\\My Drive\\gemini\\NHANES")
    
    # 1. Load Discovery Cohort (2015-2018)
    df_disc = pd.read_csv("nhanes_depression_pooled.csv")
    df_disc["const"] = 1.0
    
    # Fit exact structural OLS models on Discovery Cohort (without cycle cohort flag for perfect portability!)
    print("\nFitting structural OLS baseline equations on Discovery Cohort...")
    df_disc_clean = df_disc[['BMXBMI', 'BMXWAIST', 'LBXHSCRP', 'RIDAGEYR', 'RIAGENDR', 'const']].dropna().copy()
    
    # Visceral Adiposity Discovery OLS
    ols_visc_disc = sm.OLS(df_disc_clean['BMXWAIST'], df_disc_clean[['const', 'BMXBMI', 'RIDAGEYR', 'RIAGENDR']]).fit()
    print("Visceral Fat OLS Equation:")
    print(f"Waist = {ols_visc_disc.params['const']:.4f} + {ols_visc_disc.params['BMXBMI']:.4f}*BMI + {ols_visc_disc.params['RIDAGEYR']:.4f}*Age + {ols_visc_disc.params['RIAGENDR']:.4f}*Sex")
    
    # Adiposity-Independent Inflammation Discovery OLS
    df_disc_clean["log_CRP"] = np.log(df_disc_clean["LBXHSCRP"] + 1e-3)
    ols_crp_disc = sm.OLS(df_disc_clean['log_CRP'], df_disc_clean[['const', 'BMXBMI', 'RIDAGEYR', 'RIAGENDR']]).fit()
    print("\nCRP Inflammatory OLS Equation:")
    print(f"log(CRP) = {ols_crp_disc.params['const']:.4f} + {ols_crp_disc.params['BMXBMI']:.4f}*BMI + {ols_crp_disc.params['RIDAGEYR']:.4f}*Age + {ols_crp_disc.params['RIAGENDR']:.4f}*Sex")
    
    # 2. Get and Clean Validation Cohort (2009-2010)
    df_val = download_and_clean_validation()
    df_val["const"] = 1.0
    
    # 3. Project Discovery OLS residuals onto Validation Cohort
    print("\nProjecting structural residuals onto Validation Cohort...")
    df_val_clean = df_val[['depression_binary', 'BMXBMI', 'BMXWAIST', 'LBDHDD', 'CRP_mgL', 'RIDAGEYR', 'RIAGENDR', 'const']].dropna().copy()
    
    # Predict Visceral Residual on Validation
    val_waist_pred = (
        ols_visc_disc.params['const'] + 
        ols_visc_disc.params['BMXBMI'] * df_val_clean['BMXBMI'] + 
        ols_visc_disc.params['RIDAGEYR'] * df_val_clean['RIDAGEYR'] + 
        ols_visc_disc.params['RIAGENDR'] * df_val_clean['RIAGENDR']
    )
    df_val_clean["e_visc"] = df_val_clean['BMXWAIST'] - val_waist_pred
    df_val_clean["Z_visc"] = df_val_clean["e_visc"] / 5.9847464 # Locked-in Discovery SD
    
    # Predict CRP Residual on Validation
    val_log_crp = np.log(df_val_clean["CRP_mgL"] + 1e-3)
    val_log_crp_pred = (
        ols_crp_disc.params['const'] + 
        ols_crp_disc.params['BMXBMI'] * df_val_clean['BMXBMI'] + 
        ols_crp_disc.params['RIDAGEYR'] * df_val_clean['RIDAGEYR'] + 
        ols_crp_disc.params['RIAGENDR'] * df_val_clean['RIAGENDR']
    )
    df_val_clean["e_crp"] = val_log_crp - val_log_crp_pred
    df_val_clean["Z_crp"] = df_val_clean["e_crp"] / 1.0673420 # Locked-in Discovery SD
    
    # Ratio on Validation
    df_val_clean["ratio"] = df_val_clean["BMXBMI"] / df_val_clean["LBDHDD"]
    df_val_clean["Z_ratio"] = (df_val_clean["ratio"] - 0.6148197) / 0.2725571 # Locked-in Discovery Mean and SD
    
    # 4. Construct Master Score on Validation using EXACT Discovery Weights
    # Discovery weights: Z_ratio = 0.2136, Z_visc = 0.1591, Z_crp = 0.1005
    df_val_clean["master_index"] = (
        0.2136 * df_val_clean['Z_ratio'] + 
        0.1591 * df_val_clean['Z_visc'] + 
        0.1005 * df_val_clean['Z_crp']
    )
    df_val_clean["Z_master"] = df_val_clean["master_index"] / 0.3071588 # Locked-in Discovery SD
    
    # 5. Fit Validation Logistic Regression Model
    print("\nEvaluating Master Score on independent 2009-2010 Validation Cohort...")
    val_model = sm.Logit(
        df_val_clean['depression_binary'], 
        df_val_clean[['const', 'Z_master', 'RIDAGEYR', 'RIAGENDR']]
    )
    val_res = val_model.fit(disp=0)
    print(val_res.summary())
    
    print("\n==================================================")
    print("Master Score INDEPENDENT VALIDATION Results")
    print("==================================================")
    print(f"Validation association: beta = {val_res.params['Z_master']:.4f}, z = {val_res.tvalues['Z_master']:.2f}, p-value = {val_res.pvalues['Z_master']:.4e}")
    print(f"Validation LLR p-value = {val_res.llr_pvalue:.4e}")
    
    # Quintile risk stratification in validation
    df_val_clean["quintile"] = pd.qcut(df_val_clean["Z_master"], 5, labels=[1, 2, 3, 4, 5])
    quintile_rates = df_val_clean.groupby("quintile", observed=False)["depression_binary"].mean()
    quintile_counts = df_val_clean.groupby("quintile", observed=False)["depression_binary"].count()
    print("\nClinical Depression Rate by Validation Master Score Quintile:")
    for q in range(1, 6):
        print(f"Quintile {q}: Rate = {quintile_rates[q]*100:.2f}% (n = {quintile_counts[q]})")
        
    # 6. Plot the Validation Quintiles
    sns.set_theme(style="darkgrid", palette="muted")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'figure.titlesize': 18,
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
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot logistic regression curve on validation data
    sns.regplot(
        x="Z_master", y="depression_binary", data=df_val_clean,
        logistic=True, n_boot=200, ci=95,
        scatter_kws={"color": "#30d158", "alpha": 0.05, "s": 10},
        line_kws={"color": "#34c759", "linewidth": 3, "label": f"Validation Logistic Risk (z = {val_res.tvalues['Z_master']:.2f})"},
        ax=ax
    )
    
    # Plot empirical quintiles centers and rates
    bin_centers = []
    rates = []
    for q in range(1, 6):
        subset = df_val_clean[df_val_clean["quintile"] == q]
        bin_centers.append(subset["Z_master"].mean())
        rates.append(subset["depression_binary"].mean())
        
    ax.scatter(bin_centers, rates, color="#ff9500", s=100, edgecolor="white", zorder=5, label="Empirical Quintile Rates")
    
    # Annotate rates
    for idx, (x, y_val) in enumerate(zip(bin_centers, rates)):
        ax.annotate(
            f"{y_val*100:.1f}%", 
            (x, y_val), 
            textcoords="offset points", 
            xytext=(0,10), 
            ha='center', 
            fontsize=10, 
            weight='bold', 
            color='#ff9500'
        )
        
    ax.set_title("Clinical Depression Probability by Master Score\n(Independent Validation Cohort NHANES 2009-2010)", pad=15)
    ax.set_xlabel("Master Neuro-Metabolic Score Z-Score")
    ax.set_ylabel("Clinical Depression Probability (PHQ-9 ≥ 10)")
    ax.set_ylim(-0.02, 0.25)
    ax.legend(facecolor='#181a20', edgecolor='#2a2d37', loc='upper left')
    plt.tight_layout()
    
    brain_dir = r"C:\Users\jpbro\.gemini\antigravity\brain\1663cb7c-e3b5-41f5-a60c-8dde1ab89813"
    plt.savefig(os.path.join(brain_dir, "depression_validation_score.png"), dpi=300, facecolor='#111216')
    plt.close()
    
    print("\nMaster Score INDEPENDENT VALIDATION Plot generated successfully!")
    print("==================================================")

if __name__ == "__main__":
    main()
