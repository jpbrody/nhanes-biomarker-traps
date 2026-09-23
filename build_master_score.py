import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import os

def Z_score(series):
    return (series - series.mean()) / (series.std() + 1e-9)

def main():
    os.chdir("g:\\My Drive\\gemini\\NHANES")
    
    if not os.path.exists("nhanes_depression_pooled.csv"):
        print("Error: nhanes_depression_pooled.csv not found!")
        return
        
    df = pd.read_csv("nhanes_depression_pooled.csv")
    df["const"] = 1.0
    
    # 1. Reconstruct the geometries on a clean subset
    df_sub = df[['depression_binary', 'BMXBMI', 'BMXWAIST', 'LBDHDD', 'LBXHSCRP', 'RIDAGEYR', 'RIAGENDR', 'cycle', 'const']].dropna().copy()
    
    # Visceral Adiposity Residual
    ols_visc = sm.OLS(df_sub['BMXWAIST'], df_sub[['const', 'BMXBMI', 'RIDAGEYR', 'RIAGENDR', 'cycle']]).fit()
    df_sub["Z_visc"] = Z_score(ols_visc.resid)
    
    # Adiposity-Independent Inflammation (CRP)
    df_sub["log_CRP"] = np.log(df_sub["LBXHSCRP"] + 1e-3)
    ols_crp = sm.OLS(df_sub['log_CRP'], df_sub[['const', 'BMXBMI', 'RIDAGEYR', 'RIAGENDR', 'cycle']]).fit()
    df_sub["Z_crp"] = Z_score(ols_crp.resid)
    
    # BMI / HDL Ratio
    df_sub["ratio"] = df_sub["BMXBMI"] / df_sub["LBDHDD"]
    df_sub["Z_ratio"] = Z_score(df_sub["ratio"])
    
    # 2. Fit Multivariable Joint Model to get exact weights
    joint_model = sm.Logit(
        df_sub['depression_binary'], 
        df_sub[['const', 'Z_ratio', 'Z_crp', 'Z_visc', 'RIDAGEYR', 'RIAGENDR', 'cycle']]
    )
    joint_res = joint_model.fit(disp=0)
    
    # 3. Construct the Master Score Index
    df_sub["master_index"] = (
        joint_res.params['Z_ratio'] * df_sub['Z_ratio'] + 
        joint_res.params['Z_crp'] * df_sub['Z_crp'] + 
        joint_res.params['Z_visc'] * df_sub['Z_visc']
    )
    df_sub["Z_master"] = Z_score(df_sub["master_index"])
    
    # 4. Fit the final evaluation model
    master_model = sm.Logit(
        df_sub['depression_binary'],
        df_sub[['const', 'Z_master', 'RIDAGEYR', 'RIAGENDR', 'cycle']]
    )
    master_res = master_model.fit(disp=0)
    
    print("\n==================================================")
    print("Master Neuro-Metabolic Depression Score Validation")
    print("==================================================")
    print(f"Overall association: beta = {master_res.params['Z_master']:.4f}, z = {master_res.tvalues['Z_master']:.2f}, p-value = {master_res.pvalues['Z_master']:.4e}")
    print(f"Model AIC = {master_res.aic:.1f}, LLR p-value = {master_res.llr_pvalue:.4e}")
    
    # Calculate Quintile rates
    df_sub["quintile"] = pd.qcut(df_sub["Z_master"], 5, labels=[1, 2, 3, 4, 5])
    quintile_rates = df_sub.groupby("quintile", observed=False)["depression_binary"].mean()
    quintile_counts = df_sub.groupby("quintile", observed=False)["depression_binary"].count()
    
    # 5. Plot the Premium Dark-Mode Figure
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
    
    # Plot logistic regression curve
    sns.regplot(
        x="Z_master", y="depression_binary", data=df_sub,
        logistic=True, n_boot=200, ci=95,
        scatter_kws={"color": "#bf5af2", "alpha": 0.05, "s": 10},
        line_kws={"color": "#bf5af2", "linewidth": 3, "label": f"Master Score Logistic Risk (z = {master_res.tvalues['Z_master']:.2f})"},
        ax=ax
    )
    
    # Plot empirical quintile centers and rates
    bins = np.linspace(-3, 3, 6)
    bin_centers = []
    rates = []
    # Calculate exact mean Z_master within each quintile for plotting x-coords
    for q in range(1, 6):
        subset = df_sub[df_sub["quintile"] == q]
        bin_centers.append(subset["Z_master"].mean())
        rates.append(subset["depression_binary"].mean())
        
    ax.scatter(bin_centers, rates, color="#ff375f", s=100, edgecolor="white", zorder=5, label="Empirical Quintile Rates")
    
    # Label each quintile dot with its rate
    for idx, (x, y_val) in enumerate(zip(bin_centers, rates)):
        ax.annotate(
            f"{y_val*100:.1f}%", 
            (x, y_val), 
            textcoords="offset points", 
            xytext=(0,10), 
            ha='center', 
            fontsize=10, 
            weight='bold', 
            color='#ff375f'
        )
        
    ax.set_title("Clinical Depression Probability by Master Neuro-Metabolic Score\n(Weighted Composite of Lipid-Ratio, Visceral Fat, & Inflammation)", pad=15)
    ax.set_xlabel("Master Neuro-Metabolic Score Z-Score")
    ax.set_ylabel("Clinical Depression Probability (PHQ-9 ≥ 10)")
    ax.set_ylim(-0.02, 0.25)
    ax.legend(facecolor='#181a20', edgecolor='#2a2d37', loc='upper left')
    plt.tight_layout()
    
    brain_dir = r"C:\Users\jpbro\.gemini\antigravity\brain\1663cb7c-e3b5-41f5-a60c-8dde1ab89813"
    plt.savefig(os.path.join(brain_dir, "depression_master_score.png"), dpi=300, facecolor='#111216')
    plt.close()
    
    print("Master Neuro-Metabolic Score Figure generated successfully!")
    print("==================================================")

if __name__ == "__main__":
    main()
