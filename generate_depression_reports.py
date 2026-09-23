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
    
    # Set premium aesthetic style
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
    
    # Save directory
    brain_dir = r"C:\Users\jpbro\.gemini\antigravity\brain\1663cb7c-e3b5-41f5-a60c-8dde1ab89813"
    
    # =========================================================================
    # PLOT 1: Ectopic Visceral Adiposity (Waist-BMI Residual)
    # =========================================================================
    print("Generating Figure 1: Ectopic Visceral Adiposity vs. Depression...")
    df_w = df[['depression_binary', 'BMXBMI', 'BMXWAIST', 'RIDAGEYR', 'RIAGENDR', 'cycle', 'const']].dropna().copy()
    ols_w = sm.OLS(df_w['BMXWAIST'], df_w[['const', 'BMXBMI', 'RIDAGEYR', 'RIAGENDR', 'cycle']]).fit()
    df_w["Z_e_adj"] = Z_score(ols_w.resid)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(
        x="Z_e_adj", y="depression_binary", data=df_w,
        logistic=True, n_boot=200, ci=95,
        scatter_kws={"color": "#ff7f0e", "alpha": 0.05, "s": 10},
        line_kws={"color": "#ff3b30", "linewidth": 3, "label": "Adjusted Visceral Adiposity Residual"},
        ax=ax
    )
    
    # Calculate binned rates for visual clarity
    bins = np.linspace(-3, 3, 10)
    df_w["bin"] = pd.cut(df_w["Z_e_adj"], bins)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_rates = df_w.groupby("bin", observed=False)["depression_binary"].mean()
    bin_counts = df_w.groupby("bin", observed=False)["depression_binary"].count()
    # Filter bins with enough data
    valid = bin_counts > 30
    ax.scatter(bin_centers[valid], bin_rates[valid], color="#ff9500", s=80, edgecolor="white", zorder=5, label="Empirical Bin Rates")
    
    ax.set_title("Clinical Depression Probability by Visceral Adiposity Residual\n(Waist Circumference Adjusted for Raw BMI, Age, Sex, & Cycle)", pad=15)
    ax.set_xlabel("Visceral Adiposity Residual Z-Score (Excess Waist size relative to weight)")
    ax.set_ylabel("Clinical Depression Probability (PHQ-9 ≥ 10)")
    ax.set_ylim(-0.02, 0.25)
    ax.legend(facecolor='#181a20', edgecolor='#2a2d37', loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(brain_dir, "depression_visceral_adiposity.png"), dpi=300, facecolor='#111216')
    plt.close()

    # =========================================================================
    # PLOT 2: Adiposity-Independent Systemic Inflammation (CRP-BMI Residual)
    # =========================================================================
    print("Generating Figure 2: CRP-BMI Inflammatory Residual vs. Depression...")
    df_inf = df[['depression_binary', 'LBXHSCRP', 'BMXBMI', 'RIDAGEYR', 'RIAGENDR', 'cycle', 'const']].dropna().copy()
    # hs-CRP is highly skewed; log-transform it for structural OLS
    df_inf["log_CRP"] = np.log(df_inf["LBXHSCRP"] + 1e-3)
    ols_inf = sm.OLS(df_inf['log_CRP'], df_inf[['const', 'BMXBMI', 'RIDAGEYR', 'RIAGENDR', 'cycle']]).fit()
    df_inf["Z_e_adj"] = Z_score(ols_inf.resid)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(
        x="Z_e_adj", y="depression_binary", data=df_inf,
        logistic=True, n_boot=200, ci=95,
        scatter_kws={"color": "#30d158", "alpha": 0.05, "s": 10},
        line_kws={"color": "#34c759", "linewidth": 3, "label": "Adjusted Pure CRP Residual"},
        ax=ax
    )
    
    # Binned empirical rates
    df_inf["bin"] = pd.cut(df_inf["Z_e_adj"], bins)
    bin_rates = df_inf.groupby("bin", observed=False)["depression_binary"].mean()
    bin_counts = df_inf.groupby("bin", observed=False)["depression_binary"].count()
    valid = bin_counts > 30
    ax.scatter(bin_centers[valid], bin_rates[valid], color="#2cd158", s=80, edgecolor="white", zorder=5, label="Empirical Bin Rates")
    
    ax.set_title("Clinical Depression Probability by Adiposity-Independent Inflammation\n(hs-CRP Adjusted for BMI, Age, Sex, & Cycle)", pad=15)
    ax.set_xlabel("Adiposity-Independent CRP Inflammatory Residual Z-Score")
    ax.set_ylabel("Clinical Depression Probability (PHQ-9 ≥ 10)")
    ax.set_ylim(-0.02, 0.25)
    ax.legend(facecolor='#181a20', edgecolor='#2a2d37', loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(brain_dir, "depression_inflammatory_residual.png"), dpi=300, facecolor='#111216')
    plt.close()

    # =========================================================================
    # PLOT 3: Active Calcium Deficit (Calcium-Albumin Residual)
    # =========================================================================
    print("Generating Figure 3: Calcium-Albumin Residual vs. Depression...")
    df_ca = df[['depression_binary', 'LBXSCA', 'LBXSAL', 'RIDAGEYR', 'RIAGENDR', 'cycle', 'const']].dropna().copy()
    ols_ca = sm.OLS(df_ca['LBXSCA'], df_ca[['const', 'LBXSAL', 'RIDAGEYR', 'RIAGENDR', 'cycle']]).fit()
    df_ca["Z_e_adj"] = Z_score(ols_ca.resid)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(
        x="Z_e_adj", y="depression_binary", data=df_ca,
        logistic=True, n_boot=200, ci=95,
        scatter_kws={"color": "#64d2ff", "alpha": 0.05, "s": 10},
        line_kws={"color": "#0a84ff", "linewidth": 3, "label": "Adjusted Corrected Calcium Residual"},
        ax=ax
    )
    
    # Binned empirical rates
    df_ca["bin"] = pd.cut(df_ca["Z_e_adj"], bins)
    bin_rates = df_ca.groupby("bin", observed=False)["depression_binary"].mean()
    bin_counts = df_ca.groupby("bin", observed=False)["depression_binary"].count()
    valid = bin_counts > 30
    ax.scatter(bin_centers[valid], bin_rates[valid], color="#0a84ff", s=80, edgecolor="white", zorder=5, label="Empirical Bin Rates")
    
    ax.set_title("Clinical Depression Probability by Active Corrected Calcium Residual\n(Serum Calcium Adjusted for Albumin, Age, Sex, & Cycle)", pad=15)
    ax.set_xlabel("Corrected Calcium Residual Z-Score (Negative indicates active deficit)")
    ax.set_ylabel("Clinical Depression Probability (PHQ-9 ≥ 10)")
    ax.set_ylim(-0.02, 0.25)
    ax.legend(facecolor='#181a20', edgecolor='#2a2d37', loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(brain_dir, "depression_calcium_residual.png"), dpi=300, facecolor='#111216')
    plt.close()

    # =========================================================================
    # PLOT 4: Metabolic-Lipid Index (BMI/HDL Ratio superiority)
    # =========================================================================
    print("Generating Figure 4: BMI/HDL Ratio superiority vs. Depression...")
    df_r = df[['depression_binary', 'BMXBMI', 'LBDHDD', 'RIDAGEYR', 'RIAGENDR', 'cycle', 'const']].dropna().copy()
    df_r["ratio"] = df_r["BMXBMI"] / df_r["LBDHDD"]
    df_r["Z_BMI"] = Z_score(df_r["BMXBMI"])
    df_r["Z_HDL"] = Z_score(df_r["LBDHDD"])
    df_r["Z_Ratio"] = Z_score(df_r["ratio"])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Fit logistics for components and ratio
    res_bmi = sm.Logit(df_r['depression_binary'], df_r[['const', 'Z_BMI', 'RIDAGEYR', 'RIAGENDR', 'cycle']]).fit(disp=0)
    res_hdl = sm.Logit(df_r['depression_binary'], df_r[['const', 'Z_HDL', 'RIDAGEYR', 'RIAGENDR', 'cycle']]).fit(disp=0)
    res_ratio = sm.Logit(df_r['depression_binary'], df_r[['const', 'Z_Ratio', 'RIDAGEYR', 'RIAGENDR', 'cycle']]).fit(disp=0)
    
    xs = np.linspace(-3, 3, 200)
    
    # Predict probabilities (holding age=mean, sex=1.5, cycle=0.5 constant)
    mean_age = df_r["RIDAGEYR"].mean()
    mean_sex = df_r["RIAGENDR"].mean()
    mean_cycle = df_r["cycle"].mean()
    
    def get_pred_curve(res, x_vals):
        preds = []
        for x in x_vals:
            # linpred = b0 + b1*x + g1*age + g2*sex + g3*cycle
            lp = res.params['const'] + res.params.iloc[1]*x + res.params['RIDAGEYR']*mean_age + res.params['RIAGENDR']*mean_sex + res.params['cycle']*mean_cycle
            p = 1 / (1 + np.exp(-lp))
            preds.append(p)
        return np.array(preds)
        
    ax.plot(xs, get_pred_curve(res_bmi, xs), color="#ff9500", linestyle="--", linewidth=2, label=f"BMI alone (AIC={res_bmi.aic:.1f})")
    ax.plot(xs, get_pred_curve(res_hdl, xs), color="#bf5af2", linestyle="--", linewidth=2, label=f"HDL alone (AIC={res_hdl.aic:.1f})")
    ax.plot(xs, get_pred_curve(res_ratio, xs), color="#ff375f", linewidth=3, label=f"BMI / HDL Ratio (AIC={res_ratio.aic:.1f})")
    
    # Binned rates for the ratio
    df_r["bin"] = pd.cut(df_r["Z_Ratio"], bins)
    bin_rates = df_r.groupby("bin", observed=False)["depression_binary"].mean()
    bin_counts = df_r.groupby("bin", observed=False)["depression_binary"].count()
    valid = bin_counts > 30
    ax.scatter(bin_centers[valid], bin_rates[valid], color="#ff375f", s=80, edgecolor="white", zorder=5, label="Empirical Ratio Rates")
    
    ax.set_title("Metabolic-Lipid Coordinate (BMI / HDL Ratio) vs. Depression Risk\n(Holding Age, Sex, & Cycle Cohort Constant)", pad=15)
    ax.set_xlabel("Biomarker Coordinate Z-Score")
    ax.set_ylabel("Predicted Clinical Depression Probability")
    ax.set_ylim(-0.02, 0.25)
    ax.legend(facecolor='#181a20', edgecolor='#2a2d37', loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(brain_dir, "depression_bmi_hdl_ratio.png"), dpi=300, facecolor='#111216')
    plt.close()
    
    print("All four depression figures generated successfully!")

if __name__ == "__main__":
    main()
