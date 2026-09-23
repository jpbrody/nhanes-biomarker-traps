import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

def main():
    df_disc = pd.read_csv("g:\\My Drive\\gemini\\NHANES\\nhanes_depression_pooled.csv")
    df_val = pd.read_csv("g:\\My Drive\\gemini\\NHANES\\nhanes_depression_validation_2007_2010.csv")
    
    # Drop NaNs
    disc_features = ["LBDLDL", "LBXHSCRP", "RIDAGEYR", "RIAGENDR", "depression_binary"]
    df_disc = df_disc.dropna(subset=disc_features).copy()
    df_val = df_val.dropna(subset=disc_features).copy()
    
    ldl_p10 = df_disc["LBDLDL"].quantile(0.10)
    crp_p90 = df_disc["LBXHSCRP"].quantile(0.90)
    
    print(f"Locked Cutoffs: LDL Low P10 = {ldl_p10:.4f}, CRP High P90 = {crp_p90:.4f}")
    print(f"Discovery complete cases: {len(df_disc)}")
    print(f"Validation complete cases: {len(df_val)}")
    
    for df, name in [(df_disc, "Discovery"), (df_val, "Validation")]:
        df["LBDLDL_anom"] = (df["LBDLDL"] < ldl_p10).astype(float)
        df["LBXHSCRP_anom"] = (df["LBXHSCRP"] > crp_p90).astype(float)
        
        df["quadrant"] = 0
        df.loc[(df["LBDLDL_anom"] == 1.0) & (df["LBXHSCRP_anom"] == 0.0), "quadrant"] = 1
        df.loc[(df["LBDLDL_anom"] == 0.0) & (df["LBXHSCRP_anom"] == 1.0), "quadrant"] = 2
        df.loc[(df["LBDLDL_anom"] == 1.0) & (df["LBXHSCRP_anom"] == 1.0), "quadrant"] = 3
        
        print(f"\n--- {name} LDL-CRP Gate Quadrants ---")
        for q_idx in range(4):
            q_df = df[df["quadrant"] == q_idx]
            print(f"  Quadrant {q_idx}: N = {len(q_df)}, Depressed = {int(q_df['depression_binary'].sum())} ({q_df['depression_binary'].mean()*100:.2f}%)")
            
        print(f"\n--- {name} LDL-CRP Gate Model ---")
        model = smf.logit("depression_binary ~ RIDAGEYR + C(RIAGENDR) + LBDLDL_anom * LBXHSCRP_anom", data=df).fit()
        print(model.summary().tables[1])

if __name__ == "__main__":
    main()
