import pandas as pd
import numpy as np
from itertools import combinations
import statsmodels.formula.api as smf
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', RuntimeWarning)

def calculate_haldane_or_99(a, b, c, d):
    a_c = a + 0.5
    b_c = b + 0.5
    c_c = c + 0.5
    d_c = d + 0.5
    odds_ratio = (a_c * d_c) / (b_c * c_c)
    se = np.sqrt(1/a_c + 1/b_c + 1/c_c + 1/d_c)
    lower_99 = np.exp(np.log(odds_ratio) - 2.576 * se)
    upper_99 = np.exp(np.log(odds_ratio) + 2.576 * se)
    return odds_ratio, lower_99, upper_99

def main():
    print("Loading datasets for 3-way logical gate high-throughput screen...")
    df_disc_raw = pd.read_csv("g:\\My Drive\\gemini\\NHANES\\nhanes_depression_pooled.csv")
    df_val_raw = pd.read_csv("g:\\My Drive\\gemini\\NHANES\\nhanes_comprehensive_validation_2007_2010.csv")
    
    biomarkers = ["BMXWAIST", "LBXGH", "LBXSCR", "LBXSKSI", "LBXSNASI", "LBXSCA", "LBXHSCRP", "LBXHGB", "LBXWBCSI", "LBXPLTSI"]
    outcomes = ["diabetes_binary", "hypertension_binary", "high_cholesterol_binary", "arthritis_binary", "depression_binary", "stroke_binary", "heart_attack_binary"]
    
    # Generate all triplets of biomarkers
    triplets = list(combinations(biomarkers, 3))
    print(f"Total biomarker triplets to screen: {len(triplets)}")
    
    results = []
    
    for outcome in outcomes:
        print(f"Screening outcome: {outcome}...")
        for varA, varB, varC in triplets:
            for typeA in ["low", "high"]:
                for typeB in ["low", "high"]:
                    for typeC in ["low", "high"]:
                        
                        # Prepare data
                        features = [varA, varB, varC, outcome]
                        df_d = df_disc_raw[features].dropna().copy()
                        df_v = df_val_raw[features].dropna().copy()
                        
                        if len(df_d) < 1000 or len(df_v) < 1000:
                            continue
                            
                        # Compute thresholds strictly on Discovery
                        if typeA == "low":
                            cutoffA = df_d[varA].quantile(0.10)
                            anomA_d = df_d[varA] < cutoffA
                            anomA_v = df_v[varA] < cutoffA
                        else:
                            cutoffA = df_d[varA].quantile(0.90)
                            anomA_d = df_d[varA] > cutoffA
                            anomA_v = df_v[varA] > cutoffA
                            
                        if typeB == "low":
                            cutoffB = df_d[varB].quantile(0.10)
                            anomB_d = df_d[varB] < cutoffB
                            anomB_v = df_v[varB] < cutoffB
                        else:
                            cutoffB = df_d[varB].quantile(0.90)
                            anomB_d = df_d[varB] > cutoffB
                            anomB_v = df_v[varB] > cutoffB
                            
                        if typeC == "low":
                            cutoffC = df_d[varC].quantile(0.10)
                            anomC_d = df_d[varC] < cutoffC
                            anomC_v = df_v[varC] < cutoffC
                        else:
                            cutoffC = df_d[varC].quantile(0.90)
                            anomC_d = df_d[varC] > cutoffC
                            anomC_v = df_v[varC] > cutoffC
                            
                        # Quadrant 000: baseline (neither anomalous)
                        q0_d = df_d[~anomA_d & ~anomB_d & ~anomC_d]
                        n0_d = len(q0_d)
                        y0_d = int(q0_d[outcome].sum())
                        
                        q0_v = df_v[~anomA_v & ~anomB_v & ~anomC_v]
                        n0_v = len(q0_v)
                        y0_v = int(q0_v[outcome].sum())
                        
                        # Quadrant 111: all three anomalous (Gate)
                        q3_d = df_d[anomA_d & anomB_d & anomC_d]
                        n3_d = len(q3_d)
                        y3_d = int(q3_d[outcome].sum())
                        
                        q3_v = df_v[anomA_v & anomB_v & anomC_v]
                        n3_v = len(q3_v)
                        y3_v = int(q3_v[outcome].sum())
                        
                        # Sample size filters
                        if n3_d < 30 or n3_v < 30:
                            continue
                            
                        # Haldane OR calculations for 3-way gate relative to control
                        or_d, lower_99_d, upper_99_d = calculate_haldane_or_99(y3_d, n3_d - y3_d, y0_d, n0_d - y0_d)
                        or_v, lower_99_v, upper_99_v = calculate_haldane_or_99(y3_v, n3_v - y3_v, y0_v, n0_v - y0_v)
                        
                        rate3_d = y3_d / n3_d
                        rate3_v = y3_v / n3_v
                        
                        results.append({
                            "outcome": outcome,
                            "varA": varA, "typeA": typeA,
                            "varB": varB, "typeB": typeB,
                            "varC": varC, "typeC": typeC,
                            "N111_disc": n3_d, "rate111_disc": rate3_d, "or_disc": or_d, "lower99_disc": lower_99_d,
                            "N111_val": n3_v, "rate111_val": rate3_v, "or_val": or_v, "lower99_val": lower_99_v
                        })
                        
    df_res = pd.DataFrame(results)
    if not df_res.empty:
        # Require 99% Lower CI > 1.5 in BOTH cohorts
        df_filter = df_res[(df_res["lower99_disc"] > 1.5) & (df_res["lower99_val"] > 1.5)]
        
        # Sort by Validation Lower 99% CI
        df_filter = df_filter.sort_values(by="lower99_val", ascending=False)
        
        print(f"\n==================================================")
        print(f"Discovered {len(df_filter)} 3-way gates meeting strict 99% CI criteria!")
        print(f"==================================================")
        for idx, row in df_filter.head(15).iterrows():
            print(f"\nGate: {row['varA']} ({row['typeA']}) & {row['varB']} ({row['typeB']}) & {row['varC']} ({row['typeC']}) on {row['outcome']}")
            print(f"  Discovery:  N111={row['N111_disc']}, Rate={row['rate111_disc']*100:.2f}%, OR={row['or_disc']:.2f} (99% Lower CI={row['lower99_disc']:.2f})")
            print(f"  Validation: N111={row['N111_val']}, Rate={row['rate111_val']*100:.2f}%, OR={row['or_val']:.2f} (99% Lower CI={row['lower99_val']:.2f})")
            
        # Save results for further inspection
        df_filter.to_csv("g:\\My Drive\\gemini\\NHANES\\discovered_3way_gates.csv", index=False)
        print("\nSUCCESS: Wrote discovered 3-way gates to 'discovered_3way_gates.csv'")
    else:
        print("No 3-way gates met the strict criteria.")

if __name__ == "__main__":
    main()
