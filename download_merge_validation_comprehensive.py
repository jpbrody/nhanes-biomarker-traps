import pandas as pd
import numpy as np
from pandas_nhanes import get_dataset

def main():
    print("=========================================================")
    print("STARTING COMPREHENSIVE VALIDATION COHORT POOLING (2007-2010)")
    print("=========================================================")
    
    cycles_config = [
        {
            "name": "2007-2008",
            "demo": "DEMO_E",
            "diq": "DIQ_E",
            "bpq": "BPQ_E",
            "mcq": "MCQ_E",
            "dpq": "DPQ_E",
            "bmx": "BMX_E",
            "bpx": "PEASC_E", # Wait, PEASC_E is the exam status, let's make sure it is BPX_E!
            "bpx_ds": "BPX_E",
            "ghb": "GHB_E",
            "biopro": "BIOPRO_E",
            "crp": "CRP_E",
            "cbc": "CBC_E",
            "trigly": "TRIGLY_E",
            "tchol": "TCHOL_E",
            "hdl": "HDL_E"
        },
        {
            "name": "2009-2010",
            "demo": "DEMO_F",
            "diq": "DIQ_F",
            "bpq": "BPQ_F",
            "mcq": "MCQ_F",
            "dpq": "DPQ_F",
            "bmx": "BMX_F",
            "bpx_ds": "BPX_F",
            "ghb": "GHB_F",
            "biopro": "BIOPRO_F",
            "crp": "CRP_F",
            "cbc": "CBC_F",
            "trigly": "TRIGLY_F",
            "tchol": "TCHOL_F",
            "hdl": "HDL_F"
        }
    ]
    
    all_cleaned = []
    
    for config in cycles_config:
        print(f"\nProcessing Cycle: {config['name']}")
        try:
            # 1. Download datasets
            print("  Downloading datasets...")
            df_demo = get_dataset(config["demo"])[["SEQN", "RIDAGEYR", "RIAGENDR"]]
            df_diq = get_dataset(config["diq"])[["SEQN", "DIQ010"]]
            df_bpq = get_dataset(config["bpq"])[["SEQN", "BPQ020", "BPQ080"]]
            
            mcq_cols = ["SEQN", "MCQ160A", "MCQ160L", "MCQ160M", "MCQ160E", "MCQ160F"]
            df_mcq = get_dataset(config["mcq"])[mcq_cols]
            df_dpq = get_dataset(config["dpq"])
            df_bmx = get_dataset(config["bmx"])[["SEQN", "BMXWAIST", "BMXBMI"]]
            
            # Download blood pressure and compute averages
            df_bpx_raw = get_dataset(config["bpx_ds"])
            bp_cols = ["SEQN"]
            for col in ["BPXSY1", "BPXSY2", "BPXSY3", "BPXDI1", "BPXDI2", "BPXDI3"]:
                if col in df_bpx_raw.columns:
                    bp_cols.append(col)
            df_bpx = df_bpx_raw[bp_cols].copy()
            systolic_cols = [c for c in ["BPXSY1", "BPXSY2", "BPXSY3"] if c in df_bpx.columns]
            diastolic_cols = [c for c in ["BPXDI1", "BPXDI2", "BPXDI3"] if c in df_bpx.columns]
            if systolic_cols:
                df_bpx["systolic_bp"] = df_bpx[systolic_cols].mean(axis=1)
            if diastolic_cols:
                df_bpx["diastolic_bp"] = df_bpx[diastolic_cols].mean(axis=1)
            
            df_ghb = get_dataset(config["ghb"])[["SEQN", "LBXGH"]]
            
            # Fetch liver enzymes as well
            biopro_cols = ["SEQN", "LBXSCR", "LBXSKSI", "LBXSNASI", "LBXSCA", "LBXSAL", "LBXSBU", "LBXSUA", "LBXSASSI", "LBXSATSI", "LBXSAPSI"]
            df_biopro = get_dataset(config["biopro"])[biopro_cols]
            
            df_crp = get_dataset(config["crp"])[["SEQN", "LBXCRP"]]
            df_cbc = get_dataset(config["cbc"])[["SEQN", "LBXHGB", "LBXWBCSI", "LBXPLTSI"]]
            df_trigly = get_dataset(config["trigly"])[["SEQN", "LBXTR", "LBDLDL"]]
            df_tchol = get_dataset(config["tchol"])[["SEQN", "LBXTC"]]
            df_hdl = get_dataset(config["hdl"])[["SEQN", "LBDHDD"]]
            
            # 2. Clean and Binarize outcomes:
            print("  Binarizing clinical outcomes...")
            df_diq_c = df_diq.copy()
            df_diq_c["diabetes_binary"] = df_diq_c["DIQ010"].apply(lambda x: 1.0 if x == 1.0 else (0.0 if x in [2.0, 3.0] else np.nan))
            
            df_bpq_c = df_bpq.copy()
            df_bpq_c["hypertension_binary"] = df_bpq_c["BPQ020"].apply(lambda x: 1.0 if x == 1.0 else (0.0 if x == 2.0 else np.nan))
            df_bpq_c["high_cholesterol_binary"] = df_bpq_c["BPQ080"].apply(lambda x: 1.0 if x == 1.0 else (0.0 if x == 2.0 else np.nan))
            
            df_mcq_c = df_mcq.copy()
            df_mcq_c["arthritis_binary"] = df_mcq_c["MCQ160A"].apply(lambda x: 1.0 if x == 1.0 else (0.0 if x == 2.0 else np.nan))
            df_mcq_c["liver_condition_binary"] = df_mcq_c["MCQ160L"].apply(lambda x: 1.0 if x == 1.0 else (0.0 if x == 2.0 else np.nan))
            df_mcq_c["thyroid_binary"] = df_mcq_c["MCQ160M"].apply(lambda x: 1.0 if x == 1.0 else (0.0 if x == 2.0 else np.nan))
            df_mcq_c["heart_attack_binary"] = df_mcq_c["MCQ160E"].apply(lambda x: 1.0 if x == 1.0 else (0.0 if x == 2.0 else np.nan))
            df_mcq_c["stroke_binary"] = df_mcq_c["MCQ160F"].apply(lambda x: 1.0 if x == 1.0 else (0.0 if x == 2.0 else np.nan))
            
            dpq_cols = [f"DPQ{i:03d}" for i in range(10, 100, 10)]
            existing_dep = [c for c in dpq_cols if c in df_dpq.columns]
            if len(existing_dep) == 9:
                dep_clean = df_dpq[["SEQN"] + existing_dep].copy()
                for col in existing_dep:
                    dep_clean[col] = dep_clean[col].apply(lambda x: 0.0 if not pd.isna(x) and x < 1e-5 else (x if x in [1.0, 2.0, 3.0] else np.nan))
                dep_clean = dep_clean.dropna()
                dep_clean["depression_score"] = dep_clean[existing_dep].sum(axis=1)
                dep_clean["depression_binary"] = (dep_clean["depression_score"] >= 10).astype(float)
            else:
                print(f"  Error: Missing PHQ-9 columns in cycle {config['name']}!")
                continue
                
            # 3. Merge sequentially using outer join on SEQN to keep all possible data
            print("  Merging datasets sequentially...")
            merged = df_demo
            merged = pd.merge(merged, df_diq_c[["SEQN", "diabetes_binary"]], on="SEQN", how="outer")
            merged = pd.merge(merged, df_bpq_c[["SEQN", "hypertension_binary", "high_cholesterol_binary"]], on="SEQN", how="outer")
            merged = pd.merge(merged, df_mcq_c[["SEQN", "arthritis_binary", "liver_condition_binary", "thyroid_binary", "heart_attack_binary", "stroke_binary"]], on="SEQN", how="outer")
            merged = pd.merge(merged, dep_clean[["SEQN", "depression_score", "depression_binary"]], on="SEQN", how="outer")
            merged = pd.merge(merged, df_bmx, on="SEQN", how="outer")
            merged = pd.merge(merged, df_bpx, on="SEQN", how="outer")
            merged = pd.merge(merged, df_ghb, on="SEQN", how="outer")
            merged = pd.merge(merged, df_biopro, on="SEQN", how="outer")
            merged = pd.merge(merged, df_crp, on="SEQN", how="outer")
            merged = pd.merge(merged, df_cbc, on="SEQN", how="outer")
            merged = pd.merge(merged, df_trigly, on="SEQN", how="outer")
            merged = pd.merge(merged, df_tchol, on="SEQN", how="outer")
            merged = pd.merge(merged, df_hdl, on="SEQN", how="outer")
            
            # Harmonize hs-CRP scale (LBXCRP mg/dL * 10.0 -> LBXHSCRP mg/L)
            merged["LBXHSCRP"] = merged["LBXCRP"] * 10.0
            
            merged["cycle_name"] = config["name"]
            print(f"  Cycle {config['name']} shape: {merged.shape}")
            all_cleaned.append(merged)
            
        except Exception as e:
            print(f"  Error in cycle {config['name']}: {e}")
            
    if len(all_cleaned) == 2:
        print("\nPooling cycles...")
        pooled_val = pd.concat(all_cleaned, axis=0, ignore_index=True)
        print(f"Pooled validation set shape: {pooled_val.shape}")
        
        # Save to file
        output_filename = "nhanes_comprehensive_validation_2007_2010.csv"
        pooled_val.to_csv(output_filename, index=False)
        print(f"SUCCESS: Saved comprehensive validation cohort to '{output_filename}'")
    else:
        print("Error: Could not pool both validation cohorts!")

if __name__ == "__main__":
    main()
