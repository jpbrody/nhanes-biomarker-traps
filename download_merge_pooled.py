import pandas as pd
import numpy as np
from pandas_nhanes import get_dataset
import os

def load_variables_catalog():
    if not os.path.exists("nhanes_variables_catalog.csv"):
        raise FileNotFoundError("nhanes_variables_catalog.csv not found! Run get_nhanes_metadata.py first.")
    return pd.read_csv("nhanes_variables_catalog.csv")

def download_and_clean_cycle(cycle, demo_dataset, suffix):
    print(f"\n==================================================")
    print(f"Downloading and cleaning NHANES cycle: {cycle}")
    print(f"==================================================")
    
    target_variables = {
        # Demographics
        "RIDAGEYR": "Age in years",
        "RIAGENDR": "Gender (1=Male, 2=Female)",
        
        # Outcomes (Self-reported conditions)
        "DIQ010": "Diabetes",
        "BPQ020": "Hypertension",
        "BPQ080": "High Cholesterol",
        "MCQ160a": "Arthritis",
        "MCQ160l": "Liver Condition",
        "MCQ160m": "Thyroid Problem",
        "MCQ160e": "Heart Attack",
        "MCQ160f": "Stroke",
        
        # Continuous physical/clinical markers
        "BMXBMI": "Body Mass Index",
        "BMXWAIST": "Waist Circumference",
        "BPXSY1": "Systolic BP 1",
        "BPXSY2": "Systolic BP 2",
        "BPXSY3": "Systolic BP 3",
        "BPXDI1": "Diastolic BP 1",
        "BPXDI2": "Diastolic BP 2",
        "BPXDI3": "Diastolic BP 3",
        
        # Blood chemistry, metabolic, and inflammatory markers
        "LBXGH": "Glycohemoglobin (HbA1c)",
        "LBXGLU": "Fasting Glucose",
        "LBXIN": "Insulin",
        "LBXTC": "Total Cholesterol",
        "LBDHDD": "HDL Cholesterol",
        "LBDLDL": "LDL Cholesterol",
        "LBXTR": "Triglycerides",
        "LBXSATSI": "ALT (Alanine Aminotransferase)",
        "LBXSASSI": "AST (Aspartate Aminotransferase)",
        "LBXSAPSI": "Alkaline Phosphatase",
        "LBXSCR": "Creatinine",
        "LBXSBU": "BUN (Blood Urea Nitrogen)",
        "LBXSUA": "Uric Acid",
        "LBXSNASI": "Sodium",
        "LBXSKSI": "Potassium",
        "LBXSCA": "Calcium",
        "LBXSAL": "Albumin",
        "LBXHSCRP": "High-Sensitivity CRP",
        "LBXHGB": "Hemoglobin",
        "LBXWBCSI": "White Blood Cell Count",
        "LBXPLTSI": "Platelet Count",
        
        # PHQ-9 Depression Items
        "DPQ010": "PHQ9 Q1",
        "DPQ020": "PHQ9 Q2",
        "DPQ030": "PHQ9 Q3",
        "DPQ040": "PHQ9 Q4",
        "DPQ050": "PHQ9 Q5",
        "DPQ060": "PHQ9 Q6",
        "DPQ070": "PHQ9 Q7",
        "DPQ080": "PHQ9 Q8",
        "DPQ090": "PHQ9 Q9",
    }
    
    catalog = load_variables_catalog()
    catalog_cycle = catalog[catalog["cycle name"] == cycle]
    
    var_to_dataset = {}
    dataset_to_vars = {}
    
    for var in target_variables.keys():
        if var in ["RIDAGEYR", "RIAGENDR"]:
            dataset_name = demo_dataset
        elif var.startswith("DPQ"):
            dataset_name = f"DPQ_{suffix}"
        else:
            match = catalog_cycle[catalog_cycle["variable name"] == var]
            if not match.empty:
                dataset_name = match.iloc[0]["dataset"]
            else:
                print(f"Warning: Variable '{var}' not found in catalog for cycle '{cycle}'! Skipping...")
                continue
                
        var_to_dataset[var] = dataset_name
        if dataset_name not in dataset_to_vars:
            dataset_to_vars[dataset_name] = []
        dataset_to_vars[dataset_name].append(var)
        
    merged_df = None
    
    for ds_name, vars_in_ds in dataset_to_vars.items():
        print(f"Downloading dataset '{ds_name}'...")
        try:
            ds_df = get_dataset(ds_name)
            if "SEQN" not in ds_df.columns:
                print(f"Warning: 'SEQN' not found in dataset '{ds_name}'! Skipping...")
                continue
                
            cols_in_ds = {c.upper(): c for c in ds_df.columns}
            cols_to_keep = ["SEQN"]
            for v in vars_in_ds:
                if v.upper() in cols_in_ds:
                    cols_to_keep.append(cols_in_ds[v.upper()])
                else:
                    print(f"Warning: {v} not found in downloaded dataset {ds_name} columns!")
            ds_subset = ds_df[cols_to_keep].copy()
            
            if merged_df is None:
                merged_df = ds_subset
            else:
                merged_df = pd.merge(merged_df, ds_subset, on="SEQN", how="outer")
        except Exception as e:
            print(f"Error downloading/merging dataset '{ds_name}': {e}")
            
    if merged_df is None:
        return None
        
    # Clean features
    systolic_cols = ["BPXSY1", "BPXSY2", "BPXSY3"]
    diastolic_cols = ["BPXDI1", "BPXDI2", "BPXDI3"]
    existing_sys = [c for c in systolic_cols if c in merged_df.columns]
    existing_dia = [c for c in diastolic_cols if c in merged_df.columns]
    
    if existing_sys:
        merged_df["systolic_bp"] = merged_df[existing_sys].mean(axis=1)
    if existing_dia:
        merged_df["diastolic_bp"] = merged_df[existing_dia].mean(axis=1)
        
    # Strictly Clean and Binarize PHQ-9 Depression
    depression_cols = [f"DPQ{i:03d}" for i in range(10, 100, 10)]
    existing_dep = [c for c in depression_cols if c in merged_df.columns]
    
    if len(existing_dep) == 9:
        print("Constructing strict PHQ-9 variables...")
        dep_clean = merged_df[existing_dep].copy()
        for col in existing_dep:
            # Map values < 1e-5 to 0.0 (resolves SAS precision issue), keep 1, 2, 3, and set all others to NaN
            dep_clean[col] = dep_clean[col].apply(lambda x: 0.0 if not pd.isna(x) and x < 1e-5 else (x if x in [1.0, 2.0, 3.0] else np.nan))
            
        # Drop subjects with ANY missing PHQ-9 items to guarantee perfect data quality
        dep_clean = dep_clean.dropna()
        merged_df = merged_df.loc[dep_clean.index].copy()
        
        merged_df["depression_score"] = dep_clean.sum(axis=1)
        merged_df["depression_binary"] = (merged_df["depression_score"] >= 10).astype(float)
    else:
        print("Error: PHQ-9 items are missing or incomplete in this cycle!")
        return None
        
    # Binarize other self-reported outcomes
    outcome_mappings = {
        "DIQ010": "diabetes_binary",
        "BPQ020": "hypertension_binary",
        "BPQ080": "high_cholesterol_binary",
        "MCQ160a": "arthritis_binary",
        "MCQ160l": "liver_condition_binary",
        "MCQ160m": "thyroid_binary",
        "MCQ160e": "heart_attack_binary",
        "MCQ160f": "stroke_binary"
    }
    
    for nhanes_var, new_col in outcome_mappings.items():
        actual_var = None
        for col in merged_df.columns:
            if col.upper() == nhanes_var.upper():
                actual_var = col
                break
                
        if actual_var is not None:
            if actual_var.upper() == "DIQ010":
                merged_df[new_col] = merged_df[actual_var].apply(lambda x: 1.0 if x == 1.0 else (0.0 if x in [2.0, 3.0] else np.nan))
            else:
                merged_df[new_col] = merged_df[actual_var].apply(lambda x: 1.0 if x == 1.0 else (0.0 if x == 2.0 else np.nan))
                
    return merged_df

def main():
    os.chdir("g:\\My Drive\\gemini\\NHANES")
    
    # 1. Download and clean 2015-2016 cycle
    df_15 = download_and_clean_cycle("2015-2016", "DEMO_I", "I")
    if df_15 is not None:
        df_15["cycle"] = 0.0 # 0.0 for 2015-2016
        print(f"2015-2016 cycle clean shape: {df_15.shape}, Depression Cases: {(df_15['depression_binary'] == 1.0).sum()}")
        
    # 2. Download and clean 2017-2018 cycle
    df_17 = download_and_clean_cycle("2017-2018", "DEMO_J", "J")
    if df_17 is not None:
        df_17["cycle"] = 1.0 # 1.0 for 2017-2018
        print(f"2017-2018 cycle clean shape: {df_17.shape}, Depression Cases: {(df_17['depression_binary'] == 1.0).sum()}")
        
    if df_15 is None or df_17 is None:
        print("Error: Incomplete download or merge. Cannot pool.")
        return
        
    # 3. Concatenate (Pool) both cycles
    print("\nPooling datasets...")
    # Get intersection of columns to ensure compatibility
    common_cols = list(set(df_15.columns).intersection(set(df_17.columns)))
    
    pooled_df = pd.concat([df_15[common_cols], df_17[common_cols]], axis=0, ignore_index=True)
    
    output_filename = "nhanes_depression_pooled.csv"
    pooled_df.to_csv(output_filename, index=False)
    
    print(f"\n==================================================")
    print(f"Saved POOLED depression dataset to '{output_filename}'")
    print(f"Total shape: {pooled_df.shape}")
    print(f"Total Completed PHQ-9 Surveys: {len(pooled_df)}")
    
    counts = pooled_df["depression_binary"].value_counts(dropna=False)
    print(f"Depression status: Yes={counts.get(1.0, 0)}, No={counts.get(0.0, 0)}, NaN={counts.get(np.nan, 0)}")
    print(f"Overall Depression Prevalence: {counts.get(1.0, 0)/len(pooled_df)*100:.2f}%")
    print(f"==================================================")

if __name__ == "__main__":
    main()
