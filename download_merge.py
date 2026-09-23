import pandas as pd
import numpy as np
from pandas_nhanes import get_dataset
import os

def load_variables_catalog():
    if not os.path.exists("nhanes_variables_catalog.csv"):
        raise FileNotFoundError("nhanes_variables_catalog.csv not found! Run get_nhanes_metadata.py first.")
    return pd.read_csv("nhanes_variables_catalog.csv")

def main():
    cycle = "2017-2018"
    print(f"Starting download and merge process for NHANES cycle: {cycle}")
    
    # 1. Define variables we want to retrieve
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
        
        # Depression (PHQ-9 questions)
        "DPQ010": "Depression Q1",
        "DPQ020": "Depression Q2",
        "DPQ030": "Depression Q3",
        "DPQ040": "Depression Q4",
        "DPQ050": "Depression Q5",
        "DPQ060": "Depression Q6",
        "DPQ070": "Depression Q7",
        "DPQ080": "Depression Q8",
        "DPQ090": "Depression Q9",
        
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
    }
    
    # 2. Map variables to their datasets
    catalog = load_variables_catalog()
    catalog_cycle = catalog[catalog["cycle name"] == cycle]
    
    # We need to map each variable to its dataset code
    var_to_dataset = {}
    dataset_to_vars = {}
    
    print("\nMapping variables to NHANES datasets:")
    for var in target_variables.keys():
        match = catalog_cycle[catalog_cycle["variable name"] == var]
        if not match.empty:
            dataset_name = match.iloc[0]["dataset"]
            var_to_dataset[var] = dataset_name
            if dataset_name not in dataset_to_vars:
                dataset_to_vars[dataset_name] = []
            dataset_to_vars[dataset_name].append(var)
        else:
            print(f"Warning: Variable '{var}' ({target_variables[var]}) not found in catalog for cycle '{cycle}'!")
            
    print(f"Grouped variables into {len(dataset_to_vars)} datasets.")
    for ds, vars_in_ds in dataset_to_vars.items():
        print(f" - {ds}: {vars_in_ds}")
        
    # 3. Download and merge datasets
    merged_df = None
    
    print("\nDownloading and merging datasets sequentially...")
    for ds_name, vars_in_ds in dataset_to_vars.items():
        print(f"Downloading dataset '{ds_name}'...")
        try:
            ds_df = get_dataset(ds_name)
            # Ensure SEQN is in the columns
            if "SEQN" not in ds_df.columns:
                print(f"Warning: 'SEQN' not found in dataset '{ds_name}'! Skipping...")
                continue
                
            # Filter dataset to only SEQN + the target variables we need (case-insensitive check)
            cols_in_ds = {c.upper(): c for c in ds_df.columns}
            cols_to_keep = ["SEQN"]
            for v in vars_in_ds:
                if v.upper() in cols_in_ds:
                    cols_to_keep.append(cols_in_ds[v.upper()])
                else:
                    print(f"Warning: {v} not found in downloaded dataset {ds_name} columns!")
            ds_subset = ds_df[cols_to_keep].copy()
            
            # Merge on SEQN
            if merged_df is None:
                merged_df = ds_subset
                print(f"Initial dataset: {ds_name} with shape {merged_df.shape}")
            else:
                merged_df = pd.merge(merged_df, ds_subset, on="SEQN", how="outer")
                print(f"Merged {ds_name}, new shape: {merged_df.shape}")
        except Exception as e:
            print(f"Error downloading/merging dataset '{ds_name}': {e}")
            
    if merged_df is None:
        print("Error: No datasets were merged!")
        return
        
    # 4. Clean and engineer features
    print("\nCleaning and engineering features...")
    
    # Blood pressure average
    systolic_cols = ["BPXSY1", "BPXSY2", "BPXSY3"]
    diastolic_cols = ["BPXDI1", "BPXDI2", "BPXDI3"]
    
    # Check if they exist
    existing_sys = [c for c in systolic_cols if c in merged_df.columns]
    existing_dia = [c for c in diastolic_cols if c in merged_df.columns]
    
    if existing_sys:
        merged_df["systolic_bp"] = merged_df[existing_sys].mean(axis=1)
    if existing_dia:
        merged_df["diastolic_bp"] = merged_df[existing_dia].mean(axis=1)
        
    # Depression Score (PHQ-9)
    depression_cols = [f"DPQ{i:03d}" for i in range(10, 100, 10)]
    existing_dep = [c for c in depression_cols if c in merged_df.columns]
    
    if len(existing_dep) == 9:
        print("Constructing PHQ-9 depression variables...")
        # In NHANES, depression questions are coded:
        # 0: Not at all, 1: Several days, 2: More than half the days, 3: Nearly every day
        # 7: Refused, 9: Don't know
        # Let's clean the columns (map 7 and 9 to NaN, others kept)
        dep_clean = merged_df[existing_dep].copy()
        for col in existing_dep:
            dep_clean[col] = dep_clean[col].apply(lambda x: x if x in [0.0, 1.0, 2.0, 3.0] else np.nan)
            
        merged_df["depression_score"] = dep_clean.sum(axis=1, min_count=5) # require at least 5 completed questions
        merged_df["depression_binary"] = (merged_df["depression_score"] >= 10).astype(float)
        # If depression_score is NaN, keep depression_binary as NaN
        merged_df.loc[merged_df["depression_score"].isna(), "depression_binary"] = np.nan
        
    # Binarize self-reported outcomes
    # NHANES binary questionnaire variables are usually coded: 1=Yes, 2=No, 7=Refused, 9=Don't know.
    # We map 1 -> 1.0, 2 -> 0.0, and others to NaN.
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
        # Find the actual case-sensitive column name in merged_df
        actual_var = None
        for col in merged_df.columns:
            if col.upper() == nhanes_var.upper():
                actual_var = col
                break
                
        if actual_var is not None:
            print(f"Binarizing {actual_var} -> {new_col}")
            # Map 1 to 1, 2 to 0, and all other codes (like 3 for borderline diabetes, 7, 9) to NaN or 0
            if actual_var.upper() == "DIQ010":
                # For diabetes, let's treat 1 (Yes) as 1, 2 (No) and 3 (Borderline) as 0, others as NaN
                merged_df[new_col] = merged_df[actual_var].apply(lambda x: 1.0 if x == 1.0 else (0.0 if x in [2.0, 3.0] else np.nan))
            else:
                merged_df[new_col] = merged_df[actual_var].apply(lambda x: 1.0 if x == 1.0 else (0.0 if x == 2.0 else np.nan))
        else:
            print(f"Warning: Could not binarize {nhanes_var} because it was not merged!")
                
    # Output file
    output_filename = "nhanes_merged_2017_2018.csv"
    merged_df.to_csv(output_filename, index=False)
    print(f"\nSaved final merged dataset to '{output_filename}'")
    print(f"Final shape: {merged_df.shape}")
    print(f"Non-null counts for key outcomes:")
    for new_col in list(outcome_mappings.values()) + (["depression_binary"] if "depression_binary" in merged_df.columns else []):
        if new_col in merged_df.columns:
            counts = merged_df[new_col].value_counts(dropna=False)
            print(f" - {new_col}: Yes={counts.get(1.0, 0)}, No={counts.get(0.0, 0)}, NaN={counts.get(np.nan, 0)}")

if __name__ == "__main__":
    main()
