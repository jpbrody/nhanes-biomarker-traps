from pandas_nhanes import get_cycle_variables
import pandas as pd
import numpy as np

def main():
    print("Downloading/loading NHANES 2017-2018 data...")
    
    # Define outcomes and clinical markers to fetch
    variables_to_get = [
        # Outcomes (Self-reported conditions)
        "DIQ010",    # Doctor told you have diabetes? (1=Yes, 2=No, 3=Borderline)
        "BPQ020",    # Ever told you had high blood pressure? (1=Yes, 2=No)
        "BPQ080",    # Ever told you had high cholesterol? (1=Yes, 2=No)
        "MCQ160A",   # Ever told you had arthritis? (1=Yes, 2=No)
        "MCQ160L",   # Ever told you had any liver condition? (1=Yes, 2=No)
        "MCQ160M",   # Ever told you had a thyroid problem? (1=Yes, 2=No)
        "MCQ160E",   # Ever told you had a heart attack? (1=Yes, 2=No)
        "MCQ160F",   # Ever told you had a stroke? (1=Yes, 2=No)
        
        # Demographic
        "RIDAGEYR",  # Age in years
        "RIAGENDR",  # Gender (1=Male, 2=Female)
        
        # Clinical & Physical Markers
        "BMXBMI",    # Body Mass Index (kg/m^2)
        "BMXWAIST",  # Waist Circumference (cm)
        "BPXSY1",    # Systolic blood pressure (reading 1)
        "BPXSY2",    # Systolic blood pressure (reading 2)
        "BPXSY3",    # Systolic blood pressure (reading 3)
        "BPXDI1",    # Diastolic blood pressure (reading 1)
        "BPXDI2",    # Diastolic blood pressure (reading 2)
        "BPXDI3",    # Diastolic blood pressure (reading 3)
        
        # Laboratory Blood Chemistry / Lipids / Metabolic
        "LBXGH",     # Glycohemoglobin (HbA1c, %)
        "LBXGLU",    # Fasting glucose (mg/dL)
        "LBXIN",     # Insulin (uIU/mL)
        "LBXTC",     # Total Cholesterol (mg/dL)
        "LBDHDD",    # HDL Cholesterol (mg/dL)
        "LBDLDL",    # LDL Cholesterol (mg/dL)
        "LBXTR",     # Triglycerides (mg/dL)
        
        # Liver Function
        "LBXSALT",   # ALT / SGPT (U/L)
        "LBXSAST",   # AST / SGOT (U/L)
        "LBXSAPSI",  # Alkaline Phosphatase (U/L)
        
        # Kidney Function / Inflammatory
        "LBXSCR",    # Creatinine (mg/dL)
        "LBXSBU",    # Blood Urea Nitrogen (mg/dL)
        "LBXSUA",    # Uric Acid (mg/dL)
        "LBXHSCRP",  # High-Sensitivity C-Reactive Protein (mg/L)
        
        # Electrolytes
        "LBXSN",     # Sodium (mmol/L)
        "LBXSK",     # Potassium (mmol/L)
        "LBXSCA",    # Calcium (mg/dL)
    ]
    
    # Download the data (will print log to stdout)
    df = get_cycle_variables("2017-2018", *variables_to_get)
    
    print("\nDataset loaded successfully!")
    print(f"Shape: {df.shape}")
    
    # Create derived/cleaned columns
    print("\nCleaning and engineering columns...")
    
    # Average blood pressure
    df["systolic_bp"] = df[["BPXSY1", "BPXSY2", "BPXSY3"]].mean(axis=1)
    df["diastolic_bp"] = df[["BPXDI1", "BPXDI2", "BPXDI3"]].mean(axis=1)
    
    # Print data summary
    print("\nSummary of missingness and non-null values:")
    missing_summary = pd.DataFrame({
        "Non-Null Count": df.count(),
        "Missing Count": df.isnull().sum(),
        "Missing %": (df.isnull().mean() * 100).round(1)
    })
    print(missing_summary.to_string())
    
    # Save the raw data for analysis
    df.to_csv("nhanes_raw_2017_2018.csv", index=False)
    print("\nSaved raw dataset to nhanes_raw_2017_2018.csv")

if __name__ == "__main__":
    main()
