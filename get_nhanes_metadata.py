from pandas_nhanes import get_variables
import pandas as pd

def main():
    print("Fetching NHANES variable list...")
    variables = get_variables()
    print(f"Total variables found: {len(variables)}")
    
    # Save the full variable table to a CSV for reference
    variables.to_csv("nhanes_variables_catalog.csv", index=False)
    print("Saved variable catalog to nhanes_variables_catalog.csv")
    
    # Print summary of cycles
    print("\nUnique Cycles:")
    print(variables["cycle name"].unique())
    
    # Filter for the 2017-2018 cycle
    v_2017 = variables[variables["cycle name"] == "2017-2018"]
    print(f"\nVariables in 2017-2018 cycle: {len(v_2017)}")
    
    # Show some sample categories
    print("\nSample variables related to blood pressure, cholesterol, glucose, or diabetes:")
    keywords = ["blood pressure", "cholesterol", "glucose", "insulin", "diabetes", "liver", "kidney", "creatinine", "urea"]
    for kw in keywords:
        hits = v_2017[v_2017["variable explanation"].str.contains(kw, case=False, na=False)]
        print(f"- '{kw}': {len(hits)} variables (e.g., {hits['variable name'].head(3).tolist()})")

if __name__ == "__main__":
    main()
