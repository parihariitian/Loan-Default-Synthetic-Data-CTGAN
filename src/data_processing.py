
import pandas as pd
def print_feature_names(dataset):
    for feature in dataset.columns:
        print(feature)
data = pd.read_csv("data/raw/train.csv")
data1 = pd.read_csv("data/raw/test.csv")
#print_feature_names(data)

#print(len(data.columns))

import os

PROCESSED_FOLDER = 'data/processed'
OUTPUT_FILE = os.path.join(PROCESSED_FOLDER, 'cleaned_data.csv')
out_test = os.path.join(PROCESSED_FOLDER, 'cleaned_test.csv')

# --- 4. Define Column Changes ---
rename_map = {
    'Employment Duration': 'Home ownership',
    'Home Ownership': 'Annual Income'
}

columns_to_drop = [
    'ID',           # Just an identifier, not predictive
    'Payment Plan'  # Only has one value ('n'), so it's useless
]

# --- 5. Apply Changes ---
df_cleaned = data.rename(columns=rename_map)
df_test = data1.rename(columns=rename_map)
print("Successfully renamed columns.")

df_cleaned = df_cleaned.drop(columns=columns_to_drop, errors='ignore')
df_test = df_test.drop(columns=columns_to_drop, errors='ignore')
print("Successfully dropped useless columns.")

# --- 6. Save the Cleaned Data ---
try:
    df_cleaned.to_csv(OUTPUT_FILE, index=False)
    df_test.to_csv(out_test, index=False)
    print(f"\nSUCCESS! Cleaned data has been saved to: {OUTPUT_FILE}")
    print("\n--- First 5 rows of your new cleaned data ---")
    print(df_cleaned.head())
    
except PermissionError:
    print(f"\nERROR: Permission denied. Could not save to {OUTPUT_FILE}.")
    print("Please make sure you have write permissions, or that the file is not open elsewhere.")
    
print("\n--- Step 1: Preprocessing Script Finished ---") 