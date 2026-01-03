import pandas as pd
import os
import joblib

print("--- Step 6: Final Prediction with Custom Thresholds ---")

# --- 1. Settings ---
TEST_FILE = 'data/raw/test.csv'
MODEL_FOLDER = 'models'
REAL_MODEL_PATH = os.path.join(MODEL_FOLDER, 'real_model1.pkl')
SYNTHETIC_MODEL_PATH = os.path.join(MODEL_FOLDER, 'synthetic_model1.pkl')

# Aapke Set kiye hue Thresholds
THRESH_REAL = 0.55
THRESH_SYNTH = 0.65

# --- 2. Load Data & Models ---
try:
    df_test = pd.read_csv(TEST_FILE)
    test_ids = df_test['ID']
    
    # Cleaning
    rename_map = {'Employment Duration': 'Home ownership', 'Home Ownership': 'Annual Income'}
    columns_to_drop = ['ID', 'Payment Plan', 'Loan Status'] 
    df_test_clean = df_test.rename(columns=rename_map).drop(columns=columns_to_drop, errors='ignore')

    model_real = joblib.load(REAL_MODEL_PATH)
    model_synthetic = joblib.load(SYNTHETIC_MODEL_PATH)
    print("Models Loaded.")

except Exception as e:
    print(f"Error: {e}")
    exit()

# --- 3. Generate Predictions ---

# Real Model Prediction (Threshold 0.55)
probs_real = model_real.predict_proba(df_test_clean)[:, 1]
preds_real = (probs_real >= THRESH_REAL).astype(int)

# Synthetic Model Prediction (Threshold 0.65)
probs_synth = model_synthetic.predict_proba(df_test_clean)[:, 1]
preds_synth = (probs_synth >= THRESH_SYNTH).astype(int)



# --- 5. THE COMPARISON REPORT  ---
count_real = sum(preds_real)
count_synth = sum(preds_synth)

# Overlap: Kitne logon ko dono ne '1' bola?
overlap = sum((preds_real == 1) & (preds_synth == 1))

print("\n" + "="*40)
print("      FINAL COMPARISON REPORT      ")
print("="*40)
print(f"Total Test Rows:      {len(df_test)}")
print("-" * 40)
print(f"REAL Model (Thresh {THRESH_REAL}):")
print(f"  -> Predicted Defaulters: {count_real}")
print(f"  -> Percentage:           {(count_real/len(df_test))*100:.2f}%")
print("-" * 40)
print(f"SYNTHETIC Model (Thresh {THRESH_SYNTH}):")
print(f"  -> Predicted Defaulters: {count_synth}")
print(f"  -> Percentage:           {(count_synth/len(df_test))*100:.2f}%")
print("-" * 40)
print(f"AGREEMENT (Overlap):")
print(f"  -> Both agreed on:       {overlap} defaulters")
print("="*40)

print("\nFiles saved: 'final_submission_REAL.csv' & 'final_submission_SYNTHETIC.csv'")