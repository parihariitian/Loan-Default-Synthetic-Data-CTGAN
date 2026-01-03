"""import pandas as pd
import os

SYNTHETIC_FILE = os.path.join('data', 'processed', 'cleaned_data.csv')

try:
    df = pd.read_csv(SYNTHETIC_FILE)
    print("--- Checking Counts in Synthetic Data ---")
    print(df['Loan Status'].value_counts())
    print("\nRatio:")
    print(df['Loan Status'].value_counts(normalize=True))
except:
    print("File nahi mili.") """

"""import pandas as pd
import os
import joblib

print("--- Step 6: Generating Final Predictions (Submission Files) ---")

TEST_FILE = 'data/raw/test.csv'  
MODEL_FOLDER = 'models'
REAL_MODEL_PATH = os.path.join(MODEL_FOLDER, 'real_model1.pkl')
SYNTHETIC_MODEL_PATH = os.path.join(MODEL_FOLDER, 'synthetic_model1.pkl')
SUBMISSION_REAL = 'submission_real.csv'
SUBMISSION_SYNTHETIC = 'submission_synthetic.csv'

# Load Data
try:
    df_test = pd.read_csv(TEST_FILE)
    test_ids = df_test['ID']
    
    # Cleaning
    rename_map = {'Employment Duration': 'Home ownership', 'Home Ownership': 'Annual Income'}
    columns_to_drop = ['ID', 'Payment Plan', 'Loan Status'] 
    df_test_clean = df_test.rename(columns=rename_map).drop(columns=columns_to_drop, errors='ignore')

    # Load Models
    model_real = joblib.load(REAL_MODEL_PATH)
    model_synthetic = joblib.load(SYNTHETIC_MODEL_PATH)

    # PREDICT WITH THRESHOLD (Important: Wahi 0.40 threshold use karein)
    THRESHOLD = 0.40
    
    # Real Predictions
    probs_real = model_real.predict_proba(df_test_clean)[:, 1]
    preds_real = (probs_real >= THRESHOLD).astype(int)
    
    pd.DataFrame({'ID': test_ids, 'Loan Status': preds_real}).to_csv(SUBMISSION_REAL, index=False)
    print(f"✅ Saved Real Model predictions to: {SUBMISSION_REAL}")

    # Synthetic Predictions
    probs_synth = model_synthetic.predict_proba(df_test_clean)[:, 1]
    preds_synth = (probs_synth >= THRESHOLD).astype(int)
    
    pd.DataFrame({'ID': test_ids, 'Loan Status': preds_synth}).to_csv(SUBMISSION_SYNTHETIC, index=False)
    print(f"✅ Saved Synthetic Model predictions to: {SUBMISSION_SYNTHETIC}")

except Exception as e:
    print(f"Error: {e}")

import pandas as pd

# File load karein
sub_synth = pd.read_csv('submission_synthetic.csv')

print("--- Final Prediction Counts ---")
print(sub_synth['Loan Status'].value_counts())

print("\n--- Percentage ---")
print(sub_synth['Loan Status'].value_counts(normalize=True)) """

import pandas as pd
import os
import joblib

print("--- Adjusting Threshold to Balance Predictions ---")

# --- 1. Load Data & Model ---
TEST_FILE = 'data/raw/test.csv'
MODEL_PATH = os.path.join('models', 'real_model1.pkl') # Hum sirf Synthetic Model fix kar rahe hain abhi
SUBMISSION_FILE = 'submission_real_test_final.csv'

try:
    df_test = pd.read_csv(TEST_FILE)
    test_ids = df_test['ID']
    
    # Cleaning
    rename_map = {'Employment Duration': 'Home ownership', 'Home Ownership': 'Annual Income'}
    columns_to_drop = ['ID', 'Payment Plan', 'Loan Status'] 
    df_test_clean = df_test.rename(columns=rename_map).drop(columns=columns_to_drop, errors='ignore')

    model = joblib.load(MODEL_PATH)
    print("Model Loaded.")

except Exception as e:
    print(f"Error: {e}")
    exit()

# --- 2. Get Probabilities (Shaq ka Percentage) ---
# Hum seedha prediction nahi, probability mangenge
probs = model.predict_proba(df_test_clean)[:, 1]

# --- 3. Test Different Thresholds ---
print("\n--- Checking Counts for Different Thresholds ---")
print(f"Total Rows: {len(df_test)}")

thresholds = [0.50, 0.60, 0.70, 0.80]

for t in thresholds:
    preds = (probs >= t).astype(int)
    num_ones = sum(preds)
    percent = (num_ones / len(df_test)) * 100
    print(f"Threshold {t}:  1s = {num_ones}  ({percent:.2f}%)")

# --- 4. Generate Final File with Best Threshold ---
# Usually 10% se 20% ke beech '1' hone chahiye (Real world scenario)
# Jo threshold aapko sahi lage (maan lo 0.70), use yahan set karein
BEST_THRESHOLD = 0.55  # <-- Ise aap change kar sakte hain upar ka result dekh kar

final_preds = (probs >= BEST_THRESHOLD).astype(int)

submission_df = pd.DataFrame({
    'ID': test_ids,
    'Loan Status': final_preds
})

submission_df.to_csv(SUBMISSION_FILE, index=False)
print(f"\n✅ Final Submission saved with Threshold {BEST_THRESHOLD} to: {SUBMISSION_FILE}")
print(submission_df['Loan Status'].value_counts())