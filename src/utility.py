import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score

print("--- Step: Utility Check (TSTR Evaluation) ---")

# --- 1. Define Paths ---
REAL_DATA_FILE = os.path.join('data', 'processed', 'cleaned_data.csv')
MODEL_FOLDER = 'models'
REAL_MODEL_PATH = os.path.join(MODEL_FOLDER, 'real_model1.pkl')
SYNTHETIC_MODEL_PATH = os.path.join(MODEL_FOLDER, 'synthetic_model1.pkl')
TARGET_COLUMN = 'Loan Status'

# --- 2. Load Data & Re-create Test Set ---
# Humein wahi test set chahiye jo training ke waqt alag kiya tha
try:
    df_real_full = pd.read_csv(REAL_DATA_FILE)
    
    # Target ko string banaya tha (kyunki CTGAN waisa hi chahta tha)
    df_real_full[TARGET_COLUMN] = df_real_full[TARGET_COLUMN].astype(str)
    
    # SPLIT (Wahi random_state=42 use karein taaki same test set mile)
    train_data, test_data = train_test_split(
        df_real_full, 
        test_size=0.2, 
        random_state=42, 
        stratify=df_real_full[TARGET_COLUMN]
    )
    
    X_test = test_data.drop(columns=[TARGET_COLUMN])
    y_test = test_data[TARGET_COLUMN]
    
    print(f"Test Data Loaded: {len(X_test)} rows")

except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# --- 3. Load Models ---
try:
    model_real = joblib.load(REAL_MODEL_PATH)
    model_synth = joblib.load(SYNTHETIC_MODEL_PATH)
    print("Models Loaded Successfully (Gradient Boosting).")
except:
    print("Error: Models nahi mile. Check paths.")
    exit()

# --- 4. Prediction Helper Function (With Thresholds) ---
def get_predictions(model, X, threshold=0.5):
    # Probability nikalo (Class 1 hone ka chance)
    probs = model.predict_proba(X)[:, 1]
    # Agar probability > threshold hai, toh '1' bolo, warna '0'
    preds = (probs >= threshold).astype(int).astype(str)
    return preds

# --- 5. Run Utility Check ---

# Thresholds wahi rakhenge jo humne decide kiye the
THRESH_REAL = 0.55
THRESH_SYNTH = 0.65

print(f"\nEvaluating Real Model (Threshold: {THRESH_REAL})...")
y_pred_real = get_predictions(model_real, X_test, threshold=THRESH_REAL)

print(f"Evaluating Synthetic Model (Threshold: {THRESH_SYNTH})...")
y_pred_synth = get_predictions(model_synth, X_test, threshold=THRESH_SYNTH)

# --- 6. Generate Comparison Report ---

print("\n" + "="*50)
print("      UTILITY CHECK REPORT (TSTR)      ")
print("="*50)

# Function to print nice stats
def print_stats(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    # Recall (Class 1) sabse zaroori hai
    report = classification_report(y_true, y_pred, output_dict=True)
    recall_1 = report['1']['recall']
    precision_1 = report['1']['precision']
    
    print(f"--- {name} ---")
    print(f"Accuracy:      {acc:.2%}")
    print(f"Recall (1):    {recall_1:.2%}")
    print(f"Precision (1): {precision_1:.2%}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("-" * 30)

print_stats("REAL DATA MODEL", y_test, y_pred_real)
print_stats("SYNTHETIC DATA MODEL", y_test, y_pred_synth)

# Final Verdict

recall_real = classification_report(y_test, y_pred_real, output_dict=True)['1']['recall']
recall_synth = classification_report(y_test, y_pred_synth, output_dict=True)['1']['recall']
recall_real = classification_report(y_test, y_pred_real, output_dict=True)['1']['recall']
recall_synth = classification_report(y_test, y_pred_synth, output_dict=True)['1']['recall']

if recall_synth >= recall_real:
    print(" SUCCESS: The Synthetic Model's Utility is superior or equal to the Real Model!")
    print(f"   (Synthetic Recall: {recall_synth:.2%} >= Real Recall: {recall_real:.2%})")
else:
    print("⚠️ NOTE: The Synthetic Model performs slightly lower than the Real Model.")
    print(f"   (Synthetic Recall: {recall_synth:.2%} < Real Recall: {recall_real:.2%})")