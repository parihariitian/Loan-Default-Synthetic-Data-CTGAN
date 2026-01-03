import pandas as pd
import os
import joblib 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import warnings

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

print("--- Step 5 (Advanced): Train, Save, Load, and Test ---")

# --- 1. Define Paths ---
REAL_TRAIN_FILE = os.path.join('data', 'processed', 'cleaned_data.csv')
SYNTHETIC_TRAIN_FILE = os.path.join('data', 'synthetic', 'synthetic_data.csv')
REAL_TEST_FILE = 'data/processed/cleaned_data.csv'
TARGET_COLUMN = 'Loan Status'

# Yeh 2 nayi models save karne ki jagah
MODEL_FOLDER = 'models'
REAL_MODEL_PATH = os.path.join(MODEL_FOLDER, 'real_model.pkl')
SYNTHETIC_MODEL_PATH = os.path.join(MODEL_FOLDER, 'synthetic_model.pkl')

if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)

# --- 2. Load Data ---
print("Loading all data files...")
try:
    df_real_train = pd.read_csv(REAL_TRAIN_FILE)
    df_synthetic_train = pd.read_csv(SYNTHETIC_TRAIN_FILE)
    df_real_test = pd.read_csv(REAL_TEST_FILE)
except FileNotFoundError as e:
    print(f"\nERROR: Could not find a file. {e}")
    exit()



# --- 4. Separate X (features) and y (target) ---
print("Separating features (X) and target (y)...")
df_real_train[TARGET_COLUMN] = df_real_train[TARGET_COLUMN].astype(str)
df_synthetic_train[TARGET_COLUMN] = df_synthetic_train[TARGET_COLUMN].astype(str)
df_real_test[TARGET_COLUMN] = df_real_test[TARGET_COLUMN].astype(str)

X_real_train = df_real_train.drop(columns=[TARGET_COLUMN])
y_real_train = df_real_train[TARGET_COLUMN]
X_synthetic_train = df_synthetic_train.drop(columns=[TARGET_COLUMN])
y_synthetic_train = df_synthetic_train[TARGET_COLUMN]
X_real_test = df_real_test.drop(columns=[TARGET_COLUMN], errors='ignore')
y_real_test = df_real_test[TARGET_COLUMN]

# --- 5. Create Preprocessing Pipeline ---
categorical_cols = X_real_train.select_dtypes(include=['object', 'category']).columns
numerical_cols = X_real_train.select_dtypes(include=['number']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ],
    remainder='passthrough'
)

# --- 6. Model A: Train and Save (REAL DATA) ---
print(f"\n--- Training Model A (Real Data) and saving to {REAL_MODEL_PATH} ---")
model_A = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=100))
])
model_A.fit(X_real_train, y_real_train)
joblib.dump(model_A, REAL_MODEL_PATH)
print("Model A trained and saved.")

# --- 7. Model B: Train and Save (SYNTHETIC DATA) ---
print(f"\n--- Training Model B (Synthetic Data) and saving to {SYNTHETIC_MODEL_PATH} ---")
model_B = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=100))
])
model_B.fit(X_synthetic_train, y_synthetic_train)
joblib.dump(model_B, SYNTHETIC_MODEL_PATH)
print("Model B trained and saved.")

# --- 8. Load and Test Models ---
print("\n--- Loading models and testing on 'test.csv' ---")

# Load and Test Model A
loaded_model_A = joblib.load(REAL_MODEL_PATH)
y_pred_A = loaded_model_A.predict(X_real_test)
accuracy_A = accuracy_score(y_real_test, y_pred_A)
report_A = classification_report(y_real_test, y_pred_A)

# Load and Test Model B
loaded_model_B = joblib.load(SYNTHETIC_MODEL_PATH)
y_pred_B = loaded_model_B.predict(X_real_test)
accuracy_B = accuracy_score(y_real_test, y_pred_B)
report_B = classification_report(y_real_test, y_pred_B)

# --- 9. Final Report ---
print("\n\n--- TSTR EVALUATION RESULTS ---")
print("====================================")
print(f"     MODEL A (from {REAL_MODEL_PATH})")
print("   (Trained on REAL, Tested on REAL)")
print("====================================")
print(f"Accuracy: {accuracy_A*100:.2f}%")
print(report_A)

print("\n====================================")
print(f"     MODEL B (from {SYNTHETIC_MODEL_PATH})")
print(" (Trained on SYNTHETIC, Tested on REAL)")
print("====================================")
print(f"Accuracy: {accuracy_B*100:.2f}%")
print(report_B)
