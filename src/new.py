import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier # <-- Naya aur Tez Model
from sklearn.utils import resample
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

print("--- Step 5 (FINAL COMBO): Oversampling + Gradient Boosting + Threshold ---")

# --- 1. Define Paths ---
REAL_DATA_FILE = os.path.join('data', 'processed', 'cleaned_data.csv')
SYNTHETIC_TRAIN_FILE = os.path.join('data', 'synthetic', 'synthetic_data.csv')
TARGET_COLUMN = 'Loan Status'
MODEL_FOLDER = 'models'
REAL_MODEL_PATH = os.path.join(MODEL_FOLDER, 'real_model1.pkl')
SYNTHETIC_MODEL_PATH = os.path.join(MODEL_FOLDER, 'synthetic_model1.pkl')

if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)

# --- 2. Load Data ---
df_real_full = pd.read_csv(REAL_DATA_FILE)
df_synthetic_train = pd.read_csv(SYNTHETIC_TRAIN_FILE)

df_real_full[TARGET_COLUMN] = df_real_full[TARGET_COLUMN].astype(str)
df_synthetic_train[TARGET_COLUMN] = df_synthetic_train[TARGET_COLUMN].astype(str)

# --- 3. Split Real Data ---
df_real_train, df_real_test = train_test_split(
    df_real_full, test_size=0.2, random_state=42, stratify=df_real_full[TARGET_COLUMN]
)

# --- 4. OVERSAMPLING FUNCTION ---
def balance_data(df, target_col):
    df_majority = df[df[target_col] == '0']
    df_minority = df[df[target_col] == '1']
    
    df_minority_upsampled = resample(df_minority, 
                                     replace=True,     
                                     n_samples=len(df_majority),    
                                     random_state=42) 
    
    df_balanced = pd.concat([df_majority, df_minority_upsampled])
    return df_balanced

# Balance Training Data
df_real_train_balanced = balance_data(df_real_train, TARGET_COLUMN)
df_synthetic_train_balanced = balance_data(df_synthetic_train, TARGET_COLUMN)

# Separate X and y
X_real_train = df_real_train_balanced.drop(columns=[TARGET_COLUMN])
y_real_train = df_real_train_balanced[TARGET_COLUMN]

X_synthetic_train = df_synthetic_train_balanced.drop(columns=[TARGET_COLUMN])
y_synthetic_train = df_synthetic_train_balanced[TARGET_COLUMN]

X_real_test = df_real_test.drop(columns=[TARGET_COLUMN], errors='ignore')
y_real_test = df_real_test[TARGET_COLUMN]

# --- 5. Preprocessing ---
categorical_cols = X_real_train.select_dtypes(include=['object', 'category']).columns
numerical_cols = X_real_train.select_dtypes(include=['number']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ],
    remainder='passthrough'
)

# --- 6. Train Gradient Boosting Models ---
# Yeh model 'RandomForest' se thoda better seekhta hai mushkil data par
print("\nTraining Model A (Real) with Gradient Boosting...")
model_A = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(random_state=42, n_estimators=100))
])
model_A.fit(X_real_train, y_real_train)
joblib.dump(model_A, REAL_MODEL_PATH)

print("Training Model B (Synthetic) with Gradient Boosting...")
model_B = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(random_state=42, n_estimators=100))
])
model_B.fit(X_synthetic_train, y_synthetic_train)
joblib.dump(model_B, SYNTHETIC_MODEL_PATH)

# --- 7. PREDICT WITH THRESHOLD FUNCTION ---
def predict_with_threshold(model, X, threshold=0.5):
    probs = model.predict_proba(X)
    prob_class_1 = probs[:, 1]
    predictions = (prob_class_1 >= threshold).astype(int).astype(str)
    return predictions

# --- 8. Evaluation with LOW THRESHOLD ---
# Humne data balance kar diya hai, phir bhi hum threshold thoda girayenge
# Taaki model aur zyada sensitive ho jaye
THRESHOLD = 0.40 

print(f"\n--- Evaluating with Threshold: {THRESHOLD} ---")

print("\n=== MODEL A (Real Data) Results ===")
y_pred_A = predict_with_threshold(model_A, X_real_test, threshold=THRESHOLD)
print(confusion_matrix(y_real_test, y_pred_A))
print(classification_report(y_real_test, y_pred_A))

print("\n=== MODEL B (Synthetic Data) Results ===")
y_pred_B = predict_with_threshold(model_B, X_real_test, threshold=THRESHOLD)
print(confusion_matrix(y_real_test, y_pred_B))
print(classification_report(y_real_test, y_pred_B))

print("\n--- Step 5 Finished ---")