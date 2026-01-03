"""import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.metrics import confusion_matrix

print("--- Step 7 (Bonus): Model vs Model Comparison Plots ---")

# --- 1. Settings ---
TEST_FILE = 'data/raw/test.csv'
MODEL_FOLDER = 'models'
PLOT_FOLDER = 'comparison_plots'
REAL_MODEL_PATH = os.path.join(MODEL_FOLDER, 'real_model1.pkl')
SYNTHETIC_MODEL_PATH = os.path.join(MODEL_FOLDER, 'synthetic_model1.pkl')

# Wahi Thresholds jo aapne decide kiye
THRESH_REAL = 0.55
THRESH_SYNTH = 0.65

if not os.path.exists(PLOT_FOLDER):
    os.makedirs(PLOT_FOLDER)

# --- 2. Load Data & Models ---
try:
    df_test = pd.read_csv(TEST_FILE)
    
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

# --- 3. Get Predictions & Probabilities ---

# Real Model
probs_real = model_real.predict_proba(df_test_clean)[:, 1]
preds_real = (probs_real >= THRESH_REAL).astype(int)

# Synthetic Model
probs_synth = model_synthetic.predict_proba(df_test_clean)[:, 1]
preds_synth = (probs_synth >= THRESH_SYNTH).astype(int)

# --- 4. PLOT 1: Prediction Counts (Bar Chart) ---
print("Generating Prediction Count Plot...")

# Data prepare karna plotting ke liye
count_data = pd.DataFrame({
    'Model': ['Real Model']*len(preds_real) + ['Synthetic Model']*len(preds_synth),
    'Prediction': list(preds_real) + list(preds_synth)
})

plt.figure(figsize=(8, 6))
sns.countplot(data=count_data, x='Prediction', hue='Model', palette=['blue', 'orange'])
plt.title('Prediction Counts Comparison (0 vs 1)')
plt.xlabel('Predicted Loan Status (0=Safe, 1=Default)')
plt.ylabel('Number of Customers')
plt.savefig(os.path.join(PLOT_FOLDER, 'model_prediction_counts.png'))
plt.close()

# --- 5. PLOT 2: Confidence Distribution (KDE Plot) ---
print("Generating Probability Distribution Plot...")

plt.figure(figsize=(10, 6))
sns.kdeplot(probs_real, fill=True, label='Real Model', color='blue', alpha=0.3)
sns.kdeplot(probs_synth, fill=True, label='Synthetic Model', color='orange', alpha=0.3)

# Threshold lines draw karna
plt.axvline(THRESH_REAL, color='blue', linestyle='--', label=f'Real Threshold ({THRESH_REAL})')
plt.axvline(THRESH_SYNTH, color='orange', linestyle='--', label=f'Synth Threshold ({THRESH_SYNTH})')

plt.title('Model Confidence Comparison (Probability Distribution)')
plt.xlabel('Probability of Default (0.0 to 1.0)')
plt.ylabel('Density')
plt.legend()
plt.savefig(os.path.join(PLOT_FOLDER, 'model_confidence_dist.png'))
plt.close()

# --- 6. PLOT 3: Agreement Heatmap ---
print("Generating Agreement Heatmap...")

# Confusion Matrix between models (Real ko 'Actual' maan lo comparison ke liye)
cm = confusion_matrix(preds_real, preds_synth)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Synth Says 0', 'Synth Says 1'],
            yticklabels=['Real Says 0', 'Real Says 1'])
plt.title('Do the Models Agree? (Overlap Check)')
plt.ylabel('Real Model Prediction')
plt.xlabel('Synthetic Model Prediction')
plt.savefig(os.path.join(PLOT_FOLDER, 'model_agreement_heatmap.png'))
plt.close()

print(f"\n✅ All comparison plots saved in '{PLOT_FOLDER}'")"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore')

print("--- Step 7: Generating Distribution Comparison Plots ---")

# --- 1. Define Paths & Load Data ---
REAL_FILE = os.path.join('data', 'processed', 'cleaned_data.csv')
SYNTHETIC_FILE = os.path.join('data', 'synthetic', 'synthetic_data.csv')
PLOT_FOLDER = 'distribution_plots'

if not os.path.exists(PLOT_FOLDER):
    os.makedirs(PLOT_FOLDER)

try:
    df_real = pd.read_csv(REAL_FILE)
    df_synth = pd.read_csv(SYNTHETIC_FILE)
    print("Data Loaded.")
except FileNotFoundError:
    print("Error: 'cleaned_data.csv' or 'synthetic_data.csv' not found.")
    exit()

# --- 2. Prepare Data for Plotting ---
# Dono dataframes ko 'Source' column lagakar jod denge
df_real['Source'] = 'Real Data (Blue)'
df_synth['Source'] = 'Synthetic Data (Orange)'

# Plotting ke liye combine karein
df_combined = pd.concat([df_real, df_synth], axis=0).reset_index(drop=True)

# --- 3. Set Plot Style ---
sns.set_theme(style="whitegrid")
colors = ['blue', 'orange'] # Real = Blue, Synthetic = Orange

# --- 4. Define Key Columns to Plot ---
# (Aap chahein to aur columns add kar sakte hain)
numerical_cols = [
    'Loan Amount', 
    'Annual Income', 
    'Interest Rate', 
    'Total Accounts',
    'Total Collection Amount',
    'Total Current Balance'
]

categorical_cols = [
    'Grade', 
    'Sub Grade', 
    'Home ownership',  # Make sure spelling matches your data
    'Verification Status', 
    'Loan Status'
]

# --- 5. Generate Numerical Plots (KDE - Overlapping Mountains) ---
print("\nGenerating Numerical Distribution Plots (KDE)...")

for col in numerical_cols:
    if col in df_combined.columns:
        plt.figure(figsize=(10, 6))
        
        # KDE plot banayenge jo overlap karega
        sns.kdeplot(
            data=df_combined, 
            x=col, 
            hue='Source', 
            fill=True, 
            common_norm=False, # Zaroori hai taaki dono ki height compare ho sake
            palette=colors,
            alpha=0.4,         # Transparency taaki overlap dikhe
            linewidth=2
        )
        
        plt.title(f'Distribution Comparison: {col}')
        plt.xlabel(col)
        plt.ylabel('Density')
        
        # Save plot
        filename = os.path.join(PLOT_FOLDER, f'numeric_dist_{col.replace(" ", "_")}.png')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")
    else:
        print(f"Warning: Column '{col}' not found in data.")

# --- 6. Generate Categorical Plots (Bar Charts - Side by Side) ---
print("\nGenerating Categorical Count Plots (Bar Charts)...")

for col in categorical_cols:
    if col in df_combined.columns:
        plt.figure(figsize=(12, 6))
        
        # Agar category zyada hain (jaise Sub Grade) to graph ghumayenge
        if df_combined[col].nunique() > 10:
            xticks_rotation = 45
        else:
            xticks_rotation = 0
            
        # Countplot (Side-by-side bars)
        sns.countplot(
            data=df_combined, 
            x=col, 
            hue='Source', 
            palette=colors,
            alpha=0.8,
            order=df_combined[col].value_counts().index # Order by frequency
        )
        
        plt.title(f'Category Count Comparison: {col}')
        plt.xlabel(col)
        plt.ylabel('Count (Number of Rows)')
        plt.xticks(rotation=xticks_rotation)
        plt.legend(title='Source')
        
        # Save plot
        filename = os.path.join(PLOT_FOLDER, f'cat_count_{col.replace(" ", "_")}.png')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")
    else:
        print(f"Warning: Column '{col}' not found in data.")

print(f"\n✅ All distribution plots saved in the '{PLOT_FOLDER}' folder.")
