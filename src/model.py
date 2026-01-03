import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
import os

print("--- Step 2: Model Training Script Started ---")

# --- 1. Define Paths ---
PROCESSED_FOLDER = 'data/processed'
CLEANED_FILE = os.path.join(PROCESSED_FOLDER, 'cleaned_data.csv')
MODEL_FILE = os.path.join(PROCESSED_FOLDER, 'ctgan_model.pkl') 
# --- 2. Load the Cleaned Data ---
try:
    data = pd.read_csv(CLEANED_FILE)
    print(f"Successfully loaded {CLEANED_FILE}")
except FileNotFoundError:
    print(f"ERROR: '{CLEANED_FILE}' not found.")
    print("Please make sure you have run Step 1 successfully.")
    exit() # Stop if file not found
except Exception as e:
    print(f"An error occurred while loading the file: {e}")
    exit()

# --- 3. Auto-detect Metadata (The "Automatic Encoding" Step) ---
# Yahi woh step hai jahan model sabhi column types ko pehchaan raha hai
print("Detecting metadata (column types) automatically...")
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=data)

# Yeh line check karti hai ki 'Loan Status' ko ek category (na ki number) maana jaaye
# Taki model uske relationships ko behtar samajh sake
metadata.update_column(
    column_name='Loan Status',
    sdtype='categorical'
)

print("Metadata detection and update complete.")
# Optional: metadata check karne ke liye
# print(metadata.to_dict())

# --- 4. Initialize the CTGAN Model ---
# Hum synthesizer ko metadata pass karte hain
synthesizer = CTGANSynthesizer(
    metadata,
    enforce_rounding=False, # Financial data ke liye better hai
    epochs=300,             # Default 300 hai. Aap ise badha (better quality) ya ghata (faster training) sakte hain.
    verbose=True            # Yeh training ki progress dikhayega
)

# --- 5. Train the Model ---
print("\nStarting model training...")
print("THIS WILL TAKE A LONG TIME (e.g., 15-60 minutes). Please be patient.")
print("You will see output (Epochs/Loss) as it trains...")

synthesizer.fit(data)

print("✅ Model training complete!")

# --- 6. Save the Trained Model ---
# Model ko save karna taaki humein baar-baar train na karna pade
synthesizer.save(filepath=MODEL_FILE)
print(f"\n✅ SUCCESS! Trained model has been saved to: {MODEL_FILE}")
print("\n--- Step 2: Model Training Finished ---")