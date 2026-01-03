from sdv.single_table import CTGANSynthesizer
import pandas as pd
import os

PROCESSED_FOLDER = 'data/processed' 
synthetic_folder = 'data/synthetic'

MODEL_FILE = os.path.join(PROCESSED_FOLDER, 'ctgan_model.pkl') 

CLEANED_FILE = os.path.join(PROCESSED_FOLDER, 'cleaned_data.csv')

SYNTHETIC_FILE = os.path.join(synthetic_folder, 'synthetic_data.csv') 
syn = os.path.join(synthetic_folder, 'syn1.csv')

try:
    real_data = pd.read_csv(CLEANED_FILE)
    num_rows_to_generate = len(real_data)
    print(f"Loaded {CLEANED_FILE} to get the size. Will generate {num_rows_to_generate} new rows.")
except Exception as e:
    print(f"Error loading {CLEANED_FILE}: {e}")
    exit()


try:
    synthesizer = CTGANSynthesizer.load(filepath=MODEL_FILE)
    print(f"Successfully loaded trained model from {MODEL_FILE}")
except FileNotFoundError:
    print(f"ERROR: Model file not found at {MODEL_FILE}")
    print("Please make sure you have run Step 2 successfully.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    exit()


print("Generating new synthetic data... This should be fast.")
synthetic_data = synthesizer.sample(5000)
print("New data generated successfully.")


synthetic_data.to_csv(syn, index=False)

print(f"\nSUCCESS! Synthetic data has been saved to: {SYNTHETIC_FILE}")
print("\n--- First 5 rows of your new synthetic data ---")
print(synthetic_data.head(10))
print("\n--- Step 3: Data Generation Finished ---")