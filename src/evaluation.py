"""import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdmetrics.reports.single_table import QualityReport
from sdmetrics.visualization import get_column_plot, get_column_pair_plot
import os

print("--- Step 4: Evaluation Script Started ---")

# --- 1. Define Paths ---
PROCESSED_FOLDER = 'data/processed'
synthetic_folder = 'data/synthetic'
CLEANED_FILE = os.path.join(PROCESSED_FOLDER, 'cleaned_data.csv')
SYNTHETIC_FILE = os.path.join(synthetic_folder, 'synthetic_data.csv')

# Create a folder for the graphs
VISUAL_FOLDER = 'data/evaluation_graphs'
if not os.path.exists(VISUAL_FOLDER):
    os.makedirs(VISUAL_FOLDER)
    print(f"Created folder: {VISUAL_FOLDER}")

# --- 2. Load Data and Metadata ---
try:
    real_data = pd.read_csv(CLEANED_FILE)
    print(f"Successfully loaded REAL data from {CLEANED_FILE}")
    
    synthetic_data = pd.read_csv(SYNTHETIC_FILE)
    print(f"Successfully loaded SYNTHETIC data from {SYNTHETIC_FILE}")
    
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=real_data)
    
    metadata.update_column(
        column_name='Loan Status',
        sdtype='categorical'
    )
    
except FileNotFoundError as e:
    print(f"ERROR: Could not find file. {e}")
    print("Please make sure 'cleaned_data.csv' and 'synthetic_data.csv' are in the 'processed' folder.")
    exit()

print("\n--- Generating Report 1: Statistical Quality Report ---")

# --- 3. Report 1: Statistical Quality Report ---
# Yeh report statistics aur ML Utility, dono ko check karega.
report = QualityReport()
report.generate(real_data, synthetic_data, metadata.to_dict())

# Get the final score (0% to 100%)
quality_score = report.get_score()
print(f"\n✅ Overall Data Quality Score: {quality_score*100:.2f}%")

# Get details about which columns are good or bad
print("\n--- Column Shape Similarity (Statistics) ---")
print(report.get_details(property_name='Column Shapes'))

# Save the full report as an HTML file
report.save(filepath=os.path.join(VISUAL_FOLDER, '1_full_quality_report.html'))
print(f"Saved full statistical report to '{VISUAL_FOLDER}/1_full_quality_report.html'")


print("\n--- Generating Report 2: ML Utility (TSTR) Report ---")

# --- 4. Report 2: ML Utility (TSTR) ---
# YEH HAI CORRECTED PART
# Hum 'QualityReport' (jiska naam 'report' hai) se 'ML Efficacy' maang rahe hain.
try:
    ml_results = report.get_details(property_name='ML Efficacy')
    print(ml_results)
except Exception as e:
    print(f"Could not generate ML Efficacy report. Error: {e}")
    print("This can happen if the target column 'Loan Status' is difficult to predict.")


print("\n--- Generating Report 3: Visual Evaluation (Graphs) ---")

# --- 5. Report 3: Visual Evaluation (Graphs) ---
# Yeh code waisa hi hai, yeh graphs banayega.

# Example 1: A numerical column ('Loan Amount')
fig_loan = get_column_plot(real_data, synthetic_data, column_name='Loan Amount')
fig_loan.write_image(os.path.join(VISUAL_FOLDER, '2_graph_Loan_Amount.png'))
print("Saved graph for 'Loan Amount'")

# Example 2: A categorical column ('Grade')
fig_grade = get_column_plot(real_data, synthetic_data, column_name='Grade')
fig_grade.write_image(os.path.join(VISUAL_FOLDER, '2_graph_Grade.png'))
print("Saved graph for 'Grade'")

# Example 3: A relationship (to show the model learned patterns)
fig_pair = get_column_pair_plot(
    real_data, 
    synthetic_data, 
    column_names=['Annual Income', 'Loan Amount']
)
fig_pair.write_image(os.path.join(VISUAL_FOLDER, '2_graph_Pair_Income_vs_Loan.png'))
print("Saved graph for 'Annual Income' vs 'Loan Amount' relationship")

print(f"\n✅ All graphs saved to the '{VISUAL_FOLDER}' folder.")
print("\n--- Step 4: Evaluation Script Finished ---")"""





# ye alag hai




"""import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdmetrics.reports.single_table import QualityReport
from sdmetrics.visualization import get_column_plot, get_column_pair_plot
import os

print("--- Step 4: Evaluation Script Started (Final Fix) ---")

PROCESSED_FOLDER = 'data/processed'
synthetic_folder = 'data/synthetic'
CLEANED_FILE = os.path.join(PROCESSED_FOLDER, 'cleaned_data.csv')
SYNTHETIC_FILE = os.path.join(synthetic_folder, 'synthetic_data.csv')

# Create a folder for the graphs
VISUAL_FOLDER = 'data/evaluation_graphs'
if not os.path.exists(VISUAL_FOLDER):
    os.makedirs(VISUAL_FOLDER)
    print(f"Created folder: {VISUAL_FOLDER}")

# --- 2. Load Data ---
try:
    real_data = pd.read_csv(CLEANED_FILE)
    print(f"Successfully loaded REAL data from {CLEANED_FILE}")
    
    synthetic_data = pd.read_csv(SYNTHETIC_FILE)
    print(f"Successfully loaded SYNTHETIC data from {SYNTHETIC_FILE}")
    
except FileNotFoundError as e:
    print(f"ERROR: Could not find file. {e}")
    exit()

# --- 3. THE FIX: Force 'Loan Status' to category type ---
try:
    print("Forcing 'Loan Status' column to 'category' type...")
    real_data['Loan Status'] = real_data['Loan Status'].astype(str)
    synthetic_data['Loan Status'] = synthetic_data['Loan Status'].astype(str)
    print("'Loan Status' type converted successfully.")
except KeyError:
    print("ERROR: 'Loan Status' column not found.")
    exit()

# --- 4. Detect Metadata ---
print("Detecting metadata...")
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=real_data)
print("Metadata detection complete.")

# --- 5. Report 1: Statistical Quality Report ---
print("\n--- Generating Report 1: Statistical Quality Report ---")
report = QualityReport()
report.generate(real_data, synthetic_data, metadata.to_dict())

# Get the final score
quality_score = report.get_score()
print(f"\n✅ Overall Data Quality Score: {quality_score*100:.2f}%")

# Get statistics details
print("\n--- Column Shape Similarity (Statistics) ---")
print(report.get_details(property_name='Column Shapes'))

# Save the full report
report.save(filepath=os.path.join(VISUAL_FOLDER, '1_full_quality_report.html'))
print(f"Saved full statistical report to '{VISUAL_FOLDER}/1_full_quality_report.html'")

# --- 6. Report 2: ML Utility (TSTR) Report ---
print("\n--- Generating Report 2: ML Utility (TSTR) Report ---")
try:
    ml_results = report.get_details(property_name='ML Efficacy')
    print(ml_results)
except Exception as e:
    print(f"Could not generate ML Efficacy report. Error: {e}")

# --- 7. Report 3: Visual Evaluation (Graphs) ---
print("\n--- Generating Report 3: Visual Evaluation (Graphs) ---")

fig_loan = get_column_plot(real_data, synthetic_data, column_name='Loan Amount')
fig_loan.write_image(os.path.join(VISUAL_FOLDER, '2_graph_Loan_Amount.png'))
print("Saved graph for 'Loan Amount'")

fig_grade = get_column_plot(real_data, synthetic_data, column_name='Grade')
fig_grade.write_image(os.path.join(VISUAL_FOLDER, '2_graph_Grade.png'))
print("Saved graph for 'Grade'")

fig_pair = get_column_pair_plot(
    real_data, 
    synthetic_data, 
    column_names=['Annual Income', 'Loan Amount']
)
fig_pair.write_image(os.path.join(VISUAL_FOLDER, '2_graph_Pair_Income_vs_Loan.png'))
print("Saved graph for 'Annual Income' vs 'Loan Amount' relationship")

print(f"\n✅ All graphs saved to the '{VISUAL_FOLDER}' folder.")
print("\n--- Step 4: Evaluation Script Finished ---")"""



import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdmetrics.reports.single_table import QualityReport
from sdmetrics.visualization import get_column_plot, get_column_pair_plot
import os

print("--- Step 4: Evaluation Script Started (Final Manual Fix) ---")

# --- 1. Define Paths (Aapke project ke hisaab se) ---
CLEANED_FILE = os.path.join('data', 'processed', 'cleaned_data.csv')
SYNTHETIC_FILE = os.path.join('data', 'synthetic', 'synthetic_data.csv')
VISUAL_FOLDER = os.path.join('data', 'evaluation_graphs')

if not os.path.exists(VISUAL_FOLDER):
    os.makedirs(VISUAL_FOLDER)
    print(f"Created folder: {VISUAL_FOLDER}")

# --- 2. Load Data ---
try:
    real_data = pd.read_csv(CLEANED_FILE)
    print(f"Successfully loaded REAL data from {CLEANED_FILE}")
    synthetic_data = pd.read_csv(SYNTHETIC_FILE)
    print(f"Successfully loaded SYNTHETIC data from {SYNTHETIC_FILE}")
except FileNotFoundError as e:
    print(f"ERROR: Could not find file. {e}")
    exit()

# --- 3. THE FIX: Force 'Loan Status' to category type ---
try:
    print("Forcing 'Loan Status' column to 'category' type...")
    real_data['Loan Status'] = real_data['Loan Status'].astype(str)
    synthetic_data['Loan Status'] = synthetic_data['Loan Status'].astype(str)
    print("'Loan Status' type converted successfully.")
except KeyError:
    print("ERROR: 'Loan Status' column not found.")
    exit()

# --- 4. Detect Metadata ---
print("Detecting metadata...")
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=real_data)
print("Metadata detection complete.")

# --- 5. !!! --- YEH HAI ASLI FINAL FIX --- !!! ---
# 1. Pehle dictionary banayein
metadata_dict = metadata.to_dict()

# 2. Ab, dictionary mein target ko manually add karein
# sdmetrics is format ko expect karta hai
metadata_dict['target'] = 'Loan Status' 
print("Successfully set 'Loan Status' as the target in the metadata dictionary.")

# --- 6. Report 1: Statistical Quality Report ---
print("\n--- Generating Report 1: Statistical Quality Report (with ML) ---")
report = QualityReport()


report.generate(real_data, synthetic_data, metadata_dict) 

print("Report generation complete.")

# Get the final score
quality_score = report.get_score()
print(f"\nOverall Data Quality Score: {quality_score*100:.2f}%")

# Get statistics details
print("\n--- Column Shape Similarity (Statistics) ---")
print(report.get_details(property_name='Column Shapes'))
report.save(filepath=os.path.join(VISUAL_FOLDER, '1_full_quality_report.html'))
print(f"Saved full statistical report to '{VISUAL_FOLDER}/1_full_quality_report.html'")


print("\n--- Generating Report 2: ML Utility (TSTR) Report ---")
try:
    
    ml_results = report.get_details(property_name='ML Efficacy')
    print(ml_results)
except Exception as e:
    print(f"Could not generate ML Efficacy report. Error: {e}")

# --- 8. Report 3: Visual Evaluation (Graphs) ---
print("\n--- Generating Report 3: Visual Evaluation (Graphs) ---")
# (Graphs wala code change nahi hua hai)
fig_loan = get_column_plot(real_data, synthetic_data, column_name='Loan Amount')
fig_loan.write_image(os.path.join(VISUAL_FOLDER, '2_graph_Loan_Amount.png'))
print("Saved graph for 'Loan Amount'")

fig_grade = get_column_plot(real_data, synthetic_data, column_name='Grade')
fig_grade.write_image(os.path.join(VISUAL_FOLDER, '2_graph_Grade.png'))
print("Saved graph for 'Grade'")

fig_pair = get_column_pair_plot(
    real_data, 
    synthetic_data, 
    column_names=['Annual Income', 'Loan Amount']
)
fig_pair.write_image(os.path.join(VISUAL_FOLDER, '2_graph_Pair_Income_vs_Loan.png'))
print("Saved graph for 'Annual Income' vs 'Loan Amount' relationship")
fig_Funded = get_column_plot(real_data, synthetic_data, column_name='Funded Amount')
fig_Funded.write_image(os.path.join(VISUAL_FOLDER, '2_graph_Funded_Amount.png'))
print("Saved graph for 'Funded Amount'")
fig_FundedByInv = get_column_plot(real_data, synthetic_data, column_name='Funded Amount Investor')
fig_FundedByInv.write_image(os.path.join(VISUAL_FOLDER, '2_graph_Funded_Amount_Investor.png'))
print("Saved graph for 'Funded Amount Investor'")
fig_InterestRate = get_column_plot(real_data, synthetic_data, column_name='Interest Rate')
fig_InterestRate.write_image(os.path.join(VISUAL_FOLDER, '2_graph_Interest_Rate.png'))
print("Saved graph for 'Interest Rate'")

print(f"\n✅ All graphs saved to the '{VISUAL_FOLDER}' folder.")
print("\n--- Step 4: Evaluation Script Finished ---")