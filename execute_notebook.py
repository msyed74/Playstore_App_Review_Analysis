import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os
import sys

nb_path = r"c:\Users\Microline\Downloads\Playstore App Review\Sample\Sample_EDA_Submission_Template.ipynb"
master_path = r"c:\Users\Microline\Downloads\Playstore App Review\Sample\Playstore_App_Review_Analysis.ipynb"

print(f"Reading notebook from: {nb_path}")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

print("Configuring ExecutePreprocessor (kernel: python3, timeout: 600s)...")
ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

working_dir = r"c:\Users\Microline\Downloads\Playstore App Review\Sample"
print(f"Executing notebook in working directory: {working_dir} ...")

try:
    ep.preprocess(nb, {'metadata': {'path': working_dir}})
    print("Notebook executed successfully without errors!")
except Exception as e:
    print(f"Error during execution: {e}")
    sys.exit(1)

# Save executed notebook to both files
print("Saving executed notebook with rendered outputs and charts...")
with open(nb_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

with open(master_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("Both notebooks successfully updated with full outputs and visualizations!")
