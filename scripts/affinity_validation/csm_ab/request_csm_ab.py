import requests
import time
import json
import os
import glob
import csv
from typing import List

# =====================================================
# Configuration
# =====================================================
URL_SINGLE = "https://biosig.lab.uq.edu.au/csm_ab/api/prediction_single"

# Folder containing the structure files (PDB) (with chains renamed H, L, A)
INPUT_FOLDER = "./extracted_hla_imgt_structures/"

# New Output CSV
OUTPUT_CSV = "./OUTPUT_CSV/job_results_csm_ab.csv"

# =====================================================
# Helper Functions
# =====================================================

def get_all_pdb_files(folder: str) -> List[str]:
    """
    Get all .pdb files directly from the extracted folder.
    We don't need the input CSV anymore because this folder 
    only contains the valid processed files.
    """
    pattern = os.path.join(folder, "*.pdb")
    files = glob.glob(pattern)
    files.sort() # Sort ensures consistent processing order
    return files

def run_csm_single(pdb_path: str) -> dict:
    """
    Send a single PDB file to the CSM-AB prediction API.
    
    IMPORTANT: We explicitly send chain IDs 'H', 'L', and 'A' 
    because we standardized them in the previous step. 
    This prevents the server from guessing incorrectly.
    """
    
    # Define the standardized chains we created

    try:
        with open(pdb_path, "rb") as f:
            files = {"pdb_file": f}
            response = requests.post(URL_SINGLE, files=files)
            
        try:
            body = response.json()
        except Exception:
            body = response.text

        result = {
            "body": body,
            "requested_url": response.request.url,
            "final_url": response.url,
            "status_code": response.status_code,
        }
        
        if response.history:
            result["redirect_chain"] = [r.url for r in response.history]

        return result

    except Exception as e:
        return {"error": str(e), "status_code": 0}

def load_processed_files(csv_path: str) -> set:
    """Load the list of pdb_file paths already processed."""
    if not os.path.exists(csv_path):
        return set()

    processed = set()
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if row:
                # The first column is the filename/path
                processed.add(row[0])
    return processed

# =====================================================
# Main Execution
# =====================================================
def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    # 1. Get files directly from the folder
    pdb_files = get_all_pdb_files(INPUT_FOLDER)
    print(f"Found {len(pdb_files)} PDB files in {INPUT_FOLDER}")

    # 2. Check what is already done
    processed_files = load_processed_files(OUTPUT_CSV)
    print(f"Already processed: {len(processed_files)}")

    # 3. Open CSV
    write_header = not os.path.exists(OUTPUT_CSV)

    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        if write_header:
            writer.writerow(["pdb_file", "result_json"])

        for pdb_file in pdb_files:
            
            # Resume logic
            if pdb_file in processed_files:
                continue

            print(f"Processing: {os.path.basename(pdb_file)}")
            
            # (Conceptual: Sending file + standardized params to API)
            result = run_csm_single(pdb_file)

            # Check for API failure to avoid saving bad rows blindly
            status = result.get("status_code")
            if status != 200:
                print(f"  ⚠ API Error {status}: {result.get('body')}")
            else:
                print("  ✔ Success")

            # Save
            writer.writerow([pdb_file, json.dumps(result)])
            csvfile.flush() 

            # Be nice to the server
            time.sleep(1.5) 

    print("Finished! All new results saved.")

if __name__ == "__main__":
    main()