import requests
import re
import time
import json
import os
import glob
import csv
from typing import List

# --- Configuration ---
BASE_URL = "https://saves.mbi.ucla.edu/"
INPUT_PARENT = "./docked_structures"
OUTPUT_PARENT = "./job_results_csv"
PROGRAMS = ['errat', 'verify', 'procheck', 'whatcheck']
REQUEST_TIMEOUT = 60  # Timeout set to 60 seconds
MAX_RETRIES = 3       # Maximum number of retries for each program initiation
RETRY_DELAY = 5       # Delay in seconds between retries

# --- File Reading Function ---
def read_pdb_file(filename: str) -> str:
    """Reads the content of the PDB file."""
    print(f"Reading content from {filename}...")
    try:
        with open(filename, 'r') as f:
            content = f.read()
        return content
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The file '{filename}' was not found.")
    except Exception as e:
        raise Exception(f"An error occurred while reading the file: {e}")

# --- Upload Function ---
def upload_pdb(pdb_filename: str, pdb_content: str) -> str:
    """Uploads the PDB file and returns the Job ID."""
    print(f"1. Uploading PDB file: {pdb_filename}...")
    files = {
        'pdbfile': (pdb_filename, pdb_content, 'application/octet-stream'),
    }
    data = {
        'fname': pdb_filename,
        'startjob': 'Run programs',
    }

    response = requests.post(BASE_URL, files=files, data=data, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    match = re.search(r"job #(\d+):", response.text)
    if match:
        job_id = match.group(1)
        print(f"-> Job successfully created. Job ID: {job_id}")
        return job_id
    else:
        if "No structure file was uploaded" in response.text:
            raise Exception("Upload failed: No structure file uploaded.")
        raise Exception("Could not find Job ID in server response.")

# --- Start Programs Function ---
def start_programs(job_id: str, programs: List[str]):
    """Starts the validation programs for a given Job ID."""
    print(f"2. Starting programs for Job {job_id}...")
    for program in programs:
        ajax_url = f"{BASE_URL}ajax.php"
        params = {'job': job_id, 'lbl': program}
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = requests.get(ajax_url, params=params, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()
                try:
                    status_data = response.json()
                    if status_data.get('msg') is None or "running" not in status_data.get('msg', '').lower():
                        print(f"   [WARNING] '{program}' may not have confirmed running, but request succeeded.")
                except json.JSONDecodeError:
                    pass
                print(f"-> Started {program} successfully on attempt {attempt}.")
                break
            except requests.exceptions.RequestException as e:
                if attempt < MAX_RETRIES:
                    print(f"   [RETRY] Failed to start {program} (Attempt {attempt}/{MAX_RETRIES}). Retrying in {RETRY_DELAY}s...")
                    time.sleep(RETRY_DELAY)
                else:
                    raise Exception(f"Failed to start {program} after {MAX_RETRIES} attempts. Last Error: {e}")
    print("3. All programs initiated.")

def main():
    # Get all subdirectories
    subfolders = [f.path for f in os.scandir(INPUT_PARENT) if f.is_dir()]
    if not subfolders:
        print(f"No subfolders found in '{INPUT_PARENT}'.")
        return

    for folder in subfolders:
        folder_name = os.path.basename(folder)
        print(f"\n=== Processing folder: {folder_name} ===")

        pdb_files = glob.glob(os.path.join(folder, "*restored.pdb"))
        if not pdb_files:
            print(f"No PDB files found in subfolder '{folder}'.")
            continue

        # Output CSV path for this folder
        output_csv = os.path.join(OUTPUT_PARENT, f"{folder_name}.csv")
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        
        # ---- SKIP if CSV already exists ----
        if os.path.exists(output_csv):
            print(f"[SKIP] Output for folder '{folder_name}' already exists: {output_csv}")
            continue

        results = []

        for pdb_path in pdb_files:
            pdb_filename = os.path.basename(pdb_path)
            print(f"\nProcessing file: {pdb_filename}")

            try:
                pdb_content = read_pdb_file(pdb_path)
                if not pdb_content.strip():
                    print(f"Skipping empty file '{pdb_filename}'.")
                    continue

                job_id = upload_pdb(pdb_filename, pdb_content)
                start_programs(job_id, PROGRAMS)

                url = f"{BASE_URL}?job={job_id}"
                results.append((pdb_filename, url))
                print(f"-> URL for '{pdb_filename}': {url}\n")

            except Exception as e:
                print(f"[ERROR] Could not process '{pdb_filename}': {e}")
                results.append((pdb_filename, "ERROR"))

        # Save results for this subfolder
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["PDB Filename", "URL"])
            writer.writerows(results)

        print(f"Saved results to: {output_csv}")

    print("\nAll folders done!")


if __name__ == "__main__":
    main()
