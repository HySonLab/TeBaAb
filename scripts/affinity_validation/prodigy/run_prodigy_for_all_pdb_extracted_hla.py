import pandas as pd
import subprocess
import os
import glob

# Paths
csv_path = "sabdab_info.csv"
pdb_base_path = "extracted_hla_imgt_structures"
result_folder = "./result_prodigy/"

os.makedirs(result_folder, exist_ok=True)

df = pd.read_csv(csv_path)

H_CHAIN = "H"
L_CHAIN = "L"
AG_CHAIN = "A"

for i, row in df.iterrows():
    if not row.get("pdb_exists", False):
        continue 

    pdb_code = str(row["pdb"]).lower()
    
    # 1. Find ALL matching files for this PDB code
    search_pattern = os.path.join(pdb_base_path, f"{pdb_code}*.pdb")
    matching_files = glob.glob(search_pattern)
    
    if not matching_files:
        print(f"[-] Missing file for: {pdb_code}")
        continue
    
    # 2. Iterate through every file found in the glob
    for pdb_file in matching_files:
        filename_base = os.path.basename(pdb_file).replace(".pdb", "")
        
        # Temperature handling
        temp = row.get("temperature", 25)
        if pd.isna(temp):
            temp = 25

        # Unique output name for each specific file
        output_file = os.path.join(result_folder, f"{filename_base}_result.txt")

        cmd = [
            "prodigy",
            pdb_file,
            "--selection", AG_CHAIN, f"{H_CHAIN},{L_CHAIN}",
            "--temperature", str(temp),
            "-q"
        ]

        try:
            with open(output_file, "w") as out:
                subprocess.run(
                    cmd,
                    stdout=out,
                    stderr=subprocess.STDOUT,
                    check=True
                )
            print(f"[+] Processed: {os.path.basename(pdb_file)}")
        except Exception as e:
            print(f"[!] Error on {os.path.basename(pdb_file)}: {e}")

print("\nProcessing complete. All matched PDB files have been analyzed.")