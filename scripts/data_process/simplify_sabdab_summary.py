import pandas as pd
import os

# -----------------------
# Paths
# -----------------------
sabdab_tsv_path = "./sabdab_summary_all.tsv"
pdb_folder = "./sabdab_dataset/all_structures/imgt"
output_csv = "sabdab_info.csv"

# -----------------------
# Columns of interest
# -----------------------
columns_to_keep = [
    "pdb",
    "Hchain",
    "Lchain",
    "antigen_chain",
    "delta_g",
    "temperature"
]

# -----------------------
# Load and filter TSV
# -----------------------
df = pd.read_csv(
    sabdab_tsv_path,
    sep="\t",
    usecols=columns_to_keep
)

# Drop rows where delta_g is missing
df = df.dropna(subset=["delta_g"])

# Drop rows where any required chain is missing
df = df.dropna(subset=["Hchain", "Lchain", "antigen_chain"])

# -----------------------
# Check PDB existence
# -----------------------
def check_pdb_exists(pdb_id: str) -> bool:
    pdb_path = os.path.join(pdb_folder, f"{pdb_id}.pdb")
    return os.path.exists(pdb_path)

df["pdb_exists"] = df["pdb"].apply(check_pdb_exists)

# -----------------------
# Inspect result
# -----------------------
print(df.head())
print(f"Total rows after filtering: {len(df)}")
print(f"PDBs found: {df['pdb_exists'].sum()}")

# -----------------------
# Save final output
# -----------------------
df.to_csv(output_csv, index=False)
print(f"Saved to {output_csv}")
