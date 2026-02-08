import pandas as pd
import os
import math
import warnings
from Bio.PDB import PDBParser, PDBIO, Select, Model, Structure, Chain
from Bio import BiopythonWarning

# Suppress PDB construction warnings
warnings.simplefilter('ignore', BiopythonWarning)

# =====================================================
# Paths
# =====================================================
csv_path = "./sabdab_info.csv"
pdb_base_path = "./sabdab_dataset/all_structures/imgt/"
output_folder = "./extracted_hla_imgt_structures_fixed/"  # New folder for valid PDBs

os.makedirs(output_folder, exist_ok=True)

# =====================================================
# Load CSV
# =====================================================
df = pd.read_csv(csv_path)

# =====================================================
# Helper functions
# =====================================================
def normalize_chain_list(chain_str):
    """Convert 'A|B, C' → ['A','B','C']"""
    return [
        c.strip()
        for c in str(chain_str).replace("|", ",").split(",")
        if c.strip()
    ]

def has_duplicate_chains(h, l, ag_list):
    """Return True if any chain ID is duplicated across H, L, or Ag."""
    all_chains = [h, l] + ag_list
    return len(all_chains) != len(set(all_chains))

from Bio.PDB.Polypeptide import is_aa

def get_clean_chain(structure, source_chain_id, new_chain_id):
    """
    Extracts only amino acid residues from a chain, gives it a new ID, 
    and preserves original residue numbering.
    """
    if source_chain_id not in structure[0]:
        return None
    
    source_chain = structure[0][source_chain_id]
    new_chain = Chain.Chain(new_chain_id)
    
    found_residues = False
    for residue in source_chain:
        # is_aa(residue) checks if it's one of the 20 standard amino acids
        # residue.id[0] == ' ' ensures it is not a HETATM (which starts with 'H_')
        if is_aa(residue) and residue.id[0] == ' ':
            new_chain.add(residue.copy())
            found_residues = True
            
    return new_chain if found_residues else None

def get_merged_antigen_chain(structure, ag_chain_ids, new_chain_id="A"):
    """
    Merges multiple antigen chains into one single chain.
    Filters out non-protein HETATMs (NAG, HOH, etc.) and renumbers.
    """
    new_chain = Chain.Chain(new_chain_id)
    residue_counter = 1
    found_any = False
    
    for ag_id in ag_chain_ids:
        if ag_id not in structure[0]:
            continue
            
        source_chain = structure[0][ag_id]
        for residue in source_chain:
            # Filter for protein residues only
            if is_aa(residue) and residue.id[0] == ' ':
                found_any = True
                res_copy = residue.copy()
                # Reset the ID to be a standard ATOM record (space, index, space)
                res_copy.id = (' ', residue_counter, ' ')
                new_chain.add(res_copy)
                residue_counter += 1
                
    return new_chain if found_any else None

def build_output_filename(row, h, l, ag_list):
    """Unique filename: {PDB}_{Hchain}_{Lchain}_{Antigen}.pdb"""
    pdb = str(row["pdb"])
    ag_safe = "_".join(ag_list)
    # Filename format: 1abc_H_H_L_L_Ag_C_D.pdb (Clean and descriptive)
    return f"{pdb}_H_{h}_L_{l}_Ag_{ag_safe}.pdb"

# =====================================================
# Main loop
# =====================================================
parser = PDBParser(QUIET=True)
io = PDBIO()
valid_count = 0

print(f"Processing {len(df)} rows...")

for idx, row in df.iterrows():
    
    # 1. Validation Checks
    if not row.get("pdb_exists", False):
        continue

    h_chain = str(row["Hchain"]).strip()
    l_chain = str(row["Lchain"]).strip()
    ag_list = normalize_chain_list(row["antigen_chain"])

    if has_duplicate_chains(h_chain, l_chain, ag_list):
        print(f"⏭ Skipped (duplicate chains): {row['pdb']}")
        continue

    pdb_file = os.path.join(pdb_base_path, f"{row['pdb']}.pdb")
    if not os.path.exists(pdb_file):
        print(f"⚠ PDB file missing locally: {pdb_file}")
        continue

    # 2. Define Output Path
    out_filename = build_output_filename(row, h_chain, l_chain, ag_list)
    out_path = os.path.join(output_folder, out_filename)
    
    # Skip if already exists (optional, helpful for restarting scripts)
    if os.path.exists(out_path):
        continue

    # 3. Load Structure
    try:
        structure = parser.get_structure(row['pdb'], pdb_file)
    except Exception as e:
        print(f"⚠ Error parsing {row['pdb']}: {e}")
        continue

    # 4. Construct New Model with H, L, A
    new_model = Model.Model(0) # Create a new PDB model
    
    # -- Process Heavy Chain --
    chain_h = get_clean_chain(structure, h_chain, "H")
    if chain_h:
        new_model.add(chain_h)
    else:
        print(f"⚠ Missing H chain {h_chain} in {row['pdb']}")
        continue # Strictly require H

    # -- Process Light Chain --
    chain_l = get_clean_chain(structure, l_chain, "L")
    if chain_l:
        new_model.add(chain_l)
    else:
        print(f"⚠ Missing L chain {l_chain} in {row['pdb']}")
        continue # Strictly require L

    # -- Process Antigen Chain(s) --
    # Merges multiple chains into one 'A' chain with renumbered residues
    chain_a = get_merged_antigen_chain(structure, ag_list, "A")
    if chain_a:
        new_model.add(chain_a)
    else:
        print(f"⚠ Missing Antigen chain(s) {ag_list} in {row['pdb']}")
        continue # Strictly require Ag

    # 5. Save New Structure
    new_structure = Structure.Structure(row['pdb'])
    new_structure.add(new_model)
    
    try:
        io.set_structure(new_structure)
        io.save(out_path)
        valid_count += 1
        if valid_count % 50 == 0:
            print(f"✔ Processed {valid_count} structures...")
    except Exception as e:
        print(f"❌ Error saving {out_path}: {e}")

print("=====================================================")
print("✅ Finished extraction.")
print(f"✅ Total files created: {valid_count}")
print(f"📂 Output folder: {output_folder}")