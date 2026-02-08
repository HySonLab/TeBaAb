#!/bin/bash
# File: PrepareFilesForHaddock/preprocess_for_haddock.sh
# Description: Preprocess antibody–antigen PDB files for HADDOCK docking.

set -e  # Exit immediately if any command fails
# set -x

# --- Check input arguments ---
if [ $# -lt 5 ]; then
  echo "Usage: $0 <identifier> <antibody_dir> <antigen_dir> <output_base_dir> <dock_project_path>"
  echo "Example: $0 1a2y ./antibody_1a2y ./antigen_1a2y ./preprocessed ./dock_project_without_des"
  exit 1
fi

# --- Configuration ---
identifier="$1"
antibody_dir="$2"
antigen_dir="$3"
output_base_dir="$4"
dock_project_path="$5"

antibody_pdb="${antibody_dir}/antibody_${identifier}_1.pdb"
antigen_pdb="${antigen_dir}/${identifier}_1.pdb"
output_folder="${output_base_dir}/ab_ag_preprocessed_${identifier}"

# --- Create output directory ---
mkdir -p "${output_folder}"

echo "Processing identifier: ${identifier}"
echo "Antibody PDB: ${antibody_pdb}"
echo "Antigen PDB: ${antigen_pdb}"
echo "Output folder: ${output_folder}"
echo

# --- Step 1: Renumber antibody and identify CDR regions ---
python renumber_pdb_and_identify_cdr.py "${antibody_pdb}" "${output_folder}/ab_reformat.pdb" A > "${output_folder}/antibody_cdr_num.list"
python renumber_pdb.py "${antigen_pdb}" "${output_folder}/ag_reformat.pdb" B


# --- Step 2: Add END/TER statements to PDB files ---
pdb_tidy "${output_folder}/ab_reformat.pdb" > oo && mv oo "${output_folder}/ab_reformat_tidy.pdb"
pdb_tidy "${output_folder}/ag_reformat.pdb" > oo && mv oo "${output_folder}/ag_reformat_tidy.pdb"

# --- Step 3: Generate unambiguous restraints using haddock-tools ---
# cd ../haddock-tools
# python restrain_bodies.py "./../PrepareFilesForHaddock/${output_folder}/ab_reformat_tidy.pdb" > "./../PrepareFilesForHaddock/${output_folder}/antibody-unambig.tbl"
haddock-restraints restraint "${output_folder}/ab_reformat_tidy.pdb" > "${output_folder}/antibody-unambig.tbl"

# --- Step 4: Calculate SASA (FreeSASA) for antibody and antigen ---
# cd ../PrepareFilesForHaddock
python calc_freesasa.py -i "${output_folder}/ab_reformat_tidy.pdb" > "${output_folder}/sasa_ab_init.txt"
python calc_freesasa.py -i "${output_folder}/ag_reformat_tidy.pdb" > "${output_folder}/sasa_ag.list"

# --- Step 5: Identify antibody active sites based on CDR and SASA ---
python get_antibody_active_sites.py "${output_folder}/antibody_cdr_num.list" "${output_folder}/sasa_ab_init.txt" > "${output_folder}/sasa_ab.list"

# --- Step 6: Create ambiguous restraint configuration ---
python create_ambig_config.py "${output_folder}/sasa_ab.list" "${output_folder}/sasa_ag.list" > "${output_folder}/ambig_config.json"


# --- Step 7: Generate ambiguous restraints ---
haddock-restraints tbl "${output_folder}/ambig_config.json" | sed 's/segid/name CA and segid/g' | sed 's/2.0/3.0/g' > "${output_folder}/antibody-antigen-ambig.tbl"


# --- Step 9: Generate HADDOCK run.param file ---
# Get absolute path to output folder
output_folder_abs="$(realpath "${output_folder}")"

# Ensure the directory exists
mkdir -p "${output_folder_abs}"

# Define absolute paths for all relevant files
AMBIG_TBL="${output_folder_abs}/antibody-antigen-ambig.tbl"
UNAMBIG_TBL="${output_folder_abs}/antibody-unambig.tbl"
PDB_FILE1="${output_folder_abs}/ab_reformat_tidy.pdb"
PDB_FILE2="${output_folder_abs}/ag_reformat_tidy.pdb"
PROJECT_DIR="${dock_project_path}/${identifier}"
HADDOCK_DIR="./OtherTools/haddock2.5-2024-12"
N_COMP=2
PROT_SEGID_1=A
PROT_SEGID_2=B
RUN_NUMBER=1

RUN_PARAM_FILE="${output_folder_abs}/run.param"

# Write HADDOCK run.param file
cat > "${RUN_PARAM_FILE}" <<EOF
AMBIG_TBL=${AMBIG_TBL}
HADDOCK_DIR=${HADDOCK_DIR}
N_COMP=${N_COMP}
PDB_FILE1=${PDB_FILE1}
PDB_FILE2=${PDB_FILE2}
PROJECT_DIR=${PROJECT_DIR}
PROT_SEGID_1=${PROT_SEGID_1}
PROT_SEGID_2=${PROT_SEGID_2}
RUN_NUMBER=${RUN_NUMBER}
UNAMBIG_TBL=${UNAMBIG_TBL}
EOF

echo
echo "✅ HADDOCK run.param file successfully created at:"
echo "   ${RUN_PARAM_FILE}"
echo
cat "${RUN_PARAM_FILE}"

cd "${output_folder}"
echo
echo "Processing completed. Output files in:"
pwd
ls
