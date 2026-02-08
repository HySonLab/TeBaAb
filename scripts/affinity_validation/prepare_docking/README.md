# HADDOCK Preprocessing Scripts

This folder contains helper scripts to prepare inputs and a `run.param` file for HADDOCK docking. The main entrypoint is the shell script `preprocess_for_haddock.sh`.

## Quick Start

To run the pipeline, use the orchestrator script with the following arguments:

```bash
# Usage
./preprocess_for_haddock.sh <identifier> <antibody_dir> <antigen_dir> <output_base_dir> <dock_project_path>

# Example
./preprocess_for_haddock.sh 1a2y ./antibody_1a2y ./antigen_1a2y ./preprocessed ./dock_project_without_des
```

## What the scripts do
- `preprocess_for_haddock.sh`: orchestrates the preprocessing pipeline and calls the Python helpers.
- `renumber_pdb.py`: renumbers PDB files to a consistent scheme expected by downstream tools.
- `renumber_pdb_and_identify_cdr.py`: renumbers PDBs and identifies antibody CDR regions (complements the renumbering step).
- `get_antibody_active_sites.py`: extracts likely antibody active residues (used to define active/passive residues for HADDOCK).
- `calc_freesasa.py`: computes solvent-accessible surface area (SASA) values for PDBs (used to rank/interface residues).
- `create_ambig_config.py`: generates ambiguous interaction restraint (AIR) configuration files / `run.param` fragments for HADDOCK.

## Notes & prerequisites
- You will need HADDOCK-related tools and dependencies installed:
    - HADDOCK restraints and utilities: https://github.com/haddocking/haddock-restraints
    - Useful PDB tools: https://github.com/haddocking/pdb-tools
