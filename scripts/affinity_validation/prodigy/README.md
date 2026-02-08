# PRODIGY Affinity Validation

This folder contains scripts to predict antibody structures and validate binding affinities using the program [PRODIGY](https://github.com/haddocking/prodigy).

## Files

- `run_prodigy_for_all_pdb_extracted_hla.py`: Iterates through standardized PDB structures to calculate binding affinity ($\Delta G$) and outputs results to `./result_prodigy/`.

## Notes:
- `./extracted_hla_imgt_structures/` is processed from the script `scripts\data_process\extract_imgt_struct.py`