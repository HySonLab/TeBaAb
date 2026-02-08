# CSM-AB Affinity Validation

This folder contains scripts to predict antibody structures and validate binding affinities using the CSM-AB web service.

## Files

- `request_csm_ab.py`: Submits antibody sequences or structures to the CSM-AB service for prediction. It reads PDB files (with chains renamed H, L, A) from `./extracted_hla_imgt_structures/` and logs job details to `./OUTPUT_CSV/job_results_csm_ab.csv`.
- `fetch_result_csm_ab.py`: Retrieves completed prediction data and structures from the CSM-AB servers. It reads the job IDs from the initial CSV and outputs the final data to `./OUTPUT_CSV/ag_renumbered/results_csm_ab.csv`.

## Notes:
- `./extracted_hla_imgt_structures/` is processed from the script `scripts\data_process\extract_imgt_struct.py`