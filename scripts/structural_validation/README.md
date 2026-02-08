# Structural validation

This folder contains three helper scripts to validate PDB structures using the online SAVES tools (ERRAT, VERIFY3D, WHATCHECK, PROCHECK), download results, and aggregate them into a summary.

Files
- `request_structure_validate.py`: Upload PDB files to the online SAVES validators and start validation programs. Reads PDBs from `docked_structures/` and writes CSVs into `request_results_csv/` (one CSV per input subfolder).
- `fetch_job_results.py`: Reads the CSVs produced above, downloads result pages and files for each job into `job_results/<csv_name>/<job_id>/`.
- `parse_all_results.py`: Parses downloaded result folders under `job_results/` and writes per-PDB summary JSON files to `job_results/_parsed_results_summary/`.

