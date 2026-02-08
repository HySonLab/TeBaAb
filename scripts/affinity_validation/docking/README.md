
# HADDOCK docking

This folder contains small helper scripts to run HADDOCK docking.

First, preprocess input PDBs and generate `run.param` with `run_haddock_project_dir.sh` in `scripts/affinity_validation/prepare_docking`:

Use the produced `run.param` as input to the helper in this folder to prepare and run the HADDOCK project. Example:

```bash
# Usage
./run_haddock_project_dir.sh  <identifier> <preprocessed_base_dir> <dock_project_dir>

# Example
./run_haddock_project_dir.sh 1a2y ./preprocessed ./dock_project_without_des
```

`patch_run_cns.py` is used in the script to patch the number of structures that HADDOCK will dock.


