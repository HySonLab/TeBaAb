#!/bin/bash
# File: run_haddock_pipeline.sh
# Description: Automate HADDOCK setup and execution for a given identifier.

set -e  # Exit immediately if any command fails

pwd

# source ~/.bashrc
source "./haddock2.5-2024-12/haddock_configure.sh"

# --- Input arguments ---
if [ $# -lt 2 ]; then
    echo "Usage: $0 <identifier> <preprocessed_base_dir> <dock_project_dir>"
    echo "Example: $0 1a2y ./preprocessed ./dock_project_without_des"
    exit 1
fi

identifier="$1"
base_dir="$2"
dock_dir="$3"

# --- Paths ---
SOURCE_DIR="${base_dir}/ab_ag_preprocessed_${identifier}"
TARGET_RUN_PARAM="./run.param"
RUN_CNS_PATH="${dock_dir}/${identifier}/run1/run.cns"

# --- Step 1: Copy run.param from the prepared folder ---
if [ ! -f "${SOURCE_DIR}/run.param" ]; then
    echo "Error: run.param not found in ${SOURCE_DIR}"
    exit 1
fi

echo "Copying run.param for ${identifier}..."
cp "${SOURCE_DIR}/run.param" "${TARGET_RUN_PARAM}"

# --- Step 2: Run initial HADDOCK setup ---
echo "Running HADDOCK setup..."
haddock2.5

# --- Step 3: Remove the temporary run.param ---
echo "Cleaning up..."
rm -f "${TARGET_RUN_PARAM}"

# --- Step 4: Patch run.cns file ---
if [ ! -f "${RUN_CNS_PATH}" ]; then
    echo "Error: run.cns not found at ${RUN_CNS_PATH}"
    exit 1
fi

echo "Patching run.cns for ${identifier}..."
python patch_run_cns.py "${RUN_CNS_PATH}"

# --- Step 5: Run HADDOCK docking ---
echo "Running HADDOCK docking for ${identifier}..."
cd "${dock_dir}/${identifier}/run1"
pwd
haddock2.5

echo "HADDOCK pipeline completed for ${identifier}."
