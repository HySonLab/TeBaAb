import csv
import json
import time
import requests
import os

INPUT_CSV = "./OUTPUT_CSV/job_results_csm_ab.csv"
OUTPUT_CSV = "./OUTPUT_CSV/ag_renumbered/results_csm_ab.csv"

URL_single = "https://biosig.lab.uq.edu.au/csm_ab/api/prediction_single"


def get_result(job_id: str) -> dict:
    params = {"job_id": job_id}
    try:
        r = requests.get(URL_single, params=params, verify=False)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def extract_job_id(result_json_str):
    """Return job_id or None from the result_json column."""
    try:
        data = json.loads(result_json_str)
    except json.JSONDecodeError:
        return None

    # Expected format:
    # { "body": { "job_id": "..." }, ... }
    if isinstance(data, dict):
        body = data.get("body", {})
        if isinstance(body, dict):
            return body.get("job_id")

    return None


def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    with open(INPUT_CSV, "r") as infile, open(OUTPUT_CSV, "w", newline="") as outfile:
        reader = csv.DictReader(infile)
        writer = csv.writer(outfile)
        writer.writerow(["pdb_file", "job_id", "result_json"])

        for row in reader:
            pdb_file = row["pdb_file"]
            job_id = extract_job_id(row["result_json"])

            if not job_id:
                print(f"Skipping {pdb_file}: no job_id found.")
                continue

            print(f"Fetching result for job_id = {job_id}")
            result = get_result(job_id)
            writer.writerow([pdb_file, job_id, json.dumps(result)])
            time.sleep(1)


if __name__ == "__main__":
    main()
