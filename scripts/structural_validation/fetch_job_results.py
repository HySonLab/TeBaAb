import os
import csv
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

CSV_FOLDER = "./job_results_csv"
RESULT_BASE = "./job_results/"
BASE_URL = "https://saves.mbi.ucla.edu"

def download_file(url, outpath):
    """Download a file if it does not already exist."""
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    if os.path.exists(outpath):
        return
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        with open(outpath, "wb") as f:
            f.write(r.content)
        print(f"[OK] Downloaded: {outpath}")
    except Exception as e:
        print(f"[ERR] Failed to download {url}: {e}")


def process_job(job_id, prefix=""):
    OUTPUT_DIR = os.path.join(RESULT_BASE, prefix, job_id)
    done_flag = os.path.join(OUTPUT_DIR, "__job_done__")

    # If done file exists → skip entirely
    if os.path.exists(done_flag):
        print(f"[SKIP] Job {job_id} already completed. Skipping...")
        return
    RESULT_URL = f"https://saves.mbi.ucla.edu/results?job={job_id}"
    OUTPUT_DIR = os.path.join(RESULT_BASE, prefix, job_id)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n============================")
    print(f"[INFO] Processing job {job_id}")
    print(f"============================")

    # ------------------------------------------------------
    # 1. Download HTML page
    # ------------------------------------------------------
    try:
        response = requests.get(RESULT_URL)
        response.raise_for_status()
    except Exception as e:
        print(f"[ERR] Failed to fetch HTML for job {job_id}: {e}")
        return

    html = response.text
    with open(f"{OUTPUT_DIR}/raw_page.html", "w", encoding="utf-8") as f:
        f.write(html)

    soup = BeautifulSoup(html, "html.parser")
    print("[OK] HTML fetched.")

    # ------------------------------------------------------
    # 2. Extract information
    # ------------------------------------------------------
    info = {}

    # ERRAT
    block = soup.find(id="kberrat")
    if block:
        qf = block.find("h1")
        info["ERRAT_overall_quality_factor"] = qf.text.strip() if qf else None

    # VERIFY3D
    vblock = soup.find(id="kbverify")
    if vblock:
        percent = vblock.find("div", {"class": "vmsg"})
        status = vblock.find(id="vfpass")
        info["VERIFY3D_percent"] = percent.text.strip().replace("\n", " ") if percent else None
        info["VERIFY3D_status"] = status.text.strip() if status else None

    # WHATCHECK
    wc_items = soup.select("#kbwhatcheck .wcitem")
    info["WHATCHECK_headings"] = [x.get("tmsg", "").strip() for x in wc_items]

    # PROCHECK
    pblock = soup.find(id="kbprocheck")
    if pblock:
        stats_list = pblock.select("ul.mli li span")
        stats = {s['class'][0]: s.text.strip() for s in stats_list}
        info["PROCHECK_summary"] = stats

    # Save extracted info
    with open(f"{OUTPUT_DIR}/summary_extracted.txt", "w") as f:
        for k, v in info.items():
            f.write(f"{k}: {v}\n")

    print("[OK] Summary extracted.")

    # ------------------------------------------------------
    # 3. Find all downloadable links
    # ------------------------------------------------------
    links = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if (
            href.startswith(f"/Jobs/{job_id}/")
            or href.startswith("/class/download")
            or href.endswith(".dssp")
        ):
            links.append(href)

    links = sorted(set(links))
    print(f"[INFO] Found {len(links)} downloadable items.")

    # ------------------------------------------------------
    # 4. Download files
    # ------------------------------------------------------
    for href in links:
        full_url = urljoin(BASE_URL, href)
        save_name = href.lstrip("/").replace("/", "_")
        save_path = os.path.join(OUTPUT_DIR, save_name)
        download_file(full_url, save_path)

    # Mark job as fully completed
    with open(done_flag, "w") as f:
        f.write("1")


    print(f"[DONE] Job {job_id} completed.")


# ----------------------------------------------------------
# MAIN: Scan all CSVs and process all job IDs
# ----------------------------------------------------------

csv_files = [f for f in os.listdir(CSV_FOLDER) if f.endswith(".csv")]
print(f"[INFO] Found {len(csv_files)} CSV files to process.")

for csv_name in csv_files:
    csv_path = os.path.join(CSV_FOLDER, csv_name)
    prefix = os.path.splitext(csv_name)[0]  # group results by CSV filename

    print(f"\n==== Processing CSV: {csv_name} ====")

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = row["URL"].strip()
            
            # --- HANDLE MISSING OR ERROR URL ---
            if not url or url.upper() == "ERROR":
                print(f"[WARN] Skipping due to invalid URL value: '{url}'")
                continue
            
            match = re.search(r"job=(\d+)", url)
            if match:
                job_id = match.group(1)
                process_job(job_id, prefix=prefix)
            else:
                print(f"[WARN] No job ID found in URL: {url}")

print("\nALL DONE.")
