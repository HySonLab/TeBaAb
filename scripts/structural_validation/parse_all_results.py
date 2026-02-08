import os
import ast
import json
import re
from collections import Counter, defaultdict

BASE_ROOT = "./job_results"
OUTPUT_DIR = os.path.join(BASE_ROOT, "_parsed_results_summary")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_number_from_percent(s):
    return float(s.replace("%", "").strip())

def process_pdb_folder(pdb_path):
    errat_values = []
    verify3d_values = []
    verify3d_status_counter = Counter()
    whatcheck_counter = Counter()
    procheck_totals = Counter()

    # New: Use a defaultdict to collect lists of values for all numeric fields
    # This avoids creating 20+ separate list variables.
    # Key examples: 'rama_core', 'g_factor_overall', 'bad_contacts', etc.

    stats_collector = defaultdict(list)
    structures_count = 0

    for folder in os.listdir(pdb_path):
        folder_path = os.path.join(pdb_path, folder)
        summary_file = os.path.join(folder_path, "summary_extracted.txt")
        
        # Construct path for the PROCHECK sum file using the folder name (job_id)
        # Example: Jobs_389508_pc_saves.sum
        procheck_sum_file = os.path.join(folder_path, f"Jobs_{folder}_pc_saves.sum")

        if not os.path.isdir(folder_path):
            continue
        
        # track if we found valid data for this structure
        found_data = False 
        entry = {}

        # ---------------------------------------------------------
        # 1. Process summary_extracted.txt (Existing Logic)
        # ---------------------------------------------------------
        if os.path.isfile(summary_file):
            found_data = True
            with open(summary_file, "r") as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                
                if line.startswith("ERRAT_overall_quality_factor:"):
                    value = line.split(":", 1)[1].strip()
                    if value:
                        try:
                            entry["errat"] = float(value)
                        except ValueError:
                            print(f"[WARN] Could not parse ERRAT value in {summary_file}: {value}")
                
                elif line.startswith("VERIFY3D_percent:"):
                    value = line.split(":", 1)[1]
                    value = value.split("%")[0].strip()
                    if value:
                        try:
                            entry["verify3d"] = float(value)
                        except ValueError:
                            print(f"[WARN] Could not parse VERIFY3D_percent in {summary_file}: {value}")
                
                elif line.startswith("VERIFY3D_status:"):
                    status = line.split(":", 1)[1].strip()
                    if status:
                        entry["verify3d_status"] = status
                
                elif line.startswith("WHATCHECK_headings:"):
                    data = line.split(":", 1)[1].strip()
                    try:
                        headings_list = ast.literal_eval(data)
                        entry["whatcheck"] = headings_list
                    except Exception:
                        print(f"[WARN] Could not parse WHATCHECK_headings in {summary_file}")
                
                elif line.startswith("PROCHECK_summary:"):
                    data = line.split(":", 1)[1].strip()
                    try:
                        summary_dict = ast.literal_eval(data)
                        entry["procheck"] = summary_dict
                    except Exception:
                        print(f"[WARN] Could not parse PROCHECK_summary in {summary_file}")

        # ---------------------------------------------------------
        # 2. Process Jobs_{id}_pc_saves.sum (Detailed Logic)
        # ---------------------------------------------------------

        if os.path.isfile(procheck_sum_file):
            found_data = True
            try:
                with open(procheck_sum_file, "r") as f:
                    content = f.read()

                # A. Ramachandran Plot
                # | Ramachandran plot:   87.2% core   12.1% allow    0.3% gener    0.3% disall |
                m_rama = re.search(r"Ramachandran plot:\s*([\d\.]+)%\s*core\s*([\d\.]+)%\s*allow\s*([\d\.]+)%\s*gener\s*([\d\.]+)%\s*disall", content)
                if m_rama:
                    entry["rama_core"] = float(m_rama.group(1))
                    entry["rama_allow"] = float(m_rama.group(2))
                    entry["rama_gener"] = float(m_rama.group(3))
                    entry["rama_disall"] = float(m_rama.group(4))

                # B. Labelled Residues (All Ramachandrans)
                # | All Ramachandrans:   10 labelled residues (out of 350)                     |
                m_all_rama = re.search(r"All Ramachandrans:\s*(\d+)\s*labelled residues", content)
                if m_all_rama:
                    entry["labelled_rama_count"] = int(m_all_rama.group(1))

                # C. Chi1-chi2 plots
                # | Chi1-chi2 plots:      4 labelled residues (out of 198)                     |
                m_chi = re.search(r"Chi1-chi2 plots:\s*(\d+)\s*labelled residues", content)
                if m_chi:
                    entry["labelled_chi_count"] = int(m_chi.group(1))

                # D. Side-chain params
                # | Side-chain params:    4 better     0 inside      1 worse                   |
                m_side = re.search(r"Side-chain params:\s*(\d+)\s*better\s*(\d+)\s*inside\s*(\d+)\s*worse", content)

                if m_side:
                    entry["sidechain_better"] = int(m_side.group(1))
                    entry["sidechain_inside"] = int(m_side.group(2))
                    entry["sidechain_worse"] = int(m_side.group(3))

                # E. Residue properties (Line 1: Max deviation & Bad contacts)
                # | Residue properties: Max.deviation:     5.6              Bad contacts:    0 |
                m_res1 = re.search(r"Max\.deviation:\s*([\d\.]+)\s+Bad contacts:\s*(\d+)", content)
                if m_res1:
                    entry["max_deviation"] = float(m_res1.group(1))
                    entry["bad_contacts"] = int(m_res1.group(2))

                # F. Residue properties (Line 2: Bond len/angle)
                # |                     Bond len/angle:    2.3    Morris et al class:  1  3  3 |
                m_res2 = re.search(r"Bond len\/angle:\s*([\d\.]+)", content)
                if m_res2:
                    entry["bond_len_angle"] = float(m_res2.group(1))

                # G. Cis-peptides
                # |     2 cis-peptides                                                         |
                m_cis = re.search(r"(\d+)\s*cis-peptides", content)
                if m_cis:
                    entry["cis_peptides"] = int(m_cis.group(1))
                else:
                    # If the line isn't there, usually implies 0, but safest not to assume unless consistent.
                    # However, strictly if the text exists we parse it.
                    pass

                # H. G-factors
                # | G-factors           Dihedrals:   0.02  Covalent:   0.64    Overall:   0.26 |
                m_g = re.search(r"G-factors\s+Dihedrals:\s*([-\d\.]+)\s+Covalent:\s*([-\d\.]+)\s+Overall:\s*([-\d\.]+)", content)
                if m_g:
                    entry["g_factor_dihedral"] = float(m_g.group(1))
                    entry["g_factor_covalent"] = float(m_g.group(2))
                    entry["g_factor_overall"] = float(m_g.group(3))

                # I. Planar groups
                # | Planar groups:   100.0% within limits   0.0% highlighted                   |
                m_planar = re.search(r"Planar groups:\s*([\d\.]+)%\s*within limits", content)
                if m_planar:
                    entry["planar_groups_within_limits"] = float(m_planar.group(1))
            except Exception as e:
                print(f"[WARN] Error reading PROCHECK sum file {procheck_sum_file}: {e}")


        # ---------------------------------------------------------
        # Collect statistics into lists
        # ---------------------------------------------------------
        if found_data:
            structures_count += 1

        if "errat" in entry:
            errat_values.append(entry["errat"])

        if "verify3d" in entry:
            verify3d_values.append(entry["verify3d"])

        if "verify3d_status" in entry:
            verify3d_status_counter[entry["verify3d_status"]] += 1

        if "whatcheck" in entry:
            for heading in entry["whatcheck"]:
                whatcheck_counter[heading] += 1

        if "procheck" in entry:
            for key, val in entry["procheck"].items():
                try:
                    count = int(val.split(":")[1].strip())
                    procheck_totals[key] += count
                except:
                    pass

        # Collect Ramachandran stats
        # Auto-append all other keys (numerical ones from PROCHECK) to stats_collector
        # This handles rama, g-factors, bad contacts, etc. automatically

        for key, val in entry.items():

            if key not in ["errat", "verify3d", "verify3d_status", "whatcheck", "procheck"]:

                if isinstance(val, (int, float)):

                    stats_collector[key].append(val)

    # Final aggregated results
    result = {
        "ERRAT_average": sum(errat_values) / len(errat_values) if errat_values else None,
        "VERIFY3D_average_percent": sum(verify3d_values) / len(verify3d_values) if verify3d_values else None,
        "VERIFY3D_status_counts": dict(verify3d_status_counter),
        "WHATCHECK_heading_counts": dict(whatcheck_counter),
        "PROCHECK_summary_totals": dict(procheck_totals),
        "averages": {
            "ERRAT_score": sum(errat_values) / len(errat_values) if errat_values else None,
            "VERIFY3D_percent": sum(verify3d_values) / len(verify3d_values) if verify3d_values else None,
        },
        "N_structures_processed": structures_count
    }
    
    # Add averages for all PROCHECK detailed stats

    # Mapping keys to cleaner output names if desired, or just use keys as is
    for key, values in stats_collector.items():
        if values:
            avg_val = sum(values) / len(values)
            result["averages"][f"PROCHECK_{key}"] = avg_val

    return result


# === MAIN LOOP FOR ALL PDB FOLDERS ===
for pdb_code in os.listdir(BASE_ROOT):
    pdb_path = os.path.join(BASE_ROOT, pdb_code)

    # Ignore the summary folder itself
    if pdb_code.startswith("_"):
        continue

    if not os.path.isdir(pdb_path):
        continue

    print(f"Processing {pdb_code} ...")

    summary = process_pdb_folder(pdb_path)

    output_file = os.path.join(OUTPUT_DIR, f"{pdb_code}_summary.json")
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=4)

print("\n=== DONE! All summaries saved to _summarize_all_fixed/ ===")