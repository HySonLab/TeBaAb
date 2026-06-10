import pandas as pd
import difflib
from Bio import Align

def calculate_similarity(seq1, seq2):
    """
    Calculates sequence similarity using difflib.
    Returns a score between 0.0 (completely different) and 1.0 (identical).
    """
    if pd.isna(seq1) or pd.isna(seq2):
        return 0.0
    return difflib.SequenceMatcher(None, str(seq1), str(seq2)).ratio()

def calculate_biopython_alignment(seq1, seq2):
    """
    Calculates sequence similarity using BioPython's PairwiseAligner.
    """
    if pd.isna(seq1) or pd.isna(seq2):
        return 0.0
    aligner = Align.PairwiseAligner()
    score = aligner.score(str(seq1), str(seq2))
    return score / max(len(str(seq1)), len(str(seq2)))

def evaluate_chains(experiment_name, input_file, gt_col, designed_col, output_file, metric_func=calculate_similarity):
    """
    Evaluates generated sequences, saves the results, and extracts key examples.
    Returns a dictionary of summary statistics for global comparison.
    """
    print(f"Processing: {experiment_name}...")
    
    # Read dataset
    df = pd.read_csv(input_file)
    
    # Calculate similarity
    similarity_col_name = 'sequence_similarity'
    df[similarity_col_name] = df.apply(
        lambda row: metric_func(row[gt_col], row[designed_col]), 
        axis=1
    )
    
    # Save the updated dataframe
    df.to_csv(output_file, index=False)
    
    # --- Extract Qualitative Examples ---
    print(f"\n[Examples] {experiment_name}")
    if not df.empty:
        # Find indices for Best, Worst, and Median scores
        best_idx = df[similarity_col_name].idxmax()
        worst_idx = df[similarity_col_name].idxmin()
        # Find the value closest to the median
        median_val = df[similarity_col_name].median()
        median_idx = (df[similarity_col_name] - median_val).abs().idxmin()
        
        examples = {
            "BEST  ": df.loc[best_idx],
            "MEDIAN": df.loc[median_idx],
            "WORST ": df.loc[worst_idx]
        }
        
        for label, row in examples.items():
            score = row[similarity_col_name]
            # Truncating to 80 characters for terminal readability; remove [:80] if you want full sequences
            gt_seq = str(row[gt_col])[:80] + "..." if len(str(row[gt_col])) > 80 else str(row[gt_col])
            des_seq = str(row[designed_col])[:80] + "..." if len(str(row[designed_col])) > 80 else str(row[designed_col])
            
            print(f"  {label} (Score: {score:.4f})")
            print(f"    GT : {gt_seq}")
            print(f"    Des: {des_seq}")
    print("-" * 60)
    
    # --- Gather Summary Statistics ---
    stats = df[similarity_col_name].describe().to_dict()
    stats['Experiment'] = experiment_name
    return stats

if __name__ == "__main__":
    # Define all experiments in a list of dictionaries for easy management
    base_dir = '/home/huyhoang/Work/HysonLab/Repo/TeBaAb/scripts/inter_outputs/'
    
    configs = [
        # 1. Baseline
        # {"name": "Baseline - Heavy", "in": "results_heavy_2.csv", "out": "evaluated_results_heavy.csv", "gt": "gt_heavy", "des": "designed_heavy"},
        # {"name": "Baseline - Light", "in": "results_light_2.csv", "out": "evaluated_results_light.csv", "gt": "gt_light", "des": "designed_light"},
        
        # # 2. With Chain Description
        # {"name": "Chain Desc - Heavy", "in": "results_w_chain_desc_heavy.csv", "out": "evaluated_results_w_chain_desc_heavy.csv", "gt": "gt_heavy", "des": "designed_heavy"},
        # {"name": "Chain Desc - Light", "in": "results_w_chain_desc_light.csv", "out": "evaluated_results_w_chain_desc_light.csv", "gt": "gt_light", "des": "designed_light"},
        
        # # 3. Full Description + Chain Description
        # {"name": "Full Desc - Heavy", "in": "results_full_desc_w_chain_desc_heavy.csv", "out": "evaluated_results_full_desc_w_chain_desc_heavy.csv", "gt": "gt_heavy", "des": "designed_heavy"},
        # {"name": "Full Desc - Light", "in": "results_full_desc_w_chain_desc_light.csv", "out": "evaluated_results_full_desc_w_chain_desc_light.csv", "gt": "gt_light", "des": "designed_light"},
        
        
        {"name": "PDB_desc - Heavy", "in": "results_swissisolated_heavy.csv", "out": "evaluated_results_swissisolated_heavy.csv", "gt": "gt_heavy", "des": "designed_heavy"},
        {"name": "PDB_desc - Light", "in": "results_swissisolated_light.csv", "out": "evaluated_results_swissisolated_light.csv", "gt": "gt_light", "des": "designed_light"},
        
        # 2. With Chain Description
        {"name": "SwissProt - Heavy", "in": "results_swissisolated_w_chain_desc_heavy.csv", "out": "evaluated_results_swissisolated_w_chain_desc_heavy.csv", "gt": "gt_heavy", "des": "designed_heavy"},
        {"name": "SwissProt - Light", "in": "results_swissisolated_w_chain_desc_light.csv", "out": "evaluated_results_swissisolated_w_chain_desc_light.csv", "gt": "gt_light", "des": "designed_light"},
        
        # 3. Full Description + Chain Description
        {"name": "PDB_desc + SwissProt - Heavy", "in": "results_swissisolated_full_desc_w_chain_desc_heavy.csv", "out": "evaluated_results_swissisolated_full_desc_w_chain_desc_heavy.csv", "gt": "gt_heavy", "des": "designed_heavy"},
        {"name": "PDB_desc + SwissProt - Light", "in": "results_swissisolated_full_desc_w_chain_desc_light.csv", "out": "evaluated_results_swissisolated_full_desc_w_chain_desc_light.csv", "gt": "gt_light", "des": "designed_light"},

    ]
    
    all_stats = []
    
    # Run the evaluation loop
    for cfg in configs:
        stats = evaluate_chains(
            experiment_name=cfg["name"],
            input_file=base_dir + cfg["in"],
            output_file=cfg["out"], # Saving output locally, adjust path if needed
            gt_col=cfg["gt"],
            designed_col=cfg["des"],
            metric_func=calculate_biopython_alignment # Swap to calculate_biopython_alignment if desired
        )
        all_stats.append(stats)
        
    # Compile and print the final global comparison table
    summary_df = pd.DataFrame(all_stats)
    
    # Reorder columns to put Experiment name first, then standard describe() metrics
    cols = ['Experiment', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    summary_df = summary_df[cols]
    
    print("\n" + "="*80)
    print("FINAL COMPARISON SUMMARY - SwissProt Isolated Test Set")
    print("="*80)
    # Using markdown output for a clean, highly readable terminal table
    print(summary_df.to_markdown(index=False, floatfmt=".4f"))
    print("="*80)

    summary_df.to_csv(base_dir + "final_comparison_summary_swissisolated.csv", index=False)