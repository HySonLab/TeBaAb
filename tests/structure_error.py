import argparse
import os
import tempfile
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import math
import logging

from ImmuneBuilder import ABodyBuilder2
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ABodyBuilderEvaluator:

    def __init__(self, weights_dir: Optional[str] = None, temp_dir: Optional[str] = None):
        self.predictor = ABodyBuilder2(weights_dir=weights_dir)
        self.temp_dir = temp_dir or tempfile.gettempdir()

        # ANARCI CDR definitions (Chothia scheme)
        self.cdr_ranges = {
            'H': {
                'CDR1': (26, 32),
                'CDR2': (52, 56),
                'CDR3': (95, 102)
            },
            'L': {
                'CDR1': (24, 34),
                'CDR2': (50, 56),
                'CDR3': (89, 97)
            }
        }

        self.region_order = [
            "Framework H-chain", "CDR-H1", "CDR-H2", "CDR-H3",
            "Framework L-chain", "CDR-L1", "CDR-L2", "CDR-L3"
        ]

    def predict_structure(self, heavy: str, light: str, pdb_id: str) -> str:
        """Predict antibody structure and return PDB file path"""
        sequences = {'H': heavy, 'L': light}

        # Create temporary file
        temp_pdb = os.path.join(self.temp_dir, f"antibody_{pdb_id}.pdb")

        try:
            antibody = self.predictor.predict(sequences)
            antibody.save(temp_pdb)
            # antibody.save_all()
            return temp_pdb
        except Exception as e:
            if os.path.exists(temp_pdb):
                os.remove(temp_pdb)
            raise e

    def parse_pdb_file(self, filename: str) -> Dict[str, List[Tuple[int, float]]]:
        """Parse PDB file and extract residue numbers with B-factors by chain"""
        chain_data = defaultdict(list)

        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    atom_name = line[12:16].strip()
                    chain = line[21]
                    residue_num = int(line[22:26].strip())
                    b_factor = float(line[60:66].strip())

                    # Only consider CA atoms to avoid duplicates
                    if atom_name == 'CA':
                        chain_data[chain].append((residue_num, b_factor))

        return dict(chain_data)

    def classify_residue(self, chain: str, residue_num: int) -> str:
        """Classify residue as Framework or CDR based on ANARCI numbering"""
        chain_type = 'H' if chain == 'H' else 'L'

        for cdr_name, (start, end) in self.cdr_ranges[chain_type].items():
            if start <= residue_num <= end:
                return f"CDR-{chain}{cdr_name[-1]}"

        return f"Framework {chain}-chain"

    def calculate_region_errors(self, chain_data: Dict[str, List[Tuple[int, float]]]) -> Dict[str, float]:
        """Calculate RMS prediction error for each antibody region"""
        region_errors = defaultdict(list)

        for chain, residues in chain_data.items():
            if chain not in ['H', 'L']:
                continue

            sorted_residues = sorted(residues, key=lambda x: x[0])

            for residue_num, b_factor in sorted_residues:
                region = self.classify_residue(chain, residue_num)
                region_errors[region].append(b_factor)

        # Calculate RMS errors
        rms_errors = {}
        for region, errors in region_errors.items():
            if errors:
                sum_squares = sum(error ** 2 for error in errors)
                rms = math.sqrt(sum_squares / len(errors))
                rms_errors[region] = round(rms, 2)

        return rms_errors

    def analyze_structure(self, heavy: str, light: str, pdb_id: str) -> Dict[str, float]:
        """Analyze antibody structure and return region errors"""
        pdb_file = None
        try:
            # Predict structure
            pdb_file = self.predict_structure(heavy, light, pdb_id)

            # Parse PDB and calculate errors
            chain_data = self.parse_pdb_file(pdb_file)
            region_errors = self.calculate_region_errors(chain_data)

            return region_errors

        except Exception as e:
            logger.error(f"Error analyzing structure: {e}")
            return {}
        finally:
            # Clean up temporary file
            if pdb_file and os.path.exists(pdb_file):
                os.remove(pdb_file)

    def evaluate_antibody_results(self, input_file: str, output_file: str = None) -> pd.DataFrame:
        """Evaluate antibody generation results using ABodyBuilder2"""
        # Load data
        df = pd.read_csv(input_file)

        # Check required columns
        required_cols = ['pdb_id', 'original_ab', 'optimized_ab']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        logger.info(f"Loaded {len(df)} antibody pairs for structure analysis")

        results = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing structures"):
            try:
                pdb_id = row['pdb_id']

                # Clean sequences
                gt_heavy = str(row['original_ab']).split("|")[0].strip()
                gt_light = str(row['original_ab']).split("|")[1].strip()
                gen_heavy = str(row['optimized_ab']).split("|")[0].strip()
                gen_light = str(row['optimized_ab']).split("|")[1].strip()

                # Skip if sequences are too short
                if len(gt_heavy) < 50 or len(gen_heavy) < 50:
                    logger.warning(f"Skipping {pdb_id}: sequences too short")
                    continue

                # Analyze ground truth structure
                gt_errors = self.analyze_structure(gt_heavy, gt_light, pdb_id)

                # Analyze generated structure
                gen_errors = self.analyze_structure(gen_heavy, gen_light, pdb_id)

                # Prepare result
                result = {
                    'pdb_id': pdb_id,
                    'gt_heavy_seq': gt_heavy,
                    'gt_light_seq': gt_light,
                    'gen_heavy_seq': gen_heavy,
                    'gen_light_seq': gen_light
                }

                # Add region-specific errors
                for region in self.region_order:
                    result[f'gt_{region.lower().replace(" ", "_").replace("-", "_")}'] = gt_errors.get(region, 0.0)
                    result[f'gen_{region.lower().replace(" ", "_").replace("-", "_")}'] = gen_errors.get(region, 0.0)

                    # Calculate difference
                    gt_val = gt_errors.get(region, 0.0)
                    gen_val = gen_errors.get(region, 0.0)
                    result[f'diff_{region.lower().replace(" ", "_").replace("-", "_")}'] = gen_val - gt_val

                results.append(result)

            except Exception as e:
                logger.error(f"Error processing {row.get('pdb_id', idx)}: {e}")
                continue

        results_df = pd.DataFrame(results)

        if output_file:
            self.save_results(results_df, output_file)

        return results_df

    def save_results(self, results_df: pd.DataFrame, output_file: str):
        """Save evaluation results to CSV"""
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        results_df.to_csv(output_file, index=False)
        logger.info(f"Structure analysis results saved to: {output_file}")

    def print_statistics(self, results_df: pd.DataFrame):
        """Print summary statistics"""
        if len(results_df) == 0:
            logger.warning("No results to analyze")
            return

        logger.info("\n=== ABodyBuilder2 Structure Analysis Statistics ===")

        # Calculate average errors for each region
        for region in self.region_order:
            region_key = region.lower().replace(" ", "_").replace("-", "_")
            gt_col = f'gt_{region_key}'
            gen_col = f'gen_{region_key}'
            diff_col = f'diff_{region_key}'

            if gt_col in results_df.columns and gen_col in results_df.columns:
                gt_mean = results_df[gt_col].mean()
                gen_mean = results_df[gen_col].mean()
                diff_mean = results_df[diff_col].mean()
                diff_std = results_df[diff_col].std()

                print(f"\n{region}:")
                print(f"  GT Mean Error: {gt_mean:.3f}")
                print(f"  Generated Mean Error: {gen_mean:.3f}")
                print(f"  Difference: {diff_mean:.3f} ± {diff_std:.3f}")

        # Overall statistics
        gt_cols = [col for col in results_df.columns if col.startswith('gt_') and not col.endswith('_seq')]
        gen_cols = [col for col in results_df.columns if col.startswith('gen_') and not col.endswith('_seq')]

        if gt_cols and gen_cols:
            overall_gt_mean = results_df[gt_cols].mean().mean()
            overall_gen_mean = results_df[gen_cols].mean().mean()

            print(f"\nOverall Statistics:")
            print(f"  Average GT Error: {overall_gt_mean:.3f}")
            print(f"  Average Generated Error: {overall_gen_mean:.3f}")
            print(f"  Overall Difference: {overall_gen_mean - overall_gt_mean:.3f}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="ABodyBuilder2 Structure Analysis")

    parser.add_argument(
        "input_file",
        type=str,
        help="Input CSV file with antibody generation results"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output CSV file (default: input_file_structure_errors.csv)"
    )

    parser.add_argument(
        "--weights_dir",
        type=str,
        default=None,
        help="Directory containing ABodyBuilder2 weights"
    )

    parser.add_argument(
        "--temp_dir",
        type=str,
        default=None,
        help="Temporary directory for PDB files"
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    try:
        # Set default output path
        if args.output is None:
            base_name = os.path.splitext(os.path.basename(args.input_file))[0]
            output_dir = os.path.dirname(args.input_file) or '.'
            args.output = os.path.join(output_dir, f"{base_name}_structure_errors.csv")

        # Initialize evaluator
        evaluator = ABodyBuilderEvaluator(
            weights_dir=args.weights_dir,
            temp_dir=args.temp_dir
        )

        # Run evaluation
        results_df = evaluator.evaluate_antibody_results(args.input_file, args.output)

        # Print statistics
        evaluator.print_statistics(results_df)

        logger.info("Structure analysis completed!")

    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())