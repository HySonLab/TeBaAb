import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import pickle
from typing import Dict, List
import logging

import pandas as pd
import torch
import yaml
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm

from src.models.CVAE import CVAE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AntibodyInference:

    def __init__(self, config_path: str):
        self.cfg = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.results = []

    def _load_config(self, config_path: str) -> DictConfig:
        with open(config_path, "r") as f:
            raw_cfg = yaml.safe_load(f)

        return OmegaConf.create(raw_cfg)

    def _load_model(self) -> None:
        checkpoint_path = self.cfg.get("checkpoint_path", "checkpoints/cvae/best_wo_des.ckpt")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading model from: {checkpoint_path}")
        self.model = CVAE.load_from_checkpoint(
            checkpoint_path,
            cfg=self.cfg,
            map_location=self.device
        )
        self.model = self.model.eval().to(self.device)
        logger.info(f"Model loaded on device: {self.device}")

    def _load_sequences(self, file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        required_cols = ['heavy_chain', 'light_chain', 'antigen', 'description', 'pdb_id']

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in CSV: {missing_cols}")

        logger.info(f"Loaded {len(df)} sequences from {file_path}")
        return df

    def _load_embeddings(self, file_path: str) -> Dict:
        with open(file_path, "rb") as f:
            embeddings = pickle.load(f)
        logger.info(f"Loaded embeddings from {file_path}")
        return embeddings

    def _generate_antibody(self, des_emb: torch.Tensor, antigen_emb: torch.Tensor) -> List[str]:
        with torch.no_grad():
            return self.model.generate_new_antibody(des_emb, antigen_emb)

    def _generate_valid_antibody(self, des_emb, antigen_emb, max_retries=5):
        for _ in range(max_retries):
            generated_seqs = self._generate_antibody(des_emb, antigen_emb)
            generated_seq = generated_seqs[0]

            # Ensure it's a string
            if not isinstance(generated_seq, str):
                continue

            # Check format: must contain exactly one '|'
            if "|" in generated_seq:
                parts = generated_seq.split("|")
                if len(parts) == 2 and all(part.strip() for part in parts):
                    return generated_seq

            # else -> retry
            continue

            # If still invalid after retries
        raise ValueError("Failed to generate valid antibody sequence after retries")

    def run_inference(self,
                      test_csv_path: str = None,
                      description_pkl_path: str = None,
                      antigen_pkl_path: str = None,
                      output_path: str = None) -> pd.DataFrame:

        test_csv_path = test_csv_path or self.cfg.get("test_csv_path", "datasets/abdes/test.csv")
        description_pkl_path = description_pkl_path or self.cfg.get("description_pkl_path",
                                                                    "datasets/cvae/test_description.pkl")
        antigen_pkl_path = antigen_pkl_path or self.cfg.get("antigen_pkl_path", "datasets/cvae/test_antigen.pkl")

        if self.model is None:
            self._load_model()

        sequence_df = self._load_sequences(test_csv_path)
        des_embs = self._load_embeddings(description_pkl_path)
        antigen_embs = self._load_embeddings(antigen_pkl_path)

        self.results = []
        logger.info("Starting inference...")

        for idx, row in tqdm(sequence_df.iterrows(), total=len(sequence_df), desc="Generating antibodies"):
            try:
                pdb_id = row['pdb_id']
                antigen_key = row['antigen']

                if pdb_id not in des_embs:
                    logger.warning(f"Description embedding not found for PDB: {pdb_id}")
                    continue

                if antigen_key not in antigen_embs:
                    logger.warning(f"Antigen embedding not found for: {antigen_key}")
                    continue

                des_emb = des_embs[pdb_id].unsqueeze(0).to(self.device)
                antigen_emb = antigen_embs[antigen_key].unsqueeze(0).to(self.device)

                generated_seq = self._generate_valid_antibody(des_emb, antigen_emb, 5)

                if generated_seq:
                    if "|" in generated_seq:
                        gen_heavy, gen_light = generated_seq.split("|", 1)
                    else:
                        gen_heavy, gen_light = generated_seq, ""
                else:
                    gen_heavy, gen_light = "", ""

                result = {
                    'pdb_id': pdb_id,
                    'gt_heavy': row['heavy_chain'],
                    'gt_light': row['light_chain'],
                    'generated_heavy': gen_heavy,
                    'generated_light': gen_light,
                    'antigen': row['antigen'],
                    'description': row['description'],
                    'full_generated': generated_seq if generated_seq else "",
                }

                self.results.append(result)

            except Exception as e:
                logger.error(f"Error processing sequence {idx} (PDB: {row.get('pdb_id', 'N/A')}): {e}")
                continue

        results_df = pd.DataFrame(self.results)

        if output_path:
            self.save_results(results_df, output_path)

        return results_df

    def save_results(self, results_df: pd.DataFrame, output_path: str) -> None:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

        results_df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(results_df)} results to: {output_path}")

    def print_sample_results(self, results_df: pd.DataFrame, n_samples: int = 5) -> None:
        logger.info(f"\n=== Sample Results ({n_samples}) ===")
        for idx, row in results_df.head(n_samples).iterrows():
            print(f"\nPDB: {row['pdb_id']}")
            print(f"GT Heavy:  {row['gt_heavy'][:50]}...")
            print(f"GT Light:  {row['gt_light'][:50]}...")
            print(f"Gen Heavy: {row['generated_heavy'][:50]}...")
            print(f"Gen Light: {row['generated_light'][:50]}...")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Antibody Generation Inference")

    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/training.yaml",
        help="Path to config file"
    )

    parser.add_argument(
        "--test_csv",
        type=str,
        default=None,
        help="Path to test CSV file"
    )

    parser.add_argument(
        "--description_pkl",
        type=str,
        default=None,
        help="Path to description embeddings pickle file"
    )

    parser.add_argument(
        "--antigen_pkl",
        type=str,
        default=None,
        help="Path to antigen embeddings pickle file"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="results/inference_results.csv",
        help="Output CSV file path"
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of sample results to display"
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    try:
        inferencer = AntibodyInference(args.config)

        results_df = inferencer.run_inference(
            test_csv_path=args.test_csv,
            description_pkl_path=args.description_pkl,
            antigen_pkl_path=args.antigen_pkl,
            output_path=args.output
        )

        inferencer.print_sample_results(results_df, args.samples)

        logger.info("Inference completed!")

    except Exception as e:
        logger.error(f"Error during inference: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
