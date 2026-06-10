import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import pickle
from typing import Dict, List, Tuple
import logging

import pandas as pd
import torch
import yaml
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm

from src.models.modules.maskedLanguageModel import MaskedLanguageModel
from src.utils.constants import convert_seqs2ids, get_token2id

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AntibodyMaskedInference:
    """Inference engine for masked language model-based antibody design."""

    def __init__(self, config_path: str):
        self.cfg = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.results = []

    def _load_config(self, config_path: str) -> DictConfig:
        with open(config_path, "r") as f:
            raw_cfg = yaml.safe_load(f)
        return OmegaConf.create(raw_cfg)

    def _load_model(self, checkpoint_path: str) -> None:
        """Load the trained MLM model."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading MLM from: {checkpoint_path}")
        self.model = MaskedLanguageModel.load_from_checkpoint(
            checkpoint_path,
            cfg=self.cfg,
            map_location=self.device
        )
        self.model = self.model.eval().to(self.device)
        logger.info(f"Model loaded on device: {self.device}")

    def _load_sequences(self, file_path: str) -> pd.DataFrame:
        """Load sequences from CSV."""
        df = pd.read_csv(file_path)
        required_cols = ['heavy_chain', 'light_chain', 'antigen', 'description', 'pdb_id']

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in CSV: {missing_cols}")

        logger.info(f"Loaded {len(df)} sequences from {file_path}")
        return df

    def _load_embeddings(self, file_path: str) -> Dict:
        """Load embeddings from pickle file."""
        with open(file_path, "rb") as f:
            embeddings = pickle.load(f)
        logger.info(f"Loaded embeddings from {file_path}")
        return embeddings

    def design_masked_region(
            self,
            full_sequence: str,
            chain_to_mask: str,  # "heavy" or "light"
            des_emb: torch.Tensor,
            antigen_emb: torch.Tensor,
            temperature: float = 1.0,
            greedy: bool = True
    ) -> str:
        """
        Design a masked region of an antibody sequence.
        
        Args:
            full_sequence: Full antibody sequence (heavy|light)
            chain_to_mask: Which chain to mask ("heavy" or "light")
            des_emb: Description embedding
            antigen_emb: Antigen embedding
            temperature: Sampling temperature
            greedy: Use greedy decoding if True
            
        Returns:
            Designed sequence with the masked region filled
        """
        token2id = get_token2id()

        # Parse heavy and light chains
        parts = full_sequence.split("|")
        if len(parts) != 2:
            raise ValueError(f"Invalid sequence format: {full_sequence}")

        heavy_chain, light_chain = parts

# 1. Construct original unmasked sequence
        full_original_seq = heavy_chain + "|" + light_chain

        # 2. Convert to token IDs (add_sos=True adds <SOS> at index 0)
        seq_ids = torch.tensor(
            convert_seqs2ids([full_original_seq], add_sos=True, add_eos=True, max_length=self.cfg.model.max_ab_seq_len)[0],
            dtype=torch.long
        ).unsqueeze(0).to(self.device)

        # 3. Get the correct mask token ID (with a fallback just in case)
        mask_token_id = token2id.get("<mask>", token2id.get("<mask>", token2id.get("<pad>", 0)))

        # 4. Safely overwrite the specific chain's IDs with the mask token
        if chain_to_mask.lower() == "heavy":
            # Mask heavy chain: from index 1 (after SOS) up to the length of the heavy chain
            seq_ids[0, 1 : 1 + len(heavy_chain)] = mask_token_id
            
        elif chain_to_mask.lower() == "light":
            # Mask light chain: from after the '|' separator to the end of the light chain
            # The separator '|' is located at index (1 + len(heavy_chain))
            light_start_idx = 1 + len(heavy_chain) + 1
            light_end_idx = light_start_idx + len(light_chain)
            seq_ids[0, light_start_idx : light_end_idx] = mask_token_id
            
        else:
            raise ValueError(f"Invalid chain_to_mask: {chain_to_mask}")

        # Prepare embeddings
        des_emb = des_emb.unsqueeze(0).to(self.device) if des_emb.dim() == 1 else des_emb.to(self.device)
        antigen_emb = antigen_emb.unsqueeze(0).to(self.device) if antigen_emb.dim() == 1 else antigen_emb.to(self.device)

        # Get predictions from model
        with torch.inference_mode():
            completed_ids, completed_seqs = self.model.predict_masked_positions(
                seq_ids,
                # Add .float() to ensure this placeholder is float32, not long
                seq_ids.new_zeros((seq_ids.shape[0], self.cfg.model.ab_dimen)).float(), 
                antigen_emb,
                des_emb,
                temperature=temperature,
                greedy=greedy
            )

        return completed_seqs[0]

    def run_inference(
            self,
            checkpoint_path: str,
            test_csv_path: str = None,
            description_pkl_path: str = None,
            antigen_pkl_path: str = None,
            output_path: str = None,
            chain_to_mask: str = "heavy",
            temperature: float = 1.0
    ) -> pd.DataFrame:
        """
        Run inference to design masked antibody regions.
        
        Args:
            checkpoint_path: Path to trained MLM checkpoint
            test_csv_path: Path to test CSV
            description_pkl_path: Path to description embeddings
            antigen_pkl_path: Path to antigen embeddings
            output_path: Where to save results
            chain_to_mask: Which chain to design ("heavy" or "light")
            temperature: Sampling temperature
            
        Returns:
            DataFrame with results
        """
        # Load model
        self._load_model(checkpoint_path)

        # Load data
        # 1. Safely grab the 'data' subset of your config (default to empty dict if missing)
        data_cfg = self.cfg.get("data", {})

        # 2. Get the specific paths using the correct YAML keys
        test_csv_path = test_csv_path or data_cfg.get("test_csv", "../datasets/abdes/test.csv")
        description_pkl_path = description_pkl_path or data_cfg.get("ab_description_embedding_path", "../datasets/abdes/.pkl")
        antigen_pkl_path = antigen_pkl_path or data_cfg.get("ag_embedding_path", "../datasets/abdes/.pkl")
        sequence_df = self._load_sequences(test_csv_path)
        des_embs = self._load_embeddings(description_pkl_path)
        antigen_embs = self._load_embeddings(antigen_pkl_path)

        self.results = []
        logger.info(f"Starting inference for {chain_to_mask} chain design...")

        for idx, row in tqdm(sequence_df.iterrows(), total=len(sequence_df), desc="Designing antibodies"):
            try:
                pdb_id = row['pdb_id']
                antigen_key = row['antigen']

                # Get embeddings
                if pdb_id not in des_embs:
                    logger.warning(f"Description embedding not found for PDB: {pdb_id}")
                    continue

                if antigen_key not in antigen_embs:
                    logger.warning(f"Antigen embedding not found for: {antigen_key}")
                    continue

                des_emb = torch.tensor(des_embs[pdb_id], dtype=torch.float32)
                antigen_emb = torch.tensor(antigen_embs[antigen_key], dtype=torch.float32)

                # Original full sequence
                original_seq = row['heavy_chain'] + "|" + row['light_chain']

                # Design masked region
                designed_seq = self.design_masked_region(
                    original_seq,
                    chain_to_mask=chain_to_mask,
                    des_emb=des_emb,
                    antigen_emb=antigen_emb,
                    temperature=temperature,
                    greedy=True
                )

                # Parse designed sequence
                if "|" in designed_seq:
                    des_heavy, des_light = designed_seq.split("|", 1)
                else:
                    des_heavy, des_light = designed_seq, ""

                # Store result
                result = {
                    'pdb_id': pdb_id,
                    'gt_heavy': row['heavy_chain'],
                    'gt_light': row['light_chain'],
                    'designed_heavy': des_heavy,
                    'designed_light': des_light,
                    'chain_masked': chain_to_mask,
                    'antigen': row['antigen'],
                    'description': row['description'],
                    'full_designed': designed_seq
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
        """Save results to CSV."""
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        results_df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(results_df)} results to: {output_path}")

    def print_sample_results(self, results_df: pd.DataFrame, n_samples: int = 5) -> None:
        """Print sample results."""
        logger.info(f"\n=== Sample Results ({n_samples}) ===")
        for idx, row in results_df.head(n_samples).iterrows():
            print(f"\nPDB: {row['pdb_id']}")
            # print(f"GT {row['chain_masked'].upper()}: {row[f'gt_{row[\"chain_masked\"]}'][:60]}")
            print(f"""GT {row['chain_masked'].upper()}: {row[f'gt_{row["chain_masked"]}'][:60]}""")                                                                     
            # print(f"Designed:     {row[f'designed_{row[\"chain_masked\"]}'][:60]}")
            print(f"""Designed:     {row[f'designed_{row["chain_masked"]}'][:60]}""")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Masked Language Model Inference for Antibody Design")

    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/training.yaml",
        help="Path to config file"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained MLM checkpoint"
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
        "--output",
        type=str,
        default="inference_results/mlm_design_results.csv",
        help="Output path for results CSV"
    )

    parser.add_argument(
        "--chain",
        type=str,
        choices=["heavy", "light"],
        default="heavy",
        help="Which chain to design"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Create inference engine
    inference = AntibodyMaskedInference(args.config)

    # Run inference
    results_df = inference.run_inference(
        checkpoint_path=args.checkpoint,
        test_csv_path=args.test_csv,
        description_pkl_path=args.description_pkl,
        antigen_pkl_path=args.antigen_pkl,
        output_path=args.output,
        chain_to_mask=args.chain,
        temperature=args.temperature
    )

    # Print sample results
    if len(results_df) > 0:
        inference.print_sample_results(results_df, n_samples=5)
        logger.info(f"\nTotal designed: {len(results_df)}")
    else:
        logger.warning("No results generated!")
