import pickle
import os
import random
import torch
import pandas as pd
from torch.utils.data import Dataset

import rootutils
from tqdm import tqdm

root_dir = os.path.abspath(__file__)
root = rootutils.setup_root(root_dir, pythonpath=True)

# Valid text_mode values for ablation experiments:
#   "real"    – use the true description embedding for each sample
#   "shuffle" – randomly permute which description each sample receives
#               (same distribution, wrong alignment → tests specificity)
#   "random"  – replace every description with i.i.d. Gaussian noise
#               (tests whether any structured text signal is needed)
#   "zero"    – replace every description with an all-zero vector
#               (tests whether magnitude alone explains any effect)
TEXT_MODES = ("real", "shuffle", "random", "zero")


def load_embeddings(file_path):
    """Load embeddings from a pickle file and return the embeddings dictionary."""
    with open(file_path, "rb") as f:
        embeddings = pickle.load(f)
    return embeddings


def load_sequences(file_path):
    """Load sequences from a CSV file and return them as a dictionary."""
    df = pd.read_csv(file_path)
    print(df.columns)

    return {
        "heavy_chain": df['heavy_chain'].values,
        "light_chain": df['light_chain'].values,
        "antigen": df['antigen'].values,
        "description": df['description'].values,
        "pdb_id": df['pdb_id'].values
    }


class TraningDataset(Dataset):
    def __init__(self, ab_embedding_path, ag_embedding_path, ab_description_embedding_path, abset_path,
                 text_mode: str = "real", seed: int = 42):
        assert text_mode in TEXT_MODES, f"text_mode must be one of {TEXT_MODES}, got '{text_mode}'"
        self.text_mode = text_mode

        abset_path = os.path.join(root, abset_path)
        ab_embedding_path = os.path.join(root, ab_embedding_path)
        ag_embedding_path = os.path.join(root, ag_embedding_path)
        ab_description_embedding_path = os.path.join(root, ab_description_embedding_path)

        # Load embeddings
        self.ab_embedding = load_embeddings(ab_embedding_path)
        self.ag_embedding = load_embeddings(ag_embedding_path)
        self.ab_description_embedding = load_embeddings(ab_description_embedding_path)

        # Load sequence CSV
        sequence_df = load_sequences(abset_path)

        self.pairs = [
            {
                "antibody_key": (h, l),
                "antigen_key": ag,
                "description": d,
                "pdb_id": pdb_id
            }
            for h, l, ag, d, pdb_id in tqdm(
                zip(
                    sequence_df["heavy_chain"],
                    sequence_df["light_chain"],
                    sequence_df["antigen"],
                    sequence_df["description"],
                    sequence_df["pdb_id"]
                ),
                desc="Loading dataset"
            )
            if (h, l) in self.ab_embedding and ag in self.ag_embedding
        ]

        # Infer description embedding shape from first entry (needed for random/zero modes)
        first_des_emb = next(iter(self.ab_description_embedding.values()))
        self._des_emb_shape = first_des_emb.shape
        self._des_emb_dtype = first_des_emb.dtype

        # Pre-build a fixed shuffled index for reproducibility
        if text_mode == "shuffle":
            pdb_ids = [p["pdb_id"] for p in self.pairs]
            rng = random.Random(seed)
            self._shuffled_pdb_ids = pdb_ids[:]
            rng.shuffle(self._shuffled_pdb_ids)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]

        if self.text_mode == "real":
            des_emb = self.ab_description_embedding[item["pdb_id"]]
        elif self.text_mode == "shuffle":
            # Use a different sample's description — breaks specificity, keeps distribution
            des_emb = self.ab_description_embedding[self._shuffled_pdb_ids[idx]]
        elif self.text_mode == "random":
            des_emb = torch.randn(self._des_emb_shape, dtype=self._des_emb_dtype)
        else:  # "zero"
            des_emb = torch.zeros(self._des_emb_shape, dtype=self._des_emb_dtype)

        return {
            "pdb_id": item["pdb_id"],
            "ag_sequences": item["antigen_key"],
            "ab_sequence_heavy": item["antibody_key"][0],
            "ab_sequence_light": item["antibody_key"][1],
            "ab_sequences": item["antibody_key"][0] + "|" + item["antibody_key"][1],
            "description": item["description"],
            "description_embedding": des_emb,
            "ab_embedding": self.ab_embedding[item["antibody_key"]],
            "ag_embedding": self.ag_embedding[item["antigen_key"]],
        }
