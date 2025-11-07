import pickle
import os
import pandas as pd
from torch.utils.data import Dataset

import rootutils
from tqdm import tqdm

root_dir = os.path.abspath(__file__)
root = rootutils.setup_root(root_dir, pythonpath=True)


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
    def __init__(self, ab_embedding_path, ag_embedding_path, ab_description_embedding_path, abset_path):
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

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        return {
            "pdb_id": item["pdb_id"],
            "ag_sequences": item["antigen_key"],
            "ab_sequence_heavy": item["antibody_key"][0],
            "ab_sequence_light": item["antibody_key"][1],
            "ab_sequences": item["antibody_key"][0] + "|" + item["antibody_key"][1],
            "description": item["description"],
            "description_embedding": self.ab_description_embedding[item["pdb_id"]],
            "ab_embedding": self.ab_embedding[item["antibody_key"]],
            "ag_embedding": self.ag_embedding[item["antigen_key"]],
        }
