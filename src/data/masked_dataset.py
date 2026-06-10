import pickle
import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import rootutils

from src.utils.constants import convert_seqs2ids, get_token2id, VOCAB

root_dir = os.path.abspath(__file__)
root = rootutils.setup_root(root_dir, pythonpath=True)


def load_embeddings(file_path):
    """Load embeddings from a pickle file."""
    with open(file_path, "rb") as f:
        embeddings = pickle.load(f)
    return embeddings


def load_sequences(file_path):
    """Load sequences from a CSV file."""
    df = pd.read_csv(file_path)
    return {
        "heavy_chain": df['heavy_chain'].values,
        "light_chain": df['light_chain'].values,
        "antigen": df['antigen'].values,
        "description": df['description'].values,
        "pdb_id": df['pdb_id'].values
    }


class MaskedAntibodyDataset(Dataset):
    """
    Dataset for masked language modeling on antibody sequences.
    
    Randomly masks regions (heavy chain, light chain, or custom) and creates
    training targets for predicting masked positions.
    """

    def __init__(
            self,
            ab_embedding_path: str,
            ag_embedding_path: str,
            ab_description_embedding_path: str,
            abset_path: str,
            mask_strategy: str = "random",  # "random", "heavy", "light", "both"
            mask_ratio: float = 0.15,
            max_ab_seq_len: int = 256
    ):
        """
        Initialize the masked dataset.
        
        Args:
            ab_embedding_path: Path to antibody embeddings
            ag_embedding_path: Path to antigen embeddings
            ab_description_embedding_path: Path to description embeddings
            abset_path: Path to sequence CSV
            mask_strategy: How to mask sequences
            mask_ratio: Fraction of tokens to mask (for "random" strategy)
            max_ab_seq_len: Maximum sequence length
        """
        abset_path = os.path.join(root, abset_path)
        ab_embedding_path = os.path.join(root, ab_embedding_path)
        ag_embedding_path = os.path.join(root, ag_embedding_path)
        ab_description_embedding_path = os.path.join(root, ab_description_embedding_path)

        # Load embeddings
        self.ab_embedding = load_embeddings(ab_embedding_path)
        self.ag_embedding = load_embeddings(ag_embedding_path)
        self.ab_description_embedding = load_embeddings(ab_description_embedding_path)

        # Load sequences
        sequence_df = load_sequences(abset_path)

        # Create pairs
        self.pairs = [
            {
                "heavy_chain": h,
                "light_chain": l,
                "antigen": ag,
                "description": d,
                "pdb_id": pdb_id,
                "antibody_key": (h, l),
                "antigen_key": ag
            }
            for h, l, ag, d, pdb_id in tqdm(
                zip(
                    sequence_df["heavy_chain"],
                    sequence_df["light_chain"],
                    sequence_df["antigen"],
                    sequence_df["description"],
                    sequence_df["pdb_id"]
                ),
                desc="Loading masked dataset"
            )
            if (h, l) in self.ab_embedding and ag in self.ag_embedding
        ]

        self.mask_strategy = mask_strategy
        self.mask_ratio = mask_ratio
        self.max_ab_seq_len = max_ab_seq_len
        
        # Get token IDs
        self.token2id = get_token2id() # Store as class attribute for easier access later
        self.mask_idx = self.token2id.get("<mask>", self.token2id["<pad>"])
        self.pad_idx = self.token2id["<pad>"]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        
        # Get full antibody sequence
        full_seq = item["heavy_chain"] + "|" + item["light_chain"]
        
        # Create target sequence (original)
        target_ids = torch.tensor(
            convert_seqs2ids([full_seq], add_sos=True, add_eos=True, max_length=self.max_ab_seq_len)[0],
            dtype=torch.long
        )
        
        # Create masked sequence and mask positions
        masked_ids, mask_positions = self._create_masked_sequence(full_seq, target_ids)
        
        # --- FIX: Fetch embedding with fallback to prevent KeyError ---
        # Assuming SciBert embedding dim is 768. Adjust if your model outputs a different size.
        des_emb = self.ab_description_embedding.get(item["pdb_id"], torch.zeros(768))

        return {
            "pdb_id": item["pdb_id"],
            "ab_sequence_heavy": item["heavy_chain"],
            "ab_sequence_light": item["light_chain"],
            "ab_sequences": full_seq,
            "antigen": item["antigen"],
            "description": item["description"],
            
            "masked_seq_ids": masked_ids,
            "target_seq_ids": target_ids,
            "mask_positions": mask_positions,
            
            "ab_embedding": self.ab_embedding[item["antibody_key"]].clone().detach().to(dtype=torch.float32),
            "ag_embedding": self.ag_embedding[item["antigen_key"]].clone().detach().to(dtype=torch.float32),
            "des_embedding": des_emb.clone().detach().to(dtype=torch.float32),
        }

    def _create_masked_sequence(
            self,
            full_seq: str,
            target_ids: torch.Tensor
    ) -> tuple:
        """
        Create masked sequence based on masking strategy.
        
        Args:
            full_seq: Full sequence string (heavy|light)
            target_ids: Original sequence IDs
            
        Returns:
            masked_ids: Masked sequence IDs
            mask_positions: Binary mask (1 where masked, 0 otherwise)
        """
        masked_ids = target_ids.clone()
        mask_positions = torch.zeros_like(target_ids, dtype=torch.bool)
        
        # --- FIX: Retrieve token2id once outside the loop ---
        token2id = self.token2id 
        sep_token_id = token2id.get("|", 100)
        
        # Find separator position (where '|' is)
        sep_idx = -1
        for i, seq_id in enumerate(target_ids):
            if seq_id == sep_token_id:
                sep_idx = i
                break
        
        if self.mask_strategy == "random":
            # Randomly mask tokens (excluding special tokens)
            num_to_mask = max(1, int(len(target_ids) * self.mask_ratio))
            maskable_positions = [i for i in range(len(target_ids)) 
                                 if target_ids[i] not in [token2id["<sos>"], token2id["<eos>"], token2id["<pad>"]]]
            
            if maskable_positions:
                mask_idxs = np.random.choice(maskable_positions, size=min(num_to_mask, len(maskable_positions)), replace=False)
                for idx in mask_idxs:
                    masked_ids[idx] = self.mask_idx
                    mask_positions[idx] = True
                    
        elif self.mask_strategy == "heavy":
            # Mask heavy chain
            if sep_idx > 0:
                for i in range(1, sep_idx):  # Skip <sos>
                    if target_ids[i] not in [token2id["<pad>"]]:
                        masked_ids[i] = self.mask_idx
                        mask_positions[i] = True
                        
        elif self.mask_strategy == "light":
            # Mask light chain
            if sep_idx > 0:
                for i in range(sep_idx + 1, len(target_ids)):
                    if target_ids[i] not in [token2id["<eos>"], token2id["<pad>"]]:
                        masked_ids[i] = self.mask_idx
                        mask_positions[i] = True
                        
        elif self.mask_strategy == "both":
            # Randomly choose to mask heavy or light
            choice = np.random.randint(0, 2)
            if choice == 0 and sep_idx > 0:
                # Mask heavy chain
                for i in range(1, sep_idx):
                    if target_ids[i] not in [token2id["<pad>"]]:
                        masked_ids[i] = self.mask_idx
                        mask_positions[i] = True
            elif sep_idx > 0:
                # Mask light chain
                for i in range(sep_idx + 1, len(target_ids)):
                    if target_ids[i] not in [token2id["<eos>"], token2id["<pad>"]]:
                        masked_ids[i] = self.mask_idx
                        mask_positions[i] = True
        
        return masked_ids, mask_positions.long()