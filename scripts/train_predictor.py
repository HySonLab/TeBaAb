import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pytorch_lightning as pl
import pickle
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset
from pytorch_lightning.callbacks import EarlyStopping
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

from src.models.modules.predictor import ContrastiveModel
from src.models.modules.predictor import CrossAttentionModel


class Config:
    antibody_embedding_path = "datasets/affinity/train_antibody.pkl"
    antigen_embedding_path = "datasets/affinity/train_antigen.pkl"
    sabdab_pair_file_path = 'datasets/affinity/train.csv'

    contrastive_model_path = "checkpoints/affinity_predictor/ab_ag_clip.pth"
    cross_attention_model_path = "checkpoints/affinity_predictor/cross_attention_model.pth"
    predictions_csv_path = "checkpoints/affinity_predictor/predictions.csv"

    antigen_embedding_dim = 960
    antibody_embedding_dim = 1024
    projected_embedding_dim = 256
    cross_attention_emb_dim = projected_embedding_dim
    device = "cuda" if torch.cuda.is_available() else "cpu"


def load_embeddings(file_path):
    """Load embeddings from a pickle file and return the embeddings dictionary."""
    with open(file_path, "rb") as f:
        embeddings = pickle.load(f)

    return embeddings


def load_sequences(file_path):
    """Load sequences from a CSV file and return them as a dictionary."""
    df = pd.read_csv(file_path)
    return {
        "heavy_chain": df['heavy_chain'].values,
        "light_chain": df['light_chain'].values,
        "antigen": df['antigen'].values,
        "delta_g": df['delta_g'].values
    }


class CLIPDataset(Dataset):
    def __init__(self, sequences, antibody_embeddings, antigen_embeddings):
        self.antibody_embeddings = antibody_embeddings
        self.antigen_embeddings = antigen_embeddings
        self.pairs = []

        for h, l, ag in tqdm(zip(sequences["heavy_chain"], sequences["light_chain"], sequences["antigen"]),
                             desc="Loading dataset"):
            if (h, l) in antibody_embeddings and ag in antigen_embeddings:
                self.pairs.append(((h, l), ag))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        (h, l), ag = self.pairs[idx]
        ab_embed = self.antibody_embeddings[(h, l)].clone().detach()
        ag_embed = self.antigen_embeddings[ag].clone().detach()
        return ab_embed, ag_embed


# Contrastive Dataset
class ContrastiveDataset(Dataset):
    def __init__(self, sequences, antibody_embeddings, antigen_embeddings):
        self.antibody_embeddings = antibody_embeddings
        self.antigen_embeddings = antigen_embeddings
        self.pairs = []

        for h, l, ag, delta_g in tqdm(
                zip(sequences["heavy_chain"], sequences["light_chain"], sequences["antigen"], sequences["delta_g"]),
                desc="Loading dataset"):
            if (h, l) in antibody_embeddings and ag in antigen_embeddings:
                self.pairs.append((((h, l), ag), delta_g))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ((h, l), ag), delta_g = self.pairs[idx]
        ab_embed = self.antibody_embeddings[(h, l)].clone().detach()
        ag_embed = self.antigen_embeddings[ag].clone().detach()
        delta_g = float(delta_g)
        return ab_embed, ag_embed, delta_g


def train_contrastive_model():
    config = Config()
    # Load data
    antigen_embeddings = load_embeddings(config.antigen_embedding_path)
    antibody_embeddings = load_embeddings(config.antibody_embedding_path)
    sequences = load_sequences(config.sabdab_pair_file_path)

    dataset = CLIPDataset(sequences, antibody_embeddings, antigen_embeddings)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Train the model
    model = ContrastiveModel(ab_embed_dim=config.antibody_embedding_dim, ag_embed_dim=config.antigen_embedding_dim,
                             embed_dim=config.projected_embedding_dim)
    trainer = pl.Trainer(max_epochs=100, accelerator="gpu" if torch.cuda.is_available() else "cpu", log_every_n_steps=1)
    trainer.fit(model, dataloader)

    # Save model for each fold
    torch.save(model.state_dict(), config.contrastive_model_path)


def train_cross_attention_model():
    config = Config()
    antigen_embeddings = load_embeddings(config.antigen_embedding_path)
    antibody_embeddings = load_embeddings(config.antibody_embedding_path)
    sequences = load_sequences(config.sabdab_pair_file_path)

    dataset = ContrastiveDataset(sequences, antibody_embeddings, antigen_embeddings)

    # ✅ Split once into train and validation sets
    train_idx, val_idx = train_test_split(
        list(range(len(dataset))), test_size=0.1, random_state=42, shuffle=True
    )

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    # Create dataloaders
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

    # Instantiate model
    proj_model = ContrastiveModel(
        ab_embed_dim=config.antibody_embedding_dim,
        ag_embed_dim=config.antigen_embedding_dim,
        embed_dim=config.projected_embedding_dim
    )
    proj_model.load_state_dict(torch.load(config.contrastive_model_path))
    proj_model.eval()

    cross_attention_model = CrossAttentionModel(
        proj_model,
        proj_embed_dim=config.projected_embedding_dim,
        attn_embed_dim=config.cross_attention_emb_dim
    )

    # Early stopping
    early_stop_callback = EarlyStopping(
        monitor="val_loss", patience=10, mode="min", verbose=True
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        log_every_n_steps=10,
        callbacks=[early_stop_callback]
    )

    # Train model
    trainer.fit(cross_attention_model, train_loader, val_loader)

    # ✅ Save model (one checkpoint only)
    model_path = config.cross_attention_model_path
    torch.save(cross_attention_model.state_dict(), model_path)
    print(f"Model saved at {model_path}")

    # Predictions
    cross_attention_model.eval()
    predictions = []
    true_values = []
    predicted_values = []

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            ab_embed, ag_embed, delta_g = batch
            ab_embed = ab_embed.to(config.device)
            ag_embed = ag_embed.to(config.device)
            delta_g = delta_g.to(config.device)
            cross_attention_model = cross_attention_model.to(config.device)

            outputs = cross_attention_model(ab_embed, ag_embed).squeeze()
            predicted_values.extend(outputs.cpu().numpy())
            true_values.extend(delta_g.cpu().numpy())

            for j in range(len(delta_g)):
                predictions.append({
                    "heavy_chain": sequences["heavy_chain"][val_idx[i * val_loader.batch_size + j]],
                    "light_chain": sequences["light_chain"][val_idx[i * val_loader.batch_size + j]],
                    "antigen": sequences["antigen"][val_idx[i * val_loader.batch_size + j]],
                    "delta_g": sequences["delta_g"][val_idx[i * val_loader.batch_size + j]],
                    "predicted_output": outputs[j].item()
                })

    # ✅ Save predictions
    predictions_df = pd.DataFrame(predictions)
    predictions_csv_path = config.predictions_csv_path
    predictions_df.to_csv(predictions_csv_path, index=False)
    print(f"Predictions saved at {predictions_csv_path}")

    # Metrics
    rmse = mean_squared_error(true_values, predicted_values)
    pearson_corr, _ = pearsonr(true_values, predicted_values)

    print(f"\nValidation RMSE: {rmse:.4f}")
    print(f"Validation Pearson Correlation: {pearson_corr:.4f}")


if __name__ == "__main__":
    train_contrastive_model()
    train_cross_attention_model()
