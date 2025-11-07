import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl


class BaselineMLPModel(pl.LightningModule):
    """
    Simple baseline model that concatenates antibody and antigen embeddings
    and passes them through an MLP to predict binding affinity.
    """

    def __init__(self, ab_embed_dim=1024, ag_embed_dim=960, hidden_dims=[512, 256, 128]):
        super().__init__()

        # Concatenated input dimension
        input_dim = ab_embed_dim + ag_embed_dim

        # Build MLP layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim

        # Final prediction layer
        layers.append(nn.Linear(prev_dim, 1))

        self.mlp = nn.Sequential(*layers)

        # Metrics
        self.rmse_metric = torchmetrics.MeanSquaredError(squared=False)
        self.pearson_metric = torchmetrics.PearsonCorrCoef(num_outputs=1)

        self.val_rmse = torchmetrics.MeanSquaredError(squared=False)
        self.val_pearson = torchmetrics.PearsonCorrCoef(num_outputs=1)

    def forward(self, ab_embed, ag_embed):
        # Concatenate antibody and antigen embeddings
        combined = torch.cat([ab_embed, ag_embed], dim=-1)

        # Pass through MLP
        output = self.mlp(combined)

        return output.squeeze(-1)  # (batch,)

    def training_step(self, batch, batch_idx):
        ab_embed, ag_embed, delta_g = batch
        delta_g = delta_g.float()

        pred_score = self(ab_embed, ag_embed)
        loss = F.mse_loss(pred_score, delta_g)

        # Update metrics
        self.rmse_metric.update(pred_score, delta_g)
        self.pearson_metric.update(pred_score, delta_g)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        ab_embed, ag_embed, delta_g = batch
        delta_g = delta_g.float()

        pred_score = self(ab_embed, ag_embed)
        loss = F.mse_loss(pred_score, delta_g)

        # Update validation metrics
        self.val_rmse.update(pred_score, delta_g)
        self.val_pearson.update(pred_score, delta_g)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        rmse = self.rmse_metric.compute()
        pearson_corr = self.pearson_metric.compute()

        self.rmse_metric.reset()
        self.pearson_metric.reset()

        self.log("train_rmse", rmse, prog_bar=True, on_epoch=True)
        self.log("train_pearson", pearson_corr, prog_bar=True, on_epoch=True)

    def on_validation_epoch_end(self):
        val_rmse = self.val_rmse.compute()
        val_pearson = self.val_pearson.compute()

        self.val_rmse.reset()
        self.val_pearson.reset()

        self.log("val_rmse", val_rmse, prog_bar=True, on_epoch=True)
        self.log("val_pearson", val_pearson, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class ElementwiseInteractionModel(pl.LightningModule):
    """
    Baseline model that captures element-wise interactions between antibody
    and antigen embeddings before passing through an MLP.

    Creates features: [emb1, emb2, emb1*emb2, |emb1-emb2|]
    """

    def __init__(self, ab_embed_dim=1024, ag_embed_dim=960, hidden_dims=[512, 256, 128]):
        super().__init__()

        # First project both embeddings to same dimension for element-wise operations
        common_dim = min(ab_embed_dim, ag_embed_dim)
        self.ab_proj = nn.Linear(ab_embed_dim, common_dim)
        self.ag_proj = nn.Linear(ag_embed_dim, common_dim)

        # Input dimension: original + projected + element-wise multiplication + absolute difference
        # = ab_embed_dim + ag_embed_dim + common_dim + common_dim
        input_dim = ab_embed_dim + ag_embed_dim + 2 * common_dim

        # Build MLP layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim

        # Final prediction layer
        layers.append(nn.Linear(prev_dim, 1))

        self.mlp = nn.Sequential(*layers)

        # Metrics
        self.rmse_metric = torchmetrics.MeanSquaredError(squared=False)
        self.pearson_metric = torchmetrics.PearsonCorrCoef(num_outputs=1)

        self.val_rmse = torchmetrics.MeanSquaredError(squared=False)
        self.val_pearson = torchmetrics.PearsonCorrCoef(num_outputs=1)

    def forward(self, ab_embed, ag_embed):
        # Project to common dimension for interaction
        ab_proj = self.ab_proj(ab_embed)
        ag_proj = self.ag_proj(ag_embed)

        # Compute element-wise interactions
        multiplication = ab_proj * ag_proj  # Element-wise multiplication
        abs_difference = torch.abs(ab_proj - ag_proj)  # Absolute difference

        # Concatenate all features: original embeddings + interactions
        combined = torch.cat([
            ab_embed,  # Original antibody embedding
            ag_embed,  # Original antigen embedding
            multiplication,  # Element-wise multiplication
            abs_difference  # Absolute difference
        ], dim=-1)

        # Pass through MLP
        output = self.mlp(combined)

        return output.squeeze(-1)  # (batch,)

    def training_step(self, batch, batch_idx):
        ab_embed, ag_embed, delta_g = batch
        delta_g = delta_g.float()

        pred_score = self(ab_embed, ag_embed)
        loss = F.mse_loss(pred_score, delta_g)

        # Update metrics
        self.rmse_metric.update(pred_score, delta_g)
        self.pearson_metric.update(pred_score, delta_g)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        ab_embed, ag_embed, delta_g = batch
        delta_g = delta_g.float()

        pred_score = self(ab_embed, ag_embed)
        loss = F.mse_loss(pred_score, delta_g)

        # Update validation metrics
        self.val_rmse.update(pred_score, delta_g)
        self.val_pearson.update(pred_score, delta_g)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        rmse = self.rmse_metric.compute()
        pearson_corr = self.pearson_metric.compute()

        self.rmse_metric.reset()
        self.pearson_metric.reset()

        self.log("train_rmse", rmse, prog_bar=True, on_epoch=True)
        self.log("train_pearson", pearson_corr, prog_bar=True, on_epoch=True)

    def on_validation_epoch_end(self):
        val_rmse = self.val_rmse.compute()
        val_pearson = self.val_pearson.compute()

        self.val_rmse.reset()
        self.val_pearson.reset()

        self.log("val_rmse", val_rmse, prog_bar=True, on_epoch=True)
        self.log("val_pearson", val_pearson, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
