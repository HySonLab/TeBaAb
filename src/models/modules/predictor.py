import torch.nn.functional as F
import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl


# Model Definition
class ContrastiveModel(pl.LightningModule):
    def __init__(self, ab_embed_dim=256, ag_embed_dim=256, embed_dim=256, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.ab_encoder = nn.Linear(ab_embed_dim, embed_dim)
        self.ag_encoder = nn.Linear(ag_embed_dim, embed_dim)

    def forward(self, ab_embed, ag_embed):
        ab_proj = F.normalize(self.ab_encoder(ab_embed), dim=-1)
        ag_proj = F.normalize(self.ag_encoder(ag_embed), dim=-1)
        return ab_proj, ag_proj

    def contrastive_loss(self, ab_proj, ag_proj):
        logits = (ab_proj @ ag_proj.T) / self.temperature
        labels = torch.arange(logits.shape[0], device=self.device)
        return F.cross_entropy(logits, labels)

    def training_step(self, batch, batch_idx):
        ab_embed, ag_embed = batch
        ab_proj, ag_proj = self(ab_embed, ag_embed)
        loss = self.contrastive_loss(ab_proj, ag_proj)
        self.log("train_loss", loss)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


# New Model with Cross-Attention
class CrossAttentionModel(pl.LightningModule):
    def __init__(self, contrastive_model, proj_embed_dim=256, attn_embed_dim=256):
        super().__init__()
        self.ab_encoder = contrastive_model.ab_encoder  # Use encoder from ContrastiveModel
        self.ag_encoder = contrastive_model.ag_encoder

        self.cross_attention = nn.MultiheadAttention(proj_embed_dim, num_heads=8, batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(attn_embed_dim, attn_embed_dim // 2),
            nn.ReLU(),
            nn.Linear(attn_embed_dim // 2, 1)
        )
        self.rmse_metric = torchmetrics.MeanSquaredError(squared=False)  # RMSE metric
        self.pearson_metric = torchmetrics.PearsonCorrCoef(num_outputs=1)

        self.val_rmse = torchmetrics.MeanSquaredError(squared=False)  # RMSE metric
        self.val_pearson = torchmetrics.PearsonCorrCoef(num_outputs=1)

    def forward(self, ab_embed, ag_embed):
        proj_ab_embed = self.ab_encoder(ab_embed)
        proj_ag_embed = self.ag_encoder(ag_embed)

        # Reshaping for attention mechanism
        proj_ab_embed = proj_ab_embed.unsqueeze(1)  # (batch, 1, embed_dim)
        proj_ag_embed = proj_ag_embed.unsqueeze(1)  # (batch, 1, embed_dim)

        # Cross Attention
        attn_output, _ = self.cross_attention(proj_ab_embed, proj_ag_embed, proj_ag_embed)
        attn_output = attn_output.squeeze(1)  # (batch, embed_dim)

        attn_output2, _ = self.cross_attention(proj_ag_embed, proj_ab_embed, proj_ab_embed)
        attn_output2 = attn_output2.squeeze(1)  # (batch, embed_dim)

        # MLP to get a float value
        output = self.mlp(attn_output + attn_output2)

        return output.squeeze(-1)  # (batch,)

    def training_step(self, batch, batch_idx):
        ab_embed, ag_embed, delta_g = batch
        delta_g = delta_g.float()
        pred_score = self(ab_embed, ag_embed)
        loss = F.mse_loss(pred_score, delta_g)

        self.rmse_metric.update(pred_score, delta_g)  # Update RMSE metric
        self.pearson_metric.update(pred_score, delta_g)  # Update RMSE metric

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

        # Reset metrics for the next epoch
        self.rmse_metric.reset()
        self.pearson_metric.reset()

        # Print and log results
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
