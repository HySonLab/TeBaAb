import math
import logging
from typing import List, Tuple, Optional, Dict
from omegaconf import DictConfig
import torch
from torch import Tensor
import torch.nn as nn
from pytorch_lightning import LightningModule

from src.utils.constants import convert_seqs2ids, get_token2id, convert_ids2seqs, VOCAB

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def sinusoidal_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """Generate sinusoidal positional encoding."""
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # [1, max_len, d_model]


class MaskedLanguageModel(LightningModule):
    """
    Masked Language Model for antibody sequence completion.
    
    Given an antibody sequence with masked tokens and context (antigen, description embeddings),
    predicts the tokens at masked positions.
    """

    def __init__(self, cfg: DictConfig):
        """
        Initialize the Masked Language Model.
        
        Args:
            cfg (DictConfig): Configuration object
        """
        super(MaskedLanguageModel, self).__init__()

        # Store configuration
        self.max_len = cfg.model.max_ab_seq_len
        self.latent_dim = cfg.model.latent_dimen
        self.hidden_dim = cfg.model.decoder.hidden_dim
        self.num_layers = cfg.model.decoder.num_layers
        self.ab_dimen = cfg.model.ab_dimen
        self.ag_dimen = cfg.model.ag_dimen
        self.des_dimen = cfg.model.des_dimen
        self.nhead = 8
        self.learning_rate = cfg.optimizer.lr

        # Vocabulary
        self.num_embeddings = len(VOCAB)
        token2id = get_token2id()
        self.mask_idx = token2id.get("<mask>", token2id["<pad>"])
        self.pad_idx = token2id["<pad>"]
        self.sos_idx = token2id["<sos>"]
        self.eos_idx = token2id["<eos>"]

        # Token embedding
        self.token_embedding = nn.Embedding(self.num_embeddings, self.hidden_dim)
        self.positional_encoding = sinusoidal_positional_encoding(self.max_len, self.hidden_dim)

        # Context embedding projection: antibody embedding -> hidden_dim
        self.ab_context_projection = nn.Linear(self.ab_dimen, self.hidden_dim)
        
        # Optional: antigen and description context
        self.ag_context_projection = nn.Linear(self.ag_dimen, self.hidden_dim)
        self.des_context_projection = nn.Linear(self.des_dimen, self.hidden_dim)

        # Transformer encoder for contextual understanding
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.nhead,
            dim_feedforward=self.hidden_dim * 4,
            dropout=cfg.model.decoder.inp_dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # Output projection for token prediction
        self.output_projection = nn.Linear(self.hidden_dim, self.num_embeddings)

        # Loss function
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.pad_idx)

        logger.info("Masked Language Model initialized successfully")

    def forward(
            self,
            masked_seq_ids: Tensor,
            ab_emb: Tensor,
            ag_emb: Tensor,
            des_emb: Tensor,
            mask_positions: Tensor = None
    ) -> Tensor:
        """
        Forward pass for masked language model.
        
        Args:
            masked_seq_ids: Sequence with masked tokens [batch, seq_len]
            ab_emb: Antibody embeddings [batch, ab_dimen]
            ag_emb: Antigen embeddings [batch, ag_dimen]
            des_emb: Description embeddings [batch, des_dimen]
            mask_positions: Optional tensor indicating masked positions [batch, seq_len]
            
        Returns:
            logits: Predictions for all positions [batch, seq_len, vocab_size]
        """
        batch_size = masked_seq_ids.shape[0]
        seq_len = masked_seq_ids.shape[1]
        device = masked_seq_ids.device

        # Embed the masked sequence
        seq_emb = self.token_embedding(masked_seq_ids)  # [batch, seq_len, hidden_dim]
        
        # Add positional encoding
        pe = self.positional_encoding[:, :seq_len, :].to(device)
        seq_emb = seq_emb + pe

        # Project context embeddings
        ab_context = self.ab_context_projection(ab_emb).unsqueeze(1)  # [batch, 1, hidden_dim]
        ag_context = self.ag_context_projection(ag_emb).unsqueeze(1)  # [batch, 1, hidden_dim]
        des_context = self.des_context_projection(des_emb).unsqueeze(1)  # [batch, 1, hidden_dim]

        # Combine context
        combined_context = ab_context + ag_context + des_context  # [batch, 1, hidden_dim]

        # Concatenate context with sequence
        full_input = torch.cat([combined_context, seq_emb], dim=1)  # [batch, 1+seq_len, hidden_dim]

        # Apply transformer encoder
        encoded = self.transformer_encoder(full_input)  # [batch, 1+seq_len, hidden_dim]

        # Remove context, keep only sequence predictions
        seq_output = encoded[:, 1:, :]  # [batch, seq_len, hidden_dim]

        # Project to vocabulary size
        logits = self.output_projection(seq_output)  # [batch, seq_len, vocab_size]

        return logits

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        """
        Training step.
        
        Args:
            batch: Dictionary containing batch data
            batch_idx: Batch index
            
        Returns:
            Loss tensor
        """
        masked_seq_ids = batch['masked_seq_ids']  # [batch, seq_len]
        ab_emb = batch['ab_embedding']
        ag_emb = batch['ag_embedding']
        des_emb = batch['des_embedding']
        target_seq_ids = batch['target_seq_ids']  # [batch, seq_len]
        mask_positions = batch.get('mask_positions', None)  # [batch, seq_len]

        # Forward pass
        logits = self.forward(masked_seq_ids, ab_emb, ag_emb, des_emb, mask_positions)

        # Compute loss only on masked positions
        if mask_positions is not None:
            loss = self._compute_masked_loss(logits, target_seq_ids, mask_positions)
        else:
            # Compute loss on all positions
            logits_reshaped = logits.view(-1, self.num_embeddings)
            targets_reshaped = target_seq_ids.view(-1)
            loss = self.ce_loss(logits_reshaped, targets_reshaped)

        # Log metrics
        self.log("train_loss", loss, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> Tensor:
        """
        Validation step.
        
        Args:
            batch: Validation batch data
            batch_idx: Batch index
            
        Returns:
            Validation loss
        """
        masked_seq_ids = batch['masked_seq_ids']
        ab_emb = batch['ab_embedding']
        ag_emb = batch['ag_embedding']
        des_emb = batch['des_embedding']
        target_seq_ids = batch['target_seq_ids']
        mask_positions = batch.get('mask_positions', None)

        # Forward pass
        logits = self.forward(masked_seq_ids, ab_emb, ag_emb, des_emb, mask_positions)

        # Compute loss
        if mask_positions is not None:
            loss = self._compute_masked_loss(logits, target_seq_ids, mask_positions)
        else:
            logits_reshaped = logits.view(-1, self.num_embeddings)
            targets_reshaped = target_seq_ids.view(-1)
            loss = self.ce_loss(logits_reshaped, targets_reshaped)

        # Log metrics
        self.log("val_loss", loss, on_epoch=True, sync_dist=True)

        return loss

    def _compute_masked_loss(self, logits: Tensor, target_ids: Tensor, mask_positions: Tensor) -> Tensor:
        """
        Compute loss only on masked positions.
        
        Args:
            logits: [batch, seq_len, vocab_size]
            target_ids: [batch, seq_len]
            mask_positions: [batch, seq_len] binary mask
            
        Returns:
            Scalar loss
        """
        # Reshape for loss computation
        batch_size, seq_len, vocab_size = logits.shape
        logits_reshaped = logits.view(-1, vocab_size)
        targets_reshaped = target_ids.view(-1)
        mask_reshaped = mask_positions.view(-1).bool()

        # Compute loss only on masked positions
        masked_logits = logits_reshaped[mask_reshaped]
        masked_targets = targets_reshaped[mask_reshaped]

        if masked_targets.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        loss = self.ce_loss(masked_logits, masked_targets)
        return loss

    def predict_masked_positions(
            self,
            masked_seq_ids: Tensor,
            ab_emb: Tensor,
            ag_emb: Tensor,
            des_emb: Tensor,
            temperature: float = 1.0,
            greedy: bool = True
    ) -> Tuple[Tensor, List[str]]:
        """
        Predict tokens at masked positions.
        
        Args:
            masked_seq_ids: Sequence with masked tokens [batch, seq_len]
            ab_emb: Antibody embeddings [batch, ab_dimen]
            ag_emb: Antigen embeddings [batch, ag_dimen]
            des_emb: Description embeddings [batch, des_dimen]
            temperature: Temperature for sampling
            greedy: If True, use greedy decoding; else sample
            
        Returns:
            completed_ids: Completed sequence IDs [batch, seq_len]
            completed_seqs: List of completed sequences as strings
        """
        with torch.inference_mode():
            # Get predictions
            logits = self.forward(masked_seq_ids, ab_emb, ag_emb, des_emb)  # [batch, seq_len, vocab_size]

            if greedy:
                # Greedy decoding: take argmax
                predicted_ids = torch.argmax(logits, dim=-1)  # [batch, seq_len]
            else:
                # Sampling with temperature
                logits_scaled = logits / temperature
                probs = torch.softmax(logits_scaled, dim=-1)
                predicted_ids = torch.multinomial(
                    probs.view(-1, self.num_embeddings),
                    num_samples=1
                ).view(probs.shape[0], probs.shape[1])  # [batch, seq_len]

            # Replace masked positions in original sequence with predictions
            completed_ids = masked_seq_ids.clone()
            token2id = get_token2id()
            mask_token_id = token2id.get("<mask>", token2id["<pad>"])
            
            mask_positions = (masked_seq_ids == mask_token_id)
            completed_ids[mask_positions] = predicted_ids[mask_positions]

            # Convert to sequences
            completed_seqs = convert_ids2seqs(completed_ids.cpu().tolist())

            return completed_ids, completed_seqs

    def configure_optimizers(self):
        """Configure optimizer."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
