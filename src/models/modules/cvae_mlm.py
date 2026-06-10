import math
import logging
from typing import List, Tuple, Optional
from omegaconf import DictConfig
import torch
from torch import Tensor
from pytorch_lightning import LightningModule
from tqdm import tqdm
import torch.nn as nn

# Import custom modules exactly as in your CVAE
from src.models.modules.components import LatentEncoder
from src.models.modules.components import KLDivergence
from src.models.modules.seqDecoder import TransformerDecoder
from src.models.modules.controller import PositionalPIController
from src.models.pcgrad import PCGrad
from src.utils.constants import convert_seqs2ids, get_token2id, convert_ids2seqs, VOCAB

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class MaskedLanguageCVAE(LightningModule):
    """
    Masked Conditional Variational Autoencoder for sequence completion.
    
    Combines the latent space modeling, KL annealing, and PCGrad optimization 
    of a CVAE with the masked token prediction objective of an MLM.
    """

    def __init__(self, cfg: DictConfig):
        super(MaskedLanguageCVAE, self).__init__()

        # 1. CVAE & MLM Configuration parameters
        self.use_pcgrad = cfg.model.use_pcgrad
        self.latent_dim = cfg.model.latent_dimen
        self.max_len = cfg.model.max_ab_seq_len
        self.use_concat_condition = cfg.model.use_concat_condition
        
        # Training hyperparameters for CVAE
        self.learning_rate = cfg.optimizer.lr
        self.nll_weight = cfg.model.nll_weight
        self.kl_weight = cfg.model.kl_weight
        self.expected_kl = cfg.model.expected_kl
        self.beta_min = cfg.model.beta_min
        self.beta_max = cfg.model.beta_max
        self.Kp = cfg.model.Kp
        self.Ki = cfg.model.Ki

        if self.use_pcgrad:
            self.automatic_optimization = False
            logger.info("Using PCGrad for multi-objective optimization in Masked CVAE")

        # 2. Setup Vocabulary & Tokens
        self.num_embeddings = len(VOCAB)
        token2id = get_token2id()
        self.mask_idx = token2id.get("<mask>", token2id["<pad>"])
        self.pad_idx = token2id["<pad>"]

        # 3. Initialize CVAE Controllers and Modules
        self.pi_controller = PositionalPIController(
            self.expected_kl, self.kl_weight, self.beta_min, self.beta_max, self.Kp, self.Ki
        )
        
        self.seqDecoder = TransformerDecoder(cfg)
        
        combined_dim = (
            cfg.model.ab_dimen + cfg.model.ag_dimen
            if self.use_concat_condition
            else cfg.model.ab_dimen
        )
        self.latent_encoder = LatentEncoder(self.latent_dim, combined_dim)

        # 4. Losses
        # Note: reduction is 'none' so we can apply the mask filter later
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.pad_idx, reduction='none')
        self.kl_div = KLDivergence(reduction=cfg.model.reduction)

        logger.info("Masked Language CVAE initialized successfully")

    def encode_precomputed_emb(self, ab_emb: Tensor, ag_emb: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Encode precomputed embeddings into latent space."""
        if self.use_concat_condition:
            combined_rep = torch.cat([ab_emb, ag_emb], dim=-1)
        else:
            combined_rep = ab_emb

        z, mu, logsigma = self.latent_encoder(combined_rep)
        return z, mu, logsigma

    def model_step(self, batch: dict, batch_idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass merging VAE latent extraction and masked decoding."""
        masked_seq_ids = batch['masked_seq_ids']  # Used as input condition for decoder
        target_seq_ids = batch['target_seq_ids']  # Ground truth
        
        ab_emb = batch['ab_embedding'] 
        ag_emb = batch['ag_embedding']
        des_emb = batch['des_embedding']
        mask_positions = batch.get('mask_positions', None)

        # 1. Encode context/masked input into latent representation
        z, mu, log_var = self.encode_precomputed_emb(ab_emb, ag_emb)

        # 2. Decode using latent vector and contextual embeddings
        # We pass masked_seq_ids so the decoder knows what tokens are available and what is missing
        recon_sequence_logits, _ = self.seqDecoder(z, des_emb, ag_emb, masked_seq_ids)

        # 3. Compute masked CVAE loss
        total_loss, recon_loss, kl_div = self.loss_function(
            recon_sequence_logits, target_seq_ids, mask_positions, mu, log_var
        )

        return total_loss, recon_loss, kl_div

    def loss_function(
            self, 
            recon_logits: Tensor, 
            target_seq: Tensor, 
            mask_positions: Optional[Tensor], 
            mu: Tensor, 
            log_var: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute VAE loss specifically filtered by masked positions."""
        
        # [batch, seq_len, vocab_size] -> [batch, vocab_size, seq_len] for CrossEntropy
        recon_logits = recon_logits.permute(0, 2, 1) 
        
        # Calculate cross-entropy token by token
        unreduced_recon_loss = self.ce_loss(recon_logits, target_seq) # Shape: [batch, seq_len]

        # Apply loss ONLY to masked positions if a mask is provided
        if mask_positions is not None:
            mask_reshaped = mask_positions.bool()
            if mask_reshaped.any():
                recon_loss = unreduced_recon_loss[mask_reshaped].mean()
            else:
                recon_loss = torch.tensor(0.0, device=recon_logits.device, requires_grad=True)
        else:
            # Fallback to computing across all valid tokens
            recon_loss = unreduced_recon_loss.mean()

        # Calculate KL Divergence
        kl_div = self.kl_div(mu, log_var)

        # Total Loss
        total_loss = self.nll_weight * recon_loss + self.kl_weight * kl_div

        # PI Controller Update (Train only)
        if self.training:
            self.kl_weight = self.pi_controller(kl_div.detach())

        return total_loss, recon_loss, kl_div

    def training_step(self, batch: dict, batch_idx: int) -> Optional[Tensor]:
        """Training step with PCGrad."""
        loss, recon_loss, kl_div = self.model_step(batch, batch_idx)

        self.log("train_loss", loss, on_epoch=True, sync_dist=True)
        self.log("train_recon_loss", recon_loss, on_epoch=True, sync_dist=True)
        self.log("train_kl_div", kl_div, on_epoch=True, sync_dist=True)
        self.log("kl_weight", self.kl_weight, on_epoch=True, sync_dist=True)

        if not self.use_pcgrad:
            return loss

        # PCGrad multi-objective optimization workflow
        opt = self.optimizers()
        opt.optimizer.zero_grad()

        losses = [recon_loss, kl_div]
        opt.optimizer.pc_backward(losses)

        import torch.distributed as dist
        if self.trainer.strategy.__class__.__name__ == "DDPStrategy" and dist.is_initialized():
            for param in self.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                    param.grad.data /= dist.get_world_size()

        opt.optimizer.step()
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> Tensor:
        """Validation step."""
        loss, recon_loss, kl_div = self.model_step(batch, batch_idx)

        self.log("val_loss", loss, on_epoch=True, sync_dist=True)
        self.log("val_recon_loss", recon_loss, on_epoch=True, sync_dist=True)
        self.log("val_kl_div", kl_div, on_epoch=True, sync_dist=True)
        
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
        """Inference wrapper to complete masks utilizing latent space."""
        with torch.inference_mode():
            # 1. Encode into latent distribution
            z, _, _ = self.encode_precomputed_emb(ab_emb, ag_emb)
            
            # 2. Decode conditioned on latents and context
            logits, _ = self.seqDecoder(z, des_emb, ag_emb, masked_seq_ids)

            if greedy:
                predicted_ids = torch.argmax(logits, dim=-1)
            else:
                logits_scaled = logits / temperature
                probs = torch.softmax(logits_scaled, dim=-1)
                predicted_ids = torch.multinomial(
                    probs.view(-1, self.num_embeddings), 1
                ).view(probs.shape[0], probs.shape[1])

            # 3. Replace only the <mask> indices in the sequence
            completed_ids = masked_seq_ids.clone()
            mask_positions = (masked_seq_ids == self.mask_idx)
            completed_ids[mask_positions] = predicted_ids[mask_positions]

            completed_seqs = convert_ids2seqs(completed_ids.cpu().tolist())

            return completed_ids, completed_seqs

    def configure_optimizers(self) -> dict:
        """Optimizer logic using PCGrad if configured."""
        opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        if self.use_pcgrad:
            pcgrad_optimizer = PCGrad(opt)
            return {"optimizer": pcgrad_optimizer}
        else:
            return opt