import math
import logging
from typing import List, Tuple, Optional
from omegaconf import DictConfig
import torch
from torch import Tensor
from pytorch_lightning import LightningModule
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from src.models.modules import Upsampling
from src.models.modules.components import LatentEncoder
from src.models.modules.components import KLDivergence
from src.models.modules.seqDecoder import TransformerDecoder
from src.models.modules.controller import PositionalPIController
from src.models.pcgrad import PCGrad
from src.utils.constants import convert_seqs2ids, get_token2id, convert_ids2seqs
import torch.nn as nn


class CVAE(LightningModule):
    """
    Conditional Variational Autoencoder (CVAE) for protein sequence generation.

    This model encodes protein sequences (antibodies) along with their descriptions
    and antigens into a latent space, then reconstructs the sequences. It uses
    a variational approach with KL divergence regularization and supports
    multi-objective optimization with PCGrad.
    """

    def __init__(self, cfg: DictConfig):
        """
        Initialize the CVAE model.

        Args:
            cfg (DictConfig): Configuration object containing model hyperparameters
        """
        super(CVAE, self).__init__()

        # Store configuration parameters
        self.use_pcgrad = cfg.model.use_pcgrad
        self.latent_dim = cfg.model.latent_dimen  # Latent space dimensionality
        self.max_len = cfg.model.max_ab_seq_len  # Maximum antibody sequence length
        self.use_concat_condition = cfg.model.use_concat_condition  # Whether to concatenate all conditions
        self.use_description = bool(cfg.model.get("use_description", 1))

        # Calculate low resolution dimension for upsampling
        self.low_res_dim = math.ceil(
            self.max_len / cfg.model.upsampler.stride ** cfg.model.upsampler.num_deconv_layers
        )
        new_max_len = self.low_res_dim * cfg.model.upsampler.stride ** cfg.model.upsampler.num_deconv_layers

        logger.info(f"Adjusting max_len from {self.max_len} to {new_max_len} for upsampling compatibility")
        assert new_max_len >= self.max_len, "New max_len must be higher than original max_len"
        self.max_len = new_max_len

        # Configure optimization strategy
        if self.use_pcgrad:
            # Manual optimization for multi-objective learning with PCGrad
            self.automatic_optimization = False
            logger.info("Using PCGrad for multi-objective optimization")

        # Training hyperparameters
        self.learning_rate = cfg.optimizer.lr
        self.nll_weight = cfg.model.nll_weight  # Weight for reconstruction loss
        self.kl_weight = cfg.model.kl_weight  # Weight for KL divergence loss
        self.expected_kl = cfg.model.expected_kl  # Target KL divergence value
        self.beta_min = cfg.model.beta_min  # Minimum beta for KL annealing
        self.beta_max = cfg.model.beta_max  # Maximum beta for KL annealing
        self.Kp = cfg.model.Kp  # Proportional gain for PI controller
        self.Ki = cfg.model.Ki  # Integral gain for PI controller

        # Initialize PI controller for KL weight adjustment
        self.pi_controller = PositionalPIController(
            self.expected_kl, self.kl_weight, self.beta_min, self.beta_max, self.Kp, self.Ki
        )
        logger.info(f"PI Controller initialized with expected_kl={self.expected_kl}")

        # Initialize sequence decoder
        self.seqDecoder = TransformerDecoder(cfg)
        logger.info("Using Transformer decoder")

        # Check if attention mechanism is used
        self.use_attn = False  # cfg.model.decoder.attn_method is not None

        # Initialize upsampler if attention is used
        if self.use_attn:
            self.upsampler = Upsampling(
                self.latent_dim,
                cfg.model.upsampler.max_filter,
                self.low_res_dim,
                cfg.model.upsampler.min_deconv_dim,
                cfg.model.upsampler.num_deconv_layers,
                cfg.model.upsampler.kernel_size,
                cfg.model.upsampler.stride,
                cfg.model.upsampler.padding,
                cfg.model.upsampler.dropout,
                cfg.model.upsampler.act_type,
                cfg.device
            )
            logger.info("Upsampler initialized for attention mechanism")

        # Initialize loss functions
        token2id = get_token2id()
        self.mse_loss = nn.MSELoss(reduction=cfg.model.reduction)
        self.kl_div = KLDivergence(reduction=cfg.model.reduction)
        self.recon_loss = nn.CrossEntropyLoss(
            ignore_index=token2id["<pad>"],
            reduction=cfg.model.reduction
        )

        # Calculate combined dimension for latent encoder input
        combined_dim = (
            cfg.model.ab_dimen + cfg.model.ag_dimen
            if self.use_concat_condition
            else cfg.model.ab_dimen
        )
        if self.use_description:
            combined_dim += cfg.model.des_dimen

        # Initialize latent encoder
        self.latent_encoder = LatentEncoder(self.latent_dim, combined_dim)
        logger.info(f"Latent encoder initialized with combined_dim={combined_dim}")

    def encode_precomputed_emb(
            self,
            ab_emb: Tensor,
            des_emb: Tensor,
            ag_emb: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Encode precomputed embeddings into latent space.

        Args:
            ab_emb: Antibody embeddings [batch, ab_seq_len, ab_dimen]
            des_emb: Description embeddings [batch, des_seq_len, des_dimen]
            ag_emb: Antigen embeddings [batch, ag_seq_len, ag_dimen]

        Returns:
            z: Latent representation [batch, latent_dim]
            mu: Mean of latent distribution [batch, latent_dim]
            logsigma: Log variance of latent distribution [batch, latent_dim]

        """
        z_ab_rep = ab_emb
        if self.use_concat_condition:
            combined_rep = torch.cat([z_ab_rep, ag_emb], dim=-1)
            logger.debug(f"Combined representation shape: {combined_rep.shape}")
        else:
            combined_rep = z_ab_rep  # [batch, ab_dimen]
        if self.use_description:
            combined_rep = torch.cat([combined_rep, des_emb], dim=-1)

        # Encode to latent space
        z, mu, logsigma = self.latent_encoder(combined_rep)
        logger.debug(f"Latent encoding - z: {z.shape}, mu: {mu.shape}, logsigma: {logsigma.shape}")

        return z, mu, logsigma

    def model_step(self, batch: dict, batch_idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass for training/validation.

        Args:
            batch: Dictionary containing batch data
            batch_idx: Batch index

        Returns:
            total_loss: Combined reconstruction and KL loss
            recon_loss: Reconstruction loss
            kl_div: KL divergence loss
        """
        # Extract embeddings from batch
        ab_emb = batch['ab_embedding']  # [batch, ab_seq_len, ab_dimen]
        des_emb = batch['des_embedding']  # [batch, des_seq_len, des_dimen]
        ag_emb = batch['ag_embedding']  # [batch, ag_seq_len, ag_dimen]
        ab_sequences = batch['ab_sequences']  # List of sequence strings

        # Convert sequences to token IDs
        orig_seq_ids = torch.tensor(
            convert_seqs2ids(ab_sequences, add_sos=True, add_eos=True, max_length=self.max_len),
            dtype=torch.long,
            device=self.device
        )  # [batch, max_len]

        logger.debug(f"Original sequence IDs shape: {orig_seq_ids.shape}")

        # Encode to latent space
        z, mu, log_var = self.encode_precomputed_emb(ab_emb, des_emb, ag_emb)

        # Decode from latent space
        recon_sequence_ids, recon_seqs = self.seqDecoder(z, des_emb, ag_emb, orig_seq_ids)

        if batch_idx % 50 == 0 and batch_idx > 0:
            seqs = convert_ids2seqs(recon_seqs.tolist())
            gt_seqs = convert_ids2seqs(orig_seq_ids.tolist())
            logger.info(f"Grouth truth: {gt_seqs} - result: {seqs}")

        # Compute loss
        total_loss, recon_loss, kl_div = self.loss_function(recon_sequence_ids, orig_seq_ids, mu, log_var)

        logger.debug(f"Batch {batch_idx} - Total Loss: {total_loss:.4f}, "
                     f"Recon Loss: {recon_loss:.4f}, KL Div: {kl_div:.4f}")

        return total_loss, recon_loss, kl_div

    def training_step(self, batch: dict, batch_idx: int) -> Optional[Tensor]:
        """
        Training step with optional PCGrad optimization.

        Args:
            batch: Training batch data
            batch_idx: Batch index

        Returns:
            Loss tensor if using automatic optimization, None if using PCGrad
        """
        loss, recon_loss, kl_div = self.model_step(batch, batch_idx)

        # Log training metrics
        self.log("train_loss", loss, on_epoch=True, sync_dist=True)
        self.log("train_recon_loss", recon_loss, on_epoch=True, sync_dist=True)
        self.log("train_kl_div", kl_div, on_epoch=True, sync_dist=True)
        self.log("kl_weight", self.kl_weight, on_epoch=True, sync_dist=True)

        if not self.use_pcgrad:
            return loss

        # Manual optimization with PCGrad
        opt = self.optimizers()
        opt.optimizer.zero_grad()

        # PCGrad requires separate loss terms
        losses = [recon_loss, kl_div]
        opt.optimizer.pc_backward(losses)

        # Handle distributed training
        import torch.distributed as dist
        if self.trainer.strategy.__class__.__name__ == "DDPStrategy" and dist.is_initialized():
            for param in self.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                    param.grad.data /= dist.get_world_size()

        opt.optimizer.step()

        # Add tqdm progress for training step loss printing
        tqdm.write(f"Training Batch {batch_idx} - Loss: {loss:.4f}, Recon Loss: {recon_loss:.4f}, KL Div: {kl_div:.4f}")
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
        loss, recon_loss, kl_div = self.model_step(batch, batch_idx)

        # Log validation metrics
        self.log("val_loss", loss, on_epoch=True, sync_dist=True)
        self.log("val_recon_loss", recon_loss, on_epoch=True, sync_dist=True)
        self.log("val_kl_div", kl_div, on_epoch=True, sync_dist=True)

        tqdm.write(
            f"Validation Batch {batch_idx} - Loss: {loss:.4f}, Recon Loss: {recon_loss:.4f}, KL Div: {kl_div:.4f}")

        return loss

    def loss_function(
            self,
            recon_seq: Tensor,
            orig_seq: Tensor,
            mu: Tensor,
            log_var: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute VAE loss function.

        Args:
            recon_seq: Reconstructed sequence logits [batch, seq_len, vocab_size]
            orig_seq: Original sequence token IDs [batch, seq_len]
            mu: Latent mean [batch, latent_dim]
            log_var: Latent log variance [batch, latent_dim]

        Returns:
            total_loss: Weighted combination of reconstruction and KL loss
            recon_loss: Reconstruction loss (cross-entropy)
            kl_div: KL divergence loss
        """
        # Compute reconstruction loss
        recon_seq = recon_seq.permute(0, 2, 1)  # [batch, num_embeddings, gen_len]
        recon_loss = self.recon_loss(recon_seq, orig_seq)
        # Compute KL divergence loss
        kl_div = self.kl_div(mu, log_var)

        # Combine losses
        total_loss = self.nll_weight * recon_loss + self.kl_weight * kl_div

        # Update KL weight using PI controller during training
        if self.training:
            old_kl_weight = self.kl_weight
            self.kl_weight = self.pi_controller(kl_div.detach())
            logger.debug(f"KL weight updated from {old_kl_weight:.4f} to {self.kl_weight:.4f}")

        return total_loss, recon_loss, kl_div

    def configure_optimizers(self) -> dict:
        """
        Configure optimizers for training.

        Returns:
            Optimizer configuration
        """
        opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        if self.use_pcgrad:
            pcgrad_optimizer = PCGrad(opt)
            logger.info("Configured PCGrad optimizer")
            return {"optimizer": pcgrad_optimizer}
        else:
            logger.info("Configured Adam optimizer")
            return opt

    def generate_new_antibody(self, y_des: torch.Tensor, y_ag: torch.Tensor) -> list[str]:
        """
        Generate an antibody sequence from description and antigen embeddings, using random latent vector.

        Args:
            y_des: Description embeddings [batch, des_seq_len, des_dimen]
            y_ag: Antigen embeddings [batch, ag_seq_len, ag_dimen]

        Returns:
            List of generated antibody sequences
        """
        with torch.inference_mode():
            # Encode with description and antigen embeddings, generating random latent
            z = torch.randn(1, self.latent_dim, device=y_des.device)

            # Decode with latent and condition representations
            logits, pred_tokens = self.seqDecoder(z, y_des, y_ag)
            # Convert token IDs to sequences
            seqs = convert_ids2seqs(pred_tokens.tolist())

            logger.debug(f"Generated {len(seqs)} antibody sequences")

            return seqs

    def reconstruct_from_wt(
            self,
            y_ab: Tensor,
            y_ag: Tensor,
            y_des: Tensor,
            noise_scale: float = 0.2,
    ) -> List[str]:
        """
        Reconstruct sequences from wild-type input using antibody, antigen, and description embeddings.

        Args:
            y_ab: Antibody embeddings [batch, ab_seq_len, ab_dimen]
            y_ag: Antigen embeddings [batch, ag_seq_len, ag_dimen]
            y_des: Description embeddings [batch, des_seq_len, des_dimen]
            noise_scale: Standard deviation of noise to add to reconstruction loss

        Returns:
            List of reconstructed sequences
        """

        with torch.inference_mode():
            # Encode with description and antigen embeddings, generating random latent
            y_des = y_des.expand(y_ab.size(0), -1)
            y_ag = y_ag.expand(y_ab.size(0), -1)
            z, mu, log_var = self.encode_precomputed_emb(y_ab, y_des, y_ag)

            # Extra noise around posterior mean (fine-grained control)
            eps = torch.randn_like(mu) * noise_scale
            z = mu + eps * torch.exp(0.5 * log_var)

            # Decode with latent and condition representations
            logits, pred_tokens = self.seqDecoder(z, y_des, y_ag)
            # Convert token IDs to sequences
            seqs = convert_ids2seqs(pred_tokens.tolist())

            logger.debug(f"Generated {len(seqs)} antibody sequences")

            return seqs
