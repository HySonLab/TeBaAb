import math
from omegaconf import DictConfig
import torch
import torch.nn as nn

from src.utils.constants import VOCAB, get_token2id


def sinusoidal_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # [1, max_len, d_model]


class TransformerDecoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(TransformerDecoder, self).__init__()
        self.latent_dim = cfg.model.latent_dimen
        self.des_dimen = cfg.model.des_dimen
        self.ag_dimen = cfg.model.ag_dimen
        self.hidden_dim = cfg.model.decoder.hidden_dim
        self.num_layers = cfg.model.decoder.num_layers
        self.num_embeddings = len(VOCAB)
        self.input_dropout = cfg.model.decoder.inp_dropout
        self.nhead = 8

        # Compute sos_idx and padding_idx automatically
        token2id = get_token2id()
        self.sos_idx = token2id["<sos>"]
        self.padding_idx = token2id["<pad>"]

        self.max_len = cfg.model.max_ab_seq_len
        self.low_res_dim = math.ceil(self.max_len / cfg.model.upsampler.stride ** cfg.model.upsampler.num_deconv_layers)
        self.max_len = self.low_res_dim * cfg.model.upsampler.stride ** cfg.model.upsampler.num_deconv_layers
        self.use_teacher_forcing = cfg.model.decoder.use_teacher_forcing

        # Combined dimension for latent and conditions
        self.combined_dim = self.latent_dim + self.ag_dimen

        # Project combined latent and conditions to sequence hidden states
        self.input_projection = nn.Linear(self.combined_dim, self.max_len * self.hidden_dim)

        # Embedding for target tokens
        self.token_embedding = nn.Embedding(self.num_embeddings, self.hidden_dim)
        self.positional_encoding = sinusoidal_positional_encoding(self.max_len, self.hidden_dim)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=self.nhead,
            dropout=self.input_dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=self.num_layers
        )

        # Final output projection
        self.output_projection = nn.Linear(self.hidden_dim, self.num_embeddings)

    def forward(
            self,
            latent: torch.Tensor,
            des_rep: torch.Tensor,
            ag_rep: torch.Tensor,
            target_ids: torch.LongTensor = None,
            teacher_forcing_ratio: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Runs decoding with concatenated latent, description, and antigen representations.
        If `target_ids` is provided, uses teacher forcing based on ratio. Otherwise, uses greedy decoding.

        Args:
            latent: [batch, latent_dim]
            des_rep: [batch, des_dimen]
            ag_rep: [batch, ag_dimen]
            target_ids: [batch, seq_len]
            teacher_forcing_ratio: Probability to use ground-truth token

        Returns:
            logits: [batch, gen_len, num_embeddings]
            generated: [batch, gen_len]
        """
        teacher_forcing_ratio = self.use_teacher_forcing
        batch = latent.size(0)
        device = latent.device

        # Concatenate latent and condition representations
        combined_input = torch.cat([latent, ag_rep], dim=-1)  # [batch, combined_dim]

        # Project to sequence hidden states
        hidden = self.input_projection(combined_input)  # [batch, max_len * hidden_dim]
        memory = hidden.view(batch, self.max_len, self.hidden_dim)  # [batch, max_len, hidden_dim]

        # Generation loop
        generated = torch.full((batch, 1), self.sos_idx, dtype=torch.long, device=device)
        logits_all = []
        max_steps = target_ids.size(1) if target_ids is not None else self.max_len

        for t in range(max_steps):
            # Embed inputs + positional encoding
            seq_len = generated.size(1)
            tgt = self.token_embedding(generated)  # [batch, seq_len, hidden_dim]
            pe = self.positional_encoding[:, :seq_len, :].to(device)
            tgt = tgt + pe

            # Create target mask for autoregressive decoding
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)

            # Transformer decoder (self-attention on target, cross-attention to memory)
            dec = self.transformer_decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)  # [batch, seq_len, hidden_dim]
            logits = self.output_projection(dec)  # [batch, seq_len, num_embeddings]
            step_logits = logits[:, -1, :]  # [batch, num_embeddings]
            logits_all.append(step_logits.unsqueeze(1))  # [batch, 1, num_embeddings]

            # Decide next input
            if target_ids is not None and torch.rand(1).item() < teacher_forcing_ratio:
                next_token = target_ids[:, t]
            else:
                next_token = step_logits.argmax(-1)
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

        # Concatenate logits: [batch, gen_len, num_embeddings]
        logits_all = torch.cat(logits_all, dim=1)
        return logits_all, generated
