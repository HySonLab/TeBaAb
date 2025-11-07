from omegaconf import DictConfig
import torch
from transformers import BertModel, BertTokenizer
from esm.models.esmc import ESMC
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer


class ESMCEncoder(torch.nn.Module):
    def __init__(
            self,
            cfg: DictConfig,
    ):
        """
        Args:
            pretrained_model_name_or_path (str): Pre-trained model to load.
        """
        super(ESMCEncoder, self).__init__()
        pretrained_model_name_or_path = cfg.model.encoder.pretrained_model_name_or_path
        assert pretrained_model_name_or_path is not None

        self.tokenizer = EsmSequenceTokenizer()
        self.model = ESMC(
            d_model=960,
            n_heads=30,
            n_layers=15,
            use_flash_attn=True,
            tokenizer=self.tokenizer
        ).eval().to(torch.device('cuda:0'))
        state_dict = torch.load(
            pretrained_model_name_or_path,
            map_location="cuda"
        )
        self.model.load_state_dict(state_dict, strict=False)

    def forward(self, input: str) -> torch.Tensor:
        batch = self.tokenizer.batch_encode_plus(
            [input],
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt",
            return_attention_mask=True,
        )
        results = self.model(
            sequence_tokens=batch["input_ids"].to(torch.device('cuda:0')),
            sequence_id=batch["attention_mask"].to(torch.device('cuda:0')),
        )
        mask = batch["attention_mask"].unsqueeze(-1).expand_as(results.embeddings).to(
            torch.device('cuda:0'))  # Shape: [batch_size, seq_len, 960]
        emb = (results.embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # Shape: [batch_size, 960]

        return emb


class IgBertEncoder(torch.nn.Module):
    def __init__(
            self,
            cfg: DictConfig,
    ):
        """
        Args:
            pretrained_model_name_or_path (str): Pre-trained model to load.
        """
        super(IgBertEncoder, self).__init__()
        self.tokeniser = BertTokenizer.from_pretrained("Exscientia/IgBert", do_lower_case=False)
        self.model = BertModel.from_pretrained("Exscientia/IgBert", add_pooling_layer=False).to("cuda:0").eval()

    def forward(self, ab_sequence) -> torch.Tensor:
        paired_sequences = []
        for seq in ab_sequence:
            heavy_seq = seq.split("|")[0]
            light_seq = seq.split("|")[-1]
            new_seq = " ".join(heavy_seq) + "|" + " ".join(light_seq)
            paired_sequences.append(new_seq)

        tokens = self.tokeniser.batch_encode_plus(
            paired_sequences,
            add_special_tokens=True,
            pad_to_max_length=True,
            return_tensors="pt",
            return_special_tokens_mask=True
        )

        output = self.model(
            input_ids=tokens['input_ids'].to("cuda:0"),
            attention_mask=tokens['attention_mask'].to("cuda:0")
        )

        residue_embeddings = output.last_hidden_state
        residue_embeddings[tokens["special_tokens_mask"] == 1] = 0
        sequence_embeddings_sum = residue_embeddings.sum(1)

        sequence_lengths = torch.sum(tokens["special_tokens_mask"] == 0, dim=1).to("cuda:0")
        valid_residue_embeddings = sequence_embeddings_sum / sequence_lengths.unsqueeze(1)
        del tokens, output, residue_embeddings
        torch.cuda.empty_cache()

        return valid_residue_embeddings
