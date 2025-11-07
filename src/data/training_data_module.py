import math
import torch
from pytorch_lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader, random_split

from src.data.dataset import TraningDataset
from torch.nn.utils.rnn import pad_sequence


def pad_and_stack(tensors, fixed_len=None):
    if fixed_len is not None:
        padded = [torch.cat([t[:fixed_len], torch.zeros(max(0, fixed_len - t.size(0)), t.size(1))]) for t in tensors]
        return torch.stack(padded)
    return pad_sequence(tensors, batch_first=True)


def get_padding_mask(tensors, fixed_len=None):
    lengths = [min(len(t), fixed_len) if fixed_len else len(t) for t in tensors]
    max_len = fixed_len or max(lengths)
    return torch.tensor([[1] * l + [0] * (max_len - l) for l in lengths], dtype=torch.bool)


class CustomCollateFn:
    def __init__(self):
        pass

    def __call__(self, batch):
        ab_tensors = [item["ab_embedding"].clone().detach() for item in batch]
        ag_tensors = [item["ag_embedding"].clone().detach() for item in batch]
        des_tensors = [item["description_embedding"].clone().detach() for item in batch]

        return {
            "pdb_id": [item["pdb_id"] for item in batch],
            "ag_sequences": [item["ag_sequences"] for item in batch],
            "ab_sequence_heavy": [item["ab_sequence_heavy"] for item in batch],
            "ab_sequence_light": [item["ab_sequence_light"] for item in batch],
            "ab_sequences": [item["ab_sequences"] for item in batch],

            "ab_embedding": pad_and_stack(ab_tensors),
            "ag_embedding": pad_and_stack(ag_tensors),
            "des_embedding": pad_and_stack(des_tensors),
        }


class TrainingDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super(TrainingDataModule, self).__init__()
        self.ag_embedding_path = cfg.data.ag_embedding_path
        self.ab_embedding_path = cfg.data.ab_embedding_path
        self.ab_description_embedding_path = cfg.data.ab_description_embedding_path
        self.abset_path = cfg.data.abset_path
        self.batch_size = cfg.data.batch_size
        self.seed = cfg.seed
        self.num_workers = cfg.data.num_workers

        self.max_len = cfg.model.max_ab_seq_len
        self.low_res_dim = math.ceil(self.max_len / cfg.model.upsampler.stride ** cfg.model.upsampler.num_deconv_layers)
        self.max_len = self.low_res_dim * cfg.model.upsampler.stride ** cfg.model.upsampler.num_deconv_layers

        self.dataset = TraningDataset(ag_embedding_path=self.ag_embedding_path,
                                      ab_embedding_path=self.ab_embedding_path,
                                      ab_description_embedding_path=self.ab_description_embedding_path,
                                      abset_path=self.abset_path)
        self.dataset_size = len(self.dataset)

        self.train_size = int(cfg.data.train_size * self.dataset_size)
        self.val_size = self.dataset_size - self.train_size
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        self.train_dataset, self.val_dataset = random_split(self.dataset, lengths=[self.train_size, self.val_size],
                                                            generator=torch.Generator().manual_seed(self.seed))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,
                          collate_fn=CustomCollateFn(), persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                          collate_fn=CustomCollateFn(), persistent_workers=True)
