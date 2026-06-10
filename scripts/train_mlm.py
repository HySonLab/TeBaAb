import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import hydra
from omegaconf import DictConfig
import rootutils
import torch
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import Trainer, callbacks

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
root_dir = os.path.abspath(__file__)
root = rootutils.setup_root(root_dir, indicator='train_mlm.py', pythonpath=True)

from src.models.modules.maskedLanguageModel import MaskedLanguageModel
from src.models.modules.cvae_mlm import MaskedLanguageCVAE
from src.data.masked_dataset import MaskedAntibodyDataset


class MaskedDataModule:
    """Simple data module for masked language modeling."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.batch_size = cfg.data.batch_size
        self.num_workers = cfg.data.num_workers
        self.seed = cfg.seed

        # Create dataset with masking strategy
        self.dataset = MaskedAntibodyDataset(
            ab_embedding_path=cfg.data.ab_embedding_path,
            ag_embedding_path=cfg.data.ag_embedding_path,
            ab_description_embedding_path=cfg.data.ab_description_embedding_path,
            abset_path=cfg.data.abset_path,
            mask_strategy=cfg.model.get('mask_strategy', 'random'),
            mask_ratio=cfg.model.get('mask_ratio', 0.15),
            max_ab_seq_len=cfg.model.max_ab_seq_len
        )

        self.dataset_size = len(self.dataset)
        self.train_size = int(cfg.data.train_size * self.dataset_size)
        self.val_size = self.dataset_size - self.train_size

        # Split data
        self.train_dataset, self.val_dataset = random_split(
            self.dataset,
            lengths=[self.train_size, self.val_size],
            generator=torch.Generator().manual_seed(self.seed)
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self._collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self._collate_fn
        )

    @staticmethod
    def _collate_fn(batch):
        """Custom collate function for padding batches."""
        # Stack tensors
        masked_seq_ids = torch.stack([item['masked_seq_ids'] for item in batch])
        target_seq_ids = torch.stack([item['target_seq_ids'] for item in batch])
        mask_positions = torch.stack([item['mask_positions'] for item in batch])

        ab_embeddings = torch.stack([item['ab_embedding'] for item in batch])
        ag_embeddings = torch.stack([item['ag_embedding'] for item in batch])
        des_embeddings = torch.stack([item['des_embedding'] for item in batch])

        return {
            'pdb_id': [item['pdb_id'] for item in batch],
            'ab_sequences': [item['ab_sequences'] for item in batch],
            'antigen': [item['antigen'] for item in batch],
            
            'masked_seq_ids': masked_seq_ids,
            'target_seq_ids': target_seq_ids,
            'mask_positions': mask_positions,
            
            'ab_embedding': ab_embeddings,
            'ag_embedding': ag_embeddings,
            'des_embedding': des_embeddings,
        }


@hydra.main(config_path='../configs', config_name='training', version_base="1.1")
def train_mlm(cfg: DictConfig) -> None:
    """
    Train the Masked Language Model.
    
    Args:
        cfg: Hydra configuration
    """
    # Create data module
    data_module = MaskedDataModule(cfg)

    # Create model
    model = MaskedLanguageCVAE(cfg)

    # Setup callbacks
    callback_list = [
        callbacks.ModelCheckpoint(
            dirpath=os.path.join(cfg.trainer.model_output_dir, 'checkpoints_mlm'),
            filename="checkpoint-{epoch:02d}-{val_loss:.2f}",
            monitor="val_loss",
            verbose=True,
            save_top_k=cfg.trainer.save_top_k,
            save_weights_only=False,
            save_last=True,
            every_n_epochs=cfg.trainer.save_every_n_epochs,
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            verbose=True,
            mode="min"
        )
    ]

    # Setup trainer
    trainer = Trainer(
        max_epochs=cfg.trainer.max_epochs,
        default_root_dir=cfg.trainer.model_output_dir,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.gpus,
        strategy=cfg.trainer.strategy,
        callbacks=callback_list,
        num_nodes=cfg.trainer.num_nodes
    )

    # Train
    trainer.fit(
        model,
        train_dataloaders=data_module.train_dataloader(),
        val_dataloaders=data_module.val_dataloader()
    )


if __name__ == '__main__':
    train_mlm()
