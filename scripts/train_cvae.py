import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import hydra
from omegaconf import DictConfig
import rootutils

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
root_dir = os.path.abspath(__file__)
root = rootutils.setup_root(root_dir, indicator='train_cvae.py', pythonpath=True)

from src.data.training_data_module import TrainingDataModule
from src.models.CVAE import CVAE

from pytorch_lightning import Trainer, callbacks


@hydra.main(config_path='../configs', config_name='training', version_base="1.1")
def train(cfg: DictConfig) -> None:
    dataModule = TrainingDataModule(cfg)
    model = CVAE(cfg)

    callback_list = [
        callbacks.ModelCheckpoint(
            dirpath=os.path.join(cfg.trainer.model_output_dir, 'checkpoints'),
            filename="checkpoint-{epoch:02d}-{val_loss:.2f}",
            monitor="val_loss",
            verbose=True,
            save_top_k=cfg.trainer.save_top_k,
            save_weights_only=False,
            save_last=False,
            every_n_epochs=cfg.trainer.save_every_n_epochs,
        )
    ]

    trainer = Trainer(
        max_epochs=cfg.trainer.max_epochs,
        default_root_dir=cfg.trainer.model_output_dir,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.gpus,
        strategy=cfg.trainer.strategy,
        callbacks=callback_list,
        num_nodes=cfg.trainer.num_nodes
    )
    trainer.fit(model, dataModule)


if __name__ == '__main__':
    train()
