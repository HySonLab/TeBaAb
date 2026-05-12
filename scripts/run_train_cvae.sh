#!/bin/bash
# Train the CVAE.
#
# Usage:
#   bash scripts/run_train_cvae.sh              # With Description (default)
#   bash scripts/run_train_cvae.sh --no-desc    # Without Description
#
# Prerequisites — extract embeddings once from the full dataset CSV:
#   python3 datasets/extract_embedding.py \
#       --input_csv  datasets/abdes/abdes_all.csv \
#       --output_dir datasets/cvae \
#       --output_prefix all \
#       --modality all --embedding_type pooler --device cuda:0 \
#       --esmc_cache   checkpoints/esmc_300m_2024_12_v0.pth \
#       --scibert_model <path/to/scibert> \
#       --igbert_model  <path/to/igbert>
#
# The same pkl files are reused for optimization; the DataModule handles
# the train / val / test split automatically.

set -e

USE_DESCRIPTION=1
if [[ "$1" == "--no-desc" ]]; then
    USE_DESCRIPTION=0
fi

python3 scripts/train_cvae.py \
    --config-dir configs \
    --config-name training.yaml \
    ++device='cuda' \
    ++trainer.accelerator='cuda' \
    ++trainer.gpus=1 \
    ++trainer.max_epochs=100 \
    ++trainer.strategy='auto' \
    ++trainer.num_nodes=1 \
    ++data.ab_embedding_path='datasets/cvae/all_antibody.pkl' \
    ++data.ag_embedding_path='datasets/cvae/all_antigen.pkl' \
    ++data.ab_description_embedding_path='datasets/cvae/all_description.pkl' \
    ++data.abset_path='datasets/abdes/abdes_all.csv' \
    ++data.train_size=0.8 \
    ++data.val_size=0.1 \
    ++data.test_size=0.1 \
    ++data.batch_size=32 \
    ++data.num_workers=4 \
    ++model.use_description="${USE_DESCRIPTION}" \
    ++model.use_concat_condition=1 \
    ++model.latent_dimen=64 \
    ++model.ag_dimen=960 \
    ++model.encoder.hidden_dim=960 \
    ++model.encoder.pretrained_model_name_or_path='checkpoints/esmc_300m_2024_12_v0.pth' \
    ++model.upsampler.max_filter=512 \
    ++model.decoder.use_teacher_forcing=0.4 \
    ++model.decoder.attn_method='general' \
    ++model.use_pcgrad=0 \
    ++model.predictor.contrastive_model_path='checkpoints/affinity_predictor/ab_ag_clip.pth' \
    ++model.predictor.cross_attention_model_path='checkpoints/affinity_predictor/cross_attention_model.pth' \
    ++model.predictor.pred_hidden_dim=256 \
    ++optimizer.lr=0.0001
