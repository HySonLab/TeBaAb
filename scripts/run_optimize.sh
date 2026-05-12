#!/bin/bash
# Run directed evolution optimization.
#
# Usage:
#   bash scripts/run_optimize.sh              # With Description (default)
#   bash scripts/run_optimize.sh --no-desc    # Without Description
#
# The script uses the same pkl files and the same seed as training, so
# optimize.py automatically iterates over the held-out test split.

set -e

USE_DESCRIPTION=1
if [[ "$1" == "--no-desc" ]]; then
    USE_DESCRIPTION=0
fi

LOG_FOLDER="results/$([ "${USE_DESCRIPTION}" -eq 1 ] && echo 'with_description' || echo 'without_description')/log_optimize"

python3 scripts/optimize.py \
    --config-dir configs \
    --config-name optimize.yaml \
    ++device='cuda' \
    ++model.use_description="${USE_DESCRIPTION}" \
    ++num_samples=10 \
    ++recon_wt_batch=4 \
    ++data.ab_embedding_path='datasets/cvae/all_antibody.pkl' \
    ++data.ag_embedding_path='datasets/cvae/all_antigen.pkl' \
    ++data.ab_description_embedding_path='datasets/cvae/all_description.pkl' \
    ++data.abset_path='datasets/abdes/abdes_all.csv' \
    ++data.train_size=0.8 \
    ++data.val_size=0.1 \
    ++data.test_size=0.1 \
    ++data.num_workers=4 \
    ++model.use_concat_condition=1 \
    ++model.latent_dimen=64 \
    ++model.ag_dimen=960 \
    ++model.encoder.hidden_dim=960 \
    ++model.encoder.pretrained_model_name_or_path='checkpoints/esmc_300m_2024_12_v0.pth' \
    ++model.upsampler.max_filter=512 \
    ++model.decoder.attn_method='general' \
    ++model.decoder.use_teacher_forcing=0.4 \
    ++model.use_pcgrad=0 \
    ++model.predictor.contrastive_model_path='checkpoints/affinity_predictor/ab_ag_clip.pth' \
    ++model.predictor.cross_attention_model_path='checkpoints/affinity_predictor/cross_attention_model.pth' \
    ++model.predictor.pred_hidden_dim=256 \
    ++log_folder="${LOG_FOLDER}"
