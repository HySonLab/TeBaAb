#!/bin/bash
# Text-modality ablation experiments.
#
# Trains four CVAE variants that differ only in how the text description
# embedding is provided to the decoder.  Comparing val_loss across
# conditions shows whether real text carries useful signal beyond its
# distributional statistics.
#
# Conditions
# ----------
#   real    – true description for each sample            (full model)
#   shuffle – descriptions randomly permuted across samples (wrong alignment)
#   random  – i.i.d. Gaussian noise, same shape as real embeddings
#   zero    – all-zero vector (no text information at all)
#
# Usage: bash scripts/run_text_ablation.sh

set -e

BASE_ARGS=(
    --config-dir configs
    --config-name training.yaml
    ++device='cuda'
    ++trainer.accelerator='cuda'
    ++trainer.gpus=1
    ++trainer.max_epochs=50
    ++trainer.strategy='auto'
    ++trainer.num_nodes=1
    ++data.ab_embedding_path='datasets/cvae/train_antibody.pkl'
    ++data.ag_embedding_path='datasets/cvae/train_antigen.pkl'
    ++data.ab_description_embedding_path='datasets/cvae/train_description.pkl'
    ++data.abset_path='datasets/abdes/train.csv'
    ++model.predictor.contrastive_model_path='checkpoints/affinity_predictor/ab_ag_clip.pth'
    ++model.predictor.cross_attention_model_path='checkpoints/affinity_predictor/cross_attention_model.pth'
    ++model.predictor.pred_hidden_dim=256
    ++model.encoder.pretrained_model_name_or_path='esmc'
    ++model.ag_dimen=960
    ++model.encoder.hidden_dim=960
    ++model.use_pcgrad=0
    ++data.batch_size=1
    ++data.num_workers=4
    ++model.upsampler.max_filter=512
    ++model.latent_dimen=64
    ++model.decoder.attn_method='general'
    ++model.use_concat_condition=1
    ++optimizer.lr=0.0001
    ++model.decoder.use_teacher_forcing=0.0
)

for MODE in real shuffle random zero; do
    echo "============================================"
    echo "  Training with text_mode=${MODE}"
    echo "============================================"
    python3 scripts/train_cvae.py \
        "${BASE_ARGS[@]}" \
        ++data.text_mode="${MODE}" \
        ++trainer.model_output_dir="models/text_ablation/${MODE}"
done

echo "All ablation runs complete."
echo "Compare val_loss across: models/text_ablation/{real,shuffle,random,zero}/"
