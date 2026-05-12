#!/bin/bash
# Inference-time text ablation for the optimization (directed evolution) stage.
#
# Uses the already-trained best_w_des.ckpt checkpoint but substitutes the
# description embedding at inference time.  No retraining required.
#
# Conditions (compare against results/with_description/ which uses text_mode=real):
#   shuffle – each sample receives another sample's description (wrong alignment)
#   random  – i.i.d. Gaussian noise with the same shape as a real embedding
#   zero    – all-zero vector (no text signal whatsoever)
#
# Usage: bash scripts/run_text_ablation_optimize.sh

set -e

BASE_ARGS=(
    --config-dir configs
    --config-name optimize.yaml
    ++device='cuda'
    ++model_path="checkpoints/cvae/best_w_des.ckpt"
    ++num_samples=10
    ++recon_wt_batch=4
    ++data.ab_embedding_path='datasets/cvae/test_antibody.pkl'
    ++data.ag_embedding_path='datasets/cvae/test_antigen.pkl'
    ++data.ab_description_embedding_path='datasets/cvae/test_description.pkl'
    ++data.abset_path='datasets/abdes/test.csv'
    ++data.num_workers=4
    ++model.predictor.contrastive_model_path="checkpoints/affinity_predictor/ab_ag_clip.pth"
    ++model.predictor.cross_attention_model_path="checkpoints/affinity_predictor/cross_attention_model.pth"
    ++model.predictor.pred_hidden_dim=256
    ++model.encoder.pretrained_model_name_or_path="checkpoints/esmc_300m_2024_12_v0.pth"
    ++model.ag_dimen=960
    ++model.encoder.hidden_dim=960
    ++model.use_pcgrad=0
    ++model.upsampler.max_filter=512
    ++model.latent_dimen=64
    ++model.use_concat_condition=1
    ++model.use_des_in_encoder=1
    ++model.decoder.attn_method="general"
    ++model.decoder.use_teacher_forcing=0.4
)

for MODE in real; do
    echo "============================================"
    echo "  Optimizing with text_mode=${MODE}"
    echo "============================================"
    python3 scripts/optimize.py \
        "${BASE_ARGS[@]}" \
        ++data.text_mode="${MODE}" \
        ++log_folder="results/text_ablation_optimize/${MODE}/log_optimize"
done

echo "All inference-time text ablation runs complete."
echo "Compare optimized_fitness across:"
echo "  results/with_description/optimization_log.csv   (real text)"
echo "  results/text_ablation_optimize/{shuffle,random,zero}/log_optimize/optimization_log.csv"
