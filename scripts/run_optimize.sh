python3 scripts/optimize.py \
--config-dir configs \
--config-name optimize.yaml \
++log_folder="log_optimize/" \
++device='cuda' \
++model_path="checkpoints/cvae/best_wo_des.ckpt" \
++num_samples=10 \
++recon_wt_batch=4 \
++data.ab_embedding_path='datasets/cvae/test_antibody.pkl' \
++data.ag_embedding_path='datasets/cvae/test_antigen.pkl' \
++data.ab_description_embedding_path='datasets/cvae/test_description.pkl' \
++data.abset_path='datasets/abdes/test.csv' \
++data.num_workers=4 \
++model.predictor.contrastive_model_path="checkpoints/affinity_predictor/ab_ag_clip.pth" \
++model.predictor.cross_attention_model_path="checkpoints/affinity_predictor/cross_attention_model.pth" \
++model.predictor.pred_hidden_dim=256 \
++model.encoder.pretrained_model_name_or_path="checkpoints/esmc_300m_2024_12_v0.pth" \
++model.ag_dimen=960 \
++model.encoder.hidden_dim=960 \
++model.use_pcgrad=0 \
++model.upsampler.max_filter=512 \
++model.latent_dimen=64 \
++model.use_concat_condition=1 \
++model.decoder.attn_method="general" \
++model.decoder.use_teacher_forcing=0.4