python3 scripts/train_cvae.py \
--config-dir configs \
--config-name training.yaml \
++device='cuda' \
++trainer.accelerator='cuda' \
++trainer.gpus=1 \
++trainer.max_epochs=50 \
++trainer.strategy='auto' \
++trainer.num_nodes=1 \
++data.ab_embedding_path='datasets/cvae/train_antibody.pkl' \
++data.ag_embedding_path='datasets/cvae/train_antigen.pkl' \
++data.ab_description_embedding_path='datasets/cvae/train_description.pkl' \
++data.abset_path='datasets/abdes/test.csv' \
++model.predictor.contrastive_model_path='checkpoints/affinity_predictor/ab_ag_clip.pth' \
++model.predictor.cross_attention_model_path='checkpoints/affinity_predictor/cross_attention_model.pth' \
++model.predictor.pred_hidden_dim=256 \
++model.encoder.pretrained_model_name_or_path="esmc" \
++model.ag_dimen=960 \
++model.encoder.hidden_dim=960 \
++model.use_pcgrad=0 \
++data.batch_size=1 \
++data.num_workers=4 \
++model.upsampler.max_filter=512 \
++model.latent_dimen=64 \
++model.decoder.attn_method="general" \
++model.use_concat_condition=1 \
++optimizer.lr=0.0001 \
++model.decoder.use_teacher_forcing=0.0