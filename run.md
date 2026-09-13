python main.py \
  --run_mode inference \
  --data_mode rq1 \
  --location_inference_type model \
  --cluster_inference_type model \
  --expression_inference_type model \
  --cluster_model_checkpoint model_checkpoints/best_model_rq1.pt \
  --expression_model_checkpoint model_checkpoints/TG-base4_epoch4_model.pt \
  --metrics soft_spearman_correlation,soft_f1 \
  --metric_sampling 100 \
  --out_csv results/rq1_full.csv


  