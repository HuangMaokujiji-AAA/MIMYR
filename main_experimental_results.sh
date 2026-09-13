# python main.py --data_mode "rq1" --location_inference_type "closest_slice" --cluster_inference_type "majority_baseline" --expression_inference_type "lookup" --metrics "soft_correlation,soft_spearman_correlation,soft_f1" --out_csv "results/rq1.csv" --metric_sampling 100
# python main.py --data_mode "rq1" --location_inference_type "model" --cluster_inference_type "model" --expression_inference_type "model" --metrics "soft_correlation,soft_spearman_correlation,soft_f1" --out_csv "results/rq1.csv" --metric_sampling 100 --expression_model_checkpoint "/compute/oven-0-13/skrieger/TG-base4/epoch4_model.pt" --cluster_model_checkpoint "experimentation/best_model_cluster1.pt"

# python main.py --data_mode "rq2_v2" --location_inference_type "uniform_circle" --cluster_inference_type "majority_baseline" --expression_inference_type "lookup" --metrics "soft_correlation,soft_spearman_correlation,soft_f1" --out_csv "results/rq2.csv" --metric_sampling 100
# python main.py --data_mode "rq2" --location_inference_type "model" --cluster_inference_type "model" --expression_inference_type "model" --metrics "soft_correlation,soft_spearman_correlation,soft_f1" --out_csv "results/rq2.csv" --metric_sampling 100 --expression_model_checkpoint "/compute/oven-0-13/skrieger/TG-base4/epoch4_model.pt" --cluster_model_checkpoint "experimentation/best_model_cluster1.pt"

python main.py --data_mode "rq3" --location_inference_type "closest_slice" --cluster_inference_type "majority_baseline" --expression_inference_type "lookup" --metrics "soft_correlation,soft_spearman_correlation,soft_f1" --out_csv "results/rq3.csv" --metric_sampling 100
python main.py --data_mode "rq3" --location_inference_type "model" --cluster_inference_type "model" --expression_inference_type "model" --metrics "soft_correlation,soft_spearman_correlation,soft_f1" --out_csv "results/rq3.csv" --metric_sampling 100 --expression_model_checkpoint "/compute/oven-0-13/skrieger/TG-base4/epoch4_model.pt" --cluster_model_checkpoint "experimentation/best_model_cluster1.pt"
python main.py --data_mode "rq3" --location_inference_type "model" --cluster_inference_type "model" --expression_inference_type "model" --metrics "soft_correlation,soft_spearman_correlation,soft_f1" --out_csv "results/rq3.csv" --metric_sampling 100 --expression_model_checkpoint "/compute/oven-0-13/skrieger/Zhuang-2/epoch60_model.pt" --cluster_model_checkpoint "experimentation/best_model_rq3.pt"

# python main.py --data_mode "rq4" --location_inference_type "closest_slice" --cluster_inference_type "majority_baseline" --expression_inference_type "lookup" --metrics "soft_correlation,soft_spearman_correlation,soft_f1" --out_csv "results/rq4.csv" --metric_sampling 100
# python main.py --data_mode "rq4" --location_inference_type "model" --cluster_inference_type "model" --expression_inference_type "model" --metrics "soft_correlation,soft_spearman_correlation,soft_f1" --out_csv "results/rq4.csv" --metric_sampling 100 --expression_model_checkpoint "/compute/oven-0-13/skrieger/Zhuang-3/epoch60_model.pt" --cluster_model_checkpoint "experimentation/best_model_rq4.pt" 

# python main.py --data_mode "rq5" --location_inference_type "closest_slice" --cluster_inference_type "majority_baseline" --expression_inference_type "lookup" --metrics "soft_correlation,soft_spearman_correlation,soft_f1" --out_csv "results/rq5.csv" --metric_sampling 100
# python main.py --data_mode "rq5" --location_inference_type "model" --cluster_inference_type "model" --expression_inference_type "model" --metrics "soft_correlation,soft_spearman_correlation,soft_f1" --out_csv "results/rq5.csv" --metric_sampling 100 --expression_model_checkpoint "/compute/oven-0-13/skrieger/Trem2_5xFAD/epoch60_model.pt" --cluster_model_checkpoint "experimentation/best_model_rq5.pt"





