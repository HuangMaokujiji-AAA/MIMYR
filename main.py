import argparse
from dataclasses import dataclass
from datetime import datetime
import zipfile

import gdown

from models.combined_model import CombinedModel
from data_loader import SliceDataLoader
from models.biological_model import KDEModelForGuidance
import torch
import numpy as np
from inference import Inferernce
from evaluator import Evaluator
import copy, os, pandas as pd
import pickle as pkl

import json
import gc


np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


@dataclass
class TrainConfig:
    degree: int = 7
    hidden_sizes: tuple = (1024, 2048, 4096, 2048, 1024)
    activation: str = "silu"
    batchnorm: bool = False
    dropout: float = 0.0
    feature_type: str = "poly"
    num_rff_features: int = 256
    rff_gamma: float = 100.0
    rff_seed: int | None = None
    n_timesteps: int = 70
    schedule_type: str = "cosine"
    beta_start: float = 1e-10
    beta_end: float = 1e-9
    cosine_s: float = 0.008
    batch_size: int = 4096 * 50
    lr: float = 2e-4
    weight_decay: float = 0
    epochs: int = 1000000
    grad_clip: float = None
    ema_decay: float = 0.999

# ----------------- argparse -----------------
def get_args():
    parser = argparse.ArgumentParser(description="Run inference with config options.")

    # data / model paths
    parser.add_argument(
        "--run_mode", type=str, default="train", help="Mode for SliceDataLoader", choices=["train", "inference"]
    )
    parser.add_argument(
        "--training_slice_directory", type=str, default="data/cleaned_versions", help="Training adata, if mode is set to train"
    )
    parser.add_argument(
        "--val_slice_directory", type=str, default="data/cleaned_versions", help="Validation adata, if mode is set to train"
    )


    parser.add_argument(
        "--data_mode", type=str, default="rq1", help="Mode for SliceDataLoader"
    )
    parser.add_argument(
        "--data_label",
        type=str,
        default="cluster",
        help="Label type for SliceDataLoader",
    )
    parser.add_argument(
        "--location_model_checkpoint",
        type=str,
        default="model_checkpoints/smoothtune_conditional_ddpm_2d_checkpoint_400.pt",
        help="Path to trained location checkpoint",
    )
    parser.add_argument(
        "--cluster_model_checkpoint",
        type=str,  
        default="model_checkpoints/best_model_rq1.pt",
        help="Path to trained CelltypeModel checkpoint",
    )
    parser.add_argument(
        "--expression_model_checkpoint",
        type=str,  ### CHANGE
        default="model_checkpoints/TG-base4_epoch4_model.pt",
        help="Path to trained expression checkpoint",
    )

    parser.add_argument(
        "--location_inference_type",
        type=str,
        default="skip",
        help="Type of location inference",
    )
    parser.add_argument(
        "--kde_bandwidth", type=float, default=0.01, help="Bandwidth for KDE"
    )

    parser.add_argument(
        "--cluster_inference_type",
        type=str,
        default="model",
        help="How to infer subclass",
    )
    parser.add_argument(
        "--expression_inference_type",
        type=str,
        default="end",
        help="How to infer gene expression",
    )

    # training hyperparams
    parser.add_argument(
        "--epochs", type=int, default=500, help="Training epochs for CelltypeModel"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for CelltypeModel",
    )
    parser.add_argument(
        "--guidance_signal",
        type=float,
        default=0.01,
        help="Guidance signal for classifier-based guidance",
    )

    parser.add_argument(
        "--batch_size", type=int, default=1024, help="Batch size for CelltypeModel"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cpu/cuda)"
    )

    parser.add_argument(
        "--metrics",
        type=lambda s: s.split(","),
        help="Comma-separated list of metrics to compute (soft_accuracy,soft_correlation,neighborhood_enrichment,soft_precision)",
        default=[
            "soft_accuracy"
        ],  # ,"soft_accuracy", "soft_correlation", "neighborhood_enrichment", "soft_precision"],
    )
    parser.add_argument(
        "--metric_sampling",
        type=int,
        default=1,
        help="Percentage of samples to use for metric computation",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="results/debugging_cluster_model2.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--meta_info",
        type=str,
        default="4hierarchy_metainfo_mouse_geneunion2_DAG.pt",
        help="meta_info file path for GE prediction",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing the data",
    )

    parser.add_argument(
        "--artifact_dir",
        type=str,
        default="artifacts",
        help="Directory to save artifacts",
    )

    return parser.parse_args()


# ----------------- util -----------------
def write_row(row, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    header = not os.path.isfile(path)
    pd.DataFrame([row]).to_csv(path, mode="a", header=header, index=False)


def already_done(cfg, path):
    if not os.path.isfile(path):
        return False
    df = pd.read_csv(path, usecols=cfg.keys())
    return any((df == pd.Series(cfg)).all(axis=1))


# ----------------- main -----------------


def main():
    args = get_args()
    cfg = args.__dict__


    if not os.path.exists(cfg["artifact_dir"]):
        os.makedirs(cfg["artifact_dir"], exist_ok=True)

    if not os.path.exists(cfg["data_dir"]):
        gdown.download(id="1iJX3z9S_biGCdpc-uQWwJyFImhv2mdU8", output="data.zip")
        with zipfile.ZipFile("data.zip", 'r') as zip_ref:
            zip_ref.extractall(".")
    
    if not os.path.exists("model_checkpoints"):
        gdown.download(id="1OSh5JfXg2OXVfTIyGR33PkvQYGNqFXLD", output="model_checkpoints.zip")
        with zipfile.ZipFile("model_checkpoints.zip", 'r') as zip_ref:
            zip_ref.extractall(".")


    slice_data_loader = SliceDataLoader(
        mode=args.data_mode, label=args.data_label, cfg=copy.deepcopy(cfg)
    )
    slice_data_loader.prepare()

    cfg["full_gene_panel"] = True

    temp_test_slices = slice_data_loader.test_slices.copy()
    temp_ref_slices = slice_data_loader.reference_slices.copy()

    artifact_dir = cfg['artifact_dir']

    combined_model = CombinedModel(
        location_model_checkpoint=args.location_model_checkpoint,
        celltype_model_checkpoint=args.cluster_model_checkpoint,
        gene_exp_model_checkpoint=args.expression_model_checkpoint,
    )

    if args.run_mode == "train":
        combined_model.fit(args.training_slice_directory, args.val_slice_directory, cfg)
        exit(0)


    for i, slice in enumerate(temp_test_slices):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg["artifact_dir"] = f"{artifact_dir}/{timestamp}"
        os.makedirs(cfg["artifact_dir"], exist_ok=True)

        slice_data_loader.test_slices = [slice]

        slice_data_loader.reference_slices = temp_ref_slices[
            2 * i : 2 * i + 2
        ]
        if len(slice_data_loader.reference_slices) == 0:
            slice_data_loader.reference_slices = temp_ref_slices[-2:]

        cfg["slice_index"] = i


        closest_ref_slice = np.argsort(
            [
                np.square(
                    ref_slice.obsm["aligned_spatial"].mean(0)[-1]
                    - slice_data_loader.test_slices[0]
                    .obsm["aligned_spatial"]
                    .mean(0)[-1]
                )
                for ref_slice in slice_data_loader.reference_slices
            ]
        )[1]
        best_ref_slice = slice_data_loader.reference_slices[closest_ref_slice].copy()
        if args.data_mode == "rq4":
            best_ref_slice.obsm["aligned_spatial"][:, 0] = (
                slice_data_loader.test_slices[0].obsm["aligned_spatial"][:, 0].mean(0)
            )
        else:
            best_ref_slice.obsm["aligned_spatial"] = best_ref_slice.obsm[
                "aligned_spatial"
            ][:, :2]


        kdemodel = KDEModelForGuidance(
            [best_ref_slice], bandwidth=args.kde_bandwidth
        )
        kdemodel.fit()


        if already_done(cfg, args.out_csv):
            print("skip", cfg)
            return

        inf = Inferernce(
            combined_model, 
            kdemodel,
            slice_data_loader,
            copy.deepcopy(cfg),
        )

        pred = inf.run_inference(slice_data_loader.test_slices)
        print("Sending pred to evaluator...", pred)
        res = Evaluator(cfg).evaluate(
            pred, slice_data_loader.test_slices[0], sample=args.metric_sampling
        )
        res = {k: float(v) for k, v in res.items()}

        row = {**cfg, **res}
        write_row(row, args.out_csv)
        print("wrote", cfg)
        # save config to artifact dir as json

        # save config + results into artifact dir
        cfg_path = os.path.join(cfg["artifact_dir"], "config.json")
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)

        res_path = os.path.join(cfg["artifact_dir"], "results.json")
        with open(res_path, "w") as f:
            json.dump(res, f, indent=2)

        # optionally also save predictions
        pred_path = os.path.join(cfg["artifact_dir"], "pred.pkl")
        with open(pred_path, "wb") as f:
            pkl.dump(pred, f)

        # delete all local variables and collect garbage
        del pred, inf
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
