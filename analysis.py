import torch
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tqdm
import os

import torch
import numpy as np
import anndata as ad


def generate_anndata_from_samples(
    region_model, xyz, device="cuda", sample_from_probs=False
):
    """
    xyz: [N, D] numpy array of coordinates (must include 3 spatial dims in [:, :3])
    xyz_labels: [N] array of known tokens to use for conditioning (initial labels if gibbs=True)
    use_budget: when True and use_conditionals=False, enforces budget-based sampling per group
    graph_smooth: when True and use_conditionals=False, applies 1-step GCN-like smoothing before sampling
    """

    region_model.eval().to(device)

    with torch.no_grad():
        xyz_tensor = torch.tensor(xyz, dtype=torch.float32).to(device)
        input_tensor = xyz_tensor
        batch_size = 1000
        outputs = []
        for i in range(0, input_tensor.size(0), batch_size):
            logits_batch = region_model.model(input_tensor[i : i + batch_size])
            outputs.append(logits_batch)
        logits = torch.cat(outputs, dim=0)

        probs = torch.softmax(logits, dim=1).cpu().numpy()

        preds = np.array(
            [
                np.random.choice(len(p), p=p) if sample_from_probs else np.argmax(p)
                for p in probs
            ]
        )

    if isinstance(xyz, torch.Tensor):
        xyz = xyz.detach().cpu().numpy()

    adata = ad.AnnData(X=np.zeros((xyz.shape[0], 1)))
    adata.obsm["spatial"] = xyz[:, :3]
    adata.obs["token"] = preds
    return adata, preds, probs


def assign_shared_colors(adatas, color_key="token", palette="tab10", seed=None):
    # collect all unique labels across datasets
    all_labels = pd.Categorical(
        np.concatenate([adata.obs[color_key] for adata in adatas])
    ).categories

    # generate colors
    colors = [
        mcolors.to_hex(c) for c in sns.color_palette(palette, n_colors=len(all_labels))
    ]

    # optionally shuffle them deterministically
    if seed is not None:
        rng = np.random.default_rng(seed)
        colors = list(rng.permutation(colors))

    # assign to all AnnDatas
    for adata in adatas:
        adata.obs[color_key] = pd.Categorical(
            adata.obs[color_key], categories=all_labels
        )
        adata.uns[f"{color_key}_colors"] = colors

    return all_labels, colors


def plot_spatial_with_palette(
    adata,
    color_key="token",
    spot_size=0.001,
    size=10,
    scale_factor=1,
    show_legend=False,
    figsize=(10, 10),
    save=None,
    saggital=False,
):
    adata = adata.copy()
    if saggital:
        print("Yes it is saggital")
        adata.obsm["spatial"][:, [0, 1]] = adata.obsm["spatial"][:, [1, 2]]
    with plt.rc_context({"figure.figsize": figsize}):
        sc.pl.spatial(
            adata,
            color=color_key,
            spot_size=spot_size,
            size=size,
            scale_factor=scale_factor,
            legend_loc="right margin" if show_legend else None,
            save=None,
            title="",
            frameon=False,
        )
        if save is not None:
            os.makedirs(os.path.dirname(save), exist_ok=True)
            plt.savefig(save, bbox_inches="tight", dpi=300)
        plt.close()
