import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import cKDTree
import tqdm
from scipy.spatial import Delaunay
import numpy as np
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm


import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
import pandas as pd

from scipy.spatial import Delaunay
import numpy as np

import numpy as np
from collections import Counter
from scipy.spatial import Delaunay

import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr
import numpy as np
from scipy.spatial import cKDTree



def one_hot_encode(celltypes):
    unique_types = list(set(celltypes))
    I = np.eye(len(unique_types), dtype=np.float32)
    encoding_dict = {ctype: I[i] for i, ctype in enumerate(unique_types)}
    return encoding_dict, len(unique_types)


def soft_accuracy(
    gt_celltypes,
    gt_positions,
    pred_celltypes,
    pred_positions,
    radius=None,
    k=0,
    return_list=False,
    return_percent=False,
    sample=None,
):
    encoding_dict, num_classes = one_hot_encode(gt_celltypes + pred_celltypes)

    gt_positions = np.array(gt_positions)
    pred_positions = np.array(pred_positions)

    gt_tree = cKDTree(gt_positions)
    pred_tree = cKDTree(pred_positions)

    result = []
    gt_distributions = []
    pred_distributions = []
    gt_neighbors_all = []
    pred_neighbors_all = []

    if sample is not None:
        percent = sample
        n = int(len(gt_positions) * percent / 100)
        indices = np.random.choice(len(gt_positions), size=n, replace=False)
        samples = gt_positions[indices]

    else:
        samples = gt_positions

    print("samples",samples)

    for i, pos in tqdm(list(enumerate(samples))):
        if k > 0:
            gt_distances, gt_indices = gt_tree.query(pos, k=k + 1)
            pred_distances, pred_indices = pred_tree.query(pos, k=k + 1)
            gt_neighbors = gt_indices[1:]
            pred_neighbors = pred_indices[1:]
        else:
            gt_neighbors = gt_tree.query_ball_point(pos, radius)
            pred_neighbors = pred_tree.query_ball_point(pos, radius)
        gt_neighbors_all.append(gt_neighbors)
        pred_neighbors_all.append(pred_neighbors)

        gt_encoding_sum = np.sum(
            [encoding_dict[gt_celltypes[j]] for j in gt_neighbors], axis=0
        )
        pred_encoding_sum = np.sum(
            [encoding_dict[pred_celltypes[j]] for j in pred_neighbors], axis=0
        )

        # Normalize
        gt_norm = np.linalg.norm(gt_encoding_sum)
        pred_norm = np.linalg.norm(pred_encoding_sum)

        gt_distribution = (
            gt_encoding_sum / gt_norm if gt_norm != 0 else np.zeros(num_classes)
        )
        pred_distribution = (
            pred_encoding_sum / pred_norm if pred_norm != 0 else np.zeros(num_classes)
        )
        gt_distributions.append(gt_distribution)
        pred_distributions.append(pred_distribution)

        similarity = cosine_similarity(
            gt_distribution.reshape(1, -1), pred_distribution.reshape(1, -1)
        )[0, 0]
        result.append(similarity)

        if i % 10000 == 0:
            print(np.mean(result))

    if return_percent:
        counts = np.sum([encoding_dict[ct] for ct in gt_celltypes], axis=0) / np.sum(
            [encoding_dict[ct] for ct in gt_celltypes]
        )
        return [
            (gt_distribution * counts).sum() for gt_distribution in gt_distributions
        ]
    if return_list:
        return result

    return np.mean(result) if result else 0.0




def delauney_colocalization(
    gt_celltypes, gt_positions, pred_celltypes, pred_positions, encoding_dict=None
):
    """
    Build Delaunay graph for GT and Pred positions and compute L1 distance
    between their edge-type count maps (sparse via Counter).

    Parameters
    ----------
    gt_celltypes : list[str]
        Ground-truth cell type labels (length N).
    gt_positions : array-like, shape (N, 2)
        Ground-truth positions.
    pred_celltypes : list[str]
        Predicted cell type labels (length M).
    pred_positions : array-like, shape (M, 2)
        Predicted positions.
    encoding_dict : dict, optional
        Mapping from cell type to index. If None, builds internally.

    Returns
    -------
    l1_distance : float
        L1 distance (sum of absolute differences) between GT and Pred edge-type counts.
    """

    # Build encoding dict if not given
    if encoding_dict is None:
        all_cts = list(set(gt_celltypes) | set(pred_celltypes))
        encoding_dict = {ct: i for i, ct in enumerate(all_cts)}

    def build_count_counter(celltypes, positions):
        tri = Delaunay(positions)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                a, b = sorted((simplex[i], simplex[(i + 1) % 3]))
                edges.add((a, b))

        counter = Counter()
        for a, b in edges:
            ia, ib = encoding_dict[celltypes[a]], encoding_dict[celltypes[b]]
            counter[(ia, ib)] += 1
            if ia != ib:
                counter[(ib, ia)] += 1  # keep symmetric
        return counter

    gt_counter = build_count_counter(gt_celltypes, np.array(gt_positions))
    pred_counter = build_count_counter(pred_celltypes, np.array(pred_positions))

    all_keys = set(gt_counter.keys()) | set(pred_counter.keys())
    l1_distance = sum(abs(gt_counter[k] - pred_counter[k]) for k in all_keys)

    return l1_distance / len(gt_positions)


def gridized_l1_distance(
    gt_positions, pred_positions, radius=None, k=0, grid_size=50, return_list=False
):
    """
    Estimate weighted L1 distance between two distributions by evaluating
    density differences on a regular grid of points spanning the data space.

    Parameters
    ----------
    gt_positions : array-like, shape (n_gt, d)
        Ground-truth positions.
    pred_positions : array-like, shape (n_pred, d)
        Predicted positions.
    radius : float, optional
        Radius for density estimation (used if k == 0).
    k : int, optional
        k for kNN density estimation. If > 0, kNN mode is used.
    grid_size : int, optional
        Number of grid points per dimension.
    return_list : bool, optional
        If True, return list of per-grid-point distances. Otherwise return mean.
    """
    gt_positions = np.array(gt_positions)
    pred_positions = np.array(pred_positions)

    d = gt_positions.shape[1]
    Vd = np.pi ** (d / 2) / np.math.gamma(d / 2 + 1)  # volume of unit d-ball
    n_gt, n_pred = len(gt_positions), len(pred_positions)

    # Build trees
    gt_tree = cKDTree(gt_positions)
    pred_tree = cKDTree(pred_positions)

    # Grid bounding box
    mins = np.minimum(gt_positions.min(axis=0), pred_positions.min(axis=0))
    maxs = np.maximum(gt_positions.max(axis=0), pred_positions.max(axis=0))
    grid_axes = [np.linspace(mins[i], maxs[i], grid_size) for i in range(d)]
    mesh = np.meshgrid(*grid_axes, indexing="ij")
    grid_points = np.stack([m.ravel() for m in mesh], axis=-1)

    results = []

    for i, pos in tqdm(list(enumerate(grid_points))):
        if k > 0:
            # kNN mode
            gt_distances, _ = gt_tree.query(pos, k=k)
            pred_distances, _ = pred_tree.query(pos, k=k)
            r_gt = gt_distances[-1]
            r_pred = pred_distances[-1]

            p_hat = k / (n_gt * Vd * (r_gt**d)) if r_gt > 0 else 0
            q_hat = k / (n_pred * Vd * (r_pred**d)) if r_pred > 0 else 0
        else:
            # Radius mode
            gt_neighbors = gt_tree.query_ball_point(pos, radius)
            pred_neighbors = pred_tree.query_ball_point(pos, radius)

            k_gt = len(gt_neighbors)
            k_pred = len(pred_neighbors)

            p_hat = k_gt / (n_gt * Vd * (radius**d)) if radius > 0 else 0
            q_hat = k_pred / (n_pred * Vd * (radius**d)) if radius > 0 else 0

        results.append(abs(p_hat - q_hat))

    return results if return_list else np.mean(results) if results else 0.0



def gridized_kl_divergence(
    gt_positions,
    pred_positions,
    radius=None,
    k=0,
    grid_size=50,
    return_list=False,
    eps=1e-200,
):
    """
    Estimate KL divergence KL(P||Q) between two distributions by evaluating
    density estimates on a regular grid of points spanning the data space.

    Parameters
    ----------
    gt_positions : array-like, shape (n_gt, d)
        Ground-truth samples (distribution P).
    pred_positions : array-like, shape (n_pred, d)
        Predicted samples (distribution Q).
    radius : float, optional
        Radius for density estimation (used if k == 0).
    k : int, optional
        k for kNN density estimation. If > 0, kNN mode is used.
    grid_size : int, optional
        Number of grid points per dimension.
    return_list : bool, optional
        If True, return list of per-grid-point KL terms. Otherwise return mean.
    eps : float, optional
        Small constant to avoid log(0) and division by zero.
    """
    gt_positions = np.array(gt_positions)
    pred_positions = np.array(pred_positions)

    d = gt_positions.shape[1]
    Vd = np.pi ** (d / 2) / np.math.gamma(d / 2 + 1)  # volume of unit d-ball
    n_gt, n_pred = len(gt_positions), len(pred_positions)

    # Build trees
    gt_tree = cKDTree(gt_positions)
    pred_tree = cKDTree(pred_positions)

    # Grid bounding box
    mins = np.minimum(gt_positions.min(axis=0), pred_positions.min(axis=0))
    maxs = np.maximum(gt_positions.max(axis=0), pred_positions.max(axis=0))
    grid_axes = [np.linspace(mins[i], maxs[i], grid_size) for i in range(d)]
    mesh = np.meshgrid(*grid_axes, indexing="ij")
    grid_points = np.stack([m.ravel() for m in mesh], axis=-1)

    results = []

    for i, pos in tqdm(list(enumerate(grid_points))):
        if k > 0:
            # kNN mode
            gt_distances, _ = gt_tree.query(pos, k=k)
            pred_distances, _ = pred_tree.query(pos, k=k)
            r_gt = gt_distances[-1]
            r_pred = pred_distances[-1]

            p_hat = k / (n_gt * Vd * (r_gt**d)) if r_gt > 0 else 0.0
            q_hat = k / (n_pred * Vd * (r_pred**d)) if r_pred > 0 else 0.0
        else:
            # Radius mode
            gt_neighbors = gt_tree.query_ball_point(pos, radius)
            pred_neighbors = pred_tree.query_ball_point(pos, radius)

            k_gt = len(gt_neighbors)
            k_pred = len(pred_neighbors)

            p_hat = k_gt / (n_gt * Vd * (radius**d)) if radius > 0 else 0.0
            q_hat = k_pred / (n_pred * Vd * (radius**d)) if radius > 0 else 0.0

        # Add eps to avoid instability
        p_hat = max(p_hat, eps)
        q_hat = max(q_hat, eps)

        results.append(p_hat * np.log(p_hat / q_hat))

    return results if return_list else np.mean(results) if results else 0.0


import numpy as np
import scipy.sparse as sp


def _nnz_per_gene(X):
    if sp.issparse(X):
        return np.asarray(X.getnnz(axis=0)).ravel()
    return np.asarray((X > 0).sum(axis=0)).ravel()


def intersect_and_filter_X(gt_adata, pred_adata, min_expr_cells=0, gene_set=None):
    # 1) intersect genes (order is preserved by AnnData slicing)
    common_genes = gt_adata.var_names.intersection(pred_adata.var_names)
    if gene_set is not None:
        common_genes = common_genes.intersection(np.asarray(gene_set))

    if len(common_genes) == 0:
        raise ValueError("No overlapping genes between gt_adata and pred_adata.")
    print(gt_adata.var_names)
    gt_common = gt_adata[:, common_genes]
    pred_common = pred_adata[:, common_genes]
    print(len(common_genes))

    # 2) require expression in each adata
    gt_nnz = _nnz_per_gene(gt_common.X)
    pred_nnz = _nnz_per_gene(pred_common.X)
    keep_mask = (gt_nnz >= min_expr_cells) & (pred_nnz >= min_expr_cells)

    if not np.any(keep_mask):
        raise ValueError("No genes pass the expression filter in both adatas.")

    # 3) return filtered .X matrices and the kept gene names (same order)
    gt_X_filtered = gt_common[:, keep_mask].X
    pred_X_filtered = pred_common[:, keep_mask].X
    kept_genes = common_genes[keep_mask]
    if sp.issparse(gt_X_filtered):
        gt_X_filtered = gt_X_filtered.todense()
    if sp.issparse(pred_X_filtered):
        pred_X_filtered = pred_X_filtered.todense()

    return gt_X_filtered.tolist(), pred_X_filtered.tolist(), kept_genes




def soft_correlation(
    gt_adata,
    gt_positions,
    pred_adata,
    pred_positions,
    radius=None,
    k=0,
    sample=None,
    return_list=False,
    corr_type="pearson",
    gene_set=None,
):
    """
    gt_expressions, pred_expressions: list or array of gene expression vectors (shape [num_cells, num_genes])
    gt_positions, pred_positions: list or array of positions (shape [num_cells, 2] or [num_cells, 3])
    radius: radius for neighbor search (if k=0)
    k: number of neighbors to consider (if k>0)
    sample: if provided, percentage of gt_positions to sample
    """

    if corr_type == "pearson":
        corr_fn = pearsonr
    elif corr_type == "spearman":
        corr_fn = spearmanr
    gt_expressions, pred_expressions, genes = intersect_and_filter_X(
        gt_adata, pred_adata, 1, gene_set
    )
    print(f"running soft_correlation on {len(genes)} genes")
    gt_positions = np.array(gt_positions)
    pred_positions = np.array(pred_positions)
    gt_expressions = np.array(gt_expressions)
    pred_expressions = np.array(pred_expressions)

    gt_tree = cKDTree(gt_positions)
    pred_tree = cKDTree(pred_positions)

    gt_sums = []
    pred_sums = []

    if sample is not None:
        percent = sample
        n = int(len(gt_positions) * percent / 100)
        indices = np.random.choice(len(gt_positions), size=n, replace=False)
        samples = gt_positions[indices]
    else:
        samples = gt_positions

    correlations_all = []
    for i, pos in tqdm(list(enumerate(samples))):
        if k > 0:
            gt_distances, gt_indices = gt_tree.query(pos, k=k + 1)
            pred_distances, pred_indices = pred_tree.query(pos, k=k + 1)
            gt_neighbors = gt_indices[1:]  # exclude self
            pred_neighbors = pred_indices[1:]
        else:
            gt_neighbors = gt_tree.query_ball_point(pos, radius)
            pred_neighbors = pred_tree.query_ball_point(pos, radius)

        gt_sum = np.sum(gt_expressions[gt_neighbors], axis=0)
        pred_sum = np.sum(pred_expressions[pred_neighbors], axis=0)

        gt_sums.append(gt_sum)
        pred_sums.append(pred_sum)

        pred_sum[0] = (
            pred_sum[0] + 1e-15
        )  # to avoid NaN in pearsonr when pred_sum is all zeros

        correlations_all.append(corr_fn(gt_sum, pred_sum)[0])

        if i % 10000 == 0 and i > 0:
            print(f"Processed {i} samples...")

    gt_sums = np.array(gt_sums).flatten()
    pred_sums = np.array(pred_sums).flatten()

    if len(gt_sums) == 0 or len(pred_sums) == 0:
        return 0.0

    if return_list:
        return correlations_all

    correlation, _ = corr_fn(gt_sums, pred_sums)
    return correlation


def soft_f1(
    gt_adata,
    gt_positions,
    pred_adata,
    pred_positions,
    radius=None,
    k=0,
    sample=None,
    return_list=False,
    gene_set=None,
):
    """
    gt_expressions, pred_expressions: array of shape [num_cells, num_genes]
    gt_positions, pred_positions: array of shape [num_cells, 2] or [num_cells, 3]
    radius: radius for neighbor search (if k=0)
    k: number of neighbors to consider (if k>0)
    sample: if provided, percentage of gt_positions to sample
    """
    gt_expressions, pred_expressions, genes = intersect_and_filter_X(
        gt_adata, pred_adata, gene_set=gene_set
    )
    gt_positions = np.asarray(gt_positions)
    pred_positions = np.asarray(pred_positions)
    gt_expressions = np.asarray(gt_expressions)
    pred_expressions = np.asarray(pred_expressions)

    gt_tree = cKDTree(gt_positions)
    pred_tree = cKDTree(pred_positions)

    precisions = []
    recalls = []
    f1s = []

    # sampling
    if sample is not None:
        n = int(len(gt_positions) * sample / 100)
        idx = np.random.choice(len(gt_positions), size=n, replace=False)
        samples = gt_positions[idx]
    else:
        samples = gt_positions

    for i, pos in enumerate(tqdm(samples, desc="spots")):
        # find neighbors
        if k > 0:
            _, gt_idx = gt_tree.query(pos, k=k + 1)
            _, pred_idx = pred_tree.query(pos, k=k + 1)
            gt_nbrs = gt_idx[1:]
            pred_nbrs = pred_idx[1:]
        else:
            gt_nbrs = gt_tree.query_ball_point(pos, radius)
            pred_nbrs = pred_tree.query_ball_point(pos, radius)

        # sum expressions over neighbors
        gt_sum = np.sum(gt_expressions[gt_nbrs], axis=0)
        pred_sum = np.sum(pred_expressions[pred_nbrs], axis=0)

        # compute precision: TP / (predicted positives)
        pred_pos = pred_sum > 0
        if pred_pos.sum() > 0:
            true_pos = np.logical_and(pred_pos, gt_sum > 0).sum()
            precisions.append(true_pos / pred_pos.sum())
            recalls.append(
                true_pos / (gt_sum > 0).sum() if (gt_sum > 0).sum() > 0 else 0.0
            )
            f1s.append(
                2 * precisions[-1] * recalls[-1] / (precisions[-1] + recalls[-1])
                if (precisions[-1] + recalls[-1]) > 0
                else 0.0
            )
        else:
            precisions.append(0.0)
            recalls.append(0.0)
            f1s.append(0.0)

        if i and i % 10000 == 0:
            print(f"Processed {i} spots...")

    if return_list:
        return f1s
    return (
        float(np.mean(f1s)) if f1s else 0.0,
        float(np.mean(precisions)) if precisions else 0.0,
        float(np.mean(recalls)) if recalls else 0.0,
    )
