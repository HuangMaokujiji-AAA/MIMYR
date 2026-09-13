import torch
import torch.nn as nn
import anndata as ad
import numpy as np
from shapely.geometry import Point, Polygon
from scipy.spatial import Delaunay
from shapely.geometry import Polygon, LineString
from shapely.ops import polygonize, unary_union
from scipy.spatial import Delaunay
import numpy as np


def manual_alpha_shape_polygon(coords: np.ndarray, alpha: float) -> Polygon:
    tri = Delaunay(coords)

    def triangle_circumradius(a, b, c):
        A = np.linalg.norm(b - c)
        B = np.linalg.norm(c - a)
        C = np.linalg.norm(a - b)
        s = (A + B + C) / 2
        area = np.sqrt(max(s * (s - A) * (s - B) * (s - C), 1e-12))
        return (A * B * C) / (4.0 * area)

    edges = set()
    for simplex in tri.simplices:
        pts = coords[simplex]
        r = triangle_circumradius(pts[0], pts[1], pts[2])
        if r < 1.0 / alpha:
            for i, j in [(0, 1), (1, 2), (2, 0)]:
                edge = tuple(sorted((simplex[i], simplex[j])))
                if edge in edges:
                    edges.remove(edge)
                else:
                    edges.add(edge)

    edge_lines = [LineString([coords[i], coords[j]]) for i, j in edges]
    boundary = unary_union(edge_lines)
    polygons = list(polygonize(boundary))

    if len(polygons) == 0:
        raise ValueError("Alpha shape produced no polygons.")
    elif len(polygons) == 1:
        return polygons[0]
    else:
        # Return the largest polygon (in area)
        return max(polygons, key=lambda p: p.area)


class KDEMixture(nn.Module):
    def __init__(self, spatial_data, bandwidth=1, z_factor=1.0):
        super(KDEMixture, self).__init__()
        self.N, self.d = spatial_data.shape
        if not isinstance(spatial_data, torch.Tensor):
            spatial_data = torch.tensor(spatial_data).float()
        self.spatial_data = nn.Parameter(spatial_data, requires_grad=True)

        self.bandwidth = nn.Parameter(
            torch.tensor(bandwidth, device=spatial_data.device), requires_grad=False
        )
        self.z_factor = nn.Parameter(
            torch.tensor(z_factor, device=spatial_data.device), requires_grad=False
        )
        self.register_buffer(
            "weights", torch.ones(self.N, device=spatial_data.device) / self.N
        )

    def forward(self, points, sample_frac=1.0):
        device = points.device
        if sample_frac < 1.0:
            k = max(1, int(self.N * sample_frac))
            idx = torch.randperm(self.N, device=self.spatial_data.device)[:k]
            spatial_subset = self.spatial_data[idx].to(device)
            weight_subset = self.weights[idx].to(device)
            weight_subset /= weight_subset.sum()
        else:
            spatial_subset = self.spatial_data.to(device)
            weight_subset = self.weights.to(device)

        # Scale z-axis for sharper or flatter KDE
        points_scaled = points
        subset_scaled = spatial_subset.clone()
        if self.d >= 3:
            points_scaled[:, 2] *= self.z_factor
            subset_scaled[:, 2] *= self.z_factor

        distances = torch.cdist(points_scaled, subset_scaled)
        kernels = torch.exp(-(distances**2) / (2 * self.bandwidth.to(device) ** 2))
        return torch.sum(weight_subset * kernels, dim=-1)

    def forward(
        self,
        points,
        sample_frac: float = 1.0,
        eps: float = 1e-12,
        p_bs: int = 2 * 2048,  # points batch size
        s_bs: int = 2 * 4096,  # subset batch size
        normalize_weights: bool = True,
        rescale_grad_to_original_coords: bool = True,
    ):
        device, dtype = points.device, points.dtype

        # ----- choose subset -----
        if sample_frac < 1.0:
            k = max(1, int(self.N * sample_frac))
            idx = torch.randperm(self.N, device=self.spatial_data.device)[:k]
            spatial_subset = self.spatial_data[idx].to(device=device, dtype=dtype)
            weight_subset = self.weights[idx].to(device=device, dtype=dtype)
        else:
            spatial_subset = self.spatial_data.to(device=device, dtype=dtype)
            weight_subset = self.weights.to(device=device, dtype=dtype)

        if normalize_weights:
            weight_subset = weight_subset / (weight_subset.sum() + eps)

        # ----- scaling (z-axis) -----
        points_scaled = points.clone()
        subset_scaled = spatial_subset.clone()
        if self.d >= 3:
            points_scaled[:, 2] *= self.z_factor
            subset_scaled[:, 2] *= self.z_factor

        P = points_scaled.shape[0]
        K = subset_scaled.shape[0]
        bw2 = (self.bandwidth**2).to(device=device, dtype=dtype)
        inv_bw2 = 1.0 / bw2

        # outputs
        fvals = torch.zeros(P, device=device, dtype=dtype)
        grad = torch.zeros(P, self.d, device=device, dtype=dtype)

        # ----- two-axis tiling -----
        for p_start in range(0, P, p_bs):
            p_end = min(p_start + p_bs, P)
            p_chunk = points_scaled[p_start:p_end]  # (bp, d)

            f_chunk = torch.zeros(p_end - p_start, device=device, dtype=dtype)
            g_chunk = torch.zeros(p_end - p_start, self.d, device=device, dtype=dtype)

            for s_start in range(0, K, s_bs):
                s_end = min(s_start + s_bs, K)
                s_chunk = subset_scaled[s_start:s_end]  # (bs, d)
                w_chunk = weight_subset[s_start:s_end]  # (bs,)

                # (bp, bs, d)
                diffs = p_chunk[:, None, :] - s_chunk[None, :, :]
                # (bp, bs)
                sq_dist = (diffs**2).sum(dim=-1)

                # (bp, bs)
                kernels = torch.exp(-sq_dist * (0.5 * inv_bw2))

                # accumulate f
                # (bp,)
                f_chunk = f_chunk + (kernels * w_chunk[None, :]).sum(dim=1)

                # accumulate grad
                # -(diffs / bw^2) * kernel * w
                # (bp, bs, d)
                contrib = (
                    -(diffs * inv_bw2) * kernels[..., None] * w_chunk[None, :, None]
                )
                # (bp, d)
                g_chunk = g_chunk + contrib.sum(dim=1)

                # free temps ASAP
                del diffs, sq_dist, kernels, contrib

            # write back
            fvals[p_start:p_end] = f_chunk
            grad[p_start:p_end] = g_chunk

            del f_chunk, g_chunk

        # gradient of log f
        log_grad = grad / (fvals[:, None] + eps)

        # If you scaled z during KDE, adjust gradient back to ORIGINAL coords:
        # p_scaled = diag(1,1,z_factor) * p  => grad_p = A^T * grad_p_scaled
        if self.d >= 3 and rescale_grad_to_original_coords:
            grad[:, 2] *= self.z_factor
            log_grad[:, 2] *= self.z_factor

        return fvals, log_grad

    def log_prob(self, points, sample_frac=1.0):
        density = self.forward(points, sample_frac=sample_frac)
        return torch.log(density + 1e-10)

    def prob(self, points, sample_frac=1.0):
        return self.forward(points, sample_frac=sample_frac)

    def sample(self, num_samples):
        indices = torch.randint(
            0, self.N, (num_samples,), device=self.spatial_data.device
        )
        noise = (
            torch.randn(num_samples, self.d, device=self.spatial_data.device)
            * self.bandwidth
        )

        if self.d >= 3:
            noise[:, 2] /= self.z_factor  # Less noise in z → sharper

        return self.spatial_data[indices] + noise

    def sample_conditionally(self, conditioning_points, num_samples):
        """
        Sample conditionally based on closeness to mean z-coordinate of conditioning points.

        Parameters:
        - conditioning_points: numpy array of shape (n, 3)
        - num_samples: int, number of samples to generate

        Returns:
        - torch.Tensor of shape (num_samples, 3)
        """
        if self.d < 3:
            raise ValueError("Conditional sampling expects 3D data.")

        # Compute mean z (apply z_factor to match model's KDE scaling)
        z_mean = np.mean(conditioning_points[:, 2]) * float(self.z_factor.cpu())

        # Get spatial data in CPU numpy for selection
        spatial_np = self.spatial_data.detach().cpu().numpy()
        spatial_z_scaled = spatial_np[:, 2] * float(self.z_factor.cpu())

        # Compute absolute distance from mean z
        z_dists = np.abs(spatial_z_scaled - z_mean)

        # Pick 2n indices with smallest z difference
        n = conditioning_points.shape[0]
        topk_idxs = np.argpartition(z_dists, 4 * n)[: 4 * n]

        # Convert to tensor
        candidate_centers = self.spatial_data[topk_idxs]

        # KDE sampling from selected subset
        k = candidate_centers.shape[0]
        chosen_idxs = torch.randint(
            0, k, (num_samples,), device=self.spatial_data.device
        )
        base_points = candidate_centers[chosen_idxs]

        noise = (
            torch.randn(num_samples, self.d, device=self.spatial_data.device)
            * self.bandwidth
        )

        if self.d >= 3:
            noise[:, 2] /= self.z_factor  # same sharpness rule

        return base_points + noise


class KDEModelForGuidance(nn.Module):
    def __init__(
        self, slices, z_posn=None, pin_key="region", bandwidth=0.001, z_factor=1
    ):
        super(KDEModelForGuidance, self).__init__()
        self.bandwidth = bandwidth
        self.z_factor = z_factor
        try:
            self.xyz = ad.concat(slices).obsm["aligned_spatial"]
        except:
            self.slices = slices
            self.pin_key = pin_key
            self.z_posns = z_posn
            concatenated_slices = ad.concat(self.slices)
            z_posns_all = []
            z_posns = self.z_posns
            for i, slice in enumerate(self.slices):
                slice.obsm["3d_spatial"] = np.concatenate(
                    [
                        slice.obsm["aligned_spatial"],
                        np.array([[z_posns[i]]] * len(slice)),
                    ],
                    1,
                )
                z_posns_all.append(torch.tensor([z_posns[i]] * len(slice)).cuda())
            xy = torch.tensor(concatenated_slices.obsm["aligned_spatial"]).cuda()
            z_posns = torch.cat(z_posns_all) * self.z_factor
            self.xyz = torch.cat([xy, z_posns.unsqueeze(-1)], 1)

    def fit(self):
        self.model = KDEMixture(
            self.xyz, bandwidth=self.bandwidth, z_factor=self.z_factor
        )

    def forward(self, x, sample_frac=1.0):
        return self.model(x, sample_frac=sample_frac)

    def sample(self, z, size=1000):
        samples = []
        while len(samples) < size:
            data_points = self.model.sample(100000)
            selection = torch.abs(data_points[:, 2] - z) < self.z_factor / 2
            samples += data_points[selection].detach().cpu().numpy().tolist()
        return np.array(samples)

    def sample_slice_conditionally(
        self, slice, size=1000, alpha=1.0, interior=False, dist_cutoff=0.5
    ):
        coords_3d = slice.obsm["aligned_spatial"]
        coords_2d = coords_3d[:, :2]

        # Step 1: Fit a plane to the 3D coordinates of the slice
        xyz = torch.tensor(coords_3d, dtype=torch.float32).cuda()
        xyz_mean = xyz.mean(dim=0, keepdim=True)
        centered = xyz - xyz_mean
        _, _, V = torch.pca_lowrank(centered)
        normal = V[:, -1]  # Last component is the normal vector

        # Polygon for filtering 2D positions
        polygon = manual_alpha_shape_polygon(coords_2d, alpha=alpha)
        slice_kde = KDEMixture(xyz, bandwidth=self.bandwidth)

        accepted = []
        while len(accepted) < size:
            # candidates = self.model.sample_conditionally(xyz.cpu().numpy(), 100000)
            candidates = self.model.sample(100000)
            candidates = torch.tensor(candidates, dtype=torch.float32).cuda()

            # Step 2: Compute orthogonal distances to plane
            diffs = candidates - xyz_mean  # (N, 3)
            dists = torch.abs(
                torch.matmul(diffs, normal)
            )  # scalar projection on normal vector

            mask = dists < dist_cutoff
            candidates = candidates[mask]
            if len(candidates) == 0:
                continue

            # Step 3: 2D polygon check
            candidates_2d = candidates[:, :2].detach().cpu().numpy()
            if interior is not None:
                outside_mask = np.array(
                    [polygon.contains(Point(pt)) == interior for pt in candidates_2d]
                )
                candidates = candidates[outside_mask]

            if len(candidates) == 0:
                continue

            accepted = accepted + candidates.detach().cpu().numpy().tolist()

        return np.array(accepted)[:size]

    # def sample_slice_conditionally(self, slice, size=1000, alpha=1.0, interior=False, dist_cutoff=0.1):
    #     coords_2d = slice.obsm["aligned_spatial"]

    #     z = slice.obsm["aligned_spatial"][:,2].mean().item()

    #     # Get shapely polygon
    #     polygon = manual_alpha_shape_polygon(coords_2d[:,:2], alpha=alpha)

    #     slice_xyz = torch.tensor(slice.obsm["aligned_spatial"], dtype=torch.float32).cuda()
    #     # Build KDE for the slice
    #     slice_kde = KDEMixture(slice_xyz, bandwidth=self.bandwidth)

    #     accepted = []
    #     while len(accepted) < size:
    #         candidates = self.model.sample_conditionally(slice_xyz.cpu().numpy(),100000)

    #         z_mask = torch.abs(candidates[:, 2] - z) < dist_cutoff
    #         candidates = candidates[z_mask]
    #         if len(candidates) == 0:
    #             continue

    #         candidates_2d = candidates[:, :2].detach().cpu().numpy()
    #         outside_mask = np.array([polygon.contains(Point(pt))==interior for pt in candidates_2d])
    #         candidates = candidates[outside_mask]

    #         if len(candidates) == 0:
    #             continue

    #         accepted=accepted+candidates.detach().cpu().numpy().tolist()

    #     return np.array(accepted)[:size]


# import torchdiffeq
# import torch

# t_span = torch.tensor([0, 1], dtype=torch.float32).to("cuda")
# slice=bio_model.pre_slices[0]
# og_coords = slice.obsm["spatial"]

# # Convert numpy array to torch tensor and move to GPU
# transformed_coords = torch.tensor(og_coords, dtype=torch.float32).to("cuda")

# transformed_coords=density_model.alignment_velocity_fields[0].aff(transformed_coords)+torchdiffeq.odeint(density_model.alignment_velocity_fields[0], transformed_coords, t_span, method="rk4")[-1]
# nc=transformed_coords.cpu().detach().numpy()

# plt.scatter(nc[:,0],nc[:,1])
