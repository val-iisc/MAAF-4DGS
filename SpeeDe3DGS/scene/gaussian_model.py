#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import math
import torch

# Try to load the compiled CUDA extension; fall back to pure-PyTorch if absent.
# JIT-load via torch.utils.cpp_extension.load() avoids LD_LIBRARY_PATH issues
# (PyTorch manages its own dlopen path when using this API).
_PARZEN_CUDA_AVAILABLE = False
_parzen_cuda = None
try:
    import os as _os
    from torch.utils.cpp_extension import load as _cpp_load
    _ext_dir = _os.path.join(_os.path.dirname(__file__), '..', 'submodules', 'parzen-filter')
    _parzen_cuda = _cpp_load(
        name='parzen_filter_C',
        sources=[
            _os.path.join(_ext_dir, 'parzen_filter_kernel.cu'),
            _os.path.join(_ext_dir, 'ext.cpp'),
        ],
        extra_cuda_cflags=['-O3', '--use_fast_math'],
        extra_cflags=['-O3'],
        verbose=False,
    )
    _PARZEN_CUDA_AVAILABLE = True
except Exception:
    pass
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud, fov2focal
from utils.general_utils import strip_symmetric, build_scaling_rotation


# ── Parzen kernel functions for 3D filter estimation ─────────────────────────

def gaussian_kernel_filter(u):
    """Standard Gaussian kernel."""
    return (1.0 / math.sqrt(2 * math.pi)) * torch.exp(-0.5 * u ** 2)


def weibull_kernel_filter(u, k=10, lam=1.0):
    """Weibull kernel — defined only for u >= 0, emphasises larger values.
    With k=10 the mode sits at ~0.9*lam, giving a right-skewed response
    that naturally approximates max-like behaviour on positive f/d data.
    """
    mask = (u >= 0).float()
    safe_u = u.clamp(min=0)
    return (k / lam) * ((safe_u / lam) ** (k - 1)) * torch.exp(-((safe_u / lam) ** k)) * mask


class GaussianModel:
    def __init__(self, sh_degree: int):

        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree

        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)

        self.optimizer = None

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    # ── 3D Filter (MipSplatting-style anti-aliasing) ──────────────────────────

    def quantize_data_final(self, input_t, input_y):
        num_gaussians = input_t.shape[0]
        bins = torch.linspace(0, 1.0, steps=101, device="cuda")
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_centers = bin_centers.unsqueeze(0).expand(num_gaussians, -1)
        bin_indices = torch.bucketize(input_t.contiguous(), bins, right=True)
        valid_mask = (bin_indices >= 0) & (bin_indices < 100)
        max_y = torch.full((num_gaussians, 100), float('-inf'), device="cuda")
        max_y.scatter_reduce_(1, bin_indices * valid_mask, input_y * valid_mask, reduce="amax", include_self=False)
        max_y[max_y == float('-inf')] = 0  # unobserved bins → 0 (safe: inversion returns r_k=0 → no inflation)
        return bin_centers, max_y

    @torch.no_grad()
    def _infer_value_histogram(self, timestamp):
        """Hard-bin histogram lookup: returns max(f/d) in the bin closest to timestamp."""
        num_gaussians = self.bin_centers.shape[0]
        if num_gaussians == 0:
            return torch.zeros(0, 1, device="cuda")
        timestamp_tensor = torch.full((num_gaussians,), timestamp, device="cuda")
        step = 0.01
        lower_bound = (timestamp_tensor // step) * step
        center_val = lower_bound + step / 2
        bin_index = torch.bucketize(center_val, self.bin_centers[0]) - 1
        bin_index = torch.clamp(bin_index, 0, 99)
        return self.max_y[torch.arange(num_gaussians), bin_index].unsqueeze(1)

    @torch.no_grad()
    def _precompute_parzen_data(self):
        """Precompute all parzen data from filter_3D — called once after compute_3D_filter / load_filter.

        Stores:
          parzen_fd_norm  [N, K]   depth observations normalised to [0, 1] per Gaussian
          parzen_fd_max   [N, 1]   per-Gaussian depth scale (to rescale nu back to f/d units)
          parzen_ts_data  [N, K]   raw timestamp observations (< 0 = empty slot)
          parzen_x_grid   [D]      shared depth query grid in [0, 1]
          parzen_t_grid   [T]      shared timestamp query grid in [0, 1]
          parzen_density  [N, D, T] joint 2D KDE — looked up at render time, no recompute needed
        """
        fd_data = self.filter_3D[..., 1].contiguous()   # [N, K]
        ts_data = self.filter_3D[..., 0].contiguous()   # [N, K]
        valid   = (ts_data >= 0).float()

        fd_max = (fd_data * valid).max(dim=1, keepdim=True).values.clamp(min=1e-6)
        self.parzen_fd_norm = (fd_data / fd_max).contiguous()
        self.parzen_fd_max  = fd_max
        self.parzen_ts_data = ts_data

        N   = fd_data.shape[0]
        dev = fd_data.device
        D   = getattr(self, 'parzen_D',   10)
        T   = getattr(self, 'parzen_T',  100)
        h_x = getattr(self, 'parzen_h_x', 0.3)
        h_t = getattr(self, 'parzen_h_t', 0.1)
        wk  = float(getattr(self, 'parzen_wk', 10))

        self.parzen_x_grid = torch.linspace(0, 1, D, device=dev)   # [D]
        self.parzen_t_grid = torch.linspace(0, 1, T, device=dev)   # [T]

        if _PARZEN_CUDA_AVAILABLE and D in (5, 10, 20):
            self.parzen_density = _parzen_cuda.parzen_density(
                self.parzen_fd_norm, self.parzen_ts_data,
                self.parzen_x_grid, self.parzen_t_grid,
                h_x, h_t, wk, 1.0,
            )  # [N, D, T]  — fused CUDA kernel, no large intermediates
        else:
            self.parzen_density = self._estimate_parzen_density(
                self.parzen_fd_norm, self.parzen_ts_data,
                self.parzen_x_grid, self.parzen_t_grid,
                h_x, h_t, wk,
            )  # [N, D, T]  — chunked PyTorch fallback

    @torch.no_grad()
    def _estimate_parzen_density(self, fd_norm, ts_data, x_grid_1d, t_grid_1d,
                                  h_x, h_t, wk, chunk=5000):
        """Build 2D joint Parzen KDE density [N, D, T].

        density[n, d, t] = Σ_k  weibull((x_grid[d] - fd[n,k]) / h_x)
                                × gauss((t_grid[t]  - ts[n,k]) / h_t)
                                × valid[n,k]

        Uses bmm(x_kern [nc,D,K], t_kern.T [nc,K,T]) → [nc,D,T] to avoid
        ever materialising the [N, D, T, K] joint intermediate.

        Args:
            fd_norm    [N, K]  normalised depth observations
            ts_data    [N, K]  timestamp observations (< 0 = empty slot)
            x_grid_1d  [D]     shared depth query grid
            t_grid_1d  [T]     shared timestamp query grid
        Returns:
            density    [N, D, T]
        """
        N, K = fd_norm.shape
        D    = x_grid_1d.shape[0]
        T    = t_grid_1d.shape[0]
        dev  = fd_norm.device

        density = torch.zeros(N, D, T, device=dev)

        for start in range(0, N, chunk):
            end = min(start + chunk, N)

            fdn_c   = fd_norm[start:end]           # [nc, K]
            ts_c    = ts_data[start:end]            # [nc, K]
            valid_c = (ts_c >= 0).float()           # [nc, K]

            # Depth kernel: [nc, D, K]
            x_diff = (x_grid_1d.view(1, D, 1) - fdn_c.unsqueeze(1)) / h_x
            x_kern = weibull_kernel_filter(x_diff, k=wk)                    # [nc, D, K]

            # Temporal kernel: [nc, T, K], invalid slots masked out
            t_diff = (t_grid_1d.view(1, T, 1) - ts_c.unsqueeze(1)) / h_t
            t_kern = gaussian_kernel_filter(t_diff) * valid_c.unsqueeze(1) # [nc, T, K]

            # Joint density via bmm — never materialises [nc, D, T, K]
            density[start:end] = torch.bmm(x_kern, t_kern.transpose(1, 2)) / (2 * h_x * h_t)

        return density  # [N, D, T]

    @torch.no_grad()
    def _infer_from_density(self, timestamp):
        """Lookup modal depth from the precomputed [N, D, T] joint density.

        Finds the nearest t in parzen_t_grid (shared across Gaussians → scalar index),
        slices density → [N, D], argmax → nu [N, 1].
        """
        if not hasattr(self, 'parzen_density') or self.parzen_density.shape[0] == 0:
            return torch.zeros(0, 1, device="cuda")

        N   = self.parzen_density.shape[0]
        dev = self.parzen_density.device

        # t_grid is shared → t_idx is scalar, same for every Gaussian
        t_idx = torch.argmin(torch.abs(self.parzen_t_grid - timestamp)).item()

        density_at_t = self.parzen_density[:, :, t_idx]                    # [N, D]
        x_idx        = torch.argmax(density_at_t, dim=1)                   # [N]
        nu_norm      = self.parzen_x_grid[x_idx]                           # [N]
        return (nu_norm * self.parzen_fd_max.squeeze(1)).unsqueeze(1)      # [N, 1]

    @torch.no_grad()
    def infer_value(self, timestamp):
        """Estimate the per-Gaussian 3D filter size (f/d) for a given timestamp.

        Dispatch:
          parzen    → _infer_from_density  (O(1) lookup into precomputed [N,D,T])
          histogram → _infer_value_histogram
        The [N, D, T] density is built once in _precompute_parzen_data using
        the CUDA kernel when available, PyTorch otherwise.
        """
        if getattr(self, 'filter_estimate', 'histogram') == 'parzen':
            return self._infer_from_density(timestamp)
        return self._infer_value_histogram(timestamp)

    def _compute_filter_var(self, nu):
        """Shared helper: convert nu [N,1] → filter_var [N,1] for scale inflation."""
        lam = getattr(self, 'filter_lambda', 1.0)
        r_k = torch.where(nu > 0, 1.0 / nu, torch.zeros_like(nu))
        return lam * torch.square(r_k * (0.2 ** 0.5))

    def get_scaling_with_3D_filter(self, scales, nu):
        """Inflate scales using a pre-computed nu [N, 1] (avoids redundant KDE call)."""
        with torch.no_grad():
            filter_var = self._compute_filter_var(nu)
        return torch.sqrt(torch.square(scales) + filter_var)

    def get_opacity_with_3D_filter(self, opacity, scales, nu):
        """Compensate opacity using a pre-computed nu [N, 1] (avoids redundant KDE call)."""
        with torch.no_grad():
            filter_var = self._compute_filter_var(nu)
            scales_sq  = torch.square(scales.detach())
            det1 = scales_sq.prod(dim=1)
            det2 = (scales_sq + filter_var).prod(dim=1)
            eps  = 1e-12
            coef = torch.sqrt((det1 + eps) / (det2 + eps))
        return opacity * coef[..., None]

    @torch.no_grad()
    def compute_3D_filter(self, cameras, deform):
        print("Computing 3D filter")
        from collections import defaultdict
        xyz = self.get_xyz
        num_points = xyz.shape[0]
        device = xyz.device

        max_timestamps = 300
        timestamp_data = torch.full((num_points, max_timestamps, 2), -1.0, device=device)

        # Pre-compute global max focal length across all cameras
        max_focal_length = max(
            max(fov2focal(cam.FoVx, cam.image_width), fov2focal(cam.FoVy, cam.image_height))
            for cam in cameras
        )

        # Group cameras by unique timestamps to reuse deformation per timestamp
        timestamp_to_cameras = defaultdict(list)
        for camera in cameras:
            ts = camera.fid.item() if torch.is_tensor(camera.fid) else float(camera.fid)
            timestamp_to_cameras[ts].append(camera)

        for ts, cam_group in timestamp_to_cameras.items():
            fid = torch.tensor([[ts]], device=device)
            time_input = fid.expand(num_points, -1)
            d_xyz, _, _ = deform.step(xyz.detach(), time_input)
            new_xyz = xyz + d_xyz

            for camera in cam_group:
                R = torch.tensor(camera.R, device=device, dtype=torch.float32)
                T = torch.tensor(camera.T, device=device, dtype=torch.float32)
                xyz_cam = new_xyz @ R + T[None, :]

                z = xyz_cam[:, 2]
                valid_depth = z > 0.2

                focalx = fov2focal(camera.FoVx, camera.image_width)
                focaly = fov2focal(camera.FoVy, camera.image_height)
                x = xyz_cam[:, 0] / z * focalx + camera.image_width / 2.0
                y = xyz_cam[:, 1] / z * focaly + camera.image_height / 2.0

                in_screen = (
                    (x >= -0.15 * camera.image_width) & (x <= 1.15 * camera.image_width) &
                    (y >= -0.15 * camera.image_height) & (y <= 1.15 * camera.image_height)
                )
                valid = valid_depth & in_screen

                valid_indices = valid.nonzero().squeeze(-1)
                if valid_indices.numel() > 0:
                    slot_indices = (timestamp_data[valid_indices, :, 0] == -1).int().argmax(dim=1)
                    timestamp_data[valid_indices, slot_indices, 0] = ts
                    timestamp_data[valid_indices, slot_indices, 1] = max_focal_length / z[valid_indices]

        self.filter_3D = timestamp_data
        use_parzen = getattr(self, 'filter_estimate', 'histogram') == 'parzen'
        if use_parzen:
            self._precompute_parzen_data()
            print(f"3D filter computed [parzen]: {timestamp_data.shape[0]} Gaussians, "
                  f"{timestamp_data.shape[1]} slots/Gaussian")
        else:
            self.bin_centers, self.max_y = self.quantize_data_final(
                self.filter_3D[..., 0], self.filter_3D[..., 1]
            )
            print(f"3D filter computed [histogram]: bin_centers {self.bin_centers.shape}, "
                  f"max_y {self.max_y.shape}")

    def save_filter(self, path):
        data = {'filter_estimate': getattr(self, 'filter_estimate', 'histogram')}
        # Persist every hyper-parameter the filter depends on so the saved .pt is
        # self-describing. load_filter rebuilds the parzen density from filter_3D
        # and MUST reuse the train-time hparams (h_x/h_t/D/T/wk), not whatever CLI
        # args/defaults are live at render time; filter_lambda scales the inflation
        # at apply time. Defaults here mirror the getattr() fallbacks used elsewhere.
        data['hparams'] = {
            'parzen_h_x':    getattr(self, 'parzen_h_x', 0.3),
            'parzen_h_t':    getattr(self, 'parzen_h_t', 0.1),
            'parzen_D':      getattr(self, 'parzen_D',   10),
            'parzen_T':      getattr(self, 'parzen_T',  100),
            'parzen_wk':     getattr(self, 'parzen_wk',  10),
            'filter_lambda': getattr(self, 'filter_lambda', 1.0),
        }
        if hasattr(self, 'filter_3D'):
            data['filter_3D'] = self.filter_3D.cpu()
        if hasattr(self, 'bin_centers'):
            data['bin_centers'] = self.bin_centers.cpu()
        if hasattr(self, 'max_y'):
            data['max_y'] = self.max_y.cpu()
        torch.save(data, path)

    def load_filter(self, path):
        data = torch.load(path, map_location='cpu')
        mode = data.get('filter_estimate', 'histogram')
        self.filter_estimate = mode
        # Restore the saved hyper-parameters onto the model BEFORE rebuilding the
        # parzen density, so the reconstruction matches training exactly and takes
        # precedence over any render-time CLI args. Older checkpoints without
        # 'hparams' fall back to whatever is already set (CLI args / getattr defaults).
        for key, val in data.get('hparams', {}).items():
            setattr(self, key, val)
        if 'filter_3D' in data:
            self.filter_3D = data['filter_3D'].cuda()
        if 'bin_centers' in data:
            self.bin_centers = data['bin_centers'].cuda()
        if 'max_y' in data:
            self.max_y = data['max_y'].cuda()
        if mode == 'parzen' and hasattr(self, 'filter_3D'):
            self._precompute_parzen_data()
        print(f"Loaded 3D filter [{mode}]: {self.filter_3D.shape[0]} Gaussians")

    # ─────────────────────────────────────────────────────────────────────────

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = 5
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        self.spatial_lr_scale = 5

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr * self.spatial_lr_scale, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, og_number_points=-1):
        self.og_number_points = og_number_points
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        print("Number of points from existing .ply file : ", self.get_xyz.shape[0])

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                              new_rotation):
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values > self.percent_dense * scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values <= self.percent_dense * scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                                   new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def prune_gaussians(self, percent, import_score: list): # NEW
        num_before = self.get_xyz.shape[0]
        sorted_tensor, _ = torch.sort(import_score, dim=0)
        index_nth_percentile = int(percent * (sorted_tensor.shape[0] - 1))
        value_nth_percentile = sorted_tensor[index_nth_percentile]
        prune_mask = (import_score <= value_nth_percentile).squeeze()
        self.prune_points(prune_mask)
        num_after = self.get_xyz.shape[0]
        num_removed = num_before - num_after
        print(f"\n[Prune] Gaussians before: {num_before}, after: {num_after}, removed: {num_removed}")
        return prune_mask

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1,
                                                             keepdim=True)
        self.denom[update_filter] += 1
