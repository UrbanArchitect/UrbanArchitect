import torch
import torch.nn as nn
import torch.nn.functional as F
from hashgrid.gridencoder import GridEncoder
from hashgrid import coord, math, render, stepfun, train_utils
import numpy as np
from torch_scatter import segment_coo

class PositionEncoding(nn.Module):
    '''
    Position Encoding for xyz/directions
    Args:
        in_channels: number of input channels (typically 3)
        N_freqs: maximum frequency
        logscale: if True, use log scale for frequencies
    Inputs:
        x: (batch_size, in_channels)
    '''
    def __init__(self, in_channels, N_freqs, logscale=True):
        super(PositionEncoding, self).__init__()

        self.in_channels = in_channels
        self.N_freqs = N_freqs

        self.funcs = [torch.sin, torch.cos]

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, x):
        
        out = [x]

        for freq in self.freq_bands:
            for func in self.funcs:
                out.append(func(x * freq))
        
        out = torch.cat(out, -1)
        return out

class MLPs(nn.Module):
    def __init__(self, num_channels, width, out_channels, num_layers=2):
        super(MLPs, self).__init__()

        layers = []
        for idx in range(num_layers):
            if idx == 0:
                layers.append(nn.Linear(num_channels, width))
            else:
                layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())

        self.mlp = nn.Sequential(*layers)
        self.out_layer = nn.Linear(width, out_channels)
    
    def forward(self, x):
        x = self.mlp(x)
        return self.out_layer(x)

def set_kwargs(self, kwargs):
    for k, v in kwargs.items():
        setattr(self, k, v)

class ScalableHashGrid(nn.Module):
    """A mip-Nerf360 model containing all MLPs."""
    num_prop_samples: int = 32  # The number of samples for each proposal level.
    num_nerf_samples: int = 32  # The number of samples the final nerf level.
    num_levels: int = 3  # The number of sampling levels (3==2 proposals, 1 nerf).
    bg_intensity_range = (1., 1.)  # The range of background colors.
    anneal_slope: float = 10  # Higher = more rapid annealing.
    stop_level_grad: bool = True  # If True, don't backprop across levels.
    use_viewdirs: bool = True  # If True, use view directions as input.
    raydist_fn = "power_transformation"  # The curve used for ray dists.
    single_jitter: bool = True  # If True, jitter whole rays instead of samples.
    dilation_multiplier: float = 0.5  # How much to dilate intervals relatively.
    dilation_bias: float = 0.0025  # How much to dilate intervals absolutely.
    num_glo_features: int = 0  # GLO vector length, disabled if 0.
    num_glo_embeddings: int = 1000  # Upper bound on max number of train images.
    learned_exposure_scaling: bool = False  # Learned exposure scaling (RawNeRF).
    near_anneal_rate = None  # How fast to anneal in near bound.
    near_anneal_init: float = 0.95  # Where to initialize near bound (in [0, 1]).
    single_mlp: bool = False  # Use the NerfMLP for all rounds of sampling.
    distinct_prop: bool = True  # Use the NerfMLP for all rounds of sampling.
    resample_padding: float = 0.0  # Dirichlet/alpha "padding" on the histogram.
    opaque_background: bool = False  # If true, make the background opaque.
    power_lambda: float = -1.5
    std_scale: float = 0.5
    prop_desired_grid_size = [512, 2048]
    def __init__(self, config, device, obj_transforms, **kwargs):
        super().__init__()
        set_kwargs(self, kwargs)

        self.config = config
        self.device = device

        self.grid_scale = config.grid_scale
        self.grid_centers = {}
        self.uncertainty = nn.ParameterList()
        self.obj_scale = config.obj_scale
        self.obj_log2_T = config.obj_log2_T
        self.use_objviewdirs = config.use_objviewdirs
        self.use_viewdirs = config.use_viewdirs

        self.nerf_mlp_dict = nn.ModuleDict()

        if self.single_mlp:
            self.prop_mlp_dict = self.nerf_mlp_dict
        elif not self.distinct_prop:
            self.prop_mlp_dict = {}
        else:
            for i in range(self.num_levels - 1):
                self.add_module(f'prop_mlp_dict_{i}', nn.ModuleDict())

        self.dir_pos_enc = PositionEncoding(in_channels=3, N_freqs=4)
        self.sky_rep = config.sky_rep
        if self.sky_rep == 'mlp':
            self.sky_mlp = MLPs(num_channels=27, width=64, out_channels=3, num_layers=3)
        else:
            sky_rgb_size = (100,100,100)
            self.sky_grid = nn.Parameter(torch.ones(1,3,*sky_rgb_size))
            self.sky_grid.requires_grad = True
        self.obj_transforms = obj_transforms
        self.add_obj_grid()
        self.gradient_scaling = False
    
    def add_obj_grid(self):
        self.obj_nerf_mlp_list = nn.ModuleList()
        for _ in range(len(self.obj_transforms)):
            self.obj_nerf_mlp_list.append(NerfMLP(num_glo_features=self.num_glo_features,
                                                  num_glo_embeddings=self.num_glo_embeddings,
                                                  grid_log2_hashmap_size=self.obj_log2_T,
                                                  use_viewdirs=self.use_objviewdirs))
    
    def render_sky(self, rays_d):
        viewdirs = rays_d / (torch.norm(rays_d, dim=-1, keepdim=True)+1e-8)
        if self.sky_rep == 'mlp':
            viewdirs_enc = self.dir_pos_enc(viewdirs)
            sky_rgb = self.sky_mlp(viewdirs_enc)
            sky_rgb[...,:3] = torch.sigmoid(sky_rgb[...,:3])
        elif self.sky_rep == 'grid':
            num_dirs = viewdirs.shape[0]
            viewdirs = viewdirs.view(1, 1, 1, -1, 3)
            viewdirs = viewdirs.flip(-1)
            sky_rgb = F.grid_sample(self.sky_grid, viewdirs, align_corners=False)
            sky_rgb = sky_rgb.view(-1, num_dirs)
            sky_rgb = sky_rgb.permute(1,0)
            sky_rgb = torch.clamp(sky_rgb, min=0., max=1.)
        return sky_rgb
    
    def render(self, rays_o, rays_d, near, far, radii, train_frac, rand, empty_bkgd=False):

        if empty_bkgd:
            ray_valid_idx = far.squeeze(-1) < 200.
        else:
            ray_valid_idx = far.squeeze(-1) < 1e10
        
        rays_d_all = rays_d.clone()
        rays_o = rays_o[ray_valid_idx]
        rays_d = rays_d[ray_valid_idx]
        near = near[ray_valid_idx]
        far = far[ray_valid_idx]
        radii = radii[ray_valid_idx]

        if torch.sum(ray_valid_idx) > 0:

            _, s_to_t = coord.construct_ray_warps(self.raydist_fn, near, far, self.power_lambda)

            if self.near_anneal_rate is None:
                init_s_near = 0.
            else:
                init_s_near = np.clip(1 - train_frac / self.near_anneal_rate, 0,
                                    self.near_anneal_init)
            
            init_s_far = 1.
            sdist = torch.cat([
                torch.full_like(near, init_s_near),
                torch.full_like(far, init_s_far)
            ], dim=-1)

            weights = torch.ones_like(near)
            prod_num_samples = 1

            renderings = []
            is_prop = False
            num_rays = rays_o.shape[0]
            num_samples = self.num_prop_samples if is_prop else self.num_nerf_samples

            # Record the product of the number of samples seen so far.
            prod_num_samples *= num_samples

            # Optionally anneal the weights as a function of training iteration.
            if self.anneal_slope > 0:
                # Schlick's bias function, see https://arxiv.org/abs/2010.09714
                bias = lambda x, s: (s * x) / ((s - 1) * x + 1)
                anneal = bias(train_frac, self.anneal_slope)
            else:
                anneal = 1.

            # A slightly more stable way to compute weights**anneal. If the distance
            # between adjacent intervals is zero then its weight is fixed to 0.
            logits_resample = torch.where(
                sdist[..., 1:] > sdist[..., :-1],
                anneal * torch.log(weights + self.resample_padding),
                torch.full_like(sdist[..., :-1], -torch.inf))

            # Draw sampled intervals from each ray's current weights.
            sdist = stepfun.sample_intervals(
                rand,
                sdist,
                logits_resample,
                num_samples,
                single_jitter=self.single_jitter,
                domain=(init_s_near, init_s_far))
            
            # Optimization will usually go nonlinear if you propagate gradients
            # through sampling.
            if self.stop_level_grad:
                sdist = sdist.detach()

            # Convert normalized distances to metric distances.
            tdist = s_to_t(sdist)

            # Cast our rays, by turning our distance intervals into Gaussians.
            means, stds, ts = render.cast_rays(
                tdist,
                rays_o,
                rays_d,
                None,
                radii,
                rand,
                std_scale=self.std_scale)
            
            stds = stds / self.grid_scale
            
            pts = means.mean(2).reshape(-1, 3) # [N, 3]
            pts_h = torch.cat([pts, torch.ones_like(pts[...,:1])], dim=-1)
            means = means.view(means.shape[0] * means.shape[1], -1, 3)
            stds = stds.view(stds.shape[0] * stds.shape[1], -1)

            all_densitys = torch.zeros_like(pts[...,:1])
            all_rgbs = torch.zeros_like(pts[...,:3])
            all_counts = torch.zeros_like(pts[...,0])

            for obj_idx in range(len(self.obj_transforms)):
                obj_transform = self.obj_transforms[obj_idx]
                pts_o = ((torch.inverse(obj_transform) @ pts_h.T).T)[...,:3]
                pts_o = pts_o * 2. / self.obj_scale
                valid_idx = (pts_o[...,0] > -1) & (pts_o[...,0] < 1) \
                    & (pts_o[...,1] > -1) & (pts_o[...,1] < 1) \
                    & (pts_o[...,2] > -1) & (pts_o[...,2] < 1)
                valid_means = means[valid_idx]
                valid_stds = stds[valid_idx]
                all_counts[valid_idx] += 1
                if valid_idx.sum() == 0:
                    continue
                valid_means = valid_means.view(-1, 3)
                valid_means_h = torch.cat([valid_means, torch.ones_like(valid_means[...,:1])], dim=-1)
                valid_means = ((torch.inverse(obj_transform) @ valid_means_h.T).T)[...,:3]
                valid_means = valid_means.view(-1, 6, 3)
                valid_means = valid_means * 2 / self.obj_scale
                mlp_list = self.obj_nerf_mlp_list
                mlp = mlp_list[obj_idx]
                ray_results = mlp(rand, valid_means, valid_stds, None, None, None)
                all_densitys[valid_idx] = all_densitys[valid_idx] + ray_results['density'][...,None]
                all_rgbs[valid_idx] = all_rgbs[valid_idx] + ray_results['rgb']
            
            valid_idx = (all_counts > 0)
            all_densitys[valid_idx] = all_densitys[valid_idx] / all_counts[valid_idx,None]
            all_rgbs[valid_idx] = all_rgbs[valid_idx] / all_counts[valid_idx,None]

            means = means[~valid_idx]
            pts = pts[~valid_idx]
            stds = stds[~valid_idx]
            coords = torch.round(pts / self.grid_scale).long()
            centers = coords * self.grid_scale

            pts_centered = pts - centers
            pts_normalized = (pts_centered / self.grid_scale) * 2.

            pts_normalized[pts_normalized >= 1.] = 0.999999
            pts_normalized[pts_normalized <= -1.] = -0.999999

            means_centered = means - centers[:,None,:]
            means_normalized = (means_centered / self.grid_scale) * 2.

            unique_coords, unique_inverse = torch.unique(coords, dim=0, return_inverse=True) # [M, 3], [M, 1]
            all_pts_idx = torch.arange(pts.shape[0]).to(self.device)
            bkgd_densitys = torch.zeros(pts.shape[0], 1).float().to(self.device)
            bkgd_rgbs = torch.zeros(pts.shape[0], 3).float().to(self.device)

            reset_flag = False

            mlp_dict = self.nerf_mlp_dict

            for unique_idx in range(len(unique_coords)):
                curr_coords = unique_coords[unique_idx]
                map_id = f'x{str(curr_coords[0].item())}y{str(curr_coords[1].item())}z{str(curr_coords[2].item())}'

                curr_pts_idx = all_pts_idx[unique_inverse == unique_idx]
                curr_means_normalized = means_normalized[curr_pts_idx]
                curr_stds = stds[curr_pts_idx]

                if not map_id in mlp_dict:
                    mlp_dict.update({map_id:NerfMLP(num_glo_features=self.num_glo_features,
                                                num_glo_embeddings=self.num_glo_embeddings,
                                                use_viewdirs=self.use_viewdirs).to(self.device)})
                    reset_flag = True
                
                mlp = mlp_dict[map_id]
                ray_results = mlp(rand, curr_means_normalized, curr_stds, None, None, None)

                bkgd_densitys[curr_pts_idx] = ray_results['density'].view(-1, 1)
                bkgd_rgbs[curr_pts_idx] = ray_results['rgb']

            all_densitys[~valid_idx] = bkgd_densitys
            all_rgbs[~valid_idx] = bkgd_rgbs

            all_densitys = all_densitys.view(num_rays, num_samples)
            all_rgbs = all_rgbs.view(num_rays, num_samples, 3)

            ray_results = {
                'density': all_densitys,
                'rgb': all_rgbs
            }
            
            if self.gradient_scaling:
                ray_results['rgb'], ray_results['density'] = train_utils.GradientScaler.apply(
                    ray_results['rgb'], ray_results['density'], ts.mean(dim=-1))

            weights = render.compute_alpha_weights(
                ray_results['density'],
                tdist,
                rays_d,
                opaque_background=self.opaque_background,
            )[0]
            
            bg_rgbs_all = self.render_sky(rays_d_all)
            bg_rgbs = bg_rgbs_all[ray_valid_idx]
            # Render each ray.
            rendering = render.volumetric_rendering(
                ray_results['rgb'],
                weights,
                tdist,
                bg_rgbs,
                far,
                True)
            
            if self.training:
                # Compute the hash decay loss for this level.
                idx = mlp.encoder.idx
                param = mlp.encoder.embeddings
                loss_hash_decay = segment_coo(param ** 2,
                                                idx,
                                                torch.zeros(idx.max() + 1, param.shape[-1], device=param.device),
                                                reduce='mean'
                                                ).mean()
                ray_results['loss_hash_decay'] = loss_hash_decay
            
            rendering['reset_flag'] = reset_flag

            all_idx = torch.arange(rays_d_all.shape[0]).long().to(rays_d_all.device)
            invalid_idx = all_idx[~ray_valid_idx]
            valid_idx = all_idx[ray_valid_idx]

            rendering_new = {}
            rgbs = torch.ones_like(rays_d_all)
            rgbs[valid_idx] = rendering['rgbs']
            rgbs[invalid_idx] = bg_rgbs_all[invalid_idx]
            depths = torch.zeros_like(rays_d_all[...,0])
            depths[valid_idx] = rendering['depths']
            acc = torch.zeros_like(rays_d_all[...,0])
            acc[valid_idx] = rendering['acc']
            rendering_new['rgbs'] = rgbs
            rendering_new['depths'] = depths
            rendering_new['acc'] = acc
            rendering_new['reset_flag'] = rendering['reset_flag']
            rendering_new['sdist'] = sdist.clone()
            rendering_new['weights'] = weights.clone()

            renderings.append(rendering_new)
            ray_results['sdist'] = sdist.clone()
            ray_results['weights'] = weights.clone()
        else:
            renderings = []
            rendering = {}
            rendering['reset_flag'] = False
            bg_rgbs_all = self.render_sky(rays_d_all)
            rgbs = bg_rgbs_all
            depths = torch.zeros_like(rays_d_all[...,0])
            acc = torch.zeros_like(rays_d_all[...,0])
            rendering['rgbs'] = rgbs
            rendering['depths'] = depths
            rendering['acc'] = acc
            renderings.append(rendering)

        return renderings
    
    def load_weights(self, state_dict):
        for key in state_dict.keys():
            if key.split('.')[0] == 'nerf_mlp_dict':
                map_id = key.split('.')[1]
                self.nerf_mlp_dict.update({map_id:NerfMLP(num_glo_features=self.num_glo_features,
                                                  num_glo_embeddings=self.num_glo_embeddings,
                                                  grid_log2_hashmap_size=21,
                                                  use_viewdirs=self.use_viewdirs).to(self.device)})
            elif key.split('.')[0].startswith('prop_mlp_dict'):
                map_id = key.split('.')[1]
                if self.single_mlp:
                    self.get_submodule(key.split('.')[0]).update({map_id:NerfMLP(num_glo_features=self.num_glo_features,
                                                    num_glo_embeddings=self.num_glo_embeddings,
                                                    grid_log2_hashmap_size=19,
                                                    use_viewdirs=self.use_viewdirs).to(self.device)})
                elif not self.distinct_prop:
                    self.get_submodule(key.split('.')[0]).update({map_id:PropMLP(disable_rgb=True, disable_density_normals=True, grid_level_dim=1,
                                                grid_log2_hashmap_size=19).to(self.device)})
                else:
                    i_level = int(key.split('.')[0].split('_')[-1])
                    self.get_submodule(key.split('.')[0]).update({map_id:PropMLP(grid_disired_resolution=self.prop_desired_grid_size[i_level], disable_rgb=True, disable_density_normals=True, 
                                                grid_level_dim=1, grid_log2_hashmap_size=19).to(self.device)})
        
        self.load_state_dict(state_dict, strict=False)

class ZipNeRFMLP(nn.Module):
    """A PosEnc MLP."""
    bottleneck_width: int = 128  # The width of the bottleneck vector.
    net_depth_viewdirs: int = 2  # The depth of the second part of ML.
    net_width_viewdirs: int = 128  # The width of the second part of MLP.
    skip_layer_dir: int = 0  # Add a skip connection to 2nd MLP after Nth layers.
    num_rgb_channels: int = 3  # The number of RGB channels.
    deg_view: int = 4  # Degree of encoding for viewdirs or refdirs.
    bottleneck_noise: float = 0.0  # Std. deviation of noise added to bottleneck.
    density_bias: float = -1.  # Shift added to raw densities pre-activation.
    density_noise: float = 0.  # Standard deviation of noise added to raw density.
    rgb_premultiplier: float = 1.  # Premultiplier on RGB before activation.
    rgb_bias: float = 0.  # The shift added to raw colors pre-activation.
    rgb_padding: float = 0.001  # Padding added to the RGB outputs.
    enable_pred_normals: bool = False  # If True compute predicted normals.
    disable_density_normals: bool = True  # If True don't compute normals.
    disable_rgb: bool = False  # If True don't output RGB.
    warp_fn = 'contract'
    num_glo_features: int = 0  # GLO vector length, disabled if 0.
    num_glo_embeddings: int = 1000  # Upper bound on max number of train images.
    scale_featurization: bool = False
    grid_num_levels: int = 10
    grid_level_interval: int = 2
    grid_level_dim: int = 4
    grid_base_resolution: int = 16
    grid_disired_resolution: int = 8192
    grid_log2_hashmap_size: int = 21
    net_width_glo: int = 128  # The width of the second part of MLP.
    net_depth_glo: int = 2  # The width of the second part of MLP.
    use_viewdirs: bool = True

    def __init__(self, **kwargs):
        super().__init__()
        set_kwargs(self, kwargs)

        def dir_enc_fn(direction, _):
            return coord.pos_enc(
                direction, min_deg=0, max_deg=self.deg_view, append_identity=True)

        self.dir_enc_fn = dir_enc_fn
        dim_dir_enc = self.dir_enc_fn(torch.zeros(1, 3), None).shape[-1]
        self.grid_num_levels = int(
            np.log(self.grid_disired_resolution / self.grid_base_resolution) / np.log(self.grid_level_interval)) + 1
        self.encoder = GridEncoder(input_dim=3,
                                   num_levels=self.grid_num_levels,
                                   level_dim=self.grid_level_dim,
                                   base_resolution=self.grid_base_resolution,
                                   desired_resolution=self.grid_disired_resolution,
                                   log2_hashmap_size=self.grid_log2_hashmap_size,
                                   gridtype='hash',
                                   align_corners=False)
        last_dim = self.encoder.output_dim
        if self.scale_featurization:
            last_dim += self.encoder.num_levels
        self.density_layer = nn.Sequential(nn.Linear(last_dim, 64),
                                           nn.ReLU(),
                                           nn.Linear(64,
                                                     1 if self.disable_rgb else self.bottleneck_width))  # Hardcoded to a single channel.
        last_dim = 1 if self.disable_rgb and not self.enable_pred_normals else self.bottleneck_width
        if self.enable_pred_normals:
            self.normal_layer = nn.Linear(last_dim, 3)

        if not self.disable_rgb:
            if self.bottleneck_width > 0:
                last_dim_rgb = self.bottleneck_width
            else:
                last_dim_rgb = 0
            
            if self.use_viewdirs:
                last_dim_rgb += dim_dir_enc

            if self.num_glo_features > 0:
                last_dim_glo = self.num_glo_features
                for i in range(self.net_depth_glo - 1):
                    self.add_module(f"lin_glo_{i}", nn.Linear(last_dim_glo, self.net_width_glo))
                    last_dim_glo = self.net_width_glo
                self.add_module(f"lin_glo_{self.net_depth_glo - 1}",
                                     nn.Linear(last_dim_glo, self.bottleneck_width * 2))

            input_dim_rgb = last_dim_rgb
            for i in range(self.net_depth_viewdirs):
                lin = nn.Linear(last_dim_rgb, self.net_width_viewdirs)
                torch.nn.init.kaiming_uniform_(lin.weight)
                self.add_module(f"lin_second_stage_{i}", lin)
                last_dim_rgb = self.net_width_viewdirs
                if i == self.skip_layer_dir:
                    last_dim_rgb += input_dim_rgb
            self.rgb_layer = nn.Linear(last_dim_rgb, self.num_rgb_channels)

    def predict_density(self, means, stds, rand=False, no_warp=False):
        """Helper function to output density."""
        # Encode input positions
        if self.warp_fn is not None and not no_warp:
            means, stds = coord.track_linearize(self.warp_fn, means, stds)
            # contract [-2, 2] to [-1, 1]
            bound = 2
            means = means / bound
            stds = stds / bound
        features = self.encoder(means, bound=1).unflatten(-1, (self.encoder.num_levels, -1))
        weights = torch.erf(1 / torch.sqrt(8 * stds[..., None] ** 2 * self.encoder.grid_sizes ** 2))
        features = (features * weights[..., None]).mean(dim=-3).flatten(-2, -1)
        if self.scale_featurization:
            with torch.no_grad():
                vl2mean = segment_coo((self.encoder.embeddings ** 2).sum(-1),
                                      self.encoder.idx,
                                      torch.zeros(self.grid_num_levels, device=weights.device),
                                      self.grid_num_levels,
                                      reduce='mean'
                                      )
            featurized_w = (2 * weights.mean(dim=-2) - 1) * (self.encoder.init_std ** 2 + vl2mean).sqrt()
            features = torch.cat([features, featurized_w], dim=-1)
        x = self.density_layer(features)
        raw_density = x[..., 0]  # Hardcoded to a single channel.
        # Add noise to regularize the density predictions if needed.
        if rand and (self.density_noise > 0):
            raw_density += self.density_noise * torch.randn_like(raw_density)
        return raw_density, x, means.mean(dim=-2)

    def forward(self,
                rand,
                means, stds,
                viewdirs=None,
                imageplane=None,
                glo_vec=None,
                exposure=None,
                no_warp=False):
        """Evaluate the MLP.

        Args:
        rand: if random .
        means: [..., n, 3], coordinate means.
        stds: [..., n], coordinate stds.
        viewdirs: [..., 3], if not None, this variable will
            be part of the input to the second part of the MLP concatenated with the
            output vector of the first part of the MLP. If None, only the first part
            of the MLP will be used with input x. In the original paper, this
            variable is the view direction.
        imageplane:[batch, 2], xy image plane coordinates
            for each ray in the batch. Useful for image plane operations such as a
            learned vignette mapping.
        glo_vec: [..., num_glo_features], The GLO vector for each ray.
        exposure: [..., 1], exposure value (shutter_speed * ISO) for each ray.

        Returns:
        rgb: [..., num_rgb_channels].
        density: [...].
        """
        raw_density, x, means_contract = self.predict_density(means, stds, rand=rand, no_warp=no_warp)
        raw_grad_density = None
        grad_pred = None

        # Apply bias and activation to raw density
        density = F.softplus(raw_density + self.density_bias)

        if self.disable_rgb:
            rgb = torch.zeros(density.shape + (3,), device=density.device)
        else:
            if viewdirs is not None:
                # Output of the first part of MLP.
                if self.bottleneck_width > 0:
                    bottleneck = x
                    # Add bottleneck noise.
                    if rand and (self.bottleneck_noise > 0):
                        bottleneck += self.bottleneck_noise * torch.randn_like(bottleneck)

                    # Append GLO vector if used.
                    if glo_vec is not None:
                        for i in range(self.net_depth_glo):
                            glo_vec = self.get_submodule(f"lin_glo_{i}")(glo_vec)
                            if i != self.net_depth_glo - 1:
                                glo_vec = F.relu(glo_vec)
                        glo_vec = torch.broadcast_to(glo_vec[..., None, :],
                                                     bottleneck.shape[:-1] + glo_vec.shape[-1:])
                        scale, shift = glo_vec.chunk(2, dim=-1)
                        bottleneck = bottleneck * torch.exp(scale) + shift

                    x = [bottleneck]
                else:
                    x = []

                # Encode view (or reflection) directions.
                if self.use_viewdirs:
                    dir_enc = self.dir_enc_fn(viewdirs, roughness)
                    dir_enc = torch.broadcast_to(
                        dir_enc[..., None, :],
                        bottleneck.shape[:-1] + (dir_enc.shape[-1],))
                else:
                    pass

                # Append view (or reflection) direction encoding to bottleneck vector.
                if self.use_viewdirs:
                    x.append(dir_enc)

                # Concatenate bottleneck, directional encoding, and GLO.
                x = torch.cat(x, dim=-1)
                # Output of the second part of MLP.
                inputs = x
                for i in range(self.net_depth_viewdirs):
                    x = self.get_submodule(f"lin_second_stage_{i}")(x)
                    x = F.relu(x)
                    if i == self.skip_layer_dir:
                        x = torch.cat([x, inputs], dim=-1)
            
            rgb = torch.sigmoid(self.rgb_premultiplier *
                                self.rgb_layer(x) +
                                self.rgb_bias)

            # Apply padding, mapping color to [-rgb_padding, 1+rgb_padding].
            rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding

        return dict(
            coord=means_contract,
            density=density,
            rgb=rgb,
            raw_grad_density=raw_grad_density,
            grad_pred=grad_pred,
        )


class NerfMLP(ZipNeRFMLP):
    pass


class PropMLP(ZipNeRFMLP):
    pass