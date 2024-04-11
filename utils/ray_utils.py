import numpy as np
import torch

def get_rays(H, W, focal, c2w):
    # opencv camera
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H))
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i - W * .5) / focal, (j - H * .5) / focal, torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., np.newaxis, :].type_as(c2w) * c2w[..., :3, :3], -1)
    viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    rays_o = c2w[..., :3, -1].expand(rays_d.shape)
    rays_o, viewdirs = rays_o.reshape(-1, 3), viewdirs.reshape(-1, 3)
    return rays_o, viewdirs

def get_rays_intrinsic(intrinsic, pose, img_size):
    '''
        intrinsics: [3, 3]
        pose: [3, 4]
        img_size: [H, W]
    '''
    origin = pose[:3, 3]
    rotm = pose[:3, :3]
    H, W = img_size[0], img_size[1]

    x, y = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')

    ray_dirs = np.stack([x, y, np.ones_like(x)], axis=-1)
    intrinsic_inv = np.linalg.inv(intrinsic)
    camera_dirs = ray_dirs @ intrinsic_inv.T
    directions = ((camera_dirs[Ellipsis, None, :] *
                rotm[None, None, :3, :3]).sum(axis=-1))
    viewdirs = directions / np.linalg.norm(directions, axis=-1, keepdims=True)

    origins = np.tile(origin[None, None, :], (directions.shape[0], directions.shape[1], 1))

    dx = np.sqrt(np.sum((directions[:-1, :, :] - directions[1:, :, :]) ** 2, -1))
    dx = np.concatenate([dx, dx[-2:-1, :]], 0)

    radii = dx[..., None] * 2 / np.sqrt(12)

    return origins, directions, radii

def sample_rays(rays_o, rays_d, rgb, num_samples):
    '''
        rays_o: [H, W, 3]
        rays_d: [H, W, 3]
        rgb: [H, W, 3]
        num_samples: int
    '''
    rays_o = rays_o.view(-1, 3)
    rays_d = rays_d.view(-1, 3)
    rgb = rgb.view(-1, 3)

    if num_samples > rays_o.shape[0]:
        sample_idx = np.random.choice(rays_o.shape[0], num_samples, replace=True)
    else:
        sample_idx = np.random.choice(rays_o.shape[0], num_samples, replace=False)
    rays_o = rays_o[sample_idx]
    rays_d = rays_d[sample_idx]
    rgb = rgb[sample_idx]

    return rays_o, rays_d, rgb, sample_idx

def sample_patch(img_hw, num_samples, patch_size):
    
    num_patches = num_samples // (patch_size ** 2)
    all_idx = np.arange(img_hw[0]*img_hw[1])
    all_idx = all_idx.reshape(img_hw[0], img_hw[1])

    x0 = np.random.randint(0, img_hw[1] - patch_size + 1, size=(num_patches, 1, 1))
    y0 = np.random.randint(0, img_hw[0] - patch_size + 1, size=(num_patches, 1, 1))

    xy0 = np.concatenate([x0, y0], axis=-1)
    patch_idx = xy0 + np.stack(
        np.meshgrid(np.arange(patch_size), np.arange(patch_size), indexing='xy'),
        axis=-1).reshape(1, -1, 2)
    
    patch_idx = all_idx[patch_idx[...,1], patch_idx[...,0]]
    
    return patch_idx