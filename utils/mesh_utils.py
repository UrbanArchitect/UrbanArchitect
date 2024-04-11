import torch
import torch.nn.functional as F
import numpy as np

from pytorch3d.renderer.cameras import PerspectiveCameras, FoVPerspectiveCameras
from pytorch3d.renderer import RasterizationSettings, TexturesVertex, MeshRasterizer
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.ops import interpolate_face_attributes

def get_camera(intrinsic, pose, hw, device):
    fx = intrinsic[0,0]
    fy = intrinsic[1,1]
    cx = intrinsic[0,2]
    cy = intrinsic[1,2]
    num_cameras = 1
    focal_length = np.zeros((num_cameras, 2))
    focal_length[:, 0] = fx
    focal_length[:, 1] = fy
    principal_point = np.zeros((num_cameras, 2))
    principal_point[:, 0] = cx
    principal_point[:, 1] = cy
    focal_length = torch.from_numpy(focal_length).float()
    principal_point = torch.from_numpy(principal_point).float()
    w = hw[1]
    h = hw[0]
    half_imwidth = w/2.0
    half_imheight = h/2.0
    focal_length[:,0] /= half_imheight
    focal_length[:,1] /= half_imheight

    principal_point[:, 0] = -(principal_point[:, 0]-half_imwidth)/half_imheight
    principal_point[:, 1] = -(principal_point[:, 1]-half_imheight)/half_imheight
    mirror = torch.eye(4).unsqueeze(0).to(device).repeat(num_cameras, 1, 1)
    mirror[:, 0, 0] = -1.0
    mirror[:, 1, 1] = -1.0

    R_T_joined = torch.eye(4).unsqueeze(0).repeat(num_cameras,1,1).to(device)
    if not torch.is_tensor(pose):
        pose = torch.from_numpy(pose).float()
    pose = pose.to(device)
    R = pose[:3,:3].to(device)
    T = pose[:3,3].to(device)
    R_T_joined[:, :3, :3] = R
    R_T_joined[:, :3, 3]  = T
    R_T_joined = torch.inverse(R_T_joined)
    new_R_T = torch.bmm(mirror, R_T_joined)
    R_camera = new_R_T[:, :3, :3]
    T_camera = new_R_T[:, :3,  3]
    cameras = PerspectiveCameras(device=device, R=R_camera.transpose(1,2).float(), T=T_camera.float(), focal_length=focal_length, principal_point=principal_point)
    return cameras

def mesh_render(intrinsic, pose, img_hw, verts=None, faces=None, verts_feats=None, meshes=None, device='cpu', num_faces=1, resolution=None, use_bin=False):
    '''
    Input:
        intrinsic: [3,3]
        pose: [4,4], c2w
        img_hw: (h,w)
        obj_vertices: [[N,3]]
        obj_faces: [[N,3]]
    Ret:
        unique_obj_idx: [M]
        valid_rays: [start, end]
    '''
    camera = get_camera(intrinsic, pose, img_hw, device=device)
    if meshes is None:
        textures = TexturesVertex([verts_feats])
        meshes = Meshes([verts], [faces], textures)
    meshes = join_meshes_as_scene(meshes)

    if use_bin:
        bin_size = int(2 ** max(np.ceil(np.log2(img_hw[1]))-4,4))
        if img_hw[0] <= 128:
            max_faces_per_bin = int(meshes._F // 1)
        elif img_hw[0] <= 256:
            max_faces_per_bin = int(meshes._F // 2)
        else:
            max_faces_per_bin = int(meshes._F // 5)

        raster_settings = RasterizationSettings(
            image_size=img_hw,
            blur_radius=0.0,
            faces_per_pixel=num_faces,
            perspective_correct=True,
            bin_size=bin_size,
            max_faces_per_bin=max_faces_per_bin,
            cull_to_frustum=False)
    else:
        raster_settings = RasterizationSettings(
            image_size=img_hw,
            blur_radius=0.0,
            bin_size=0,
            faces_per_pixel=num_faces,
            perspective_correct=True)
    
    rasterizer = MeshRasterizer(
        cameras=camera,
        raster_settings=raster_settings)
    
    fragments = rasterizer(meshes)

    pix_to_face = fragments.pix_to_face # [1,H,W,N]
    zbuf = fragments.zbuf # [1,H,W,N]
    bary_coords = fragments.bary_coords # [1,H,W,N,3]

    textures_packed = meshes.textures.verts_features_packed()
    faces = meshes.faces_packed()
    faces_verts_features = textures_packed[faces]
    sampled_texture = interpolate_face_attributes(pix_to_face, bary_coords, faces_verts_features)

    zbuf_min = zbuf[...,:1].repeat(1,1,1,num_faces)
    zbuf_range = zbuf.clone()
    zbuf_range[zbuf_range == -1] = zbuf_min[zbuf_range == -1]
    zbuf_max = torch.max(zbuf_range, dim=-1, keepdim=False)[0]
    zbuf_min = zbuf_min[...,0]
    zbuf_max[zbuf_max == zbuf_min] = zbuf_min[zbuf_max == zbuf_min] + 0.5
    threshold = 5.
    zbuf_max[(zbuf_max - zbuf_min) < threshold] = zbuf_min[(zbuf_max - zbuf_min) < threshold] + threshold
    zbuf_max[zbuf_min == -1] = 200
    zbuf_min[zbuf_min == -1] = 199.5
    sampled_texture = sampled_texture[:,:,:,:1,:]
    zbuf = zbuf[:,:,:,:1]

    zbuf_min = zbuf_min.squeeze()
    zbuf_max = zbuf_max.squeeze()

    if resolution is not None:
        zbuf_min = zbuf_min.view(1, 1, img_hw[0], img_hw[1])
        zbuf_max = zbuf_max.view(1, 1, img_hw[0], img_hw[1])
        if isinstance(resolution, int):
            resolution = (resolution, resolution)
        zbuf_min = F.interpolate(zbuf_min, size=(resolution[0], resolution[1]), mode='bilinear')
        zbuf_max = F.interpolate(zbuf_max, size=(resolution[0], resolution[1]), mode='bilinear')

    return sampled_texture, zbuf, [zbuf_min, zbuf_max]