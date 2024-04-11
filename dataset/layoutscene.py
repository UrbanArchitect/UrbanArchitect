import os
import torch
from torch.utils.data import Dataset
import numpy as np
from io import BytesIO
import sys

from pytorch3d.ops import SubdivideMeshes, interpolate_face_attributes, sample_points_from_meshes
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.renderer import TexturesVertex

id2rgb = {
    0: [0, 0, 255],
    1: [255, 0, 0],
    2: [0, 0, 110],
    3: [153,153,153],
    4: [0, 128,192],
    5: [128,64,128],
    6: [244, 35,232],
    7: [250,170,160],
    8: [70, 70, 70],
    9: [190,153,153],
    10: [107, 142, 35],
    11: [152,251,152],
    12: [230,150,140],
    13: [255,255,255]
}

name2id = {
    'car': 0,
    'truck': 0,
    'bus': 0,
    'van': 0,
    'trailer': 0,
    'caravan': 0,
    'person': 1,
    'rider': 1,
    'bicycle': 2,
    'motorcycle': 2,
    'motorbike': 2,
    'pole': 3,
    'traffic light': 3,
    'traffic sign': 3,
    'smallpole': 3,
    'lamp': 3,
    'trash bin': 4,
    'ground': 5,
    'road': 5,
    'sidewalk': 6,
    'parking': 7,
    'building': 8,
    'garage': 8,
    'fence': 8,
    'gate': 8,
    'vegetation': 10,
    'terrain': 11,
    'rail track': 12,
    'wall': 8,
    'box':13,
    'vending machine':13
}

class LayoutScene(Dataset):
    def __init__(self, config, device='cuda:0', get_pcd=False):
        super().__init__()

        self.config = config
        self.device = device
        self.get_pcd = get_pcd
        assert config.layout_path is not None
        self.layout_path = config.layout_path

    def load_layout(self):
        layout = np.load(self.layout_path, allow_pickle=True).item()

        obj_transforms = layout['obj_transforms']
        cameras = layout['camera']
        objs = layout['objs']

        # KITTI360 intrinsic
        self.intrinsic = np.array([[552.554261, 0., 256-(1408/2-682.049453)],
                                    [0., 552.554261, 256.-(376/2-238.769549)],
                                    [0., 0., 1.]])

        obj_transforms = []
        meshes = []
        obj_verts = []
        obj_faces = []
        obj_textures = []
        bkgd_meshes = []

        num_objs = 0

        obj_nums = {}
        mesh_id = 0
        for obj in objs:
            if obj['name'] in ['trailer', 'caravan']:
                continue
            verts = obj['verts']
            faces = obj['faces']
            transform = obj['transform']
            verts_h = np.concatenate([verts, np.ones_like(verts[...,:1])], axis=-1)

            if obj['name'] in ['car', 'truck', 'bus']:
                verts_ = torch.from_numpy(verts).float().to(self.device)
                obj_verts.append(verts_)
                obj_faces.append(torch.from_numpy(faces).float().to(self.device))
                rgb = np.array(id2rgb[label]) / 255.
                rgbs = torch.from_numpy(rgb[None,...]).to(verts_) * torch.ones_like(verts_[...,:1])
                labels = label * torch.ones_like(verts_[...,:1])
                mesh_ids = mesh_id * torch.ones_like(verts_[...,:1])
                vert_feats = torch.cat([rgbs, labels, mesh_ids], dim=-1)
                obj_textures.append(vert_feats)
                
            verts_h = (transform @ (verts_h.T)).T
            
            if obj['name'] in obj_nums:
                obj_nums[obj['name']] += 1
            else:
                obj_nums[obj['name']] = 1

            if obj['name'] in ['car', 'truck', 'bus']:
                obj_transforms.append(transform)
                num_objs += 1

            verts = verts_h[...,:3]
            mesh = Meshes(verts=[torch.from_numpy(verts).float().cuda()], faces=[torch.from_numpy(faces).long().cuda()])
            
            if obj['name'] in ['building']:
                num_subdivision = 3
            elif obj['name'] in ['road', 'sidewalk', 'ground']:
                num_subdivision = 5
            elif obj['name'] in ['pole', 'trafficsign', 'trafficlight']:
                num_subdivision = 1
            elif obj['name'] in ['vegetation']:
                num_subdivision = 0
            elif obj['name'] in ['car', 'truck', 'bus']:
                num_subdivision = 1
            elif obj['name'] in ['terrain']:
                num_subdivision = 1
            elif obj['name'].startswith('rail'):
                num_subdivision = 2
            else:
                num_subdivision = 0

            for i in range(num_subdivision):
                mesh = SubdivideMeshes()(mesh)

            if obj['name'] in name2id:
                label = name2id[obj['name']]
            else:
                label = 13
            rgb = np.array(id2rgb[label]) / 255.

            rgbs = torch.from_numpy(rgb[None,...]).to(mesh.verts_packed()) * torch.ones_like(mesh.verts_packed()[...,:1])
            labels = label * torch.ones_like(mesh.verts_packed()[...,:1])
            mesh_ids = mesh_id * torch.ones_like(mesh.verts_packed()[...,:1])
            mesh_id += 1
            vert_feats = torch.cat([rgbs, labels, mesh_ids], dim=-1)
            textures = TexturesVertex([vert_feats])
            mesh.textures = textures

            meshes.append(mesh)
            
            if not obj['name'] in ['car', 'truck', 'bus']:
                bkgd_meshes.append(mesh)
        
        num_meshes = len(meshes)
        join_meshes = join_meshes_as_scene(meshes)
        verts = join_meshes.verts_packed()
        faces = join_meshes.faces_packed()
        textures = join_meshes.textures.verts_features_packed()

        join_bkgd_meshes = join_meshes_as_scene(bkgd_meshes)
        bkgd_verts = join_bkgd_meshes.verts_packed()
        bkgd_faces = join_bkgd_meshes.faces_packed()
        bkgd_textures = join_bkgd_meshes.textures.verts_features_packed()
        
        if len(obj_transforms) > 0:
            transforms = torch.from_numpy(np.stack(obj_transforms)).to(verts)
        else:
            transforms = []
        num_verts = verts.shape[0]
        num_faces = faces.shape[0]

        print(f'Get {num_meshes} meshes, {num_verts} verts, {num_faces} faces.')
        print(obj_nums)
        sys.stdout.flush()

        data_dict = {}
        data_dict['verts'] = verts
        data_dict['faces'] = faces
        data_dict['rgbs'] = textures
        data_dict['poses'] = cameras
        data_dict['transforms'] = transforms
        data_dict['meshes'] = meshes
        data_dict['intrinsic'] = self.intrinsic
        if len(obj_verts) > 0:
            data_dict['bkgd_verts'] = bkgd_verts
            data_dict['bkgd_faces'] = bkgd_faces
            data_dict['bkgd_rgbs'] = bkgd_textures
            data_dict['obj_verts'] = obj_verts
            data_dict['obj_faces'] = obj_faces
            data_dict['obj_rgbs'] = obj_textures

        return data_dict
    
    def __getitem__(self, index):

        data_dict = self.load_layout()
        return data_dict