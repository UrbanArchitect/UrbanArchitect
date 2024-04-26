import os
import io
from random import random
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import cv2
from io import BytesIO
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR
from PIL import Image
import time

from hashgrid.models import ScalableHashGrid
from utils.clip import CLIPScore

from diffusers import (
    StableDiffusionInpaintPipeline, 
    StableDiffusionPipeline, 
    ControlNetModel, 
    UNet2DConditionModel, 
    StableDiffusionControlNetPipeline,
    StableDiffusionXLControlNetPipeline,
    AutoencoderKL,
    DDPMScheduler,
    UniPCMultistepScheduler,
    DDIMScheduler
)
from transformers import AutoTokenizer, PretrainedConfig

from scipy.spatial.transform import Rotation as Rot

from dataset.layoutscene import LayoutScene
from guidance.lgvsd import LGVSD
from guidance.pipeline_stable_diffusion_controlnet_resample import StableDiffusionControlNetResamplePipeline

from utils import ray_utils
import torch.nn.functional as F

from utils.train_utils import get_logger, resize_imgs_torch, setup_seed
from utils.losses import dist_loss
from tqdm import tqdm
from utils.mesh_utils import mesh_render

from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.renderer import TexturesVertex

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda')
        self.train_scale_factor = self.config.train_scale_factor
        train_img_size = int(512 / self.train_scale_factor)
        self.pid = os.getpid()

        self.checkpoint_dir = config.checkpoint_dir
        self.save_dir = config.log_dir
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.num_train_step = config.num_train_step

        self.val_freq = config.val_freq
        self.checkpoint_freq = config.checkpoint_freq

        self.controlnet_conditioning_scale = config.controlnet_conditioning_scale

        self.base_lr = config.lr
        self.pre_generation = config.pre_generation
        self.use_clip_loss = config.use_clip_loss
        self.finetune = config.finetune
        self.finetune_style = self.config.finetune_style
        self.empty_bkgd = self.config.empty_bkgd
        self.text_prompt = self.config.text_prompt

        self.logger = get_logger()

        self.logger.info(f'Train scale factor: {self.train_scale_factor} Train img size: {train_img_size}')
        self.logger.info(f'Loading data from {self.config.layout_path}')

        self.dataset = LayoutScene(self.config)
        self.seq = self.dataset[0]
        if torch.is_tensor(self.seq['poses']):
            self.poses = self.seq['poses'].flip(0).to('cpu').numpy()
        else:
            self.poses = self.seq['poses']
        intrinsic = self.seq['intrinsic']

        self.verts = self.seq['verts'].to(self.device)
        self.faces = self.seq['faces'].to(self.device)
        self.vert_rgbs = self.seq['rgbs'].to(self.device)
        self.verts = self.verts[...,:3]
        self.obj_transforms = self.seq['transforms']

        self.obj_verts = self.seq['obj_verts']
        self.obj_faces = self.seq['obj_faces']
        self.obj_rgbs = self.seq['obj_rgbs']
        self.bkgd_verts = self.seq['bkgd_verts']
        self.bkgd_faces = self.seq['bkgd_faces']
        self.bkgd_rgbs = self.seq['bkgd_rgbs']
        
        if torch.is_tensor(intrinsic):
            self.intrinsic = intrinsic.to('cpu').numpy()
        else:
            self.intrinsic = intrinsic
        
        if len(self.obj_transforms) > 0:
            self.obj_transforms = self.obj_transforms.to(self.device)
        self.logger.info(f'Num of objs: {len(self.obj_transforms)}')

        if self.config.resume is not None:
            resume_dict = torch.load(self.config.resume)
            grid_state_dict = resume_dict['state_dict']
            if 'lora' in resume_dict:
                lora_dict = resume_dict['lora']

        self.grid_render = ScalableHashGrid(self.config, self.device, obj_transforms=self.obj_transforms).to(self.device)

        if self.config.resume is not None:
            self.grid_render.load_weights(grid_state_dict)
            self.grid_render.to(self.device)
        
        self.finetune = self.config.finetune
        
        if self.config.render_video:
            assert self.config.resume is not None
            self.render_video()
            os._exit(0)

        self.lgvsd = LGVSD(self.config, device=self.device, fp16=False)
        self.tokenizer = self.lgvsd.tokenizer

        if self.config.resume is not None:
            if not self.finetune_style:
                self.lgvsd.lora_layers.load_state_dict(lora_dict)
        
        if self.use_clip_loss and not self.finetune:
            self.clip_model = CLIPScore(model_path=self.config.clip_model_path, device=self.device).to(self.device)
            self.sd_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                        self.config.sd_model_path,
                        vae=self.lgvsd.vae,
                        text_encoder=self.lgvsd.text_encoder,
                        tokenizer=self.tokenizer,
                        unet=self.lgvsd.unet,
                        controlnet=self.lgvsd.controlnet,
                        safety_checker=None,
                        revision=config.revision,
                        torch_dtype='torch.float32',
                    )
            self.sd_pipeline.scheduler = UniPCMultistepScheduler.from_config(self.sd_pipeline.scheduler.config)
            self.sd_pipeline = self.sd_pipeline.to(self.device)
        
        if self.finetune:
            self.sd_resample_pipeline = StableDiffusionControlNetResamplePipeline.from_pretrained(
                    self.config.sd_model_path,
                    vae=self.lgvsd.vae,
                    text_encoder=self.lgvsd.text_encoder,
                    tokenizer=self.tokenizer,
                    unet=self.lgvsd.unet,
                    controlnet=self.lgvsd.controlnet,
                    safety_checker=None,
                    revision=config.revision,
                    torch_dtype='torch.float32',
                )
            self.sd_resample_pipeline.scheduler = DDIMScheduler.from_config(self.sd_resample_pipeline.scheduler.config)
            self.sd_resample_pipeline = self.sd_resample_pipeline.to(self.device)

            if config.finetune_depth:
                from utils.depth_prediction import MiDasDepthPrediction
                import utils.depth_alignment as depth_alignment
                self.logger.info('Loading MiDas...')
                depth_type = 'dpt_beit_large_512'
                self.depth_prediction = MiDasDepthPrediction(config.depth_model_path, depth_type, self.device)
                self.depth_state_dict = self.depth_prediction.model.state_dict()
            
            if config.finetune_sky:
                from utils.sem_seg import SemSegOneformer
                self.sem_seg = SemSegOneformer(config.semantic_model_path)

    def reset_optimizer(self, lr=1e-3):
        if self.config.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        self.optimizer = optimizer_class([
                {'params':self.grid_render.parameters(), 'lr':lr},
                ])
        self.optimizer.add_param_group({'params':self.lgvsd.lora_layers.parameters(), 'lr':1e-3})
    
    def prepare_text_embeddings(self, text):
        negative_text = ''
        text_embedding = self.lgvsd.get_text_embeds([text], [negative_text])
        return text_embedding
    
    def disp_to_depth(self, disp):
        self.depth_scale = 0.00006016
        self.depth_shift = 0.00579
        depth = self.depth_scale * disp + self.depth_shift
        depth[depth < 1e-8] = 1e-8
        depth = 1.0 / depth

        return depth
    
    def depth_alignment(self, rendered_depth, predicted_depth, inpaint_mask=None, fuse=True, ret_numpy=True):
        if not isinstance(rendered_depth, torch.Tensor):
            rendered_depth = torch.from_numpy(rendered_depth).float().to(self.device_1)
        if not isinstance(predicted_depth, torch.Tensor):
            predicted_depth = torch.from_numpy(predicted_depth).float().to(self.device_1)
        
        if inpaint_mask is None:
            inpaint_mask = (rendered_depth > 1.) & (predicted_depth > 1.) & (rendered_depth < 100.) & (predicted_depth < 100.)
        
        aligned_depth = depth_alignment.scale_shift_linear(
            rendered_depth=rendered_depth,
            predicted_depth=predicted_depth,
            mask=inpaint_mask,
            fuse=fuse)
        
        if ret_numpy:
            aligned_depth = aligned_depth.to('cpu').numpy()

        return aligned_depth
    
    @torch.no_grad()
    def sky_mask_prediction(self, img):
        if isinstance(img, torch.Tensor):
            img = img.to('cpu').numpy()
        img = (img * 255.).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        sem_labels = self.sem_seg(img)
        sem_labels = torch.argmax(sem_labels, dim=0)
        sem_labels = sem_labels.cpu().numpy()
        sky_masks = (sem_labels == 10)
        return sky_masks
    
    def finetune_depth_model(self, img, depth, mask=None, step=100, ret_numpy=True):
        if mask is None:
            mask = (depth > 1.) & (depth < 100.)
        
        valid_mask = mask

        depth_optimizer = torch.optim.AdamW(self.depth_prediction.parameters(), lr=1e-6)
        depth_finetune_steps = step
        depth_finetune_print_steps = 20
        if not isinstance(depth, torch.Tensor):
            depth = torch.from_numpy(depth).float().to(self.device_1)
        self.logger.info('Finetuning depth...')
        for depth_step in range(depth_finetune_steps):
            pred_depth = self.depth_prediction(img).squeeze(0).float()
            pred_depth = self.disp_to_depth(pred_depth)
            aligned_pred_depth = pred_depth
            depth_loss = ((aligned_pred_depth - depth).abs() * (valid_mask)).sum() / (valid_mask.sum() + 1e-8)
            depth_loss = depth_loss / 100.
            
            depth_optimizer.zero_grad()
            depth_loss.backward()
            depth_optimizer.step()
            if depth_step % depth_finetune_print_steps == 0:
                self.logger.info(f'Finetuned depth loss: {depth_loss.item()}')
        
        self.depth_prediction.model.load_state_dict(self.depth_state_dict)
        del depth_optimizer
        
        if ret_numpy:
            return aligned_pred_depth.detach().cpu().numpy()
        else:
            return aligned_pred_depth.detach()
    
    def refinement_step(self, data, step):
        self.optimizer.zero_grad()
        c2w = data['c2w']
        intrinsic = data['intrinsic']
        img_hw = data['img_hw']
        resample_image = data['resample_image'].to(self.device)
        intrinsic = intrinsic / self.train_scale_factor
        intrinsic[2,2] = 1.
        img_hw = (int(img_hw[0]/self.train_scale_factor), int(img_hw[1]/self.train_scale_factor))

        num_samples = self.config.finetune_num_rays
        all_num_rays = int(img_hw[0] * img_hw[1])
        sample_idx = np.random.choice(all_num_rays, num_samples)
        sample_idx = sample_idx.reshape(-1)

        rays_o, rays_d, radii = ray_utils.get_rays_intrinsic(intrinsic, c2w, img_hw)
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)
        radii = radii.reshape(-1, 1)
        
        rays_o = torch.from_numpy(rays_o).float().to(self.device)
        rays_d = torch.from_numpy(rays_d).float().to(self.device)
        radii = torch.from_numpy(radii).float().to(self.device)

        rays_o = rays_o[sample_idx]
        rays_d = rays_d[sample_idx]
        radii = radii[sample_idx]
        
        near = data['near'].view(-1, 1)
        far = data['far'].view(-1, 1)
        near = near[sample_idx]
        far = far[sample_idx]
        near = near.to(rays_o)
        far = far.to(rays_d)

        ret_dict = self.grid_render.render(rays_o, rays_d, near, far, radii, train_frac=0, rand=True)[-1]

        pred_rgb = ret_dict['rgbs'].view(-1, 3)
        pred_depth = ret_dict['depths'].view(-1)

        gt_depth = data['resample_depth'].to(self.device)
        gt_depth = gt_depth.view(-1)[sample_idx]

        if self.config.finetune_sky:
            sky_mask = data['resample_sky_mask'].to(self.device)
            sky_mask = sky_mask.view(-1)[sample_idx]

        if self.config.finetune_depth:
            aligned_gt_depth = gt_depth.view(-1)
            depth_loss = (aligned_gt_depth - pred_depth).abs().view(-1)
            aligned_gt_depth = aligned_gt_depth.view(-1)
            pred_depth = pred_depth.view(-1)
            valid_depth_mask = (aligned_gt_depth > 1e-3) & (aligned_gt_depth < 100) & (pred_depth > 1e-3) & (pred_depth < 100)
            valid = valid_depth_mask.view(-1)
            if self.config.finetune_sky:
                valid = valid & (~sky_mask.view(-1))
            depth_loss = depth_loss[valid].mean()
        else:
            depth_loss = 0.

        if self.config.finetune_sky:
            acc_loss = ret_dict['acc'].view(-1)[sky_mask.view(-1)].sum() / (sky_mask.view(-1).sum() + 1e-8)
        else:
            acc_loss = 0.

        resample_image = resample_image.view(-1, 3)[sample_idx]
        mae_loss = ((pred_rgb - resample_image).abs()).mean()
        sdists = ret_dict['sdist']
        weights = ret_dict['weights']

        if self.config.finetune_sky:
            d_loss = dist_loss(sdists[~sky_mask], weights[~sky_mask])
        else:
            d_loss = dist_loss(sdists, weights)
        
        loss = mae_loss + acc_loss * 0.10 + depth_loss * 0.05 + d_loss * 0.10

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train_step(self, data, step):
        self.grid_render.train()
        self.optimizer.zero_grad()
        c2w = data['c2w']
        text = data['text']
        intrinsic = data['intrinsic']
        img_hw = data['img_hw']
        intrinsic = intrinsic / self.train_scale_factor
        intrinsic[2,2] = 1.
        img_hw = (int(img_hw[0]/self.train_scale_factor), int(img_hw[1]/self.train_scale_factor))
        
        controlnet_image = data['controlnet_image']
        near = data['near']
        far = data['far']
        near = near.view(1,512,512,-1).permute(0,3,1,2)
        far = far.view(1,512,512,-1).permute(0,3,1,2)
        near = F.interpolate(near, size=img_hw)
        far = F.interpolate(far, size=img_hw)

        rays_o, rays_d, radii = ray_utils.get_rays_intrinsic(intrinsic, c2w, img_hw)
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)
        radii = radii.reshape(-1, 1)
        rays_o = torch.from_numpy(rays_o).float().to(self.device)
        rays_d = torch.from_numpy(rays_d).float().to(self.device)
        radii = torch.from_numpy(radii).float().to(self.device)

        near = near.view(-1, 1).to(rays_o)
        far = far.view(-1, 1).to(rays_d)

        ret_dict = self.grid_render.render(rays_o, rays_d, near, far, radii, train_frac=0, rand=True, empty_bkgd=self.empty_bkgd)[-1]

        if ret_dict['reset_flag']:
            self.reset_optimizer(lr=self.base_lr)
        pred_rgb = ret_dict['rgbs']
        pred_rgb = pred_rgb.view(img_hw[0], img_hw[1], -1)
        pred_rgb = pred_rgb.permute(2,0,1).contiguous().unsqueeze(0)
        if self.use_clip_loss:
            sample_image = data['sample_image']
            sample_image = sample_image.to(self.device)
        
        grad_clip = 8
        text_z = self.prepare_text_embeddings(text)

        if not torch.is_tensor(c2w):
            camera_condition = torch.from_numpy(c2w).float().to(self.device).view(1, -1)
        else:
            camera_condition = c2w.view(1, -1).to(self.device)
        loss_lgvsd, loss_lora = self.lgvsd.train_step(
            text_z, pred_rgb, controlnet_image, as_latent=False, grad_clip=grad_clip, guidance_scale=7.5, guidance_scale_lora=1., 
            camera_condition=camera_condition, conditioning_scale=self.controlnet_conditioning_scale)
        loss = loss_lgvsd + loss_lora
        
        if self.use_clip_loss:
            sample_image = F.interpolate(sample_image, size=pred_rgb.shape[2:])
            with torch.no_grad():
                sample_clip_features = self.clip_model.get_image_features(sample_image)
                pred_clip_features = self.clip_model.get_image_features(pred_rgb)
            clip_loss = F.mse_loss(pred_clip_features, sample_clip_features)
            loss = loss + clip_loss * 1000.

        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def val_step(self, data):
        c2w = data['c2w']
        intrinsic = data['intrinsic']
        
        img_hw = data['img_hw']
        
        intrinsic = intrinsic / self.train_scale_factor
        intrinsic[2,2] = 1.
        img_hw = (int(img_hw[0]/self.train_scale_factor), int(img_hw[1]/self.train_scale_factor))

        near = data['near'].view(1,512,512,-1).permute(0,3,1,2)
        far = data['far'].view(1,512,512,-1).permute(0,3,1,2)
        near = F.interpolate(near, size=img_hw, mode='bilinear')
        far = F.interpolate(far, size=img_hw, mode='bilinear')

        rays_o, rays_d, radii = ray_utils.get_rays_intrinsic(intrinsic, c2w, img_hw)
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)
        radii = radii.reshape(-1, 1)
        rays_o = torch.from_numpy(rays_o).float().to(self.device)
        rays_d = torch.from_numpy(rays_d).float().to(self.device)
        radii = torch.from_numpy(radii).float().to(self.device)

        near = near.view(-1, 1).to(rays_o)
        far = far.view(-1, 1).to(rays_d)

        with torch.no_grad():
            chunk_size = self.config.chunk_size
            num_chunks = rays_o.shape[0] // chunk_size + 1
            ret_dict_list = []
            for chunk_idx in range(num_chunks):
                rays_o_chunk = rays_o[chunk_idx*chunk_size:(chunk_idx+1)*chunk_size]
                rays_d_chunk = rays_d[chunk_idx*chunk_size:(chunk_idx+1)*chunk_size]
                near_chunk = near[chunk_idx*chunk_size:(chunk_idx+1)*chunk_size]
                far_chunk = far[chunk_idx*chunk_size:(chunk_idx+1)*chunk_size]
                radii_chunk = radii[chunk_idx*chunk_size:(chunk_idx+1)*chunk_size]
                if rays_o_chunk.shape[0] == 0:
                    break
                ret_dict_chunk = self.grid_render.render(rays_o_chunk, rays_d_chunk, near_chunk, far_chunk, radii_chunk, train_frac=0, rand=False, empty_bkgd=self.empty_bkgd)[-1]
                ret_dict_list.append(ret_dict_chunk)

            ret_dict = {}
            for key in ret_dict_list[0].keys():
                if key in ['rgbs', 'depths', 'acc']:
                    ret_dict[key] = torch.cat([ret_dict_chunk[key] for ret_dict_chunk in ret_dict_list], dim=0)
            pred_rgb = ret_dict['rgbs']
            pred_rgb = pred_rgb.view(img_hw[0], img_hw[1], -1)
            pred_depth = ret_dict['depths']
            pred_depth = pred_depth.view(img_hw[0], img_hw[1])
            pred_acc = ret_dict['acc']
            pred_acc = pred_acc.view(img_hw[0], img_hw[1])

        return pred_rgb, pred_depth, pred_acc
    
    @torch.no_grad()
    def resample_images(self, resample_steps=5, repeat=2):
        self.train_scale_factor = 1
        resample_images = []
        pred_depths = []
        self.logger.info('Resampling images')

        for idx in tqdm(range(len(self.poses))):
            c2w = self.poses[idx]
            img_hw = [512, 512]
            intrinsic = self.intrinsic / self.train_scale_factor
            intrinsic[2,2] = 1.
            img_hw = (int(img_hw[0]/self.train_scale_factor), int(img_hw[1]/self.train_scale_factor))
            rays_o, rays_d, radii = ray_utils.get_rays_intrinsic(intrinsic, c2w, img_hw)
            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)
            radii = radii.reshape(-1, 1)
            rays_o = torch.from_numpy(rays_o).float().to(self.device)
            rays_d = torch.from_numpy(rays_d).float().to(self.device)
            radii = torch.from_numpy(radii).float().to(self.device)

            sample_scale_factor = 4
            img_hw_scale = (int(512 // sample_scale_factor), int(512 // sample_scale_factor))
            intrinsic_scale = self.intrinsic / sample_scale_factor
            intrinsic_scale[2,2] = 1.
            with torch.no_grad():
                control_sem, control_depth, near_far = mesh_render(intrinsic_scale, c2w, verts=self.verts.float(), faces=self.faces.long(), verts_feats=self.vert_rgbs.float(), 
                                                                    device=self.device, num_faces=8, img_hw=img_hw_scale, resolution=self.config.resolution, use_bin=True)
            near = near_far[0]
            far = near_far[1]
            near = near.view(1,512,512,-1).permute(0,3,1,2)
            far = far.view(1,512,512,-1).permute(0,3,1,2)
            near = F.interpolate(near, size=img_hw)
            far = F.interpolate(far, size=img_hw)
            near = near.view(-1, 1).to(rays_o)
            far = far.view(-1, 1).to(rays_d)

            chunk_size = 128 * 128
            num_chunks = rays_o.shape[0] // chunk_size + 1
            ret_dict_list = []
            for chunk_idx in range(num_chunks):
                rays_o_chunk = rays_o[chunk_idx*chunk_size:(chunk_idx+1)*chunk_size]
                rays_d_chunk = rays_d[chunk_idx*chunk_size:(chunk_idx+1)*chunk_size]
                near_chunk = near[chunk_idx*chunk_size:(chunk_idx+1)*chunk_size]
                far_chunk = far[chunk_idx*chunk_size:(chunk_idx+1)*chunk_size]
                radii_chunk = radii[chunk_idx*chunk_size:(chunk_idx+1)*chunk_size]
                if rays_o_chunk.shape[0] == 0:
                    break
                ret_dict_chunk = self.grid_render.render(rays_o_chunk, rays_d_chunk, near_chunk, far_chunk, radii_chunk, train_frac=0, rand=False, empty_bkgd=False)[-1]
                ret_dict_list.append(ret_dict_chunk)
            ret_dict = {}
            for key in ret_dict_list[0].keys():
                if key in ['rgbs', 'depths', 'acc']:
                    ret_dict[key] = torch.cat([ret_dict_chunk[key] for ret_dict_chunk in ret_dict_list], dim=0)
            control_sem = control_sem.squeeze()[...,:3]
            control_depth = control_depth.squeeze()
            control_depth[control_depth==-1] = 0.
            control_sem = control_sem.unsqueeze(0).permute(0,3,1,2)
            control_sem = 2 * (control_sem - 0.5)
            control_depth = control_depth.unsqueeze(0).unsqueeze(0) / 200.
            control_sem = resize_imgs_torch(control_sem, self.config.resolution)
            control_depth = resize_imgs_torch(control_depth, self.config.resolution)
            controlnet_image = torch.cat([control_sem, control_depth], dim=1)
                
            pred_rgb = ret_dict['rgbs']
        
            generator = torch.Generator(device=self.device).manual_seed(seed)

            pred_rgb = pred_rgb.view(img_hw[0], img_hw[1], 3).permute(2,0,1).unsqueeze(0)
            
            pred_rgb = F.interpolate(pred_rgb, size=(512, 512), mode='bilinear')

            pred_depth = ret_dict['depths'].view(img_hw[0], img_hw[1])
            pred_depth = pred_depth.unsqueeze(0).unsqueeze(0)
            pred_depth = F.interpolate(pred_depth, size=(512, 512))
            pred_depth = pred_depth.squeeze()
            
            resample_image = self.sd_resample_pipeline.resample(
                self.text_prompt, controlnet_image, num_steps=50, resample_steps=resample_steps, repeat=repeat, start_image=pred_rgb, generator=generator, controlnet_conditioning_scale=self.controlnet_conditioning_scale
            )

            resample_image = resample_image.images[0]
            resample_image.save(f'{self.save_dir}/resample_img_{idx}.png')
            resample_image = np.array(resample_image).astype(np.float32) / 255.
            resample_image = torch.from_numpy(resample_image).to(self.device)
            resample_image = resample_image.to('cpu')
            pred_depth = pred_depth.to('cpu')
            resample_images.append(resample_image)
            pred_rgb = pred_rgb.squeeze().permute(1,2,0)
            pred_rgb = pred_rgb.cpu().numpy()
            pred_rgb = Image.fromarray((pred_rgb * 255).astype(np.uint8))
            pred_rgb.save(f'{self.save_dir}/original_img_{idx}.png')
            pred_depths.append(pred_depth)
        
        return resample_images, pred_depths
    
    def train(self):

        self.reset_optimizer(self.base_lr)

        self.poses = self.poses[0:len(self.poses):5]

        if self.pre_generation and not self.finetune:
            control_images = []
            self.logger.info('Pre-generating SD images for CLIP loss...')
            gen_step = 0
            for c2w in tqdm(self.poses):
                control_sem, control_depth, near_far = mesh_render(self.intrinsic, c2w, verts=self.verts.float(), faces=self.faces.long(), verts_feats=self.vert_rgbs.float(), 
                                                                        device=self.device, num_faces=16, img_hw=(512,512), resolution=512, use_bin=True)
                control_sem = control_sem.squeeze()[...,:3]
                control_depth = control_depth.squeeze()
                control_depth[control_depth==-1] = 0.
                control_sem = control_sem.unsqueeze(0).permute(0,3,1,2)
                control_sem = 2 * (control_sem - 0.5)
                control_depth = control_depth.unsqueeze(0).unsqueeze(0) / 200.
                control_sem = resize_imgs_torch(control_sem, self.config.resolution)
                control_depth = resize_imgs_torch(control_depth, self.config.resolution)
                control_signal = torch.cat([control_sem, control_depth], dim=1)
                generator = torch.Generator(device=self.device).manual_seed(seed)
                text_prompt = self.text_prompt
                negative_prompt = ''
                with torch.autocast('cuda'):
                    sample_image = self.sd_pipeline(
                            text_prompt, image=control_signal, num_inference_steps=20, generator=generator, controlnet_conditioning_scale=self.controlnet_conditioning_scale, negative_prompt=negative_prompt
                    ).images[0]
                control_sem = control_sem.squeeze().permute(1,2,0).to('cpu').numpy()[...,:3]
                control_sem = control_sem / 2. + 0.5
                control_sem = Image.fromarray((control_sem * 255.).astype(np.uint8))
                control_sem.save(f'{self.save_dir}/control_sem_{gen_step}.png')
                sample_image.save(f'{self.save_dir}/sample_img_{gen_step}.png')
                gen_step += 1
                sample_image = np.array(sample_image).astype(np.float32) / 255.
                sample_image = torch.from_numpy(sample_image).permute(2,0,1).unsqueeze(0)
                control_images.append(sample_image)
            
            del self.sd_pipeline
            torch.cuda.empty_cache()

        if self.finetune:
            resample_images, pred_depths = self.resample_images(resample_steps=5, repeat=2)
            resample_depths = []
            if self.config.finetune_sky:
                resample_sky_masks = []
                for resample_image in resample_images:
                    resample_sky_mask = self.sky_mask_prediction(resample_image.cpu().numpy())
                    resample_sky_mask = torch.from_numpy(resample_sky_mask).to(self.device).bool()
                    resample_sky_mask = resample_sky_mask.to('cpu')
                    resample_sky_masks.append(resample_sky_mask)
                del self.sem_seg
            
            for resample_idx in range(len(resample_images)):
                resample_image = resample_images[resample_idx]
                pred_depth = pred_depths[resample_idx].to(self.device)
                depth_mask = (pred_depth > 0.) & (pred_depth < 100.)
                sky_mask = resample_sky_masks[resample_idx].to(self.device)
                depth_mask = depth_mask & (~sky_mask)
                resample_depth = self.finetune_depth_model(resample_image.cpu().numpy(), pred_depth.detach(), mask=depth_mask, step=10, ret_numpy=False)
                resample_depth = resample_depth.to(self.device)
                resample_depth = resample_depth.to('cpu')
                resample_depths.append(resample_depth)

            del self.depth_state_dict
            del self.depth_prediction
            del self.sd_resample_pipeline
            torch.cuda.empty_cache()

        if not self.finetune:
            poses_x = np.array(self.poses)[:,0,3]
            poses_z = np.array(self.poses)[:,2,3]
            poses_xz = np.stack([poses_x, poses_z], axis=1)
            poses_dist = np.linalg.norm(poses_xz[:,None,:] - poses_xz[None,:,:], axis=-1) # [NxN]
            sorted_dist = np.sort(poses_dist, axis=1) # [NxN-1]
            sorted_dist = sorted_dist[:,1:]
            min_dist = np.min(sorted_dist, axis=1)
            mean_dist = np.mean(min_dist)
            
        for step in tqdm(range(self.num_train_step), ncols=80):
            
            random_idx = np.random.randint(0, len(self.poses))
            random_pose = self.poses[random_idx]
            angle_range = self.config.random_angle
            if self.finetune:
                angle_range = 0.0
            random_rot_angle = np.random.uniform(-angle_range, angle_range)
            rot_angle = np.array([0., -random_rot_angle, 0.]) * np.pi / 180.
            rotm = Rot.from_euler('zyx', rot_angle, degrees=False)
            rotm = rotm.as_matrix()
            transm = np.eye(4)
            transm[:3,:3] = rotm

            c2w = random_pose.copy()
            c2w[:3,:3] = transm[:3,:3] @ c2w[:3,:3]

            if not self.finetune:
                if np.linalg.det(c2w[:3,:3]) < 0:
                    rot = Rot.from_matrix(-c2w[:3,:3])
                else:
                    rot = Rot.from_matrix(c2w[:3,:3])
                rotvec = rot.as_euler('yzx', degrees=True)

                deltaz = np.cos(np.deg2rad(rotvec[0]))
                deltax = np.sin(np.deg2rad(rotvec[0]))
                random_xz = np.random.uniform(-1., 1.) * mean_dist
                deltaz = deltaz * random_xz
                deltax = deltax * random_xz
                c2w[0,3] += deltax
                c2w[2,3] += deltaz

            data = {}
            data['intrinsic'] = self.intrinsic
            data['c2w'] = c2w
            data['text'] = self.text_prompt
            data['img_hw'] = [512, 512]
            if self.pre_generation and not self.finetune:
                data['sample_image'] = control_images[random_idx]
            
            if self.finetune:
                data['resample_image'] = resample_images[random_idx]
                data['resample_depth'] = resample_depths[random_idx]
                if self.config.finetune_sky:
                    data['resample_sky_mask'] = resample_sky_masks[random_idx]

            sample_scale_factor = 4
            img_hw_scale = (int(512 // sample_scale_factor), int(512 // sample_scale_factor))
            intrinsic_scale = self.intrinsic / sample_scale_factor
            intrinsic_scale[2,2] = 1.
            with torch.no_grad():
                control_sem, control_depth, near_far = mesh_render(intrinsic_scale, c2w, verts=self.verts.float(), faces=self.faces.long(), verts_feats=self.vert_rgbs.float(), 
                                                                    device=self.device, num_faces=8, img_hw=img_hw_scale, resolution=self.config.resolution, use_bin=True)
            control_sem = control_sem.squeeze()[...,:3]
            control_depth = control_depth.squeeze()
            control_depth[control_depth==-1] = 0.
            control_sem = control_sem.unsqueeze(0).permute(0,3,1,2)
            control_sem = 2 * (control_sem - 0.5)
            control_depth = control_depth.unsqueeze(0).unsqueeze(0) / 200.
            control_sem = resize_imgs_torch(control_sem, self.config.resolution)
            control_depth = resize_imgs_torch(control_depth, self.config.resolution)
            controlnet_image = torch.cat([control_sem, control_depth], dim=1)
            data['controlnet_image'] = controlnet_image
            data['near'] = near_far[0]
            data['far'] = near_far[1]
            if self.finetune:
                loss = self.refinement_step(data, step)
            else:
                loss = self.train_step(data, step)

            if step % self.val_freq == 0:
                pred_rgb, pred_depth, pred_acc = self.val_step(data)
                pred_rgb = pred_rgb.to('cpu').numpy()
                pred_rgb_pil = Image.fromarray((pred_rgb*255.).astype(np.uint8))
                pred_rgb_pil.save(f'{self.save_dir}/val_{step}.png')
                control_sem_numpy = control_sem.squeeze().permute(1,2,0).to('cpu').numpy()[...,:3]
                control_sem_numpy = control_sem_numpy / 2. + 0.5
                control_sem_numpy_pil = Image.fromarray((control_sem_numpy * 255.).astype(np.uint8))
                control_sem_numpy_pil.save(f'{self.save_dir}/control_sem_{step}.png')
            
            if step % self.checkpoint_freq == 0 and step != 0:
                self.save_checkpoint('ckpt')
    
    def save_checkpoint(self, filename):
        filename = os.path.join(self.checkpoint_dir, f'{filename}.pth')

        data_dict = {}
        data_dict['verts'] = self.verts
        data_dict['faces'] = self.faces
        data_dict['vert_rgbs'] = self.vert_rgbs
        data_dict['poses'] = self.poses
        data_dict['intrinsic'] = self.intrinsic
        data_dict['obj_transforms'] = self.obj_transforms

        state = {
            'state_dict': self.grid_render.state_dict(),
            'config': self.config,
            'data_dict': data_dict,
        }
        state['lora'] = self.lgvsd.lora_layers.state_dict()
        torch.save(state, filename)
        self.logger.info(f'Save checkpoints in {filename}')
    
    def render_video(self):
        from scipy.spatial.transform import Rotation
        from scipy.spatial.transform import Slerp
        assert self.config.resume is not None

        c2ws = self.poses

        num_poses = len(c2ws)

        all_poses = c2ws
        
        print('Num of poses: ', len(all_poses))
        output_dir = os.path.join(self.config.log_dir, 'results')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        img_save_dir = f'{output_dir}/imgs'
        if not os.path.exists(img_save_dir):
            os.makedirs(img_save_dir)
        cond_sem_save_dir = f'{output_dir}/control_sem'
        if not os.path.exists(cond_sem_save_dir):
            os.makedirs(cond_sem_save_dir)

        for pose_idx in range(len(all_poses)):
            print(f'{pose_idx}/{len(all_poses)}')
            sys.stdout.flush()
            save_id = '%03d' % pose_idx
            curr_c2w = all_poses[pose_idx]
            data = {}
            data['intrinsic'] = self.intrinsic.copy()
            data['c2w'] = curr_c2w
            data['text'] = self.text_prompt
            data['img_hw'] = [512, 512]

            sample_scale_factor = 1
            img_hw_scale = (int(512 // sample_scale_factor), int(512 // sample_scale_factor))
            intrinsic_scale = self.intrinsic / sample_scale_factor
            intrinsic_scale[2,2] = 1.
            control_sem, control_depth, near_far = mesh_render(intrinsic_scale, curr_c2w, verts=self.verts.float(), faces=self.faces.long(), verts_feats=self.vert_rgbs.float(), 
                                                                device=self.device, num_faces=8, img_hw=img_hw_scale, resolution=self.config.resolution, use_bin=True)
            control_sem = control_sem.squeeze()[...,:3]
            control_depth = control_depth.squeeze()
            control_depth[control_depth==-1] = 0.
            control_sem = control_sem.unsqueeze(0).permute(0,3,1,2)
            control_sem = 2 * (control_sem - 0.5)
            control_depth = control_depth.unsqueeze(0).unsqueeze(0) / 200.

            data['controlnet_image'] = control_sem
            data['near'] = near_far[0]
            data['far'] = near_far[1]
            img, render_depth, acc = self.val_step(data)
            
            render_depth[acc < 0.01] = 100.
            render_depth[render_depth > 100.] = 100.

            control_sem = control_sem.squeeze().permute(1,2,0).to('cpu').numpy()[...,:3]
            control_sem = control_sem / 2. + 0.5
            control_sem = Image.fromarray((control_sem * 255.).astype(np.uint8))
            control_sem.save(f'{cond_sem_save_dir}/control_sem_{save_id}.png')

            img = img.to('cpu').numpy()
            img_pil = Image.fromarray((img*255.).astype(np.uint8))
            img_pil.save(f'{output_dir}/imgs/rendered_img_{save_id}.png')
            
if __name__ == '__main__':
    from utils.config import get_args
    config = get_args()
    import cv2
    import os
    seed = np.random.randint(0, 1e8)
    setup_seed(seed)
    device = 'cuda:0'
    trainer = Trainer(config)
    trainer.train()