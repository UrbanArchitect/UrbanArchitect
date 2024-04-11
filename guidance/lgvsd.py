from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import (
    AutoencoderKL, 
    UNet2DConditionModel, 
    PNDMScheduler, 
    DDIMScheduler, 
    DDPMScheduler,
    StableDiffusionPipeline, 
    EulerDiscreteScheduler,
    StableDiffusionControlNetPipeline,
    ControlNetModel)
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.embeddings import TimestepEmbedding

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from torch.cuda.amp import custom_bwd, custom_fwd
from dataclasses import dataclass

class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None

class ToWeightsDType(nn.Module):
    def __init__(self, module: nn.Module, dtype: torch.dtype):
        super().__init__()
        self.module = module
        self.dtype = dtype

    def forward(self, x):
        return self.module(x).to(self.dtype)
    
@dataclass
class UNet2DConditionOutput:
    sample: torch.HalfTensor # Not sure how to check what unet_traced.pt contains, and user wants. HalfTensor or FloatTensor

class LGVSD(nn.Module):
    def __init__(self, config, device, fp16=False):
        super().__init__()
        self.config = config
        self.device = device

        print(f'[INFO] loading stable diffusion...')

        self.weight_dtype = self.precision_t = precision_t = torch.float16 if fp16 else torch.float32
        sd_model_path = config.sd_model_path

        self.pipe = StableDiffusionPipeline.from_pretrained(sd_model_path, local_files_only=True).to(self.device)
        self.pipe_lora = StableDiffusionPipeline.from_pretrained(sd_model_path, local_files_only=True).to(self.device)

        if is_xformers_available():
            import xformers
            print('Use memory efficient')
            self.pipe.enable_xformers_memory_efficient_attention()
            self.pipe_lora.enable_xformers_memory_efficient_attention()
        else:
            pass
        
        self.unet_lora = self.pipe_lora.unet

        self.lora_n_timestamp_samples = 1
        self.lora_cfg_training = True
        self.weights_dtype = torch.float32

        self.vae = self.pipe.vae
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder
        self.unet = self.pipe.unet
        self.vae_scaling_factor = self.vae.config.scaling_factor
        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)
        for p in self.unet_lora.parameters():
            p.requires_grad_(False)

        conditioning_channels = 4
        self.controlnet = ControlNetModel.from_unet(self.unet, conditioning_channels=conditioning_channels)
        checkpoint = torch.load(config.controlnet_model_path)['state_dict']
        self.controlnet.load_state_dict(checkpoint)
        self.controlnet.to(self.device, self.weight_dtype)

        self.scheduler = DDPMScheduler.from_pretrained(sd_model_path, subfolder="scheduler", local_files_only=True, torch_dtype=precision_t)
        self.scheduler_lora = DDPMScheduler.from_pretrained(
            sd_model_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
        )

        # FIXME: hard-coded dims
        self.camera_embedding = ToWeightsDType(
            TimestepEmbedding(16, 1280), self.weights_dtype
        )
        self.unet_lora.class_embedding = self.camera_embedding.to(self.device)

        # set up LoRA layers
        lora_attn_procs = {}
        for name in self.unet_lora.attn_processors.keys():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else self.unet_lora.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = self.unet_lora.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet_lora.config.block_out_channels))[
                    block_id
                ]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet_lora.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
            )

        self.unet_lora.set_attn_processor(lora_attn_procs)
        self.unet_lora.to(self.device)

        self.lora_layers = AttnProcsLayers(self.unet_lora.attn_processors)
        self.lora_layers._load_state_dict_pre_hooks.clear()
        self.lora_layers._state_dict_hooks.clear()

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * 0.02)
        self.max_step = int(self.num_train_timesteps * 0.98)
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience
    
    def get_text_embeds(self, prompt, negative_prompt):
        # prompt, negative_prompt: [str]

        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')

        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings
    
    def train_lora(
        self,
        latents,
        text_embeddings,
        camera_condition,
        down_block_res_samples,
        mid_block_res_sample,
    ):
        B = latents.shape[0]
        latents = latents.detach().repeat(self.lora_n_timestamp_samples, 1, 1, 1)

        t = torch.randint(
            int(self.num_train_timesteps * 0.0),
            int(self.num_train_timesteps * 1.0),
            [B * self.lora_n_timestamp_samples],
            dtype=torch.long,
            device=self.device,
        )

        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler_lora.add_noise(latents, noise, t)
        if self.scheduler_lora.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler_lora.config.prediction_type == "v_prediction":
            target = self.scheduler_lora.get_velocity(latents, noise, t)
        else:
            raise ValueError(
                f"Unknown prediction type {self.scheduler_lora.config.prediction_type}"
            )
        _, text_embeddings = text_embeddings.chunk(2)
        if self.lora_cfg_training and random.random() < 0.1:
            camera_condition = torch.zeros_like(camera_condition)
        down_block_res_samples = [
                sample.chunk(2)[1] for sample in down_block_res_samples
            ]
        mid_block_res_sample = mid_block_res_sample.chunk(2)[1]
        noise_pred = self.unet_lora(
            noisy_latents,
            t,
            encoder_hidden_states=text_embeddings.repeat(self.lora_n_timestamp_samples, 1, 1),
            class_labels=camera_condition.view(B, -1).repeat(
                self.lora_n_timestamp_samples, 1
            ),
            down_block_additional_residuals=[
                sample.to(dtype=self.precision_t) for sample in down_block_res_samples
            ],
            mid_block_additional_residual=mid_block_res_sample.to(dtype=self.weight_dtype),
        ).sample
        
        return F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
    
    def train_step(self, 
                   text_embeddings, 
                   pred_rgb, 
                   controlnet_image, 
                   guidance_scale=7.5, 
                   guidance_scale_lora=1.0, 
                   camera_condition=None, 
                   as_latent=False, 
                   grad_clip=None, 
                   conditioning_scale=1.):
        
        if as_latent:
            latents = F.interpolate(pred_rgb, (64, 64), mode='bilinear', align_corners=False) * 2 - 1
        else:
            # interp to 512x512 to be fed into vae.
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
            controlnet_image = F.interpolate(controlnet_image, (512, 512), mode='bilinear', align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_512)

        B = latents.shape[0]

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)

            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)

            down_block_res_samples, mid_block_res_sample = self.controlnet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
                controlnet_cond=controlnet_image,
                conditioning_scale=conditioning_scale,
                return_dict=False,
            )
            
            noise_pred_pretrain = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
                down_block_additional_residuals=[
                    sample.to(dtype=self.precision_t) for sample in down_block_res_samples
                ],
                mid_block_additional_residual=mid_block_res_sample.to(dtype=self.weight_dtype),
            ).sample

            noise_pred_est = self.unet_lora(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
                down_block_additional_residuals=[
                    sample.to(dtype=self.precision_t) for sample in down_block_res_samples
                ],
                mid_block_additional_residual=mid_block_res_sample.to(dtype=self.weight_dtype),
                class_labels=torch.cat(
                    [
                        camera_condition.view(B, -1),
                        torch.zeros_like(camera_condition.view(B, -1)),
                    ],
                    dim=0,
                )
            ).sample

        noise_pred_pretrain_uncond, noise_pred_pretrain_text = noise_pred_pretrain.chunk(2)
        noise_pred_pretrain = noise_pred_pretrain_uncond + guidance_scale * (noise_pred_pretrain_text - noise_pred_pretrain_uncond)

        assert self.scheduler.config.prediction_type == "epsilon"
        if self.scheduler_lora.config.prediction_type == "v_prediction":
            alphas_cumprod = self.scheduler_lora.alphas_cumprod.to(
                device=latents_noisy.device, dtype=latents_noisy.dtype
            )
            alpha_t = alphas_cumprod[t] ** 0.5
            sigma_t = (1 - alphas_cumprod[t]) ** 0.5

            noise_pred_est = latent_model_input * torch.cat([sigma_t] * 2, dim=0).view(
                -1, 1, 1, 1
            ) + noise_pred_est * torch.cat([alpha_t] * 2, dim=0).view(-1, 1, 1, 1)
        
        noise_pred_est_uncond, noise_pred_est_text = noise_pred_est.chunk(2)
        noise_pred_est = noise_pred_est_uncond + guidance_scale_lora * (
            noise_pred_est_text - noise_pred_est_uncond
        )

        w = (1 - self.alphas[t])
        grad = w * (noise_pred_pretrain - noise_pred_est)

        if grad_clip is not None:
            grad = grad.clamp(-grad_clip, grad_clip)
        grad = torch.nan_to_num(grad)
        # print('grad:', grad)
        target = (latents - grad).detach()
        loss_vsd = 0.5 * F.mse_loss(latents, target, reduction="sum") / B
        loss_lora = self.train_lora(latents, text_embeddings, camera_condition, down_block_res_samples, mid_block_res_sample)

        return loss_vsd, loss_lora
    
    def decode_latents(self, latents):

        latents = 1 / self.vae_scaling_factor * latents

        with torch.no_grad():
            imgs = self.vae.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        
        return imgs
    
    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae_scaling_factor

        return latents