# Urban Architect: Steerable 3D Urban Scene Generation with Layout Prior

[![arXiv](https://img.shields.io/badge/arXiv-2307.10173-b31b1b.svg)](https://arxiv.org/abs/2404.06780) <a href="https://urbanarchitect.github.io/">
<img alt="Project" src="https://img.shields.io/badge/-Project%20Page-lightgrey?logo=Google%20Chrome&color=informational&logoColor=white"></a> 
<a href="https://youtu.be/_TQQoQLnvPc"><img alt="Demo" src="https://img.shields.io/badge/-Demo-ea3323?logo=youtube"></a> 

## Introduction
The repository contains the official implementation of source code and pre-trained models of our paper:*"[Urban Architect: Steerable 3D Urban Scene Generation with Layout Prior]()"*. It is a novel pipeline for large-scale 3D urban scene generation!

## Updates
- 2024.04.11: The:fire::fire::fire:**[pre-print](https://arxiv.org/abs/2404.06780)**:fire::fire::fire:is released! Refer to it for more details!
- 2024.04.10: The [project page](https://urbanarchitect.github.io/) is created. Check it out for an overview of our work!

## Environments
1. Main requirements:
- PyTorch (tested on torch-2.0)
- [pytorch3d](https://pytorch3d.org/)
- [diffusers](https://github.com/huggingface/diffusers/tree/main)
2. Download the pretrained model weights of [Stable Diffusion](https://huggingface.co/stabilityai/stable-diffusion-2-1)
3. Download the pretrained model weights of [CLIP](https://huggingface.co/openai/clip-vit-base-patch32)
4. Other requirements are provided in `requirements.txt`

## Generate a Scene

### Coarse stage
Please refer to `train.sh`.

### Refinement stage
Please refer to `refine.sh`.

### Render video
Please refer to `render.sh`.