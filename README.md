# Urban Architect: Steerable 3D Urban Scene Generation with Layout Prior

[![arXiv](https://img.shields.io/badge/arXiv-2404.06780-b31b1b.svg)](https://arxiv.org/abs/2404.06780) <a href="https://urbanarchitect.github.io/">
<img alt="Project" src="https://img.shields.io/badge/-Project%20Page-lightgrey?logo=Google%20Chrome&color=informational&logoColor=white"></a> 
<a href="https://www.youtube.com/watch?v=CEPquApsPjI&ab_channel=FanLu"><img alt="Demo" src="https://img.shields.io/badge/-Demo-ea3323?logo=youtube"></a> 




https://github.com/UrbanArchitect/UrbanArchitect/assets/13656744/ffd06914-f522-44a2-9a44-93a516506e36


## Introduction
The repository contains the official implementation of source code and pre-trained models of our paper:*"[Urban Architect: Steerable 3D Urban Scene Generation with Layout Prior]()"*. It is a novel pipeline for large-scale 3D urban scene generation!

## Updates
- 2024.04.26: We release samples of layout in `dataset/data` and pretrained [ControlNet weights](https://drive.google.com/file/d/1IdhzxrwfqZ7yr5Ka1VeVV70J8q29YX-1/view?usp=sharing)! 
- 2024.04.11: The :fire::fire::fire: **[pre-print](https://arxiv.org/abs/2404.06780)** :fire::fire::fire: is released! Refer to it for more details!
- 2024.04.10: The [project page](https://urbanarchitect.github.io/) is created. Check it out for an overview of our work!

## Environments
1. Main requirements:
- PyTorch (tested on torch-2.0)
- [pytorch3d](https://pytorch3d.org/)
- [diffusers](https://github.com/huggingface/diffusers/tree/main)
2. Download the pretrained model weights of [Stable Diffusion](https://huggingface.co/stabilityai/stable-diffusion-2-1)
3. Download the pretrained model weights of [CLIP](https://huggingface.co/openai/clip-vit-base-patch32)
4. Download the pretrained model weights of [ControlNet]()
5. Other requirements are provided in `requirements.txt`

## Generate a Scene

### Coarse stage
Please refer to `train.sh`.

### Refinement stage
Please refer to `refine.sh`.

### Render video
Please refer to `render.sh`.

## To-Do List

- [ ] Release 3D layout data
- [x] Technical Report
- [x] Project page

## Related Work
* (ICCV 2023) **Urban Radiance Field Representation with Deformable Neural Mesh Primitives**, Fan Lu et al. [[Paper](https://arxiv.org/abs/2307.10776)], [[Project Page](https://dnmp.github.io/)]


## Citation
If you find this project useful for your work, please consider citing:
```
@article{lu2024urban,
  title={Urban Architect: Steerable 3D Urban Scene Generation with Layout Prior},
  author={Lu, Fan and Lin, Kwan-Yee and Xu, Yan and Li, Hongsheng and Chen, Guang and Jiang, Changjun},
  journal={arXiv preprint arXiv:2404.06780},
  year={2024}
}
```
