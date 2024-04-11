import nntplib
from tkinter import Image
from turtle import forward
import detectron2
import sys
import os
ONEFORMER_PATH=os.getenv('ONEFORMER_PATH')
sys.path.append(ONEFORMER_PATH)
import numpy as np
import torch
import torch.nn as nn
import cv2

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog

from demo.defaults import DefaultPredictor, DefaultPredictorTrainer
from demo.visualizer import Visualizer, ColorMode

from oneformer import (
    add_oneformer_config,
    add_common_config,
    add_swin_config,
    add_dinat_config,
    add_convnext_config,
)

SWIN_CFG_DICT = {"cityscapes": f"{ONEFORMER_PATH}/configs/cityscapes/oneformer_swin_large_IN21k_384_bs16_90k.yaml",
                "coco": f"{ONEFORMER_PATH}/configs/coco/oneformer_swin_large_IN21k_384_bs16_100ep.yaml",
                "ade20k": f"{ONEFORMER_PATH}/configs/ade20k/oneformer_swin_large_IN21k_384_bs16_160k.yaml",}

DINAT_CFG_DICT = {"cityscapes": f"{ONEFORMER_PATH}/configs/cityscapes/convnext/mapillary_pretrain_oneformer_convnext_large_bs16_90k.yaml",
            "coco": f"{ONEFORMER_PATH}/configs/coco/oneformer_dinat_large_bs16_100ep.yaml",
            "ade20k": f"{ONEFORMER_PATH}/configs/ade20k/oneformer_dinat_large_IN21k_384_bs16_160k.yaml",}

cpu_device = torch.device("cpu")

def setup_cfg(dataset, model_path, use_swin, device='cuda'):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_common_config(cfg)
    add_swin_config(cfg)
    add_dinat_config(cfg)
    add_convnext_config(cfg)
    add_oneformer_config(cfg)
    if use_swin:
        cfg_path = SWIN_CFG_DICT[dataset]
    else:
        cfg_path = DINAT_CFG_DICT[dataset]
    cfg.merge_from_file(cfg_path)
    cfg.MODEL.DEVICE = device
    cfg.MODEL.WEIGHTS = model_path
    cfg.freeze()
    return cfg

def setup_modules(dataset, model_path, use_swin, device='cuda'):
    cfg = setup_cfg(dataset, model_path, use_swin, device)
    # predictor = DefaultPredictor(cfg)
    predictor = DefaultPredictorTrainer(cfg)
    
    metadata = MetadataCatalog.get(
        cfg.DATASETS.TEST_PANOPTIC[0] if len(cfg.DATASETS.TEST_PANOPTIC) else "__unused"
    )
    if 'cityscapes_fine_sem_seg_val' in cfg.DATASETS.TEST_PANOPTIC[0]:
        from cityscapesscripts.helpers.labels import labels
        stuff_colors = [k.color for k in labels if k.trainId != 255]
        metadata = metadata.set(stuff_colors=stuff_colors)
    return predictor, metadata

def panoptic_run(img, predictor, metadata):
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, instance_mode=ColorMode.IMAGE)
    predictions = predictor(img, "panoptic")
    panoptic_seg, segments_info = predictions["panoptic_seg"]
    out = visualizer.draw_panoptic_seg_predictions(
    panoptic_seg.to(cpu_device), segments_info, alpha=0.5
)
    return predictions, out

def instance_run(img, predictor, metadata):
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, instance_mode=ColorMode.IMAGE)
    predictions = predictor(img, "instance")
    instances = predictions["instances"].to(cpu_device)
    out = visualizer.draw_instance_predictions(predictions=instances, alpha=0.5)
    return predictions, out

def semantic_run(img, predictor, metadata):
    # visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, instance_mode=ColorMode.IMAGE)
    predictions = predictor(img, "semantic")
    out = None
    return predictions, out

class SemSegOneformer(nn.Module):
    def __init__(self, model_path):
        super(SemSegOneformer, self).__init__()
        use_swin = False
        self.predictor, self.metadata = setup_modules("cityscapes", model_path, use_swin)
    
    def forward(self, image):
        '''
            image: np.ndarray, cv2.BGR
        Ret:
            sem_id: [H,W]
        '''
        predictions = self.predictor(image, 'semantic')
        semantics = predictions['sem_seg']
        return semantics