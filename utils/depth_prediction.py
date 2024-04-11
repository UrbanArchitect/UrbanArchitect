import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
MIDAS_PATH=os.getenv('MIDAS_PATH')
import sys
sys.path.append(MIDAS_PATH)
from midas.model_loader import default_models, load_model
from transformers import DPTForDepthEstimation

class MiDasDepthPrediction(nn.Module):

    def __init__(self, 
                 model_path, 
                 model_type, 
                 device, 
                 optimize=False, 
                 side=False,
                 height=None,
                 square=False,
                 grayscale=False):
        super(MiDasDepthPrediction, self).__init__()
        self.device = device

        self.model, self.transform, self.net_w, self.net_h = load_model(
            device, model_path, model_type, optimize=False)
    
    def forward(self, img):
        '''
            img: (H,W,3)
        '''
        img_h = img.shape[0]
        img_w = img.shape[1]
        img = self.transform({"image":img})["image"]
        img = torch.from_numpy(img).to(torch.float32).to(self.device).unsqueeze(0)
        prediction = self.model.forward(img)
        prediction = prediction[None,...]
        prediction = F.interpolate(prediction, size=(img_h, img_w))

        return prediction[0]
