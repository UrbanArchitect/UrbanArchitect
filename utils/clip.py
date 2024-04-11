import torch
from transformers import CLIPProcessor, CLIPModel
import torch.nn as nn
import torch.nn.functional as F

class CLIPScore(nn.Module):
    def __init__(self, model_path, device):
        super(CLIPScore, self).__init__()

        self.model = CLIPModel.from_pretrained(model_path, local_files_only=True)
        self.processor = CLIPProcessor.from_pretrained(model_path, local_files_only=True)
        self.device = device
        self.model.to(device)
    
    @torch.no_grad()
    def forward(self, image, text):
        '''
            img: PIL Image
            text: ['text']
        '''
        inputs = self.processor(text=text, images=image, return_tensors="pt", padding=True)
        for key in inputs.keys():
            inputs[key] = inputs[key].to(self.device)
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        logits = logits_per_image.squeeze().item()

        return logits, outputs
    
    def get_image_features(self, images):
        images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        images = images * 2 - 1
        image_features = self.model.get_image_features(pixel_values=images)
        return image_features