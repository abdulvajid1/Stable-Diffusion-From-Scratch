from typing import Any, Optional

import torch
import torchvision
import torch.nn as nn
from torchvision.models import vgg19_bn, vgg16_bn, VGG19_BN_Weights, VGG16_BN_Weights

class LPIPS(nn.Module):
    def __init__(self, 
                 image_range='zero_to_one', 
                 use_dropout: bool = False) -> None:
        super().__init__()
        # Load & Set pretained model weights to No gradient on params
        pretrained_model_features = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT,
                                             progress=True).features
        for param in pretrained_model_features.parameters():
            param.requires_grad_(False)
        
        vgg_imagenet_scale_constants = [
            torch.tensor([0.485, 0.456, 0,405]), 
            torch.tensor([0.229, 0.224, 0.225])
        ]
             
        if image_range is "zero_to_one":
            self.register_buffer('mean', vgg_imagenet_scale_constants[0])
            self.register_buffer('std', vgg_imagenet_scale_constants[1])
        else:
            self.register_buffer("mean", vgg_imagenet_scale_constants[0] * 2 - 1)
            self.register_buffer('mean', vgg_imagenet_scale_constants[1] * 2)
            
        
        self.layer_groups = [
            (0, 2), (3, 9), (10, 16), (17, 22), (23, 29)
        ]
        
        self.section_layers = {}
        for i, (start, end) in enumerate(self.layer_groups):
            self.section_layers[f"section_{i}_layer"] = pretrained_model_features[start: end+1] # pyright: ignore[reportIndexIssue]
            
        self.section_layers = nn.ModuleDict(self.section_layers)
        
        self.section_out_channels = [
            64, 128, 256, 256, 512
        ]
        
        self.section_proj_layers = nn.ModuleDict()
        
        for i, in_channels in enumerate(self.section_out_channels):
            proj_layers = [nn.Dropout()] if use_dropout else []
            proj_layers += [nn.Conv2d(in_channels=in_channels, 
                                      out_channels=1, 
                                      kernel_size=1, 
                                      stride=1)]
            self.section_proj_layers[f'section_proj_{i}_layer'] = proj_layers # pyright: ignore[reportArgumentType]
        
        self.pool_layer = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    
    def forward_vgg(self, x: torch.Tensor):
        slice1 = self.section_layers["section_1_layer"](x)
        slice2 = self.section_layers["section_2_layer"](slice1)
        slice3 = self.section_layers["section_3_layer"](slice2)
        slice4 = self.section_layers["section_4_layer"](slice3)
        slice5 = self.section_layers["section_5_layer"](slice4)
        
        return {
            "slice1": slice1,
            "slice2": slice2,
            "slice3": slice3,
            "slice4": slice4,
            "slice5": slice5
        }
    
    def scale(self, x):
        return (x - self.mean) / self.std

    def _unit_norm(self, x, epsilon=1e-8):
        return x / torch.norm(x, p=2, dim=1, keepdim=True) + epsilon
    
    def forward(self, input, target):
        input, target = self.scale(input), self.scale(target)
        
        if not self.train_backbone:
            with torch.no_grad():
                input_features = self.forward_vgg(input)
                target_features = self.forward_vgg(target)
        else:
            input_features = self.forward_vgg(input)
            target_features = self.forward_vgg(input)
        
        self.pooled_outs = []
        for i, key in enumerate(input_features.keys()):
            input = input_features[key]
            target = target_features[key]
            
            # Norm & substract as in paper
            input, target = self._unit_norm(input), self._unit_norm(input)
            delta = (input - target) ** 2
            
            proj_out = self.section_proj_layers[f"section_proj_{i}_layer"](delta) # Bxcxhxw -> bx1xcxwxh
            pooled_out = self.pool_layer(proj_out) # bx1x1x1
            self.pooled_outs.append(pooled_out)
            
        
        val = 0
        for i in self.pooled_outs:
            val += i
        
        return val # bx1x1x1
    

class LpipsDiffToLogits(nn.Module):
    def __init__(self, middle_channel=32) -> None:
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=middle_channel, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=middle_channel, out_channels=middle_channel, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=middle_channel, out_channels=1, kernel_size=1)
        )
    
    def forward(self, diff1, diff2, eps=1e-8): 
        diff = diff1 - diff2
        ratio1 = diff1 / (diff2 + eps)
        ratio2 = diff2 / (diff1 + eps)
        x_cat = torch.concat([diff1, diff2, diff, ratio1, ratio2], dim=1)
        return self.model(x_cat)
        
        
        
        
        
        
        



if __name__ == "__main__":
    lpips = LPIPS()
    print("Succesfull")
        
        