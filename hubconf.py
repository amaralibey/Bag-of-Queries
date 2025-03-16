dependencies = ['torch', 'torchvision']

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))  # Add repo root to Python path

import torch
from src.backbones import ResNet, DinoV2
from src.boq import BoQ
    

class VPRModel(torch.nn.Module):
    def __init__(self, 
                 backbone,
                 aggregator):
        super().__init__()
        self.backbone = backbone
        self.aggregator = aggregator
        
    def forward(self, x):
        x = self.backbone(x)
        x, attns = self.aggregator(x)
        return x, attns


AVAILABLE_BACKBONES = {
    # this list will be extended
    # "resnet18": [8192 , 4096],
    "resnet50": [16384],
    "dinov2": [12288],
}

MODEL_URLS = {
    "resnet50_16384": "https://github.com/amaralibey/Bag-of-Queries/releases/download/v1.0/resnet50_16384.pth",
    "dinov2_12288": "https://github.com/amaralibey/Bag-of-Queries/releases/download/v1.0/dinov2_12288.pth",
    # "resnet50_4096": "",
}

def get_trained_boq(backbone_name="resnet50", output_dim=16384):
    if backbone_name not in AVAILABLE_BACKBONES:
        raise ValueError(f"backbone_name should be one of {list(AVAILABLE_BACKBONES.keys())}")
    try:
        output_dim = int(output_dim)
    except:
        raise ValueError(f"output_dim should be an integer, not a {type(output_dim)}")
    if output_dim not in AVAILABLE_BACKBONES[backbone_name]:
        raise ValueError(f"output_dim should be one of {AVAILABLE_BACKBONES[backbone_name]}")
    
    if "dinov2" in backbone_name:
        # load the backbone
        backbone = DinoV2()
        # load the aggregator
        aggregator = BoQ(
            in_channels=backbone.out_channels,  # make sure the backbone has out_channels attribute
            proj_channels=384,
            num_queries=64,
            num_layers=2,
            row_dim=output_dim//384, # 32 for dinov2
        )
        
    elif "resnet" in backbone_name:
        backbone = ResNet(
                backbone_name=backbone_name,
                crop_last_block=True,
            )
        aggregator = BoQ(
                in_channels=backbone.out_channels,  # make sure the backbone has out_channels attribute
                proj_channels=512,
                num_queries=64,
                num_layers=2,
                row_dim=output_dim//512, # 32 for resnet
            )

    vpr_model = VPRModel(
            backbone=backbone,
            aggregator=aggregator
        )
    
    vpr_model.load_state_dict(
        torch.hub.load_state_dict_from_url(
            MODEL_URLS[f"{backbone_name}_{output_dim}"],
            map_location=torch.device('cpu')
        )
    )
    return vpr_model
