dependencies = ['torch', 'torchvision']

import torch
from backbones import ResNet
from boq import BoQ
    

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
    "resnet50": [16384],
    # "resnet18": [8192 , 4096],
    # "dinov2": [8192],
}

MODEL_URLS = {
    "resnet50_16384": "",
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
    
    # load the backbone
    backbone = ResNet(
            backbone_name=backbone_name,
            crop_last_block=True,
        )

    aggregator = BoQ(
            in_channels=backbone.out_channels,  # make sure the backbone has out_channels attribute
            proj_channels=512,
            num_queries=64,
            num_layers=2,
            row_dim=output_dim//512, # the output dimension will be proj_channels*row_dim
        )

    vpr_model = VPRModel(
            backbone=backbone,
            aggregator=aggregator
        )
    
    # vpr_model.load_state_dict(torch.hub.load_state_dict_from_url(MODEL_URLS[f"{backbone_name}_{output_dim}"]))
    return vpr_model


if __name__ == "__main__":
    model = get_trained_boq("resnet50", 16384)
    print('Testing model with a mock input of shape (4, 3, 320, 320)')
    x = torch.randn(4, 3, 320, 320)
    o, attns = model(x)
    print(f"output descriptors {o.shape}")
    print(f"attention shape of first BoQ block: {attns[0].shape}")
