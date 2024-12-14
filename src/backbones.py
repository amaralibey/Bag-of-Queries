import torch
import torch.nn as nn
import torchvision

class DinoV2(torch.nn.Module):
    AVAILABLE_MODELS = [
        'dinov2_vits14',
        'dinov2_vitb14',
        'dinov2_vitl14',
        'dinov2_vitg14'
    ]
    
    def __init__(
        self,
        backbone_name="dinov2_vitb14",
        unfreeze_n_blocks=2,
    ):
        super().__init__()
        
        self.backbone_name = backbone_name
        self.unfreeze_n_blocks = unfreeze_n_blocks
        
        # make sure the backbone_name is in the available models
        if self.backbone_name not in self.AVAILABLE_MODELS:
            print(f"Backbone {self.backbone_name} is not recognized!, using dinov2_vitb14")
            self.backbone_name = "dinov2_vitb14"                             
                
        self.dino = torch.hub.load('facebookresearch/dinov2', self.backbone_name)
        
        # freeze the patch embedding and positional encoding
        self.dino.patch_embed.requires_grad_(False)
        self.dino.pos_embed.requires_grad_(False)
        
        # freeze the first blocks, keep only the last unfreeze_n_blocks trainable
        for i in range(len(self.dino.blocks) - self.unfreeze_n_blocks):
            self.dino.blocks[i].requires_grad_(False)
        
        self.out_channels = self.dino.embed_dim

    def forward(self, x):
        B, _, H, W = x.shape
        # No need to compute gradients for frozen layers
        with torch.no_grad():
            x = self.dino.prepare_tokens_with_masks(x)
            for blk in self.dino.blocks[ : -self.unfreeze_n_blocks]:
                x = blk(x)

        # Last blocks are trained
        for blk in self.dino.blocks[-self.unfreeze_n_blocks : ]:
            x = blk(x)
            
        
        x = x[:, 1:] # remove the [CLS] token
        
        # reshape the output tensor to B, C, H, W
        _, _, C = x.shape # we know C == self.dino.embed_dim, but still...
        x = x.permute(0, 2, 1).contiguous().view(B, C, H//14, W//14)
        return x
    
    
class ResNet(nn.Module):
    AVAILABLE_MODELS = {
        "resnet18": torchvision.models.resnet18,
        "resnet34": torchvision.models.resnet34,
        "resnet50": torchvision.models.resnet50,
        "resnet101": torchvision.models.resnet101,
        "resnet152": torchvision.models.resnet152,
        "resnext50": torchvision.models.resnext50_32x4d,
    }

    def __init__(
        self,
        backbone_name="resnet50",
        pretrained=True,
        unfreeze_n_blocks=1,
        crop_last_block=True,
    ):
        """Class representing the resnet backbone used in the pipeline.
        
        Args:
            backbone_name (str): The architecture of the resnet backbone to instantiate.
            pretrained (bool): Whether the model is pretrained or not.
            unfreeze_n_blocks (int): The number of residual blocks to unfreeze (starting from the end).
            crop_last_block (bool): Whether to crop the last residual block.
        
        Raises:
            ValueError: if the backbone_name corresponds to an unknown architecture.
        """
        super().__init__()

        

        self.backbone_name = backbone_name
        self.pretrained = pretrained
        self.unfreeze_n_blocks = unfreeze_n_blocks
        self.crop_last_block = crop_last_block

        if backbone_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Backbone {backbone_name} is not recognized!" 
                             f"Supported backbones are: {list(self.AVAILABLE_MODELS.keys())}")

        # Load the model
        weights = "IMAGENET1K_V1" if pretrained else None
        resnet = self.AVAILABLE_MODELS[backbone_name](weights=weights)

        all_layers = [
            nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool),
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        ]
        
        if crop_last_block:
            all_layers.remove(resnet.layer4)
        nb_layers = len(all_layers)

        # Check if the number of unfrozen blocks is valid
        assert (
            isinstance(unfreeze_n_blocks, int) and 0 <= unfreeze_n_blocks <= nb_layers
        ), f"unfreeze_n_blocks must be an integer between 0 and {nb_layers} (inclusive)"

        if pretrained:
            # Split the resnet into frozen and unfrozen parts
            self.frozen_layers = nn.Sequential(*all_layers[:nb_layers - unfreeze_n_blocks])
            self.unfrozen_layers = nn.Sequential(*all_layers[nb_layers - unfreeze_n_blocks:])
            
            # this is helful to make PyTorch count the right number of trainable params
            # because it doesn't detect the torch.no_grad() context manager at init time
            self.frozen_layers.requires_grad_(False)
        else:
            # If the model is not pretrained, we keep all layers trainable
            if self.unfreeze_n_blocks > 0:
                print("Warning: unfreeze_n_blocks is ignored when pretrained=False. Setting it to 0.")
                self.unfreeze_n_blocks = 0
            self.frozen_layers = nn.Identity()
            self.unfrozen_layers = nn.Sequential(*all_layers)
        
        # Calculate the output channels from the last conv layer of the model
        if backbone_name in ["resnet18", "resnet34"]:
            self.out_channels = all_layers[-1][-1].conv2.out_channels
        else:
            self.out_channels = all_layers[-1][-1].conv3.out_channels
        
       
    def forward(self, x):
        with torch.no_grad():
            x = self.frozen_layers(x)
        
        x = self.unfrozen_layers(x)
        return x