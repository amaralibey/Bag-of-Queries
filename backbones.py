import torch
import torchvision

class ResNet(torch.nn.Module):
    def __init__(
        self,
        backbone_name="resnet50",
        crop_last_block=True,
    ):
        super().__init__()
        
        self.crop_last_block = crop_last_block

        
        if "18" in backbone_name:
            model = torchvision.models.resnet18()
        elif "34" in backbone_name:
            model = torchvision.models.resnet34()
        elif "50" in backbone_name:
            model = torchvision.models.resnet50()
        elif "101" in backbone_name:
            model = torchvision.models.resnet101()
        else:
            raise NotImplementedError("Backbone architecture not recognized!")


        # create backbone with only the necessary layers
        self.net = torch.nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            *([] if crop_last_block else [model.layer4]),
        )

        # calculate output channels
        out_channels = 2048
        if "34" in backbone_name or "18" in backbone_name:
            out_channels = 512

        self.out_channels = out_channels // 2 if crop_last_block else out_channels

    def forward(self, x):
        x = self.net(x)
        return x

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
        num_unfrozen_blocks=2,
    ):
        super().__init__()
        
        self.backbone_name = backbone_name
        self.num_unfrozen_blocks = num_unfrozen_blocks
        
        # make sure the backbone_name is in the available models
        if self.backbone_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Backbone {self.backbone_name} is not recognized!" 
                             f"Supported backbones are: {self.AVAILABLE_MODELS}")
                             
                
        self.dino = torch.hub.load('facebookresearch/dinov2', self.backbone_name)
        
        # freeze the patch embedding and positional encoding
        self.dino.patch_embed.requires_grad_(False)
        self.dino.pos_embed.requires_grad_(False)
        
        # freeze the first blocks, keep only the last num_unfrozen_blocks trainable
        for i in range(len(self.dino.blocks) - self.num_unfrozen_blocks):
            self.dino.blocks[i].requires_grad_(False)
        
        self.out_channels = self.dino.embed_dim

    def forward(self, x):
        B, _, H, W = x.shape
        # No need to compute gradients for frozen layers
        with torch.no_grad():
            x = self.dino.prepare_tokens_with_masks(x)
            for blk in self.dino.blocks[ : -self.num_unfrozen_blocks]:
                x = blk(x)
        x = x.detach()

        # Last blocks are trained
        for blk in self.dino.blocks[-self.num_unfrozen_blocks : ]:
            x = blk(x)
            
        
        x = x[:, 1:] # remove the [CLS] token
        
        # reshape the output tensor to B, C, H, W
        _, _, C = x.shape # we know C == self.dino.embed_dim, but still...
        x = x.permute(0, 2, 1).contiguous().view(B, C, H//14, W//14)
        return x