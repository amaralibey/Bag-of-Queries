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

