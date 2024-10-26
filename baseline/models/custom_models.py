import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from semseg.models.heads import UPerHead
from semseg.models.backbones import *
from semseg.models.layers import trunc_normal_
import math


class CustomVIT(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        backbone_class, backbone_name = self.config["train_model"]["backbone"].split(sep='-')
        in_channels = self.config["train_model"]["num_channels"]
        self.backbone = eval(backbone_class)(in_channels, model_name=backbone_name)
        self.decode_head = UPerHead(self.backbone.channels, 192, 1)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out // m.groups  # TODO wtf??
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        y = self.backbone(x)
        y = self.decode_head(y)   # 4x reduction in image size
        y = F.sigmoid(y)
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)    # to original image shape
        return y
