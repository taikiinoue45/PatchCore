from typing import Tuple

import torch
from torch import Tensor
from torch.nn import Module
from torchvision.models import wide_resnet50_2


class WideResNet50(Module):
    def __init__(self) -> None:

        super().__init__()
        self.wide_resnet50 = wide_resnet50_2(pretrained=True)
        self.wide_resnet50.layer2[-1].register_forward_hook(self.layer2_hook)
        self.wide_resnet50.layer3[-1].register_forward_hook(self.layer3_hook)
        self.avg_pool_2d = torch.nn.AvgPool2d(3, 1, 1)

    def layer2_hook(self, module: Module, x: Tensor, y: Tensor) -> None:

        self.feature2 = self.avg_pool_2d(y)

    def layer3_hook(self, module: Module, x: Tensor, y: Tensor) -> None:

        self.feature3 = self.avg_pool_2d(y)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:

        self.wide_resnet50(x)
        return (self.feature2, self.feature3)
