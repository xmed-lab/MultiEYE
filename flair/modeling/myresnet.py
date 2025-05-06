from typing import Any, List, Optional, Type, Union

import torch
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck, ResNet50_Weights
from torchvision.models._api import WeightsEnum, register_model


class MyResNet(ResNet):
    def forward_inter(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x1 = torch.flatten(self.avgpool(x), 1)
        x = self.layer2(x)
        x2 = torch.flatten(self.avgpool(x), 1)
        x = self.layer3(x)
        x3 = torch.flatten(self.avgpool(x), 1)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x4 = x
        x = self.fc(x)

        return x, x1, x2, x3, x4

def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = MyResNet(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model

@register_model()
@handle_legacy_interface(weights=("pretrained", ResNet50_Weights.IMAGENET1K_V1))
def resnet50(*, weights: Optional[ResNet50_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    weights = ResNet50_Weights.verify(weights)

    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)
