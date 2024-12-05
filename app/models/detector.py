import torch
import torch.nn as nn
import torchvision
from loguru import logger
from torchvision.models import resnet18, resnet34, resnet50
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator


def map_backbone(backbone: str):
    mapping = {"resnet18": resnet18, "resnet34": resnet34, "resnet50": resnet50}
    if backbone in mapping:
        return mapping[backbone]
    else:
        logger.error(
            f"[ERROR] Unsupported backbone: {backbone}. Using default: resnet18. Available options are: {list(mapping.keys())}"
        )
        return resnet18


class AerialDetector(nn.Module):
    def __init__(self, num_classes, backbone_name="resnet18"):
        super(AerialDetector, self).__init__()

        backbone_fn = map_backbone(backbone_name)
        backbone = backbone_fn(pretrained=True)

        modules = list(backbone.children())[:-2]
        self.backbone = nn.Sequential(*modules)

        if backbone_name in ["resnet18", "resnet34"]:
            self.backbone.out_channels = 512
        elif backbone_name == "resnet50":
            self.backbone.out_channels = 2048

        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),)
        )

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=["0"], output_size=7, sampling_ratio=2
        )

        self.model = FasterRCNN(
            self.backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
        )

    def forward(self, images, targets=None):
        return self.model(images, targets)
