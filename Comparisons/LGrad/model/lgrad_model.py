import torch.nn as nn
import torchvision.models as models


class LGradClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2
        )

        in_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Linear(
            in_features,
            2
        )

    def forward(self,x):

        return self.backbone(x)