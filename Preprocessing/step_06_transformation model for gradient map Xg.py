import torch
import torch.nn as nn
import torchvision.models as models

class ResNetTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(weights=None)  # NOT pretrained
        self.model.fc = nn.Linear(2048, 2)

    def forward(self, x):
        return self.model(x)