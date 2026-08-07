import torch
import torch.nn as nn
from models.drn import DRN_Light
from models.resnet_transformer import ResNetTransformer

class HybridModel(nn.Module):
    def __init__(self, transformer_path):
        super().__init__()

        # Load frozen transformer
        self.transformer = ResNetTransformer()
        self.transformer.load_state_dict(torch.load(transformer_path))
        self.transformer.eval()

        for p in self.transformer.parameters():
            p.requires_grad = False

        # Color branch
        self.color_branch = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU()
        )

        # Gradient branch
        self.grad_branch = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU()
        )

        # Frequency branch
        self.freq_branch = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU()
        )

        # Final DRN
        self.drn = DRN_Light(in_ch=192, num_classes=2)

    def compute_gradient(self, x):
        x = x.clone().detach().requires_grad_(True)
        logits = self.transformer(x)

        grad = torch.autograd.grad(
            outputs=logits.sum(),
            inputs=x,
            create_graph=False
        )[0]

        return grad.detach()

    def forward(self, x, xdwt):

        # Compute gradient map
        xg = self.compute_gradient(x)

        # Branch outputs
        Fx = self.color_branch(x)
        Fg = self.grad_branch(xg)
        Fr = self.freq_branch(xdwt)

        Fmap = torch.cat([Fx, Fg, Fr], dim=1)

        return self.drn(Fmap)