import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================
# IBN BLOCK
# =====================================================
class IBN(nn.Module):
    """
    IBN: Half InstanceNorm, Half BatchNorm
    """
    def __init__(self, channels):
        super().__init__()
        half = channels // 2
        self.IN = nn.InstanceNorm2d(half, affine=True)
        self.BN = nn.BatchNorm2d(channels - half)

    def forward(self, x):
        c = x.size(1)
        x1, x2 = torch.split(x, [c // 2, c - c // 2], dim=1)
        return torch.cat([self.IN(x1), self.BN(x2)], dim=1)


# =====================================================
# UNI-RESIDUAL BLOCKS
# =====================================================
class UniResidualBlockIBN(nn.Module):
    """
    Uni-Residual Block with IBN (used in early layers)
    """
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.ibn = IBN(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return x + self.relu(self.ibn(self.conv(x)))


class UniResidualBlock(nn.Module):
    """
    Uni-Residual Block with BatchNorm (used in deeper layers)
    """
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return x + self.relu(self.bn(self.conv(x)))


# =====================================================
# DUAL RESIDUAL BLOCK (KEEP AS IS)
# =====================================================
class DualResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()

        # Local path
        self.local = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        # Global path
        self.global_path = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.proj = None
        if stride != 1 or in_ch != out_ch:
            self.proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.local(x) + self.global_path(x)
        res = x if self.proj is None else self.proj(x)
        return self.relu(out + res)


# =====================================================
# 🔥 LIGHTWEIGHT DRN (FINAL)
# =====================================================
class DRN_Light(nn.Module):
    def __init__(self, in_ch=3, num_classes=2):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
        self.ibn1 = IBN(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        # Early strong modeling (KEEP)
        self.uni1 = nn.Sequential(
            UniResidualBlockIBN(64),
            UniResidualBlockIBN(64),
        )

        self.dual1 = DualResidualBlock(64, 128, stride=2)

        # Reduced depth (SAFE)
        self.uni2 = UniResidualBlock(128)

        self.dual2 = DualResidualBlock(128, 256, stride=2)

        self.uni3 = UniResidualBlock(256)

        # Lightweight head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_out = nn.Linear(256, num_classes)

    def forward(self, x, return_features=False):
        x = self.conv1(x)
        x = self.ibn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.uni1(x)
        x = self.dual1(x)
        x = self.uni2(x)
        x = self.dual2(x)
        x = self.uni3(x)

        x = self.avgpool(x)
        feat = torch.flatten(x, 1)   # (B, 256)

        logits = self.fc_out(feat)

        if return_features:
            return logits, feat
        return logits


# =====================================================
# SUPERVISED CONTRASTIVE LOSS
# =====================================================
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        features = F.normalize(features, dim=1)
        device = features.device

        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        sim = torch.matmul(features, features.T) / self.temperature
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        exp_sim = torch.exp(sim) * (1 - torch.eye(sim.size(0)).to(device))
        log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        return -mean_log_prob_pos.mean()


# =====================================================
# TRAIN STEP — CE + SUPCON
# =====================================================
def train_step(model, batch, optimizer, device, supcon_loss, alpha=0.3):
    model.train()
    images, labels = batch
    images = images.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()

    logits, features = model(images, return_features=True)

    loss_ce = F.cross_entropy(logits, labels)
    loss_con = supcon_loss(features, labels)

    loss = loss_ce + alpha * loss_con
    loss.backward()
    optimizer.step()

    return loss.item(), loss_ce.item(), loss_con.item()


# =====================================================
# EXAMPLE MAIN
# =====================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DRN_Light(in_ch=3, num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    supcon_loss = SupConLoss(temperature=0.07)

    print("✅ Lightweight DRN + IBN + CE + SupCon READY")
