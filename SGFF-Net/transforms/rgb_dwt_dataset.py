import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class RGBDWTDataset(Dataset):

    def __init__(self, rgb_dir, dwt_dir):

        self.rgb_dir = rgb_dir
        self.dwt_dir = dwt_dir

        self.classes = ["Real","Fake"]

        self.samples = []

        for label,cls in enumerate(self.classes):

            rgb_cls = os.path.join(rgb_dir,cls)
            files = os.listdir(rgb_cls)

            for f in files:

                rgb_path = os.path.join(rgb_cls,f)
                dwt_path = os.path.join(dwt_dir,cls,f)

                self.samples.append((rgb_path,dwt_path,label))

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,idx):

        rgb_path,dwt_path,label = self.samples[idx]

        rgb = Image.open(rgb_path).convert("RGB")
        dwt = Image.open(dwt_path).convert("RGB")

        rgb = self.to_tensor(rgb)
        dwt = self.to_tensor(dwt)

        return (rgb,dwt), label
