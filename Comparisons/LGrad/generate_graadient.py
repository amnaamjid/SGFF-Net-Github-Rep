import os
import cv2
import torch
import numpy as np

from tqdm import tqdm
from PIL import Image

from torchvision import models
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_ROOT = "/home/amna/HPC_DATA/GAN/Same-Training/RGB/Same-Training-RGB"

OUTPUT_ROOT = "/home/amna/HPC_DATAGAN/Same-Training/RGB/Same-Training_Gradient"

IMAGE_SIZE = 224

BATCH_SIZE = 32

NUM_WORKERS = 8

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# IMAGE PREPROCESSING
# (Exactly as LGrad)
# ============================================================

transform = transforms.Compose([

    transforms.Resize((224,224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5,0.5,0.5],
        std=[0.5,0.5,0.5]
    )

])


# ============================================================
# TRANSFORMATION MODEL
# (ImageNet Pretrained ResNet50)
# ============================================================

transform_model = models.resnet50(
    weights=models.ResNet50_Weights.IMAGENET1K_V2
)

transform_model.to(DEVICE)

transform_model.eval()

for p in transform_model.parameters():

    p.requires_grad=False


# ============================================================
# NORMALIZATION FUNCTION
# (Exactly same as paper)
# ============================================================

def normalize_gradient(g):

    g = g - g.min()

    if g.max() > 0:

        g = g / g.max()

    g = g * 255.0

    return g.astype(np.uint8)


# ============================================================
# GENERATE GRADIENTS
# ============================================================

splits = ["Train","Val","Test"]


for split in splits:

    print(f"\nProcessing {split}")

    dataset = ImageFolder(

        root=os.path.join(INPUT_ROOT,split),

        transform=transform

    )


    loader = DataLoader(

        dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=True

    )


    image_index = 0


    for images,labels in tqdm(loader):

        images = images.to(DEVICE)

        images.requires_grad_(True)


        transform_model.zero_grad()


        outputs = transform_model(images)


        score = outputs.sum()


        gradients = torch.autograd.grad(

            outputs=score,

            inputs=images,

            create_graph=False,

            retain_graph=False

        )[0]


        gradients = gradients.detach().cpu()


        batch_size = gradients.size(0)


        for i in range(batch_size):

            original_path = dataset.samples[image_index][0]

            class_name = os.path.basename(
                os.path.dirname(original_path)
            )

            filename = os.path.basename(original_path)

            filename = os.path.splitext(filename)[0] + ".jpeg"


            save_dir = os.path.join(

                OUTPUT_ROOT,

                split,

                class_name

            )

            os.makedirs(

                save_dir,

                exist_ok=True

            )


            grad = gradients[i].permute(1,2,0).numpy()

            grad = normalize_gradient(grad)


            cv2.imwrite(

                os.path.join(save_dir,filename),

                cv2.cvtColor(grad,cv2.COLOR_RGB2BGR)

            )


            image_index += 1


print("\nGradient generation completed successfully.")