from torchvision import transforms
import torch

class DualTransform:

    def __init__(self, dwt_transform, rgb_transform=None):

        self.dwt_transform = dwt_transform
        self.rgb_transform = rgb_transform
        self.to_tensor = transforms.ToTensor()

    def __call__(self, img):

        # Apply augmentation (should return PIL)
        if self.rgb_transform is not None:
            aug_img = self.rgb_transform(img)
        else:
            aug_img = img

        # RGB tensor
        rgb = self.to_tensor(aug_img)

        # DWT computed on same augmented image
        dwt = self.dwt_transform(aug_img)

        return (rgb, dwt)