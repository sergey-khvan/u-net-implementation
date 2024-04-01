import os

import cv2
from torch.utils.data import Dataset


class EMDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(self.img_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, flags=cv2.IMREAD_GRAYSCALE)
        img = img / 255
        mask[mask < 150.0] = 0.0
        mask[mask >= 150.0] = 1.0
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]
        return (img, mask)
