import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class MyDataset(Dataset):
    def __init__(self, image_dir, pos_dir, transform):
        self.image_dir = image_dir
        self.pos_dir = pos_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.pos = os.listdir(pos_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        pos_path = os.path.join(self.pos_dir, self.pos[index])
        image = np.array(Image.open(img_path).convert("RGB"))
        pos = np.array(Image.open(pos_path).convert("RGB"))

        #if self.transform is not None:
        augmentations = self.transform(image=image)
        image = augmentations["image"]
        augmentations = self.transform(image=pos)
        pos = augmentations["image"]

        return image, pos
