import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset


class PVDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file: path to train.csv / val.csv / test.csv
            root_dir: data/raw directory
            transform: torchvision transforms
        """

        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        # unique labels
        self.classes = sorted(self.data["label"].unique())

        # label → index mapping
        self.label_map = {label: idx for idx, label in enumerate(self.classes)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        img_path = row["image_path"]
        label = row["label"]

        image = Image.open(img_path).convert("RGB")

        label_idx = self.label_map[label]

        if self.transform:
            image = self.transform(image)

        return image, label_idx