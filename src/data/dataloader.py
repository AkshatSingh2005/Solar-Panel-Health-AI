import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data.pv_dataset import PVDataset



def get_dataloaders():

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    train_dataset = PVDataset(
        csv_file="data/processed/train.csv",
        root_dir=None,
        transform=transform
    )

    val_dataset = PVDataset(
        csv_file="data/processed/val.csv",
        root_dir=None,
        transform=transform
    )

    test_dataset = PVDataset(
        csv_file="data/processed/test.csv",
        root_dir=None,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, test_loader