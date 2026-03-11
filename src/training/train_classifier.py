import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.logger import get_logger
logger = get_logger()

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from data.dataloader import get_dataloaders
from models.cnn_model import get_model


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = get_dataloaders()

    num_classes = 12

    model = get_model(num_classes).to(device)

    # -------------------------------
    # Compute class weights
    # -------------------------------

    train_labels = train_loader.dataset.data["label"]

    label_map = train_loader.dataset.label_map

    numeric_labels = [label_map[l] for l in train_labels]

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=numeric_labels
    )

    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    logger.info(f"Class weights: {class_weights}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    epochs = 30

    logger.info("Training started")
    logger.info(f"Epochs: {epochs}")
    logger.info("Model: ResNet50")
    logger.info("Loss: Weighted CrossEntropyLoss")

    # -------------------------------
    # Training loop
    # -------------------------------

    for epoch in range(epochs):

        model.train()

        running_loss = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} Loss: {running_loss:.4f}")
        logger.info(f"Epoch {epoch+1}/{epochs} Loss: {running_loss:.4f}")

    # -------------------------------
    # Save model
    # -------------------------------

    torch.save(model.state_dict(), "model.pth")

    print("Model saved")
    logger.info("Model saved")


if __name__ == "__main__":
    main()