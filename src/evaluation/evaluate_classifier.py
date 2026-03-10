import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np

from utils.logger import get_logger

logger = get_logger()

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from data.dataloader import get_dataloaders
from models.cnn_model import get_model


def evaluate():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader = get_dataloaders()

    model = get_model(12)

    model.load_state_dict(torch.load("model.pth", map_location=device))

    model.to(device)

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")
    f1 = f1_score(all_labels, all_preds, average="weighted")

    print("\nOverall Metrics")
    print("----------------")

    print("Accuracy:", acc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    logger.info("Starting evaluation")

    logger.info(f"Accuracy: {acc}")
    logger.info(f"Precision: {precision}")
    logger.info(f"Recall: {recall}")
    logger.info(f"F1 Score: {f1}")

    print("\nClassification Report")
    print("----------------")

    print(classification_report(all_labels, all_preds))

    print("\nConfusion Matrix")
    print("----------------")

    print(confusion_matrix(all_labels, all_preds))

    logger.info(f"Confusion Matrix:\n{confusion_matrix(all_labels, all_preds)}")


if __name__ == "__main__":
    evaluate()