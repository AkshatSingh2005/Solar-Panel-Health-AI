import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import cv2

DATA_DIR = "data/raw"
METADATA_PATH = os.path.join(DATA_DIR, "module_metadata.json")

# Load metadata
with open(METADATA_PATH) as f:
    metadata = json.load(f)

records = []

# Parse dataset
for key, info in metadata.items():

    image_path = os.path.join(DATA_DIR, info["image_filepath"])
    label = info["anomaly_class"]

    records.append({
        "image_path": image_path,
        "label": label
    })

df = pd.DataFrame(records)

print("Dataset size:", len(df))

print("\nClass distribution:")
print(df["label"].value_counts())

# Show sample images
plt.figure(figsize=(10,6))

for i in range(6):

    img_path = df.iloc[i]["image_path"]
    label = df.iloc[i]["label"]

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.subplot(2,3,i+1)
    plt.imshow(img)
    plt.title(label)
    plt.axis("off")

plt.tight_layout()
plt.show()