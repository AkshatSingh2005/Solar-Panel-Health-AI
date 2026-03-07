import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = "data/raw"
METADATA_PATH = os.path.join(DATA_DIR, "module_metadata.json")

# Load metadata
with open(METADATA_PATH) as f:
    metadata = json.load(f)

records = []

for key, info in metadata.items():

    image_path = os.path.join(DATA_DIR, info["image_filepath"])
    label = info["anomaly_class"]

    records.append({
        "image_path": image_path,
        "label": label
    })

df = pd.DataFrame(records)

# Train / Validation / Test split
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42
)

os.makedirs("data/processed", exist_ok=True)

train_df.to_csv("data/processed/train.csv", index=False)
val_df.to_csv("data/processed/val.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)

print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))