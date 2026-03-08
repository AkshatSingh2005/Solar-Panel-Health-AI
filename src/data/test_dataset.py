from torchvision import transforms
from pv_dataset import PVDataset

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

dataset = PVDataset(
    csv_file="data/processed/train.csv",
    root_dir="data/raw",
    transform=transform
)

print("Dataset size:", len(dataset))

image, label = dataset[0]

print("Image shape:", image.shape)
print("Label:", label)