import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import EmotionDataset  # Make sure this import path is correct
from model import EmotionRecognition

# Define transforms
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

# Load datasets
train_dataset = EmotionDataset(root_dir="data/train", transform=transform)
test_dataset = EmotionDataset(root_dir="data/test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("Train samples:", len(train_dataset))
print("Test samples:", len(test_dataset))


model = EmotionRecognition()
lossFunction = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(params = model.parameters(), lr = 0.01)

