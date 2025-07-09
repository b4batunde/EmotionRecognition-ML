import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import EmotionDataset  # Custom Dataset class
from model import EmotionRecognition  # Your model definition

# Define transforms (should match your model’s expected input)
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

# Load training and test datasets
train_dataset = EmotionDataset(root_dir="data/train", transform=transform)
test_dataset = EmotionDataset(root_dir="data/test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize model, loss function, and optimizer
model = EmotionRecognition()
lossFunction = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.01)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        # Forward pass
        outputs = model(images)
        loss = lossFunction(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(avg_loss)


torch.save(model.state_dict(), "emotion_model.pth")
print("Model saved to emotion_model.pth ✅")
