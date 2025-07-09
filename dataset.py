import os
from PIL import Image
from torch.utils.data import Dataset

class EmotionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))  # Ensure consistent order
        self.image_paths = []
        self.labels = []

        for label_index, class_name in enumerate(self.classes):
            class_folder = os.path.join(root_dir, class_name)
            for file_name in os.listdir(class_folder):
                if file_name.lower().endswith((".jpg", ".png", ".jpeg")):
                    self.image_paths.append(os.path.join(class_folder, file_name))
                    self.labels.append(label_index)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
