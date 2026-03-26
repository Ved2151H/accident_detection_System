import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os

# ==========================
# 1. Device Configuration
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

# ==========================
# 2. Dataset Path
# ==========================
data_dir = r"D:\Subjects_Languages\Languages\AICT_SureTrust_Internship\ALL_Assignments\MLP\Cat_VS_Dog\dataset"

# ==========================
# 3. Transformations
# ==========================
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
])

# ==========================
# 4. Load Dataset
# ==========================
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

print("Classes:", dataset.classes)
print("Class Index:", dataset.class_to_idx)

# ==========================
# 5. Train / Validation Split
# ==========================
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ==========================
# 6. CNN Model Definition
# ==========================
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 18 * 18, 128),
            nn.ReLU(),
            nn.Linear(128, 2)   # 2 classes (cat, dog)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

model = CNN().to(device)

# ==========================
# 7. Loss & Optimizer
# ==========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==========================
# 8. Training Loop
# ==========================
epochs = 5

for epoch in range(epochs):

    # ---- Training ----
    model.train()
    train_correct = 0
    train_total = 0
    train_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        train_correct += (predicted == labels).sum().item()
        train_total += labels.size(0)

    train_accuracy = 100 * train_correct / train_total

    # ---- Validation ----
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_accuracy = 100 * val_correct / val_total

    print(f"Epoch [{epoch+1}/{epochs}] "
          f"Train Loss: {train_loss:.4f} "
          f"Train Acc: {train_accuracy:.2f}% "
          f"Val Acc: {val_accuracy:.2f}%")

# ==========================
# 9. Save Model
# ==========================
torch.save(model.state_dict(), "cat_dog_model.pth")

print("Training Complete 🔥 Model Saved.")