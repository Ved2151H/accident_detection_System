import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define same CNN architecture
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
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# Load model
model = CNN().to(device)
model.load_state_dict(torch.load("cat_dog_model.pth"))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((150,150)),
    transforms.ToTensor(),
])

# Load test image
image_path = "test.jpg"
image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output, 1)

classes = ["Cat", "Dog"]
print("Prediction:", classes[predicted.item()])