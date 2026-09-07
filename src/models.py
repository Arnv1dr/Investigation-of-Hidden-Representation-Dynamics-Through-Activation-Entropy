import torch
from torch import nn


class MLP(nn.Module):
    """
    Simple ReLU-based Multi-Layer Perceptron for 28x28 grayscale images.
    Input shape: [batch_size, 1, 28, 28]
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


class SimpleCNN(nn.Module):
    """
    Simple ReLU-based Convolutional Neural Network for 28x28 grayscale images.
    Input shape: [batch_size, 1, 28, 28]
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x


def create_model(model_name: str) -> nn.Module:
    """Factory function for creating models by name."""
    model_name = model_name.lower()

    if model_name == "mlp":
        return MLP()
    if model_name == "cnn":
        return SimpleCNN()

    raise ValueError(f"Unsupported model: {model_name}. Use 'mlp' or 'cnn'.")


def get_monitored_layer_names(model_name: str) -> list[str]:
    """Return hidden post-activation layers monitored for a model."""
    model_name = model_name.lower()

    if model_name == "mlp":
        return ["relu1", "relu2"]

    if model_name == "cnn":
        return ["relu1", "relu2", "relu3"]

    raise ValueError(f"Unsupported model: {model_name}. Use 'mlp' or 'cnn'.")
