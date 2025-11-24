import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    """
    MNIST architecture as specified in the PRD (Section 4.1).
    Matches BlockDFL paper architecture.
    """
    def __init__(self):
        super(MNISTNet, self).__init__()
        # 1. Conv2D: 1 -> 32 channels, kernel=3x3, stride=1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        # 3. Conv2D: 32 -> 64 channels, kernel=3x3, stride=1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        # 5. MaxPool2D: kernel=2x2
        self.pool = nn.MaxPool2d(kernel_size=2)
        # 6. Dropout: p=0.25
        self.dropout1 = nn.Dropout(0.25)
        # 8. Linear: 9216 -> 128
        # Calculation: 28x28 -> conv1(3x3) -> 26x26 -> conv2(3x3) -> 24x24 -> pool(2x2) -> 12x12
        # 64 channels * 12 * 12 = 9216
        self.fc1 = nn.Linear(9216, 128)
        # 10. Dropout: p=0.5
        self.dropout2 = nn.Dropout(0.5)
        # 11. Linear: 128 -> 10
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # 1. Conv2D
        x = self.conv1(x)
        # 2. ReLU
        x = F.relu(x)
        # 3. Conv2D
        x = self.conv2(x)
        # 4. ReLU
        x = F.relu(x)
        # 5. MaxPool2D
        x = self.pool(x)
        # 6. Dropout
        x = self.dropout1(x)
        # 7. Flatten
        x = torch.flatten(x, 1)
        # 8. Linear
        x = self.fc1(x)
        # 9. ReLU
        x = F.relu(x)
        # 10. Dropout
        x = self.dropout2(x)
        # 11. Linear
        x = self.fc2(x)
        # 12. LogSoftmax
        output = F.log_softmax(x, dim=1)
        return output
