import torch
import torch.nn as nn
import torch.nn.functional as F

class CIFAR10Net(nn.Module):
    """
    CIFAR-10 architecture (CIFARNET) as specified in the PRD (Section 4.2).
    Matches BlockDFL paper architecture.
    """
    def __init__(self):
        super(CIFAR10Net, self).__init__()
        # 64C3x3
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        # 64C3x3
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # MaxPool2
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        
        # 128C3x3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # 128C3x3
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        # AvgPool2
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # 256C3x3
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        # 256C3x3
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # AvgPool8
        # Note: Input is 32x32. After pool1 (2x2) -> 16x16. After pool2 (2x2) -> 8x8.
        # AvgPool8 on 8x8 will result in 1x1.
        self.pool3 = nn.AvgPool2d(kernel_size=8)
        
        # Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        
        # FC256
        # Input to FC is 256 channels * 1 * 1 = 256
        self.fc1 = nn.Linear(256, 256)
        
        # FC10
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # Block 1
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Block 2
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        
        # Block 3
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # FC Layers
        x = self.dropout2(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)
