import torch
from torch import nn
import torch.nn.functional as F

class ConvNetPooling(nn.Module):
    def __init__(self, height, width, output):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 4) # 28 - 3 = 25
        height = height - 4 + 1
        width = width - 4 + 1
        self.pol = nn.MaxPool2d(2) # 25 / 2 = 12
        height = height // 2
        width = width // 2
        self.conv2 = nn.Conv2d(4, 10, 4) # 12 - 4 + 1 = 9
        height = height - 4 + 1
        width = width - 4 + 1
        # extra poling: 19 / 2 = 9
        height = height // 2
        width = width // 2
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(10 * height * width, 100)
        self.fc2 = nn.Linear(100, 10)
        self.fc3 = nn.Linear(10, output)
    
    def forward(self, x):
        x = self.pol(F.relu(self.conv1(x)))
        x = self.pol(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x
    
    def show_conv(self, x):
        tensors_list = []
        x = self.conv1(x)
        tensors_list.append(x.clone().detach())
        x = F.relu(x)
        tensors_list.append(x.clone().detach())
        x = self.pol(x)
        tensors_list.append(x.clone().detach())
        x = self.conv2(x)
        tensors_list.append(x.clone().detach())
        x = F.relu(x)
        tensors_list.append(x.clone().detach())
        x = self.pol(x)
        tensors_list.append(x.clone().detach())
        return tensors_list