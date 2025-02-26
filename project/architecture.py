import torch
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda")
class ConvNetPooling(nn.Module):
    def __init__(self, height, width, output, channels):
        super().__init__()
        self.conv_layers = []
        input_size = 1 # we assume images are GS
        kernel_size = 4 # probably doesn't matter
        pooling_kernel = 2 # probbaly doesn't matter
        i = 1
        for channel_count in channels:
            channel_count = int(channel_count)
            conv_layer = nn.Conv2d(input_size, channel_count, kernel_size).to(device)
            relu = nn.ReLU()
            dropout = nn.Dropout2d(p=0.2)
            pooling_layer = nn.MaxPool2d(pooling_kernel).to(device)
            self.conv_layers.append((conv_layer, relu, dropout, pooling_layer))
            height = (height - kernel_size + 1) // 2 # we assume no padding and stride=1
            width = (width - kernel_size + 1) // 2 # we assume no padding and stride=1
            input_size = channel_count
        self.flatten = nn.Flatten().to(device)
        self.fc1 = nn.Linear(channels[-1] * height * width, 100).to(device)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(100, 10).to(device)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(10, output).to(device)
    
    def forward(self, x):
        for layers in self.conv_layers:
            conv_layer, relu, dropout, pooling_layer = layers
            x = conv_layer(x)
            x = relu(x)
            x = dropout(x)
            x = pooling_layer(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        return x
    
    def show_conv(self, x):
        tensors_list = []
        for layers in self.conv_layers:
            conv_layer = layers[0]
            pooling_layer = layers[1]
            x = conv_layer(x)
            tensors_list.append(x.clone().detach())
            x = F.relu(x)
            tensors_list.append(x.clone().detach())
            x = pooling_layer(x)
            tensors_list.append(x.clone().detach())
        return tensors_list