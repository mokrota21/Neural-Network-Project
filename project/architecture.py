import torch
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda")
class ConvNetPooling(nn.Module):
    def __init__(self, height, width, output, channels):
        super().__init__()
        self.conv_seqs = nn.ModuleList()
        input_size = 1 # we assume images are GS
        kernel_size = 4 # probably doesn't matter
        pooling_kernel = 2 # probbaly doesn't matter
        self.summary = channels[::]
        i = 1
        for channel_count in channels:
            channel_count = int(channel_count)
            conv_seq = nn.Sequential(
                nn.Conv2d(input_size, channel_count, kernel_size, padding='same').to(device),
                nn.ReLU(),
                nn.Dropout2d(p=0.2),
                nn.MaxPool2d(pooling_kernel).to(device)
            )
            self.conv_seqs.append(conv_seq)
            # self.conv_layers.append((conv_layer, relu, pooling_layer))
            # height = (height - kernel_size + 1) // 2 # if no padding and stride=1
            # width = (width - kernel_size + 1) // 2 # if no padding and stride=1
            height = height // 2 # if padding
            width = width // 2 # if padding
            input_size = channel_count
        self.flatten = nn.Flatten().to(device)
        self.fc1 = nn.Linear(channels[-1] * height * width, 512).to(device)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 84).to(device)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(84, output).to(device)
        self.softmax = nn.Softmax()
    
    def forward(self, x):
        for conv_seq in self.conv_seqs:
            x = conv_seq(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x
    
    def show_conv(self, x):
        tensors_list = []
        for layers in self.conv_seqs:
            conv_layer = layers[0]
            relu_layer = layers[1]
            pooling_layer = layers[2]
            x = conv_layer(x)
            tensors_list.append(x.clone().detach())
            x = relu_layer(x)
            tensors_list.append(x.clone().detach())
            x = pooling_layer(x)
            tensors_list.append(x.clone().detach())
        return tensors_list