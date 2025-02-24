
import os
cur_path = os.path.abspath(__file__)
dir_path = os.path.join(cur_path, '..')
os.chdir(dir_path)

from architecture import ConvNetPooling, nn
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder, EMNIST
import torchvision.transforms as transforms

load = True
device = torch.device("cuda")
batch_size = 32
cur_path = os.path.abspath(__file__)
train_path = os.path.join(dir_path, 'data', 'train')
save_path = os.path.join(dir_path, 'model', 'cnn.pth')

to_tensor = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), transforms.ToTensor()
])
dataset = EMNIST(root="data_emnist", split="letters", train=True, transform=to_tensor)
labels = [label for _, label in dataset]
output_size = len(set(labels))
sample_image, _ = dataset[0]
_, width, height = list(sample_image.shape)
trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

conv_network = ConvNetPooling(height=height, width=width, output=output_size).to(device)
if load:
    conv_network.load_state_dict(torch.load(save_path, weights_only=True))
epochs = 10
alpha = 0.001
optimizer = torch.optim.SGD(params=conv_network.parameters(), lr=alpha)
mse = nn.CrossEntropyLoss()

for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        image, label = data
        label = (label - 1).to(device)
        image = image.to(device)

        optimizer.zero_grad()
        outputs = conv_network(image).to(device)
        # print(label)
        # print(outputs)
        loss = mse(outputs, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:
            print(f"Epoch {epoch + 1}, batch {i + 1}, loss {running_loss / 100:.3f}")
            running_loss = 0.0

torch.save(conv_network.state_dict(), save_path)

