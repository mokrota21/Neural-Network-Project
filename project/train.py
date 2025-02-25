
import os
cur_path = os.path.abspath(__file__)
dir_path = os.path.join(cur_path, '..')
os.chdir(dir_path)

from architecture import ConvNetPooling, nn
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder, EMNIST
import torchvision.transforms as transforms
import pandas as pd
torch.manual_seed(162179)

device = torch.device("cuda")
batch_size = 32
validation_counter = 20
cur_path = os.path.abspath(__file__)
save_path = os.path.join(dir_path, 'model')
loss_path = os.path.join(dir_path, "loss.json")

to_tensor = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), transforms.ToTensor()
])
dataset = EMNIST(root="data_emnist", split="letters", train=True, transform=to_tensor)
train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validationloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

labels = [label for _, label in dataset]
output_size = len(set(labels))
sample_image, _ = dataset[0]
_, width, height = list(sample_image.shape)

conv_networks_parameters = [torch.randint(1, 10, (2,)).sort(descending=False).values for _ in range(validation_counter)]
conv_networks = []
for parameter in conv_networks_parameters:
    conv_networks.append(ConvNetPooling(height=height, width=width, output=output_size, channels=parameter).cuda())
exp_powers = torch.randint(-20, -5, (validation_counter,))
alphas = exp_powers.exp2()
epochs = 10
mse = nn.CrossEntropyLoss()

best_model = None
best_loss = None
best_model_no = None

losses_list = []

for model_no, (conv_network, alpha) in enumerate(zip(conv_networks, alphas)):
    optimizer = torch.optim.SGD(params=conv_network.parameters(), lr=alpha)
    loss_list = []
    # training
    print(f"TRAINING {model_no}")
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
            loss_list.append(loss.item())
            if i % 100 == 99:
                print(f"Epoch {epoch + 1}, batch {i + 1}, loss {running_loss / 100:.3f}")
                running_loss = 0.0
    # validating
    losses_list.append(loss_list)
    print(f"VALIDATION {model_no}")
    running_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(trainloader, 0):
            image, label = data
            label = (label - 1).to(device)
            image = image.to(device)

            optimizer.zero_grad()
            outputs = conv_network(image).to(device)
            # print(label)
            # print(outputs)
            loss = mse(outputs, label)
            running_loss += loss.item()
            if i % 100 == 99:
                print(f"Epoch {epoch + 1}, batch {i + 1}, total loss {running_loss:.3f}")
        if best_loss is None or best_loss > running_loss:
            best_loss = running_loss
            best_model = conv_network
            best_model_no = model_no

print("Saving losses")
import json
with open(loss_path, 'a') as file:
    file.write(json.dumps(losses_list) + '\n')
print(f"Best model is model with channels {conv_networks_parameters[best_model_no]} and alpha rate of {alphas[best_model_no]} with loss: {best_loss}")
for i, model in enumerate(conv_networks):
    layers = conv_networks_parameters[i]
    alpha = exp_powers[i]
    torch.save(model.state_dict(), os.path.join(save_path, f"cnn_{int(alpha)}_{int(layers[0])}_{int(layers[1])}.pth"))

