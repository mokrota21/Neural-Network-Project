
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

def get_unique_folder(path, folder_name):
    counter = 0
    folder_path = os.path.join(path, folder_name + str(counter))
    while os.path.exists(folder_path):
        counter += 1
        folder_path = os.path.join(path, folder_name + str(counter))
    os.mkdir(folder_path)
    return folder_path

descending = True
device = torch.device("cuda")
batch_size = 256
validation_counter = 5
cur_path = os.path.abspath(__file__)
save_path = os.path.join(dir_path, 'model')
folder_name = "training"
if descending:
    folder_name += "_pyramid"
else:
    folder_name += "_reverse_pyramid"
save_path = get_unique_folder(save_path, folder_name)
loss_path = os.path.join(save_path, "loss.json")

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

conv_networks_parameters = [torch.randint(4, 16, (2,)).sort(descending=descending) .values for _ in range(validation_counter)]

conv_networks = []
for parameter in conv_networks_parameters:
    conv_networks.append(ConvNetPooling(height=height, width=width, output=output_size, channels=parameter).cuda())
exp_powers = torch.randint(-13, -4, (validation_counter,))
alphas = exp_powers.exp2()

epochs = 10
mse = nn.CrossEntropyLoss()
best_model = None
best_accuracy = None
best_model_no = None

losses_list = {}

for model_no, (conv_network, alpha) in enumerate(zip(conv_networks, alphas)):
    optimizer = torch.optim.Adam(params=conv_network.parameters(), lr=alpha)
    loss_list = []
    # training
    print(f"TRAINING {model_no}")
    for epoch in range(epochs):
        running_loss = 0.0
        trainloader
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
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for i, data in enumerate(trainloader, 0):
            image, label = data
            label = (label - 1).to(device)
            image = image.to(device)

            optimizer.zero_grad()
            outputs = conv_network(image).to(device)
            # print(label)
            # print(outputs)
            outputs = outputs.argmax(1)
            correct += (label == outputs).sum().item()
            total += label.shape[0]
            if i % 100 == 99:
                print(f"Epoch {epoch + 1}, batch {i + 1}, accuracy {correct / total * 100:.3f}")
        if best_accuracy is None or best_accuracy < correct / total:
            best_accuracy = correct / total
            best_model = conv_network
            best_model_no = model_no

print("Saving losses")
import json
with open(loss_path, 'a') as file:
    file.write(json.dumps(losses_list) + '\n')
print(f"Best model is model with channels {conv_networks_parameters[best_model_no]} and alpha rate of {alphas[best_model_no]} with loss: {best_accuracy}")
for i, model in enumerate(conv_networks):
    layers = conv_networks_parameters[i]
    alpha = exp_powers[i]
    if i != best_model_no:
        file_name = f"cnn_{int(alpha)}_{int(layers[0])}_{int(layers[1])}.pth"
    else:
        file_name = f"cnn_{int(alpha)}_{int(layers[0])}_{int(layers[1])}_best.pth"
    torch.save(model.state_dict(), os.path.join(save_path, file_name))

