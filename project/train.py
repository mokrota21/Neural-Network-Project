
import os
cur_path = os.path.abspath(__file__)
dir_path = os.path.dirname(cur_path)
os.chdir(dir_path)

from architecture import ConvNetPooling, nn
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder, EMNIST
import torchvision.transforms as transforms
import pandas as pd
import json

def get_unique_path(path, name, suffix=""):
    counter = 0
    new_path = os.path.join(path, name + str(counter) + suffix)
    while os.path.exists(new_path):
        counter += 1
        new_path = os.path.join(path, name + str(counter) + suffix)
    return new_path

descending = False
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
save_path = get_unique_path(save_path, folder_name)
os.mkdir(save_path)
loss_path = os.path.join(save_path, "loss.json")

to_tensor = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), transforms.ToTensor()
])
dataset = EMNIST(root="data_emnist", split="letters", train=True, transform=to_tensor)
train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

def set_train_val(batch_size, seed=None):
    if seed:
        torch.manual_seed(seed)
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validationloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    return trainloader, validationloader

labels = [label for _, label in dataset]
output_size = len(set(labels))
sample_image, _ = dataset[0]
_, width, height = list(sample_image.shape)

def models_rand(n, channel_low, channel_high, exp_low, exp_high, num_l=2, width=width, height=height):
    conv_networks_parameters = [torch.randint(channel_low, channel_high, (num_l,)).sort(descending=descending).values for _ in range(n)]
    conv_networks = []
    for parameter in conv_networks_parameters:
        conv_networks.append(ConvNetPooling(height=height, width=width, output=output_size, channels=parameter.tolist()).cuda())

    exp_powers = torch.randint(exp_low, exp_high, (n,))
    alphas = exp_powers.exp2().tolist()

    return conv_networks, alphas



def validation(model: ConvNetPooling, validationloader, save_folder=save_path, seed=None):
    if seed:
        torch.manual_seed(seed)
    model.eval()
    predicted = torch.tensor(data=[]).to(device)
    gt = torch.tensor(data=[]).to(device)

    with torch.no_grad():
        for i, data in enumerate(validationloader, 0):
            image, label = data
            label = (label - 1).to(device)
            image = image.to(device)

            outputs = model(image).to(device)
            batch_predictions = outputs.argmax(1).flatten()
            batch_gt = label.flatten()

            predicted = torch.cat((predicted, batch_predictions))
            gt = torch.cat((gt, batch_gt))
    
    result = {
        "predicted": predicted.tolist(),
        "expected": gt.tolist()
    }
    if save_folder:
        path = "validation"
        path = get_unique_path(save_folder, path, suffix=".json")
        with open(path, 'w') as f:
            f.write(json.dumps(result))
    model.train()
    return result

def train(model: ConvNetPooling, hyperparams, batch_size, save_folder=save_path, epochs=10, seed=None):
    if seed:
        torch.manual_seed(seed)
    mse = nn.CrossEntropyLoss()

    trainloader, validationloader = set_train_val(batch_size, seed)
    optimizer = torch.optim.Adam(params=model.parameters(), **hyperparams)
    loss_list = []
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            image, label = data
            label = (label - 1).to(device)
            image = image.to(device)

            optimizer.zero_grad()
            outputs = model(image).to(device)
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

    if save_folder:
        model_path = "cnn"
        model_path = get_unique_path(save_folder, model_path, suffix=".pth")
        torch.save(model.state_dict(), model_path)

        loss_path = "loss"
        loss_path = get_unique_path(save_folder, loss_path, suffix=".json")
        with open(loss_path, 'w') as f:
            f.write(json.dumps(loss_list))

        validation_path = "validation"
        validation_path = get_unique_path(save_folder, validation_path, suffix=".json")
        validation(model, validationloader, save_folder, seed=seed)

        metadata = {
            "model": os.path.basename(model_path),
            "loss": os.path.basename(loss_path),
            "validation": os.path.basename(validation_path),
            "hyperparameters": hyperparams,
            "channels": model.summary,
            "epochs": epochs
        }
        metadata_path = "train_metadata"
        metadata_path = get_unique_path(save_folder, metadata_path, suffix=".json")
        with open(metadata_path, 'w') as f:
            f.write(json.dumps(metadata))


