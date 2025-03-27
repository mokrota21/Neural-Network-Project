import os
cur_path = os.path.abspath(__file__)
dir_path = os.path.join(cur_path, '..')
os.chdir(dir_path)

from architecture import ConvNetPooling, nn
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder, EMNIST
import torchvision.transforms as transforms
import json
import numpy as np
from sklearn.metrics import confusion_matrix

models_folder = os.path.join(dir_path, 'model')

    

device = torch.device("cuda")
batch_size = 32
cur_path = os.path.abspath(__file__)
to_tensor = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), transforms.ToTensor()
])
dataset = EMNIST(root="data_emnist", split="letters", train=False, transform=to_tensor)
labels = [label for _, label in dataset]
output_size = len(set(labels))
sample_image, _ = dataset[0]
_, width, height = list(sample_image.shape)
testloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

label_mapping = [chr(ord('a') + i) for i in range(0, 26)]

def load_model(model_path, metadata_path):
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    model = ConvNetPooling(height=height, width=width, output=output_size, channels=metadata['channels'])
    model.load_state_dict(torch.load(model_path, weights_only=True))
    return model

def evaluate(model: ConvNetPooling, test_loader=testloader, seed=None): # note testloader is uniformal if not specified
    if seed:
        torch.manual_seed(seed)
    model.eval()
    accuracies = []

    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            image, label = data
            label = (label - 1).to(device)
            image = image.to(device)

            outputs = model(image).to(device)
            batch_predictions = outputs.argmax(1).flatten()
            batch_gt = label.flatten()
            
            accuracies.append((batch_predictions == batch_gt).sum().item() / label.size(0))
    
    return accuracies

def validation(model: ConvNetPooling, validationloader=testloader, seed=None):
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
    return result