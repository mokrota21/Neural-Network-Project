import os
cur_path = os.path.abspath(__file__)
dir_path = os.path.join(cur_path, '..')
os.chdir(dir_path)

from architecture import ConvNetPooling, nn
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder, EMNIST
import torchvision.transforms as transforms
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import json

device = torch.device("cuda")
batch_size = 50
cur_path = os.path.abspath(__file__)
model_path = os.path.join(dir_path, 'best_models', "best_reverse_pyramid", "cnn10.pth")
metadata_path = os.path.join(os.path.dirname(model_path), 'train_metadata10.json')

output_prefix = "emnist_reverse_pyramid"
save_path = os.path.join(dir_path, 'predictions')
counter = 0
folder_name = "{output_prefix}_visuals_{counter}"
folder_path = os.path.join(save_path, folder_name.format(output_prefix=output_prefix, counter=counter))
while os.path.exists(folder_path):
    counter += 1
    folder_path = os.path.join(save_path, folder_name.format(output_prefix=output_prefix, counter=counter))
os.mkdir(folder_path)

label_mapping = [chr(ord('a') + i) for i in range(0, 26)]

to_tensor = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), transforms.ToTensor()
])
dataset = EMNIST(root="data_emnist", split="letters", train=False, transform=to_tensor)
labels = [label for _, label in dataset]
output_size = len(set(labels))
sample_image, sample_label = dataset[3]
# F.to_pil_image(F.hflip(F.rotate(sample_image, -90))).show()
# F.to_pil_image(F.rotate(sample_image, -90)).show()
_, width, height = list(sample_image.shape)
trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

def load_model(model_path, metadata_path):
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    model = ConvNetPooling(height, width, output_size, channels=metadata['channels'])
    model.load_state_dict(torch.load(model_path, weights_only=True))
    return model

conv_net = load_model(model_path=model_path, metadata_path=metadata_path)
dataiter = iter(trainloader)
images, labels = next(dataiter)
images = images.to(device)
conv_out = conv_net.show_conv(images)  # shape: [batch_size, 4, H, W]

for image in range(conv_out[0].shape[0]):
    counter = 0
    img_folder_name = f'image_{image}'
    img_folder_path = os.path.join(folder_path, img_folder_name)
    os.mkdir(img_folder_path)
    label = label_mapping[int(labels[image] - 1)]
    original_path = os.path.join(img_folder_path, f"original_{label}.jpg")
    F.to_pil_image(F.hflip(F.rotate(images[image], -90))).save(original_path)
    for layer in range(len(conv_out)):
        layer_out = conv_out[layer]
        layer_path = os.path.join(img_folder_path, f"layer_{layer}")
        os.mkdir(layer_path)
        for channel in range(layer_out.shape[1]):
            image_tensor = layer_out[image][channel].unsqueeze(0)
            file_path = os.path.join(layer_path, f"visual_channel_{channel}.jpg")
            F.to_pil_image(F.hflip(F.rotate(image_tensor, -90))).save(file_path)
