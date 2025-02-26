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
test_path = os.path.join(dir_path, 'data', 'test')
save_path = os.path.join(dir_path, 'model', 'training_pyramid0', 'cnn_-8_12_12_best.pth')
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

conv_net = ConvNetPooling(height=height, width=width, output=output_size, channels=[12, 12]).to(device)
conv_net.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))

label_score = {}
with torch.no_grad():
    for image, label in testloader:
        label = (label - 1).to(device)
        image = image.to(device)
        res = conv_net(image).to(device)
        res = res.argmax(1)
        for label_int, predict_int in zip(list(label), list(res)):
            label_score[int(label_int)] = label_score.get(int(label_int), [0, 0])
            label_score[int(label_int)][0] += int(predict_int == label_int)
            label_score[int(label_int)][1] += 1

sorted_scores = label_score.items()
sorted_scores = sorted(sorted_scores, key= lambda x: x[1][0] / x[1][1])
for item in sorted_scores:
    positive, total = item[1]
    key = item[0]
    print(f"{label_mapping[int(key)]}: {positive / total * 100:.3f}%")
