import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
transform = transforms.Compose(
    [
     transforms.Resize([416,416]),
     transforms.Grayscale(num_output_channels=1),
     transforms.ToTensor(),
     # transforms.Normalize([0.5],[0.5]),
    ])
#IR_VIS
ir = torchvision.datasets.ImageFolder('D:/dataset/95000/0',transform=transform)
vis = torchvision.datasets.ImageFolder('D:/dataset/95000/1',transform=transform)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
