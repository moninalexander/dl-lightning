# This file downloads the data and returns dataloaders for training and testing

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Normalize


BATCH_SIZE = 64

img_transform = Compose([ToTensor(),Normalize((0.1307,),(0.3081,))])

def loaders(batch_size=BATCH_SIZE):
    train_data = datasets.MNIST(download=True, train=True,root='../../../data', 
                            transform=img_transform)
    test_data = datasets.MNIST(root='../../../data',train=False,download=True,
                           transform=img_transform)
    train_loader = DataLoader(train_data, batch_size=batch_size,shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size,shuffle=False)
    X, _ = train_data[0]
    return train_loader, len(train_data), test_loader, len(test_data), X.shape[1], X.shape[0], len(train_data.classes)