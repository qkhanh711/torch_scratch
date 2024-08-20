import torch
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader

def get_dataloader(dataset, batch_size, num_workers, root="data"):
    dataset = dataset.lower()
    if dataset == "mnist":
        transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
        train_dataset = MNIST(root=root, train=True, download=True, transform=transform)
        test_dataset  = MNIST(root=root, train=False, download=True, transform=transform)
    elif dataset == "cifar10":
        transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = CIFAR10(root=root, train=True, download=True, transform=transform)
        test_dataset  = CIFAR10(root=root, train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


classes ={
    "mnist": [str(i) for i in range(10)],
    "cifar10": ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
}

import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

