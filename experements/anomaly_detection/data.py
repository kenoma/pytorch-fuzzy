import torch
from torchvision import datasets, transforms


def get_target_and_mask(target_label):
    t = target_label
    return t

def norm_and_transform(x):
    nimg = 2.0*(x.view(-1, 28, 28) - 0.5)
    nimg = torch.clamp(nimg, -1, 1)
    return nimg

def get_mnist_loaders(bs, class_anomaly=None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(size=26),
        transforms.Resize(size=(28, 28)),
        norm_and_transform
    ])

    train_data = datasets.MNIST(
        '~/.pytorch/MNIST_data/',
        download=True,
        train=True,
        transform=transform,
        target_transform=transforms.Lambda(lambda x: get_target_and_mask(x))
    )

    if class_anomaly is not None:
        idx = (train_data.targets != class_anomaly)
        train_data.targets = train_data.targets[idx]
        train_data.data = train_data.data[idx]

    test_data = datasets.MNIST(
        '~/.pytorch/MNIST_data/',
        download=True,
        train=False,
        transform=transform,
        target_transform=transforms.Lambda(lambda x: get_target_and_mask(x))
    )

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=bs,
        shuffle=True,

    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=bs,
        shuffle=False,
    )

    return train_loader, test_loader