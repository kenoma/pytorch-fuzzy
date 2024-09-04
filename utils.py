import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torchvision import datasets, transforms


def get_target_and_mask(target_label, unknown_ratio, labels: int, device):
    is_multi_label = labels > 10
    t = F.one_hot(torch.LongTensor([target_label]), labels)
    if is_multi_label:
        if (target_label == 0) or (target_label == 6) or (target_label == 8) or (target_label == 9):
            t[0][labels - 2] = 1
        else:
            t[0][labels - 1] = 1
    m = torch.ones((1, labels)) if torch.rand(1) > unknown_ratio else torch.zeros((1, labels))

    return torch.cat((t, m), 0).to(device)


def get_data(is_multi_class: bool, batch_size: int, device, unknown_classes_ratio: float = 0):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1, 28, 28) - 0.5),
    ])

    train_data = datasets.MNIST(
        '~/.pytorch/MNIST_data/',
        download=True,
        train=True,
        transform=transform,
        target_transform=transforms.Lambda(lambda x: get_target_and_mask(x, unknown_classes_ratio, 12 if is_multi_class else 10, device))
    )
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
    )

    test_data = datasets.MNIST(
        '~/.pytorch/MNIST_data/',
        download=True,
        train=False,
        transform=transform,
        target_transform=transforms.Lambda(lambda x: get_target_and_mask(x, 0, 12 if is_multi_class else 10, device))
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader


def get_emnist():
    emnist_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1, 28, 28) - 0.5),
    ])

    emnist_test = pd.read_csv("./data/EMNIST/emnist-letters.csv")
    emnist_y = emnist_test["label"]#[:-int(len(emnist_test) * drop)]
    emnist_x = emnist_test.drop(labels=["label"], axis=1)#[:-int(len(emnist_test) * drop)]
    del emnist_test

    emnist_x = emnist_x / 255.0
    emnist_x = emnist_x.values.reshape(-1, 28, 28)
    emnist_x = [torch.tensor(emnist_transforms(a), dtype=torch.float32) for a in emnist_x]

    emnist_mapping = pd.read_csv("./data/EMNIST/emnist-letters-mapping.txt", sep=' ', header=None)
    emnist_mapping.columns = ("EMNIST", "UP", "LO")
    emnist_mapping["Letter"] = emnist_mapping.apply(lambda row: chr(row["UP"]) + chr(row["LO"]), axis=1)
    emnist_mapping = dict(zip(emnist_mapping["EMNIST"] - 1, emnist_mapping["Letter"]))

    return emnist_x, emnist_y, emnist_mapping


def predict(model, data_loader, is_multi_class, is_emnist: bool, device):
    if is_emnist and is_multi_class:
        raise Exception()

    gt_digits = []
    gt_shape = []
    pred_mu = []
    pred_digits = []
    pred_shape = []

    with torch.no_grad():
        for data, target in data_loader:
            data = data.view((-1, 1, 28, 28)).to(device)
            mu, logvar, z, labels = model.half_pass(data)

            pred_mu.append(mu.cpu().numpy())
            if is_emnist:
                gt_digits.append(target - 1)
            else:
                gt_digits.append(np.argmax(target[:, 0, :10].cpu().numpy(), axis=1))
            pred_digits.append(np.argmax(labels[:, :10].cpu().numpy(), axis=1))
            if is_multi_class:
                gt_shape.append(np.argmax(target[:, 0, 10:].cpu().numpy(), axis=1))
                pred_shape.append(np.argmax(labels[:, 10:].cpu().numpy(), axis=1))

    pred_mu = np.concatenate(pred_mu, axis=0)
    if is_emnist:
        gt_digits = np.array(gt_digits)
    else:
        gt_digits = np.concatenate(gt_digits, axis=0)
    pred_digits = np.concatenate(pred_digits, axis=0)
    if is_multi_class:
        gt_shape = np.concatenate(gt_shape, axis=0)
        pred_shape = np.concatenate(pred_shape, axis=0)

    return pred_mu, (pred_digits, pred_shape), (gt_digits, gt_shape)


def pred_per_class(mu, labeles):
    class_data = [[] for _ in range(len(np.unique(labeles)))]

    for mu, y in zip(mu, labeles):
        class_data[y].append(mu)

    return [np.array(cd) for cd in class_data]


def class_scatter(ax, x, y, labels, x_axis_label, y_axis_label, title, **kwargs):
    unique = list(set(labels))
    colors = [plt.cm.tab10(float(i)/len(unique)) for i, _ in enumerate(unique)]
    for i, u in enumerate(unique):
        x1i = [x[j] for j in range(len(x)) if labels[j] == u]
        x2i = [y[j] for j in range(len(y)) if labels[j] == u]
        ax.scatter(x1i, x2i, c=colors[i], label=str(u), **kwargs)
    if ax is plt:
        ax.xlabel(x_axis_label)
        ax.ylabel(y_axis_label)
        ax.title(title)
    else:
        ax.set_xlabel(x_axis_label)
        ax.set_ylabel(y_axis_label)
        ax.set_title(title)
    lgnd = ax.legend(loc='upper right', numpoints=1, fontsize=10)

    # change the marker size manually for both lines
    for i in range(len(unique)):
        lgnd.legendHandles[i]._sizes = [30]
