import glob
from dataclasses import dataclass
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from tqdm.auto import tqdm
from torchfuzzy.fuzzy_layer import FuzzyLayer
import pandas as pd

DATA = [
    ("pl", "mlp", "PL", "MLP", 'orange'),
    ("cut", "mlp", "Cut", "MLP", 'green'),
    ("pl", "fz", "PL", "Fuzzy", 'blue'),
("cut", "fz", "Cut", "Fuzzy", 'm'),
]

if __name__ == '__main__':
    fig, ax = plt.subplots(1, figsize=(5, 5))

    for task_type, model_type, task_type_label, model_type_label, color in DATA:
        data_paths = glob.glob(f"./papers/iiti24/{task_type}*{model_type}.csv")

        x = []
        for data_file in data_paths:
            df = pd.read_csv(data_file, sep=',')
            unique_ratios = np.unique(df['ratio'].values)
            max_acc = [np.nan for _ in range(11)]
            for i, r in enumerate(unique_ratios):
                ratio_data = df.loc[df['ratio'] == r]
                max_acc[i] = np.max([v for v in ratio_data['test_accuracy'].values if not np.isnan(v)])
            x.append(max_acc)
        x = np.stack(x)
        mu = x.mean(axis=0)
        sigma = x.std(axis=0)
        if len(unique_ratios) < 10:
            unique_ratios = np.linspace(0, 1, 11)
        ax.plot(unique_ratios, mu, lw=2, label=f'{task_type_label} {model_type_label} C-VAE', color=color)
        ax.fill_between(unique_ratios, mu + sigma, mu - sigma, facecolor=color, alpha=0.3)

    ax.set_title(r'Learning on partial labeled dataset')
    ax.legend(loc='lower left')
    ax.set_xlabel('Rate of unlabeled data')
    ax.set_ylabel('Accuracy')

    plt.savefig(f'./papers/iiti24/fig6-accuracy-vs-unlabeled-rate.png')
    plt.show()

