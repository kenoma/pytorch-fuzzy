import glob
import os.path
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
    ("pl", "fz", "PL", "Fuzzy", 'blue'),
    ("pl", "mlp", "PL", "MLP", 'orange'),
]


if __name__ == '__main__':
    fig, ax = plt.subplots(1, figsize=(10, 5))

    for task_type, model_type, task_type_label, model_type_label, color in DATA:
        data_paths = glob.glob(f"./papers/iiti24/pl/3{task_type}*{model_type}.csv")

        x = []
        for data_file in data_paths:
            df = pd.read_csv(data_file, sep=',')
            unique_ratios = np.unique(df['ratio'].values)
            data = []
            for i, r in enumerate(unique_ratios):
                ratio_data = df.loc[df['ratio'] == r]
                data.append(np.max([v for v in ratio_data['test_accuracy'].values if not np.isnan(v)]))
            x.append(data)
        x = np.stack(x)
        mu = x.mean(axis=0)
        sigma = x.std(axis=0)
        ax.plot(unique_ratios, mu, lw=2, label=f'{model_type_label} C-VAE', color=color)
        ax.plot(unique_ratios, mu + sigma, lw=1, color=color, ls=':')
        ax.plot(unique_ratios, mu - sigma, lw=1, color=color, ls=':')
        #ax.fill_between(unique_ratios, mu + sigma, mu - sigma, facecolor=color, alpha=0.3)

    # ax.set_title(r'Learning on cut dataset')
    ax.legend(loc='lower left')
    ax.set_xlabel('Dataset unlabeled rate')
    ax.set_ylabel('Accuracy')

    plt.savefig(f'./papers/iiti24/fig6-accuracy-vs-unlabeled-rate.eps', format='eps')
    plt.show()
