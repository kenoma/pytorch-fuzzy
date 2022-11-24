#%%
import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from fuzzy_layer import FuzzyLayer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#%%
def plot_clusters(z, labels):
    plt.scatter(z[:, 0], z[:, 1], c=labels, s=3)
    plt.colorbar()
#%%
class SimpleClustering(nn.Module):
    def __init__(self):
        super(SimpleClustering, self).__init__()
        self.fuzzy = FuzzyLayer.fromdimentions(2, 5, trainable=True)

    def forward(self, x):
        return self.fuzzy(x)

class NoisyClustersDataset(Dataset):
    def __init__(self, clusters, size=1000):
        self.clusters = clusters
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        cluster = np.random.randint(len(self.clusters))
        rcl = self.clusters[cluster]
        return rcl[0] * np.random.randn(2) + rcl[1:], cluster

#%%
ds = NoisyClustersDataset([[1,-1,-1], 
                           [0.5,1,1], 
                           [0.1,-1,1], 
                           [0.3,1,-1],
                           [0.1, 0, 0]], 10000)
dl = DataLoader(ds, batch_size=256)
#%%
model = SimpleClustering()
features, labels = next(iter(dl))
processed_dataset = model(features.float()).detach().numpy()
plot_clusters(features, labels)
#%%
res = model(features.float()).detach().numpy()
assigned_classes =[np.argmax(a) for a in res]
plot_clusters(features, assigned_classes)
#%%
def train(model, dataloader, epochs=20):
    opt = torch.optim.RMSprop(model.parameters())
    loss = nn.CrossEntropyLoss(reduction="sum")

    for epoch in range(epochs):
        sum_loss = 0
        for x, y in dataloader:
            opt.zero_grad()
            f_c = model(x.float())
            loss_value = loss(f_c, y)
            loss_value.backward()
            opt.step()
            sum_loss = loss_value
                    
        print(f"Epoch {epoch}: ploss {sum_loss}")
        count = 0
        sum_ploss = 0

    return model

train(model, dl, 2)
#%%
res = model(features.float()).detach().numpy()
assigned_classes =[np.argmax(a) for a in res]
plot_clusters(features, assigned_classes)
# %%
model.fuzzy.A
