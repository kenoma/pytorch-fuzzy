#%%
import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchfuzzy import FuzzyLayer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#%%
def plot_clusters(z, labels, title = ''):
    fig = plt.figure(figsize=[5, 5])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.scatter(z[:, 0], z[:, 1], c=labels, cmap='Accent', s=2)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_title(title)
#%%
class SimpleClustering(nn.Module):
    def __init__(self):
        super(SimpleClustering, self).__init__()
        self.fuzzy = FuzzyLayer.fromdimentions(2, 4, trainable=True)

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
ds = NoisyClustersDataset([[0.1,-0.5,-0.5], 
                           [0.1, 0.5, 0.5],
                           [0.1, 0.5,-0.5],
                           [0.1,-0.5, 0.5]], 10000)
dl = DataLoader(ds, batch_size=256)
features, labels = next(iter(dl))
plot_clusters(features, labels, title='Initial dataset')
#%%
model = SimpleClustering()
res = model(features.float()).detach().numpy()
assigned_classes =[np.argmax(a) for a in res]
plot_clusters(features, assigned_classes, title='Initial cluster assignment')
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
            sum_loss += loss_value
                    
        print(f"Epoch {epoch}: loss {sum_loss}")

    return model

train(model, dl, 35)
#%%
res = model(features.float()).detach().numpy()
assigned_classes =[np.argmax(a) for a in res]
plot_clusters(features, assigned_classes, title='Cluster assignment after few epoches')
# %%
uniform_distr = np.random.uniform(-1, 1, (10000, 2))
res = model(torch.FloatTensor(uniform_distr)).detach().numpy()
assigned_classes =[np.argmax(a) if a[np.argmax(a)]>6e-1 else -1 for a in res]
plot_clusters(uniform_distr, assigned_classes,title='Uniform distribution classification (threshold 6e-1, after 300 iterations)')

# %%
