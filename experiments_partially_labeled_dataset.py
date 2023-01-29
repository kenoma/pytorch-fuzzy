#%%
import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from fuzzy_layer import FuzzyLayer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#%%
def plot_clusters(z, labels, title = ''):
    colors=["black", "red", "green", "blue", "yellow"]
    cmap = ListedColormap(colors)
    fig = plt.figure(figsize=[5, 5])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.scatter(z[:, 0], z[:, 1], c=labels, cmap=cmap, s=2)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_title(title)
#%%
class SemiSupervisedModel(nn.Module):
    def __init__(self):
        super(SemiSupervisedModel, self).__init__()
        self.fuzzy = FuzzyLayer.fromdimentions(2, 4, trainable=True)

    def forward(self, x):
        return self.fuzzy(x)

class NoisyClustersWithUnmarkedItemsDataset(Dataset):
    def __init__(self, clusters, size=1000, unmarked_items_ratio=0.8):
        self.clusters = clusters
        self.size = size
        self.unmarked_items_ratio = unmarked_items_ratio

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        label = np.random.randint(len(self.clusters))
        rcl = self.clusters[label]

        cluster_label = label
        if np.random.ranf() < self.unmarked_items_ratio:
            cluster_label = -1
        return rcl[0] * np.random.randn(2) + rcl[1:], cluster_label, label

#%%
ds = NoisyClustersWithUnmarkedItemsDataset([[0.1,-0.5,-0.5], 
                           [0.1, 0.5, 0.5],
                           [0.1, 0.5,-0.5],
                           [0.1,-0.5, 0.5]], 10000, 0.8)
dl = DataLoader(ds, batch_size=1024)
features, labels_masked, labels_true = next(iter(dl))
plot_clusters(features, labels_masked, title='Initial dataset')
#%%
model = SemiSupervisedModel()
res = model(features.float()).detach().numpy()
assigned_classes =[np.argmax(a) for a in res]
plot_clusters(features, assigned_classes, title='Initial cluster assignment')
#%%
def train(model, dataloader, epochs=20):
    opt = torch.optim.RMSprop(model.parameters())
    general_loss = nn.CrossEntropyLoss(reduction="sum")
    norm_loss = nn.MSELoss()

    for epoch in range(epochs):
        sum_loss = 0
        sum_norm_loss = 0
        for x, label_masked, _ in dataloader:
            opt.zero_grad()
            fuzzy_out = model(x.float())
            
            masked_fuzzy_out = fuzzy_out[label_masked != -1]
            loss_value = general_loss(masked_fuzzy_out, label_masked[label_masked != -1])
            loss_value.backward(retain_graph=True)

            unmasked_fuzzy_out = fuzzy_out[label_masked == -1].sum(axis=1)
            norm_loss_value = norm_loss(unmasked_fuzzy_out, torch.ones(unmasked_fuzzy_out.shape))
            norm_loss_value.backward()
            
            opt.step()
            sum_loss += loss_value
            sum_norm_loss += norm_loss_value
                    
        print(f"Epoch {epoch}: loss {sum_loss} norm loss {sum_norm_loss}")

    return model

train(model, dl, 100)
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
