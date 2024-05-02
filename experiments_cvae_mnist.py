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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#%%
class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, latent_dims)
        self.linear3 = nn.Linear(512, latent_dims)
        self.fuzzy = FuzzyLayer.fromcenters(initial_centers=[
            [ 0.8090169943,  0.5877852524], 
            [ 0.3090169938,  0.9510565165], 
            [-0.3090169938,  0.9510565165], 
            [-0.8090169943,  0.5877852524], 
            [-1.,            0.          ], 
            [-0.8090169943, -0.5877852524], 
            [-0.3090169938, -0.9510565165], 
            [ 0.3090169938, -0.9510565165], 
            [ 0.8090169943, -0.5877852524], 
            [ 1.,            0.          ]], trainable=True)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() 
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu =  self.linear2(x)
        fz = self.fuzzy(mu)

        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, fz

class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, 784)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z.reshape((-1, 1, 28, 28))

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z, fz = self.encoder(x)
        return self.decoder(z), fz
#%%
def train(autoencoder, data, epochs=20):
    opt = torch.optim.Adadelta(autoencoder.parameters())
    ploss = nn.MSELoss(reduction="sum")

    for epoch in range(epochs):
        sum_floss = 0
        sum_loss = 0
        count= 0
        for x, y in data:
            x = x.to(device) 
            opt.zero_grad()
            x_hat,fz_x = autoencoder(x)
            gy = F.one_hot(y, num_classes=10).to(device)
            fuzzy_loss_value = ploss(fz_x, gy.float())
            loss = ((x - x_hat)**2).sum() + autoencoder.encoder.kl

            loss.backward(retain_graph=True)
            fuzzy_loss_value.backward()
            

            opt.step()
            count += 1
            sum_floss += fuzzy_loss_value
            sum_loss += loss
                    
        print(f"Epoch {epoch}: loss {sum_loss/count} ploss {sum_floss/count}")
        count = 0
        sum_loss = 0
        sum_floss = 0

    return autoencoder

latent_dims = 2
data = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data', 
               transform=torchvision.transforms.ToTensor(), 
               download=True),
        batch_size=256,
        shuffle=True)
vae = VariationalAutoencoder(latent_dims).to(device)
vae = train(vae, data, 20)
#%%
def plot_latent(autoencoder, data, num_batches=100):
    for i, (x, y) in enumerate(data):
        z,fz = autoencoder.encoder(x.to(device))
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            break

def plot_reconstructed(autoencoder, r0=(-5, 10), r1=(-10, 5), n=12):
    w = 28
    img = np.zeros((n*w, n*w))
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[x, y]]).to(device)
            x_hat = autoencoder.decoder(z)
            x_hat = x_hat.reshape(28, 28).to('cpu').detach().numpy()
            img[(n-1-i)*w:(n-1-i+1)*w, j*w:(j+1)*w] = x_hat
    plt.imshow(img, extent=[*r0, *r1])

#%%
plot_latent(vae, data)
#%%
plot_reconstructed(vae, r0=(-3, 3), r1=(-3, 3))



# %%
