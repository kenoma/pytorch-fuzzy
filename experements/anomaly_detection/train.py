import os
from datetime import datetime

import piqa
import torch
from matplotlib import pyplot as plt
from sklearn import metrics
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import get_mnist_loaders
from model import VAE

batch_size = 256
learning_rate = 2e-3
num_epochs = 300
latent_dim = 8
mnist_class_anomaly = [2, 3, 4, 5, 6, 7, 8, 9, 0]
fuzzy_rules_count = 8
beta = 1e-3
gamma = 1

prefix = f"fuzzy_cvae_mamdani_anomaly"
os.makedirs('./runs', exist_ok=True)
os.makedirs('./images', exist_ok=True)
writer = SummaryWriter(f'runs/{prefix}_{datetime.now().strftime("%Y%m%d-%H%M%S")}')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ssim = piqa.SSIM(window_size = 14, n_channels=1, reduction='none').to(device)

def compute_loss(x, recon_x, mu, logvar, fz, most_active_layer):
    diff = ssim((x + 1) / 2, (recon_x + 1) / 2)

    most_active_layer_index, most_active_layer_value = most_active_layer
    fz_loss = (1 - fz.sum(-1)) ** 2
    fz_loss = fz_loss.mean()

    if most_active_layer_value > 1 / fuzzy_rules_count:
        most_active_loss_value = (fz[:, most_active_layer_index] ** 2).mean()
        fz_loss += most_active_loss_value

    loss_recon = (
                1 - diff).abs().mean()  # F.binary_cross_entropy((recon_x+1)/2, (x + 1)/2, reduction='none').sum(-1).mean()#

    tsquare = torch.square(mu)
    tlogvar = torch.exp(logvar)
    kl_loss = -0.5 * (1 + logvar - tsquare - tlogvar)
    loss_kl = kl_loss.sum(-1).mean()

    loss = loss_recon + beta * loss_kl

    return loss, loss_recon, loss_kl, fz_loss


def keep_eigenvals_positive_loss(layer, eps = 1e-15):
    ev = layer.get_transformation_matrix_eigenvals().real.min()
    ev = torch.clamp(ev, max=eps)
    return -ev


def get_arate(inp):
    _, _, fz = model.forward(inp)
    return fz.log().quantile(0.5, dim=-1).abs().cpu().numpy()  # ssim((inp + 1)/2, (recon_x+1)/2).cpu().numpy() #


def train(model, dataloader, optimizer, prev_updates, writer=None):
    model.train()

    for batch_idx, (data, _) in enumerate(tqdm(dataloader, disable=True)):
        n_upd = prev_updates + batch_idx

        data = data.to(device)

        optimizer.zero_grad()

        mu, logvar, z = model.half_pass(data)
        recon_x, fz = model.decoder_pass(z)

        loss, _, _, fz_loss = compute_loss(data, recon_x, mu, logvar, fz,
                                           model.decoder.activation_memorizer.get_most_active())

        centroids = model.decoder.fuzzy.get_centroids()
        # spatial_loss = torch.cdist(mu, centroids).mean() + (2e-1 - torch.cdist(centroids, centroids).topk(k=2, dim=-1, largest=False).values.mean())

        ev_loss = keep_eigenvals_positive_loss(model.decoder.fuzzy)

        loss.backward(retain_graph=True)
        # spatial_loss.backward(retain_graph=True)

        if ev_loss.item() > 0:
            fz_loss.backward(retain_graph=True)
            ev_loss.backward()
        else:
            fz_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

    return prev_updates + len(dataloader)


fixed_random_z = torch.randn(16, latent_dim).to(device)


def val(model, dataloader, cur_step, writer=None):
    model.eval()
    test_recon_loss = 0
    test_kl_loss = 0
    test_fz_loss = 0

    lab_true = []
    lab_pred = []

    with torch.no_grad():
        for data, lab in tqdm(test_loader, desc='Test MNIST', disable=True):
            data = data.view((-1, 1, 28, 28)).to(device)
            _, _, fz = model.forward(data)
            rates = fz.log().quantile(0.5, dim=-1).abs().cpu().numpy()

            for oie, f, l in zip(range(len(rates)), rates, lab):
                lab_pred.append(f)
                if l == mnist_class_anomaly:
                    # print(f"ANOMALY {fz[oie]}")
                    lab_true.append(1)
                else:
                    # print(f"NORMAL {fz[oie]}")
                    lab_true.append(0)

    fpr, tpr, _ = metrics.roc_curve(lab_true, lab_pred)
    roc_auc = metrics.auc(fpr, tpr)

    with torch.no_grad():
        for data, _ in tqdm(dataloader, desc='Testing', disable=True):
            data = data.to(device)
            mu, logvar, z = model.half_pass(data)
            recon_x, fz = model.decoder_pass(z)

            _, loss_recon, loss_kl, fz_loss = compute_loss(data, recon_x, mu, logvar, fz,
                                                           model.decoder.activation_memorizer.get_most_active())

            test_recon_loss += loss_recon.item()
            test_kl_loss += loss_kl.item()
            test_fz_loss += fz_loss.item()

    test_recon_loss /= len(dataloader)
    test_kl_loss /= len(dataloader)
    test_fz_loss /= len(dataloader)

    print(
        f'[{cur_step}] Reconstruction loss: {test_recon_loss:.4f}, KLD: {test_kl_loss:.4f} AUC {roc_auc:.4f} FZ {test_fz_loss:.4f}')

    if writer is not None:
        writer.add_scalar('ADFVAE/AUC', roc_auc, global_step=cur_step)
        writer.add_scalar('ADFVAE/Reconstruction', test_recon_loss, global_step=cur_step)
        writer.add_scalar('ADFVAE/KLD', test_kl_loss, global_step=cur_step)
        writer.add_scalar('ADFVAE/Fuzzy', test_fz_loss, global_step=cur_step)

        samples, _ = model.decoder_pass(fixed_random_z)
        writer.add_images('ADFVAE/Samples', samples.view(-1, 1, 28, 28), global_step=cur_step)

def get_activation_stats(model, dataloader):
    rulestat = {}
    with torch.no_grad():
        for _, (data, _) in enumerate(tqdm(dataloader)):
            data = data.to(device)
            _, _, fz = model.forward(data)
            act_fz = fz.max(-1).indices.cpu().numpy()
            for ind in act_fz:
                rulestat[ind] = rulestat.get(ind, 0) + 1
    return rulestat


if __name__ == '__main__':
    for mnist_class_anomaly_digit in mnist_class_anomaly:
        print(mnist_class_anomaly_digit)
        train_loader, test_loader = get_mnist_loaders(batch_size, mnist_class_anomaly_digit)
        model = VAE(latent_dim, fuzzy_rules_count, batch_size * 32).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        prev_updates = 0
        for epoch in range(num_epochs):
            prev_updates = train(model, train_loader, optimizer, prev_updates, writer=writer)
            val(model, test_loader, prev_updates, writer=writer)
            scheduler.step()

        torch.save(model, f'./runs/fuzzy_mnist_{mnist_class_anomaly_digit}.pt')

        train_stat = get_activation_stats(model, train_loader)
        test_stat = get_activation_stats(model, test_loader)

        plt.bar(list(train_stat.keys()), train_stat.values(), 1, color='g')
        plt.bar(list(test_stat.keys()), test_stat.values(), 1, color='r')
        plt.savefig(f'./images/fig1-{mnist_class_anomaly_digit}-distribution.eps', format='eps')

        writer.close()