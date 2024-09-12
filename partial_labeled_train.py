import sys
from dataclasses import dataclass
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from tqdm.auto import tqdm
from torchfuzzy.fuzzy_layer import FuzzyLayer

batch_size = 256
learning_rate = 2e-3
weight_decay = 1e-2
num_epochs = 50
latent_dim = 6
output_dims = 2
beta = 1
gamma = 1
fuzzy_labels = 10
is_fuzzy_loss_active = True

is_fuzzy = bool(int(sys.argv[1]))
device = torch.device(f'cuda:{sys.argv[2]}')
is_cut = bool(int(sys.argv[3]))


def get_target_and_mask(target_label, unknown_ratio):
    """
    Возвращает вектор целевого значения и маску в виде сдвоенного тензора

    Args:
        target_label (int): Метка класса
        unknown_ratio (float): Доля примеров в датасете, чья разметка будет игнорироваться при обучении

    Returns:
        tensor (2, 12)
    """
    t = F.one_hot(torch.LongTensor([target_label]), fuzzy_labels)
    m = torch.ones((1, fuzzy_labels)) if torch.rand(1) > unknown_ratio else torch.zeros((1, fuzzy_labels))

    return torch.cat((t, m), 0).to(device)


class Encoder(nn.Module):
    """
    Компонент энкодера для VAE

    Args:
        latent_dim (int): Размер латентного вектора.
    """

    def __init__(self, latent_dim):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5),
            nn.SiLU(),
            nn.Conv2d(8, 16, kernel_size=5),
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=5),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(9216, 625),
            nn.BatchNorm1d(625),
            nn.SiLU(),
            nn.Linear(625, 2 * latent_dim),  # mean + variance.
        )
        self.softplus = nn.Softplus()

    def forward(self, x, eps: float = 1e-8):
        """
        Выход энкодера для чистого VAE.

        Args:
            x (torch.Tensor): Входной вектор.
            eps (float): Небольшая поправка к скейлу для лучшей сходимости и устойчивости.

        Returns:
            mu, logvar, z, dist
        """

        x = self.encoder(x)
        mu, logvar = torch.chunk(x, 2, dim=-1)
        scale = self.softplus(logvar) + eps
        scale_tril = torch.diag_embed(scale)
        dist = torch.distributions.MultivariateNormal(mu, scale_tril=scale_tril)
        z = dist.rsample()

        return mu, logvar, z


class Decoder(nn.Module):
    """
    Компонент декодера для VAE

    Args:
        latent_dim (int): Размер латентного вектора.
    """

    def __init__(self, latent_dim):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 625),
            nn.BatchNorm1d(625),
            nn.SiLU(),
            nn.Linear(625, 9216),
            nn.BatchNorm1d(9216),
            nn.SiLU(),
            nn.Unflatten(1, (64, 12, 12)),
            nn.ConvTranspose2d(64, 32, 5),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 16, 5),
            nn.SiLU(),
            nn.ConvTranspose2d(16, 8, 5),
            nn.SiLU(),
            nn.ConvTranspose2d(8, 1, 5),
            nn.Sigmoid()
        )

    def forward(self, z):
        """
        Декодирует латентный вектор в исходное представление

        Args:
            z (torch.Tensor): Латентный вектор.

        Returns:
            x
        """

        x = self.decoder(z)

        return x


class CVAE(nn.Module):
    """
    Conditional Variational Autoencoder (C-VAE)

    Args:
        latent_dim (int): Размер латентного вектора.
        labels_count (int): Количество выходов классификатора
    """

    def __init__(self, latent_dim, labels_count, output_dims, fuzzy):
        super(CVAE, self).__init__()

        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.output_dims = output_dims

        if fuzzy:
            self.output = nn.Sequential(
                FuzzyLayer.fromdimentions(output_dims, labels_count, trainable=True)
            )
        else:
            self.output = nn.Sequential(
                nn.Linear(output_dims, labels_count),
                nn.Sigmoid(),
                nn.Linear(labels_count, labels_count),
                nn.Sigmoid()
            )

    def forward(self, x):
        """
        Возвращает компоненты внутренних слоев CVAE, результаты реконструкции и классификации

        Args:
            x (torch.Tensor): Входной вектор.

        Returns:
            mu, x_recon, labels
        """

        mu, _, _, = self.encoder(x)
        x_recon = self.decoder(mu)

        labels = self.output(mu[:, :self.output_dims])

        return mu, x_recon, labels

    def half_pass(self, x):
        """
        Возвращает результаты работы энкодера и классификатора
        """
        mu, logvar, z = self.encoder(x)
        labels = self.output(mu[:, :self.output_dims])

        return mu, logvar, z, labels

    def decoder_pass(self, x):
        return self.decoder(x)


def compute_loss(x, recon_x, mu, logvar, z, target_labels, predicted_labels):
    loss_recon = F.binary_cross_entropy(recon_x, x + 0.5, reduction='none').sum(-1).mean()

    tsquare = torch.square(mu)
    tlogvar = torch.exp(logvar)
    kl_loss = -0.5 * (1 + logvar - tsquare - tlogvar)
    loss_kl = kl_loss.sum(-1).mean()

    target_firings = target_labels[:, 0, :]
    mask = target_labels[:, 1, :]
    loss_fuzzy = (mask * torch.square(target_firings - predicted_labels)).sum(-1).mean()

    loss = loss_recon + beta * loss_kl + gamma * loss_fuzzy

    return loss, loss_recon, loss_kl, loss_fuzzy


def train(model, dataloader, optimizer, prev_updates, writer=None):
    model.train()
    latest_losses = []
    for batch_idx, (data, target) in enumerate(dataloader):
        n_upd = prev_updates + batch_idx

        data = data.to(device)

        optimizer.zero_grad()

        mu, logcar, z, labels = model.half_pass(data)
        recon_x = model.decoder_pass(z)
        loss, loss_recon, loss_kl, loss_fuzzy = compute_loss(data, recon_x, mu, logcar, z, target, labels)

        loss.backward()

        if n_upd % 100 == 0:
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)

            latest_losses = [loss.item(), loss_recon.item(), loss_kl.item(), loss_fuzzy.item()]

            if writer is not None:
                global_step = n_upd
                writer.add_scalar('Loss/Train', loss.item(), global_step)
                writer.add_scalar('Loss/Train/BCE', loss_recon.item(), global_step)
                writer.add_scalar('Loss/Train/KLD', loss_kl.item(), global_step)
                writer.add_scalar('Fuzzy/Train/Loss', loss_fuzzy.item(), global_step)
                writer.add_scalar('GradNorm/Train', total_norm, global_step)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

    return prev_updates + len(dataloader), latest_losses


def test(model, dataloader, cur_step, writer=None):
    model.eval()
    test_loss = 0
    test_recon_loss = 0
    test_kl_loss = 0
    test_fuzzy_loss = 0
    test_accuracy = 0

    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device)
            mu, logcar, z, labels = model.half_pass(data)
            recon_x = model.decoder_pass(z)
            loss, loss_recon, loss_kl, loss_fuzzy = compute_loss(data, recon_x, mu, logcar, z, target, labels)

            test_loss += loss.item()
            test_recon_loss += loss_recon.item()
            test_kl_loss += loss_kl.item()
            test_fuzzy_loss += loss_fuzzy.item()
            pred_target = np.argmax(labels[:, 0:10].cpu().numpy(), axis=1)
            target_labels = np.argmax(target[:, 0, 0:10].cpu().numpy(), axis=1)
            test_accuracy += np.sum(target_labels == pred_target) / len(pred_target)

    test_loss /= len(dataloader)
    test_recon_loss /= len(dataloader)
    test_kl_loss /= len(dataloader)
    test_fuzzy_loss /= len(dataloader)
    test_accuracy /= len(dataloader)

    if writer is not None:
        writer.add_scalar('Loss/Test', test_loss, global_step=cur_step)
        writer.add_scalar('Loss/Test/BCE', loss_recon.item(), global_step=cur_step)
        writer.add_scalar('Loss/Test/KLD', loss_kl.item(), global_step=cur_step)
        writer.add_scalar('Fuzzy/Test/Loss', loss_fuzzy.item(), global_step=cur_step)
        writer.add_scalar('Fuzzy/Test/Accuracy', test_accuracy, global_step=cur_step)

        z = torch.randn(16, latent_dim).to(device)
        samples = model.decoder_pass(z)
        writer.add_images('Test/Samples', samples.view(-1, 1, 28, 28), global_step=cur_step)

    return [test_loss, loss_recon.item(), loss_kl.item(), loss_fuzzy.item(), test_accuracy]


def train_pl(number_of_exp, is_fuzzy_cvae):
    ratios = np.linspace(0.0, 1, 11)

    exp_bar = tqdm(total=number_of_exp, desc=f"exp {'fz' if is_fuzzy_cvae else 'mlp'}")
    ratio_bar = tqdm(total=len(ratios), desc=f"ratio")
    epoch_bar = tqdm(total=num_epochs, desc=f"epoch")

    for i in range(number_of_exp):
        ratio_bar.reset(len(ratios))
        with open(f"./papers/iiti24/LOCAL{'pl' if not is_cut else 'cut'}_{i}_{'fz' if is_fuzzy_cvae else 'mlp'}.csv", 'w') as f:
            f.write(f"ratio,epoch,train_loss,train_loss_recon,train_loss_kl,train_loss_fuzzy,test_loss,test_loss_recon,test_loss_kl,test_loss_fuzzy,test_accuracy\n")
            for ratio in ratios:
                epoch_bar.reset(num_epochs)

                if is_cut:
                    train_data = datasets.MNIST(
                        '~/.pytorch/MNIST_data/',
                        download=True,
                        train=True,
                        transform=transform,
                        target_transform=transforms.Lambda(lambda x: get_target_and_mask(x, 0))
                    )
                    cut_train, _ = train_test_split(train_data, test_size=min(0.99, max(0.01, ratio)))
                else:
                    train_data = datasets.MNIST(
                        '~/.pytorch/MNIST_data/',
                        download=True,
                        train=True,
                        transform=transform,
                        target_transform=transforms.Lambda(lambda x: get_target_and_mask(x, ratio))
                    )
                    cut_train = train_data

                train_loader = torch.utils.data.DataLoader(
                    cut_train,
                    batch_size=batch_size,
                    shuffle=True,
                )
                model = CVAE(latent_dim=latent_dim, labels_count=fuzzy_labels, output_dims=output_dims,
                             fuzzy=is_fuzzy_cvae).to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

                prev_updates = 0
                for epoch in range(num_epochs):
                    prev_updates, train_data = train(model, train_loader, optimizer, prev_updates, writer=writer)
                    test_data = test(model, test_loader, prev_updates, writer=writer)
                    f.write(f"{ratio},{epoch},{','.join(map(str, train_data))},{','.join(map(str, test_data))}\n")
                    f.flush()
                    epoch_bar.update(1)
                ratio_bar.update(1)
        exp_bar.update(1)


if __name__ == '__main__':
    writer = SummaryWriter(f'runs/mnist/{"pl" if not is_cut else "cut"}vae_{datetime.now().strftime("%Y%m%d-%H%M%S")}')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1, 28, 28) - 0.5),
    ])

    test_data = datasets.MNIST(
        '~/.pytorch/MNIST_data/',
        download=True,
        train=False,
        transform=transform,
        target_transform=transforms.Lambda(lambda x: get_target_and_mask(x, 0))
    )

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
    )

    train_pl(6, is_fuzzy)
