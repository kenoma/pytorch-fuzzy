import torch
import torch.nn as nn
import numpy as np
from torch import Tensor

from torchfuzzy import FuzzyLayer, DefuzzyLinearLayer, FuzzyBellLayer

KERNELS = 2

class Encoder(nn.Module):
    """
    Компонент энкодера для VAE

    Args:
        latent_dim (int): Размер латентного вектора.
    """

    def __init__(self, latent_dim):
        super(Encoder, self).__init__()

        self.input = nn.Sequential(
            nn.Conv2d(1, KERNELS, kernel_size=3, padding=2, stride=1),
            nn.BatchNorm2d(KERNELS, track_running_stats=False),
            nn.SiLU(),
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(KERNELS, 2 * KERNELS, kernel_size=2, stride=2, padding=2),
            nn.BatchNorm2d(2 * KERNELS, track_running_stats=False),
            nn.SiLU(),
            nn.Conv2d(2 * KERNELS, 2 * KERNELS, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(2 * KERNELS, track_running_stats=False),
            nn.SiLU(),
        )

        self.block_3 = nn.Sequential(
            nn.Conv2d(2 * KERNELS, 4 * KERNELS, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(4 * KERNELS, track_running_stats=False),
            nn.SiLU(),
            nn.Conv2d(4 * KERNELS, 4 * KERNELS, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(4 * KERNELS, track_running_stats=False),
            nn.SiLU(),
        )

        self.block_4 = nn.Sequential(
            nn.Conv2d(4 * KERNELS, 8 * KERNELS, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(8 * KERNELS, track_running_stats=False),
            nn.SiLU(),
            nn.Conv2d(8 * KERNELS, 8 * KERNELS, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(8 * KERNELS, track_running_stats=False),
            nn.SiLU(),
        )

        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * KERNELS, 2 * latent_dim),  # mean + variance.
        )

        self.downscale_1 = nn.Conv2d(KERNELS, 8 * KERNELS, kernel_size=30)
        self.downscale_2 = nn.Conv2d(2 * KERNELS, 8 * KERNELS, kernel_size=10)
        self.downscale_3 = nn.Conv2d(4 * KERNELS, 8 * KERNELS, kernel_size=6)
        self.after_sum = nn.SiLU()
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

        x = self.input(x)
        # print(x.shape)
        res_1 = self.downscale_1(x)

        x = self.block_2(x)
        # print(x.shape)
        res_2 = self.downscale_2(x)

        x = self.block_3(x)
        # print(x.shape)
        res_3 = self.downscale_3(x)

        x = self.block_4(x)

        x = self.after_sum(x + res_1 + res_2 + res_3)
        x = self.out(x)

        mu, logvar = torch.chunk(x, 2, dim=-1)
        scale = self.softplus(logvar) + eps
        scale_tril = torch.diag_embed(scale)
        dist = torch.distributions.MultivariateNormal(mu, scale_tril=scale_tril)
        z = dist.rsample()

        return mu, logvar, z

class ActivationMemorizer(torch.nn.Module):
    def __init__(self, input_size: int, memory_size: int):
        super().__init__()
        self.memory_size = memory_size
        self.memory = torch.zeros((self.memory_size, input_size), dtype=torch.bool)
        self.counter = torch.tensor(0, dtype=torch.long)

    def forward(self, input: Tensor) -> Tensor:
        most_active = torch.argmax(input, dim=1)
        memory_slice = self.memory[self.counter:self.counter + len(most_active), :]
        most_active = most_active[:len(memory_slice)]
        memory_slice[torch.arange(len(most_active)), :] = False
        memory_slice[torch.arange(len(most_active)), most_active] = True
        self.counter = (self.counter + len(most_active)) % self.memory_size

        return input

    def get_most_active(self):
        all_activations_per_item = torch.sum(self.memory, dim=0)
        all_activations = torch.sum(self.memory)
        most_active = np.argmax(all_activations_per_item)

        return most_active, all_activations_per_item[most_active] / all_activations


class Decoder(nn.Module):
    """
    Компонент декодера для VAE

    Args:
        latent_dim (int): Размер латентного вектора.
    """

    def __init__(self, latent_dim, fuzzy_rules_count, memory_size):
        super(Decoder, self).__init__()

        initial_centroids = np.random.rand(fuzzy_rules_count, latent_dim)
        initial_scales = 1e-2 * np.ones((fuzzy_rules_count, latent_dim))
        self.fuzzy = FuzzyBellLayer.from_centers_and_scales(initial_centroids, initial_scales, trainable=True)
        self.activation_memorizer = ActivationMemorizer(fuzzy_rules_count, memory_size)

        self.decoder = nn.Sequential(
            nn.Linear(fuzzy_rules_count, 16 * KERNELS),
            nn.SiLU(),
            nn.BatchNorm1d(16 * KERNELS, track_running_stats=False),
            nn.Unflatten(1, (16 * KERNELS, 1, 1)),
            nn.ConvTranspose2d(16 * KERNELS, 8 * KERNELS, 12),
            nn.SiLU(),
            nn.BatchNorm2d(8 * KERNELS, track_running_stats=False),
            nn.ConvTranspose2d(8 * KERNELS, 4 * KERNELS, 5),
            nn.SiLU(),
            nn.BatchNorm2d(4 * KERNELS, track_running_stats=False),
            nn.ConvTranspose2d(4 * KERNELS, 2 * KERNELS, 5),
            nn.SiLU(),
            nn.BatchNorm2d(2 * KERNELS, track_running_stats=False),
            nn.ConvTranspose2d(2 * KERNELS, KERNELS, 5),
            nn.SiLU(),
            nn.BatchNorm2d(KERNELS, track_running_stats=False),
            nn.ConvTranspose2d(KERNELS, 1, 5),
            nn.Tanh()
        )

    def forward(self, z):
        """
        Декодирует латентный вектор в исходное представление

        Args:
            z (torch.Tensor): Латентный вектор.

        Returns:
            x
        """
        fz = self.fuzzy(z)
        fz = self.activation_memorizer(fz)
        x = self.decoder(fz)
        return x, fz


class VAE(nn.Module):
    """
    Args:
        latent_dim (int): Размер латентного вектора.
    """

    def __init__(self, latent_dim, fuzzy_rules_count, memory_size):
        super(VAE, self).__init__()

        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim, fuzzy_rules_count, memory_size)

    def forward(self, x):
        """

        """
        mu, _, _, = self.encoder(x)
        x_recon, fz = self.decoder(mu)
        return mu, x_recon, fz

    def half_pass(self, x):
        """

        """
        mu, logvar, z = self.encoder(x)
        return mu, logvar, z

    def decoder_pass(self, x):
        return self.decoder(x)
