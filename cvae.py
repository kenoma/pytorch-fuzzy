import torch
import torch.nn as nn
from torchfuzzy.fuzzy_layer import FuzzyLayer


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
