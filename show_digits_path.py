import matplotlib.pyplot as plt
import numpy as np
import torch
import string
from cvae import CVAE, Encoder, Decoder
from torchfuzzy.fuzzy_layer import FuzzyLayer

from utils import get_data, predict, class_scatter, get_emnist

device = 'cuda:0'
latent_dim = 2
fraction = 10

model_path = f"./runs/mnist/fz_{latent_dim}_12.pt"

points = [
        # 6 to 1
        ([-1.578, -0.277], [-0.81, -1.026]),
        ([-0.625, -0.222], [-0.521, 0.762]),
        # 6 to 6
        ([-1.578, -0.277], [-0.625, -0.222])
    ]


axbig_size = 3
fig, axs = plt.subplots(len(points) + axbig_size, fraction, figsize=(fraction, len(points) + axbig_size * 2))
gs = axs[0, 0].get_gridspec()
for i in range(axbig_size):
    for ax in axs[i, :]:
        ax.remove()
space_ax = fig.add_subplot(gs[:axbig_size, :])

dedicated_space_fig, dedicated_space_ax = plt.subplots(1, figsize=(6, 5))

if __name__ == '__main__':
    _, test_data = get_data(is_multi_class=True, batch_size=128, device=device)
    model = torch.load(model_path, map_location=device).eval()
    pred_mu, (pred_digits, pred_shapes), (gt_digits, gt_shapes) = predict(model, test_data, is_multi_class=True,
                                                                          is_emnist=False, device=device)
    class_scatter(space_ax, *pred_mu.T[:2], gt_digits, "", "", "", s=3)
    class_scatter(dedicated_space_ax, *pred_mu.T[:2], gt_digits, "Fuzzy C-VAE component 1", "Fuzzy C-VAE component 2", "Fuzzy C-VAE latent space", s=3)

    for points_index, (from_point, to_point) in enumerate(points):
        space_ax.plot([from_point[0], to_point[0]], [from_point[1], to_point[1]], 'o-', color='black')
        dedicated_space_ax.plot([from_point[0], to_point[0]], [from_point[1], to_point[1]], 'o-', color='black')
        path = np.linspace(from_point, to_point, fraction)
        path = [torch.FloatTensor(a) for a in path]
        space_ax.text(*(path[fraction // 2] + 0.03), string.ascii_lowercase[points_index])
        dedicated_space_ax.text(*(path[fraction // 2] + 0.03), string.ascii_lowercase[points_index])

        z = torch.stack(path, dim=0).to(device)

        samples = model.decoder_pass(z)
        samples = torch.sigmoid(samples)

        dedicated_fig, dedicated_ax = plt.subplots(1, 10, figsize=(12, 2))
        for i in range(fraction):
            dedicated_ax[i].imshow(samples[i].view(28, 28).cpu().detach().numpy(), cmap='gray')
            dedicated_ax[i].axis('off')
            axs[points_index + axbig_size, i].imshow(samples[i].view(28, 28).cpu().detach().numpy(), cmap='gray')
            axs[points_index + axbig_size, i].axis('off')
            if i == fraction // 2:
                axs[points_index + axbig_size, i].set_title(string.ascii_lowercase[points_index])
        dedicated_fig.savefig(f'./papers/iiti24/fig4{string.ascii_lowercase[points_index + 1]}-sample-generation.eps', format='eps')
        plt.close(dedicated_fig)

    dedicated_space_fig.savefig(f'./papers/iiti24/fig4a-sample-generation.eps', format='eps')
    plt.close(dedicated_space_fig)
    plt.show()
