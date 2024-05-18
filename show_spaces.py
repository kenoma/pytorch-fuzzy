import matplotlib.pyplot as plt
import torch

from cvae import CVAE, Encoder, Decoder
from torchfuzzy.fuzzy_layer import FuzzyLayer

from utils import get_data, predict, class_scatter

device = 'cuda:0'
is_multi_class = True
batch_size = 128
latent_dim = 2
dot_size = 3

DATA = [
    ("vae", "VAE"),
    ("mlp", "MLP C-VAE"),
    ("fz", "Fuzzy C-VAE"),
]

if __name__ == '__main__':
    _, test_data = get_data(is_multi_class, batch_size, device)
    fig, axs = plt.subplots(1 if not is_multi_class else 2, len(DATA), figsize=(5 * len(DATA), 12 if is_multi_class else 5))
    fig_suffix = 'a' if not is_multi_class else 'b'

    for i, (model_type, model_desc) in enumerate(DATA):
        model_path = f"./runs/mnist/{model_type}_{latent_dim}_{12 if is_multi_class else 10}.pt"
        print(model_path)
        model = torch.load(model_path, map_location=device).eval()

        x_axis_label = f"{model_desc} component 1"
        y_axis_label = f"{model_desc} component 2"
        title = f'{model_desc} latent space'

        pred_mu, (pred_digits, pred_shapes), (gt_digits, gt_shapes) = predict(model, test_data, is_multi_class, is_emnist=False, device=device)
        target_ax = axs[0, i] if is_multi_class else axs[i]
        class_scatter(target_ax, *pred_mu.T[:2], gt_digits, x_axis_label, y_axis_label, title, s=dot_size)
        dedicated_fig, dedicated_ax = plt.subplots(1, 1, figsize=(5, 5))
        class_scatter(dedicated_ax, *pred_mu.T[:2], gt_digits, x_axis_label, y_axis_label, title, s=dot_size)
        dedicated_fig.savefig(f'./papers/iiti24/fig2{fig_suffix}-{model_type}-all-features.eps', format='eps')
        plt.close(dedicated_fig)

        if is_multi_class:
            human_gt = ["Closed loop contour" if x == 0 else "Not closed loop contour" for x in gt_shapes]
            dedicated_fig, dedicated_ax = plt.subplots(1, 1, figsize=(5, 5))
            class_scatter(axs[1, i], *pred_mu.T[:2], human_gt, x_axis_label, y_axis_label, title, s=dot_size)
            class_scatter(dedicated_ax, *pred_mu.T[:2], human_gt, x_axis_label, y_axis_label, title, s=dot_size)
            dedicated_fig.savefig(f'./papers/iiti24/fig2c-{model_type}-all-features.eps', format='eps')
            plt.close(dedicated_fig)

    plt.show()
