import matplotlib.pyplot as plt
import numpy as np
import torch

from cvae import CVAE, Encoder, Decoder
from torchfuzzy.fuzzy_layer import FuzzyLayer
from sklearn.ensemble import IsolationForest

from utils import get_data, predict, class_scatter, get_emnist, pred_per_class

device = 'cuda:0'
is_multi_class = False
batch_size = 128
latent_dim = 6
dot_size = 3

DATA = [
    ("fz", "Fuzzy C-VAE"),
    ("mlp", "MLP C-VAE"),
]


def get_anomaly_rates(data, labels, detectors):
    anomaly_rates = [0 for _ in range(len(labels))]
    for label_index in range(len(labels)):
        scores = [detectors[a].predict(data[label_index]) for a in detectors.keys()]
        overall_score = np.max(scores, axis=0)
        overall_score = [1 if a == -1 else 0 for a in overall_score]
        anomaly_rates[label_index] = np.mean(overall_score)

    return anomaly_rates


def metrics(anomaly_rates):
    mean_ar = np.mean(anomaly_rates)
    plow_ar = np.percentile(anomaly_rates, 20)
    phigh_ar = np.percentile(anomaly_rates, 80)

    return mean_ar, plow_ar, phigh_ar


def draw_diff(ax, ar_0, ar_1, labels, sep):
    x_pos = np.arange(len(labels),)

    # if not sep:
    #     ar_common = np.array(ar_0) - np.array(ar_1)
    #     ar_0 = np.maximum(ar_common, 0)
    #     ar_1 = np.minimum(ar_common, 0)

    def ax_draw(target_ax):
        if sep:
            target_ax.bar(x_pos - 0.2, ar_0, width=0.4, align='center', label=DATA[0][1])
            target_ax.bar(x_pos + 0.2, ar_1, width=0.4, align='center', label=DATA[1][1])
            target_ax.set_ylim(0, 0.3)
        else:
            target_ax.bar(x_pos, ar_0, width=0.8, align='center', label=DATA[0][1])
            target_ax.bar(x_pos, ar_1, width=0.8, align='center', label=DATA[1][1])
            target_ax.set_ylim(0, 1)
            #target_ax.set_yticks(np.linspace(-0.3, 0.3, 7), labels=np.abs(np.round(np.linspace(-0.3, 0.3, 7), 1)))

        target_ax.set_xticks(x_pos, labels=labels)
        target_ax.set_xlabel(f'{"E" if len(labels) > 10 else ""}MNIST classes')
        target_ax.set_ylabel('Anomaly rate')
        target_ax.set_title(f'{"E" if len(labels) > 10 else ""}MNIST anomaly rates per class')
        target_ax.grid(True)
        target_ax.legend(loc='lower left')

    ax_draw(ax)

    dedicated_fig, dedicated_ax = plt.subplots(1, 1, figsize=(10, 5))
    ax_draw(dedicated_ax)
    dedicated_fig.savefig(f'./papers/iiti24/fig5{"a" if len(labels) > 10 else "b"}-anomaly-ratio.eps', format='eps')
    plt.close(dedicated_fig)


def draw_common(ax, m_0, m_1, labels):
    m0_mean, m0_plow, m0_phigh = m_0
    m1_mean, m1_plow, m1_phigh = m_1

    M = [m0_mean, m1_mean]
    error = [
        [m0_mean - m0_plow, m1_mean - m1_plow],
        [m0_phigh - m0_mean, m1_phigh - m1_mean]
    ]
    exps = [DATA[0][1], DATA[1][1]]
    x_pos = np.arange(len(exps))

    def ax_draw(target_ax):
        target_ax.bar(x_pos, M, yerr=error, align='center', capsize=10)
        target_ax.set_xticks(x_pos, labels=exps)
        target_ax.invert_xaxis()
        target_ax.set_ylabel('Rate of detected anomalies')
        target_ax.set_title(f'{"E" if len(labels) > 10 else ""}MNIST')
        if len(labels) > 10:
            target_ax.set_ylim(0.5, 1)
        else:
            target_ax.set_ylim(0.0, 0.25)
        target_ax.yaxis.grid(True)
        #target_ax.legend(loc='upper left')

    ax_draw(ax)

    dedicated_fig, dedicated_ax = plt.subplots(1, 1, figsize=(5, 5))
    ax_draw(dedicated_ax)
    dedicated_fig.savefig(f'./papers/iiti24/fig5{"c" if len(labels) > 10 else "d"}-anomaly-ratio.eps', format='eps')
    plt.close(dedicated_fig)


if __name__ == '__main__':
    *emnist_dataset, emnist_mapping = get_emnist()
    _, test_data = get_data(is_multi_class, batch_size, device)

    result = {}
    for i, (model_type, model_desc) in enumerate(DATA):
        model_path = f"./runs/mnist/{model_type}_{latent_dim}_{12 if is_multi_class else 10}.pt"
        print(model_path)
        model = torch.load(model_path, map_location=device).eval()

        mnist_mu, (pred_mnist, _), _ = predict(model, test_data, is_multi_class, is_emnist=False, device=device)
        mnist_mu_per_class = pred_per_class(mnist_mu, pred_mnist)

        detectors = {}
        for digit in range(0, 10):
            detector = IsolationForest(n_estimators=100, max_samples='auto', contamination='auto', max_features=1.0)
            detector.fit(mnist_mu_per_class[digit])
            detectors[digit] = detector

        emnist_mu, _, (gt_emnist, _) = predict(model, zip(*emnist_dataset), is_multi_class, is_emnist=True, device=device)
        emnist_mu_per_class = pred_per_class(emnist_mu, gt_emnist)
        emnist_anomaly_rates = get_anomaly_rates(emnist_mu_per_class, emnist_mapping.values(), detectors)
        mnist_anomaly_rates = get_anomaly_rates(mnist_mu_per_class, list(map(str, range(0, 10))), detectors)

        emnist_metrics = metrics(emnist_anomaly_rates)
        mnist_metrics = metrics(mnist_anomaly_rates)
        print("Anomaly rate on EMNIST mean {} plow {} phigh {}".format(*emnist_metrics))
        print("Anomaly rate on MNIST mean {} plow {} phigh {}".format(*mnist_metrics))

        result[(model_type, model_desc)] = (emnist_anomaly_rates, emnist_metrics, mnist_anomaly_rates, mnist_metrics)

    fig, axs = plt.subplots(2, 2, figsize=(20, 20))

    (emnist_ar_0, emnist_m_0, mnist_ar_0, mnist_m_0) = result[DATA[0]]
    (emnist_ar_1, emnist_m_1, mnist_ar_1, mnist_m_1) = result[DATA[1]]
    draw_common(axs[0, 0], emnist_m_0, emnist_m_1, emnist_mapping.values())
    draw_common(axs[0, 1], mnist_m_0, mnist_m_1, list(map(str, range(0, 10))))
    draw_diff(axs[1, 0], emnist_ar_0, emnist_ar_1, emnist_mapping.values(), sep=False)
    draw_diff(axs[1, 1], mnist_ar_0, mnist_ar_1, list(map(str, range(0, 10))), sep=True)
    plt.show()
