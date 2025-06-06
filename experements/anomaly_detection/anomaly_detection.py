import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import IsolationForest
from tqdm import tqdm

from data import get_mnist_loaders
from model import VAE

latent_dim = 7
fuzzy_rules_count = 20
mnist_anomaly_digit = 1
batch_size = 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    model = VAE(latent_dim, fuzzy_rules_count, batch_size * 32).to(device)
    state_dict = torch.load(f'./runs/fuzzy_mnist_{mnist_anomaly_digit}.pt')
    model.load_state_dict(state_dict)

    train_loader, test_loader = get_mnist_loaders(batch_size, None)
    X_train, X_test, y_train, y_test = [], [], [], []
    with torch.no_grad():
        for data, target in tqdm(train_loader, desc='Encoding'):
            data = data.view((-1, 1, 28, 28)).to(device)
            fz, _, _ = model.forward(data)
            for t, f in zip(target.cpu().numpy(), fz.cpu().numpy()):
                if t == mnist_anomaly_digit:
                    continue
                X_train.append(f)
                y_train.append(t)
        for data, target in tqdm(test_loader, desc='Encoding'):
            data = data.view((-1, 1, 28, 28)).to(device)
            fz, _, _ = model.forward(data)
            for t, f in zip(target.cpu().numpy(), fz.cpu().numpy()):
                X_test.append(f)
                if t == mnist_anomaly_digit:
                    y_test.append(0)
                else:
                    y_test.append(1)

    X_train, X_test, y_train, y_test = np.stack(X_train), np.stack(X_test), np.stack(y_train), np.stack(y_test)

    anomaly_model = IsolationForest(contamination=0.22)
    anomaly_model.fit(X_train)

    y_pred = anomaly_model.predict(X_test)
    y_pred_proba = anomaly_model.decision_function(X_test)
    y_pred = [int(a == 1) for a in y_pred]

    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_proba)
    roc_auc = metrics.auc(fpr, tpr)

    fig = plt.figure(figsize=(4, 4))
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(f'./images/fig2-{mnist_anomaly_digit}-roc.eps', format='eps')
