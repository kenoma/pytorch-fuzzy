from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from cvae import CVAE, Encoder, Decoder
from torchfuzzy.fuzzy_layer import FuzzyLayer

from utils import get_data, predict, class_scatter


device = 'cuda:0'

batch_size = 64
num_epochs = 35
learning_rate = 2e-3
weight_decay = 1e-2

latent_dim = 2
output_dims = 2

is_multi_class = True
is_fuzzy_cvae = True
is_cvae = True

beta = 1
if is_cvae:
    gamma = 1
else:
    gamma = 0


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


def train_one_epoch(model, dataloader, optimizer, prev_updates, writer=None):
    model.train()

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

            print(
                f'Step {n_upd:,} (N samples: {n_upd * batch_size:,}), Loss: {loss.item():.4f} (Recon: {loss_recon.item():.4f}, KL: {loss_kl.item():.4f} Fuzzy: {loss_fuzzy.item():.4f}) Grad: {total_norm:.4f}')

            if writer is not None:
                global_step = n_upd
                writer.add_scalar('Loss/Train', loss.item(), global_step)
                writer.add_scalar('Loss/Train/BCE', loss_recon.item(), global_step)
                writer.add_scalar('Loss/Train/KLD', loss_kl.item(), global_step)
                writer.add_scalar('Fuzzy/Train/Loss', loss_fuzzy.item(), global_step)
                writer.add_scalar('GradNorm/Train', total_norm, global_step)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

    return prev_updates + len(dataloader)


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

    print(
        f'====> Test set loss: {test_loss:.4f} (BCE: {test_recon_loss:.4f}, KLD: {test_kl_loss:.4f} Fuzzy: {test_fuzzy_loss:.4f} Accuracy {test_accuracy:.4f})')

    if writer is not None:
        writer.add_scalar('Loss/Test', test_loss, global_step=cur_step)
        writer.add_scalar('Loss/Test/BCE', loss_recon.item(), global_step=cur_step)
        writer.add_scalar('Loss/Test/KLD', loss_kl.item(), global_step=cur_step)
        writer.add_scalar('Fuzzy/Test/Loss', loss_fuzzy.item(), global_step=cur_step)
        writer.add_scalar('Fuzzy/Test/Accuracy', test_accuracy, global_step=cur_step)

        z = torch.randn(16, latent_dim).to(device)
        samples = model.decoder_pass(z)
        writer.add_images('Test/Samples', samples.view(-1, 1, 28, 28), global_step=cur_step)

    return test_accuracy


if __name__ == '__main__':
    train_data, test_data = get_data(is_multi_class, batch_size, device)
    model = CVAE(latent_dim=latent_dim, labels_count=12 if is_multi_class else 10, output_dims=output_dims, fuzzy=is_fuzzy_cvae).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if is_cvae:
        if is_fuzzy_cvae:
            prefix = 'fz'
        else:
            prefix = 'mlp'
    else:
        prefix = 'vae'
    print(f"Training {prefix}-{latent_dim}-{12 if is_multi_class else 10}")
    writer = SummaryWriter(f'runs/mnist/{prefix}_{datetime.now().strftime("%Y%m%d-%H%M%S")}')

    prev_updates = 0
    best_acc = 0
    best_acc_epoch = 0
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        prev_updates = train_one_epoch(model, train_data, optimizer, prev_updates, writer=writer)
        acc = test(model, test_data, prev_updates, writer=writer)
        if acc > best_acc:
            print(f"Updating model {acc} ==> {best_acc}")
            best_acc = acc
            best_acc_epoch = epoch
            torch.save(model, f'runs/mnist/{prefix}_{latent_dim}_{12 if is_multi_class else 10}.pt')
    print(f"Training complete. Best model at {best_acc_epoch} epoch with {best_acc} acc")
    print(f"Model is here: {f'runs/mnist/{prefix}-{latent_dim}-{12 if is_multi_class else 10}.pt'}")
