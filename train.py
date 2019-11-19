import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from dataset import SkeletonDataset
from util import one_hot_embedding, create_cmx


def loss_function(args, recon_x, x, mu_qz, logvar_qz, mu_pz, logvar_pz):
    batch_size, dimensionality, time_step, joints, people = x.size()
    x = x.permute(0, 2, 4, 3, 1).contiguous().view(batch_size, time_step, people * dimensionality * joints).float()

    # Gaussian Nll
    x_mu, x_logvar = torch.chunk(recon_x.view(-1, args.time_step, args.x_dim*2), 2, dim=2)
    rec_loss = torch.sum(0.5 * (math.log(2 * math.pi) + x_logvar
                         + (x - x_mu).pow(2) * torch.exp(-x_logvar)))

    # Kullback–Leibler Divergence
    kld_loss = -0.5 * torch.sum(1 + logvar_qz - logvar_pz - (logvar_qz.exp()
               + (mu_qz - mu_pz).pow(2)) * (-logvar_pz).exp())

    rec_loss /= batch_size * time_step
    kld_loss /= batch_size * time_step
    return  rec_loss + args.beta * kld_loss, rec_loss, kld_loss


def train_dssm(args, model, device, train_loader_kwargs, test_loader_kwargs):
    train_loader = torch.utils.data.DataLoader(
        dataset=SkeletonDataset(**train_loader_kwargs),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_worker)
    test_loader = torch.utils.data.DataLoader(
        dataset=SkeletonDataset(**test_loader_kwargs),
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_worker)
    optimizer = optim.Adam(model.parameters(), lr=args.lr_dssm, weight_decay=1e-5)

    min_loss = best_epoch = float('inf')
    for epoch in range(1, args.dssm_epochs + 1):
        train_dssm_epoch(epoch, args, model, device, train_loader, optimizer)
        test_loss = test_dssm_epoch(epoch, args, model, device, test_loader)
        if test_loss <= min_loss:
            min_loss = test_loss
            best_epoch = epoch
        if epoch % 5 == 0:
            print('=====================================')
            print('Best Epoch: {} Min Loss: {:.6f}'.format(best_epoch, min_loss))
            print('=====================================')


def train_disc(args, model, device, train_loader_kwargs, test_loader_kwargs):
    train_loader = torch.utils.data.DataLoader(
        dataset=SkeletonDataset(**train_loader_kwargs),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_worker)
    test_loader = torch.utils.data.DataLoader(
        dataset=SkeletonDataset(**test_loader_kwargs),
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_worker)

    # freeze combiner weight
    for param in model.combiner.parameters():
        param.requires_grad = False
    optimizer = optim.Adam(model.discriminator.parameters(), lr=args.lr_disc, weight_decay=1e-5)

    max_acc = best_epoch = 0
    for epoch in range(1, args.disc1_epochs + 1):
        train_disc_epoch(epoch, args, model, device, train_loader, optimizer)
        test_acc, np_pred, np_true = test_disc_epoch(epoch, args, model, device, test_loader)
        if test_acc >= max_acc:
            max_acc = test_acc
            best_epoch = epoch
            np.save(args.save_path + '/pred.npy', np_pred)
            np.save(args.save_path + '/true.npy', np_true)
            torch.save(model.state_dict(), args.save_path + '/model.pth')
            create_cmx(args, np_pred, np_true)
        if epoch % 20 == 0:
            print('=====================================')
            print('Best Epoch: {} Max Acc: {:.6f}'.format(best_epoch, max_acc))
            print('=====================================')

    # fine tuning
    for param in model.parameters():
        param.requires_grad = True
    optimizer = optim.Adam(model.parameters(), lr=args.lr_disc, weight_decay=1e-5)

    max_acc = best_epoch = 0
    for epoch in range(args.disc1_epochs + 1, args.disc1_epochs + args.disc2_epochs + 1):
        train_disc_epoch(epoch, args, model, device, train_loader, optimizer)
        test_acc, np_pred, np_true = test_disc_epoch(epoch, args, model, device, test_loader)
        if test_acc >= max_acc:
            max_acc = test_acc
            best_epoch = epoch
            np.save(args.save_path + '/pred.npy', np_pred)
            np.save(args.save_path + '/true.npy', np_true)
            create_cmx(args, np_pred, np_true)
        if epoch % 20 == 0:
            print('=====================================')
            print('Best Epoch: {} Max Acc: {:.6f}'.format(best_epoch, max_acc))
            print('=====================================')

def test_disc(args, model, device, test_loader_kwargs):
    test_loader = torch.utils.data.DataLoader(
        dataset=SkeletonDataset(**test_loader_kwargs),
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_worker)

    for param in model.parameters():
        param.requires_grad = True
    optimizer = optim.Adam(model.parameters(), lr=args.lr_disc, weight_decay=1e-5)

    max_acc = best_epoch = 0
    test_acc, _, _ = test_disc_epoch(0, args, model, device, test_loader)
    print('=====================================')
    print('Max Acc: {:.6f}'.format(test_acc))
    print('=====================================')

def train_dssm_epoch(epoch, args, model, device, train_loader, optimizer):
    print('DSSM Train Epoch: {}'.format(epoch))
    model.train()

    reckld_loss = rec_loss = kld_loss = 0
    train_loss = 0
    for step, (data, target) in enumerate(train_loader):
        target = one_hot_embedding(target, args.class_dim) # (batch_size, class_dim)
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        recon_x, qz, mu_qz, logvar_qz, pz, mu_pz, logvar_pz = model(
            data.float(), discriminate=False, train=True)

        loss = loss_function(args, recon_x, data, mu_qz, logvar_qz, mu_pz, logvar_pz)

        reckld_loss += loss[0].item() / len(train_loader.dataset)
        rec_loss += loss[1].item() / len(train_loader.dataset)
        kld_loss += loss[2].item() / len(train_loader.dataset)

        loss[0].backward()

        optimizer.step()

    print('Train Loss: {:.4f}  REC Loss: {:.4f}  KLD Loss: {:.4f}'.format(reckld_loss, rec_loss, kld_loss))


def test_dssm_epoch(epoch, args, model, device, test_loader):
    print('DSSM Test  Epoch: {}'.format(epoch))
    model.eval()

    reckld_loss = rec_loss = kld_loss = 0
    train_loss = 0
    for step, (data, target) in enumerate(test_loader):
        target = one_hot_embedding(target, args.class_dim) # (batch_size, class_dim)
        data, target = data.to(device), target.to(device)

        recon_x, qz, mu_qz, logvar_qz, pz, mu_pz, logvar_pz = model(
            data.float(), discriminate=False, train=False)

        loss = loss_function(args, recon_x, data, mu_qz, logvar_qz, mu_pz, logvar_pz)

        reckld_loss += loss[0].item() / len(test_loader.dataset)
        rec_loss += loss[1].item() / len(test_loader.dataset)
        kld_loss += loss[2].item() / len(test_loader.dataset)

    print('Test Loss: {:.4f}  REC Loss: {:.4f}  KLD Loss: {:.4f}'.format(reckld_loss, rec_loss, kld_loss))
    return reckld_loss


def train_disc_epoch(epoch, args, model, device, train_loader, optimizer):
    print('Discriminator Train Epoch: {}'.format(epoch))
    model.train()

    train_loss = class_accuracy = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(data.float(), discriminate=True, train=True)

        loss = F.nll_loss(output, target)

        pred = output.argmax(dim=1, keepdim=True)
        class_accuracy += pred.eq(target.view_as(pred)).sum().item()

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader.dataset)
    class_accuracy /= len(train_loader.dataset)

    print('Train Loss: {:.6f} Train Accuracy: {:.4f}'.format(train_loss, class_accuracy))


def test_disc_epoch(epoch, args, model, device, test_loader):
    print('Discriminator Test  Epoch: {}'.format(epoch))
    model.eval()

    test_loss = class_accuracy = 0
    np_pred = np_true = np.array([])
    with torch.no_grad():
        for data, target in test_loader:
            batch_size, dimensionality, time_step, joints, people = data.size()

            data, target = data.to(device), target.to(device)

            output = model(data.float(), discriminate=True, train=False)

            test_loss += F.nll_loss(output, target.to(device), reduction='sum').item()

            pred = output.argmax(dim=1, keepdim=True)
            class_accuracy += pred.eq(target.to(device).view_as(pred)).sum().item()

            np_pred = np.append(np_pred, pred.view(-1).cpu().detach().numpy())
            np_true = np.append(np_true, target.view(-1).cpu().detach().numpy())

    test_loss /= len(test_loader.dataset)
    class_accuracy /= len(test_loader.dataset)

    print('Test Loss: {:.4f} Test Accuracy: {:.4f}'.format(test_loss, class_accuracy))
    return class_accuracy, np_pred, np_true
