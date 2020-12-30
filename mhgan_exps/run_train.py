import numpy as np
import random
import torch, torch.nn as nn
import time
import datetime
import os
from matplotlib import pyplot as plt
import argparse
import itertools
from sklearn.preprocessing import StandardScaler
import joblib
import torchvision.datasets as dset
import torchvision.transforms as transforms
from pathlib import Path
import sys

from models import (
    Generator_2d, 
    Discriminator_2d, 
    weights_init_1, 
    weights_init_2,
    Generator_DCGAN,
    Discriminator_DCGAN)
from train import (
        JS_GAN_Trainer, 
        JS_GAN_Gaussians_Trainer,
        WGAN_Gaussians_Trainer,  
        WGAN_Trainer, 
        )
from utils import Evolution


device_default = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def prepare_25gaussian_data(batch_size=1000, sigma=0.05):
    dataset = []
    means = np.array(list(itertools.product(np.arange(-2,3), repeat=2)))

    pts_centres = []

    for i in range(batch_size // 25):
        for x in range(-2, 3):
            for y in range(-2, 3):
                point = np.random.randn(2)*sigma
                point[0] += x
                point[1] += y
                dataset.append(point)
                pts_centres.append((x, y))
    dataset = np.array(dataset, dtype=np.float32)
    np.random.shuffle(dataset)
    return dataset, means, sigma, np.array(pts_centres)


def random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def parse_arguments():
    parser = argparse.ArgumentParser('run_train.py args')
    parser.add_argument('--loss_type', type=str, choices=['jensen', 'wasserstein'], default='jensen')
    parser.add_argument('--n_critic', type=int, default=1)
    parser.add_argument('--data_type', type=str, choices=['25_gaussians', 'cifar10'], default='25_gaussians')
    parser.add_argument('--train_dataset_size', type=int, default=64000)
    parser.add_argument('--n_test_pts', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=25)
    parser.add_argument('--ndim', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--b1', type=float, default=0.5)
    parser.add_argument('--b2', type=float, default=0.9)
    parser.add_argument('--num_epochs', type=int, default=361)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--batch_size_sample', type=int, default=10000)
    parser.add_argument('--path_to_save', type=str, default='../../logs')
    parser.add_argument('--save_data', action='store_true')
    parser.add_argument('--device', type=str, default=device_default)
    parser.add_argument('--seed', type=int)

    args = parser.parse_args()

    return args


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if args.data_type == '25_gaussians':
        X_train, means, sigma, _ = prepare_25gaussian_data(args.train_dataset_size, sigma=0.05)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        trainloader = torch.utils.data.DataLoader(torch.FloatTensor(X_train), batch_size=args.batch_size) 

        n_dim = args.ndim

        G = Generator_2d(n_dim=n_dim, n_hidden=100).to(device)
        D = Discriminator_2d(n_hidden=100, top_nonlin=nn.Sigmoid() if args.loss_type == 'jensen' else None).to(device)

    elif args.data_type == 'cifar10':
        class IgnoreLabelDataset(torch.utils.data.Dataset):
            def __init__(self, orig):
                self.orig = orig

            def __getitem__(self, index):
                return self.orig[index][0]

            def __len__(self):
                return len(self.orig)

        cifar = dset.CIFAR10(root='../cifar10', download=True,
                                transform=transforms.Compose([
                                    transforms.Resize(64),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])
        )
        scaler = None

        trainloader = torch.utils.data.DataLoader(IgnoreLabelDataset(cifar), batch_size=args.batch_size)

        G = Generator_DCGAN().to(device)
        D = Discriminator_DCGAN().to(device)

    g_optimizer = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.b1, args.b2), weight_decay=1e-5)
    d_optimizer = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(args.b1, args.b2), weight_decay=1e-5)

    if args.data_type == '25_gaussians':
        if args.loss_type == 'jensen':
            trainer = JS_GAN_Gaussians_Trainer(trainloader, G, g_optimizer, D, d_optimizer, 
                            path_to_save=args.path_to_save, 
                            batch_size_sample=args.batch_size_sample,
                            true_js=False, 
                            n_critic=args.n_critic)
        elif args.loss_type == 'wasserstein':
            trainer = WGAN_Gaussians_Trainer(trainloader, G, g_optimizer, D, d_optimizer, 
                            path_to_save=args.path_to_save, 
                            batch_size_sample=args.batch_size_sample, 
                            n_critic=args.n_critic,
                            use_gradient_penalty=True)
    else:
        if args.loss_type == 'jensen':                    
            trainer = JS_GAN_Trainer(trainloader, G, g_optimizer, D, d_optimizer, path_to_save=args.path_to_save, true_js=False, n_critic=args.n_critic)
        elif args.loss_type == 'wasserstein':
            trainer = WGAN_Trainer(trainloader, G, g_optimizer, D, d_optimizer, path_to_save=args.path_to_save, n_critic=args.n_critic, use_gradient_penalty=True)

    if scaler is not None:
        joblib.dump(scaler, Path(args.path_to_save, 'std_scaler.bin'), compress=True)

    if args.save_data is True:
        if X_train is not None:
            np.savez_compressed(Path(args.path_to_save, 'x_train'), x_train=X_train)
    
    trainer.train(
                batch_size=args.batch_size,
                device=device,
                num_epochs=args.num_epochs, 
                num_epoch_for_save=args.save_every,
                batch_size_sample=args.batch_size_sample)


if __name__ == "__main__":
    args = parse_arguments()
    if args.seed is not None:
        random_seed(args.seed)
    main(args)
