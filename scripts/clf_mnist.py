import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from pathlib import Path
import json
import argparse
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms as T
from torch.optim import Adam
from tqdm import trange

from utils import random_seed, DUMP_DIR


class Classifier(nn.Module):
    def __init__(self, ngpu, nc=1, ndf=16):
        super().__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
                
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Flatten(),

                nn.Dropout(0.3),
                nn.Linear(1024, 128),
                nn.ReLU(),
                nn.Linear(128, 10)
            )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=None)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--n_epoch', type=int, default=10)
    parser.add_argument('--save_dir', type=str)

    args = parser.parse_args()
    return args


def main(args):
    if args.device is None:
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if args.seed is not None:
        random_seed(args.seed)

    data_dir = Path(args.data_dir)
    assert data_dir.exists()

    train_dataset = MNIST(data_dir, train=True, transform=T.Compose([
        T.ToTensor(), T.Normalize((0.1307,), (0.3081,))
        ]))
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataset = MNIST(data_dir, train=False, transform=T.Compose([
        T.ToTensor(), T.Normalize((0.1307,), (0.3081,))
        ]))
    validloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    model = Classifier(args.ngpu).to(args.device)
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss().to(args.device)

    for epoch in trange(args.n_epoch):
        model.train()
        acc = 0
        n_ex = 0
        for x, y in trainloader:
            x = x.to(args.device)
            y = y.to(args.device)

            model.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            acc += (out.argmax(-1) == y).sum().item()
            n_ex += y.shape[0]

        acc /= n_ex

        print(f'Epoch: {epoch}/{args.n_epoch}. Train loss: {loss.item()}. Train acc: {acc}')

        with torch.no_grad():
            model.eval()
            acc = 0
            n_ex = 0
            for x, y in validloader:
                x = x.to(args.device)
                y = y.to(args.device)

                out = model(x)
                loss = criterion(out, y)

                acc += (out.argmax(-1) == y).sum().item()
                n_ex += y.shape[0]

            acc /= n_ex

            print(f'Epoch: {epoch}/{args.n_epoch}. Valid loss: {loss.item()}. Valid acc: {acc}')

    torch.save(model.state_dict(), Path(args.save_dir, 'mnist_clf.pth'))


if __name__ == '__main__':
    args = parse_arguments()
    main(args)