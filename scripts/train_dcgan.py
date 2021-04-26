import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.utils import data
from torch.utils.data import DataLoader
from torch import optim
from pathlib import Path
import argparse
import datetime
import torchvision.utils as vutils
from easydict import EasyDict as edict
import json

from utils import random_seed, DUMP_DIR
from tools.stacked_mnist_utils.data_utils import StackedMNIST
from tools.stacked_mnist_utils.dcgan import Generator, Discriminator


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_epoch', type=int, default=150)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--n_d', type=int, default=1)
    parser.add_argument('--device', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--nz', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--save_name', type=str)

    args = parser.parse_args()
    return args
    

def main(args):
    if args.seed is not None:
        random_seed(args.seed)

    if args.device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    path = Path(args.data_dir)
    assert path.exists()
    dataset = StackedMNIST(path)
    dataset.build()

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    #cudnn.benchmark = True
    nc = 3
    ngpu = int(args.ngpu)
    nz = int(args.nz)
    ngf = int(args.ngf)
    ndf = int(args.ndf)

    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    generator_params = dict(nc=nc, nz=nz, ngf=ngf)
    netG = Generator(ngpu, **generator_params).to(device)
    netG.apply(weights_init)
    # if opt.netG != '':
    #     netG.load_state_dict(torch.load(args.netG))
    #print(netG)
 
    discriminator_params = dict(nc=nc, ndf=ndf)
    netD = Discriminator(ngpu, **discriminator_params).to(device)
    netD.apply(weights_init)
    # if opt.netD != '':
    #     netD.load_state_dict(torch.load(args.netD))
    # print(netD)

    criterion = nn.BCEWithLogitsLoss()

    fixed_noise = torch.randn(args.batch_size, nz, 1, 1, device=device)
    real_label = 1
    fake_label = 0

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    cur_time = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
    path_to_save_local =  Path(DUMP_DIR, 'StackedMNIST')
    new_dir = Path(path_to_save_local, f'{cur_time}_{args.save_name}')
    new_dir.mkdir(parents=True)

    path_to_plots = Path(new_dir, 'plots')
    path_to_models = Path(new_dir, 'models')
    path_to_plots.mkdir()
    path_to_models.mkdir()
    json.dump(generator_params, Path(path_to_models, 'generator_params.json').open('w'))
    json.dump(discriminator_params, Path(path_to_models, 'discriminator_params.json').open('w'))

    netD.train()
    netG.train()
    for epoch in range(args.n_epoch):
        for i, data in enumerate(dataloader, 0):
            netG.train()
            netD.train()

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            real_cpu = data[0].to(device)
            batch_size = real_cpu.size(0)
            #noise = torch.randn(batch_size, nz, 1, 1, device=device)
            for _ in range(args.n_d):
                netD.zero_grad()
                
                #label = torch.full((batch_size,), real_label, device=device)
                label = torch.ones((batch_size,)).to(device)

                output = netD(real_cpu)
                errD_real = criterion(output, label)
                errD_real.backward()
                D_x = output.mean().item()

                # train with fake
                noise = torch.randn(batch_size, nz, 1, 1, device=device)
                fake = netG(noise)
                label.fill_(fake_label)
                output = netD(fake.detach())
                errD_fake = criterion(output, label)
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                errD = errD_real + errD_fake
                optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            if i % 100 == 0:
                print(f'Epoch {epoch}/{args.n_epoch}. Iter {i}/{len(dataloader)}. Loss_G {errG.item()}, Loss_D {errD.item()}')

                r = real_cpu
                #r = dataset.back_normalize(real_cpu)
                vutils.save_image(r,
                        Path(path_to_plots, f'real_samples.png'), 
                        normalize=False)
                netG.eval()
                fake = netG(fixed_noise)
                f = fake
                #f = dataset.back_normalize(fake)
                f = f.reshape(-1, 1, 28, 28)
                vutils.save_image(f.detach(),
                        Path(path_to_plots, f'fake_samples_epoch_{epoch}.png'),
                        normalize=False)

        torch.save(netG.state_dict(), Path(path_to_models, f'netG_epoch_{epoch}.pth'))
        torch.save(netD.state_dict(), Path(path_to_models, f'netD_epoch_{epoch}.pth'))


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
