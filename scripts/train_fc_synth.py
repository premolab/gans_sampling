import numpy as np
import sklearn.datasets
from sklearn.preprocessing import StandardScaler
import random
import datetime
from pathlib import Path
import json
from matplotlib import pyplot as plt
import argparse

import torch, torch.nn as nn

# from paths import (path_to_save_remote, 
#                    #path_to_save_local,
#                    port_to_remote) 

from tools.toy_examples_utils.toy_examples_utils import (prepare_2d_ring_data,
                                prepare_25gaussian_data,
                                prepare_swissroll_data,
                                prepare_gaussians,
                                # prepare_train_batches, 
                                prepare_dataloader, 
                                logging)
from tools.toy_examples_utils.gan_fc_models import (Generator_fc, 
                           Discriminator_fc, 
                           weights_init_1, 
                           weights_init_2)
from tools.toy_examples_utils.gan_train import train_gan
from scripts.utils import random_seed, DUMP_DIR


def get_fc_gan_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_dim', '--n_dim', dest='n_dim', type=int, default=256) #params_module.n_dim)
    parser.add_argument('--n_layers_g', type=int, default=2) #params_module.n_layers_g)
    parser.add_argument('--n_hid_g', type=int, default=128) #params_module.n_hid_g)
    parser.add_argument('--n_out', type=int, default=2) #params_module.n_out)

    parser.add_argument('--n_layers_d', type=int, default=1) #params_module.n_layers_d)
    parser.add_argument('--n_hid_d', type=int, default=128) #params_module.n_hid_d)

    parser.add_argument('--disc_lr', type=float, default=1e-4)
    parser.add_argument('--gen_lr', type=float, default=1e-4)
    parser.add_argument('--betas', type=float, nargs='+', default=(0.5, 0.999))

    parser.add_argument('--k_d', type=int, default=10)
    parser.add_argument('--k_g', type=int, default=1)
    parser.add_argument('--loss_type', type=str, choices=['Jensen_minimax', 'Jensen_nonsaturating'], default='Jensen_minimax')
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--Lambda', type=float, default=0.)
    parser.add_argument('--gp', action='store_true')
    parser.add_argument('--num_epoch_for_save', type=int, default=50)

    return parser


def parse_arguments(parser):
    parser.add_argument('--data_type', type=str, choices=['2d_ring', '2d_grid', '5d_grid', '1200d', 'swissroll'], default='2d_ring')
    parser.add_argument('--train_dataset_size', type=int, default=512*100)
    parser.add_argument('--sigma', type=float, default=0.02)
    parser.add_argument('--rad', type=float, default=2.)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--batch_size_sample', type=int, default=2500)
    # parser.add_argument('--std_scale', action='store_true')
    # parser.add_argument('--scale_discriminator ')
    parser.add_argument('--save_name', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', type=int, default=None)

    args = parser.parse_args()
    return args


def main(args):
    if args.seed is not None:
        random_seed(args.seed)

    if args.device is None:
            args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if args.data_type == '2d_ring':
        X_train, means = prepare_2d_ring_data(args.train_dataset_size,
                                                args.sigma, 
                                                args.seed)
    elif args.data_type == '2d_grid':
        args.rad = 2
        X_train, means = prepare_25gaussian_data(args.train_dataset_size,
                                                args.sigma, 
                                                args.seed)
    elif args.data_type == 'swissroll':
        X_train = prepare_swissroll_data(args.train_dataset_size)
        means = None
    # if args.std_scale:
    #     scaler = StandardScaler()
    #     X_train_peprocessed = scaler.fit_transform(X_train)
    #     #X_train_batches = prepare_train_batches(X_train, args.batch_size)
    # else:
    #     scaler = None
    #     X_train_peprocessed = X_train.copy()
    scaler = StandardScaler()
    scaler.fit(X_train)
    scale = scaler.scale_[0]

    # if args.data_type == '2d_ring':
    #     scale = 2**.
    print(f'Scale: {scale:.4f}')
    scaler = None
    X_train_peprocessed = X_train.copy()

    train_dataloader = prepare_dataloader(X_train_peprocessed, args.batch_size, 
                                        random_seed=args.seed)

    generator_params = dict(n_dim=args.n_dim, n_layers=args.n_layers_g, n_hid=args.n_hid_g,
        n_out=args.n_out)
    G = Generator_fc(**generator_params,
                    non_linear=nn.ReLU(),
                    device=args.device).to(args.device)

    discriminator_params = dict(n_in=X_train.shape[1], n_layers=args.n_layers_d, 
        n_hid=args.n_hid_d, scale=scale)
    D = Discriminator_fc(**discriminator_params,
                        non_linear=nn.ReLU(),
                        device=args.device).to(args.device)
    print(D)

    # for n, p in G.named_parameters():
    #     if 'weight' in n:
    #         torch.nn.init.orthogonal_(p, gain=0.8)
        # elif 'bias' in n:
        #     torch.nn.init.zeros_(p)

    # weights_init_2(D)

    d_optimizer = torch.optim.Adam(D.parameters(), 
                                betas=args.betas, 
                                lr=args.disc_lr)
    g_optimizer = torch.optim.Adam(G.parameters(), 
                                betas=args.betas, 
                                lr=args.gen_lr)

    cur_time = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
    path_to_save_local =  Path(DUMP_DIR, args.data_type)
    new_dir = Path(path_to_save_local, f'{cur_time}_{args.save_name}')
    new_dir.mkdir(parents=True)

    np.savez(str(Path(new_dir, 'data')), 
        X_train=X_train, 
        locs=means,
        scaler=scaler, 
        sigma=args.sigma, 
        x_range=(-args.rad*1.25, args.rad*1.25),
        y_range=(-args.rad*1.25, args.rad*1.25),)

    path_to_plots = Path(new_dir, 'plots')
    path_to_models = Path(new_dir, 'models')
    path_to_plots.mkdir()
    path_to_models.mkdir()
    json.dump(generator_params, Path(path_to_models, 'generator_params.json').open('w'))
    json.dump(discriminator_params, Path(path_to_models, 'discriminator_params.json').open('w'))

    path_to_logs = Path(new_dir, 'logs.txt')
    
    logging(path_to_logs = path_to_logs,
            mode = args.data_type, 
            train_dataset_size = args.train_dataset_size, 
            batch_size = args.batch_size, 
            n_dim = args.n_dim, 
            n_layers_g = args.n_layers_g, 
            n_layers_d = args.n_layers_d, 
            n_hid_g = args.n_hid_g, 
            n_hid_d = args.n_hid_d, 
            n_out = args.n_out, 
            loss_type = args.loss_type, 
            lr_init = args.gen_lr, 
            Lambda = args.Lambda, 
            num_epochs = args.n_epochs, 
            k_g = args.k_g, 
            k_d = args.k_d)

    print("Start to train GAN")
    train_gan(X_train=X_train,
            train_dataloader=train_dataloader, 
            generator=G, 
            g_optimizer=g_optimizer, 
            discriminator=D, 
            d_optimizer=d_optimizer,
            loss_type=args.loss_type,
            batch_size=args.batch_size,
            device=args.device,
            use_gradient_penalty=args.gp,
            Lambda=args.Lambda,
            num_epochs=args.n_epochs, 
            num_epoch_for_save=args.num_epoch_for_save,
            batch_size_sample=args.batch_size_sample,
            k_g=args.k_g,
            k_d=args.k_d,
            n_calib_pts=0, #n_calib_pts,
            normalize_to_0_1=True, #args.loss_type.startswith('Jensen'), #normalize_to_0_1, 
            scaler=scaler,
            mode=args.data_type, 
            path_to_logs=path_to_logs,
            path_to_models=path_to_models,
            path_to_plots=path_to_plots,
            path_to_save_remote=None, #path_to_save_remote,
            port_to_remote=None, #port_to_remote,
            plot_mhgan=False) #plot_mhgan)


if __name__ == '__main__':
    parser = get_fc_gan_parser()
    args = parse_arguments(parser)
    main(args)
