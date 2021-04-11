import torch
from torch import nn
import numpy as np
import argparse
import json
from pathlib import Path
from functools import partial
from easydict import EasyDict as edict
from matplotlib import pyplot as plt
import re
from sklearn.utils import shuffle
import torchvision.utils as vutils

from tools.sampling_utils.ebm_sampling import (
    gan_energy,
    IndependentNormal,
    langevin_dynamics,
    langevin_sampling,
    #mala_dynamics,
    mala_dynamics,
    mala_sampling,
    #citerais_ula_dynamics,
    citerais_ula_sampling,
    citerais_mala_sampling,
    sampling_f
)
from tools.sampling_utils.classification import LogisticOverLogits
#from tools.sampling_utils.visualization import plot_chain_metrics
#from tools.sampling_utils.metrics import Evolution
#from tools.sampling_utils.distributions import Gaussian_mixture
# from tools.toy_examples_utils.toy_examples_utils import \
#     prepare_2d_ring_data, prepare_dataloader
from tools.stacked_mnist_utils.dcgan import Generator, Discriminator
from tools.stacked_mnist_utils.data_utils import StackedMNIST
from utils import random_seed, load_dict_for_sampling#, plot_gen_dist
from clf_mnist import Classifier

name_to_sampling_f = {'ula': langevin_sampling,
                    'mala_sampling': mala_sampling, 
                    'citerais_ula': citerais_ula_sampling,
                    'citerais_mala': citerais_mala_sampling,}


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gan_data_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    # parser.add_argument('--models_dir', type=str, required=True)
    parser.add_argument('--model_idx', type=int, nargs='+', default=None)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--save_prefix', type=str, default='')
    parser.add_argument('--sampler', type=str, nargs='+', choices=[
        'ula', 'mala', 'citerais_ula', 'citerais_mala'
        ], default=['ula'])
    parser.add_argument('--n_steps', type=int, default=300)
    parser.add_argument('--invoke_every', type=int, default=10)
    parser.add_argument('--grad_step', type=float, default=1e-3) # 1e-3
    parser.add_argument('--eps_scale', type=float, default=1e-4) #(2e-4)**.5)#1e-4)
    parser.add_argument('-T', '--T', dest='T', type=int, default=100)
    parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, default=100)
    parser.add_argument('-N', '--N', dest='N', type=int, default=6)
    parser.add_argument('--pbern', type=float, default=1.0)
    parser.add_argument('--rho', type=float, default=0.95)
    #
    parser.add_argument('--beta_deg', type=float, default=0.5)
    parser.add_argument('--n_save', type=int)
    parser.add_argument('--mala_afterall', action='store_true')
    parser.add_argument('--calibrate', action='store_true')
    parser.add_argument('--burn_in', type=float, default=0.2)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', type=int, default=None)
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--clf_path', type=str, required=False)

    args = parser.parse_args()
    return args


def main(args):
    if args.device is None:
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if args.seed is not None:
        random_seed(args.seed)

    args.save_dir = Path(args.gan_data_dir, 'mcmc_sampling')
    args.save_dir.mkdir(exist_ok=True)
    if len(args.save_prefix) > 0:
        args.save_prefix = f'{args.save_prefix}_'
    
    ngpu = int(args.ngpu)

    models_dir = Path(args.gan_data_dir, 'models')
    assert models_dir.exists()
    gen_params = json.load(Path(models_dir, 'generator_params.json').open('r'))
    G = Generator(ngpu, **gen_params)
    discr_params = json.load(Path(models_dir, 'discriminator_params.json').open('r'))
    D = Discriminator(ngpu, **discr_params)

    path = Path(args.data_dir)
    assert path.exists()
    dataset = StackedMNIST(path)
    dataset.build()

    clf = None
    if args.clf_path is not None:
        clf_path = Path(args.clf_path)
        assert clf_path.exists()
        clf = Classifier(args.ngpu).to(args.device)
        clf.load_state_dict(torch.load(clf_path))
        for p in clf.parameters():
            p.requires_grad = False
    
    args.model_idx = [-1] if args.model_idx is None else args.model_idx
    for idx in args.model_idx:
        def glob_sorted(model):
            glob = list(models_dir.glob(f'*{model}*')) #(f"*_{model}.pth"))
            #print(glob)
            models = list(filter(lambda x: 'opt' not in x.parts[-1], glob))
            paths = sorted(models, key=lambda x: float(re.findall(r'\d+', str(x.parts[-1]))[-1]))
            return paths
        discriminator_path = glob_sorted('netD')[idx] #discriminator')[idx]
        generator_path = glob_sorted('netG')[idx] #generator')[idx]
        print(f'Generator file: {generator_path.parts[-1]}')

        load_dict_for_sampling(G, generator_path, args.device)
        load_dict_for_sampling(D, discriminator_path, args.device)

        if args.calibrate:
            real_sample = torch.stack([dataset[i][0] for i in np.random.choice(len(dataset), int(0.1 * len(dataset)))], 0)
            #real_sample = X_train[np.random.choice(X_train.shape[0], int(0.1 * X_train.shape[0]), replace=False)]

            noise = torch.randn(int(0.1 * len(dataset)), 64, 1, 1, device=args.device)
            fake_sample = G(noise) #int(0.1 * len(dataset)))#.detach().cpu().numpy()
            #print(fake_sample.shape, real_sample.shape)
            calib_sample = torch.cat([real_sample, fake_sample], 0)

            y_true = np.concatenate([np.ones(real_sample.shape[0]), np.zeros(fake_sample.shape[0])], axis=0)
            y_pred = D(calib_sample.to(args.device)).detach().cpu().numpy()
            #y_pred = np.concatenate([y_pred, np.ones(y_pred.shape)], 1)
            
            calibrator = LogisticOverLogits()
            calibrator.fit(y_pred, y_true)
            w = calibrator.clf.coef_
            print(f'Calibration layer: {w}')
            grad_step = args.grad_step * w[0, 0]

        else:
            grad_step = args.grad_step

        n_dim = 64 #G.n_dim
        loc = torch.zeros(n_dim).to(args.device)
        scale = torch.ones(n_dim).to(args.device)
        normalize_to_0_1 = True 
        log_prob = True

        proposal_args = edict()
        proposal_args.device = args.device
        proposal_args.loc = loc
        proposal_args.scale = scale
        proposal = IndependentNormal(proposal_args)

        target_gan = partial(gan_energy,
                            generator = G, 
                            discriminator = D, 
                            proposal = proposal,
                            normalize_to_0_1 = normalize_to_0_1,
                            log_prob = log_prob)


        if clf is not None:
            Xs_gen = G(torch.randn(args.batch_size*args.n_steps*args.T, 64, 1, 1).to(args.device)).detach()
            labels  = []
            fake_data =  Xs_gen.reshape(-1, 3, 28, 28).permute(1, 0, 2, 3)
            for x in fake_data:
                out = clf(x.unsqueeze(1))
                labels.append(out.argmax(1).detach())
            labels = torch.stack(labels, 0)
            nmodes = torch.unique(labels, dim=1).shape[1]

            print(f'Found {nmodes} modes out of 1000')

        #evols = dict()
        for method in args.sampler:
            #evolution = Evolution(target_sample, locs=locs, target_log_prob=true_target, sigma=sigma, scaler=scaler)
            kwargs = edict(batch_size=args.batch_size,
                                n=args.batch_size,
                                grad_step=grad_step,
                                eps_scale=args.eps_scale,)
            if method.startswith('citerais'):
                kwargs.betas = np.linspace(1.0, 0.0, args.T)**args.beta_deg
                kwargs.rhos = [args.rho]*args.T  #
                kwargs.pbern = args.pbern
                kwargs.N = args.N
                kwargs.n_steps = args.n_steps
                kwargs.n_save = args.n_save

                every = args.invoke_every

            elif method == 'ula' or method =='mala':
                kwargs.n_steps = args.T * args.n_steps

                every = args.invoke_every * args.T
            #                        acceptance_rule='Hastings')
            
            _, zs = name_to_sampling_f[method](target_gan, proposal, **kwargs)

            if method == 'citerais_ula' and args.mala_afterall:
                l = len(zs[0]) 
                zs = zs.reshape(-1, zs.shape[-1])
                zs, _ = mala_dynamics(torch.from_numpy(zs).to(args.device), target_gan, proposal, kwargs.n_steps, grad_step, kwargs.eps_scale)
                zs = zs[-1].data.cpu().numpy()
                #zs = np.stack([o.data.cpu().numpy() for o in zs], axis=0)
                zs = zs.reshape(1, l, -1, zs.shape[-1])

            #zs = zs[0, ::every]
            zs = zs[0, :(zs.shape[1] // every) * every, :].reshape(-1, every, zs.shape[-2], zs.shape[-1])\
                .reshape(-1, every*zs.shape[-2], zs.shape[-1])

            print(zs.shape)

            Xs_gen = G(torch.FloatTensor(zs[-1], device=args.device))#.detach().cpu().numpy()
            # if scaler is not None:
            #     Xs_gen = scaler.inverse_transform(Xs_gen.reshape(-1, Xs_gen.shape[-1])).reshape(Xs_gen.shape)
            #Xs_gen = dataset.back_normalize(Xs_gen)

            # for X_gen in Xs_gen:
            #     evolution.invoke(torch.FloatTensor(X_gen))
            # evol = evolution.as_dict()
            # evols[f'{method}'] = evol
            # json.dump(evol, Path(args.save_dir, f'{idx}_{args.save_prefix}{method}_evol.json').open('w'))

            # print('Estimate density...')
            # save_path = Path(args.save_dir, f'{idx}_{args.save_prefix}{method}.pdf')
            # plot_gen_dist(Xs_gen[int(args.burn_in * Xs_gen.shape[0]):].reshape(-1, locs.shape[1])[-500:], x_range, y_range)
            # plt.savefig(save_path)
            # plt.close()
            vutils.save_image(Xs_gen.reshape(-1, 1, 28, 28)[-100:].detach().cpu(),
                        Path(args.save_dir, f'{idx}_{args.save_prefix}{method}.png'),
                        normalize=False)

            if clf is not None:
                labels  = []
                fake_data =  Xs_gen.reshape(-1, 3, 28, 28).permute(1, 0, 2, 3)
                for x in fake_data:
                    out = clf(x.unsqueeze(1))
                    labels.append(out.argmax(1).detach())
                labels = torch.stack(labels, 0)
                nmodes = torch.unique(labels, dim=1).shape[1]

                print(f'Found {nmodes} modes out of 1000')
                

        # save_path = Path(args.save_dir, f'{idx}_{args.save_prefix}metrics.pdf')
        # plot_chain_metrics(every=every, savepath=save_path, sigma=sigma, **evols)
        # plt.close()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
