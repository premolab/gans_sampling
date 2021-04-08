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
from tools.sampling_utils.visualization import plot_chain_metrics
from tools.sampling_utils.metrics import Evolution
from tools.sampling_utils.distributions import Gaussian_mixture
# from tools.toy_examples_utils.toy_examples_utils import \
#     prepare_2d_ring_data, prepare_dataloader
from tools.toy_examples_utils.gan_fc_models import Generator_fc, Discriminator_fc
from utils import random_seed, load_dict_for_sampling, plot_gen_dist

name_to_sampling_f = {'ula': langevin_sampling,
                    'mala_sampling': mala_sampling, 
                    'citerais_ula': citerais_ula_sampling,
                    'citerais_mala': citerais_mala_sampling,}


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gan_data_dir', type=str, required=True)
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

    args = parser.parse_args()
    return args


def main(args):
    if args.device is None:
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if args.seed is not None:
        random_seed(args.seed)

    # if args.save_dir is None:
    #     args.save_dir = args.models_dir
    args.save_dir = Path(args.gan_data_dir, 'mcmc_sampling')
    args.save_dir.mkdir(exist_ok=True)
    if len(args.save_prefix) > 0:
        args.save_prefix = f'{args.save_prefix}_'
    
    models_dir = Path(args.gan_data_dir, 'models')
    assert models_dir.exists()
    gen_params = json.load(Path(models_dir, 'generator_params.json').open('r'))
    G = Generator_fc(**gen_params, non_linear=nn.ReLU(), device=args.device)
    discr_params = json.load(Path(models_dir, 'discriminator_params.json').open('r'))
    D = Discriminator_fc(**discr_params, non_linear=nn.ReLU(), device=args.device)

    tar_data = np.load(Path(args.gan_data_dir, 'data.npz'))
    X_train = tar_data['X_train']
    locs = tar_data['locs']
    sigma = tar_data['sigma']
    x_range = tar_data['x_range']
    y_range = tar_data['y_range']
    if False: #'scaler' in tar_data.keys(): #can't save scaler in npz format, only params 
        scaler = tar_data['scaler']
    else:
        scaler = None
    target_sample = X_train[np.random.choice(np.arange(X_train.shape[0]), 1000)]
    
    target_args = edict()
    target_args.device = args.device
    target_args.num_gauss = locs.shape[0]
    target_args.p_gaussians = [torch.tensor(1./target_args.num_gauss)]*target_args.num_gauss
    target_args.locs = torch.from_numpy(locs).float()
    target_args.covs = [(sigma**2)*torch.eye(locs.shape[1]).to(args.device)]*target_args.num_gauss
    target_args.dim = locs.shape[1]
    true_target = Gaussian_mixture(target_args).log_prob
    
    args.model_idx = [-1] if args.model_idx is None else args.model_idx
    for idx in args.model_idx:
        def glob_sorted(model):
            glob = list(models_dir.glob(f"*_{model}.pth"))
            models = filter(lambda x: 'opt' not in x.parts[-1], glob)
            paths = sorted(models, key=lambda x: float(re.match(r'\d+', str(x.parts[-1])).group(0)))
            return paths
        discriminator_path = glob_sorted('discriminator')[idx]
        generator_path = glob_sorted('generator')[idx]
        print(f'Generator file: {generator_path.parts[-1]}')

        # if args.calibrate:
        #     D.calib_layer = None

        load_dict_for_sampling(G, generator_path, args.device)
        load_dict_for_sampling(D, discriminator_path, args.device)

        if args.calibrate:
            real_sample = X_train[np.random.choice(X_train.shape[0], int(0.1 * X_train.shape[0]), replace=False)]
            fake_sample = G.sampling(int(0.1 * X_train.shape[0])).detach().cpu().numpy()
            calib_sample = np.concatenate([real_sample, fake_sample], 0)

            y_true = np.concatenate([np.ones(real_sample.shape[0]), np.zeros(fake_sample.shape[0])], axis=0)
            y_pred = D(torch.from_numpy(calib_sample).float().to(args.device)).detach().cpu().numpy()
            #y_pred = np.concatenate([y_pred, np.ones(y_pred.shape)], 1)
            
            calibrator = LogisticOverLogits()
            calibrator.fit(y_pred, y_true)
            w = calibrator.clf.coef_
            print(f'Calibration layer: {w}')
            grad_step = args.grad_step * w[0, 0]
            # D.calib_layer = nn.Linear(1, 1, bias=False)
            # D.calib_layer.weight = torch.nn.Parameter(torch.from_numpy(w[:, 0].reshape(1, 1)).float().to(args.device))
            # D.calib_layer.weight.requires_grad = False
            #D.calib_layer.bias.data =  torch.nn.Parameter(torch.from_numpy(w[:, 1]).float().to(args.device))
        else:
            grad_step = args.grad_step

        n_dim = G.n_dim
        loc = torch.zeros(n_dim).to(G.device)
        scale = torch.ones(n_dim).to(G.device)
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

        evols = dict()
        for method in args.sampler:
            evolution = Evolution(target_sample, locs=locs, target_log_prob=true_target, sigma=sigma, scaler=scaler)
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

            Xs_gen = G(torch.FloatTensor(zs, device=args.device)).detach().cpu().numpy()
            if scaler is not None:
                Xs_gen = scaler.inverse_transform(Xs_gen.reshape(-1, Xs_gen.shape[-1])).reshape(Xs_gen.shape)

            for X_gen in Xs_gen:
                evolution.invoke(torch.FloatTensor(X_gen))
            evol = evolution.as_dict()
            evols[f'{method}'] = evol
            json.dump(evol, Path(args.save_dir, f'{idx}_{args.save_prefix}{method}_evol.json').open('w'))

            print('Estimate density...')
            save_path = Path(args.save_dir, f'{idx}_{args.save_prefix}{method}.pdf')
            plot_gen_dist(Xs_gen[int(args.burn_in * Xs_gen.shape[0]):].reshape(-1, locs.shape[1])[-1500:], x_range, y_range)
            plt.savefig(save_path)
            plt.close()

        save_path = Path(args.save_dir, f'{idx}_{args.save_prefix}metrics.pdf')
        plot_chain_metrics(every=every, savepath=save_path, sigma=sigma, **evols)
        plt.close()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
