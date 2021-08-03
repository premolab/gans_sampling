import numpy as np
from tqdm import tqdm
from typing import Tuple
from pyro.infer import MCMC, NUTS, HMC
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from easydict import EasyDict as edict
import jax
import ot
from collections import defaultdict
from functools import partial
import seaborn as sns
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
import argparse
import yaml
import pickle
import datetime

from iterative_sir.sampling_utils.adaptive_mc import CISIR, FlowMCMC, Ex2MCMC
from iterative_sir.sampling_utils.adaptive_sir_loss import *
from iterative_sir.sampling_utils.ebm_sampling import MALA
from iterative_sir.sampling_utils.flows import RNVP
from iterative_sir.sampling_utils.distributions import (Distribution,
                           IndependentNormal,
                           Funnel, Banana, HalfBanana, CauchyMixture)

from iterative_sir.sampling_utils.total_variation import average_total_variation
from iterative_sir.sampling_utils.metrics import ESS, acl_spectrum

from utils import DotConfig

sns.set_theme(style="ticks", palette="deep")


def plot_learned_density(flow, proposal, fig=None, device='cpu', xlim=[-1, 1], ylim=[-1, 1], rest=0.):
    if fig is None:
        fig = plt.figure()

    z = proposal.sample((10000,))
    x = np.linspace(*xlim, 100)
    y = np.linspace(*ylim, 100)
    xx, yy = np.meshgrid(x, y)
    zz = np.stack([xx, yy], -1)
    z[:, :2] = torch.FloatTensor(zz).view(-1, 2).to(device)
    z[:, 2:] = torch.FloatTensor([rest]*10000)[:, None]

    inv, minus_log_jac = flow.inverse(z)
    minus_log_jac = minus_log_jac.reshape(100, 100)
    inv = inv.reshape(100, 100, -1)
    vals = (proposal(inv) + minus_log_jac).exp().detach()

    plt.contourf(xx, yy, vals.reshape(100, 100), cmap='GnBu')
    return fig


def plot_hist(samples, title = 'Histogram', bins=200, 
              gamma=0.5, dims=[0, 1], ax_lims = [(-2,9), (-2,4)],
              save_path = 'pics/histogram_test.pdf', density = True):
    plt.close()
    plt.figure(figsize=(5, 5), dpi=300)
    plt.hist2d(samples[:,dims[0]], samples[:,dims[1]],bins=bins, density=density, range=[ax_lims[0], ax_lims[1]],norm=mcolors.PowerNorm(gamma))
    plt.title(title)
    plt.axis('off')
    plt.xlim((ax_lims[0][0], ax_lims[0][1]))
    plt.ylim((ax_lims[1][0], ax_lims[1][1]))
    plt.savefig(save_path, bbox_inches='tight')
    plt.show() 


def compute_metrics(xs_true, xs_pred, name=None, n_samples=1000, scale=1., trunc_chain_len=None, ess_rar=1):
    metrics = dict()
    key = jax.random.PRNGKey(0)
    n_steps = 10
    # n_samples = 100

    ess = ESS(acl_spectrum((xs_pred[::ess_rar] - xs_pred[::ess_rar].mean(0)[None, ...]))).mean()
    metrics['ess'] = ess

    xs_pred = xs_pred[-trunc_chain_len:]

    tracker = average_total_variation(key, xs_true, xs_pred, n_steps=n_steps, n_samples=n_samples)

    metrics['tv_mean'] = tracker.mean()
    metrics['tv_conf_sigma'] = tracker.std_of_mean()

    mean = tracker.mean()
    std = tracker.std()

    M = ot.dist(xs_true / scale, xs_pred / scale)
    emd = ot.lp.emd2([], [], M)

    metrics['emd'] = emd

    if name is not None:
        print(f'===={name}====')
    print(f'TV distance. Mean: {mean:.3f}, Std: {std:.3f}. \nESS: {ess:.3f} \nEMD: {emd:.3f}')

    return metrics


def sample_nuts(target, proposal, num_samples=1000):
    true_target_energy = lambda z: - target(z)

    def energy(z):
        z = z['points']
        return true_target_energy(z).sum()


    #kernel = HMC(potential_fn=energy, step_size = 0.1, num_steps = K, full_mass = False)
    kernel_true = NUTS(potential_fn=energy, full_mass = False)
    init_samples = proposal.sample((1,))
    dim = init_samples.shape[-1]

    init_params = {'points': init_samples}
    mcmc_true = MCMC(kernel=kernel_true, num_samples = num_samples, initial_params=init_params)
    mcmc_true.run()

    q_true = mcmc_true.get_samples(group_by_chain=True)["points"].squeeze()
    samples_true = np.array(q_true.view(-1, dim))

    return samples_true


def plot_metrics(metrics, ndims, savepath=None, scale=1.):
    axs_names = ['Sliced TV', 'ESS', 'Euclidean EMD'] # (on scaled data)']
    ncols = len(axs_names)
    fig, axs = plt.subplots(ncols=ncols, figsize=(5*ncols + ncols, 5))
    
    for name, res in metrics.items():
        for k, v in res.items():
            res[k] = np.array(v)

        arr = res['tv_mean']
        axs[0].plot(ndims, arr, label=name, marker='o')

        axs[0].fill_between(ndims, res['tv_mean'] - 1.96 * res['tv_conf_sigma'], res['tv_mean'] + 1.96 * res['tv_conf_sigma'], alpha=0.2)

        arr = res['ess']
        axs[1].plot(ndims, arr, label=name, marker='o')

        arr = res['emd']
        axs[2].plot(ndims, arr, label=name, marker='o')

    for ax, name in zip(axs, axs_names):
        ax.grid()
        ax.set_title(name)
        ax.set_xlabel('dim')
        ax.legend()

    fig.tight_layout()

    if savepath is not None:
        plt.savefig(savepath)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--dist_config', type=str)
    # parser.add_argument('--dims', type=int, nargs='+')
    # parser.add_argument
    args = parser.parse_args()

    return args


def main(config):
    device = config.device
    method_metric_dict = defaultdict(lambda : defaultdict(list))

    for dim in config.dims:
        dist_class = eval(config.dist_class)
        target = dist_class(dim=dim, device=device, **config.dist_params.dict)

        loc_proposal = torch.zeros(dim).to(device)
        scale_proposal = config.scale_proposal * torch.ones(dim).to(device)
        proposal = IndependentNormal(loc=loc_proposal, scale=scale_proposal, device=device)

        print('========== NUTS ==========')
        samples_true = sample_nuts(target, proposal, num_samples=config.trunc_chain_len)
        samples = []
        names = []
        for method_name, info in config.methods.items():
            names.append(method_name)
            print(f'========== {method_name} =========== ')
            params = info.params 
            try:
                mcmc_class = eval(info.mcmc_class)
            except KeyError:
                print('Can\'t understand class')

            params = params.dict
            if 'lr' in params:
                params['lr'] = eval(params['lr'])

            mcmc = mcmc_class(**params, dim=dim)

            if 'flow' in info.dict.keys():
                verbose = mcmc.verbose
                mcmc.verbose = False
                flow = RNVP(info.flow.num_flows, dim=dim)

                flow_mcmc = FlowMCMC(target, proposal, flow, mcmc, batch_size=info.flow.batch_size, lr=info.flow.lr)
                flow.train()
                out_samples = flow_mcmc.train(n_steps=info.flow.n_steps)
                assert not torch.isnan(next(flow.parameters())[0, 0]).item()

                flow.eval()
                mcmc.flow = flow
                mcmc.verbose = verbose

                if 'figpath' in config.dict:
                    fig = plot_learned_density(flow, proposal, xlim=target.xlim, ylim=target.ylim)
                    plt.savefig(Path(config.figpath, f'flow_{config.dist}_{dim}.pdf'))


            start = proposal.sample((1,))

            #s = time.time()
            out = mcmc(start, target, proposal, n_steps=info.n_steps)
            #e = time.time()
            #elapsed = (e - s)
            if isinstance(out, tuple):
                sample = out[0]
            else:
                sample = out

            sample = np.array([_.detach().numpy() for _ in sample]).reshape(-1, dim)

            metrics = compute_metrics(samples_true, sample, name=method_name, trunc_chain_len=config.trunc_chain_len, ess_rar=info.ess_rar)
            for k, v in metrics.items():
                method_metric_dict[method_name][k] = list(method_metric_dict[method_name][k])
                method_metric_dict[method_name][k].append(v)

            sample = sample[-config.trunc_chain_len:]
            samples.append(sample)

    sub = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M")

    if 'respath' in config.dict:
        method_metric_dict = dict(method_metric_dict)
        resdir = Path(config.respath, config.dist)
        resdir.mkdir(parents=True, exist_ok=True)
        respath = Path(resdir, f'{sub}.npy')
        pickle.dump(method_metric_dict, respath.open('wb'))
        #method_metric_dict = pickle.load(respath.open('rb'))

    if 'figpath' in config.dict:
        SMALL_SIZE = 12 #8
        MEDIUM_SIZE = 15 #10
        BIGGER_SIZE = 15 #12

        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        plot_metrics(method_metric_dict, config.dims, savepath=Path(config.figpath, f'{config.dist}_metrics.pdf'))

        fig, axs = plt.subplots(ncols=len(names), figsize=(24, 8))
        for ax, sample in zip(axs, samples):
            _, xlim, ylim = target.plot_2d(fig, ax)

            ax.scatter(sample[:, 0], sample[:, 1], alpha=0.3, s=2, color='black')
            # plt.axis('equal')
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
        
        plt.savefig(Path(config.figpath, f'{config.dist}_proj.pdf'))
        #plt.savefig(Path(config.figpath, '{config.dist}_proj.png'))


if __name__ == "__main__":
    args = parse_arguments()
    config = yaml.load(Path(args.config).open('r'), Loader=yaml.FullLoader)
    config = DotConfig(config)

    if args.dist_config is not None:
        dist_config = yaml.load(Path(args.dist_config).open('r'), Loader=yaml.FullLoader)
        dist_config = DotConfig(dist_config)
    else:
        raise NotImplementedError

    config.n_steps = dist_config.n_steps
    config.dist = dist_config.dist
    config.dims = dist_config.dims
    config.scale_proposal = dist_config.scale_proposal
    config.dist_class = dist_config.dist_class
    config.dist_params = dist_config.params

    main(config)
