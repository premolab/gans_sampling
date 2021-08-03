from dataclasses import fields
import numpy as np
import time
import random
from tqdm import tqdm
from functools import partial
import matplotlib.pyplot as plt
from collections import defaultdict
from easydict import EasyDict as edict
import argparse
import yaml
from pathlib import Path
import torch
import datetime
import seaborn as sns

from pyro.infer import MCMC, NUTS, HMC

from iterative_sir.sampling_utils.mcmc_base import AbstractMCMC
from iterative_sir.sampling_utils.ebm_sampling import MALA
from iterative_sir.sampling_utils.adaptive_mc import CISIR, Ex2MCMC, FlowMCMC
from iterative_sir.sampling_utils.adaptive_sir_loss import mix_kl, forward_kl, backward_kl
from iterative_sir.sampling_utils.flows import RNVP
from iterative_sir.sampling_utils.distributions import IndependentNormal, BayesianLogRegression
from iterative_sir.sampling_utils.logistic_regression import ClassificationDatasetFactory
from iterative_sir.sampling_utils.metrics import ESS, acl_spectrum, MetricsTracker

from utils import DotConfig


def classification(theta, x, y):
    #pdb.set_trace()
    P = 1. / (1. + torch.exp(-torch.matmul(x,theta.transpose(0,1))))
    ll =  y[...,None]*torch.log(torch.clamp(P, min=1e-10)) + (1-y[...,None])*torch.log(torch.clamp(1 - P, min=1e-10))
    return ll


def compute_average_mean_posterior(samples, dataset):
    ll_post = []
    for test_samples in samples:
        dim = test_samples.shape[-1]
        chunk_len = 1000
        ll = 0
        for j in range(0, len(dataset.x_test), chunk_len):
            x_chunk = dataset.x_test[j : j + chunk_len]
            y_chunk = dataset.y_test[j : j + chunk_len]
            ll = ll + classification(torch.tensor(test_samples).view(-1, dim), x_chunk, y_chunk).exp().numpy().sum(0).reshape(test_samples.shape[:-1]).mean(0)
        ll_post.append(ll / len(dataset.x_test))
    return ll_post


def plot(res, method_names):
    sns.set_theme(style="ticks", palette="deep")

    fig = plt.figure(figsize = (15, 10))

    SMALL_SIZE = 15 #8
    MEDIUM_SIZE = 15 #10
    BIGGER_SIZE = 18 #12

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    #sns.set_style("darkgrid", {"axes.facecolor": ".9"})
    #sns.violinplot(data = ll_post_plot)
    sns.boxplot(data=res) #, s=6)

    plt.title('Average $\hat{p}(y\mid x)$', fontsize = 22)
    plt.xticks(np.arange(len(method_names)), method_names, fontsize = 20 )
    plt.grid()

    return fig


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--dataset_config', type=str)
    parser.add_argument('--dataset', type=str, default='covertype')
    parser.add_argument('--n_steps', type=int, default=None)
    #parser.add_argument('--save_metric', action='store_true')
    # parser.add_argument('--plot')
    args = parser.parse_args()

    return args

def main(dataset, config):
    n_steps = config.n_steps
    trunc_chain_len = int(0.5 * n_steps)

    target = BayesianLogRegression(dataset, device=config.device)
    dim = target.d
    proposal = IndependentNormal(dim=dim, device=config.device)

    metrics = MetricsTracker(fields=['method', 'ess', 'ess_per_s', 'time'])
    samples = []

    for method_name, info in config.methods.items():
        print(f'========= {method_name} ========== ')
        params = info.params #['params']
        try:
            mcmc_class = eval(info.mcmc_class)
            #print(mcmc_class)
            #assert isinstance(mcmc_class, AbstractMCMC)
        except KeyError:
            print('Can\'t understand class')

        params = params.dict
        #print(params)
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
            #
            assert not torch.isnan(next(flow.parameters())[0, 0]).item()

            flow.eval()
            mcmc.flow = flow
            mcmc.verbose = verbose

        start = proposal.sample([config.batch_size])

        s = time.time()
        out = mcmc(start, target, proposal, n_steps=n_steps)
        e = time.time()
        elapsed = (e - s) #/ 60
        if isinstance(out, tuple):
            sample = out[0]
        else:
            sample = out

        # ess_arr = []
        sample = torch.stack(sample, 0).detach().cpu().numpy()
        trunc_sample = sample[-trunc_chain_len:]
        batch_size = sample.shape[1]
        ess = ESS(acl_spectrum((trunc_sample - trunc_sample.mean(0)[None, ...])))
        assert ess.shape[0] == batch_size
        print(f'Method: {method_name}, ESS: {ess.mean():.4f}, sampling time: {elapsed:.2f}, ESS/s: {ess.mean()*n_steps/elapsed:.2f}')

        samples.append(sample)
        metrics.stor.method.append(method_name)
        metrics.stor.ess.append(ess)
        metrics.stor.time.append(elapsed)

    mean_post = compute_average_mean_posterior(samples, dataset)
    
    print(np.array(mean_post).mean(1))
    print(np.median(mean_post, 1))

    sub = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M")

    if 'respath' in config.dict:
        resdir = Path(config.respath, config.dataset)
        resdir.mkdir(parents=True, exist_ok=True)
        respath = Path(resdir, f'{sub}.npy')
        np.save(respath.open('wb'), mean_post)

    if 'figpath' in config.dict:
        fig = plot(mean_post, metrics.stor.method)
        plt.savefig(Path(config.figpath, f'bayesian_lr_{config.dataset}.pdf'))


if __name__ == '__main__':
    args = parse_arguments()
    config = yaml.load(Path(args.config).open('r'), Loader=yaml.FullLoader)
    config = DotConfig(config)

    data_root = config.data_root

    if args.dataset_config is not None:
        dataset_config = yaml.load(Path(args.dataset_config).open('r'), Loader=yaml.FullLoader)
        dataset_config = DotConfig(dataset_config)
    else:
        dataset_config = DotConfig(dict(dataset=args.dataset, c1=0, c2=1, n_steps=5000))

    config.n_steps = dataset_config.n_steps
    config.dataset = dataset_config.dataset
    dataset = ClassificationDatasetFactory(data_root).get_dataset(dataset_config.dataset, c1=dataset_config.c1, c2=dataset_config.c2)

    main(dataset, config)
