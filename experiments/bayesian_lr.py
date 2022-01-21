import argparse
import datetime
from os import name
import random
import time
from collections import defaultdict
from dataclasses import fields
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml
from easydict import EasyDict as edict
from pyro.infer import HMC, MCMC, NUTS
from seaborn.miscplot import palplot
from tqdm import tqdm
from utils import DotConfig

from iterative_sir.sampling_utils.adaptive_mc import CISIR, Ex2MCMC, FlowMCMC
from iterative_sir.sampling_utils.adaptive_sir_loss import (
    backward_kl,
    forward_kl,
    mix_kl,
)
from iterative_sir.sampling_utils.distributions import (
    BayesianLogRegression,
    IndependentNormal,
)
from iterative_sir.sampling_utils.ebm_sampling import MALA
from iterative_sir.sampling_utils.flows import RNVP
from iterative_sir.sampling_utils.logistic_regression import (
    ClassificationDatasetFactory,
)
from iterative_sir.sampling_utils.mcmc_base import AbstractMCMC
from iterative_sir.sampling_utils.metrics import (
    ESS,
    MetricsTracker,
    acl_spectrum,
)


sns.set_theme(style="ticks", palette="deep")


def sample_nuts(target, proposal, num_samples=1000, batch_size=1):
    def true_target_energy(z):
        return -target(z)

    def energy(z):
        z = z["points"]
        return true_target_energy(z).sum()

    # kernel = HMC(potential_fn=energy, step_size = 0.1, num_steps = K, full_mass = False)
    kernel_true = NUTS(potential_fn=energy, full_mass=False)
    #kernel_true = HMC(potential_fn=energy, full_mass=False)
    init_samples = proposal.sample((batch_size,))
    dim = init_samples.shape[-1]

    init_params = {"points": init_samples}
    mcmc_true = MCMC(
        kernel=kernel_true,
        num_samples=num_samples,
        warmup_steps=0,
        initial_params=init_params,
    )
    mcmc_true.run()

    q_true = mcmc_true.get_samples(group_by_chain=True)["points"] #.squeeze()
    samples_true = np.array(q_true.view(num_samples, batch_size, dim))

    return samples_true


def classification(theta, x, y):
    # pdb.set_trace()
    P = []
    for theta_ in theta.split(10000):
        P.append(1.0 / (1.0 + torch.exp(-torch.matmul(x, theta_.transpose(0, 1)))))
    P = torch.cat(P, 1)
    #P = 1.0 / (1.0 + torch.exp(-torch.matmul(x, theta.transpose(0, 1))))
    ll = y[..., None] * torch.log(torch.clamp(P, min=1e-10)) + (
        1 - y[..., None]
    ) * torch.log(torch.clamp(1 - P, min=1e-10))
    return ll


def compute_average_mean_posterior(samples, dataset):
    ll_post = []
    for test_samples in samples:
        dim = test_samples.shape[-1]
        chunk_len = 1000
        ll = 0
        for j in range(0, len(dataset.x_test), chunk_len):
            #test_samples = np.ones((300000, 50, 65)).astype(float)
            #print(test_samples.shape, max(1, test_samples.shape[0] // 10000))
            for sample in np.array_split(test_samples, max(1, test_samples.shape[0] // 1000)):
                x_chunk = dataset.x_test[j : j + chunk_len]
                y_chunk = dataset.y_test[j : j + chunk_len]
                ll = (
                    ll
                    + classification(
                        torch.FloatTensor(sample).view(-1, dim),
                        x_chunk,
                        y_chunk,
                    )
                    .exp()
                    .numpy()
                    .sum(0)
                    .reshape(sample.shape[:-1])
                    .sum(0) / len(test_samples)
                )
        ll_post.append(ll / len(dataset.x_test))
    return ll_post


def plot(res, method_names, colors=None):
    fig, ax = plt.subplots(figsize=(4, 6))

    SMALL_SIZE = 15  # 8
    MEDIUM_SIZE = 20  # 10
    BIGGER_SIZE = 20  # 12

    plt.rc("font", size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

    # sns.set_style("darkgrid", {"axes.facecolor": ".9"})
    # sns.violinplot(data = ll_post_plot)
    palette = colors  # {name: c for name, c in zip(method_names, colors)}
    sns.boxplot(data=res, palette=palette)  # , s=6)

    plt.title(r"Average $\hat{p}(y\mid x, \mathcal{D})$")  # , fontsize = 22)
    plt.xticks(np.arange(len(method_names)), method_names)  # , fontsize = 20 )
    plt.grid()

    # degrees = 70
    # plt.xticks(rotation=degrees)

    plt.setp(
        ax.xaxis.get_majorticklabels(),
        rotation=60,
        fontsize=MEDIUM_SIZE,
        ha="right",
        rotation_mode="anchor",
    )
    plt.setp(
        ax.yaxis.get_majorticklabels(), fontsize=MEDIUM_SIZE
    )  # , ha="right", rotation_mode="anchor")

    fig.tight_layout()

    return fig


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--dataset_config", type=str)
    parser.add_argument("--dataset", type=str, default="covertype")
    parser.add_argument("--n_steps", type=int, default=None)
    parser.add_argument("--result_path", type=str)
    # parser.add_argument('--save_metric', action='store_true')
    # parser.add_argument('--plot')
    args = parser.parse_args()

    return args


def main(dataset, config, run=True):
    n_steps = config.n_steps
    #trunc_chain_len = int(0.5 * n_steps)

    target = BayesianLogRegression(dataset, device=config.device)
    dim = target.d
    metrics = MetricsTracker(fields=["method", "ess", "ess_per_s", "time"])

    if run:
        proposal = IndependentNormal(dim=dim, device=config.device, scale=config.scale)
        colors = []
        samples = []

        for method_name, info in config.methods.items():
            print(f"========= {method_name} ========== ")
            params = info.params  # ['params']
            colors.append(info.color)
            try:
                mcmc_class = eval(info.mcmc_class)
                # print(mcmc_class)
                # assert isinstance(mcmc_class, AbstractMCMC)
            except KeyError:
                print("Can't understand class")

            params = params.dict
            # print(params)
            if "lr" in params:
                params["lr"] = eval(params["lr"])

            if method_name == 'NUTS':
                s = time.time()
                sample = sample_nuts(target, proposal, info.n_steps, batch_size=config.batch_size)
                e = time.time()
                elapsed = e - s
                print(sample.shape)
            else:
                mcmc = mcmc_class(**params, dim=dim)

                if "flow" in info.dict.keys():
                    verbose = mcmc.verbose
                    mcmc.verbose = False
                    flow = RNVP(info.flow.num_flows, dim=dim)

                    flow_mcmc = FlowMCMC(
                        target,
                        proposal,
                        flow,
                        mcmc,
                        batch_size=info.flow.batch_size,
                        lr=info.flow.lr,
                        jump_tol=1e6,
                    )
                    flow.train()
                    out_samples, nll = flow_mcmc.train(n_steps=info.flow.n_steps)
                    #
                    assert not torch.isnan(next(flow.parameters())[0, 0]).item()

                    flow.eval()
                    mcmc.flow = flow
                    mcmc.verbose = verbose

                start = proposal.sample([config.batch_size])

                s = time.time()
                out = mcmc(start, target, proposal, n_steps=info.burn_in + info.n_steps)
                e = time.time()
                elapsed = e - s  # / 60
                print(len(out))
                if isinstance(out, tuple):
                    sample = out[0]
                    print((out[1]*(info.burn_in + info.n_steps)).numpy().astype(int))
                else:
                    sample = out

                # ess_arr = []
                sample = torch.stack(sample, 0).detach().cpu().numpy()
            trunc_sample = sample[-info.n_steps:]
            #batch_size = trunc_sample.shape[1]
            #ess = ESS(
            #    acl_spectrum(trunc_sample[-3000:] - trunc_sample[-3000:].mean(0)[None, ...]),
            #)
            #assert ess.shape[0] == batch_size
            ess = np.array([0])
            print(
                f"Method: {method_name}, ESS: {ess.mean():.4f}, sampling time: {elapsed:.2f}, ESS/s: {ess.mean()*n_steps/elapsed:.2f}",
            )

            samples.append(trunc_sample)
            metrics.stor.method.append(method_name)
            metrics.stor.ess.append(ess)
            metrics.stor.time.append(elapsed)

        mean_post = compute_average_mean_posterior(samples, dataset)

        print(np.array(mean_post))
        print(np.array(mean_post).mean(1))
        print(np.array(mean_post).std(1))
        print(np.median(mean_post, 1))

        sub = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M")

        mean_post = {
            k: v for (k, _), v in zip(config.methods.items(), mean_post)
        }

        # if "respath" in config.dict:
        #     resdir = Path(config.respath, config.dataset)
        #     resdir.mkdir(parents=True, exist_ok=True)
        #     respath = Path(resdir, f"{sub}.npy")
        #     np.save(respath.open("wb"), mean_post)

    else:
        mean_post = np.load(Path(config.respath).open("rb"))
        colors = []
        for method_name, info in config.methods.items():
            colors.append(info.color)
            metrics.stor.method.append(method_name)
        # print(mean_post)
        mean_post = {
            method_name: mean_post[method_name]
            for method_name, _ in config.methods.items()
        }

    # if "figpath" in config.dict:
    #     fig = plot(
    #         list(mean_post.values()), list(mean_post.keys()), colors
    #     )  # metrics.stor.method, colors)
    #     plt.savefig(Path(config.figpath, f"bayesian_lr_{config.dataset}.pdf"))


if __name__ == "__main__":
    args = parse_arguments()
    config = yaml.load(Path(args.config).open("r"), Loader=yaml.FullLoader)
    config = DotConfig(config)

    data_root = config.data_root

    if args.dataset_config is not None:
        dataset_config = yaml.load(
            Path(args.dataset_config).open("r"),
            Loader=yaml.FullLoader,
        )
        dataset_config = DotConfig(dataset_config)
    else:
        dataset_config = DotConfig(
            dict(dataset=args.dataset, c1=0, c2=1, n_steps=5000),
        )

    config.n_steps = dataset_config.n_steps
    config.dataset = dataset_config.dataset
    dataset = ClassificationDatasetFactory(data_root).get_dataset(
        dataset_config.dataset,
        c1=dataset_config.c1,
        c2=dataset_config.c2,
    )

    if args.result_path is not None:
        run = False
        config.respath = args.result_path
    else:
        run = True

    main(dataset, config, run)
