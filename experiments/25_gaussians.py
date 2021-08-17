import argparse
import pickle
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import seaborn as sns
import torch
import yaml
from easydict import EasyDict as edict
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from utils import DotConfig

from iterative_sir.sampling_utils.adaptive_mc import CISIR, Ex2MCMC
from iterative_sir.sampling_utils.distributions import (
    GaussianMixture,
    IndependentNormal,
)
from iterative_sir.sampling_utils.ebm_sampling import MALA, ULA
from iterative_sir.sampling_utils.metrics import Evolution
from iterative_sir.sampling_utils.visualization import plot_chain_metrics
from iterative_sir.toy_examples_utils import prepare_25gaussian_data


sns.set_theme(style="ticks", palette="deep")
warnings.filterwarnings("ignore")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--result_path", type=str)
    args = parser.parse_args()
    return args


def main(config, run=True):
    colors = []

    if run:
        device = config.device
        X_train, means = prepare_25gaussian_data(
            config.train_size,
            config.sigma,
            config.random_seed,
        )

        scaler = StandardScaler()
        X_train_std = scaler.fit_transform(X_train)

        dim = 2
        num_gauss = 25
        target_args = edict()
        target_args.num_gauss = num_gauss
        n_col = 5
        n_row = num_gauss // n_col
        s = 1
        # create points
        coef_gaussian = 1.0 / num_gauss
        target_args.p_gaussians = [
            torch.tensor(coef_gaussian),
        ] * target_args.num_gauss
        locs = torch.stack(
            [
                torch.tensor([(i - 2) * s, (j - 2) * s] + [0] * (dim - 2)).to(
                    device,
                )
                for i in range(n_col)
                for j in range(n_row)
            ],
            0,
        )
        target_args.locs = locs
        target_args.covs = [
            (config.sigma ** 2) * torch.eye(dim).to(device),
        ] * target_args.num_gauss
        target_args.dim = dim
        true_target = GaussianMixture(device=device, **target_args)

        loc_proposal = torch.zeros(dim).to(device)
        scale_proposal = torch.ones(dim).to(device)
        proposal = IndependentNormal(
            device=device,
            loc=loc_proposal,
            scale=scale_proposal,
        )

        evols = dict()

        batch_size = config.batch_size

        target_sample = X_train[
            np.random.choice(np.arange(X_train.shape[0]), 1000)
        ]

        for method_name, info in config.methods.items():
            print(f"========= {method_name} ========== ")
            colors.append(info.color)
            mcmc_class = info.mcmc_class
            mcmc_class = eval(mcmc_class)
            mcmc = mcmc_class(**info.params.dict, dim=dim)

            z_0 = proposal.sample((config.batch_size,))
            out = mcmc(z_0, true_target, proposal, info.params.n_steps)

            if isinstance(out, tuple):
                sample = out[0]
            else:
                sample = out

            sample = torch.stack(sample, 0).detach().numpy()
            sample = sample[-config.n_chunks * info.every :].reshape(
                (config.n_chunks, batch_size, -1, sample.shape[-1]),
            )
            Xs_gen = sample.reshape(
                batch_size,
                config.n_chunks,
                -1,
                sample.shape[-1],
            )

            evol = defaultdict(list)
            for X_gen in Xs_gen:
                evolution = Evolution(
                    target_sample,
                    locs=locs,
                    target_log_prob=true_target,
                    sigma=config.sigma,
                    scaler=scaler,
                )
                for chunk in X_gen:
                    evolution.invoke(torch.FloatTensor(chunk))
                evol_ = evolution.as_dict()
                for k, v in evol_.items():
                    evol[k].append(v)

            for k, v in evol.items():
                evol[k] = (
                    np.mean(np.array(v), 0),
                    np.std(np.array(v), 0, ddof=1) / np.sqrt(batch_size),
                )
            evols[method_name] = evol

        if "respath" in config.dict:
            pickle.dump(
                evols,
                Path(config.respath, "gaussians_2d_metrics.pkl").open("wb"),
            )

    else:
        evols = pickle.load(Path(config.respath).open("rb"))
        for method_name, info in config.methods.items():
            colors.append(info.color)

    if "figpath" in config.dict:
        SMALL_SIZE = 15  # 8
        MEDIUM_SIZE = 20  # 10
        BIGGER_SIZE = 20  # 12

        plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
        plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
        plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

        plot_chain_metrics(
            evols,
            colors=colors,
            every=info.every,
            savepath=Path(config.figpath, "gaussians_2d.pdf"),
        )


if __name__ == "__main__":
    args = parse_arguments()
    config = yaml.load(Path(args.config).open("r"), Loader=yaml.FullLoader)
    config = DotConfig(config)

    if args.result_path is not None:
        run = False
        config.respath = args.result_path
    else:
        run = True
    main(config, run)
