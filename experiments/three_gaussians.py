import argparse
import time
from collections import defaultdict
from pathlib import Path
import scipy.stats
from tqdm import tqdm
import pandas as pd
import numpy as np
import seaborn as sns
import torch
import yaml
from easydict import EasyDict as edict
from matplotlib import pyplot as plt
from utils import DotConfig

from iterative_sir.sampling_utils.adaptive_mc import CISIR, Ex2MCMC, FlowMCMC
from iterative_sir.sampling_utils.distributions import (
    GaussianMixture,
    IndependentNormal,
    init_independent_normal,
)
from iterative_sir.sampling_utils.ebm_sampling import MALA
from iterative_sir.sampling_utils.flows import RNVP
from iterative_sir.sampling_utils.metrics import ESS, Evolution, acl_spectrum

from bayesian_lr import sample_nuts

sns.set_theme(style="ticks", palette="deep")


def define_target(
    dim=2,
    mode_weights=(1., 1., 1.),
    a=3,
    sigma=1.0,
    device="cpu",
    n_pts=10000
):
    if dim != 2:
        raise NotImplementedError

    mu = a*np.array([[0.0,1.0],[np.sqrt(3)/2,-0.5],[-np.sqrt(3)/2,-0.5]])

    target_args = edict()
    target_args.device = device
    target_args.num_gauss = 3

    target_args.p_gaussians = torch.FloatTensor(mode_weights)
    target_args.p_gaussians = target_args.p_gaussians / sum(target_args.p_gaussians)
    locs = torch.FloatTensor(mu).to(device)
    target_args.locs = locs
    target_args.covs = [
        (sigma ** 2) * torch.eye(dim, dtype=torch.float64).float().to(device),
    ] * target_args.num_gauss
    target_args.dim = dim
    target = GaussianMixture(**target_args)

    data = []
    for loc, p in zip(locs, target_args.p_gaussians):
        data.append(torch.randn(int(n_pts * p), dim) * sigma + loc[None, :])

    data = torch.cat(data, 0)
    return target, data


def compute_kl(samples_p, logp, logq):
    return ((logp(samples_p) - logq(samples_p))).mean().item()


def compute_tv(logp, logq, xlims, ylims, ax_pts=100):
    xs = torch.linspace(*xlims, ax_pts)
    ys = torch.linspace(*ylims, ax_pts)
    pts = torch.stack(torch.meshgrid(xs, ys), 2).reshape(-1, 2)
    volume = (xlims[1] - xlims[0]) * (ylims[1] - ylims[0])

    return 0.5 * torch.abs(logp(pts).exp() - logq(pts).exp()).mean().item() * volume


def compute_metrics(sample, target, steps, data, xlims, ylims, ax_pts=100, per_chain=True):
    result = {"forward KL": [], "backward KL": [], "TV": []}
    for step_id, step in tqdm(enumerate(steps)):
        loc_result = {"forward KL": [], "backward KL": [], "TV": []}
        if per_chain:
            for x in sample[:step].transpose(1, 0):
                kde = scipy.stats.gaussian_kde(torch.unique(x, dim=0).T)
                logp = lambda x: torch.from_numpy(kde.logpdf(x.T))
                xs = torch.linspace(*xlims, ax_pts)
                ys = torch.linspace(*ylims, ax_pts)
                volume = (xlims[1] - xlims[0]) * (ylims[1] - ylims[0])
                pts = torch.stack(torch.meshgrid(xs, ys), 2).reshape(-1, x.shape[-1])
                f_kl = (logp(pts).exp() * (logp(pts) - target(pts))).mean().item() * volume
                loc_result["forward KL"].append(f_kl)
                loc_result["backward KL"].append(compute_kl(data, target, logp))
                loc_result["TV"].append(compute_tv(logp, target, xlims, ylims, ax_pts))

            result["forward KL"].append(
                (np.mean(loc_result["forward KL"]), np.std(loc_result["forward KL"]))
                )
            result["backward KL"].append(
                (np.mean(loc_result["backward KL"]), np.std(loc_result["backward KL"]))
                )
            result["TV"].append((np.mean(loc_result["TV"]), np.std(loc_result["TV"])))
        else:
            prev_step = step - 10 #10 #0 if step_id  == 0 else steps[step_id-1]
            x = sample[prev_step : step].reshape(-1, sample.shape[-1])
            kde = scipy.stats.gaussian_kde(torch.unique(x, dim=0).T)
            logp = lambda y: torch.from_numpy(kde.logpdf(y.T))
            xs = torch.linspace(*xlims, ax_pts)
            ys = torch.linspace(*ylims, ax_pts)
            volume = (xlims[1] - xlims[0]) * (ylims[1] - ylims[0])
            pts = torch.stack(torch.meshgrid(xs, ys), 2).reshape(-1, 2)
            f_kl = (logp(pts).exp() * (logp(pts) - target(pts))).mean().item() * volume
            result["forward KL"].append((f_kl, 0))
            result["backward KL"].append((compute_kl(data, target, logp), 0))
            result["TV"].append((compute_tv(logp, target, xlims, ylims, ax_pts), 0))
    return result
    

def plot_metrics(
    steps, forward_kl, backward_kl, tv, colors=None, savedir=None, xscale=None
):
    SMALL_SIZE = 18  # 8
    MEDIUM_SIZE = 20  # 10
    BIGGER_SIZE = 20  # 12

    plt.rc("font", size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

    fig, axs = plt.subplots(ncols=4, figsize=(20, 4))
    figs = []
    axs = []
    for _ in range(3):
        fig, ax = plt.subplots(ncols=1, figsize=(5, 4))
        figs.append(fig)
        axs.append(ax)
    
    ax_id = 0
    for i, (method_name, arr) in enumerate(forward_kl.items()):
        if colors is not None:
            color = colors[i]
            axs[ax_id].plot(steps, arr[0], label=method_name, marker="o", color=color)
        else:
            axs[ax_id].plot(steps, arr[0], label=method_name, marker="o")
        #axs[ax_id].fill_between(steps, arr[0] - 1.96 * arr[1], arr[0] + 1.96 * arr[1], alpha=0.2, color=color)
    axs[ax_id].set_xlabel("step")
    axs[ax_id].set_ylabel(r"forward KL") #$\hat{KL}(p || p^*)$")
    axs[ax_id].grid()
    axs[ax_id].legend()
    if xscale:
        axs[ax_id].set_xscale(xscale)
    ax_id += 1

    for i, (method_name, arr) in enumerate(backward_kl.items()):
        if colors is not None:
            color = colors[i]
            axs[ax_id].plot(steps, arr[0], label=method_name, marker="o", color=color)
        else:
            axs[ax_id].plot(steps, arr[0], label=method_name, marker="o")
        #axs[ax_id].fill_between(steps, arr[0] - 1.96 * arr[1], arr[0] + 1.96 * arr[1], alpha=0.2, color=color)
    axs[ax_id].set_xlabel("step")
    axs[ax_id].set_ylabel(r"backward KL")
    axs[ax_id].grid()
    axs[ax_id].legend()
    if xscale:
        axs[ax_id].set_xscale(xscale)
    ax_id += 1

    for i, (method_name, arr) in enumerate(tv.items()):
        if colors is not None:
            color = colors[i]
            axs[ax_id].plot(steps, arr[0], label=method_name, marker="o", color=color)
        else:
            axs[ax_id].plot(steps, arr[0], label=method_name, marker="o")
        #axs[ax_id].fill_between(steps, arr[0] - 1.96 * arr[1], arr[0] + 1.96 * arr[1], alpha=0.2, color=color)
    axs[ax_id].set_xlabel("step")
    axs[ax_id].set_ylabel(r"$\|p-p^*\|_{TV}$")
    axs[ax_id].grid()
    axs[ax_id].legend()
    if xscale:
        axs[ax_id].set_xscale(xscale)
    ax_id += 1

    for ax, fig, name in zip(
        axs, figs, ["forward KL", "backward KL", "TV"]
    ):
        fig.tight_layout()
        fig.savefig(Path(savedir, f"3_gauss_{name}.pdf"))
        fig.savefig(Path(savedir, f"3_gauss_{name}.png"))


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--result_path")
    args = parser.parse_args()

    return args


def main(config, run=True):
    device = torch.device(config.device)
    if "figpath" in config.dict:
        Path(config.figpath).mkdir(exist_ok=True)

    forward_kl = defaultdict(list)
    backward_kl = defaultdict(list)
    tv = defaultdict(list)

    samples_collection = []

    target, data = define_target(
        dim=config.dim,
        mode_weights=config.mode_weights,
        a=config.a,
        sigma=config.sigma,
        n_pts=config.n_data_pts,
        device=device,
    )
    proposal = init_independent_normal(
        config.scale_proposal,
        config.dim,
        device,
        config.loc_proposal,
    )

    colors = []
    for method_name, info in config.methods.items():
        color = info.color
        colors.append(color)
        print(f"============ {method_name} =============")
        if info.mcmc_class == 'NUTS':
            s = time.time()
            sample = sample_nuts(target, proposal, num_samples=info.n_steps - info.burn_in, batch_size=info.batch_size, warmup_steps=info.burn_in)
            e = time.time()
            elapsed = e - s
            sample = torch.as_tensor(sample)
        else:
            mcmc_class = eval(info.mcmc_class)
            mcmc = mcmc_class(**info.params.dict, dim=config.dim)

            start = proposal.sample([info.batch_size]).float()

            s = time.time()
            out = mcmc(start, target, proposal, n_steps=info.n_steps)
            e = time.time()
            elapsed = e - s
            if isinstance(out, tuple):
                sample = out[0]
            else:
                sample = out

        trunc_chain_len = info.n_steps - info.burn_in
        if trunc_chain_len is not None:
                sample = sample[(-trunc_chain_len - 1) : -1]
        if isinstance(sample, list):
            sample = torch.stack(sample, axis=0).detach().cpu()

        local_result = compute_metrics(
            sample,
            target,
            config.steps,
            data,
            config.xlims,
            config.ylims,
            config.ax_pts,
            per_chain=config.per_chain
        )
        print(method_name, local_result)
        print(f"Elapsed: {elapsed:.2f} s")

        forward_kl[method_name] = np.array(local_result["forward KL"]).T
        backward_kl[method_name] = np.array(local_result["backward KL"]).T
        tv[method_name] = np.array(local_result["TV"]).T

        samples_collection.append(sample)

    if "figpath" in config.dict:
        for i, (method_name, info) in enumerate(config.methods.items()):
            delta = 0.025
            x = np.arange(*config.xlims, delta)
            y = np.arange(*config.ylims, delta)
            Xs, Ys = np.meshgrid(x, y)
            sigma = target.covs[0][0, 0] ** .5
            X = samples_collection[i][:config.plot_step, [config.d1, config.d2]]
            if isinstance(X, list):
                X = torch.stack(X, axis=0).detach().cpu()
            chain_id = 0

            plt.figure(figsize=(5, 5))
            for loc in [config.loc_1_target, config.loc_2_target, config.loc_3_target]:
                loc = np.array(loc)
                loc_d1, loc_d2 = loc[0], loc[1]

                rv = scipy.stats.multivariate_normal([loc_d1, loc_d2], [[sigma, 0], [0, sigma]])
                Z = rv.pdf(np.dstack((Xs, Ys)))
                z1 = rv.pdf(np.array([loc_d1, loc_d2 + 1*sigma]))
                z2 = rv.pdf(np.array([loc_d1, loc_d2 + 2*sigma]))
                z3 = rv.pdf(np.array([loc_d1, loc_d2 + 3*sigma]))

                plt.contour(Z, [z3, z2, z1], origin='lower', colors='black', linestyles='dashed', extent=config.xlims+config.ylims)
                plt.scatter(loc_d1, loc_d2, marker='x', color='r')

            plt.plot(*X[:, chain_id].permute(1, 0), marker='o', linewidth=2, markersize=2, alpha=0.5)
            plt.grid()
            plt.xlim(*config.xlims)
            plt.ylim(*config.ylims)
            plt.axis('off')
            plt.savefig(Path(config.figpath, f'{method_name}.pdf'))
            plt.savefig(Path(config.figpath, f'{method_name}.png'))
            plt.close()

            plt.figure(figsize=(5, 5))
            dframe = pd.DataFrame(X[:, chain_id].numpy(),  columns=['X','Y'])
            ax = sns.kdeplot(data=dframe, x='X', y='Y', shade = True, cmap = "PuBu")
            ax.patch.set_facecolor('white')
            ax.collections[0].set_alpha(0)
            ax.set_xlabel(r'$x_0$')
            ax.set_ylabel(r'$x_1$')
            ax.axis('off')
            plt.xlim(*config.xlims)
            plt.ylim(*config.ylims)
            plt.savefig(Path(config.figpath, f'{method_name}_kde.pdf'))
            plt.savefig(Path(config.figpath, f'{method_name}_kde.png'))
            plt.close()

        _, ax = plt.subplots(1, 1, figsize=(5, 5))
        xs = torch.linspace(*config.xlims, 100)
        ys = torch.linspace(*config.ylims, 100)
        pts = torch.stack(torch.meshgrid(xs, ys), 2).reshape(-1, 2)
        ax.contourf(xs, ys, target(pts).exp().reshape(100, 100).T, cmap = "PuBu")
        ax.patch.set_facecolor('white')
        ax.collections[0].set_alpha(0)
        ax.set_xlabel(r'$x_0$')
        ax.set_ylabel(r'$x_1$')
        ax.axis('off')
        plt.xlim(*config.xlims)
        plt.ylim(*config.ylims)
        plt.savefig(Path(config.figpath, f'real_dens.pdf'))
        plt.savefig(Path(config.figpath, f'real_dens.png'))
        plt.close()

    if "figpath" in config.dict:
        Path(config.figpath).mkdir(exist_ok=True)

        plot_metrics(
            config.steps,
            forward_kl,
            backward_kl,
            tv,
            colors=colors,
            savedir=Path(config.figpath),
            xscale=config.xscale
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
