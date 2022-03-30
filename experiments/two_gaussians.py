import argparse
import time
from collections import defaultdict
from pathlib import Path
import scipy.stats

import numpy as np
import seaborn as sns
import torch
import yaml
from easydict import EasyDict as edict
from matplotlib import pyplot as plt
from utils import DotConfig

from iterative_sir.sampling_utils.adaptive_mc import CISIR, Ex2MCMC, FlowMCMC
from iterative_sir.sampling_utils.distributions import (
    Distribution,
    GaussianMixture,
    IndependentNormal,
    init_independent_normal,
    init_independent_normal_scale,
)
from iterative_sir.sampling_utils.ebm_sampling import MALA
from iterative_sir.sampling_utils.flows import RNVP
from iterative_sir.sampling_utils.metrics import ESS, Evolution, acl_spectrum

from bayesian_lr import sample_nuts

sns.set_theme(style="ticks", palette="deep")


def define_target(
    loc_1_target=-3,
    loc_1_nonzero=1,
    loc_2_target=3,
    loc_2_nonzero=1,
    scale_target=1,
    dim=100,
    device="cpu",
):
    target_args = edict()
    target_args.device = device
    target_args.num_gauss = 2

    coef_gaussian = 1.0 / target_args.num_gauss
    target_args.p_gaussians = [
        torch.tensor(coef_gaussian),
    ] * target_args.num_gauss
    locs = torch.stack(
        [
            loc_1_target * torch.ones(dim, dtype=torch.float64).to(device),
            loc_2_target * torch.ones(dim, dtype=torch.float64).to(device),
        ],
        0,
    )
    locs[0, loc_1_nonzero:] = 0
    locs[1, loc_2_nonzero:] = 0

    # locs_numpy = locs.cpu().numpy()
    target_args.locs = locs

    if scale_target == 'adjoint':
        scale_target = torch.norm(locs[0] - locs[1], dim=-1, p=2).item() / 6.
    target_args.covs = [
        (scale_target ** 2) * torch.eye(dim, dtype=torch.float64).to(device),
    ] * target_args.num_gauss
    target_args.dim = dim
    target = GaussianMixture(**target_args)
    return target


def compute_metrics(sample, target):


    batch_size, dim = sample.shape[1:]

    locs = target.locs
    evolution = Evolution(None, locs=locs.cpu(), sigma=target.covs[0][0, 0] ** .5)

    result_np = sample.detach().cpu().numpy()

    modes_var_arr = []
    modes_mean_arr = []
    hqr_arr = []
    hqr2_arr = []
    jsd_arr = []
    #ess_arr = []
    means_est_1 = torch.zeros(dim)
    means_est_2 = torch.zeros(dim)
    num_found_1_mode = np.zeros(2)
    num_found_2_mode = np.zeros(2)
    num_found_both_modes = np.zeros(2)

    ess_arr = ESS(
        acl_spectrum(
            (sample - sample.mean(0)[None, ...])
            .detach()
            .cpu()
            .numpy(),
        ),
    )

    for i in range(batch_size):
        X_gen = sample[:, i, :]

        assignment_chi2, assignment_2sigma = Evolution.make_assignment(
            X_gen,
            evolution.locs,
            evolution.sigma,
            q=0.95
        )
        mode_var = Evolution.compute_mode_std(X_gen, assignment_chi2)[0].item() ** 2
        modes_mean, found_modes_ind = Evolution.compute_mode_mean(
            X_gen,
            assignment_chi2,
        )
        
        if 0 in found_modes_ind and 1 in found_modes_ind:
            num_found_both_modes[0] += 1
        if 0 in found_modes_ind:
            num_found_1_mode[0] += 1
            means_est_1 += modes_mean[0]
        if 1 in found_modes_ind:
            num_found_2_mode[0] += 1
            means_est_2 += modes_mean[1]

        modes_mean, found_modes_ind = Evolution.compute_mode_mean(
            X_gen,
            assignment_2sigma,
        )

        if 0 in found_modes_ind and 1 in found_modes_ind:
            num_found_both_modes[1] += 1
        if 0 in found_modes_ind:
            num_found_1_mode[1] += 1
            # means_est_1 += modes_mean[0]
        if 1 in found_modes_ind:
            num_found_2_mode[1] += 1
            # means_est_2 += modes_mean[1]

        hqr = Evolution.compute_high_quality_rate(assignment_chi2).item()
        jsd = Evolution.compute_jsd(assignment_chi2).item()

        hqr2 = Evolution.compute_high_quality_rate(assignment_2sigma).item()

        modes_var_arr.append(mode_var)
        hqr_arr.append(hqr)
        hqr2_arr.append(hqr2)
        jsd_arr.append(jsd)

    jsd_arr = np.array(jsd_arr)
    modes_var = np.array(modes_var_arr).mean()
    hqr_arr = np.array(hqr_arr)
    hqr2_arr = np.array(hqr2_arr)

    if num_found_1_mode[0] == 0:
        print(
            "Unfortunalely, no points were assigned to 1st mode, default estimation - zero",
        )
        modes_mean_1_result = np.nan  # 0.0
    else:
        modes_mean_1_result = (means_est_1 / num_found_1_mode[0]).mean().item()
    if num_found_2_mode[0] == 0:
        print(
            "Unfortunalely, no points were assigned to 2nd mode, default estimation - zero",
        )
        modes_mean_2_result = np.nan  # 0.0
    else:
        modes_mean_2_result = (means_est_2 / num_found_2_mode[0]).mean().item()
    if num_found_1_mode[0] == 0 and num_found_2_mode[0] == 0:
        modes_mean_1_result = modes_mean_2_result = sample.mean().item()

    result = dict(
        jsd=jsd_arr.mean(),
        jsd_std=jsd_arr.std(),
        modes_var=modes_var,
        hqr=hqr_arr.mean(),
        hqr_std=hqr_arr.std(),
        hqr2=hqr2_arr.mean(),
        hqr2_std=hqr2_arr.std(),
        mode1_mean=modes_mean_1_result,
        mode2_mean=modes_mean_2_result,
        fraction_found2_modes=num_found_both_modes / batch_size,
        fraction_found1_mode=(
            num_found_1_mode + num_found_2_mode - 2 * num_found_both_modes
        )
        / batch_size,
        ess=ess_arr.mean(),
        ess_std=ess_arr.std(),
    )
    return result


def plot_metrics(
    dims, found_both, ess, ess_per_sec, hqr, hqr2, colors=None, savedir=None
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
    for _ in range(6):
        fig, ax = plt.subplots(ncols=1, figsize=(5, 4))
        figs.append(fig)
        axs.append(ax)

    ax_id = 0
    for i, (method_name, arr) in enumerate(found_both.items()):
        if colors is not None:
            color = colors[i]
            axs[ax_id].plot(dims, arr[0], label=method_name, marker="o", color=color)
        else:
            axs[ax_id].plot(dims, arr[0], label=method_name, marker="o")
    axs[ax_id].set_xlabel("dim")
    axs[ax_id].set_ylabel(r"captured 2 modes ($\chi^2$)")
    axs[ax_id].grid()
    ax_id += 1

    for i, (method_name, arr) in enumerate(found_both.items()):
        if colors is not None:
            color = colors[i]
            axs[ax_id].plot(dims, arr[1], label=method_name, marker="o", color=color)
        else:
            axs[ax_id].plot(dims, arr[1], label=method_name, marker="o")
    axs[ax_id].set_xlabel("dim")
    axs[ax_id].set_ylabel(r"captured 2 modes ($2\sigma$)")
    axs[ax_id].grid()
    ax_id += 1

    for i, (method_name, arr) in enumerate(ess.items()):
        if colors is not None:
            color = colors[i]
            axs[ax_id].plot(dims, arr, label=method_name, marker="o", color=color)
        else:
            axs[ax_id].plot(dims, arr, label=method_name, marker="o")
    axs[ax_id].set_xlabel("dim")
    axs[ax_id].set_ylabel("ESS")
    axs[ax_id].grid()
    ax_id += 1

    for i, (method_name, arr) in enumerate(ess_per_sec.items()):
        if colors is not None:
            color = colors[i]
            axs[ax_id].plot(dims, arr, label=method_name, marker="o", color=color)
        else:
            axs[ax_id].plot(dims, arr, label=method_name, marker="o")
    axs[ax_id].set_xlabel("dim")
    axs[ax_id].set_ylabel("ESS/s")
    axs[ax_id].grid()
    ax_id += 1

    for i, (method_name, arr) in enumerate(hqr.items()):
        if colors is not None:
            color = colors[i]
            axs[ax_id].plot(dims, arr, label=method_name, marker="o", color=color)
        else:
            axs[ax_id].plot(dims, arr, label=method_name, marker="o")
    # axs[3].hline
    axs[ax_id].set_xlabel("dim")
    axs[ax_id].set_ylabel(r"HQR ($\chi^2$)")
    axs[ax_id].grid()
    axs[ax_id].legend()
    ax_id += 1

    for i, (method_name, arr) in enumerate(hqr2.items()):
        if colors is not None:
            color = colors[i]
            axs[ax_id].plot(dims, arr, label=method_name, marker="o", color=color)
        else:
            axs[ax_id].plot(dims, arr, label=method_name, marker="o")
    # axs[3].hline
    axs[ax_id].set_xlabel("dim")
    axs[ax_id].set_ylabel(r"HQR ($2\sigma$)")
    axs[ax_id].grid()
    axs[ax_id].legend()
    ax_id += 1

    for ax, fig, name in zip(
        axs, figs, ["captured_chi2", "captured_2sigma", "ESS", "ESS_per_sec", "HQR_chi2", "HQR_2sigma"]
    ):
        fig.tight_layout()
        fig.savefig(Path(savedir, f"2_gauss_{name}.pdf"))
        fig.savefig(Path(savedir, f"2_gauss_{name}.png"))


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

    dim_range = np.arange(config.min_dim, config.max_dim + 1, config.step_dim)

    result = defaultdict(list)
    found_both = defaultdict(list)
    ess = defaultdict(list)
    sampling_time = defaultdict(list)
    ess_per_sec = defaultdict(list)
    hqr_dict = defaultdict(list)
    hqr2_dict = defaultdict(list)

    samples_collection = defaultdict(list)

    for dim in dim_range:
        print(f"dim = {dim}")

        target = define_target(
            config.loc_1_target,
            config.loc_1_nonzero,
            config.loc_2_target,
            config.loc_2_nonzero,
            config.scale_target,
            dim,
            device=device,
        )  # .log_prob
        proposal = init_independent_normal(
            config.scale_proposal,
            dim,
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
                elapsed = e - s  # / 60
                sample = torch.as_tensor(sample)
            else:
                mcmc_class = eval(info.mcmc_class)
                mcmc = mcmc_class(**info.params.dict, dim=dim)

                start = proposal.sample([info.batch_size])

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
                    )
                    flow.train()
                    out_samples, nll = flow_mcmc.train(
                        n_steps=info.flow.n_steps,
                    )
                    assert not torch.isnan(
                        next(flow.parameters())[0, 0],
                    ).item()

                    flow.eval()
                    mcmc.flow = flow
                    mcmc.verbose = verbose

                s = time.time()
                out = mcmc(start, target, proposal, n_steps=info.n_steps)
                e = time.time()
                elapsed = e - s  # / 60
                if isinstance(out, tuple):
                    sample = out[0]
                else:
                    sample = out

            trunc_chain_len = info.n_steps - info.burn_in
            if trunc_chain_len is not None:
                    sample = sample[(-trunc_chain_len - 1) : -1]

            if isinstance(sample, list):
                sample = torch.stack(sample, axis=0).detach().cpu() #.numpy()

            local_result = compute_metrics(
                sample,
                target,
            )
            print(method_name, local_result)
            print(f"Elapsed: {elapsed:.2f} s")

            # result[method_name].append()

            found_both[method_name].append(local_result["fraction_found2_modes"])
            ess[method_name].append(local_result["ess"])
            sampling_time[method_name].append(elapsed)
            ess_per_sec[method_name].append(
                local_result["ess"] * trunc_chain_len / elapsed,
            )
            hqr_dict[method_name].append(local_result["hqr"])
            hqr2_dict[method_name].append(local_result["hqr2"])

            if dim in config.plot_dims:
                samples_collection[dim].append(sample)

        if "figpath" in config.dict and dim in config.plot_dims:
            for i, (method_name, info) in enumerate(config.methods.items()):
                fig = plt.figure(figsize=(5, 5))

                delta = 0.025
                x = np.arange(*config.xlim, delta)
                y = np.arange(*config.ylim, delta)
                X, Y = np.meshgrid(x, y)
                sigma = target.covs[0][0, 0] ** .5
                for loc in [config.loc_1_target, config.loc_2_target]:
                    loc = np.ones(dim) * loc
                    loc[config.loc_1_nonzero:] = 0
                    loc_d1, loc_d2 = loc[config.d1], loc[config.d2]

                    rv = scipy.stats.multivariate_normal([loc_d1, loc_d2], [[sigma, 0], [0, sigma]])
                    Z = rv.pdf(np.dstack((X, Y)))
                    z1 = rv.pdf(np.array([loc_d1, loc_d2 + 1*sigma]))
                    z2 = rv.pdf(np.array([loc_d1, loc_d2 + 2*sigma]))
                    z3 = rv.pdf(np.array([loc_d1, loc_d2 + 3*sigma]))

                    plt.contour(Z, [z3, z2, z1], origin='lower', colors='black', linestyles='dashed', extent=config.xlim+config.ylim)
                    plt.scatter(loc_d1, loc_d2, marker='x', color='r')
                #

                X = samples_collection[dim][i]
                if isinstance(X, list):
                    X = torch.stack(X, axis=0).detach().cpu().numpy()
                chain_id = 0

                plt.plot(X[:, chain_id, config.d1], X[:, chain_id, config.d2], marker='o', linewidth=2, markersize=2, alpha=0.5)
                plt.grid()
                plt.xlim(*config.xlim)
                plt.ylim(*config.ylim)
                plt.savefig(Path(config.figpath, f'{method_name}_{dim}.pdf'))
                plt.savefig(Path(config.figpath, f'{method_name}_{dim}.png'))
                plt.close()

    for method_name, _ in config.methods.items():
        found_both[method_name] = np.stack(found_both[method_name], 1)

    if "figpath" in config.dict:
        Path(config.figpath).mkdir(exist_ok=True)

        plot_metrics(
            dim_range,
            found_both,
            ess,
            ess_per_sec,
            hqr_dict,
            hqr2_dict,
            colors=colors,
            savedir=Path(config.figpath),
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
