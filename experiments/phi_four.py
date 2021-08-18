import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import seaborn as sns
import torch
import yaml
from easydict import EasyDict as edict
from matplotlib import pyplot as plt
from torch.distributions.multivariate_normal import MultivariateNormal
from utils import DotConfig, random_seed

from iterative_sir.sampling_utils.adaptive_mc import (
    CISIR,
    AugmentedULA,
    Ex2MCMC,
    FlowMCMC,
)
from iterative_sir.sampling_utils.adaptive_sir_loss import MixKLLoss
from iterative_sir.sampling_utils.distributions import (
    IndependentNormal,
    PhiFour,
)
from iterative_sir.sampling_utils.ebm_sampling import MALA, ULA
from iterative_sir.sampling_utils.flows import RNVP, RealNVP_MLP


sns.set_theme(style="ticks", palette="deep")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    # parser.add_argument('-id', '--slurm-id', type=str, default=str(random_id))

    config = parser.parse_args()
    return config


def construct_init(n_init, dim, target, proposal, ratio_pos_init=0.5):
    if ratio_pos_init == -1:
        return proposal.sample(n_init)
    else:
        x_init = torch.ones(n_init, dim)  # , device=device)
        n_pos = int(ratio_pos_init * n_init)  # config.batch_size)
        if target.tilt is None:
            x_init[n_pos:, :] = -1
        else:
            n_tilt = int(target.tilt["val"] * dim)
            x_init[n_pos:, n_tilt:] = -1
            x_init[:n_pos, : (dim - n_tilt)] = -1
        return x_init


def main(config):
    device = torch.device(config.device)

    if config.tilt_value is not None:
        tilt = {"val": config.tilt_value, "lambda": config.tilt_lambda}
    else:
        tilt = None

    target = PhiFour(
        config.a_coupling,
        config.b_field,
        config.N,
        dim_phys=config.dim_phys,
        beta=config.beta,
        tilt=tilt,
    )

    dim = config.N

    prior_arg = {
        "type": config.prior_type,
        "alpha": config.a_coupling,
        "beta": config.beta,
    }
    beta_prior = prior_arg["beta"]
    coef = prior_arg["alpha"] * dim
    prec = torch.eye(dim) * (3 * coef + 1 / coef)
    prec -= coef * torch.triu(
        torch.triu(torch.ones_like(prec), diagonal=-1).T,
        diagonal=-1,
    )
    prec = prior_arg["beta"] * prec
    prior_prec = prec.to(device)
    prior_log_det = -torch.logdet(prec)
    proposal = MultivariateNormal(
        torch.zeros((dim,), device=device),
        precision_matrix=prior_prec,
    )

    class DistWrapper:
        def __init__(self, dist, prior_prec, prior_log_det, dim):
            self.dist = dist
            self.prior_prec = prior_prec
            self.prior_log_det = prior_log_det
            self.dim = dim

        def __call__(self, z):
            prior_ll = -0.5 * torch.einsum(
                "ki,ij,kj->k",
                z,
                self.prior_prec,
                z,
            )
            prior_ll -= 0.5 * (
                self.dim * np.log(2 * np.pi) + self.prior_log_det
            )
            return prior_ll

        def sample(self, n):
            return self.dist.sample(n)

    proposal = DistWrapper(proposal, prior_prec, prior_log_det, dim)

    flow_samples = []
    mixing_samples = []
    neg_log_likelihood = []
    for method_name, info in config.methods.items():
        print(f"========== {method_name} ===========")
        mcmc_class = eval(info.mcmc_class)
        mcmc = mcmc_class(**info.params.dict, dim=dim, beta=config.beta)

        if "flow" in info.dict.keys():
            # flow = RNVP(info.flow.num_flows, dim=dim, init_weight_scale=1e-6).to(device)
            flow = RealNVP_MLP(
                dim,
                config.depth_blocks,
                1,
                hidden_dim=config.hidden_dim,  # 100,
                init_weight_scale=1e-6,
                prior_arg=prior_arg,
            ).to(device)
            x_init = construct_init(
                config.batch_size,
                dim,
                target,
                proposal,
            ).to(device)

            # burn-in
            if (
                "burn_in_steps" in info.flow.dict.keys()
            ):  # MALA(**info.params.dict, dim=dim, beta=config.beta)
                x_init_ = mcmc(
                    x_init,
                    target,
                    proposal,
                    flow=flow,
                    n_steps=info.flow.burn_in_steps,
                    verbose=True,
                )
                if isinstance(x_init_, Tuple):
                    x_init_ = x_init_[0]
                x_init_ = x_init_[-1]

                if "figpath" in config.dict.keys():
                    for i in range(x_init_.shape[0]):
                        plt.plot(
                            x_init_[i, :].detach().cpu(),
                            alpha=0.2,
                            c="b",
                        )

                    plt.savefig(Path(config.figpath, "allen_cahn_burn_in.pdf"))
                    # plt.savefig(Path(config.figpath, "allen_cahn.pdf"))
                    plt.close()

            verbose = mcmc.verbose
            mcmc.verbose = False
            loss = MixKLLoss(target, proposal, flow, gamma=0.0)
            flow_mcmc = FlowMCMC(
                target,
                proposal,
                flow,
                mcmc,
                batch_size=info.flow.batch_size,
                lr=info.flow.lr,
                loss=loss,
            )
            flow.train()
            out_samples, nll = flow_mcmc.train(
                n_steps=info.flow.n_steps,
                init_points=x_init,
                start_optim=info.flow.start_optim,
                alpha=1.0,
            )
            flow.eval()
            mcmc.flow = flow
            mcmc.verbose = verbose
            neg_log_likelihood.append(nll)

            # torch.Size([config.batch_size,])).to(device)
            prop = proposal.sample((100,))

            x_gen = flow.forward(prop)[0]
            flow_samples.append(x_gen)

            # prop = proposal.sample((10,))
            prop = torch.zeros(10, dim)
            x_gen = mcmc(prop, target, proposal, flow=flow, n_steps=10)  # 10)
            if isinstance(x_gen, Tuple):
                x_gen = x_gen[0]
            # print(x_gen)
            mixing_samples.append(x_gen[-1])

    if "figpath" in config.dict.keys():
        names = config.methods.dict.keys()
        fig, axs = plt.subplots(ncols=len(names), figsize=(6 * len(names), 5))
        for ax, sample, name in zip(axs, flow_samples, names):
            for i in range(sample.shape[0]):
                ax.plot(
                    sample[i, :].detach().cpu(),
                    alpha=0.05,
                    c="b",
                    linewidth=3,
                )
            ax.set_title(fr"{name}")

        plt.savefig(Path(config.figpath, "allen_cahn_flow.pdf"))
        plt.close()

        fig = plt.subplot()
        for name, nlls in zip(names, neg_log_likelihood):
            plt.plot(np.arange(len(nlls)), nlls, label=fr"{name}")

        plt.xlabel("Training iterations")
        plt.ylabel("- Log Likelihood + Const")
        plt.xscale("log")
        plt.grid()
        plt.legend()
        plt.savefig(Path(config.figpath, "allen_cahn_nll.pdf"))
        plt.close()

        fig, axs = plt.subplots(ncols=len(names), figsize=(6 * len(names), 5))
        for ax, sample, name in zip(axs, mixing_samples, names):
            for i in range(len(sample)):
                ax.plot(
                    sample[i].detach().cpu(),
                    alpha=0.3,
                    c="g",
                    linewidth=3,
                )
            ax.set_title(fr"{name}")

        plt.savefig(Path(config.figpath, "allen_cahn_mixing.pdf"))
        plt.close()

        # x_gen = out_samples[-1]
        # for i in range(x_gen.shape[0]):
        #     plt.plot(x_gen[i, :].detach().cpu(), alpha=0.2, c='b')

        # plt.savefig(Path(config.figpath, "allen_cahn_mcmc_out.pdf"))
        # plt.close()


if __name__ == "__main__":
    args = parse_arguments()
    config = yaml.load(Path(args.config).open("r"), Loader=yaml.FullLoader)
    config = DotConfig(config)
    main(config)