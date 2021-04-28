import os
import numpy as np
from numpy import core
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.distributions import (MultivariateNormal, 
                                 Normal, 
                                 Independent, 
                                 Uniform)

from distributions import (Target, 
                           Gaussian_mixture, 
                           IndependentNormal,
                           init_independent_normal)

import pdb
import torchvision
from scipy.stats import gamma, invgamma

import pyro
from pyro.infer import MCMC, NUTS, HMC

from functools import partial
from tqdm import tqdm, trange
from easydict import EasyDict as edict
import copy

from ebm_sampling import MALATransition, grad_energy
#from sir_ais_sampling import compute_sir_log_weights as regular_sir_log_weights


def compute_sir_log_weights(x, target, proposal, flow):
    x_pushed, log_jac = flow(x)
    log_weights = target(x_pushed) + log_jac - proposal.log_prob(x)
    return log_weights, x_pushed



def adaptive_sir_correlated_dynamics(z, target, proposal, n_steps, N, alpha, flow):
    z_sp = [] ###vector of samples
    batch_size, z_dim = z.shape[0], z.shape[1]

    for _ in range(n_steps):
        z_pushed, _ = flow(z)
        #for _ in range(K_mala):
        #    z_pushed = mala_step(z_pushed)
        z_sp.append(z_pushed)
        
        z_copy = z.unsqueeze(1).repeat(1, N, 1)
        ind = torch.randint(0, N, (batch_size,)).tolist() ###hy not [0]*batcjsize ?
        W = proposal.sample([batch_size, N])
        U = proposal.sample([batch_size]).unsqueeze(1).repeat(1, N, 1)
      #print(W.shape, U.shape, z_copy.shape)
        X = torch.zeros((batch_size, N, z_dim), dtype = z.dtype).to(z.device)
        X =  (alpha**2)*z_copy + alpha*((1- alpha**2)**0.5)*U + W*((1- alpha**2)**0.5)
        X[np.arange(batch_size), ind, :] = z
        X_view = X.view(-1, z_dim)

        log_weight, z_pushed = compute_sir_log_weights(X_view, target, proposal, flow)
        log_weight = log_weight.view(batch_size, N)
        max_logs = torch.max(log_weight, dim = 1)[0][:, None]
        log_weight = log_weight - max_logs
        weight = torch.exp(log_weight)
        sum_weight = torch.sum(weight, dim = 1)
        weight = weight/sum_weight[:, None]        

        weight[weight != weight] = 0.
        weight[weight.sum(1) == 0.] = 1.

        indices = torch.multinomial(weight, 1).squeeze().tolist()

        z = X[np.arange(batch_size), indices, :]
        z = z.data
        
    z_pushed, _ = flow(z)
    z_sp.append(z_pushed)
    return z_sp


def ex2_mcmc_mala(z, 
                target, 
                proposal, 
                n_steps, 
                N,
                grad_step,
                noise_scale, 
                corr_coef=0., 
                bernoulli_prob_corr=0., 
                flow=None):
    z_sp = [] ###vector of samples
    batch_size, z_dim = z.shape[0], z.shape[1]

    mala_transition = MALATransition(z_dim, z.device)
    bern = torch.distributions.Bernoulli(bernoulli_prob_corr)

    for _ in range(n_steps):
        if flow is not None:
            z_pushed, _ = flow(z)
            # for _ in range(K_mala):
            #     z_pushed = mala_step(z_pushed)
            #     z_sp.append(z_pushed)
        else:
            z_sp.append(z)

        if corr_coef == 0 and bernoulli_prob_corr == 0: # isir
            z_new = proposal.sample([batch_size, N - 1])
        else:
            # for simplicity assume proposal is N(0, \sigma^2 Id), need to be updated
            correlation = corr_coef * bern.sample((batch_size,))
            latent_var = correlation[:, None] * z + (1. - correlation[:, None]**2) * proposal.sample((batch_size,)) #torch.randn((batch_size, z_dim))
            correlation_new = corr_coef * bern.sample((batch_size, N - 1))
            z_new = correlation_new[..., None] * latent_var[:, None, :] + (1. - correlation_new[..., None]**2)**.5 * proposal.sample((batch_size, N - 1,)) #torch.randn((batch_size, N - 1, z_dim))
        
        X = torch.cat([z.unsqueeze(1), z_new], 1)
        X_view = X.view(-1, z_dim)

        log_weight = target(X_view) -  proposal.log_prob(X_view) #regular_sir_log_weights(X, target, proposal)
        log_weight = log_weight.view(batch_size, N)
        max_logs = torch.max(log_weight, dim = 1)[0][:, None]
        log_weight = log_weight - max_logs
        weight = torch.exp(log_weight)
        sum_weight = torch.sum(weight, dim = 1)
        weight = weight/sum_weight[:, None]        

        weight[weight != weight] = 0.
        weight[weight.sum(1) == 0.] = 1.

        indices = torch.multinomial(weight, 1).squeeze().tolist()
        z = X[np.arange(batch_size), indices, :]
        z = z.data

        # mala transition
        E, grad = grad_energy(z, target)
        z_new = z - grad_step * grad + noise_scale * proposal.sample([batch_size])
        z, _, _, mask = mala_transition.do_transition_step(z, z_new, E, grad, grad_step, noise_scale, target)

    if flow is not None:
        z_pushed, _ = flow(z)
        z_sp.append(z_pushed)
    else:
        z_sp.append(z)

    return z_sp



###Write equivalent with given normalziing flows


##write function get optimizer


###write update function


