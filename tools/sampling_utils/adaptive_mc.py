import os
import numpy as np
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





###Write equivalent with given normalziing flows


##write function get optimizer


###write update function


