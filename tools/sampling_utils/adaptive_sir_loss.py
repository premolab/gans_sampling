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

from flows import RNVP

###Write here f divergence


#write here forward/backward KL

def forward_kl(target, proposal, flow, y):
    ##Here, y \sim \target
    ###PROPOSAL INITIAL DISTR HERE
    y_ = y.detach().requires_grad_()
    u, log_jac = flow.inverse(y_)
    est = target.log_prob(y) - proposal.log_prob(u) - log_jac
    grad_est = - proposal.log_prob(u) - log_jac
    return est, grad_est



def backward_kl(target, proposal, flow, y):
    u = proposal.sample(y.shape)
    x, log_jac = flow(u)
    est = proposal.log_prob(u) - log_jac - target.log_prob(x)
    grad_est = - log_jac - target.log_prob(x)
    return est, grad_est

def mix_kl(target, proposal, flow, y, alpha):
    est_f, grad_est_f = forward_kl(target, proposal, flow, y)
    est_b, grad_est_b = backward_kl(target, proposal, flow, y)
    return alpha * est_f + (1. - alpha) * est_b, alpha * grad_est_f + (1. - alpha) * grad_est_b 





