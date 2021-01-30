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
from tqdm import tqdm
from easydict import EasyDict as edict
    
def grad_energy(point, target, x=None):
    point = point.detach().requires_grad_()
    if x is not None:
        energy = -target(z=point, x=x)
    else:
        energy = -target(point)
    grad = torch.autograd.grad(energy.sum(), point)[0]
    return energy, grad    

def gan_energy(z, generator, discriminator, proposal, normalize_to_0_1, log_prob=False):
    generator_points = generator(z)
    if normalize_to_0_1:
        gan_part = -discriminator(generator_points).view(-1)
    else:
        sigmoid_gan_part = discriminator(generator_points)
        gan_part = -(torch.log(sigmoid_gan_part) - \
                     torch.log1p(-sigmoid_gan_part)).view(-1)
        
    proposal_part = -proposal.log_prob(z)
    # print(z.shape)
    # print(proposal_part.shape)
    # print(gan_part.shape)
    energy = gan_part + proposal_part
    if not log_prob:
       return energy
    else:
       return -energy

def langevin_dynamics(z, target, proposal, n_steps, grad_step, eps_scale):
    z_sp = []
    batch_size, z_dim = z.shape[0], z.shape[1]

    for _ in range(n_steps):
        z_sp.append(z)
        eps = eps_scale*proposal.sample([batch_size])

        E, grad = grad_energy(z, target, x=None)
        z = z - grad_step * grad + eps        
        z = z.data
        z.requires_grad_(True)
    z_sp.append(z)
    return z_sp

def langevin_sampling(target, proposal, n_steps, grad_step, eps_scale, n, batch_size):
    z_last = []
    zs = []
    for i in tqdm(range(0, n, batch_size)):
        z = proposal.sample([batch_size])
        z.requires_grad_(True)
        z_sp = langevin_dynamics(z, target, proposal, n_steps, grad_step, eps_scale)
        last = z_sp[-1].data.cpu().numpy()
        z_last.append(last)
        zs.append(np.stack([o.data.cpu().numpy() for o in z_sp], axis=0))

    z_last_np = np.asarray(z_last).reshape(-1, z.shape[-1])
    zs = np.stack(zs, axis=0)
    return z_last_np, zs
    
def mala_dynamics(z, target, proposal, n_steps, grad_step, eps_scale, acceptance_rule='Hastings'):
    z_sp = [z.clone().detach()]
    batch_size, z_dim = z.shape[0], z.shape[1]
    device = z.device

    uniform = Uniform(low = 0.0, high = 1.0)
    acceptence = torch.zeros(batch_size).to(device)

    for _ in range(n_steps):
        eps = eps_scale * proposal.sample([batch_size])

        E, grad = grad_energy(z, target, x=None)
        
        new_z = z - grad_step * grad + eps
        new_z = new_z.data
        new_z.requires_grad_(True)
        
        E_new, grad_new = grad_energy(new_z, target, x=None)
        
        energy_part = E - E_new
        
        propose_vec_1 = z - new_z + grad_step*grad_new
        propose_vec_2 = new_z - z + grad_step*grad
        
        propose_part_1 = proposal.log_prob(propose_vec_1/eps_scale)
        propose_part_2 = proposal.log_prob(propose_vec_2/eps_scale)
        
        propose_part = propose_part_1 - propose_part_2

        if acceptance_rule == 'Hastings':
            log_accept_prob = propose_part + energy_part

        elif acceptance_rule == 'Barker':
            log_ratio = propose_part + energy_part
            log_accept_prob = -torch.log(1. + torch.exp(-log_ratio))

        generate_uniform_var = uniform.sample([batch_size]).to(z.device)
        log_generate_uniform_var = torch.log(generate_uniform_var)
        mask = log_generate_uniform_var < log_accept_prob
        
        acceptence += mask
        with torch.no_grad():
            z[mask] = new_z[mask].detach().clone()
            z = z.data
            z.requires_grad_(True)
            z_sp.append(z.clone().detach())
        
    return z_sp, acceptence
    
def mala_sampling(target, proposal, n_steps, grad_step, eps_scale, n, batch_size, acceptance_rule='Hastings'):
    z_last = []
    zs = []
    for i in tqdm(range(0, n, batch_size)):
        z = proposal.sample([batch_size])
        z.requires_grad_(True)
        z_sp, acceptence = mala_dynamics(z, target, proposal, n_steps, grad_step, eps_scale, acceptance_rule=acceptance_rule)
        last = z_sp[-1].data.cpu().numpy()
        z_last.append(last)
        zs.append(np.stack([o.data.cpu().numpy() for o in z_sp], axis=0))

    z_last_np = np.asarray(z_last).reshape(-1, z.shape[-1])
    zs = np.stack(zs, axis=0)
    return z_last_np, zs


def xtry_langevin_dynamics(y, target, proposal, n_steps, grad_step, eps_scale, N):
    y_arr = [y.detach().clone()]

    batch_size, z_dim = y.shape[0], y.shape[1]

    for _ in range(n_steps):
        U = torch.randint(0, N, (batch_size,)).tolist()
        X = y.unsqueeze(1).repeat(1, N, 1)
        noise = proposal.sample([batch_size, N])

        _, grad_y = grad_energy(y, target, x=None)

        g = y - grad_step * grad_y
        g = g.data
        
        g_N = g.unsqueeze(1).repeat(1, N, 1)
        X = g_N + eps_scale*noise
        X[np.arange(batch_size), U, :] = y

        # X_batch = X.view((batch_size*N, z_dim)).detach().clone()
        # X_batch.requires_grad_(True)
        # minus_log_pi_x, minus_g_log_pi_x = grad_energy(X_batch, target, x=None)
        # T_x = X_batch - grad_step * minus_g_log_pi_x
        # T_x = T_x.reshape(X.shape)
        # minus_log_pi_x = minus_log_pi_x.reshape(X.shape[:-1])
        # log_gauss = -(X[:, None, :] - T_x[:, :, None]).norm(p=2, dim=-1)**2 / (2 * eps_scale**2)
        # sum_log_gauss = log_gauss.sum(1) - torch.diagonal(log_gauss, dim1=1, dim2=2)
        
        # log_weight = -minus_log_pi_x + sum_log_gauss

        # max_logs = torch.max(log_weight, dim = 1)[0].unsqueeze(-1).repeat((1, N))
        # log_weight = log_weight - max_logs
        # weight = torch.exp(log_weight)
        # sum_weight = torch.sum(weight, dim = 1).unsqueeze(-1).repeat((1, N))
        # weight = weight/sum_weight

        X_batch = torch.transpose(X, 0, 1).reshape((batch_size*N, z_dim)).detach().clone()

        X_batch.requires_grad_(True)
        minus_log_pi_x, minus_g_log_pi_x = grad_energy(X_batch, target, x=None) #e_grad(X_batch, normal, gen, dis, alpha, e_batch=True, ret_e=True)
        T_x = X_batch - grad_step * minus_g_log_pi_x
        T_x = torch.transpose(T_x.reshape(list(X.shape[:-1][::-1]) + [X.shape[-1]]), 0, 1)
        minus_log_pi_x = minus_log_pi_x.reshape(X.shape[:-1][::-1]).T

        log_gauss = -(X[:, :, None] - T_x[:, None, :]).norm(p=2, dim=-1)**2 / (2 * eps_scale**2)
        sum_log_gauss = log_gauss.sum(1) - torch.diagonal(log_gauss, dim1=1, dim2=2)

        log_weight = -minus_log_pi_x + sum_log_gauss

        max_logs = torch.max(log_weight, dim = 1)[0].unsqueeze(-1).repeat((1, N))
        log_weight = log_weight - max_logs
        weight = torch.exp(log_weight)
        sum_weight = torch.sum(weight, dim = 1)
        weight = weight/sum_weight[:, None]

        weight[weight != weight] = 0.
        weight[weight.sum(1) == 0.] = 1.

        indices = torch.multinomial(weight, 1).squeeze().tolist()

        y = X[np.arange(batch_size), indices, :]
        y = y.data
        y.requires_grad_(True)
        y_arr.append(y.detach().clone())

    return y_arr   


def xtry_langevin_sampling(target, proposal, n_steps, grad_step, eps_scale, N, n, batch_size):
    z_last = []
    zs = []
    for i in tqdm(range(0, n, batch_size)):
        z = proposal.sample([batch_size])
        z.requires_grad_(True)
        z_sp = xtry_langevin_dynamics(z, target, proposal, n_steps, grad_step, eps_scale, N)
        last = z_sp[-1].data.cpu().numpy()
        z_last.append(last)
        zs.append(np.stack([o.data.cpu().numpy() for o in z_sp], axis=0))

    z_last_np = np.asarray(z_last).reshape(-1, z.shape[-1])
    zs = np.stack(zs, axis=0)
    return z_last_np, zs


def get_mh_kernel_log_prob(log_pi1, log_pi2, log_transition_forward, log_transition_backward):
    return (log_pi2 + log_transition_backward) - (log_pi1 + log_transition_forward)


def get_langevin_transition_kernel(z1, z2, grad, grad_step, sigma, stand_normal=None):
    loc = z2 - z1 + grad_step*grad
    log_prob = stand_normal.log_prob(loc/sigma)
    return log_prob


def do_transition_step(z, z_new, energy, grad, grad_step, sigma, stand_normal=None, uniform=None, target=None):
    energy_new, grad_new = grad_energy(z_new, target, x=None)
    log_transition_forward = get_langevin_transition_kernel(z, z_new, grad, grad_step, sigma, stand_normal)
    log_transition_backward = get_langevin_transition_kernel(z_new, z, grad_new, grad_step, sigma, stand_normal)
    acc_log_prob = get_mh_kernel_log_prob(-energy, -energy_new, log_transition_forward, log_transition_backward)
    
    generate_uniform_var = uniform.sample([z.shape[0]]).to(z.device)
    log_generate_uniform_var = torch.log(generate_uniform_var)
    mask = log_generate_uniform_var < acc_log_prob
    
    with torch.no_grad():
        z[mask] = z_new[mask].detach().clone()
        z = z.data
        z.requires_grad_(True)
        energy[mask] = energy_new[mask]
        energy[~mask] = energy[~mask]
        
        grad[mask] = grad_new[mask]
        grad[~mask] = grad[~mask]

    return z, energy, grad


def tempered_transitions_dynamics(z, target, proposal, n_steps, grad_step, eps_scale, *betas):
    z_sp = [z.clone().detach()]
    batch_size, z_dim = z.shape[0], z.shape[1]
    device = z.device

    uniform = Uniform(low = 0.0, high = 1.0)

    loc = torch.zeros(z_dim).to(device)
    scale = torch.ones(z_dim).to(device)
    dist_args = edict()
    dist_args.device = device
    dist_args.loc = loc
    dist_args.scale = scale
    stand_normal = IndependentNormal(dist_args)

    acceptence = torch.zeros(batch_size).to(device)

    betas = np.array(betas)
    betas_diff = torch.FloatTensor(betas[:-1] - betas[1:])

    for _ in range(n_steps):
        z_forward = z.clone().detach()
        z_forward.requires_grad_(True)
        energy_forward = torch.zeros(batch_size, len(betas))
        E, grad = grad_energy(z_forward, target, x=None)
        energy_forward[:, 0] = E
        
        for i, beta in enumerate(betas):
            if beta == 1.0:
                continue
            eps = eps_scale * proposal.sample([batch_size])

            z_forward_new = z_forward - grad_step * beta * grad + eps
            z_forward, E, grad = do_transition_step(z_forward, z_forward_new, E, grad, grad_step * beta, eps_scale, stand_normal, uniform, target)
            energy_forward[:, i] = E

        z_backward = z_forward.clone().detach()
        z_backward.requires_grad_(True)
        energy_backward = torch.zeros(batch_size, len(betas))
        for i, beta in enumerate(betas[::-1]):
            if beta == 1.0:
                continue
            eps = eps_scale * proposal.sample([batch_size])

            z_backward_new = z_backward - grad_step * beta * grad + eps
            z_backward, E, grad = do_transition_step(z_backward, z_backward_new, E, grad, grad_step * beta, eps_scale, stand_normal, uniform, target)
            j = len(betas) - i - 2
            energy_backward[:, j] = E

        F_forward = (betas_diff * energy_forward[:, :-1]).sum(-1)
        F_backward = (betas_diff * energy_backward[:, :-1]).sum(-1)
        log_accept_prob = F_forward - F_backward
        
        generate_uniform_var = uniform.sample([batch_size]).to(z.device)
        log_generate_uniform_var = torch.log(generate_uniform_var)
        mask = log_generate_uniform_var < log_accept_prob
        
        acceptence += mask
        with torch.no_grad():
            z[mask] = z_backward[mask].detach().clone()
            z = z.data
            z.requires_grad_(True)
            z_sp.append(z.clone().detach())
        
    return z_sp, acceptence


def tempered_transitions_sampling(target, proposal, n_steps, grad_step, eps_scale, n, batch_size, betas):
    z_last = []
    zs = []
    for i in tqdm(range(0, n, batch_size)):
        z = proposal.sample([batch_size])
        z.requires_grad_(True)
        z_sp, acceptence = tempered_transitions_dynamics(z, target, proposal, n_steps, grad_step, eps_scale, *betas)
        last = z_sp[-1].data.cpu().numpy()
        z_last.append(last)
        zs.append(np.stack([o.data.cpu().numpy() for o in z_sp], axis=0))

    z_last_np = np.asarray(z_last).reshape(-1, z.shape[-1])
    zs = np.stack(zs, axis=0)
    return z_last_np, zs


def ais_vanilla_dynamics(z, target, proposal, n_steps, grad_step, eps_scale, N, *betas):
    z_sp = [z.clone().detach()]
    batch_size, z_dim = z.shape[0], z.shape[1]
    device = z.device

    uniform = Uniform(low = 0.0, high = 1.0)

    loc = torch.zeros(z_dim).to(device)
    scale = torch.ones(z_dim).to(device)
    dist_args = edict()
    dist_args.device = device
    dist_args.loc = loc
    dist_args.scale = scale
    stand_normal = IndependentNormal(dist_args)

    acceptence = torch.zeros(batch_size).to(device)

    betas = np.array(betas)
    betas_diff = torch.FloatTensor(betas[:-1] - betas[1:])

    for _ in range(n_steps):
        Z = z.unsqueeze(1).repeat(1, N, 1)
        z_batch = torch.transpose(Z, 0, 1).reshape((batch_size*N, z_dim)).detach().clone()
        z_batch.requires_grad_(True)

        E, grad = grad_energy(z_batch, target, x=None)

        z_backward = z_batch.clone().detach()
        z_backward.requires_grad_(True)
        energy_backward = torch.zeros(batch_size, N, len(betas))
        for i, beta in enumerate(betas[::-1]):
            if beta == 1.0:
                continue
            eps = eps_scale * proposal.sample([batch_size*N])

            z_backward_new = z_backward - grad_step * beta * grad + eps
            z_backward, E, grad = do_transition_step(z_backward, z_backward_new, E, grad, grad_step * beta, eps_scale, stand_normal, uniform, target)
            j = len(betas) - i - 2
            E_ = E.reshape(Z.shape[:-1][::-1]).T
            energy_backward[..., j] = E_

        z_backward = torch.transpose(z_backward.reshape(list(Z.shape[:-1][::-1]) + [Z.shape[-1]]), 0, 1)

        F_backward = (betas_diff[None, None, :] * energy_backward[..., :-1]).sum(-1)
        log_weights = -F_backward

        max_logs = torch.max(log_weights, dim = 1)[0].unsqueeze(-1).repeat((1, N))
        log_weights = log_weights - max_logs
        sum_weights = torch.logsumexp(log_weights, dim = 1)
        log_weights = log_weights- sum_weights[:, None]
        weights = log_weights.exp()
        weights[weights != weights] = 0.
        weights[weights.sum(1) == 0.] = 1.

        indices = torch.multinomial(weights, 1).squeeze().tolist()

        z = z_backward[np.arange(batch_size), indices, :]
        z = z.data
        z.requires_grad_(True)
        z_sp.append(z.detach().clone())
        
    return z_sp


def ais_vanilla_sampling(target, proposal, n_steps, grad_step, eps_scale, n, batch_size, N, betas):
    z_last = []
    zs = []
    for i in tqdm(range(0, n, batch_size)):
        z = proposal.sample([batch_size])
        z.requires_grad_(True)
        z_sp = ais_vanilla_dynamics(z, target, proposal, n_steps, grad_step, eps_scale, N, *betas)
        last = z_sp[-1].data.cpu().numpy()
        z_last.append(last)
        zs.append(np.stack([o.data.cpu().numpy() for o in z_sp], axis=0))

    z_last_np = np.asarray(z_last).reshape(-1, z.shape[-1])
    zs = np.stack(zs, axis=0)
    return z_last_np, zs
