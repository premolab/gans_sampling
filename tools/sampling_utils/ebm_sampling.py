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
    loc = z2 - (z1 - grad_step*grad)
    log_prob = stand_normal.log_prob(loc/sigma)
    return log_prob


def do_transition_step(z, z_new, energy, grad, grad_step, sigma, stand_normal=None, uniform=None, target=None, beta=1.0):
    energy_new, grad_new = grad_energy(z_new, target, x=None)
    log_transition_forward = get_langevin_transition_kernel(z, z_new, grad, beta * grad_step, sigma, stand_normal)
    log_transition_backward = get_langevin_transition_kernel(z_new, z, grad_new, beta * grad_step, sigma, stand_normal)
    acc_log_prob = get_mh_kernel_log_prob(-beta * energy, -beta * energy_new, log_transition_forward, log_transition_backward)
    
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

def compute_log_probs(z, z_new, energy, grad, grad_step, sigma, stand_normal=None, uniform=None, target=None, beta=1.0):
    energy_new, grad_new = grad_energy(z_new, target, x=None)
    log_transition_forward = get_langevin_transition_kernel(z, z_new, grad, beta * grad_step, sigma, stand_normal)
    log_transition_backward = get_langevin_transition_kernel(z_new, z, grad_new, beta * grad_step, sigma, stand_normal)
    log_prob = get_mh_kernel_log_prob(-beta * energy, -beta * energy_new, log_transition_forward, log_transition_backward)

    return log_prob#, energy, grad


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
    E0, grad0 = grad_energy(z, target, x=None)
    for _ in range(n_steps):
        z_forward = z.clone().detach()
        z_forward.requires_grad_(True)
        energy_forward = torch.zeros(batch_size, len(betas)-1)
        energy_forward[:, 0] = E0

        E = E0#.data #.detach().clone()
        grad = grad0#.data #detach().clone()
        
        for i, beta in enumerate(betas[1:-1]):
            eps = eps_scale * proposal.sample([batch_size])

            z_forward_new = z_forward - grad_step * beta * grad + eps
            z_forward, E, grad = do_transition_step(z_forward, z_forward_new, E, grad, grad_step, eps_scale, stand_normal, uniform, target, beta=beta)
            energy_forward[:, i+1] = E

        z_backward = z_forward.clone().detach()
        z_backward.requires_grad_(True)
        energy_backward = torch.zeros(batch_size, len(betas)-1)
        energy_backward[:, -1] = E
        for i, beta in enumerate(betas[::-1][1:-1]):
            eps = eps_scale * proposal.sample([batch_size])

            z_backward_new = z_backward - grad_step * beta * grad + eps
            z_backward, E, grad = do_transition_step(z_backward, z_backward_new, E, grad, grad_step, eps_scale, stand_normal, uniform, target, beta=beta)
            j = len(betas) - i - 3
            energy_backward[:, j] = E

        F_forward = (betas_diff * energy_forward).sum(-1)
        F_backward = (betas_diff * energy_backward).sum(-1)
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

            E0[mask] = E[mask]#.data #detach().clone()
            #E0.requires_grad_(True)
            
            grad0[mask] = grad[mask].data #detach().clone()
            grad0.requires_grad_(True)

        
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
    betas_diff = torch.FloatTensor(betas[:-1] - betas[1:]) #n-1

    E, grad = grad_energy(z, target, x=None)
    for _ in range(n_steps):
        Z = z.unsqueeze(1).repeat(1, N, 1)
        z_batch = torch.transpose(Z, 0, 1).reshape((batch_size*N, z_dim)).detach().clone()
        z_batch.requires_grad_(True)

        E = E.unsqueeze(1).repeat(1, N)
        grad = grad.unsqueeze(1).repeat(1, N, 1)
        grad = torch.transpose(grad, 0, 1).reshape((batch_size*N, z_dim)).data #detach().clone()

        z_backward = z_batch
        z_backward.requires_grad_(True)
        energy_backward = torch.zeros(batch_size, N, len(betas)-1) # n-1
        energy_backward[..., len(betas)-2] = E#.detach().clone() #E_.detach().clone()
        
        E = E.T.reshape(-1)
        for i, beta in enumerate(betas[::-1][1:-1]):  #betas[0] = 1, betas[n-1] = 0, betas[::-1][1:-1] - increasing, lenghts is n-2
            j = len(betas) - i - 3
            eps = eps_scale * proposal.sample([batch_size*N])

            z_backward_new = z_backward - beta * grad_step * grad + eps
            z_backward, E, grad = do_transition_step(z_backward, z_backward_new, E, grad, grad_step, eps_scale, stand_normal, uniform, target, beta=beta)
            
            E_ = E.reshape(Z.shape[:-1][::-1]).T
            energy_backward[..., j] = E_.detach().clone()

        z_backward = torch.transpose(z_backward.reshape(list(Z.shape[:-1][::-1]) + [Z.shape[-1]]), 0, 1)
        grad = torch.transpose(grad.reshape(list(Z.shape[:-1][::-1]) + [Z.shape[-1]]), 0, 1)
        E = E.reshape(Z.shape[:-1][::-1]).T

        F_backward = (betas_diff[None, None, :] * energy_backward).sum(-1)
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

        E = E[np.arange(batch_size), indices].data

        grad = grad[np.arange(batch_size), indices, :].data
        grad.requires_grad_(True)

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


def ais_dynamics(z, target, proposal, n_steps, grad_step, eps_scale, N, betas, rhos):
    z_sp = [z.clone().detach()]
    batch_size, T, z_dim = z.shape[0], z.shape[1], z.shape[2]
    T = T - 1  #??
    # T = len(betas) - 2
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
    betas_diff = torch.FloatTensor(betas[:-1] - betas[1:]) #n-1

    def compute_probs_from_log_probs(log_probs):
        mask_zeros = log_probs > 0.
        log_probs[mask_zeros] = 0.
        probs = log_probs.exp()
        return probs

    z_flat = torch.transpose(z, 0, 1).reshape((batch_size*z.shape[1], z_dim)).detach().clone()
    E_flat, grad_flat = grad_energy(z_flat, target, None)
    grad = torch.transpose(grad_flat.reshape(list(z.shape[:-1][::-1]) + [z.shape[-1]]), 0, 1).detach().clone()
    E = E_flat.reshape(z.shape[:-1][::-1]).T.data

    for _ in trange(n_steps):
        v = torch.zeros((batch_size, T + 1, N, z_dim), dtype = z.dtype).to(z.device)
        u = torch.zeros((batch_size, T + 1, N), dtype = z.dtype).to(z.device)

        #step 1
        kappa = torch.zeros((batch_size, T + 1, z_dim), dtype = z.dtype).to(z.device)
        kappa_t_noise = proposal.sample([batch_size, T + 1])
        kappa[:, 0, :] = rhos[-1]*z[:, 0, :] + ((1 - rhos[-1]**2)**0.5) * kappa_t_noise[:, 0, :]
        
        for t in range(1, T + 1):
            beta = betas[::-1][t]
            rho = rhos[::-1][t]

            not_equal_mask = torch.ne(z[:, t, :], z[:, t - 1, :]).max(dim=-1)[0]
            equal_mask = ~not_equal_mask             
            num_not_equal = not_equal_mask.sum()
            num_equal = equal_mask.sum()
            
            if num_not_equal > 0:
                #print(f"start to do updates for not equal batches")
                z_t_not_equal = z[not_equal_mask, t, :].detach().clone()
                z_t_1_not_equal = z[not_equal_mask, t - 1, :].detach().clone()
                z_t_not_equal.requires_grad_(True)
                z_t_1_not_equal.requires_grad_(True)

                #E_t_1_not_equal, grad_t_1_not_equal = grad_energy(z_t_1_not_equal, target, x=None)
                E_t_1_not_equal = E[not_equal_mask, t - 1]
                grad_t_1_not_equal = grad[not_equal_mask, t - 1, :]
                
                log_probs = compute_log_probs(z_t_1_not_equal, 
                                                z_t_not_equal, 
                                                E_t_1_not_equal, 
                                                grad_t_1_not_equal, 
                                                grad_step, eps_scale, stand_normal, uniform, target, beta=beta)
                

                # v[not_equal_mask, t, 0, :] = (z_t_not_equal - (1. - grad_step)*z_t_1_not_equal \
                #                                             + grad_step*betas[t]*grad_point1)/eps_scale
                v[not_equal_mask, t, 0, :] = (z_t_not_equal - z_t_1_not_equal + grad_step * beta * grad_t_1_not_equal)/eps_scale

                probs = compute_probs_from_log_probs(log_probs)
                generate_uniform_var = uniform.sample([probs.shape[0]]).to(probs.device)
                weight_uniform_var = generate_uniform_var * probs

                u[not_equal_mask, t, 0] = weight_uniform_var
            
            if num_equal > 0:
                z_t_equal = z[equal_mask, t, :]
                z_t_1_equal = z[equal_mask, t - 1, :].detach().clone()
                z_t_1_equal.requires_grad_(True)
                
                #E_t_1_equal, grad_t_1_equal = grad_energy(z_t_1_equal, target, x=None)
                E_t_1_equal = E[equal_mask, t - 1].detach().clone()
                grad_t_1_equal = grad[equal_mask, t - 1, :].detach().clone()
                
                #second_part_no_noise = (1. - grad_step)*z_t_1_equal - grad_step*betas[t]*grad_t_1_equal
                second_part_no_noise = z_t_1_equal - grad_step * beta * grad_t_1_equal

                stop = False
                num_updates = 0
                update_mask = torch.zeros(num_equal, dtype = torch.bool).to(z_t_equal.device)
                z_t_1_equal = z_t_1_equal.detach().clone()
                z_t_1_equal.requires_grad_(True)
                #print("start to while sampling")
                l = 0
                while not stop:
                    l += 1
                    cur_u = uniform.sample([num_equal]).to(z_t_equal.device)
                    cur_v = proposal.sample([num_equal]).to(z_t_equal.device)
                    second_part = second_part_no_noise + cur_v*eps_scale
                    second_part = second_part.detach().clone()
                    second_part.requires_grad_(True)
                    
                    log_probs = compute_log_probs(z_t_1_equal, 
                                                    second_part, 
                                                    E_t_1_equal, 
                                                    grad_t_1_equal, 
                                                    grad_step, eps_scale, stand_normal, uniform, target, beta=beta)

                    probs = compute_probs_from_log_probs(log_probs)
                    mask_assign = (cur_u <= probs)
                    new_assign = torch.logical_and(mask_assign, ~update_mask) 
                    
                    u[equal_mask, t, 0][new_assign] = cur_u[new_assign]
                    v[equal_mask, t, 0, :][new_assign] = cur_v[new_assign]
                    
                    update_mask = torch.logical_or(update_mask, new_assign)
                    updates_num = update_mask.sum()
                    if updates_num == num_equal:
                        stop = True
            
            kappa[:, t, :] = rho*v[:, t, 0, :] + ((1 - rho**2)**0.5) * kappa_t_noise[:, t, :]

        #step 2
        W = proposal.sample([batch_size, N - 1])
        #Z - tensor (bs, T + 1, N, dim)
        Z = torch.zeros((batch_size, T + 1, N, z_dim), dtype = z.dtype).to(z.device)
        
        kappa_repeat = kappa[:, 0, :].unsqueeze(1).repeat(1, N - 1, 1)
        kappa_N_noise = proposal.sample([batch_size, N - 1])
        kappa_repeat_N = kappa.unsqueeze(2).repeat(1, 1, N - 1, 1)

        Z[:, :, 0, :] = z
        Z[:, 0, 1:, :] = rhos[-1]*kappa_repeat + ((1 - rhos[-1]**2)**0.5) * kappa_N_noise

        kappa_flat = torch.transpose(Z[:, 0, 1:, :], 0, 1).reshape((batch_size*(N-1), z_dim)).detach().clone()
        kappa_E_flat, kappa_grad_flat = grad_energy(kappa_flat, target)
        kappa_E = kappa_E_flat.reshape(N-1, batch_size).T
        kappa_grad = torch.transpose(kappa_grad_flat.reshape(N-1, batch_size, z_dim), 0, 1)
        
        energy = torch.zeros(batch_size, T+1, N)
        
        #z_flat = torch.transpose(z, 0, 1).reshape((batch_size*(T+1), z_dim)).detach().clone()
        #E_flat, grad_flat = grad_energy(z_flat, target, None)
        #E_ = E_flat.reshape(T+1, batch_size).T
        #grad_ = torch.transpose(grad_flat.reshape(T+1, batch_size, z_dim), 0, 1)
        
        energy[:, :, 0] = E.data
        energy[:, 0, 1:] = kappa_E

        grads = torch.zeros(batch_size, T+1, N, z_dim)
        grads[:, :, 0, :] = grad.detach().clone()
        grads[:, 0, 1:, :] = kappa_grad
        
        W_2 = proposal.sample([batch_size, T, N - 1])
        
        for t in range(1, T + 1):
            beta = betas[::-1][t]
            rho = rhos[::-1][t]

            v[:, t, 1:, :] = rho*kappa_repeat_N[:, t - 1, :, :] + ((1 - rho**2)**0.5) * W_2[:, t - 1, :, :]
            z_t_1_j_shape = Z[:, t - 1, 1:, :].shape
            z_t_1_j_flatten = torch.transpose(Z[:, t - 1, 1:, :], 0, 1).reshape((batch_size*(N-1), z_dim)).detach().clone()
            z_t_1_j_flatten.requires_grad_(True)
            
            #_, grad_z_t_1_j_flatten = grad_energy(z_t_1_j_flatten, target, x=None)
            #grad_z_t_1_j = torch.transpose(grad_z_t_1_j_flatten.reshape(list(z_t_1_j_shape[:-1][::-1]) + [z_t_1_j_shape[-1]]), 0, 1)
            grad_z_t_1_j = grads[:, t - 1, 1:, :]
            grad_z_t_1_j_flatten = torch.transpose(grad_z_t_1_j, 0, 1).reshape((batch_size*(N - 1), z_dim)).detach().clone()
            E_z_t_1_j = energy[:, t - 1, 1:]
            E_z_t_1_j_flatten = E_z_t_1_j.T.reshape(batch_size*(N - 1))

            Z_t_1_j = Z[:, t - 1, 1:, :]
            Z_t_1_j_shape = Z_t_1_j.shape
            # p_t_j = (1. - grad_step)*Z_t_1_j - grad_step*betas[t]*grad_z_t_1_j \
            #                                         + eps_scale*v[:, t, 1:, :]
            p_t_j = Z_t_1_j - grad_step*beta*grad_z_t_1_j  + eps_scale*v[:, t, 1:, :]
            
            p_t_j_flatten = torch.transpose(p_t_j, 0, 1).reshape((batch_size*(N-1), z_dim)).detach().clone()
            p_t_j_flatten.requires_grad_(True)
            
            Z_t_1_j_flatten = torch.transpose(Z_t_1_j, 0, 1).reshape((batch_size*(N-1), z_dim)).detach().clone()
            Z_t_1_j_flatten.requires_grad_(True)
            
            z_t, E_t, grad_t = do_transition_step(Z_t_1_j_flatten, 
                                            p_t_j_flatten, 
                                            E_z_t_1_j_flatten, 
                                            grad_z_t_1_j_flatten, 
                                            grad_step, eps_scale, stand_normal, uniform, target, beta)

            z_t = torch.transpose(z_t.reshape(N - 1, batch_size, z_dim), 0, 1)
            E_t = E_t.reshape(N - 1, batch_size).T
            grad_t = torch.transpose(grad_t.reshape(N - 1, batch_size, z_dim), 0, 1)
            Z[:, t, 1:, :] = z_t
            energy[:, t, 1:] = E_t.detach().clone()
            grads[:, t, 1:, :] = grad_t.detach().clone()
            
        #print("start step3")
        log_weights = -(betas_diff[None, :, None] * energy[:, 1:, :]).sum(1)
            
        max_logs = torch.max(log_weights, dim = 1)[0]
        log_weights = log_weights - max_logs[:, None]
        weights = torch.exp(log_weights)
        sum_weights = torch.sum(weights, dim = 1)
        weights = weights/sum_weights[:, None]
        weights[weights != weights] = 0.
        weights[weights.sum(1) == 0.] = 1.
        
        indices = torch.multinomial(weights, 1).squeeze().tolist()
        z = Z[np.arange(batch_size), :, indices, :]
        E = energy[np.arange(batch_size), :, indices]
        grad = grads[np.arange(batch_size), :, indices, :]
        #print("end step4")
        z_sp.append(z.detach().clone())

    return z_sp