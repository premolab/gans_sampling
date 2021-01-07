import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.distributions import (MultivariateNormal, 
                                 Normal, 
                                 Independent, 
                                 Uniform)

import pyro
from pyro.infer import MCMC, NUTS, HMC

from functools import partial
from tqdm import tqdm

def calculate_energy(params, generator, discriminator, P, normalize_to_0_1):
    generator_points = generator(params)
    if normalize_to_0_1:
        GAN_part = -discriminator(generator_points).view(-1)
    else:
        sigmoid_GAN_part = discriminator(generator_points)
        GAN_part = -(torch.log(sigmoid_GAN_part) - \
                     torch.log1p(-sigmoid_GAN_part)).view(-1)
        
    prior_part = -torch.sum(P.log_prob(params), dim=1)
    return GAN_part + prior_part

def e_grad(z, P, gen, dis, alpha, ret_e=False):
    logp_z = torch.sum(P.log_prob(z), dim=1)
    x = gen(z)
    d = dis(x).view(-1)
    E = (-logp_z - alpha * d).sum()
    # E = - alpha * d
    E.backward()
    grad = z.grad
    # prior_grad = chainer.grad((-logp_z, ), (z, ))
    # d_grad = chainer.grad((d, ), (z, ))
    # import pdb
    # pdb.set_trace()
    if ret_e:
        return E, grad
    return grad

def langevin_dynamics(z, gen, dis, alpha, n_steps, step_lr, eps_std):
    z_sp = []
    batch_size, z_dim = z.shape[0], z.shape[1]
    loc = torch.zeros(z_dim).to(gen.device)
    scale = torch.ones(z_dim).to(gen.device)
    normal = Normal(loc, scale)

    for _ in range(n_steps):
        z_sp.append(z)
        eps = eps_std * normal.sample([batch_size])

        E, grad = e_grad(z, normal, gen, dis, alpha, ret_e=True)
        z = z - step_lr * grad + eps        
        z = z.data
        #z = torch.clamp(z.data, -1., 1.)
        z.requires_grad_(True)
    z_sp.append(z)
    # print(n_steps, len(z_sp), z.shape)
    return z_sp

def langevin_sampling(gen, dis, alpha, n_steps, step_lr, eps_std, n, batchsize):
    ims = []
    zs = []
    for i in tqdm(range(0, n, batchsize)):
        z = gen.make_hidden(batchsize)
        z.requires_grad_(True)
        z_sp = langevin_dynamics(z, gen, dis, alpha, n_steps, step_lr, eps_std)
        x = gen(z_sp[-1]).data.cpu().numpy()
        ims.append(x)
        zs.append(np.stack([o.data.cpu().numpy() for o in z_sp], axis=0))

    ims = np.asarray(ims).reshape(-1, z.shape[-1])
    zs = np.stack(zs, axis=0)
    return ims, zs
    
def mala_dynamics(z, gen, dis, alpha, n_steps, step_lr, eps_std):
    z_sp = [z.clone().detach()]
    batch_size, z_dim = z.shape[0], z.shape[1]
    loc = torch.zeros(z_dim).to(gen.device)
    scale = torch.ones(z_dim).to(gen.device)
    
    normal = Normal(loc, scale)
    uniform = Uniform(low = 0.0, high = 1.0)
    acceptence = torch.zeros(batch_size).to(gen.device)

    for _ in range(n_steps):
        noise = normal.sample([batch_size])
        eps = eps_std * noise

        E, grad = e_grad(z, normal, gen, dis, alpha, ret_e=True)
        
        new_z = z - step_lr * grad + eps
        new_z = new_z.data
        new_z.requires_grad_(True)
        
        E_new, grad_new = e_grad(new_z, normal, gen, dis, alpha, ret_e=True)
        
        energy_part = E - E_new
        
        propose_vec = z - new_z + step_lr*grad_new
        
        propose_part_1 = normal.log_prob(noise)
        propose_part_2 = normal.log_prob(propose_vec/eps_std)
        
        propose_part = torch.sum(propose_part_2 - propose_part_1, dim=1)

        log_accept_prob = propose_part + energy_part
        generate_uniform_var = uniform.sample([batch_size]).to(gen.device)
        log_generate_uniform_var = torch.log(generate_uniform_var)

        mask = log_generate_uniform_var < log_accept_prob
        acceptence += mask
        #print(mask)
        #print(z)
        #print(new_z)
        with torch.no_grad():
            z[mask] = new_z[mask].detach().clone()
            z = z.data
            #print(z)
            z.requires_grad_(True)
            z_sp.append(z.clone().detach())
        
    return z_sp, acceptence
    
def mala_sampling(gen, dis, alpha, n_steps, step_lr, eps_std, n, batchsize):
    ims = []
    zs = []
    for i in tqdm(range(0, n, batchsize)):
        z = gen.make_hidden(batchsize)
        z.requires_grad_(True)
        z_sp, acceptence = mala_dynamics(z, gen, dis, alpha, n_steps, step_lr, eps_std)
        last_z_sp = z_sp[-1].to(gen.device)
        x = gen(last_z_sp).data.cpu().numpy()
        ims.append(x)
        zs.append(np.stack([o.data.cpu().numpy() for o in z_sp], axis=0))

    ims = np.asarray(ims).reshape(-1, z.shape[-1])
    zs = np.stack(zs, axis=0)
    return ims, zs
