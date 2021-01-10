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

def e_grad(z, P, gen, dis, alpha, ret_e=False, e_batch=False):
    logp_z = torch.sum(P.log_prob(z), dim=1)
    x = gen(z)
    d = dis(x).view(-1)
    E_batch = (-logp_z - alpha * d)
    E = E_batch.sum()
    # E = - alpha * d
    E.backward()
    grad = z.grad
    # prior_grad = chainer.grad((-logp_z, ), (z, ))
    # d_grad = chainer.grad((d, ), (z, ))
    # import pdb
    # pdb.set_trace()
    if (ret_e and not e_batch):
        return E, grad
    elif (ret_e and e_batch):
        return E_batch, grad
    else:
        return grad

def compute_log_weight(first, second, gen, dis, P, alpha, step_lr, eps_std, clip = 50.0):
    #print(first.requires_grad)
    E_first, grad_first = e_grad(first, P, gen, dis, alpha, ret_e=True, e_batch = True)
        
    new_first = first - step_lr * grad_first
    #new_first = new_first.data
    
    E_second, grad_second = e_grad(second, P, gen, dis, alpha, ret_e=True, e_batch = True)
        
    new_second = second - step_lr * grad_second
    new_second = new_second.data
    
    log_energy = E_first - E_second
    vec_1 = (first - new_second)/eps_std
    vec_2 = (second - new_first)/eps_std
    #print(vec_1)
    #print(vec_2)
    
    propose_numerator = P.log_prob(vec_1).sum(dim=1)
    propose_denumerator = P.log_prob(vec_2).sum(dim=1)
    #print(propose_numerator)
    #print(propose_denumerator)

    log_propose_part = propose_numerator - propose_denumerator
    #print(log_propose_part)
    
    log_weight = log_energy + log_propose_part
    #log_weight = torch.clamp(log_weight, -clip, clip)
    #log_weight = log_weight - torch.max(log_weight, 0)[0]

    #weight = torch.exp(log_weight)
    #weight = weight/torch.sum(weight, 0)
    log_weight = log_weight.detach()
    
    return log_weight
    
def compute_log_weight_2(first, second, gen, dis, P, alpha, step_lr, P_sigma):
    #print(first.requires_grad)
    E_first, grad_first = e_grad(first, P, gen, dis, alpha, ret_e=True, e_batch = True)
        
    new_first = first - step_lr * grad_first
    #new_first = new_first.data
    
    E_second, grad_second = e_grad(second, P, gen, dis, alpha, ret_e=True, e_batch = True)
        
    new_second = second - step_lr * grad_second
    new_second = new_second.data
    
    log_energy = E_first - E_second
    vec_1 = (first - new_second)
    vec_2 = (second - new_first)
    #print(vec_1)
    #print(vec_2)
    
    propose_numerator = P_sigma.log_prob(vec_1).sum(dim=1)
    propose_denumerator = P_sigma.log_prob(vec_2).sum(dim=1)
    #print(propose_numerator)
    #print(propose_denumerator)

    log_propose_part = propose_numerator - propose_denumerator
    #print(log_propose_part)
    
    log_weight = log_energy + log_propose_part
    #log_weight = torch.clamp(log_weight, -clip, clip)
    #log_weight = log_weight - torch.max(log_weight, 0)[0]

    #weight = torch.exp(log_weight)
    #weight = weight/torch.sum(weight, 0)
    log_weight = log_weight.detach()
    
    return log_weight
    

    
def xtry_mala_dynamics(y0, y1, gen, dis, alpha, n_steps, step_lr, eps_std, N):
    y0_arr = [y0.detach().clone()]
    y1_arr = [y1.detach().clone()]
    batch_size, z_dim = y0.shape[0], y0.shape[1]
    loc = torch.zeros(z_dim).to(gen.device)
    scale = torch.ones(z_dim).to(gen.device)
    normal = Normal(loc, scale)

    for _ in range(n_steps):
        #print(f"step = {_}")
        U = np.random.randint(N)
        Z0 = y0.unsqueeze(1).repeat(1, N, 1)
        Z1 = y1.unsqueeze(1).repeat(1, N, 1)
        noise = normal.sample([batch_size, N])
        
        E_g_j, grad_g_j = e_grad(y0, normal, gen, dis, alpha, ret_e=True)
        g_j = y0 - step_lr * grad_g_j
        g_j = g_j.data
        
        g_j_N = g_j.unsqueeze(1).repeat(1, N, 1)
        Z1 = g_j_N + eps_std*noise
        Z1[:, U, :] = y1
        
        Z0_batch = Z0.view((batch_size*N, z_dim)).detach().clone()
        Z1_batch = Z1.view((batch_size*N, z_dim)).detach().clone()
        Z0_batch.requires_grad_(True)
        Z1_batch.requires_grad_(True)
        #print(Z0_batch.requires_grad)
        #print(Z1_batch.requires_grad)
        
        log_weight = compute_log_weight(Z0_batch, Z1_batch, gen, dis, normal, alpha, step_lr, eps_std)
        log_weight = log_weight.view((batch_size, N))
        #print(log_weight)
        max_logs = torch.max(log_weight, dim = 1)[0].unsqueeze(-1).repeat((1, N))
        log_weight = log_weight - max_logs
        weight = torch.exp(log_weight)
        sum_weight = torch.sum(weight, dim = 1).unsqueeze(-1).repeat((1, N))
        
        weight = weight/sum_weight
        #print(weight)
        
        indices = torch.multinomial(weight, 1)
        indices = indices.repeat(1, z_dim).unsqueeze(1)
        with torch.no_grad():
            y1 = torch.gather(Z1, 1, indices).squeeze()
            y1 = y1.data
            #print(z)
        y1.requires_grad_(True)
        y1_arr.append(y1.detach().clone())
            
        E_y1, grad_y1 = e_grad(y1, normal, gen, dis, alpha, ret_e=True)
        y0 = y1 - step_lr * grad_y1 + eps_std*noise[:, U, :]
        y0 = y0.data
        y0.requires_grad_(True)
        y0_arr.append(y0.detach().clone())
        
    return y0_arr, y1_arr
    
def xtry_mala_dynamics_v2(y0, y1, gen, dis, alpha, n_steps, step_lr, eps_std, N):
    print(f"y1 = {y1}")
    print(f"y0 = {y0}")
    y0_arr = [y0.detach().clone()]
    y1_arr = [y1.detach().clone()]
    batch_size, z_dim = y0.shape[0], y0.shape[1]
    loc = torch.zeros(z_dim).to(gen.device)
    scale = torch.ones(z_dim).to(gen.device)
    normal = Normal(loc, scale)

    for _ in tqdm(range(n_steps)):
        print(f"step = {_}")
        print(f"y1 = {y1}")
        print(f"y0 = {y0}")
        U = np.random.randint(N)
        print(f"U = {U}")
        Z0 = y0.unsqueeze(1).repeat(1, N, 1)
        Z1 = y1.unsqueeze(1).repeat(1, N, 1)
        noise = normal.sample([batch_size, N])
        print(f"noise_std = {eps_std*noise}")
        
        E_g_j, grad_g_j = e_grad(y0, normal, gen, dis, alpha, ret_e=True)
        g_j = y0 - step_lr * grad_g_j
        g_j = g_j.data
        print(f"g_j = {g_j}")
        
        g_j_N = g_j.unsqueeze(1).repeat(1, N, 1)
        Z1 = g_j_N + eps_std*noise
        Z1[:, U, :] = y1
        print(f"Z1 = {Z1}")
        print(f"Z0 = {Z0}")
        for i in range(batch_size):
            print(f"num start = {i}")
            first = Z0[i].data
            second = Z1[i].data
            first.requires_grad_(True)
            second.requires_grad_(True)
            
            log_weight = compute_log_weight(first, second, gen, dis, normal, alpha, step_lr, eps_std)
            max_logs = torch.max(log_weight, dim = 0)[0]
            log_weight = log_weight - max_logs
            weight = torch.exp(log_weight)
            sum_weight = torch.sum(weight, dim = 0)

            weight = weight/sum_weight
            print(f"weight = {weight}")

            indices = torch.multinomial(weight, 1)
            print(f"indice = {indices}")
            y1[i] = Z1[i][indices]
        
        with torch.no_grad():
            y1 = y1.data
            #print(z)
        y1.requires_grad_(True)
        y1_arr.append(y1.detach().clone())
        print(f"y1 = {y1}")
            
        E_y1, grad_y1 = e_grad(y1, normal, gen, dis, alpha, ret_e=True)
        add_noise = eps_std*noise[:, U, :]
        print(f"add noise = {add_noise}")
        y0 = y1 - step_lr * grad_y1 + add_noise
        print(f"y0 = {y0}") 
        y0 = y0.data
        y0.requires_grad_(True)
        y0_arr.append(y0.detach().clone())
        print(f"y0 = {y0}")        
        print("-----------------------")
        
    print(f"y1_arr = {y1_arr}")
    print(f"y0_arr = {y0_arr}")
    return y0_arr, y1_arr
    
def xtry_mala_dynamics_v3(y0, y1, gen, dis, alpha, n_steps, step_lr, eps_std, N):
    print(f"y1 = {y1}")
    print(f"y0 = {y0}")
    y0_arr = [y0.detach().clone()]
    y1_arr = [y1.detach().clone()]
    batch_size, z_dim = y0.shape[0], y0.shape[1]
    loc = torch.zeros(z_dim).to(gen.device)
    scale = torch.ones(z_dim).to(gen.device)
    normal = Normal(loc, scale)
    normal_std = Normal(loc, eps_std*scale)

    for _ in tqdm(range(n_steps)):
        print(f"step = {_}")
        print(f"y1 = {y1}")
        print(f"y0 = {y0}")
        U = np.random.randint(N)
        print(f"U = {U}")
        Z0 = y0.unsqueeze(1).repeat(1, N, 1)
        Z1 = y1.unsqueeze(1).repeat(1, N, 1)
        noise = normal.sample([batch_size, N])
        print(f"noise_std = {eps_std*noise}")
        
        E_g_j, grad_g_j = e_grad(y0, normal, gen, dis, alpha, ret_e=True)
        g_j = y0 - step_lr * grad_g_j
        g_j = g_j.data
        print(f"g_j = {g_j}")
        
        g_j_N = g_j.unsqueeze(1).repeat(1, N, 1)
        Z1 = g_j_N + eps_std*noise
        Z1[:, U, :] = y1
        print(f"Z1 = {Z1}")
        print(f"Z0 = {Z0}")
        for i in range(batch_size):
            print(f"num start = {i}")
            first = Z0[i].data
            second = Z1[i].data
            first.requires_grad_(True)
            second.requires_grad_(True)
            
            log_weight = compute_log_weight_2(first, second, gen, dis, normal, alpha, step_lr, normal_std)
            max_logs = torch.max(log_weight, dim = 0)[0]
            log_weight = log_weight - max_logs
            weight = torch.exp(log_weight)
            sum_weight = torch.sum(weight, dim = 0)

            weight = weight/sum_weight
            print(f"weight = {weight}")

            indices = torch.multinomial(weight, 1)
            print(f"indice = {indices}")
            y1[i] = Z1[i][indices]
        
        with torch.no_grad():
            y1 = y1.data
            #print(z)
        y1.requires_grad_(True)
        y1_arr.append(y1.detach().clone())
        print(f"y1 = {y1}")
            
        E_y1, grad_y1 = e_grad(y1, normal, gen, dis, alpha, ret_e=True)
        add_noise = eps_std*noise[:, U, :]
        print(f"add noise = {add_noise}")
        y0 = y1 - step_lr * grad_y1 + add_noise
        print(f"y0 = {y0}") 
        y0 = y0.data
        y0.requires_grad_(True)
        y0_arr.append(y0.detach().clone())
        print(f"y0 = {y0}")        
        print("-----------------------")
        
    print(f"y1_arr = {y1_arr}")
    print(f"y0_arr = {y0_arr}")
    return y0_arr, y1_arr

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
