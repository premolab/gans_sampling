import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.distributions import (MultivariateNormal, 
                                 Normal, 
                                 Independent, 
                                 Uniform)

import pdb
import torchvision
from scipy.stats import gamma, invgamma

import pyro
from pyro.infer import MCMC, NUTS, HMC

from functools import partial
from tqdm import tqdm

torchType = torch.float32
class Target(nn.Module):
    """
    Base class for a custom target distribution
    """

    def __init__(self, kwargs):
        super().__init__()
        self.device = kwargs.device
        self.torchType = torchType
        self.device_zero = torch.tensor(0., dtype=self.torchType, device=self.device)
        self.device_one = torch.tensor(1., dtype=self.torchType, device=self.device)

    def prob(self, x):
        """
        The method returns target density, estimated at point x
        Input:
        x - datapoint
        Output:
        density - p(x)
        """
        # You should define the class for your custom distribution
        raise NotImplementedError

    def log_prob(self, x):
        """
        The method returns target logdensity, estimated at point x
        Input:
        x - datapoint
        Output:
        log_density - log p(x)
        """
        # You should define the class for your custom distribution
        raise NotImplementedError

    def sample(self, n):
        """
        The method returns samples from the distribution
        Input:
        n - amount of samples
        Output:
        samples - samples from the distribution
        """
        # You should define the class for your custom distribution
        raise NotImplementedError



class Gaussian_mixture(Target):
    """
    Mixture of n gaussians (multivariate)
    """

    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.device = kwargs.device
        self.torchType = torchType
        self.num = kwargs['num_gauss']
        self.pis = kwargs['p_gaussians']
        self.locs = kwargs['locs']  # list of locations for each of these gaussians
        self.covs = kwargs['covs']  # list of covariance matrices for each of these gaussians
        self.peak = [None] * self.num
        for i in range(self.num):
            self.peak[i] = torch.distributions.MultivariateNormal(loc=self.locs[i], covariance_matrix=self.covs[i])

    def get_density(self, x):
        """
        The method returns target density
        Input:
        x - datapoint
        z - latent vaiable
        Output:
        density - p(x)
        """
        density = self.log_prob(x).exp()
        return density

    def log_prob(self, z, x=None):
        """
        The method returns target logdensity
        Input:
        x - datapoint
        z - latent variable
        Output:
        log_density - log p(x)
        """
        log_p = torch.tensor([], device=self.device)
        #pdb.set_trace()
        for i in range(self.num):
            log_paux = (torch.log(self.pis[i]) + self.peak[i].log_prob(z)).view(-1, 1)
            log_p = torch.cat([log_p, log_paux], dim=-1)
        log_density = torch.logsumexp(log_p, dim=1)  # + torch.tensor(1337., device=self.device)
        return log_density

class DotDict(dict):
    """
    a dictionary that supports dot notation
    as well as dictionary access notation
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
def get_grad(q, target, x=None):
    q = q.detach().requires_grad_()
    if x is not None:
        s = -target(z=q, x= x)
    else:
        s = -target(q)
    grad = torch.autograd.grad(s.sum(), q)[0]
    return s, grad    


def calculate_energy(params, generator, discriminator, P, normalize_to_0_1, log_prob=False):
    generator_points = generator(params)
    if normalize_to_0_1:
        GAN_part = -discriminator(generator_points).view(-1)
    else:
        sigmoid_GAN_part = discriminator(generator_points)
        GAN_part = -(torch.log(sigmoid_GAN_part) - \
                     torch.log1p(-sigmoid_GAN_part)).view(-1)
        
    prior_part = -torch.sum(P.log_prob(params), dim=1)
    energy = GAN_part + prior_part
    if not log_prob:
       return energy
    else:
       return -energy

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
    propose_denomerator = P.log_prob(vec_2).sum(dim=1)
    #print(propose_numerator)
    #print(propose_denomerator)

    log_propose_part = propose_numerator - propose_denomerator
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
    weights_arr = []
    batch_size, z_dim = y0.shape[0], y0.shape[1]
    loc = torch.zeros(z_dim).to(gen.device)
    scale = torch.ones(z_dim).to(gen.device)
    normal = Normal(loc, scale)

    for _ in range(n_steps):
        #print(f"step = {_}")
        U = torch.randint(high = N, size = (batch_size,)).tolist()
        #print(f"U = {U}")
        #print(f"y1 = {y1}")
        #print(f"y0 = {y0}")
                          
        Z0 = y0.unsqueeze(1).repeat(1, N, 1)
        noise = normal.sample([batch_size, N])
        #print(f"noise = {noise}")
        
        E_g_j, grad_g_j = e_grad(y0, normal, gen, dis, alpha, ret_e=True)
        g_j = y0 - step_lr * grad_g_j
        g_j = g_j.data
        
        g_j_N = g_j.unsqueeze(1).repeat(1, N, 1)
        Z1 = g_j_N + eps_std*noise
        #print(f"Z1 = {Z1}")
        #print(f"Z0 = {Z0}")
        Z1[np.arange(batch_size), U, :] = y0
        #Z1[np.arange(batch_size), U, :] = y0
        #print(f"Z1 = {Z1}")
        
        Z0_batch = Z0.view((batch_size*N, z_dim)).detach().clone()
        Z1_batch = Z1.view((batch_size*N, z_dim)).detach().clone()
        Z0_batch.requires_grad_(True)
        Z1_batch.requires_grad_(True)
        
        log_weight = compute_log_weight(Z0_batch, Z1_batch, gen, dis, normal, alpha, step_lr, eps_std)
        log_weight = log_weight.view((batch_size, N))
        max_logs = torch.max(log_weight, dim = 1)[0].unsqueeze(-1).repeat((1, N))
        log_weight = log_weight - max_logs
        weight = torch.exp(log_weight)
        sum_weight = torch.sum(weight, dim = 1).unsqueeze(-1).repeat((1, N))
        
        weight = weight/sum_weight
        weights_arr.append(weight.detach().clone())
        #print(f"weight = {weight}")
        
        indices = torch.multinomial(weight, 1)
        indices_squeeze = indices.squeeze().tolist()
        #print(f"indices = {indices_squeeze}")
        #indices = indices.repeat(1, z_dim).unsqueeze(1)
        #with torch.no_grad():
        #    y1 = torch.gather(Z1, 1, indices).squeeze()
        #    y1 = y1.data
        #y1.requires_grad_(True)
        y1 = Z1[np.arange(batch_size), indices_squeeze, :]
        y1 = y1.data
        y1.requires_grad_(True)
        y1_arr.append(y1.detach().clone())
        #print(f"y1 = {y1}")
            
        E_y1, grad_y1 = e_grad(y1, normal, gen, dis, alpha, ret_e=True)
        
        noise_U = noise[np.arange(batch_size), U, :]
        #print(f"noise U = {noise_U}")
        y0 = y1 - step_lr * grad_y1 + eps_std*noise_U
        y0 = y0.data
        y0.requires_grad_(True)
        y0_arr.append(y0.detach().clone())
        #print("------------------------")
        
    return y0_arr, y1_arr, weights_arr
    
def xtry_langevin_on_target(y, target, n_steps, step_lr, eps_std, N):
    y_arr = [y.detach().clone()]

    batch_size, z_dim = y.shape[0], y.shape[1]
    loc = torch.zeros(z_dim).to(y.device)
    scale = torch.ones(z_dim).to(y.device)
    normal = Normal(loc, scale)

    for _ in tqdm(range(n_steps)):
        U = torch.randint(0, N, (batch_size,)).tolist()
        X = y.unsqueeze(1).repeat(1, N, 1)
        noise = normal.sample([batch_size, N])

        _, grad_y = get_grad(y, target, x=None)

        g = y - step_lr * grad_y
        g = g.data
        
        g_N = g.unsqueeze(1).repeat(1, N, 1)
        X = g_N + eps_std*noise
        X[np.arange(batch_size), U, :] = y

        X_batch = X.view((batch_size*N, z_dim)).detach().clone()
        X_batch.requires_grad_(True)
        minus_log_pi_x, minus_g_log_pi_x = get_grad(X_batch, target, x=None)
        T_x = X_batch - step_lr * minus_g_log_pi_x
        T_x = T_x.reshape(X.shape)
        minus_log_pi_x = minus_log_pi_x.reshape(X.shape[:-1])
        log_gauss = -(X[:, :, None] - T_x[:, None, :]).norm(p=2, dim=-1)**2 / (2 * eps_std**2)
        sum_log_gauss = log_gauss.sum(1) - torch.diagonal(log_gauss, dim1=1, dim2=2)
        
        log_weight = -minus_log_pi_x + sum_log_gauss

        max_logs = torch.max(log_weight, dim = 1)[0].unsqueeze(-1).repeat((1, N))
        log_weight = log_weight - max_logs
        weight = torch.exp(log_weight)
        sum_weight = torch.sum(weight, dim = 1).unsqueeze(-1).repeat((1, N))
        weight = weight/sum_weight

        indices = torch.multinomial(weight, 1)
        #print((indices == torch.tensor(U).unsqueeze(1)).sum().item() / float(batch_size))
        indices = indices.repeat(1, z_dim).unsqueeze(1)

        with torch.no_grad():
            y = torch.gather(X, 1, indices).squeeze()
            y = y.data
        y.requires_grad_(True)
        y_arr.append(y.detach().clone())

    return y_arr

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
    
def mala_dynamics(z, gen, dis, alpha, n_steps, step_lr, eps_std, acceptance_rule='hastings'):
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

        if acceptance_rule == 'Hastings':
            log_accept_prob = propose_part + energy_part

        elif acceptance_rule == 'Barker':
            log_ratio = propose_part + energy_part
            log_accept_prob = -torch.log(1. + torch.exp(-1.*log_ratio))

        generate_uniform_var = uniform.sample([batch_size]).to(gen.device)
        log_generate_uniform_var = torch.log(generate_uniform_var)
        mask = log_generate_uniform_var < log_accept_prob
        
        acceptence += mask
        with torch.no_grad():
            z[mask] = new_z[mask].detach().clone()
            z = z.data
            z.requires_grad_(True)
            z_sp.append(z.clone().detach())
        
    return z_sp, acceptence
    
def mala_sampling(gen, dis, alpha, n_steps, step_lr, eps_std, n, batchsize, acceptance_rule='Hastings'):
    ims = []
    zs = []
    for i in tqdm(range(0, n, batchsize)):
        z = gen.make_hidden(batchsize)
        z.requires_grad_(True)
        z_sp, acceptence = mala_dynamics(z, gen, dis, alpha, n_steps, step_lr, eps_std, acceptance_rule=acceptance_rule)
        last_z_sp = z_sp[-1].to(gen.device)
        x = gen(last_z_sp).data.cpu().numpy()
        ims.append(x)
        zs.append(np.stack([o.data.cpu().numpy() for o in z_sp], axis=0))

    ims = np.asarray(ims).reshape(-1, z.shape[-1])
    zs = np.stack(zs, axis=0)
    return ims, zs
    
def xtry_mala_sampling(gen, dis, alpha, n_steps, step_lr, eps_std, N, n, batchsize):
    ims = []
    zs = []
    for i in tqdm(range(0, n, batchsize)):
        y0 = gen.make_hidden(batchsize)
        y0.requires_grad_(True)
        y1 = gen.make_hidden(batchsize)
        y1.requires_grad_(True)
        y0_arr, y1_arr, weights = xtry_mala_dynamics(y0, y1, gen, dis, alpha, 
                                                     n_steps, step_lr, eps_std, N)
        last_z_sp = y0_arr[-1].to(gen.device)
        x = gen(last_z_sp).data.cpu().numpy()
        ims.append(x)
        zs.append(np.stack([o.data.cpu().numpy() for o in y0_arr], axis=0))

    ims = np.asarray(ims).reshape(-1, y0.shape[-1])
    zs = np.stack(zs, axis=0)
    return ims, zs

def compute_log_weight_general(z_1, z_2, target, proposal, gamma, eps_std):
    #print(first.requires_grad)
    E_first, grad_first = get_grad(z_1, target, x=None)
        
    new_first = z_1 - gamma * grad_first
    #new_first = new_first.data
    
    E_second, grad_second = get_grad(z_2, target, x=None)
        
    new_second = z_2 - gamma * grad_second
    new_second = new_second.data
    
    log_energy = E_first - E_second
    vec_1 = (z_1 - new_second)/eps_std
    vec_2 = (z_2 - new_first)/eps_std
    #print(vec_1)
    #print(vec_2)
    
    propose_numerator = proposal.log_prob(vec_1).sum(dim=1)
    propose_denumerator = proposal.log_prob(vec_2).sum(dim=1)
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

def xtry_mala_dynamics_general(y0, y1, target, alpha, n_steps, gamma, eps_std, N):
    y0_arr = [y0.detach().clone()]
    y1_arr = [y1.detach().clone()]
    weights_arr = []
    batch_size, z_dim = y0.shape[0], y0.shape[1]
    loc = torch.zeros(z_dim).to(y0.device)
    scale = torch.ones(z_dim).to(y0.device)
    normal = Normal(loc, scale)

    for _ in tqdm(range(n_steps)):
        #print(f"step = {_}")
        U = torch.randint(high = N, size = (batch_size,)).tolist()
        #print(f"U = {U}")
        #print(f"y1 = {y1}")
        #print(f"y0 = {y0}")
                          
        Z0 = y0.unsqueeze(1).repeat(1, N, 1)
        noise = normal.sample([batch_size, N])
        #print(f"noise = {noise}")
        
        E_g_j, grad_g_j = get_grad(y0, target)
        g_j = y0 - gamma * grad_g_j
        g_j = g_j.data
        
        g_j_N = g_j.unsqueeze(1).repeat(1, N, 1)
        Z1 = g_j_N + eps_std*noise
        #print(f"Z1 = {Z1}")
        #print(f"Z0 = {Z0}")
        Z1[np.arange(batch_size), U, :] = y0
        #Z1[np.arange(batch_size), U, :] = y0
        #print(f"Z1 = {Z1}")
        
        Z0_batch = Z0.view((batch_size*N, z_dim)).detach().clone()
        Z1_batch = Z1.view((batch_size*N, z_dim)).detach().clone()
        Z0_batch.requires_grad_(True)
        Z1_batch.requires_grad_(True)
        
        log_weight =   compute_log_weight_general(Z0_batch, Z1_batch, target, normal, gamma, eps_std)
        log_weight = log_weight.view((batch_size, N))
        max_logs = torch.max(log_weight, dim = 1)[0].unsqueeze(-1).repeat((1, N))
        log_weight = log_weight - max_logs
        weight = torch.exp(log_weight)
        sum_weight = torch.sum(weight, dim = 1).unsqueeze(-1).repeat((1, N))
        
        weight = weight/sum_weight
        weights_arr.append(weight.detach().clone())
        #print(f"weight = {weight}")
        
        indices = torch.multinomial(weight, 1)
        indices_squeeze = indices.squeeze().tolist()
        #print(f"indices = {indices_squeeze}")
        #indices = indices.repeat(1, z_dim).unsqueeze(1)
        #with torch.no_grad():
        #    y1 = torch.gather(Z1, 1, indices).squeeze()
        #    y1 = y1.data
        #y1.requires_grad_(True)
        y1 = Z1[np.arange(batch_size), indices_squeeze, :]
        y1 = y1.data
        y1.requires_grad_(True)
        y1_arr.append(y1.detach().clone())
        #print(f"y1 = {y1}")
            
        E_y1, grad_y1 = get_grad(y1, target)
        noise_U = noise[np.arange(batch_size), U, :]
        #print(f"noise U = {noise_U}")
        y0 = y1 - gamma * grad_y1 + eps_std*noise_U
        y0 = y0.data
        y0.requires_grad_(True)
        y0_arr.append(y0.detach().clone())
        #print("------------------------")
        
    return y0_arr, y1_arr, weights_arr
