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

def MALA_sampling(generator, discriminator, 
                  z_dim, eps, num_iter, device):
   loc = torch.zeros(z_dim).to(device)
   scale = torch.ones(z_dim).to(device)
   normal = Normal(loc, scale)
   diagn = Independent(normal, 1)
   uniform_sampler = Uniform(low = 0.0, high = 1.0)
   cur_z = diagn.sample()
   cur_z.requires_grad_(True)
   latent_arr = [cur_z.clone()]
   for i in range(num_iter):
      GAN_part = -discriminator(generator(cur_z))
      latent_part = -diagn.log_prob(cur_z)
      cur_energy = GAN_part + latent_part 
      cur_energy.backward()
      noise = diagn.sample()
      gamma = eps/2
      with torch.no_grad():
         new_z = (cur_z - gamma*cur_z.grad + (eps ** 0.5)*noise)
      new_z = new_z.clone()
      new_z.requires_grad_(True)
      new_energy = -discriminator(generator(new_z)) - diagn.log_prob(new_z)
      new_energy.backward()
      energy_part = cur_energy - new_energy
      with torch.no_grad():
         vec_for_propose_2 = cur_z - new_z + gamma*new_z.grad
      propose_part_2 = (vec_for_propose_2 @ vec_for_propose_2)/4.0/gamma
      propose_part = (noise @ noise)/2.0 - propose_part_2
      log_accept_prob = propose_part + energy_part
      generate_uniform_var = uniform_sampler.sample().to(device)
      log_generate_uniform_var = torch.log(generate_uniform_var)
      if log_generate_uniform_var < log_accept_prob:
          latent_arr.append(new_z.clone())
          cur_z = new_z
          cur_z.grad.data.zero_()

   latent_arr = torch.stack(latent_arr, dim = 0)
   return latent_arr

def NUTS_sampling(generator, discriminator, z_dim, num_samples, device):
   cur_calculate_energy = partial(calculate_energy, 
                                  generator = generator,
                                  discriminator = discriminator)
   kernel = NUTS(potential_fn = cur_calculate_energy)
   loc = torch.zeros(z_dim).to(device)
   scale = torch.ones(z_dim).to(device)
   normal = Normal(loc, scale)
   diagn = Independent(normal, 1)
   init_params = diagn.sample()
   init_params = {'points': init_params}
   mcmc = MCMC(kernel = kernel, 
               num_samples = num_samples, 
               initial_params = init_params,
               num_chains = 1)
   mcmc.run()
   latent_arr = mcmc.get_samples()['points']
   return latent_arr, mcmc
