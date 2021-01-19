import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F

from distributions import (Target, 
                           Gaussian_mixture, 
                           IndependentNormal,
                           init_independent_normal)

def compute_sir_weights(x):
    return (-x.norm(p=2, dim=-1)**2 / 4.0).exp()

def sir_independent_dynamics(z, proposal, n_steps, N):
    z_sp = []
    batch_size, z_dim = z.shape[0], z.shape[1]

    for _ in range(n_steps):
        z_sp.append(z)
        U = torch.randint(0, N, (batch_size,)).tolist()
        X = proposal.sample([batch_size, N])
        X[np.arange(batch_size), U, :] = z
        
        weight = compute_sir_weights(X)
        sum_weight = torch.sum(weight, dim = 1)
        weight = weight/sum_weight[:, None]
        
        weight[weight != weight] = 0.
        weight[weight.sum(1) == 0.] = 1.

        indices = torch.multinomial(weight, 1).squeeze().tolist()
        
        z = X[np.arange(batch_size), indices, :]
        z = z.data
        
    z_sp.append(z)
    return z_sp
    
def sir_correlated_dynamics(z, sigma_proposal, n_steps, N, alpha):
    z_sp = []
    batch_size, z_dim = z.shape[0], z.shape[1]
    scale = 1.0
    normal = init_independent_normal(scale, z_dim, z.device)

    for _ in range(n_steps):
        z_sp.append(z)
        z_copy = z.unsqueeze(1).repeat(1, N, 1)
        ind = torch.randint(0, N, (batch_size,)).tolist()
        W = normal.sample([batch_size, N])
        U = normal.sample([batch_size]).unsqueeze(1).repeat(1, N, 1)
        #print(W.shape, U.shape, z_copy.shape)
        X = torch.zeros((batch_size, N, z_dim), dtype = z.dtype).to(z.device)
        X =  (alpha**2)*z_copy + alpha*((1- alpha**2)**0.5)*sigma_proposal*U + W*sigma_proposal*((1- alpha**2)**0.5)
        X[np.arange(batch_size), ind, :] = z
        
        
        weight = compute_sir_weights(X)
        sum_weight = torch.sum(weight, dim = 1)
        weight = weight/sum_weight[:, None]
        
        weight[weight != weight] = 0.
        weight[weight.sum(1) == 0.] = 1.

        indices = torch.multinomial(weight, 1).squeeze().tolist()
        
        z = X[np.arange(batch_size), indices, :]
        z = z.data
        
    z_sp.append(z)
    return z_sp
    
def run_experiments_gaussians(dim_arr,  
                              scale_proposal, 
                              scale_target, 
                              num_points_in_chain, 
                              strategy_mean,
                              device,
                              batch_size,
                              method_params,
                              random_seed=42,
                              method='sir_independent',
                              print_results=True):
    modes_init = ['target', 'proposal']
    dict_results = {'target': {'mean_loc': [], 'mean_var': [], 
                               'ess': [], 'history_first': [], 
                                          'history_norm': []}, 
                    'proposal': {'mean_loc': [], 'mean_var': [], 
                                 'ess': [], 'history_first': [], 
                                 'history_norm': []}}
    
    for mode_init in modes_init:
       if print_results:
          print("------------------")
          print(f"mode = {mode_init}")
    
       for dim in dim_arr:
          if print_results:
             print(f"dim = {dim}")
          target = init_independent_normal(scale_target, dim, device)
          proposal = init_independent_normal(scale_proposal, dim, device)
          torch.manual_seed(random_seed)
          np.random.seed(random_seed)
          random.seed(random_seed)
          if mode_init == 'target':
             start = target.sample([batch_size])
          elif mode_init == 'proposal':
             start = proposal.sample([batch_size])
          if method == 'sir_correlated':
             alpha = (1 - method_params['c']/dim)**0.5
             history = sir_correlated_dynamics(start, 
                                               method_params['scale_proposal'], 
                                               method_params['n_steps'], 
                                               method_params['N'],
                                               alpha)
          elif method == 'sir_independent':
             history = sir_independent_dynamics(start, 
                                                proposal, 
                                                method_params['n_steps'], 
                                                method_params['N'])
          else:
             raise ValueError('Unknown method')    
          result = history[(-num_points_in_chain - 1):-1]
          result_np = torch.stack(result, axis = 0).cpu().numpy()
          if strategy_mean == 'starts':
             result_var = np.var(result_np, axis = 1, ddof=1).mean(axis = 0).mean()
             result_mean = np.mean(result_np, axis = 1).mean(axis = 0).mean()
             diff = (result_np_1 == result_np_2).sum(axis = 1)
             num_new = (diff != dim).sum()
             ess = num_new/diff.shape[0]
              
          elif strategy_mean == 'chain':
             result_var = np.var(result_np, axis = 0, ddof=1).mean(axis = 0).mean()
             result_mean = np.mean(result_np, axis = 0).mean(axis = 0).mean()
          
          else:
             raise ValueError('Unknown method of mean') 
             
          result_np_1 = result_np[:-1]
          result_np_2 = result_np[1:]
          diff = (result_np_1 == result_np_2).sum(axis = 2)
          ess_bs = (diff != dim).mean(axis = 0)
          ess = ess_bs.mean()
          first_coord_history = result_np[:, :, 0]
          norm_history = np.linalg.norm(result_np, axis = -1)
          
          if print_results:
             print(f"mean estimation of variance = {result_var}")
             print(f"mean estimation of mean = {result_mean}")
             print(f"mean estimation of ess = {ess}")
             print("------")
          dict_results[mode_init]['mean_loc'].append(result_mean)
          dict_results[mode_init]['mean_var'].append(result_var)
          dict_results[mode_init]['ess'].append(ess)
          dict_results[mode_init]['history_first'].append(first_coord_history)
          dict_results[mode_init]['history_norm'].append(norm_history)
    
    return dict_results
