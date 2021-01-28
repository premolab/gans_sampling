import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
import random
import sklearn

from .distributions import (Target, 
                           Gaussian_mixture, 
                           IndependentNormal,
                           init_independent_normal)



from general_utils import DotDict
from metrics import Evolution
from ebm_sampling import grad_energy

def compute_sir_log_weights(x, target, proposal):
    return target.log_prob(x) -  proposal.log_prob(x)

def sir_independent_dynamics(z, target, proposal, n_steps, N):
    z_sp = []
    batch_size, z_dim = z.shape[0], z.shape[1]

    for _ in range(n_steps):
        z_sp.append(z)
        U = torch.randint(0, N, (batch_size,)).tolist()
        X = proposal.sample([batch_size, N])
        X[np.arange(batch_size), U, :] = z
        X_view = X.view(-1, z_dim)

        log_weight = compute_sir_log_weights(X_view, target, proposal)
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

    z_sp.append(z)
    return z_sp
    
def sir_correlated_dynamics(z, target, proposal, n_steps, N, alpha):
    z_sp = []
    batch_size, z_dim = z.shape[0], z.shape[1]

    for _ in range(n_steps):
        z_sp.append(z)
        z_copy = z.unsqueeze(1).repeat(1, N, 1)
        ind = torch.randint(0, N, (batch_size,)).tolist()
        W = proposal.sample([batch_size, N])
        U = proposal.sample([batch_size]).unsqueeze(1).repeat(1, N, 1)
        #print(W.shape, U.shape, z_copy.shape)
        X = torch.zeros((batch_size, N, z_dim), dtype = z.dtype).to(z.device)
        X =  (alpha**2)*z_copy + alpha*((1- alpha**2)**0.5)*U + W*((1- alpha**2)**0.5)
        X[np.arange(batch_size), ind, :] = z
        X_view = X.view(-1, z_dim)

        log_weight = compute_sir_log_weights(X_view, target, proposal)
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

    z_sp.append(z)
    return z_sp

def compute_log_probs(point1, point2, target, proposal, beta, grad_step, eps_scale):
    E_point1, grad_point1 = grad_energy(point1, target, x=None)
    E_point2, grad_point2 = grad_energy(point2, target, x=None)
    
    energy_part = beta*(E_point1 - E_point2)
    
    propose_vec_1 = point1 - (1. - grad_step)*point2 + grad_step*beta*point2
    propose_vec_2 = point2 - (1. - grad_step)*point1 + grad_step*beta*grad_point1
    
    propose_part_1 = proposal.log_prob(propose_vec_1/eps_scale)
    propose_part_2 = proposal.log_prob(propose_vec_2/eps_scale)
    propose_part = propose_part_1 - propose_part_2
    
    log_probs = propose_part + energy_part
    return (E_point1.detach().clone(), 
            grad_point1.detach().clone(), 
            E_point2.detach().clone(), 
            grad_point2.detach().clone(), 
            log_probs.detach().clone())

def compute_probs_from_log_probs(log_probs):
    mask_zeros = log_probs > 0.
    num_big_zeros = mask_zeros.sum().item()
    log_probs[mask_zeros] = torch.zeros(num_big_zeros, 
                                        dtype = log_probs.dtype).to(log_probs.device)
    probs = log_probs.exp()
    return probs

def ais_vanilla(z, target, proposal, n_steps, p0, p, N, betas, grad_step, eps_scale):
    z_sp = []
    batch_size, T, z_dim = z.shape[0], z.shape[1], z.shape[2]
    T = T - 1
    uniform = Uniform(low = 0.0, high = 1.0)
    v = torch.zeros((batch_size, T + 1, N, z_dim), dtype = z.dtype).to(z.device)
    u = uniform.sample((batch_size, T + 1, N)).to(z.device)
    Z = torch.zeros((batch_size, T + 1, N, z_dim), dtype = z.dtype).to(z.device)
    for _ in range(n_steps):
        z_sp.append(z.detach().clone())
        W = proposal.sample([batch_size, N - 1])
        Z[:, :, 0, :] = z ###Set current particle at idx 0
        Z[:, 0, 1:, :] = W ###Random initial points
        W_2 = proposal.sample([batch_size, T, N - 1]) ##Innovation noise during all trjacetories except current one
        for t in range(1, T + 1):
            #print(f"t = {t}")
            v[:, t, 1:, :] = W_2[:, t - 1, :, :]
            z_t_1_j_shape = Z[:, t - 1, 1:, :].shape
            z_t_1_j_flatten = Z[:, t - 1, 1:, :].reshape(-1, z_dim).detach().clone()
            z_t_1_j_flatten.requires_grad_(True)
            
            _, grad_z_t_1_j_flatten = grad_energy(z_t_1_j_flatten, target, x=None)
            grad_z_t_1_j = grad_z_t_1_j_flatten.reshape(z_t_1_j_shape)
            p_t_j = (1. - grad_step)*Z[:, t - 1, 1:, :] - grad_step*betas[t]*grad_z_t_1_j \
                                                    + eps_scale*v[:, t, 1:, :]
            Z_t_1_j = Z[:, t - 1, 1:, :]
            Z_t_1_j_shape = Z_t_1_j.shape
            
            p_t_j_flatten = p_t_j.view(-1, z_dim).detach().clone()
            p_t_j_flatten.requires_grad_(True)
            
            Z_t_1_j_flatten = Z_t_1_j.reshape(-1, z_dim).detach().clone()
            Z_t_1_j_flatten.requires_grad_(True)
            
            _, _, _, _, log_probs = compute_log_probs(Z_t_1_j_flatten, 
                                                      p_t_j_flatten, 
                                                      target, 
                                                      proposal, 
                                                      betas[t], 
                                                      grad_step, 
                                                      eps_scale)
            probs_flatten = compute_probs_from_log_probs(log_probs)
            probs = probs_flatten.view(batch_size, N - 1)
            u_t_1 = u[:, t, 0].unsqueeze(1).repeat(1, N - 1)
            
            mask_leq = (u_t_1 <= probs)
            mask_ge = ~mask_leq
            
            mask_leq_big = mask_leq.unsqueeze(-1).repeat(1, 1, z_dim)
            mask_ge_big = mask_ge.unsqueeze(-1).repeat(1, 1, z_dim)
            
            Z[:, t, 1:, :][mask_leq_big] = p_t_j[mask_leq_big]
            Z[:, t, 1:, :][mask_ge_big] = Z[:, t - 1, 1:, :][mask_ge_big]
        
        log_weights = torch.zeros((T, batch_size, N), dtype = z.dtype).to(z.device)
            
        for t in range(1, T + 1):
            cur_z = Z[:, t - 1, :, :]
            
            z_flatten = cur_z.reshape(-1, z_dim)
            E_flatten = -target.log_prob(z_flatten)
            
            E = E_flatten.reshape((batch_size, N))
            
            log_weights[t - 1, :, :] = -(betas[t] - betas[t - 1])*E
        
        log_weights = log_weights.sum(axis = 0)
        max_logs = torch.max(log_weights, dim = 1)[0].unsqueeze(-1).repeat((1, N))
        log_weights = log_weights - max_logs
        sum_weights = torch.logsumexp(log_weights, dim = 1)
        log_weights = log_weights- sum_weights[:, None]
        weights = log_weights.exp()
        weights[weights != weights] = 0.
        weights[weights.sum(1) == 0.] = 1.
        
        indices = torch.multinomial(weights, 1).squeeze().tolist()
        z[:, :, :] = Z[np.arange(batch_size), :, indices, :]
        
        z_sp.append(z.detach().clone()) 
    return z_sp



def ais_dynamics(z, target, proposal, n_steps, N, betas, rhos, grad_step, eps_scale):
    #z - tensor (bs, T + 1, dim)
    z_sp = []
    batch_size, T, z_dim = z.shape[0], z.shape[1], z.shape[2]
    T = T - 1
    uniform = Uniform(low = 0.0, high = 1.0)
    
    for _ in range(n_steps):
        #print(f"iter = {_}")
        z_sp.append(z.detach().clone())
        v = torch.zeros((batch_size, T + 1, N, z_dim), dtype = z.dtype).to(z.device)
        u = torch.zeros((batch_size, T + 1, N), dtype = z.dtype).to(z.device)
        #print("step1")
        #step 1
        kappa = torch.zeros((batch_size, T + 1, z_dim), dtype = z.dtype).to(z.device)
        
        kappa_t_noise = proposal.sample([batch_size, T + 1])
        
        kappa[:, 0, :] = rhos[0]*z[:, 0, :] + ((1 - rhos[0]**2)**0.5) * kappa_t_noise[:, 0, :]
        
        for t in range(1, T + 1):
            #print(f"t = {t}")
            not_equal_mask = (torch.norm(z[:, t, :] - z[:, t - 1, :], p=2, dim=-1) > 1e-13)
            equal_mask = ~not_equal_mask             
            
            num_not_equal = not_equal_mask.sum()
            num_equal = equal_mask.sum()
            #print(f"num_not_equal = {num_not_equal}")
            #print(f"num_equal = {num_equal}")
            
            if num_not_equal > 0:
                #print(f"start to do updates for not equal batches")
                z_t_not_equal = z[not_equal_mask, t, :].detach().clone()
                z_t_1_not_equal = z[not_equal_mask, t - 1, :].detach().clone()
                #print(z_t_not_equal.shape)
                z_t_not_equal.requires_grad_(True)
                z_t_1_not_equal.requires_grad_(True)
                
                E_point1, grad_point1, E_point2, grad_point2, log_probs = compute_log_probs(z_t_1_not_equal, 
                                                                                            z_t_not_equal, 
                                                                                            target, 
                                                                                            proposal, 
                                                                                            betas[t], 
                                                                                            grad_step, 
                                                                                            eps_scale)

                v[not_equal_mask, t, 0, :] = (z_t_not_equal - (1. - grad_step)*z_t_1_not_equal \
                                                            + grad_step*betas[t]*grad_point1)/eps_scale
                #print(log_probs.shape)
                probs = compute_probs_from_log_probs(log_probs)
                generate_uniform_var = uniform.sample([probs.shape[0]]).to(probs.device)
                weight_uniform_var = generate_uniform_var * probs

                u[not_equal_mask, t, 0] = weight_uniform_var
            
            if num_equal > 0:
                #print(f"start to do updates for equal batches")
                z_t_equal = z[equal_mask, t, :]
                z_t_1_equal = z[equal_mask, t - 1, :].detach().clone()
                z_t_1_equal.requires_grad_(True)
                
                E_t_1_equal, grad_t_1_equal = grad_energy(z_t_1_equal, target, x=None)
                
                second_part_no_noise = (1. - grad_step)*z_t_1_equal - grad_step*betas[t]*grad_t_1_equal

                stop = False
                num_updates = 0
                update_mask = torch.zeros(num_equal, dtype = torch.bool).to(z_t_equal.device)
                #print(f"update_mask.sum() = {updates_num}")
                z_t_1_equal = z_t_1_equal.detach().clone()
                z_t_1_equal.requires_grad_(True)
                #print("start to while sampling")
                while not stop:
                    cur_u = uniform.sample([num_equal]).to(z_t_equal.device)
                    cur_v = proposal.sample([num_equal]).to(z_t_equal.device)
                    second_part = second_part_no_noise + cur_v*eps_scale
                    second_part = second_part.detach().clone()
                    second_part.requires_grad_(True)
                    
                    _, _, _, _, log_probs = compute_log_probs(z_t_1_equal, 
                                                              second_part, 
                                                              target, 
                                                              proposal, 
                                                              betas[t], 
                                                              grad_step, 
                                                              eps_scale)
                    probs = compute_probs_from_log_probs(log_probs)
                    mask_assign = (cur_u > probs)
                    new_assign = torch.logical_and(mask_assign, ~update_mask) 
                    #print(new_assign.shape)
                    #print(u.shape)
                    #print(cur_u.shape)
                    
                    u[equal_mask, t, 0][new_assign] = cur_u[new_assign]
                    v[equal_mask, t, 0, :][new_assign] = cur_v[new_assign]
                    
                    
                    update_mask = torch.logical_or(update_mask, new_assign)
                    updates_num = update_mask.sum()
                    
                    #print(f"update_mask = {updates_num}")
                    
                    
                    if updates_num == num_equal:
                        stop = True
            
            kappa[:, t, :] = rhos[t]*v[:, t, 0, :] + ((1 - rhos[t]**2)**0.5) * kappa_t_noise[:, t, :]
        #print("end step 1")   
        #print("step2")
        #step 2
        W = proposal.sample([batch_size, N - 1])
        #Z - tensor (bs, T + 1, N, dim)
        Z = torch.zeros((batch_size, T + 1, N, z_dim), dtype = z.dtype).to(z.device)
        kappa_repeat = kappa[:, 0, :].unsqueeze(1).repeat(1, N - 1, 1)
        kappa_N_noise = proposal.sample([batch_size, N - 1])
        #print(z_repeat.shape)
        #print(W.shape)
        Z[:, :, 0, :] = z
        Z[:, 0, 1:, :] = rhos[0]*kappa_repeat + ((1 - rhos[0]**2)**0.5) * kappa_N_noise
        
        kappa_repeat_N = kappa.unsqueeze(2).repeat(1, 1, N - 1, 1)
        W_2 = proposal.sample([batch_size, T, N - 1])
        
        for t in range(1, T + 1):
            #print(f"t = {t}")
            v[:, t, 1:, :] = rhos[t]*kappa_repeat_N[:, t - 1, :, :] + ((1 - rhos[t]**2)**0.5) * W_2[:, t - 1, :, :]
            z_t_1_j_shape = Z[:, t - 1, 1:, :].shape
            z_t_1_j_flatten = Z[:, t - 1, 1:, :].reshape(-1, z_dim).detach().clone()
            z_t_1_j_flatten.requires_grad_(True)
            
            _, grad_z_t_1_j_flatten = grad_energy(z_t_1_j_flatten, target, x=None)
            grad_z_t_1_j = grad_z_t_1_j_flatten.reshape(z_t_1_j_shape)
            p_t_j = (1. - grad_step)*Z[:, t - 1, 1:, :] - grad_step*betas[t]*grad_z_t_1_j \
                                                    + eps_scale*v[:, t, 1:, :]
            Z_t_1_j = Z[:, t - 1, 1:, :]
            Z_t_1_j_shape = Z_t_1_j.shape
            
            p_t_j_flatten = p_t_j.view(-1, z_dim).detach().clone()
            p_t_j_flatten.requires_grad_(True)
            
            Z_t_1_j_flatten = Z_t_1_j.reshape(-1, z_dim).detach().clone()
            Z_t_1_j_flatten.requires_grad_(True)
            
            _, _, _, _, log_probs = compute_log_probs(Z_t_1_j_flatten, 
                                                      p_t_j_flatten, 
                                                      target, 
                                                      proposal, 
                                                      betas[t], 
                                                      grad_step, 
                                                      eps_scale)
            probs_flatten = compute_probs_from_log_probs(log_probs)
            probs = probs_flatten.view(batch_size, N - 1)
            u_t_1 = u[:, t, 0].unsqueeze(1).repeat(1, N - 1)
            
            mask_leq = (u_t_1 <= probs)
            mask_ge = ~mask_leq
            
            mask_leq_big = mask_leq.unsqueeze(-1).repeat(1, 1, z_dim)
            mask_ge_big = mask_ge.unsqueeze(-1).repeat(1, 1, z_dim)
            
            Z[:, t, 1:, :][mask_leq_big] = p_t_j[mask_leq_big]
            Z[:, t, 1:, :][mask_ge_big] = Z[:, t - 1, 1:, :][mask_ge_big]
            
        #Z[:, :, 0, :] = z
        #print("end step2")
        #print("start step3")
        log_weights = torch.zeros((T, batch_size, N), dtype = z.dtype).to(z.device)
        for t in range(1, T + 1):
            cur_z = Z[:, t - 1, :, :]
            
            z_flatten = cur_z.reshape(-1, z_dim)
            E_flatten = -target.log_prob(z_flatten)
            
            E = E_flatten.reshape((batch_size, N))
            
            log_weights[t - 1, :, :] = -(betas[t] - betas[t - 1])*E
            
        
        log_weights = log_weights.sum(axis = 0)
        max_logs = torch.max(log_weights, dim = 1)[0].unsqueeze(-1).repeat((1, N))
        log_weights = log_weights - max_logs
        weights = torch.exp(log_weights)
        sum_weights = torch.sum(weights, dim = 1)
        weights = weights/sum_weights[:, None]

        weights[weights != weights] = 0.
        weights[weights.sum(1) == 0.] = 1.
        
        indices = torch.multinomial(weights, 1).squeeze().tolist()
        z[:, :, :] = Z[np.arange(batch_size), :, indices, :]
        
        #print("end step4")
        
    z_sp.append(z.detach().clone()) 
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
                              mode_init='proposal',
                              method='sir_independent',
                              print_results=True):
    dict_results = {mode_init: {'mean_loc': [], 'mean_var': [], 
                               'ess': [], 'history_first': [], 
                                          'history_norm': []}}
   
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
    
       if (mode_init == 'target') and (method != 'ais'):
          start = target.sample([batch_size])
       elif mode_init == 'proposal' and (method != 'ais'):
          start = proposal.sample([batch_size])
       elif (mode_init == 'target') and (method == 'ais'):
        
          start = target.sample([batch_size, len(method_params['betas'])])
       elif mode_init == 'proposal' and (method == 'ais'):
          start = proposal.sample([batch_size, len(method_params['betas'])])
       else:
          raise ValueError('Unknown initialization method')
       if method == 'sir_correlated':
          alpha = (1 - method_params['c']/dim)**0.5
          history = sir_correlated_dynamics(start, 
                                            target,
                                            proposal, 
                                            method_params['n_steps'], 
                                            method_params['N'],
                                            alpha)
       elif method == 'sir_independent':
          history = sir_independent_dynamics(start, 
                                             target,
                                             proposal, 
                                             method_params['n_steps'], 
                                             method_params['N'])
       elif method == 'ais':
          history = ais_dynamics(start, 
                                 target,
                                 proposal, 
                                 method_params['n_steps'], 
                                 method_params['N'], 
                                 method_params['betas'], 
                                 method_params['rhos'], 
                                 method_params['grad_step'], 
                                 method_params['eps_scale'])
          history = [history[i][:, -1, :] for i in range(len(history))]
          
       else:
          raise ValueError('Unknown sampling method')    
       last_history = history[(-num_points_in_chain - 1):-1]
       all_history_np = torch.stack(history, axis = 0).cpu().numpy()

       result_np = torch.stack(last_history, axis = 0).cpu().numpy()
       if strategy_mean == 'starts':
          result_var = np.var(result_np, axis = 1, ddof=1).mean(axis = 0).mean()
          result_mean = np.mean(result_np, axis = 1).mean(axis = 0).mean()
              
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
       first_coord_history = all_history_np[:, :, 0]
       norm_history = np.linalg.norm(all_history_np, axis = -1)
          
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

def run_experiments_2_gaussians(dim_arr,  
                                scale_proposal, 
                                scale_target,
                                loc_1_target,
                                loc_2_target,
                                num_points_in_chain, 
                                strategy_mean,
                                device,
                                batch_size,
                                method_params,
                                random_seed=42,
                                mode_init='proposal',
                                method='sir_independent',
                                print_results=True):
    dict_results = {mode_init: {'mean_loc_1': [],
                                'mean_loc_2': [],
                                'mean_var': [], 
                                'mean_jsd': [],
                                'mean_hqr': [],
                                'ess': [], 
                                'history_first': [],            
                                'history_norm': []}}
    
    if print_results:
       print("------------------")
       print(f"mode = {mode_init}")
    
    for dim in dim_arr:
       if print_results:
          print(f"dim = {dim}")
          
       target_args = DotDict()
       target_args.device = device
       target_args.num_gauss = 2

       coef_gaussian = 1./target_args.num_gauss
       target_args.p_gaussians = [torch.tensor(coef_gaussian)]*target_args.num_gauss
       locs = [loc_1_target*torch.ones(dim, dtype = torch.float64).to(device),
               loc_2_target*torch.ones(dim, dtype = torch.float64).to(device)]
       locs_numpy = torch.stack(locs, axis = 0).cpu().numpy()
       target_args.locs = locs
       target_args.covs = [(scale_target**2)*torch.eye(dim, 
                                                       dtype = torch.float64).to(device)]*target_args.num_gauss
       target_args.dim = dim
       target = Gaussian_mixture(target_args)
       proposal = init_independent_normal(scale_proposal, dim, device)
       torch.manual_seed(random_seed)
       np.random.seed(random_seed)
       random.seed(random_seed)
       if mode_init == 'target':
          dataset = sklearn.datasets.make_blobs(n_samples = batch_size, 
                                                n_features = dim, 
                                                centers = locs_numpy, 
                                                cluster_std = scale_target,
                                                random_state = random_seed)[0]
          start = torch.FloatTensor(dataset).to(device)
       elif mode_init == 'proposal':
          start = proposal.sample([batch_size])
          #print(f"start = {start}")
       else:
          raise ValueError('Unknown initialization method')
       if method == 'sir_correlated':
          alpha = (1 - method_params['c']/dim)**0.5
          history = sir_correlated_dynamics(start, 
                                            target,
                                            proposal, 
                                            method_params['n_steps'], 
                                            method_params['N'],
                                            alpha)
       elif method == 'sir_independent':
          history = sir_independent_dynamics(start, 
                                             target,
                                             proposal, 
                                             method_params['n_steps'], 
                                             method_params['N'])
       else:
          raise ValueError('Unknown sampling method')    
       last_history = history[(-num_points_in_chain - 1):-1]
       all_history_np = torch.stack(history, axis = 0).cpu().numpy()
       torch_last_history = torch.stack(last_history, axis = 0).cpu()
          
       evolution = Evolution(None, locs=torch.stack(locs, 0).cpu(), sigma=scale_target)  

       result_np = torch.stack(last_history, axis = 0).cpu().numpy()
            
       modes_var_arr = []
       modes_mean_arr = []
       h_q_r_arr = []
       jsd_arr = []
       means_est_1 = torch.zeros(dim)
       means_est_2 = torch.zeros(dim)
       num_found_1_mode = 0
       num_found_2_mode = 0  
            
       if strategy_mean == 'starts':
            
          for i in range(num_points_in_chain):
             X_gen = torch_last_history[i, :, :]
             assignment = Evolution.make_assignment(X_gen, evolution.locs, evolution.sigma)
             mode_var = Evolution.compute_mode_std(X_gen, assignment).item()**2
             modes_mean, found_modes_ind = Evolution.compute_mode_mean(X_gen, assignment)
             if 0 in found_modes_ind:
                num_found_1_mode += 1
                means_est_1 += modes_mean[0]
             if 1 in found_modes_ind:
                num_found_2_mode += 1
                means_est_2 += modes_mean[1]
                
             h_q_r = Evolution.compute_high_quality_rate(assignment).item()
             jsd = Evolution.compute_jsd(assignment).item()
                
             modes_var_arr.append(mode_var)
             modes_mean_arr.append(modes_mean)
             h_q_r_arr.append(h_q_r)
             jsd_arr.append(jsd)

       elif strategy_mean == 'chain':            
             #print(evolution.locs)
          for i in range(batch_size):
             X_gen = torch_last_history[:, i, :]
             assignment = Evolution.make_assignment(X_gen, evolution.locs, evolution.sigma)
             mode_var = Evolution.compute_mode_std(X_gen, assignment).item()**2
                
             modes_mean, found_modes_ind = Evolution.compute_mode_mean(X_gen, assignment)
             #print(f"found_modes_ind = {found_modes_ind}")
             if 0 in found_modes_ind:
                num_found_1_mode += 1
                means_est_1 += modes_mean[0]
             if 1 in found_modes_ind:
                num_found_2_mode += 1
                means_est_2 += modes_mean[1]
             #print(f"batch = {i}, modes_mean = {modes_mean}")
             h_q_r = Evolution.compute_high_quality_rate(assignment).item()
             jsd = Evolution.compute_jsd(assignment).item()
                
             modes_var_arr.append(mode_var)

             h_q_r_arr.append(h_q_r)
             jsd_arr.append(jsd)     

       else:
          raise ValueError('Unknown method of mean') 
          
       jsd_result = np.array(jsd_arr).mean()
       modes_var_result = np.array(modes_var_arr).mean()
       h_q_r_result = np.array(h_q_r_arr).mean()
       if num_found_1_mode == 0:
          print("Unfortunalely, no points were assigned to 1st mode, default estimation - zero")
          modes_mean_1_result = 0.0
       else:
          modes_mean_1_result = (means_est_1/num_found_1_mode).mean().item()
       if num_found_2_mode == 0:
          print("Unfortunalely, no points were assigned to 2nd mode, default estimation - zero")
          modes_mean_2_result = 0.0
       else:
          modes_mean_2_result = (means_est_2/num_found_2_mode).mean().item()
            
       result_np_1 = result_np[:-1]
       result_np_2 = result_np[1:]
       diff = (result_np_1 == result_np_2).sum(axis = 2)
       ess_bs = (diff != dim).mean(axis = 0)
       ess = ess_bs.mean()
       first_coord_history = all_history_np[:, :, 0]
       norm_history = np.linalg.norm(all_history_np, axis = -1)

       if print_results:
          print(f"mean estimation of target variance = {modes_var_result}")
          print(f"mean estimation of 1 mode mean  = {modes_mean_1_result}")
          print(f"mean estimation of 2 mode mean  = {modes_mean_2_result}")
          print(f"mean estimation of JSD  = {jsd_result}")
          print(f"mean estimation of HQR  = {h_q_r_result}")
          print(f"mean estimation of ESS = {ess}")
          print("------")
       dict_results[mode_init]['mean_loc_1'].append(modes_mean_1_result)
       dict_results[mode_init]['mean_loc_2'].append(modes_mean_2_result)
       dict_results[mode_init]['mean_var'].append(modes_var_result)
       dict_results[mode_init]['mean_jsd'].append(jsd_result)
       dict_results[mode_init]['mean_hqr'].append(h_q_r_result)
       dict_results[mode_init]['ess'].append(ess)
       dict_results[mode_init]['history_first'].append(first_coord_history)
       dict_results[mode_init]['history_norm'].append(norm_history)
    
    return dict_results
