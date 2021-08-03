from os import X_OK
from typing import Callable, Tuple
import numpy as np
import torch
from scipy.stats import gamma, invgamma
from tqdm import tqdm, trange
import copy

from .mcmc_base import AbstractMCMC, increment_steps, adapt_stepsize_dec
from .ebm_sampling import MALATransition, grad_energy
from .adaptive_sir_loss import get_loss


class CorrelatedKernel:
    def __init__(self, corr_coef=0, bernoulli_prob_corr=0):
        self.corr_coef = corr_coef
        self.bern = torch.distributions.Bernoulli(bernoulli_prob_corr)

    def __call__(self, z, proposal, N, batch_size=1):
        correlation = self.corr_coef * self.bern.sample((batch_size,))
        #latent_var = correlation[:, None] * z_backward_pushed + (1. - correlation[:, None]**2)**.5 * proposal.sample((batch_size,)) #torch.randn((batch_size, z_dim))
        latent_var = correlation[:, None] * z + (1. - correlation[:, None]**2)**.5 * proposal.sample((batch_size,)) #torch.randn((batch_size, z_dim))
        correlation_new = self.corr_coef * self.bern.sample((batch_size, N))
        z_new = correlation_new[..., None] * latent_var[:, None, :] + (1. - correlation_new[..., None]**2)**.5 * proposal.sample((batch_size, N,)) #torch.randn((batch_size, N - 1, z_dim))
        return z_new


def compute_sir_log_weights(x, target, proposal, flow):
    x_pushed, log_jac = flow(x)
    log_weights = target(x_pushed) + log_jac - proposal(x)
    return log_weights, x_pushed


def adaptive_sir_correlated_dynamics(z, target, proposal, n_steps, N, corr_coef=0., bernoulli_prob_corr=0., flow=None, verbose=True): # z assumed from proposal !
    z_sp = [] ###vector of samples
    batch_size, z_dim = z.shape[0], z.shape[1]
    acceptance = torch.zeros(batch_size)

    range_gen = trange if verbose else range

    corr_ker = CorrelatedKernel(corr_coef, bernoulli_prob_corr)

    for _ in range_gen(n_steps):
        if flow is not None:
           z_pushed, _ = flow(z)
        else:
            z_pushed = z

        z_sp.append(z_pushed)
        #z_copy = z.unsqueeze(1).repeat(1, N, 1)
        #W = proposal.sample([batch_size, N])
        #U = proposal.sample([batch_size]).unsqueeze(1).repeat(1, N, 1)

        #X = torch.zeros((batch_size, N, z_dim), dtype = z.dtype).to(z.device)
        X = corr_ker(z, proposal, N, batch_size)
        #X =  (alpha**2)*z_copy + alpha*((1- alpha**2)**0.5)*U + W*((1- alpha**2)**0.5)
        X[np.arange(batch_size), [0] * batch_size, :] = z
        X_view = X.view(-1, z_dim)

        if flow is not None:
            log_weight, z_pushed = compute_sir_log_weights(X_view, target, proposal, flow)
        else:
            z_pushed = X_view
            log_weight = target(z_pushed) - proposal(z_pushed)
            
        log_weight = log_weight.view(batch_size, N)
        max_logs = torch.max(log_weight, dim = 1)[0][:, None]
        log_weight = log_weight - max_logs
        weight = torch.exp(log_weight)
        sum_weight = torch.sum(weight, dim = 1)
        weight = weight/sum_weight[:, None]        

        weight[weight != weight] = 0.
        weight[weight.sum(1) == 0.] = 1.

        indices = torch.multinomial(weight, 1).squeeze().tolist()
        mask = (indices == 0)
        acceptance += mask

        z = X[np.arange(batch_size), indices, :]
        
    if flow is not None:
        z_pushed, _ = flow(z)
    else:
        z_pushed = z

    z_sp.append(z_pushed)
    acceptance /= n_steps

    return z_sp, acceptance


class CISIR(AbstractMCMC):
    def __init__(self,
                N=2,
                corr_coef=0., 
                bernoulli_prob_corr=0., 
                flow=None,
                verbose=True,
                **kwargs):
        super().__init__()
        self.N = N
        self.corr_coef = corr_coef 
        self.bernoulli_prob_corr = bernoulli_prob_corr
        self.flow = flow
        self.verbose = verbose
        self.n_steps = kwargs.get('n_steps', 1) # 

    #@increment_steps
    #@adapt_stepsize_dec
    def __call__(self, start : torch.Tensor, target, proposal, n_steps=None, N=None, corr_coef=None, bernoulli_prob_corr=None, flow=None, verbose=None):
        N = self.N if N is None else N
        corr_coef = self.corr_coef if corr_coef is None else corr_coef
        bernoulli_prob_corr = self.bernoulli_prob_corr if bernoulli_prob_corr is None else bernoulli_prob_corr
        flow = self.flow if flow is None else flow
        n_steps = self.n_steps if n_steps is None else n_steps
        verbose = self.verbose if verbose is None else verbose

        return adaptive_sir_correlated_dynamics(start, target, proposal, n_steps, N, corr_coef, bernoulli_prob_corr, flow, verbose)


def ex2_mcmc_mala(z, 
                target, 
                proposal, 
                n_steps, 
                N,
                grad_step,
                noise_scale, 
                mala_steps = 5,
                corr_coef=0., 
                bernoulli_prob_corr=0., 
                flow=None,
                adapt_stepsize=True,
                verbose=False):
    z_sp = [] ###vector of samples
    batch_size, z_dim = z.shape[0], z.shape[1]

    mala_transition = MALATransition(z_dim, z.device)
    mala_transition.adapt_grad_step = grad_step
    mala_transition.adapt_sigma = noise_scale

    corr_ker = CorrelatedKernel(corr_coef, bernoulli_prob_corr)
    
    acceptance = torch.zeros(batch_size)
    range_gen = trange if verbose else range

    if flow is not None:
        z_pushed, log_jac = flow(z)
    else:
        z_pushed = z

    for step_id in range_gen(n_steps):
        z_sp.append(z_pushed)

        if corr_coef == 0 and bernoulli_prob_corr == 0: # isir
            #z_new = proposal.sample([batch_size, N - 1])
            X = proposal.sample([batch_size, N])
        else:
            X = corr_ker(z, proposal, N, batch_size)
            # for simplicity assume proposal is N(0, \sigma^2 Id), need to be updated
            # correlation = corr_coef * bern.sample((batch_size,))

            # #latent_var = correlation[:, None] * z_backward_pushed + (1. - correlation[:, None]**2)**.5 * proposal.sample((batch_size,)) #torch.randn((batch_size, z_dim))
            # latent_var = correlation[:, None] * z + (1. - correlation[:, None]**2)**.5 * proposal.sample((batch_size,)) #torch.randn((batch_size, z_dim))
            # correlation_new = corr_coef * bern.sample((batch_size, N))
            # z_new = correlation_new[..., None] * latent_var[:, None, :] + (1. - correlation_new[..., None]**2)**.5 * proposal.sample((batch_size, N,)) #torch.randn((batch_size, N - 1, z_dim))
        
        #if flow is not None:
        #    z_new, log_jac_new = flow(z_new.view(batch_size * (N - 1), -1))
        #    z_new = z_new.view(batch_size, N - 1, -1)
            #log_jacs  = torch.cat([log_jac[:, None], log_jac_new.view(batch_size, N - 1)], 1)
        #else:
            #log_jacs = torch.zeros(batch_size, N)

        #X = torch.zeros((batch_size, N, z_dim), dtype = z.dtype).to(z.device)
        #X = z_new
        ind = [0] * batch_size
        X[np.arange(batch_size), ind, :] = z

        # X = torch.cat([z.unsqueeze(1), z_new], 1)
        
        X_view = X.view(-1, z_dim)
        
        # if flow is not None:
        #     x_pushed, log_jacs = flow(X_view)
        # else:
        #     x_pushed = X_view
        #     log_jacs = torch.zeros(batch_size * N)

        #X = torch.cat([z_pushed.unsqueeze(1), z_new], 1)
        #X_view = X.view(-1, z_dim)

        if flow is not None:
            log_weight, z_pushed = compute_sir_log_weights(X_view, target, proposal, flow)
        else:
            z_pushed = X_view
            log_weight = target(X_view) - proposal(X_view)

        log_weight = log_weight.view(batch_size, N)
        z_pushed = z_pushed.view(X.shape)
        
        max_logs = torch.max(log_weight, dim = 1)[0][:, None]
        log_weight = log_weight - max_logs
        weight = torch.exp(log_weight)
        sum_weight = torch.sum(weight, dim = 1)
        weight = weight / sum_weight[:, None]        

        weight[weight != weight] = 0.
        weight[weight.sum(1) == 0.] = 1.

        indices = torch.multinomial(weight, 1).squeeze().tolist()
        #z = X[np.arange(batch_size), indices, :]
        z_pushed = z_pushed[np.arange(batch_size), indices, :]

        #if flow is not None:
        #    log_jac = log_jacs[np.arange(batch_size), indices]

        # mala transition
        for _ in range(mala_steps):
            E, grad = grad_energy(z_pushed, target)
            z_pushed, _, _, mask = mala_transition(z_pushed, E, grad, target=target, adapt_stepsize=adapt_stepsize)
            acceptance += mask.float() / mala_steps

        if step_id != n_steps - 1:
            if flow is not None:
                z, _ = flow.inverse(z_pushed)
            else:
                z = z_pushed

    z_sp.append(z_pushed)
    acceptance /= n_steps

    return z_sp, acceptance, mala_transition.adapt_grad_step


class Ex2MCMC(AbstractMCMC):
    def __init__(self,
                N=2,
                grad_step=1e-2,
                noise_scale=(2e-2)**.5, 
                mala_steps=5,
                corr_coef=0., 
                bernoulli_prob_corr=0., 
                flow=None,
                adapt_stepsize=False,
                verbose=True,
                **kwargs):
        super().__init__()
        self.N = N
        self.grad_step = grad_step
        self.noise_scale = noise_scale
        self.mala_steps = mala_steps
        self.corr_coef = corr_coef 
        self.bernoulli_prob_corr = bernoulli_prob_corr
        self.flow = flow
        self.adapt_stepsize = adapt_stepsize
        self.verbose = verbose
        self.n_steps = kwargs.get('n_steps', 1) # 

    @increment_steps
    @adapt_stepsize_dec
    def __call__(self, start : torch.Tensor, target, proposal, *args, **kwargs):
        self_kwargs = copy.copy(self.__dict__)
        self_kwargs.update(kwargs)

        n_steps = self_kwargs.pop('n_steps')
        if len(args) > 0:
            n_steps = args[0]

        self_kwargs.pop('_steps_done')

        return ex2_mcmc_mala(start, target, proposal, n_steps, **self_kwargs)


class FlowMCMC:
    def __init__(self, target, proposal, flow, mcmc_call : callable, **kwargs):
        self.flow = flow
        self.proposal = proposal
        self.target = target
        self.batch_size = kwargs.get('batch_size', 64)
        self.mcmc_call = mcmc_call
        self.grad_clip = kwargs.get('grad_clip', 1.)
        optimizer = kwargs.get('optimizer', 'adam')
        loss = kwargs.get('loss', 'mix_kl')

        if isinstance(loss, Callable):
            self.loss = loss
        elif isinstance(loss, str):
            self.loss = get_loss(loss)
        else:
            ValueError

        lr = kwargs.get('lr', 1e-3)
        wd = kwargs.get('wd', 1e-4)
        if isinstance(optimizer, torch.optim.Optimizer):
            self.optimizer = optimizer
        elif isinstance(optimizer, str):
            if optimizer.lower() == 'adam':
                self.optimizer = torch.optim.Adam(flow.parameters(), lr=lr, weight_decay=wd)

    def train_step(self, inp=None, alpha=0.5, do_step=True):
        if inp is None:
            inp = self.proposal.sample((self.batch_size,))
        else:
            inp, _ = self.flow.inverse(inp)

        out = self.mcmc_call(inp, self.target, self.proposal, flow=self.flow)
        if isinstance(out, Tuple):
            acc_rate = out[1].mean()
            out = out[0]
        else:
            acc_rate = 1
        out = out[-1]
        loss_est, loss = self.loss(self.target, self.proposal, self.flow, out, acc_rate=acc_rate, alpha=alpha)

        if do_step:
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.flow.parameters(), self.grad_clip)
            self.optimizer.step()

        #print('p', next(self.flow.parameters()).mean())

        return out

    def train(self, n_steps=100, start_optim=10):
        samples = []
        inp = self.proposal.sample((self.batch_size,))
        for step_id in trange(n_steps):
            alpha = min(1., 3*step_id / n_steps)
            out = self.train_step(inp, alpha, do_step=step_id >= start_optim)
            inp = out.detach()
            samples.append(inp)

        return samples

    def sample(self):
        pass
    

# def ex2_mcmc_ula(z, 
#                 target, 
#                 proposal, 
#                 n_steps, 
#                 N,
#                 grad_step,
#                 noise_scale, 
#                 ula_steps = 5,
#                 corr_coef=0., 
#                 bernoulli_prob_corr=0., 
#                 flow=None,
#                 verbose=False):
#     z_sp = [] ###vector of samples
#     batch_size, z_dim = z.shape[0], z.shape[1]

#     mala_transition = MALATransition(z_dim, z.device)
#     mala_transition.adapt_grad_step = grad_step
#     mala_transition.adapt_sigma = noise_scale
#     bern = torch.distributions.Bernoulli(bernoulli_prob_corr)

#     acceptance = torch.zeros(batch_size)

#     if flow is not None:
#         z, log_jac = flow(z)

#     range_gen = trange if verbose else range
#     for step_id in range_gen(n_steps):
#         z_sp.append(z.detach().clone())

#         if corr_coef == 0 and bernoulli_prob_corr == 0: # isir
#             z_new = proposal.sample([batch_size, N - 1])
#         else:
#             # for simplicity assume proposal is N(0, \sigma^2 Id), need to be updated
#             correlation = corr_coef * bern.sample((batch_size,))
#             if flow is not None:
#                 z_backward_pushed, log_jac_minus = flow.inverse(z)
#                 log_jac = -log_jac_minus
#             else:
#                 z_backward_pushed = z

#             latent_var = correlation[:, None] * z_backward_pushed + (1. - correlation[:, None]**2)**.5 * proposal.sample((batch_size,)) #torch.randn((batch_size, z_dim))
#             correlation_new = corr_coef * bern.sample((batch_size, N - 1))
#             z_new = correlation_new[..., None] * latent_var[:, None, :] + (1. - correlation_new[..., None]**2)**.5 * proposal.sample((batch_size, N - 1,)) #torch.randn((batch_size, N - 1, z_dim))
        
#         if flow is not None:
#             z_new, log_jac_new = flow(z_new.view(batch_size * (N - 1), -1))
#             z_new = z_new.view(batch_size, N - 1, -1)
#             log_jacs  = torch.cat([log_jac[:, None], log_jac_new.view(batch_size, N - 1)], 1)
#         else:
#             log_jacs = torch.zeros(batch_size, N)
                
#         X = torch.cat([z.unsqueeze(1), z_new], 1)
#         X_view = X.view(-1, z_dim)

#         log_weight = target(X_view) -  proposal.log_prob(X_view)
#         log_weight = log_weight.view(batch_size, N)
#         if flow is not None:
#             log_weight += log_jacs
        
#         max_logs = torch.max(log_weight, dim = 1)[0][:, None]
#         log_weight = log_weight - max_logs
#         weight = torch.exp(log_weight)
#         sum_weight = torch.sum(weight, dim = 1)
#         weight = weight/sum_weight[:, None]        

#         weight[weight != weight] = 0.
#         weight[weight.sum(1) == 0.] = 1.

#         indices = torch.multinomial(weight, 1).squeeze().tolist()
#         z = X[np.arange(batch_size), indices, :]
#         z = z.data

#         #if flow is not None:
#         #    log_jac = log_jacs[np.arange(batch_size), indices]

#         # mala transition
#         for _ in range(ula_steps):
#             E, grad = grad_energy(z, target)
#             z = z - grad_step * grad + noise_scale * torch.randn([batch_size, z_dim])
#             #z, _, _, mask = mala_transition.do_transition_step(z, z_new, E, grad, grad_step, noise_scale, target, adapt_stepsize=adapt_stepsize)
#             #acceptance += mask.float() / mala_steps
#         #print(mala_transition.adapt_grad_step)
#     z_sp.append(z.detach().clone())

#     return z_sp



###Write equivalent with given normalziing flows


##write function get optimizer


###write update function

