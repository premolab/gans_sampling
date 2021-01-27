import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F

from general_utils import DotDict

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
        
    def energy(self, x):
        """
        The method returns target logdensity, estimated at point x
        Input:
        x - datapoint
        Output:
        energy = -log p(x)
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
        log_density = torch.logsumexp(log_p, dim=1) 
        return log_density
        
    def energy(self, z, x=None):
        return -self.log_prob(z, x)
        
class IndependentNormal(Target):
    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.device = kwargs.device
        self.torchType = torchType
        self.loc = kwargs['loc']  
        self.scale = kwargs['scale']  
        self.distribution = torch.distributions.Normal(loc=self.loc, scale=self.scale)

    def get_density(self, x):
        density = self.log_prob(x).exp()
        return density

    def log_prob(self, z, x=None):
        log_density = (self.distribution.log_prob(z.to(self.device))).sum(dim=-1)
        return log_density
    
    def sample(self, n):
        return self.distribution.sample(n)
        
    def energy(self, z, x=None):
        return -self.log_prob(z, x)
        
def init_independent_normal(scale, n_dim, device, loc = 0.0):
    loc = loc*torch.ones(n_dim).to(device)
    scale = scale*torch.ones(n_dim).to(device)
    target_args = DotDict()
    target_args.device = device
    target_args.loc = loc
    target_args.scale = scale
    target = IndependentNormal(target_args)
    return target
