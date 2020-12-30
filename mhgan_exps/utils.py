import numpy as np
import torch
import itertools
from pathlib import Path
from sklearn import mixture
from scipy.stats import gaussian_kde
from tqdm import tqdm
import re


class GaussianMixture():
    def __init__(self, **kwargs):
        self.n = 25
        self.d = 2
        self.sigma = kwargs.get('sigma', 0.05)
        self.means = np.array(list(itertools.product(np.arange(-2,3), repeat=2)))
        self.torch_dist = torch.distributions.Normal(torch.FloatTensor(self.means), torch.FloatTensor([self.sigma]))
        
    def sample(self, n_pts):
        pass

    def pdf(self, x):
        assert len(x.shape) == 2
        x = x.reshape(-1, 2)
        x = torch.FloatTensor(x)
        assert x.shape[1] == self.d
        pis = self.torch_dist.log_prob(x.unsqueeze(1).repeat(1, len(self.means), 1)).sum(2).exp().sum(1) / float(len(self.means))
        return pis.detach().cpu().numpy()


@torch.no_grad()
def get_pis_estimate(X_gen, n_pts=4000, sample_method='grid', density_method='gmm', sigma=0.05):
    gm = GaussianMixture(sigma=sigma)

    if density_method == 'kde':
        G_ker = gaussian_kde(X_gen.detach().cpu().numpy().reshape(2, -1))
        pi_g_f = lambda x: G_ker.pdf(x.reshape(2, -1))

    elif density_method == 'gmm':
        gm_g = mixture.GaussianMixture(n_components=25)
        gm_g.fit(X_gen.detach())
        pi_g_f = lambda x: np.exp(gm_g.score_samples(x))
    optD = lambda x: gm.pdf(x) / (gm.pdf(x) + pi_g_f(x) + 1e-8)

    if sample_method == 'grid':
        n_pts = int(n_pts**.5)
        X = np.mgrid[-3:3:4./n_pts, -3:3:4./n_pts].reshape(2, -1).T
    elif sample_method == 'mc':
        pass
        #X = 4*torch.rand(n_pts, 2) - 2.

    pi_d = gm.pdf(X)
    pi_g = pi_g_f(X)

    opt_ds = optD(X)
    opt_ds[pi_d < 1e-9] = 0.

    return pi_d, pi_g, opt_ds, X


class Evolution(object):
    def __init__(self, means, sigma=0.05, scaler=None):
        self.means = means
        self.sigma = sigma
        self.scaler = scaler
        self.mode_std = []
        self.high_quality_rate = []
        self.jsd = []

        self.kl_pis = []
        self.js_pis = []
        self.l2_div = []

    @staticmethod
    def make_assignment(X_gen, means, sigma=0.05):
        n_modes, x_dim = means.shape
        dists = torch.norm((X_gen[:, None, :] - means[None, :, :]), p=2, dim=-1)
        assignment = dists < 4 * sigma
        return assignment

    @staticmethod
    def compute_mode_std(X_gen, assignment):
        """
        X_gen(torch.FloatTensor) - (n_pts, x_dim)
        
        """
        x_dim = X_gen.shape[-1]
        n_modes = assignment.shape[1]
        std = 0
        for mode_id in range(n_modes):
            xs = X_gen[assignment[:, mode_id]]
            if xs.shape[0] > 1:
                std_ = (1 / (2**(x_dim - 1) * (xs.shape[0] - 1)) * ((xs - xs.mean(0))**2).sum())**.5
                std += std_
        std /= n_modes
        return std

    @staticmethod
    def compute_high_quality_rate(assignment):
        high_quality_rate = assignment.max(1)[0].sum() / float(assignment.shape[0])
        return high_quality_rate

    @staticmethod
    def compute_jsd(assignment):
        n_modes = assignment.shape[1]
        assign_ = torch.cat([assignment, torch.zeros(assignment.shape[0]).unsqueeze(1)], -1)
        assign_[:, -1][assignment.sum(1) == 0] = 1
        sample_dist = assign_.sum(dim=0) / float(assign_.shape[0])
        sample_dist /= sample_dist.sum()
        uniform_dist = torch.FloatTensor([1. / n_modes for _ in range(n_modes)] + [0]).to(assignment.device)
        M = .5 * (uniform_dist + sample_dist)
        JSD = .5 * (sample_dist * torch.log((sample_dist + 1e-7) / M)).sum() + .5 * (uniform_dist * torch.log((uniform_dist + 1e-7) / M)).sum()

        return JSD

    def invoke(self, X_gen, D=None, compute_discrim_div=False):
        assignment = Evolution.make_assignment(X_gen, self.means, self.sigma)
        mode_std = Evolution.compute_mode_std(X_gen, assignment)
        self.mode_std.append(mode_std.item())
        h_q_r = Evolution.compute_high_quality_rate(assignment)
        self.high_quality_rate.append(h_q_r.item())
        jsd = Evolution.compute_jsd(assignment)
        self.jsd.append(jsd.item())

        pi_d, pi_g, opt_ds, X = get_pis_estimate(X_gen, n_pts=4000, sample_method='grid', density_method='gmm', sigma=self.sigma)
        kl = pi_g * (np.log(pi_g) - np.log(pi_d + 1e-10))
        kl[pi_g == 0.] = 0.
        kl = np.mean(kl).item()
        self.kl_pis.append(kl)

        m = .5 * (pi_g + pi_d)
        js = .5 * (pi_g * (np.log(pi_g) - np.log(m)) + pi_d * (np.log(pi_d) - np.log(m)))
        js[(pi_g == 0.) + (pi_d == 0.)] = 0.
        js = np.mean(js).item()
        self.js_pis.append(js)

        if self.scaler is not None and compute_discrim_div and D is not None:
            ds = D(torch.FloatTensor(self.scaler.transform(X)))[:, 0]
            div = torch.abs(ds - torch.FloatTensor(opt_ds))
            inf_dist = torch.max(div).item()
            l2_dist = ((div**2).mean()).item()**.5
            self.l2_div.append(l2_dist)

    def as_dict(self):
        d = dict(mode_std=self.mode_std, 
                hqr=self.high_quality_rate,
                jsd=self.jsd,
                kl_pis=self.kl_pis,
                js_pis=self.js_pis,
                l2_div=self.l2_div)
        return d


def collect_evolution(samples, means, sigma, scaler, D, G, models_path, device):
    models_path = Path(models_path)
    discriminator_regexp = "*_discriminator.pth"
    generator_regexp = "*_generator.pth"
    discriminator_name = list(sorted([f for f in models_path.glob(discriminator_regexp)], key=lambda x: int(re.findall(r'\d+', x.name)[-1])))
    generator_name = list(sorted([f for f in models_path.glob(generator_regexp)], key=lambda x: int(re.findall(r'\d+', x.name)[-1])))

    evolution = Evolution(means=torch.FloatTensor(means).to(device), sigma=sigma, scaler=scaler)
    for sample, d_name, g_name in tqdm(zip(samples, discriminator_name, generator_name)):
        G.load_state_dict(torch.load(g_name, map_location=device))
        D.load_state_dict(torch.load(d_name, map_location=device))

        evolution.invoke(torch.FloatTensor(scaler.inverse_transform(sample)), compute_discrim_div=True, D=D)

    return evolution

