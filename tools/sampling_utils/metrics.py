import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from torchvision.models.inception import inception_v3
from scipy.stats import gaussian_kde
from sklearn import mixture
import ot
from matplotlib import pyplot as plt
from scipy.stats import chi2, entropy

from general_utils import to_var

def inception_score(imgs, device=None, batch_size=32, resize=False, splits=10):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [0, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    code from - https://github.com/sbarratt/inception-score-pytorch
    """
    N = len(imgs)
    #print(imgs[0].shape)
    #print(imgs[0].min())
    #print(imgs[0].max())
    #print(imgs[0].mean())

    assert batch_size > 0
    assert N > batch_size, "{} <= {}".format(N, batch_size)

    # Set up dtype
    if device not in [torch.device('cpu'), None]:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set device with cuda")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)    

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False).type(dtype)
    if device is not None:
       inception_model = inception_model.to(device)
       up = up.to(device)
    inception_model.eval();
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x, dim=1).data.cpu().numpy()

    preds = np.zeros((N, 1000))
    print("Start to make predictions")
    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = to_var(batch, device)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []
    print("Start to compute KL divergence")
    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

@torch.no_grad()
def get_pis_estimate(X_gen, target_log_prob, n_pts=4000, sample_method='grid', density_method='gmm'):
    target_pdf = lambda x: target_log_prob(torch.FloatTensor(x)).exp().detach().cpu().numpy()

    if density_method == 'kde':
        G_ker = gaussian_kde(X_gen.detach().cpu().numpy().reshape(2, -1))
        pi_g_f = lambda x: G_ker.pdf(x.reshape(2, -1))

    elif density_method == 'gmm':
        gm_g = mixture.GaussianMixture(n_components=25)
        gm_g.fit(X_gen.detach().cpu().numpy())
        pi_g_f = lambda x: np.exp(gm_g.score_samples(x))
    optD = lambda x: target_pdf(x) / (target_pdf(x) + pi_g_f(x) + 1e-8)

    if sample_method == 'grid':
        n_pts = int(n_pts**.5)
        X = np.mgrid[-3:3:4./n_pts, -3:3:4./n_pts].reshape(2, -1).T
    elif sample_method == 'mc':
        pass
        

    pi_d = target_pdf(X)
    pi_g = pi_g_f(X)

    opt_ds = optD(X)
    opt_ds[pi_d < 1e-9] = 0.

    return pi_d, pi_g, opt_ds, X


class Evolution(object):
    def __init__(self, target_sample, **kwargs):#locs, sigma, target_log_prob, target_sample, sigma=0.05, scaler=None):
        self.locs = kwargs.get('locs', None)
        self.sigma = kwargs.get('sigma', None)
        self.target_log_prob = kwargs.get('target_log_prob', None)
        self.target_sample = target_sample
        self.scaler = kwargs.get('scaler', None)
        self.q = kwargs.get('q', 0.999)

        self.mode_std = []
        self.high_quality_rate = []
        self.jsd = []
        self.emd = []

        self.kl_pis = []
        self.js_pis = []
        self.l2_div = []

    @staticmethod
    def make_assignment(X_gen, locs, sigma=0.05, q=0.95):
        n_modes, x_dim = locs.shape
        dists = torch.norm((X_gen[:, None, :] - locs[None, :, :]), p=2, dim=-1)
        chi2_quantile = chi2.ppf(q, x_dim)
        test_dist = (chi2_quantile * (sigma**2))**0.5
        assignment = dists < test_dist
        return assignment

    @staticmethod
    def compute_mode_std(X_gen, assignment):
        """
        X_gen(torch.FloatTensor) - (n_pts, x_dim)
        
        """
        x_dim = X_gen.shape[-1]
        n_modes = assignment.shape[1]
        std = torch.FloatTensor([0.0]).to(X_gen.device)
        found_modes = 0
        for mode_id in range(n_modes):
            xs = X_gen[assignment[:, mode_id]]
            #print(xs.shape)
            if xs.shape[0] > 1:
                std_ = (1 / (x_dim * (xs.shape[0] - 1)) * ((xs - xs.mean(0))**2).sum())**.5
                std += std_
                found_modes += 1
        std /= found_modes
        return std

    @staticmethod
    def compute_mode_mean(X_gen, assignment):
        """
        X_gen(torch.FloatTensor) - (n_pts, x_dim)
        """
        x_dim = X_gen.shape[-1]
        n_modes = assignment.shape[1]
        means = torch.zeros((n_modes, x_dim))
        found_modes_ind = []
        for mode_id in range(n_modes):
            xs = X_gen[assignment[:, mode_id]]
            #print(xs.shape)
            if xs.shape[0] > 1:
                means[mode_id] = xs.mean(0)
                found_modes_ind.append(mode_id)

        return means, found_modes_ind    

    @staticmethod
    def compute_high_quality_rate(assignment):
        high_quality_rate = assignment.max(1)[0].sum() / float(assignment.shape[0])
        return high_quality_rate

    @staticmethod
    def compute_jsd(assignment):
        device = assignment.device
        n_modes = assignment.shape[1]
        assign_ = torch.cat([assignment, torch.zeros(assignment.shape[0]).to(device).unsqueeze(1)], -1)
        assign_[:, -1][assignment.sum(1) == 0] = 1
        sample_dist = assign_.sum(dim=0) / float(assign_.shape[0])
        sample_dist /= sample_dist.sum()
        uniform_dist = torch.FloatTensor([1. / n_modes for _ in range(n_modes)] + [0]).to(assignment.device)
        M = .5 * (uniform_dist + sample_dist)
        # JSD = .5 * (sample_dist * torch.log((sample_dist + 1e-7) / M)) + .5 * (uniform_dist * torch.log((uniform_dist + 1e-7) / M))

        # JSD[sample_dist == 0.] = 0.
        # JSD[uniform_dist == 0.] = 0.
        # JSD = JSD.sum()

        KL1 = sample_dist * (torch.log(sample_dist) - torch.log(M))
        KL1[sample_dist == 0.] = 0.
        KL2 = uniform_dist * (torch.log(uniform_dist)  -torch.log(M))
        KL2[uniform_dist == 0.] = 0.
        JSD = .5 * (KL1 + KL2).sum()

        return JSD

    @staticmethod
    def compute_emd(target_sample, gen_sample):
        gen_sample = gen_sample[np.random.choice(np.arange(gen_sample.shape[0]), target_sample.shape[0])]
        M = np.linalg.norm(target_sample[None, :, :] - gen_sample[:, None, :], axis=-1, ord=2)**2
        emd = ot.lp.emd2([], [], M)

        return emd

    def invoke(self, X_gen, D=None, compute_discrim_div=False):
        emd = Evolution.compute_emd(self.target_sample, X_gen.detach().cpu().numpy())
        self.emd.append(emd)
        
        if self.locs is not None and self.sigma is not None:
            assignment = Evolution.make_assignment(X_gen, self.locs, self.sigma, self.q)
            #print(assignment.shape)
            mode_std = Evolution.compute_mode_std(X_gen, assignment)
            #print(mode_std)
            self.mode_std.append(mode_std.item())
            h_q_r = Evolution.compute_high_quality_rate(assignment)
            self.high_quality_rate.append(h_q_r.item())
            jsd = Evolution.compute_jsd(assignment)
            self.jsd.append(jsd.item())

        if self.target_log_prob is not None:
            pi_d, pi_g, opt_ds, X = get_pis_estimate(X_gen, self.target_log_prob, 
                                                     n_pts=4000, sample_method='grid', density_method='gmm')
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
                emd=self.emd,
                kl_pis=self.kl_pis,
                js_pis=self.js_pis,
                l2_div=self.l2_div)
        return d
