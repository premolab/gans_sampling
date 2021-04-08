import torch
import numpy as np
import random
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt

DUMP_DIR = 'dump'


def random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_dict_for_sampling(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)

    for p in model.parameters():  
        p.requires_grad = False


def plot_gen_dist(X_gen, x_range, y_range):
    kernel = gaussian_kde(np.unique(X_gen, axis=0).T)

    x = np.linspace(*x_range, 100)
    y = np.linspace(*y_range, 100)
    xx, yy = np.meshgrid(x, y)
    stacked = np.stack([xx.reshape(-1), yy.reshape(-1)], 1)
    vals = kernel(stacked.T)
    vals = vals.reshape(xx.shape)

    fig = plt.figure(figsize=(8, 8))
    plt.xlim(*x_range)
    plt.ylim(*y_range)
    plt.contourf(xx, yy, vals, 20, cmap='Greens')
    plt.legend()
    plt.grid(True)

    return fig
