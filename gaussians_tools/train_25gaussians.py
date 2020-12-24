import numpy as np
import sklearn.datasets
import time
import random
import os
import datetime

from matplotlib import pyplot as plt

import torch, torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch import autograd

from paths import path_to_save
from utils import (prepare_25gaussian_data, 
                   prepare_train_batches,
                   prepare_dataloader, 
                   logging)
from gan_fc_models import (Generator_fc, 
                           Discriminator_fc, 
                           weights_init_1, 
                           weights_init_2)
from gan_train import train_gan

random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

train_dataset_size = 64000
batch_size = 128           
X_train = prepare_25gaussian_data(train_dataset_size)
#X_train_batches = prepare_train_batches(X_train, BATCH_SIZE) 
train_dataloader = prepare_dataloader(X_train, batch_size, 
                                      random_seed=random_seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_dim = 2
n_layers_d = 4
n_layers_g = 4
n_hid_d = 100
n_hid_g = 100
n_out = 2

G = Generator_fc(n_dim=n_dim, 
                 n_layers=n_layers_g,
                 n_hid=n_hid_g,
                 n_out=n_out,
                 non_linear=nn.ReLU(),
                 device=device).to(device)
D = Discriminator_fc(n_in=n_dim, 
                     n_layers=n_layers_d,
                     n_hid=n_hid_d,
                     non_linear=nn.ReLU(),
                     device=device).to(device)
G.init_weights(weights_init_2, random_seed=random_seed)
D.init_weights(weights_init_2, random_seed=random_seed)

loss_type='Jensen'
#loss_type='Wasserstein'
lr_init = 1e-3
d_optimizer = torch.optim.Adam(D.parameters(), betas = (0.5, 0.9), lr = lr_init)
g_optimizer = torch.optim.Adam(G.parameters(), betas = (0.5, 0.9), lr = lr_init)
use_gradient_penalty = True
Lambda = 0.01
num_epochs = 5000
num_epoch_for_save = 100
batch_size_sample = 5000  
k_g = 1
k_d = 100   

cur_time = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
new_dir = os.path.join(path_to_save, cur_time)
os.mkdir(new_dir)
plots_dir = os.path.join(new_dir, 'plots')
models_dir = os.path.join(new_dir, 'models')
os.mkdir(plots_dir)
os.mkdir(models_dir)
path_to_logs = os.path.join(new_dir, 'logs.txt')

logging(path_to_logs, train_dataset_size, 
        batch_size, n_dim, n_layers_g, 
        n_layers_d, n_hid_g, n_hid_d, 
        n_out, loss_type, lr_init, 
        Lambda, num_epochs, k_g, k_d)

print("Start to train GAN")
train_gan(X_train=X_train,
          train_dataloader=train_dataloader, 
          generator=G, 
          g_optimizer=g_optimizer, 
          discriminator=D, 
          d_optimizer=d_optimizer,
          path_to_save=new_dir,
          loss_type=loss_type,
          batch_size=batch_size,
          device=device,
          use_gradient_penalty=use_gradient_penalty,
          Lambda=Lambda,
          num_epochs=num_epochs, 
          num_epoch_for_save=num_epoch_for_save,
          batch_size_sample=batch_size_sample,
          k_g=k_g, k_d=k_d, 
          path_to_logs=path_to_logs)
