import torch

random_seed = 42

train_dataset_size = 64000
batch_size = 512         
sigma = 0.02

n_dim = 256
n_layers_d = 2
n_layers_g = 2
n_hid_d = 128
n_hid_g = 128
n_out = 2

normalize_to_0_1 = True

#loss_type='Jensen_nonsaturing'
loss_type='Jensen_minimax'

# lr_init = 1e-4

disc_lr = 1e-4
gen_lr = 1e-4
betas = (0.5, 0.999)

use_gradient_penalty = False #True
Lambda = 0.01
num_epochs = 5000
num_epoch_for_save = 50
batch_size_sample = 5000  
k_g = 1
k_d = 10
mode = '2d_ring'
n_calib_pts = 10000

plot_mhgan = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
