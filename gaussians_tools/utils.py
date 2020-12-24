import numpy as np
import sklearn.datasets
import os
import matplotlib.pyplot as plt
import datetime
import random

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

class PoolSet(Dataset):
    def __init__(self, p_x):
        ## input: torch.tensor (NOT CUDA TENSOR)
        self.len = len(p_x)
        self.x = p_x    ##[N, p]
    
    def __getitem__(self, index):
        return self.x[index]
    
    def __len__(self):
        return self.len

def prepare_swissroll_data(batch_size=1000):
    data = sklearn.datasets.make_swiss_roll(
                    n_samples=batch_size,
                    noise=0.25
                )[0]
    data = data.astype('float32')[:, [0, 2]]
    data /= 7.5 # stdev plus a little
    return data

def prepare_25gaussian_data(batch_size=1000):
    dataset = []
    for i in range(batch_size//25):
        for x in range(-2, 3):
            for y in range(-2, 3):
                point = np.random.randn(2)*0.05
                point[0] += 2*x
                point[1] += 2*y
                dataset.append(point)
    dataset = np.array(dataset, dtype=np.float32)
    np.random.shuffle(dataset)
    dataset /= 2.828 # stdev
    return dataset 

def prepare_gaussians(num_samples_in_cluster, dim, 
                      num_gaussian_per_dim, coord_limits, 
                      std = 0.1, random_state = 42,
                      scale = 2.828):
    num_clusters = num_gaussian_per_dim ** dim
    num_samples = num_samples_in_cluster * num_clusters
    coords_per_dim = np.linspace(-coord_limits, 
                                 coord_limits, 
                                 num = num_gaussian_per_dim)
    copy_coords = list(np.tile(coords_per_dim, (dim, 1)))
    centers = np.array(np.meshgrid(*copy_coords)).T.reshape(-1, dim)
    dataset = sklearn.datasets.make_blobs(n_samples = num_samples, 
                                          n_features = dim, 
                                          centers = centers, 
                                          cluster_std = std,
                                          random_state = random_state)[0]
    dataset /= scale
    return dataset

def prepare_train_batches(dataset, batch_size):
    while True:
        for i in range(len(dataset) // batch_size):
            yield dataset[i * batch_size:(i + 1) * batch_size]

def prepare_dataloader(dataset, batch_size, random_seed=None):
    dataset = torch.Tensor(dataset) 
    poolset = PoolSet(dataset)
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)    
    dataloader = DataLoader(poolset, batch_size=batch_size, shuffle=True)
    return dataloader

def sample_fake_data(generator, X_train, epoch, path_to_save, batch_size_sample = 5000):
    fake_data = generator.sampling(batch_size_sample).data.cpu().numpy()
    plt.figure(figsize=(8, 8))
    plt.xlim(-2., 2.)
    plt.ylim(-2., 2.)
    plt.title("Training and generated samples", fontsize=20)
    plt.scatter(X_train[:,:1], X_train[:,1:], alpha=0.5, color='gray', 
                marker='o', label = 'training samples')
    plt.scatter(fake_data[:,:1], fake_data[:,1:], alpha=0.5, color='blue', 
                marker='o', label = 'samples by G')
    plt.legend()
    plt.grid(True)
    if path_to_save is not None:
       cur_time = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
       plot_name = cur_time + f'_gan_sampling_{epoch}_epoch.pdf'
       path_to_plot = os.path.join(path_to_save, plot_name)
       plt.savefig(path_to_plot)

    else:
       plt.show()

def plot_fake_data_mode(fake, X_train, mode, path_to_save):
    fake_data = fake.data.cpu().numpy()
    plt.figure(figsize=(8, 8))
    plt.xlim(-2., 2.)
    plt.ylim(-2., 2.)
    plt.title(f"Training and {mode} samples", fontsize=20)
    plt.scatter(X_train[:,:1], X_train[:,1:], alpha=0.5, color='gray', 
                marker='o', label = 'training samples')
    plt.scatter(fake_data[:,:1], fake_data[:,1:], alpha=0.5, color='blue', 
                marker='o', label = f'{mode} samples')
    plt.legend()
    plt.grid(True)
    if path_to_save is not None:
       cur_time = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
       plot_name = cur_time + f'_{mode}_sampling.pdf'
       path_to_plot = os.path.join(path_to_save, plot_name)
       plt.savefig(path_to_plot)

    else:
       plt.show()

def visualize_fake_data_projection(fake_data, X_train, path_to_save, proj_1, proj_2, 
                                   title,
                                   mode):
    fake_data_proj = fake_data[:, [proj_1, proj_2]]
    X_train_proj = X_train[:, [proj_1, proj_2]]

    plt.figure(figsize=(8, 8))
    plt.xlim(-2., 2.)
    plt.ylim(-2., 2.)
    plt.title(title, fontsize=20)
    X_train_proj = X_train[:, [proj_1, proj_2]]


    plt.scatter(X_train_proj[:, 0], X_train_proj[:, 1], alpha=0.5, color='gray',
                marker='o', label = 'training samples')
    plt.scatter(fake_data_proj[:, 0], fake_data_proj[:, 1], alpha=0.5, color='blue',
                marker='o', label = 'samples by G')
    plt.xlabel(f"proj ind = {proj_1 + 1}")
    plt.ylabel(f"proj ind = {proj_2 + 1}")
    plt.legend()
    plt.grid(True)
    if path_to_save is not None:
       cur_time = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
       plot_name = cur_time + f"_gan_sampling_" + mode + f"_proj1_{proj_1}_proj2_{proj_2}.pdf"
       path_to_plot = os.path.join(path_to_save, plot_name)
       plt.savefig(path_to_plot)

    else:
       plt.show()

def epoch_visualization(X_train, generator, 
                        use_gradient_penalty, 
                        discriminator_mean_loss_arr, 
                        epoch, Lambda,
                        generator_mean_loss_arr, 
                        path_to_save,
                        batch_size_sample = 5000,
                        loss_type='Jensen',
                        proj_list = None):
    subtitle_for_losses = "Training process for discriminator and generator"
    if (use_gradient_penalty):
        subtitle_for_losses += f" with gradient penalty, $\lambda = {Lambda}$"
    fig, axs = plt.subplots(1, 2, figsize = (20, 5))
    fig.suptitle(subtitle_for_losses)
    axs[0].set_xlabel("#epoch")
    axs[0].set_ylabel("loss")
    axs[1].set_xlabel("#epoch")
    axs[1].set_ylabel("loss")
    axs[0].grid(True)
    axs[1].grid(True)
    axs[0].set_title('D-loss')
    axs[1].set_title('G-loss')
    axs[0].plot(discriminator_mean_loss_arr, 'b', 
                label = f'discriminator loss = {loss_type}')
    axs[1].plot(generator_mean_loss_arr, 'r', label = 'generator loss')
    axs[0].legend()
    axs[1].legend()
    cur_time = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
    plot_name = cur_time + f'_gan_losses_{epoch}_epoch.pdf'
    path_to_plot = os.path.join(path_to_save, plot_name)
    fig.savefig(path_to_plot)

    if proj_list is None:
        sample_fake_data(generator, X_train, epoch, path_to_save, batch_size_sample)

    else:
        fake_data = generator.sampling(batch_size_sample).data.cpu().numpy()
        title = "Training and generated samples"
        mode = f"{epoch}_epoch"
        for i in range(len(proj_list)):
            visualize_fake_data_projection(fake_data, X_train, path_to_save, 
                                           proj_list[i][0], proj_list[i][1],
                                           title,
                                           mode)

def logging(path_to_logs, train_dataset_size,
            batch_size, n_dim, n_layers_g, 
            n_layers_d, n_hid_g, n_hid_d, 
            n_out, loss_type, lr_init, 
            Lambda, num_epochs, k_g, k_d):
    f = open(path_to_logs, "w")
    f.write("Setup for training GANs:\n")
    f.write(f"Train dataset size = {train_dataset_size}\n")
    f.write(f"Batch size = {batch_size}\n")
    f.write(f"Hidden dim for prior of generator = {n_dim}\n")
    f.write(f"Number of hidden layers in generator = {n_layers_g}\n")
    f.write(f"Number of hidden layers in discriminator = {n_layers_d}\n")
    f.write(f"Number of hidden neurons in generator = {n_hid_g}\n")
    f.write(f"Number of hidden neurons in discriminator = {n_hid_d}\n")
    f.write(f"Dim of output for generator = {n_out}\n")
    f.write(f"Loss type = {loss_type}\n")
    f.write(f"Learning rate = {lr_init}\n")
    f.write(f"Lambda for gradient penalization = {Lambda}\n")
    f.write(f"Number of epochs = {num_epochs}\n")
    f.write(f"Number of generator learning passes = {k_g}\n")
    f.write(f"Number of discriminator learning passes = {k_d}\n")
    f.write(f"-------------------------------------------\n")
    f.close()
