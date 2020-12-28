import numpy as np
import sklearn.datasets
import os
import matplotlib.pyplot as plt
import datetime

import random
import itertools

import torch

from mh_sampling import mh_sampling

def send_file_to_remote(path_to_file,
                        port_to_remote, 
                        path_to_save_remote):
   if (port_to_remote is not None) and (path_to_save_remote is not None):
          command = f'scp -P {port_to_remote} '.format(port_to_remote)
          command += path_to_file
          command += ' localhost:'
          command += path_to_save_remote
          print(f"Try to send file {path_to_file} to remote server....".format(path_to_file))
          os.system(command)

def sample_fake_data(generator, X_train, epoch, 
                     path_to_save,
                     scaler = None, 
                     batch_size_sample = 5000,
                     path_to_save_remote = None,
                     port_to_remote = None):
    fake_data = generator.sampling(batch_size_sample).data.cpu().numpy()
    if scaler is not None:
       fake_data = scaler.inverse_transform(fake_data)
    plt.figure(figsize=(8, 8))
    plt.xlim(-3., 3.)
    plt.ylim(-3., 3.)
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
       send_file_to_remote(path_to_plot,
                           port_to_remote, 
                           path_to_save_remote)

    else:
       plt.show()

def plot_fake_data_mode(fake, X_train, mode, path_to_save, 
                        scaler = None,
                        path_to_save_remote = None,
                        port_to_remote = None):
    #fake_data = fake.data.cpu().numpy()
    if scaler is not None:
       fake = scaler.inverse_transform(fake)
    plt.figure(figsize=(8, 8))
    plt.xlim(-3., 3.)
    plt.ylim(-3., 3.)
    plt.title(f"Training and {mode} samples", fontsize=20)
    plt.scatter(X_train[:,:1], X_train[:,1:], alpha=0.5, color='gray', 
                marker='o', label = 'training samples')
    plt.scatter(fake[:,:1], fake[:,1:], alpha=0.5, color='blue', 
                marker='o', label = f'{mode} samples')
    plt.legend()
    plt.grid(True)
    if path_to_save is not None:
       plt.savefig(path_to_save)
       send_file_to_remote(path_to_save,
                           port_to_remote, 
                           path_to_save_remote)
    else:
       plt.show()

def visualize_fake_data_projection(fake_data, X_train, path_to_save, 
                                   proj_1, proj_2, 
                                   title,
                                   mode,
                                   path_to_save_remote = None,
                                   port_to_remote = None):
    fake_data_proj = fake_data[:, [proj_1, proj_2]]
    X_train_proj = X_train[:, [proj_1, proj_2]]

    plt.figure(figsize=(8, 8))
    plt.xlim(-3., 3.)
    plt.ylim(-3., 3.)
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
       send_file_to_remote(path_to_plot,
                           port_to_remote, 
                           path_to_save_remote)

    else:
       plt.show()

def discriminator_2d_visualization(discriminator,
                                   x_range,
                                   y_range,
                                   path_to_save,
                                   epoch,
                                   scaler = None,
                                   port_to_remote = None, 
                                   path_to_save_remote = None,
                                   num_points = 700):
    x = torch.linspace(-x_range, x_range, num_points)
    y = torch.linspace(-y_range, y_range, num_points)
    x_t = x.view(-1, 1).repeat(1, y.size(0))
    y_t = y.view(1, -1).repeat(x.size(0), 1)
    x_t_batch = x_t.view(-1 , 1)
    y_t_batch = y_t.view(-1 , 1)
    batch = torch.zeros((x_t_batch.shape[0], 2))
    batch[:, 0] = x_t_batch[:, 0]
    batch[:, 1] = y_t_batch[:, 0]
    if scaler is not None:
       batch = batch.numpy()
       batch = scaler.transform(batch)
       batch = torch.FloatTensor(batch)
    discr_batch = discriminator(batch.to(discriminator.device))
    heatmap = discr_batch[:, 0].view((num_points, 
                                      num_points)).detach().cpu()
    sigmoid_heatmap = heatmap.sigmoid().numpy()
    x_numpy = x.numpy()
    y_numpy = y.numpy()
    y, x = np.meshgrid(x_numpy, y_numpy)
    l_x=x_numpy.min()
    r_x=x_numpy.max()
    l_y=y_numpy.min()
    r_y=y_numpy.max()
    #small_heatmap = sigmoid_heatmap[:-1, :-1]
    figure, axes = plt.subplots(figsize=(8, 8))
    z = axes.contourf(x, y, sigmoid_heatmap, 10, cmap='viridis')
    title = f"Discriminator heatmap, epoch = {epoch}"
    axes.set_title(title)
    axes.axis([l_x, r_x, l_y, r_y])
    figure.colorbar(z)
    if path_to_save is not None:
       figure.savefig(path_to_save)
       send_file_to_remote(path_to_save,
                           port_to_remote, 
                           path_to_save_remote)

def mh_sampling_visualize(generator, 
                          discriminator,
                          X_train, epoch, 
                          path_to_save,
                          n_calib_pts = 10000,
                          scaler = None, 
                          batch_size_sample = 5000,
                          path_to_save_remote = None,
                          port_to_remote = None,
                          type_calibrator = 'iso',
                          normalize_to_0_1 = True):
    X_mh = mh_sampling(X_train, 
                       generator, 
                       discriminator, 
                       generator.device, 
                       n_calib_pts, 
                       batch_size=batch_size_sample,
                       normalize_to_0_1=normalize_to_0_1,
                       type_calibrator=type_calibrator)
    mode = 'MHGAN'
    plot_fake_data_mode(X_mh, X_train, mode, path_to_save, 
                        scaler = scaler,
                        path_to_save_remote = path_to_save_remote,
                        port_to_remote = port_to_remote)

def epoch_visualization(X_train, 
                        generator, 
                        discriminator,
                        use_gradient_penalty, 
                        discriminator_mean_loss_arr, 
                        epoch, Lambda,
                        generator_mean_loss_arr, 
                        path_to_save,
                        batch_size_sample = 5000,
                        loss_type='Jensen',
                        mode = '25_gaussians',
                        port_to_remote = None, 
                        path_to_save_remote = None,
                        scaler = None,
                        proj_list = None,
                        n_calib_pts = 10000,
                        normalize_to_0_1 = True):
    if path_to_save is not None:
        cur_time = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
        plot_name = cur_time + f'_gan_losses_{epoch}_epoch.pdf'
        path_to_plot = os.path.join(path_to_save, plot_name)
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
        fig.savefig(path_to_plot)
        if mode == '25_gaussians':
            x_range = 3.0
            y_range = 3.0
            plot_name = cur_time + f'_discriminator_{epoch}_epoch.pdf'
            path_to_plot_discriminator = os.path.join(path_to_save, plot_name)
            discriminator_2d_visualization(discriminator,
                                           x_range,
                                           y_range,
                                           path_to_plot_discriminator,
                                           epoch,
                                           scaler=scaler,
                                           port_to_remote=port_to_remote, 
                                           path_to_save_remote=path_to_save_remote)
            plot_name = cur_time + f'_{mode}_sampling.pdf'
            path_to_plot_mhgan = os.path.join(path_to_save, plot_name)
            mh_sampling_visualize(generator, 
                                  discriminator,
                                  X_train, epoch, 
                                  path_to_plot_mhgan,
                                  n_calib_pts = n_calib_pts,
                                  scaler = scaler, 
                                  batch_size_sample = batch_size_sample,
                                  port_to_remote=port_to_remote,
                                  path_to_save_remote = path_to_save_remote,
                                  normalize_to_0_1 = normalize_to_0_1)

        if proj_list is None:
            sample_fake_data(generator, X_train, epoch, path_to_save, 
                             scaler=scaler,
                             batch_size_sample=batch_size_sample,
                             port_to_remote=port_to_remote, 
                             path_to_save_remote=path_to_save_remote)

        else:
            fake_data = generator.sampling(batch_size_sample).data.cpu().numpy()
            if scaler is not None:
                fake_data = scaler.inverse_transform(fake_data)
            title = f"Training and generated samples, epoch = {epoch}"
            mode = f"{epoch}_epoch"
            for i in range(len(proj_list)):
                visualize_fake_data_projection(fake_data, X_train, path_to_save, 
                                               proj_list[i][0], proj_list[i][1],
                                               title,
                                               mode,
                                               port_to_remote=port_to_remote, 
                                               path_to_save_remote=path_to_save_remote)
