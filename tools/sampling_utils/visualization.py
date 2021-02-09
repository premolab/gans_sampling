import numpy as np
import sklearn.datasets
import os
import matplotlib.pyplot as plt
import datetime
from functools import partial

import random
import itertools

import torch
from torch.distributions import Normal

from general_utils import send_file_to_remote

from mh_sampling import mh_sampling
from ebm_sampling import (langevin_sampling, 
                          mala_sampling, 
                          xtry_langevin_sampling,
                          gan_energy)

figsize=(8,8)
 
def sample_fake_data(generator, X_train, 
                     x_range,
                     y_range, 
                     path_to_save = None,
                     epoch = None,
                     scaler = None, 
                     batch_size_sample = 5000,
                     path_to_save_remote = None,
                     port_to_remote = None):
    fake_data = generator.sampling(batch_size_sample).data.cpu().numpy()
    if scaler is not None:
       fake_data = scaler.inverse_transform(fake_data)
    plt.figure(figsize=(8, 8))
    plt.xlim(-x_range, x_range)
    plt.ylim(-y_range, y_range)
    plt.title("Training and generated samples", fontsize=20)
    plt.scatter(X_train[:,:1], X_train[:,1:], alpha=0.5, color='gray', 
                marker='o', label = 'training samples')
    plt.scatter(fake_data[:,:1], fake_data[:,1:], alpha=0.5, color='blue', 
                marker='o', label = 'samples by G')
    plt.legend()
    plt.grid(True)
    if path_to_save is not None:
       cur_time = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
       if epoch is not None:
          plot_name = cur_time + f'_gan_sampling_{epoch}_epoch.pdf'
       else:
          plot_name = cur_time + f'_gan_sampling.pdf'

       path_to_plot = os.path.join(path_to_save, plot_name)
       plt.savefig(path_to_plot)
       send_file_to_remote(path_to_plot,
                           port_to_remote, 
                           path_to_save_remote)
       plt.close()
    else:
       plt.show()

def plot_fake_data_mode(fake, X_train, mode, 
                        path_to_save = None, 
                        scaler = None,
                        path_to_save_remote = None,
                        port_to_remote = None,
                        params = None):
    #fake_data = fake.data.cpu().numpy()
    if scaler is not None:
       fake = scaler.inverse_transform(fake)
    plt.figure(figsize=figsize)
    plt.xlim(-3., 3.)
    plt.ylim(-3., 3.)
    plt.title(f"Training and {mode} samples", fontsize=20)
    plt.scatter(X_train[:,:1], X_train[:,1:], alpha=0.3, color='gray', 
                marker='o', label = 'training samples')
    label = f'{mode} samples'
    if params is not None:
       label += (', ' + params)
    plt.scatter(fake[:,:1], fake[:,1:], alpha=0.3, color='blue', 
                marker='o', label = label)
    plt.legend()
    plt.grid(True)
    if path_to_save is not None:
       plt.savefig(path_to_save)
       send_file_to_remote(path_to_save,
                           port_to_remote, 
                           path_to_save_remote)
       plt.close()
    else:
       plt.show()

def plot_fake_data_projection(fake, X_train, 
                              proj_1, proj_2, 
                              title,
                              fake_label,
                              path_to_save = None,
                              scaler = None,
                              path_to_save_remote = None,
                              port_to_remote = None):
    if scaler is not None:
       fake = scaler.inverse_transform(fake)
    fake_proj = fake[:, [proj_1, proj_2]]

    plt.figure(figsize=figsize)
    plt.xlim(-3., 3.)
    plt.ylim(-3., 3.)
    plt.title(title, fontsize=20)
    X_train_proj = X_train[:, [proj_1, proj_2]]

    plt.scatter(X_train_proj[:, 0], X_train_proj[:, 1], alpha=0.3, color='gray',
                marker='o', label = 'training samples')
    plt.scatter(fake_proj[:, 0], fake_proj[:, 1], alpha=0.3, color='blue',
                marker='o', label = fake_label)
    plt.xlabel(f"proj ind = {proj_1 + 1}")
    plt.ylabel(f"proj ind = {proj_2 + 1}")
    plt.legend()
    plt.grid(True)
    if path_to_save is not None:
       plt.savefig(path_to_save)
       send_file_to_remote(path_to_save,
                           port_to_remote, 
                           path_to_save_remote)
       plt.close()
    else:
       plt.show()

def plot_discriminator_2d(discriminator,
                          x_range,
                          y_range,
                          path_to_save = None,
                          epoch = None,
                          scaler = None,
                          port_to_remote = None, 
                          path_to_save_remote = None,
                          normalize_to_0_1 = True,
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
    if normalize_to_0_1:
       heatmap = heatmap.sigmoid().numpy()
    else:
       heatmap = heatmap.numpy()
       
    x_numpy = x.numpy()
    y_numpy = y.numpy()
    y, x = np.meshgrid(x_numpy, y_numpy)
    l_x=x_numpy.min()
    r_x=x_numpy.max()
    l_y=y_numpy.min()
    r_y=y_numpy.max()
    #small_heatmap = sigmoid_heatmap[:-1, :-1]
    figure, axes = plt.subplots(figsize=figsize)
    z = axes.contourf(x, y, heatmap, 10, cmap='viridis')
    if epoch is not None:
       title = f"Discriminator heatmap, epoch = {epoch}"
    else:
       title = f"Discriminator heatmap"
    axes.set_title(title)
    axes.axis([l_x, r_x, l_y, r_y])
    figure.colorbar(z)
    if path_to_save is not None:
       figure.savefig(path_to_save)
       send_file_to_remote(path_to_save,
                           port_to_remote, 
                           path_to_save_remote)
       plt.close()

def plot_potential_energy(target_energy,
                          x_range,
                          y_range,
                          device,
                          path_to_save = None,
                          norm_grads = False,
                          normalize_to_0_1 = True,
                          num_points = 100):
    x = torch.linspace(-x_range, x_range, num_points)
    y = torch.linspace(-y_range, y_range, num_points)
    x_t = x.view(-1, 1).repeat(1, y.size(0))
    y_t = y.view(1, -1).repeat(x.size(0), 1)
    x_t_batch = x_t.view(-1 , 1)
    y_t_batch = y_t.view(-1 , 1)
    batch = torch.zeros((x_t_batch.shape[0], 2))
    batch[:, 0] = x_t_batch[:, 0]
    batch[:, 1] = y_t_batch[:, 0]

    batch = batch.to(device)

    if norm_grads:
        batch.requires_grad_(True)
        batch_energy = target_energy(batch).sum() 
        batch_energy.backward()
        batch_grads = batch.grad.detach().cpu()
        batch_grads_norm = torch.norm(batch_grads, p=2, dim=-1)
        result = batch_grads_norm.view((num_points, 
                                        num_points)).detach().cpu().numpy()
        title = f"Latent energy norm gradients"
    else:
        batch_energy = target_energy(batch)
        result = batch_energy.view((num_points, 
                                    num_points)).detach().cpu().numpy()
        title = "Latent energy"
    
    x_numpy = x.numpy()
    y_numpy = y.numpy()
    y, x = np.meshgrid(x_numpy, y_numpy)
    l_x=x_numpy.min()
    r_x=x_numpy.max()
    l_y=y_numpy.min()
    r_y=y_numpy.max()
    #small_heatmap = sigmoid_heatmap[:-1, :-1]
    figure, axes = plt.subplots(figsize=figsize)
    z = axes.contourf(x, y, result, 10, cmap='viridis')
    axes.set_title(title)
    axes.axis([l_x, r_x, l_y, r_y])
    figure.colorbar(z)
    if path_to_save is not None:
        plt.savefig(path_to_save)
    else:
        plt.show()

def langevin_sampling_plot_2d(target,
                              proposal,
                              X_train,  
                              path_to_save = None,
                              scaler = None, 
                              batch_size_sample = 5000,
                              path_to_save_remote = None,
                              port_to_remote = None,
                              grad_step = 1e-3,
                              eps_scale = 1e-2,
                              n_steps = 5000,
                              n_batches = 1,
                              latent_transform = None):
    batchsize = batch_size_sample // n_batches
    X_langevin, zs = langevin_sampling(target, proposal, n_steps, grad_step, eps_scale, batch_size_sample, batchsize)
    if latent_transform is not None:
        X_langevin = torch.FloatTensor(X_langevin).to(proposal.device)
        X_langevin = latent_transform(X_langevin).data.cpu().numpy()
    mode = 'ULA'
    params = f'lr = {grad_step}, std noise = {round(eps_scale, 3)}'
    plot_fake_data_mode(X_langevin, X_train, mode, 
                        path_to_save = path_to_save, 
                        scaler = scaler,
                        path_to_save_remote = path_to_save_remote,
                        port_to_remote = port_to_remote,
                        params = params)
                        
def mala_sampling_plot_2d(target,
                          proposal,
                          X_train,  
                          path_to_save = None,
                          scaler = None, 
                          batch_size_sample = 5000,
                          path_to_save_remote = None,
                          port_to_remote = None,
                          grad_step = 1e-3,
                          eps_scale = 1e-2,
                          n_steps = 5000,
                          n_batches = 1,
                          acceptance_rule = 'Hastings',
                          latent_transform = None):
    batchsize = batch_size_sample // n_batches
    X_mala, zs = mala_sampling(target, proposal, n_steps, grad_step, eps_scale, batch_size_sample, batchsize, acceptance_rule=acceptance_rule)
    if latent_transform is not None:
        X_mala = torch.FloatTensor(X_mala).to(proposal.device)
        X_mala = latent_transform(X_mala).data.cpu().numpy()
    mode = f'MALA/{acceptance_rule}'
    params = f'lr = {grad_step}, std noise = {round(eps_scale, 3)}'
    plot_fake_data_mode(X_mala, X_train, mode, 
                        path_to_save = path_to_save, 
                        scaler = scaler,
                        path_to_save_remote = path_to_save_remote,
                        port_to_remote = port_to_remote,
                        params = params)
                        
def xtry_langevin_sampling_plot_2d(target,
                                   proposal,
                                   X_train,  
                                   path_to_save = None,
                                   scaler = None, 
                                   batch_size_sample = 5000,
                                   path_to_save_remote = None,
                                   port_to_remote = None,
                                   N = 2,
                                   grad_step = 1e-3,
                                   eps_scale = 1e-2,
                                   n_steps = 5000,
                                   n_batches = 1,
                                   latent_transform = None):
    batchsize = batch_size_sample // n_batches
    X_xtry_langevin, zs = xtry_langevin_sampling(target, proposal, n_steps, grad_step, eps_scale, N, batch_size_sample, batchsize)
    if latent_transform is not None:
        X_xtry_langevin = torch.FloatTensor(X_xtry_langevin).to(proposal.device)
        X_xtry_langevin = latent_transform(X_xtry_langevin).data.cpu().numpy()
    mode = 'X-Try-ULA'
    params = f'lr = {grad_step}, std noise = {round(eps_scale, 3)}, N = {N}'
    plot_fake_data_mode(X_xtry_langevin, X_train, mode, 
                        path_to_save = path_to_save, 
                        scaler = scaler,
                        path_to_save_remote = path_to_save_remote,
                        port_to_remote = port_to_remote,
                        params = params)                        

def mh_sampling_plot_2d(generator, 
                        discriminator,
                        X_train, 
                        path_to_save = None,
                        n_calib_pts = 10000,
                        scaler = None, 
                        batch_size_sample = 5000,
                        path_to_save_remote = None,
                        port_to_remote = None,
                        type_calibrator = 'iso',
                        normalize_to_0_1 = True):
    if scaler is not None:
       X_train_scale = scaler.transform(X_train)
    else:
       X_train_scale = X_train
    print("Start to do MH sampling....")
    X_mh = mh_sampling(X_train_scale, 
                       generator, 
                       discriminator, 
                       generator.device, 
                       n_calib_pts, 
                       batch_size_sample=batch_size_sample,
                       normalize_to_0_1=normalize_to_0_1,
                       type_calibrator=type_calibrator)
    mode = 'MHGAN'
    params = f'calibrator = {type_calibrator}'
    plot_fake_data_mode(X_mh, X_train, mode, 
                        path_to_save = path_to_save, 
                        scaler = scaler,
                        path_to_save_remote = path_to_save_remote,
                        port_to_remote = port_to_remote,
                        params = params)

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
                        normalize_to_0_1 = True,
                        plot_mhgan = False):
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
        if (mode in ['25_gaussians', 'swissroll']):
            if mode == '25_gaussians':
               x_range = 3.0
               y_range = 3.0
            else:
               x_range = 2.0
               y_range = 2.0
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
            if plot_mhgan:
               mh_mode = 'mhgan'
               plot_name = cur_time + f'_{mh_mode}_sampling.pdf'
               path_to_plot_mhgan = os.path.join(path_to_save, plot_name)
              
               type_calibrator = 'iso'
               mh_sampling_visualize(generator, 
                                     discriminator,
                                     X_train,  
                                     path_to_plot_mhgan,
                                     n_calib_pts = n_calib_pts,
                                     scaler = scaler, 
                                     batch_size_sample = batch_size_sample,
                                     port_to_remote=port_to_remote,
                                     path_to_save_remote = path_to_save_remote,
                                     normalize_to_0_1 = normalize_to_0_1,
                                     type_calibrator = type_calibrator)

            sample_fake_data(generator, X_train,
                             x_range,
                             y_range,  
                             path_to_save,
                             epoch=epoch, 
                             scaler=scaler,
                             batch_size_sample=batch_size_sample,
                             port_to_remote=port_to_remote, 
                             path_to_save_remote=path_to_save_remote)
        
        

        elif mode == '5d_gaussians':
            fake_generator = generator.sampling(batch_size_sample).data.cpu().numpy()
            cur_time = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')

            if plot_mhgan:
               if scaler is not None:
                  X_train_scale = scaler.transform(X_train)
               else:
                  X_train_scale = X_train

               print("Start to do MH sampling....")
               type_calibrator = 'iso'
               X_mh = mh_sampling(X_train_scale, 
                                  generator, 
                                  discriminator, 
                                  generator.device, 
                                  n_calib_pts, 
                                  batch_size_sample=batch_size_sample,
                                  normalize_to_0_1=normalize_to_0_1,
                                  type_calibrator=type_calibrator)

            for i in range(len(proj_list)):
                proj_1 = proj_list[i][0]
                proj_2 = proj_list[i][1]
                plot_name = cur_time + f"_gan_sampling_epoch_{epoch}_proj1_{proj_1}_proj2_{proj_2}.pdf"
                path_to_plot_generator = os.path.join(path_to_save, plot_name)

                title_generator = "Training and generated samples"
                fake_label_generator = "samples by G"

                plot_fake_data_projection(fake = fake_generator, 
                                          X_train = X_train,
                                          path_to_save = path_to_plot_generator, 
                                          proj_1 = proj_1, 
                                          proj_2 = proj_2,
                                          title = title_generator,
                                          fake_label = fake_label_generator, 
                                          scaler = scaler,
                                          path_to_save_remote = path_to_save_remote,
                                          port_to_remote = port_to_remote)
                if plot_mhgan:
                   title_mhgan = "Training and MHGAN samples"
                   fake_label_mhgan = 'MHGAN samples'
                   mh_mode = 'mhgan'
                   plot_name = cur_time + f'_{mh_mode}_epoch_{epoch}_proj1_{proj_1}_proj2_{proj_2}.pdf'
                   path_to_plot_mhgan = os.path.join(path_to_save, plot_name)
                   plot_fake_data_projection(fake = X_mh, 
                                             X_train = X_train,
                                             path_to_save = path_to_plot_mhgan, 
                                             proj_1 = proj_1, 
                                             proj_2 = proj_2,
                                             title = title_mhgan,
                                             fake_label = fake_label_mhgan, 
                                             scaler = scaler,
                                             path_to_save_remote = path_to_save_remote,
                                             port_to_remote = port_to_remote)


def plot_chain_metrics(every=50, name=None, savepath=None, sigma=0.05, **evols):
    instance = list(evols.values())[0]
    keys = ['mode_std', 'hqr', 'jsd', 'emd']
    ncols = np.sum([int(len(instance[k]) > 0) for k in keys])
    
    fig, axs = plt.subplots(ncols=ncols, nrows=1, figsize=(6*ncols, 6))

    if name is not None:
        fig.suptitle(name)
    k = 0
    if sigma is not None and len(instance['mode_std']) > 0:
        axs[k].axhline(sigma, label='real', color='black')
        axs[k].set_xlabel('iter')
        axs[k].set_ylabel('mode std')
        k += 1

    if len(instance['hqr']) > 0:
        axs[k].axhline(1, label='real', color='black')
        axs[k].set_xlabel('iter')
        axs[k].set_ylabel('high quality rate')
        k += 1

    if len(instance['jsd']) > 0:
        axs[k].axhline(0, label='real', color='black')
        axs[k].set_xlabel('iter')
        axs[k].set_ylabel('JSD')
        k += 1

    emd_ax = axs if k == 0 else axs[k]
    emd_ax.axhline(0, label='real', color='black')
    emd_ax.set_xlabel('iter')
    emd_ax.set_ylabel('EMD')

    for label, evol in evols.items():
        k == 0
        if len(instance['mode_std']) > 0:
            k += 1
            axs[0].plot(np.arange(0, len(evol['mode_std'])) * every, evol['mode_std'], label=label, marker='o')
        if len(instance['hqr']) > 0:
            k += 1   
            axs[1].plot(np.arange(0, len(evol['hqr'])) * every, evol['hqr'], label=label, marker='o')
        if len(instance['jsd']) > 0:
            k += 1
            axs[2].plot(np.arange(0, len(evol['jsd'])) * every, evol['jsd'], label=label, marker='o')
        emd_ax.plot(np.arange(0, len(evol['emd'])) * every, evol['emd'], label=label, marker='o')

    if k > 0:
         for ax in axs:
            ax.grid()
            ax.legend()
    else:
      emd_ax.grid()
      emd_ax.legend()
        

    if savepath is not None:
        plt.savefig(savepath)
    plt.show()