import numpy as np
import random
import torch, torch.nn as nn
from torch.autograd import Variable
from torch import autograd
import time
import datetime
import os
from matplotlib import pyplot as plt

import sys
sys.path.append('ebm-wgan')
from utils import prepare_train_batches, sample_fake_data, visualize_fake_data_projection

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
device_default = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Evolution(object):
    def __init__(self, means, sigma=0.05):
        self.means = means
        self.sigma = sigma
        self.mode_std = []
        self.high_quality_rate = []
        self.jsd = []

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

    def invoke(self, X_gen):
        assignment = Evolution.make_assignment(X_gen, self.means, self.sigma)
        mode_std = Evolution.compute_mode_std(X_gen, assignment)
        self.mode_std.append(mode_std.item())
        h_q_r = Evolution.compute_high_quality_rate(assignment)
        self.high_quality_rate.append(h_q_r.item())
        jsd = Evolution.compute_jsd(assignment)
        self.jsd.append(jsd.item())


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


def epoch_visualization(X_train, generator, 
                        use_gradient_penalty, 
                        discriminator_mean_loss_arr, 
                        epoch, Lambda,
                        generator_mean_loss_arr, 
                        path_to_save,
                        batch_size_sample = 10000,
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
                label = f'discriminator loss = js')
    axs[1].plot(generator_mean_loss_arr, 'r', label = 'generator loss')
    axs[0].legend()
    axs[1].legend()
    cur_time = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
    plot_name = cur_time + f'_gan_losses_{epoch}_epoch.pdf'
    path_to_plot = os.path.join(path_to_save, plot_name)
    fig.savefig(path_to_plot)

    if proj_list is None:
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
            plot_name = cur_time + f'_wgan_sampling_{epoch}_epoch.pdf'
            path_to_plot = os.path.join(path_to_save, plot_name)
            plt.savefig(path_to_plot)

        else:
            plt.show()

    else:
        fake_data = generator.sampling(batch_size_sample).data.cpu().numpy()
        title = "Training and generated samples"
        mode = f"{epoch}_epoch"
        for i in range(len(proj_list)):
            visualize_fake_data_projection(fake_data, X_train, path_to_save, 
                                           proj_list[i][0], proj_list[i][1],
                                           title,
                                           mode)


def train_gan(X_train,
               X_train_batches, 
               generator, g_optimizer, 
               discriminator, d_optimizer,
               path_to_save,
               batch_size = 256,
               device = device_default,
               num_epochs = 20000, 
               num_epoch_for_save = 100,
               batch_size_sample = 5000
               ):

    adversarial_loss = torch.nn.BCELoss()
    adversarial_loss = adversarial_loss.to(device)           

    generator_loss_arr = []
    generator_mean_loss_arr = []
    discriminator_loss_arr = []
    discriminator_mean_loss_arr = []
    path_to_save_models = os.path.join(path_to_save, 'models')
    path_to_save_plots = os.path.join(path_to_save, 'plots')

    try:
        for epoch in range(num_epochs):
            print(f"Start epoch = {epoch}")

            start_time = time.time()

            for batch_id, real_data in enumerate(X_train_batches):
                if (real_data.shape[0] != batch_size):
                    continue

                #real_data = torch.FloatTensor(real_data, device=device)
                real_data = real_data.to(device)

                g_optimizer.zero_grad()

                # Do an update
                noise = generator.make_hidden(batch_size)
                noise = noise.to(device)
                fake_data = generator(noise)

                #generator_loss = discriminator(fake_data).mean()
                valid = torch.full((real_data.shape[0], ), 1, dtype=real_data.dtype, device=device)
                #valid = autograd.Variable(torch.FloatTensor(real_data.shape[0], 1, device=device).fill_(1.0), requires_grad=False)

                g_loss = adversarial_loss(discriminator(fake_data)[:, 0], valid)
                generator_loss_arr.append(g_loss.data.cpu().numpy())

                g_loss.backward()
                g_optimizer.step()

                d_optimizer.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                fake = torch.full((real_data.shape[0], ), 0, dtype=real_data.dtype, device=device)

                real_loss = adversarial_loss(discriminator(real_data)[:, 0], valid)
                fake_loss = adversarial_loss(discriminator(fake_data.detach())[:, 0], fake)
                d_loss = (real_loss + fake_loss) / 2

                discriminator_loss_arr.append(d_loss.data.cpu().numpy())

                d_loss.backward()
                d_optimizer.step()

            end_time = time.time()
            calc_time = end_time - start_time
            discriminator_mean_loss_arr.append(np.mean(discriminator_loss_arr[-len(X_train_batches):]))
            discriminator_loss_arr = []
            generator_mean_loss_arr.append(np.mean(generator_loss_arr[-len(X_train_batches):]))
            generator_loss_arr = []
            print("Epoch {} of {} took {:.3f}s".format(
                   epoch + 1, num_epochs, calc_time))
            print("Discriminator last mean loss: \t{:.6f}".format(
                   discriminator_mean_loss_arr[-1]))
            print("Generator last mean loss: \t{:.6f}".format(
                   generator_mean_loss_arr[-1])) 
            if epoch % num_epoch_for_save == 0:
               # Visualize
                epoch_visualization(X_train, generator, 
                                   False, 
                                   discriminator_mean_loss_arr, 
                                   epoch, None,
                                   generator_mean_loss_arr, 
                                   path_to_save_plots,
                                   batch_size_sample)

                cur_time = f'{epoch}' #datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')

                discriminator_model_name = cur_time + '_discriminator.pth'
                generator_model_name = cur_time + '_generator.pth'

                path_to_discriminator = os.path.join(path_to_save_models, discriminator_model_name)
                path_to_generator = os.path.join(path_to_save_models, generator_model_name)

                torch.save(discriminator.state_dict(), path_to_discriminator)
                torch.save(generator.state_dict(), path_to_generator)
                
    except KeyboardInterrupt:
        pass
