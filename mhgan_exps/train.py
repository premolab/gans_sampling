import numpy as np
import random
import torch, torch.nn as nn
import time
from matplotlib import pyplot as plt
from abc import ABC, abstractmethod
from pathlib import Path
from scipy.stats import gaussian_kde

# import sys
# sys.path.append('ebm-wgan/api')

device_default = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class GAN_Trainer(ABC):
    def __init__(self, trainloader, G, g_optimizer, D, d_optimizer, path_to_save=None, **kwargs):
        self.trainloader = trainloader
        self.G = G
        self.g_optimizer = g_optimizer
        self.D = D
        self.d_optimizer = d_optimizer

        self.discriminator_mean_loss_arr = []
        self.generator_mean_loss_arr = []
        self.path_to_save = path_to_save
        self.device = kwargs.get('device', device_default)
        self.n_critic = kwargs.get('n_critic', 1)
        self.n_gen = kwargs.get('n_gen', 1)
        self.use_gradient_penalty = kwargs.get('use_gradient_penalty', False)
        self.Lambda = kwargs.get('Lambda', 0.1)

    @abstractmethod
    def train_step(self, real_data):
        pass

    @torch.no_grad()
    def epoch_visualization(self, epoch, path_to_save):
        self.D.eval()
        self.G.eval()

        subtitle_for_losses = "Training process for discriminator and generator"
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
        axs[0].plot(self.discriminator_mean_loss_arr, 'b', 
                    label = f'discriminator loss')
        axs[1].plot(self.generator_mean_loss_arr, 'r', label = 'generator loss')
        axs[0].legend()
        axs[1].legend()
        #cur_time = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
        plot_name = f'gan_losses_{epoch}_epoch.png'
        path_to_plot = Path(path_to_save, plot_name)
        fig.savefig(path_to_plot)
        plt.close()

    def train(
            self, 
            path_to_save=None,
            batch_size = 256,
            device = device_default,
            num_epochs = 350, 
            num_epoch_for_save = 10,
            batch_size_sample = 10000
                ):

        self.device = device

        path_to_save = path_to_save if path_to_save is not None else self.path_to_save        

        generator_loss_arr = []
        discriminator_loss_arr = []
        path_to_save_models = Path(path_to_save, 'models')
        path_to_save_models.mkdir(exist_ok=True, parents=True)
        path_to_save_plots = Path(path_to_save, 'plots')
        path_to_save_plots.mkdir(exist_ok=True, parents=True)

        try:
            for epoch in range(1, num_epochs + 1):
                print(f"Start epoch = {epoch}")

                start_time = time.time()

                self.G.train()
                self.D.train()

                critic_step = 0

                for batch_id, real_data in enumerate(self.trainloader):
                    if (real_data.shape[0] != batch_size):
                        continue
                    real_data = real_data.to(device)
                    g_loss, d_loss = self.train_step(real_data, critic_step)
                    if critic_step < self.n_critic - 1:
                        critic_step += 1
                    else:
                        critic_step = 0
                    discriminator_loss_arr.append(d_loss.data.cpu().numpy())
                    if g_loss is not None:
                        generator_loss_arr.append(g_loss.data.cpu().numpy())

                end_time = time.time()
                calc_time = end_time - start_time
                self.discriminator_mean_loss_arr.append(np.mean(discriminator_loss_arr[-len(self.trainloader):]))
                discriminator_loss_arr = []
                self.generator_mean_loss_arr.append(np.mean(generator_loss_arr[-len(self.trainloader):]))
                generator_loss_arr = []
                print("Epoch {} of {} took {:.3f}s".format(
                    epoch + 1, num_epochs, calc_time))
                print("Discriminator last mean loss: \t{:.6f}".format(
                    self.discriminator_mean_loss_arr[-1]))
                print("Generator last mean loss: \t{:.6f}".format(
                    self.generator_mean_loss_arr[-1])) 
                if epoch % num_epoch_for_save == 0:
                # Visualize
                    if path_to_save is not None:
                        self.epoch_visualization(epoch, path_to_save_plots)

                    cur_ep = f'{epoch}'
                    discriminator_model_name = cur_ep + '_discriminator.pth'
                    generator_model_name = cur_ep + '_generator.pth'

                    path_to_discriminator = Path(path_to_save_models, discriminator_model_name)
                    path_to_generator = Path(path_to_save_models, generator_model_name)

                    torch.save(self.D.state_dict(), path_to_discriminator)
                    torch.save(self.G.state_dict(), path_to_generator)
                    
        except KeyboardInterrupt:
            pass


class JS_GAN_Trainer(GAN_Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.true_js = kwargs.get('true_js', True)
        self.adversarial_loss = torch.nn.BCELoss().to(self.device)

    def train_step(self, real_data, critic_step=0):
        device = real_data.device 
        batch_size = real_data.shape[0]

        valid = torch.full((real_data.shape[0], ), 1, dtype=real_data.dtype, device=device)
        fake = torch.full((real_data.shape[0], ), 0, dtype=real_data.dtype, device=device)

        if critic_step == 0:
            self.d_optimizer.zero_grad()
        fake_data = self.G.sampling(batch_size, device=device)
        real_loss = self.adversarial_loss(self.D(real_data)[:, 0], valid)
        fake_loss = self.adversarial_loss(self.D(fake_data.detach())[:, 0], fake)
        d_loss = (real_loss + fake_loss) / 2.
        d_loss.backward()
        self.d_optimizer.step()
        if critic_step == self.n_critic - 1:
            self.g_optimizer.zero_grad()
            fake_data = self.G.sampling(batch_size, device=device)
            if self.true_js is True:
                g_loss = -self.adversarial_loss(self.D(fake_data)[:, 0], fake)
            else:
                g_loss = self.adversarial_loss(self.D(fake_data)[:, 0], valid)
            g_loss.backward()
            self.g_optimizer.step()
        else:
            g_loss = None

        # self.g_optimizer.zero_grad()
        # fake_data = self.G.sampling(batch_size, device=device)

        # valid = torch.full((real_data.shape[0], ), 1, dtype=real_data.dtype, device=device)
        # fake = torch.full((real_data.shape[0], ), 0, dtype=real_data.dtype, device=device)

        # if self.true_js is True:
        #     g_loss = -adversarial_loss(self.D(fake_data)[:, 0], fake)
        # else:
        #     g_loss = adversarial_loss(self.D(fake_data)[:, 0], valid)
        # g_loss.backward()
        # self.g_optimizer.step()

        # fake_data = self.G.sampling(batch_size, device=device)
        # self.d_optimizer.zero_grad()
        # real_loss = adversarial_loss(self.D(real_data)[:, 0], valid)
        # fake_loss = adversarial_loss(self.D(fake_data.detach())[:, 0], fake)
        # d_loss = (real_loss + fake_loss) / 2.
        # d_loss.backward()
        # self.d_optimizer.step()

        return g_loss, d_loss


def calc_gradient_penalty(D, real_data, fake_data, batch_size, Lambda = 0.1,
                          device = device_default):
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(real_data.size()).to(device)

    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates = interpolates.to(device)
    interpolates.requires_grad = True
    discriminator_interpolates = D(interpolates)
    ones = torch.ones(discriminator_interpolates.size()).to(device)
    gradients = torch.autograd.grad(outputs = discriminator_interpolates, 
                              inputs = interpolates,
                              grad_outputs = ones,
                              create_graph = True, 
                              retain_graph = True, 
                              only_inputs = True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1.0) ** 2).mean() * Lambda
    return gradient_penalty


class WGAN_Trainer(GAN_Trainer):
    def train_step(self, real_data, critic_step=0):
        device = real_data.device 
        batch_size = real_data.shape[0]

        valid = torch.full((real_data.shape[0], ), 1, dtype=real_data.dtype, device=device)
        fake = torch.full((real_data.shape[0], ), 0, dtype=real_data.dtype, device=device)

        if critic_step == 0:
            self.d_optimizer.zero_grad()
        fake_data = self.G.sampling(batch_size, device=device)
        d_loss =  self.D(fake_data).mean() - self.D(real_data).mean()

        if (self.use_gradient_penalty) and (self.Lambda > 0):
            gradient_penalty = calc_gradient_penalty(self.D, 
                                                        real_data.data, 
                                                        fake_data.data, 
                                                        batch_size,
                                                        self.Lambda)
            d_loss += gradient_penalty

        d_loss.backward()
        self.d_optimizer.step()

        if critic_step == self.n_critic - 1:
            self.g_optimizer.zero_grad()
            fake_data = self.G.sampling(batch_size, device=device)
            g_loss = -self.D(fake_data).mean()
            g_loss.backward()
            self.g_optimizer.step()
        else:
            g_loss = None

        return g_loss, d_loss


class Gaussians_Trainer(GAN_Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size_sample = kwargs.get('batch_size_sample', True)
        self.X_train = np.vstack([x.cpu().numpy() for x in self.trainloader.dataset])

    def plot_discriminator_heatmap(self, epoch, path_to_save):
        n_pts = 300
        batch = torch.FloatTensor(np.mgrid[-2:2:4./n_pts, -2:2:4./n_pts]).reshape(2, -1).T
        discr_batch = self.D(batch.to(self.device))
        heatmap = discr_batch[:, 0].view((n_pts, n_pts)).detach().cpu().numpy()
        plt.figure(figsize=(8, 8))
        plt.imshow(heatmap, cmap='viridis')
        ticks = [f'{x:.2f}' for x in np.linspace(-2, 2, 6)]
        plt.xticks(np.arange(0, n_pts+1, n_pts//5), ticks)
        plt.yticks(np.arange(0, n_pts+1, n_pts//5), ticks)
        plt.colorbar()
        plt.savefig(Path(path_to_save, f'heatmap_{epoch}.png'))
        plt.close()

    def plot_discriminator_scores_dist(self, epoch, path_to_save, X_test):
        nonlin = self.D.top_nonlin
        self.D.top_nonlin = None
        Xs = [self.X_train, X_test]
        labels = ['real', 'GAN']
        dist_space = np.linspace(-10, 10)
        plt.figure(figsize=(9, 7))
        for x, label in zip(Xs, labels):
            x = torch.FloatTensor(x).to(self.device)
            logits = self.D(x)[:, 0]
            ker = gaussian_kde(logits.detach().cpu().numpy())
            plt.plot(dist_space, ker.pdf(dist_space), label=label)

        plt.xlabel('logit D(x\')', fontsize=15)
        plt.ylabel('PDF', fontsize=15)
        plt.title('Scores', fontsize=15)
        plt.legend(fontsize=15)
        plt.grid()
        plt.savefig(Path(path_to_save, f'scores_{epoch}.png'))
        plt.close()

        self.D.top_nonlin = nonlin

    def wrapper(f):
        @torch.no_grad()
        def epoch_visualization(*args):
            self, epoch, path_to_save = args
            f(*args)

            fake_data = self.G.sampling(self.batch_size_sample).data.cpu().numpy()
            plt.figure(figsize=(8, 8))
            plt.xlim(-2., 2.)
            plt.ylim(-2., 2.)
            plt.title("Training and generated samples", fontsize=20)
            plt.scatter(self.X_train[:,:1], self.X_train[:,1:], alpha=0.5, color='gray', 
                        marker='o', label = 'training samples')
            plt.scatter(fake_data[:,:1], fake_data[:,1:], alpha=0.5, color='blue', 
                        marker='o', label = 'samples by G')
            plt.legend()
            plt.grid(True)

            plot_name = f'gan_sampling_{epoch}_epoch.png'
            path_to_plot = Path(path_to_save, plot_name)
            plt.savefig(path_to_plot)
            plt.close()
            
            self.plot_discriminator_heatmap(epoch, path_to_save)
            self.plot_discriminator_scores_dist(epoch, path_to_save, fake_data)

        return epoch_visualization

    epoch_visualization = wrapper(GAN_Trainer.epoch_visualization)


class JS_GAN_Gaussians_Trainer(JS_GAN_Trainer, Gaussians_Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class WGAN_Gaussians_Trainer(WGAN_Trainer, Gaussians_Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
