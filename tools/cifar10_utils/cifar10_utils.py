import matplotlib.pyplot as plt
import numpy as np
import torch
import random

def save_predictions_on_test(G,
                             real_dataloader,
                             name_fake_test,
                             name_real_test,
                             z_dim,
                             device, 
                             random_seed):
    fake_list = []
    real_list = []
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    for  i, data_real in enumerate(real_dataloader, 0):
        batch_real = data_real[0]
        batch_size = batch_real.shape[0]
        fixed_noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
        fake_images = G(fixed_noise)
        fake_norm_np = ((1. + fake_images)/2).detach().cpu().numpy()
        real_norm_np = ((1. + batch_real)/2).detach().cpu().numpy()
        fake_list.append(fake_norm_np)
        real_list.append(real_norm_np)

    fake_np = np.concatenate(fake_list)
    real_np = np.concatenate(real_list)

    np.save(name_fake_test, fake_np)
    np.save(name_real_test, real_np)

def plot_images(images_torch, figsize = (10, 10)):
    batch_size_sample = images_torch.shape[0]
    numpy_images = images_torch.detach().cpu().numpy().transpose(0, 2, 3, 1)
    numpy_images = (numpy_images - numpy_images.min())/(numpy_images.max() - numpy_images.min())
    nrow = int(batch_size_sample**0.5)
    fig = plt.figure(figsize=figsize)
    axes = fig.subplots(nrow, nrow)
    for k in range(batch_size_sample):
        i = k // nrow
        j = k % nrow
        #axes[i][j].imshow(np.clip(numpy_images[k], 0, 1))
        axes[i][j].imshow(numpy_images[k])
        axes[i][j].axis('off')
    plt.show()
