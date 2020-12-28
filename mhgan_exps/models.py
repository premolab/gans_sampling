import numpy as np
import random
import torch, torch.nn as nn
from torch.nn import functional as F


__all__ = [
    "Generator_2d", "Discriminator_2d", "", "weights_init_1", "weights_init_2",
    "Discriminator_DCGAN", "Generator_DCGAN",
]

device_default = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Generator_2d(nn.Module):
    def __init__(self, 
                 n_dim = 2,
                 n_hidden = 100, 
                 device = device_default, 
                 non_linear = nn.ReLU()):
        super().__init__()
        self.non_linear = non_linear
        self.device = device
        self.n_hidden = n_hidden
        self.n_dim = n_dim
        self.layers = nn.ModuleList([nn.Linear(self.n_dim, self.n_hidden),
                                     nn.Linear(self.n_hidden, self.n_hidden), 
                                     nn.Linear(self.n_hidden, self.n_hidden),
                                     nn.Linear(self.n_hidden, 2)])
        self.num_layer = len(self.layers)
        for i in range(4):
           std_init = 0.8 * (2/self.layers[i].in_features)**0.5
           torch.nn.init.normal_(self.layers[i].weight, std = std_init)

    def forward(self, z):
        for i in range(self.num_layer - 1):
            z = self.non_linear((self.layers[i])(z))
        z = (self.layers[self.num_layer - 1])(z)
        return z

    def make_hidden(self, batch_size, device=device_default):
        return torch.randn(batch_size, self.n_dim, device=device)

    def sampling(self, batch_size, device=device_default):
        z = self.make_hidden(batch_size, device)
        return self.forward(z)


class Discriminator_2d(nn.Module):
    def __init__(self, 
                 n_hidden = 100,
                 device = device_default, 
                 non_linear = nn.ReLU(),
                 top_nonlin = None):
        super().__init__()
        self.non_linear = non_linear
        self.device = device
        self.n_hidden = n_hidden
        self.layers = nn.ModuleList([nn.Linear(2, self.n_hidden),
                                     nn.Linear(self.n_hidden, self.n_hidden), 
                                     nn.Linear(self.n_hidden, self.n_hidden),
                                     nn.Linear(self.n_hidden, 1)])
        self.num_layer = len(self.layers)
        self.top_nonlin = top_nonlin
        for i in range(4):
           std_init = 0.8 * (2/self.layers[i].in_features)**0.5
           torch.nn.init.normal_(self.layers[i].weight, std = std_init)
      
    def forward(self, z):
        for i in range(self.num_layer - 1):
            z = self.non_linear((self.layers[i])(z))
        z = (self.layers[self.num_layer - 1])(z)
        if self.top_nonlin is None:
            return z
        else:
            z = self.top_nonlin(z)
        return z

        
def weights_init_1(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
        
def weights_init_2(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        std_init = 0.8 * (2/m.in_features)**0.5
        m.weight.data.normal_(0.0, std = std_init)


class Discriminator_DCGAN(nn.Module):
    def __init__(self):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        out = self.main(input)
        out = torch.flatten(out).unsqueeze(1)
        return out


class Generator_DCGAN(nn.Module):
    def __init__(self):
        super().__init__()

        self.n_dim = 100

        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        out = self.main(input)
        return out

    def make_hidden(self, batch_size, device=device_default):
        return torch.randn(batch_size, self.n_dim, 1, 1, device=device)

    def sampling(self, batch_size, device=device_default):
        z = self.make_hidden(batch_size, device)
        return self.forward(z)

#
# def _gan(arch, pretrained, progress):
#     model = Generator()
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
#         model.load_state_dict(state_dict)
#     return model


# def discriminator() -> Discriminator:
#     model = Discriminator()
#     return model


# def cifar10(pretrained: bool = False, progress: bool = True) -> Generator:
#     # r"""GAN model architecture from the
#     # `"One weird trick..." <https://arxiv.org/abs/1511.06434>`_ paper.
#     # Args:
#     #     pretrained (bool): If True, returns a model pre-trained on ImageNet
#     #     progress (bool): If True, displays a progress bar of the download to stderr
#     # """
#     return _gan("cifar10", pretrained, progress)
