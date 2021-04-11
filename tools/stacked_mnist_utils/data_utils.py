import torch
from torch.utils import data
from torchvision import datasets, transforms as T
#import argparse
import numpy as np
from pathlib import Path


class StackedMNIST(data.Dataset):
    def __init__(self, mnist_path, transform=T.Compose([T.ToTensor(),
                                                        T.Normalize((0.1307,), (0.3081,))
                                                          ])):
        self.mnist = datasets.MNIST(mnist_path, train=True)
        self.samples = []
        self.transform = transform

    def __len__(self):
        if len(self.samples) == 3:
            return len(self.samples[0])
        else:
            return 0

    def build(self, size: int=50000, random_seed: int=None):
        #self.dataset = []
        self.samples = []
        for i in range(3):
            indices = np.random.choice(len(self.mnist), size=size, replace=True)
            sample = data.Subset(self.mnist, indices)
            self.samples.append(sample)

    def __getitem__(self, index: int):
        tensor = torch.stack([self.transform(x[index][0]) for x in self.samples], 1)
        return tensor

    def back_normalize(self, x):
        return x * 0.3081 + 0.1307

    def _test(self):
        self.build()
        im = self[45]

        from matplotlib import pyplot as plt
        plt.imshow(im.permute(1, 2, 0))
        plt.show()


#StackedMNIST(Path(Path.home(), 'data'))._test()

    