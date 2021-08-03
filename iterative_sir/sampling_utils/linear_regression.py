# Packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import torchvision
import imp
import matplotlib as mpl
from matplotlib import cm
import statsmodels.tsa.stattools as stat
from sklearn.model_selection import KFold
import time
from tqdm import tqdm
import pandas as pd
import torch

from pathlib import Path
from dataclasses import dataclass

from .logistic_regression import preprocess, split, import_csv_dataset


@dataclass        
class RegressionDataset:
    x_train : torch.Tensor
    x_test : torch.Tensor
    y_train : torch.Tensor
    y_test : torch.Tensor

    @property
    def d(self):
        return self.x_train.shape[1]

    @property
    def n(self):
        self.y_train.shape[0]
        

class RegressionDatasetFactory:
    def __init__(self, data_root=None, device='cpu', **kwargs):
        self._instance = None
        self.data_root = data_root
        self.device = device

    def get_dataset(self, name : str, **kwargs) -> RegressionDataset:
        if name == 'covertype':
            x_train, x_test, y_train, y_test = import_covertype(**kwargs)
        elif name == 'breast':
            x_train, x_test, y_train, y_test = import_breast(**kwargs)
        elif name == 'digits':
            x_train, x_test, y_train, y_test = import_digits(**kwargs)
        else:
            # csv
            if not name.endswith('.csv'):
                name = f'{name}.csv'
            path = Path(self.data_root, name)
            x_train, x_test, y_train, y_test = import_csv_dataset(path, **kwargs)

        x_train = torch.tensor(x_train, dtype=torch.float32, device=self.device)
        x_test =  torch.tensor(x_test , dtype=torch.float32, device=self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32, device=self.device)
        y_test =  torch.tensor(y_test , dtype=torch.float32, device=self.device)

        return RegressionDataset(x_train, x_test, y_train, y_test)