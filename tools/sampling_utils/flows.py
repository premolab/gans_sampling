import torch
import torch.nn as nn
import torch.nn.functional as F

from pyro.distributions.transforms import AffineCoupling
from pyro.nn import DenseNN


class RNVP(nn.Module):
    def __init__(self, num_flows, dim, flows = None):
        super().__init__()
        split_dim = dim // 2
        param_dims = [dim - split_dim, dim - split_dim]
        if flows is not None:
            self.flow = nn.ModuleList(flows)
        else:
            hypernet = DenseNN(split_dim, [2 * dim], param_dims)
            self.flow = nn.ModuleList(
                [AffineCoupling(split_dim, hypernet) for _ in range(num_flows)])

        even = [i for i in range(0, dim, 2)]
        odd = [i for i in range(1, dim, 2)]
        reverse_eo = [i // 2 if i % 2 == 0 else (i // 2 + len(even)) for i in range(dim)]
        reverse_oe = [(i // 2 + len(odd)) if i % 2 == 0 else i // 2 for i in range(dim)]
        self.register_buffer('eo', torch.tensor(even + odd, dtype=torch.int64))
        self.register_buffer('oe', torch.tensor(odd + even, dtype=torch.int64))
        self.register_buffer('reverse_eo', torch.tensor(reverse_eo, dtype=torch.int64))
        self.register_buffer('reverse_oe', torch.tensor(reverse_oe, dtype=torch.int64))


    def to(self, *args, **kwargs):
        """
        overloads to method to make sure the manually registered buffers are sent to device
        """
        self = super().to(*args, **kwargs)
        self.eo = self.eo.to(*args, **kwargs)
        self.oo = self.oe.to(*args, **kwargs)
        self.reverse_eo = self.reverse_eo.to(*args, **kwargs)
        self.reverse_oe = self.reverse_oe.to(*args, **kwargs)
        return self


    def permute(self, z, i, reverse=False):
        if not reverse:
            if i % 2 == 0:
                z = torch.index_select(z, 1, self.eo)
            else:
                z = torch.index_select(z, 1, self.oe)
        else:
            if i % 2 == 0:
                z = torch.index_select(z, 1, self.reverse_eo)
            else:
                z = torch.index_select(z, 1, self.reverse_oe)
        return z


    def forward(self, z):

        log_jacob = torch.zeros_like(z[:, 0], dtype=torch.float32)
        for i, current_flow in enumerate(self.flow):
            z = self.permute(z, i)
            z_new = current_flow(z)
            log_jacob += current_flow.log_abs_det_jacobian(z, z_new)
            z_new = self.permute(z_new, i, reverse=True)
            z = z_new
        return z, log_jacob


    def inverse(self, z):
        log_jacob = torch.zeros_like(z[:, 0], dtype=torch.float32)
        n = len(self.flow)-1
        for i, current_flow in enumerate(self.flow[::-1]):
            z = self.permute(z, n-i)
            z_new = current_flow._inverse(z)
            log_jacob -= current_flow.log_abs_det_jacobian(z_new, z)
            z_new = self.permute(z_new, n-i, reverse=True)
            z = z_new
        return z, log_jacob
