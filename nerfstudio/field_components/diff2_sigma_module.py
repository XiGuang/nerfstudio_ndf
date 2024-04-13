from torch import Tensor, nn
import torch


class Diff2SigmaModule(nn.Module):
    def __init__(self, params_init=None, beta_min=0.0001):
        super().__init__()
        if params_init is None:
            params_init = {"beta": 0.1}
        self.beta_min = torch.tensor(beta_min).cuda()
        for p in params_init:
            param = nn.Parameter(torch.tensor(params_init[p]))
            setattr(self, p, param)

    def density_func(self, diff_density: Tensor, beta=None):
        if beta is None:
            beta = self.get_beta()

        return 1e4 * beta / (1 + torch.exp(-(diff_density - 0.8) * 10))

    def get_beta(self):
        beta = self.beta.abs() + self.beta_min
        return beta

    def forward(self, sdf, beta=None):
        return self.density_func(sdf, beta=beta)
