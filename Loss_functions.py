"""
This file is from the MIT-licensed repository:

    Learning-based Design of Luenberger Observers for Nonlinear Systems
    Umar Niazi, John Cao, Xudong Sun, Amritam Das, and Karl Johansson
    Version: 1.0.0
    Released: 2022-10-04
    Repository: https://github.com/Mudhdhoo/ACC_KKL_Observer
"""

import torch
from torch import nn

class PDELoss(nn.Module):
    def __init__(self, M, K, system, loss_calculator, lmbda, reduction='mean'):
        super(PDELoss, self).__init__()
        self.M = M
        self.K = K
        self.system = system
        self.loss_calc = loss_calculator
        self.reduction = reduction
        self.lmbda = lmbda

    def forward(self, x, y, z_hat):
        loss = self.loss_calc.calc_pde_loss_xz(x, y, z_hat, self.system, self.M, self.K, self.reduction)
        return self.lmbda * loss


class PdeLoss_xz(nn.Module):
    def __init__(self, M, K, system, loss_calculator, lmbda, reduction = 'mean'):
        super(PdeLoss_xz, self).__init__()
        self.M = M
        self.K = K
        self.system = system
        self.loss_calc = loss_calculator
        self.reduction = reduction
        self.lmbda = lmbda

    def forward(self, x, y, z_hat, return_per_point=False):
        loss = self.loss_calc.calc_pde_loss_xz(x, y, z_hat, self.system, self.M, self.K, self.reduction, return_per_point=return_per_point)
        return self.lmbda*loss

class MSELoss(nn.Module):
    def __init__(self, loss_calculator):
        super(MSELoss, self).__init__()
        self.loss_calc = loss_calculator
        self.lmbda = 1.0

    def forward(self, x_hat=None, z_hat=None, x=None, z=None):
        if x_hat is not None and x is not None and z_hat is None and z is None:
            # Inverse map: z -> x
            return self.loss_cal.calc_backward_loss(x_hat, x)
        elif z_hat is not None and z is not None and x_hat is None and x is None:
            return self.loss_calc.calc_forward_loss(z_hat, z)
        else:
            raise ValueError("Invalid combination of inputs")
        