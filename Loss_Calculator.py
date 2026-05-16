"""
Portions of this file were adapted from the MIT-licensed repository:

    Learning-based Design of Luenberger Observers for Nonlinear Systems
    Umar Niazi, John Cao, Xudong Sun, Amritam Das, and Karl Johansson
    Version: 1.0.0
    Released: 2022-10-04
    Repository: https://github.com/Mudhdhoo/ACC_KKL_Observer
"""

import torch
from torch.autograd.functional import jacobian
import numpy as np
import normflows as nf


class Loss_Calculator:
    def __init__(self, loss_fn, net, dataset, device, method, mode='forward', normalizer=None):
        self.loss_fn = loss_fn
        self.net = net              # either forward or inverse model
        self.mode = mode    # 'forward' or 'inverse'
        self.device = device
        self.dataset = dataset
        self.method = method
        self.normalizer = normalizer
        
    # Normal loss calculation
    def calc_loss(self, x_hat, z_hat, x, z):
        loss_xz = self.loss_fn(z_hat, z)
        loss_zx = self.loss_fn(x_hat, x)

        if self.method == 'unsupervised_AE':
            loss = loss_zx
        else:
            loss = loss_xz + loss_zx

        return loss
    
    # Normal forward loss calculation for sequential training
    def calc_forward_loss(self, z_hat, z, x=None, y=None, system=None, M=None, K=None, v=1.0, reduction='mean'):
        return self.loss_fn(z_hat, z)

    # Normal backward loss calculation for sequential training
    def calc_backward_loss(self, x_hat, x):
        return self.loss_fn(x_hat, x)
    
    # PDE constrain loss for PINN from x --> z
    def calc_pde_loss_xz(self, x, y, z_hat, system, M, K, reduction = 'mean', return_per_point=False):

        M = torch.from_numpy(M).to(self.device)
        K = torch.from_numpy(K).to(self.device)
        
        # Jacobian
        if hasattr(self.net, "flows"): dTdx = self.calc_J(x, NN='flow')
        elif hasattr(self.net, "beta"): dTdx = self.calc_J(x, NN='elm')
        else: dTdx = self.calc_J(x, NN='net1')
        
        # Computation of f(x)
        f = []
        u = 0
        for state in x:
            #f.append(system.function(u, state.detach().numpy()))
            f.append(system.function(u, state.detach().cpu().numpy()))
        f = torch.from_numpy(np.array(f)).float().to(self.device)
        # dT/dx * f(x)
        dTdx_mul_f = torch.bmm(dTdx, torch.unsqueeze(f,2))

        z_hat = torch.unsqueeze(z_hat, 2)
        M = M.to(torch.float32)
        M_mul_T = torch.matmul(M, z_hat)    # MT(x)
        
        # Check if y elements are scalar
        K = K.to(torch.float32)
        y = y.to(torch.float32)
        if y[0].shape == torch.Size([]):
            K_mul_h = torch.matmul(K, y.view(y.shape[0],1,1))    # Kh(x)
        else:
            y = torch.unsqueeze(y, 2)
            K_mul_h = torch.matmul(K, y)    # Kh(x)
            
        pde = dTdx_mul_f - M_mul_T - K_mul_h    # dT/dx*f(x) - MT(x) - Kh(x) = 0
        if return_per_point: return pde
        loss_batch = torch.linalg.norm(pde, dim = 1)    # Element-wise norm

        # Type of loss reduction
        if reduction == 'mean':
            samples = loss_batch.shape[0]
            loss_pde = torch.sum(loss_batch) / samples
            
        if reduction == 'sum':
            loss_pde = torch.sum(loss_batch)
        
        return loss_pde
    
    def calc_J(self, x, NN='net1'):
        m = x.shape[0]
        if NN == 'net1':
            net = self.net.net
        elif NN == 'net2':
            net = self.net.net2
        else:
            raise ValueError("Unknown network type.")
        
        dTdx = jacobian(net, x, create_graph=False)  
        ind = torch.arange(0, m)
        return dTdx[ind, :, ind, :]  

    def invertibility_loss(self, J, eps=1e-3):
        """
        J: Jacobian [batch, m, n]
        eps: minimum allowed singular value
        returns: scalar loss
        """
        # SVD per batch element
        # (u, s, v) = torch.linalg.svd(J) would work, but heavy
        # For small n, it’s fine.
        s = torch.linalg.svdvals(J)   # shape (batch, min(m,n))
        sigma_min = s.min(dim=1).values
        # penalty if sigma_min < eps
        loss = torch.relu(eps - sigma_min)**2
        return loss.mean()
    
    def calc_roundtrip_loss(self, x):
        with torch.no_grad():
            z_hat = self.net.net1(x)
        x_hat = self.net.net2(z_hat)

        if self.normalizer:
            x = self.normalizer.Normalize(x, mode='normal').float()
            x_hat = self.normalizer.Normalize(x_hat, mode='normal').float()

        return self.loss_fn(x_hat, x)