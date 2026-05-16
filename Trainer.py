"""
Portions of this file were adapted from the MIT-licensed repository:

    Learning-based Design of Luenberger Observers for Nonlinear Systems
    Umar Niazi, John Cao, Xudong Sun, Amritam Das, and Karl Johansson
    Version: 1.0.0
    Released: 2022-10-04
    Repository: https://github.com/Mudhdhoo/ACC_KKL_Observer
"""

import torch
import Loss_Calculator as L
from Loss_functions import *
from typing import TYPE_CHECKING, Optional
import torch.nn as nn
import torch.optim as optim
from Dataset import DataSet
import numpy as np
from NN import Main_Network
import time
from smt.sampling_methods import FullFactorial
from Loss_Calculator import Loss_Calculator

class Trainer:
    def __init__(self, dataset, epochs, optimizer, net, loss_fn, batch_size, lmbda,
                 method, shuffle=True, scheduler=None, reduction='mean', mode='forward',
                 net_type='simple_net', reduce_data=0, adaptive_sampling=False,
                 adapt_ratio=0.2, num_adapt_points=1000):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = dataset
        self.trainset = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        self.epochs = epochs
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.reduce_data = reduce_data
        self.net = net.to(self.device)
        self.mode = mode
        self.loss_fn = loss_fn
        self.loss_calulator = L.Loss_Calculator(loss_fn, self.net, self.dataset, self.device, method, self.mode)
        self.normalizer = getattr(net, "normalizer", None)
        self.reduction = reduction
        self.pde1 = PdeLoss_xz(self.dataset.M, self.dataset.K, self.dataset.system,
                                self.loss_calulator, lmbda, reduction=self.reduction)
        self.with_pde = False if method == 'supervised_NN' else True
        self.net_type = net_type

        # Adaptive sampling parameters
        self.adaptive_sampling = adaptive_sampling
        self.adapt_ratio = adapt_ratio          # fraction of PDE points to focus on
        self.num_adapt_points = num_adapt_points  # number of new collocation points generated
        print('Device: ', self.device)

    def generate_adaptive_points(self, n_total, n_candidates = 10000, frac_adaptive = 0.8):
        start_time = time.time()
        n_adaptive = int(n_total * frac_adaptive)
        n_uniform = n_total - n_adaptive
        box_size = 2.
        limits = np.array([[-box_size, box_size], [-box_size, box_size]])
        x_cand = torch.from_numpy(FullFactorial(xlimits = limits)(n_candidates)).float().to(self.device)
        x_cand.requires_grad_()
        y_ph = x_cand[:, 1]
        with torch.no_grad():
            self.net.eval()
            z_hat_ph = self.net(x_cand)[0]
            pde_err = self.pde1(x_cand, y_ph, z_hat_ph, return_per_point=True).detach().reshape(x_cand.shape[0], self.dataset.system.z_size)
            # print(pde_err)
            pde_losses = torch.norm(pde_err, dim=1)
            # print(pde_losses)
            self.net.train()
            # print("eval: ", time.time()-start_time)
            start_time = time.time()
            _, idx = torch.topk(pde_losses, n_adaptive)
            x_adaptive = x_cand[idx]
            y_adaptive = x_adaptive[:, 1]
            # print("find max: ", time.time()-start_time)
            # 4. Uniform random points
            limits = torch.from_numpy(limits).float().to(self.device)
            low = limits[:, 0]   # [-2, -3]
            high = limits[:, 1]  # [2, 5]

            x_uni = low + (high - low) * torch.rand(n_uniform, self.dataset.system.x_size).to(self.device)
            y_uni = x_uni[:, 1]

            # 5. Combine
            x_new = torch.cat([x_adaptive, x_uni], dim=0)
            y_new = torch.cat([y_adaptive, y_uni], dim=0)

        return x_new, y_new

    def train(self, forward_net=None):
        MSE = MSELoss(self.loss_calulator)
        inv_lmbda = 0.1
        for epoch in range(self.epochs):
            loss_sum = 0
            pde_loss_sum = 0
            for idx, data in enumerate(self.trainset):
                x, z, y, x_ph, y_ph = data
                x, z, y = x.to(self.device), z.to(self.device), y.to(self.device)

                if self.with_pde:
                    # if (epoch+1)%2==0: x_ph, y_ph = x_ph.to(self.device), y_ph.to(self.device)
                    # else: x_ph, y_ph = self.generate_adaptive_points(n_total=x.shape[0], n_candidates=100)
                    x_ph, y_ph = x_ph.to(self.device), y_ph.to(self.device)

                if self.reduce_data != 0:
                    x = x[::self.reduce_data]
                    z = z[::self.reduce_data]
                    y = y[::self.reduce_data]

                self.optimizer.zero_grad()
                if hasattr(self.net, "mode"): self.net.mode = 'normal'
                if self.normalizer is not None:
                    label_x = self.normalizer.Normalize(x, mode='normal').float()
                    label_z = self.normalizer.Normalize(z, mode='normal').float()

                # --------------- Forward Mode -----------------
                if self.mode == 'forward':
                    if self.net_type == 'simple_net':
                        z_hat, norm_z_hat = self.net(x)
                        loss_normal = MSE(z_hat=norm_z_hat, z=label_z)
                        if self.with_pde:
                            if hasattr(self.net, "mode"): self.net.mode = 'physics'
                            z_hat_ph = self.net(x_ph)[0]
                            loss_pde1 = self.pde1(x_ph, y_ph, z_hat_ph)
                            J = self.loss_calulator.calc_J(x_ph)
                            J_inv_loss = self.loss_calulator.invertibility_loss(J)
                            loss = loss_normal + loss_pde1+inv_lmbda*J_inv_loss
                        else:
                            loss = loss_normal
                    elif self.net_type == "flow":
                        z_hat = self.net(x)
                        loss_normal = MSE(z_hat=z_hat, z=z)
                        if self.with_pde:
                            if hasattr(self.net, "mode"): self.net.mode = 'physics'
                            z_hat_ph = self.net(x_ph)
                            loss_pde1 = self.pde1(x_ph, y_ph, z_hat_ph)
                            loss = loss_normal + loss_pde1
                        else:
                            loss = loss_normal
                # --------------- Backward Mode -----------------
                else:
                    if self.loss_calulator.method == 'unsupervised_AE':
                        assert forward_net is not None, "Forward network must be passed for AE training."
                        loss = self.loss_calulator.calc_roundtrip_loss(x, forward_net)
                    else:
                        x_hat, norm_x_hat = self.net(z)
                        if self.normalizer is not None:
                            label_x = self.normalizer.Normalize(x, mode='normal').float()
                        else:
                            label_x = x
                        loss = self.loss_calulator.calc_backward_loss(norm_x_hat, label_x)

                loss_sum += loss
                if self.with_pde: pde_loss_sum += loss_pde1
                loss.backward()
                self.optimizer.step()

            training_loss = (loss_sum / (idx + 1)).item()
            if self.with_pde: training_pde_loss = (pde_loss_sum / (idx + 1))
            else: training_pde_loss = None

            if self.scheduler:
                self.scheduler.step(training_loss)

            print(f"Epoch: {epoch + 1}, Loss: {training_loss:.6f}")
            if self.with_pde: 
                print(f"Epoch: {epoch + 1}, PDE Loss: {training_pde_loss:.6f}")
                # print(epoch+1)
                # if (epoch+1) % 4==0:
                #     self.dataset.x_data_ph, self.dataset.output_data_ph = self.generate_adaptive_points(n_total=self.dataset.x_data_ph.shape[0])
        return training_pde_loss
    
    def mixed_adaptive_sample_2d(self, model, n_candidates, n_total, limits, frac_adaptive=0.8, device="cuda"):
        n_adaptive = int(n_total * frac_adaptive)
        n_uniform = n_total - n_adaptive

        # 1. Candidate pool
        x_cand = torch.from_numpy(FullFactorial(xlimits = limits)(n_candidates)).float().to(device)
        x_cand.requires_grad_()
        print(x_cand)

        # 2. PDE residuals on candidates
        with torch.no_grad():
            model.eval()
            res = pde_residual(x_cand, model, self.dataset.system.torch_function_batch, self.dataset.system.output_batch, self.dataset.M, self.dataset.K)
            model.train()
            print(res)

        # 3. Select top residual points
        print(res.shape)
        print(n_adaptive)
        _, idx = torch.topk(res, n_adaptive)
        x_adaptive = x_cand[idx]
        y_adaptive = x_adaptive[:, 1]

        # 4. Uniform random points
        low = limits[:, 0]   # [-2, -3]
        high = limits[:, 1]  # [2, 5]

        x_uni = low + (high - low) * torch.rand(n_uniform, 2)
        y_uni = x_uni[:, 1]

        # 5. Combine
        x_new = torch.cat([x_adaptive, x_uni], dim=0)
        y_new = torch.cat([y_adaptive, y_uni], dim=0)

        return x_new, y_new
    
    def train_lbfgs(self, x_ph, y_ph):
        # Fine-tuning optimizer
        lbfgs = optim.LBFGS(
            self.net.parameters(),
            lr=1.0,
            max_iter=500,
            history_size=50,
            tolerance_grad=1e-8,
            tolerance_change=1e-9,
            line_search_fn="strong_wolfe"
        )
        # ---- L-BFGS fine-tuning ----
        def closure():
            lbfgs.zero_grad()
            z_hat_ph = self.net(x_ph)[0]
            loss_pde1 = self.pde1(x_ph, y_ph, z_hat_ph)
            L_pde = (loss_pde1**2).mean()

            loss = L_pde
            loss.backward()
            return loss

        print("Starting L-BFGS...")
        # training loop (often just a few steps are enough after Adam)
        for step in range(2):  # each step does up to max_iter LBFGS iters
            loss = lbfgs.step(closure)
            print(f"Step {step}, Loss {loss.item()}")
        print(f"[L-BFGS] Final Loss {loss.item():.3e}")

def pde_residual(x, T_net, f, h, A, b):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    A = torch.as_tensor(A, dtype=x.dtype, device=device)
    b = torch.as_tensor(b, dtype=x.dtype, device=device)
    T_val = T_net(x)        # [batch, n]
    f_val = f(x)            # [batch, dim]
    h_val = h(x).unsqueeze(1)
    jac_list = []
    T_val.requires_grad_(True)
    x.requires_grad_(True)
    for i in range(T_val.shape[1]):
        grad_i = torch.autograd.grad(
            T_val[:, i].sum(), x, create_graph=True
        )[0]  # [batch, dim]
        jac_list.append(grad_i)
    J = torch.stack(jac_list, dim=1)  # [batch, n, dim]

    dTfx = torch.bmm(J, f_val.unsqueeze(-1)).squeeze(-1)  # [batch, n]

    residual = dTfx - (T_val @ A.T) - (h_val * b.T)

    return residual