"""
This file is from the MIT-licensed repository:

    Learning-based Design of Luenberger Observers for Nonlinear Systems
    Umar Niazi, John Cao, Xudong Sun, Amritam Das, and Karl Johansson
    Version: 1.0.0
    Released: 2022-10-04
    Repository: https://github.com/Mudhdhoo/ACC_KKL_Observer
"""

import torch
import numpy as np
import data_generation as data
from torch.autograd.functional import jacobian
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class System_z:
    def __init__(self, M, K, system):
        self.M = M
        self.K = K
        self.is_autonomous = True if system.input is None else False
        self.y_size = system.y_size
    
    def z_dynamics(self):
        if self.is_autonomous:
            if self.y_size > 1:
                z_dot = lambda y, z: np.matmul(self.M, z) + np.matmul(self.K, np.expand_dims(y, 1))
            else:
                z_dot = lambda y, z: np.matmul(self.M, z) + self.K * y
        else:
            if self.y_size > 1:
                z_dot = lambda y, q, z: np.matmul(self.M, z) + np.matmul(self.K, np.expand_dims(y, 1)) + q
            else:
                z_dot = lambda y, q, z: np.matmul(self.M, z) + self.K * y + q
        
        return z_dot
    
class Observer:
    def __init__(self, system, z_system, decoder, a, b, N, init_z_zero=True, encoder=None, ic_z = None):
        self.system = system
        self.z_system = z_system
        self.ic_z = ic_z
        self.f = z_system.z_dynamics()
        self.a = a
        self.b = b
        self.N = N
        self.init_z_zero = init_z_zero

        if encoder is not None:
            self.T = encoder
            self.T_inv = decoder
        else:
            self.T = None
            self.T_inv = decoder
    
    def simulate_NA(self, ic, u0, g):
        x, y, t = self.system.generate_data(ic, self.a, self.b, self.N)
        x = torch.from_numpy(np.reshape(x, (self.N+1, self.system.x_size))).to(device)
        u = self.system.input
        size_z = self.z_system.M.shape[0]
        h = (self.b - self.a) / self.N

        z = [[0] * size_z]
        v = np.array(z).T

        y = np.squeeze(y)
        if y.ndim > 2:
            y = y[1:, :]
        else:
            y = np.delete(y, 0)

        for idx, output in enumerate(y):
            if hasattr(self.T_inv, 'predict'):
                x_hat = torch.from_numpy(self.T_inv.predict(np.array([z[-1]]))).float().squeeze().to(device)
            else:
                x_hat = self.T_inv(torch.tensor(z[-1]).float().to(device))
            u_sub_u0 = u(t[idx].item())[0] - u0(t[idx].item())[0]
            dTdx = jacobian(self.T, x_hat).numpy()
            dTdx_mul_g = np.matmul(dTdx, g)

            q = dTdx_mul_g * u_sub_u0

            k1 = self.f(output, q, v)
            k2 = self.f(output, q, v + h/2 * k1)
            k3 = self.f(output, q, v + h/2 * k2)
            k4 = self.f(output, q, v + h * k3)

            v = v + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
            z.append(np.reshape(v.T, size_z).tolist())

        z = torch.from_numpy(np.array(z)).float().to(device)
        if hasattr(self.T_inv, 'predict'):
            x_hat = torch.from_numpy(self.T_inv.predict(z.numpy())).float().to(device)
        else:
            x_hat = self.T_inv(z)

        error = torch.abs(x - x_hat).to(device)
        return x, x_hat, t, error


    def simulate(self, ic, noise_mean=0, noise_std=0.3, add_noise=False, perturb_std=0.0):
        # Step 1: Generate true state/output trajectories
        x_raw, y, t = self.system.generate_data(ic, self.a, self.b, self.N)

        # Handle tuple-valued system.x_size safely
        x_size = self.system.x_size[0] if isinstance(self.system.x_size, tuple) else self.system.x_size
        x = torch.from_numpy(np.reshape(x_raw, (self.N+1, x_size))).float().to(device)

        if add_noise:
            np.random.seed(123)
            noise = np.random.normal(noise_mean, noise_std, y.shape)
            y = y + noise

        # Step 2: Observer simulation using z dynamics
        z_size = self.system.z_size[0] if isinstance(self.system.z_size, tuple) else self.system.z_size

        if self.init_z_zero:
            ic_z = -10 * np.ones([1, z_size])
        else:
            if self.T is None:
                raise ValueError("Cannot initialize z from x0 because model has no encoder T.")
            if hasattr(self.T, 'predict'):
                ic_np = ic if isinstance(ic, np.ndarray) else ic.detach().cpu().numpy()
                ic_z = self.T.predict(ic_np)
            else:
                with torch.no_grad():
                    ic_z = self.T(torch.from_numpy(ic).float())[0].detach().cpu().numpy()
        if self.ic_z is not None:
            ic_z = self.ic_z

        ic_z += np.random.normal(scale=perturb_std, size=ic_z.shape)

        # Integrate KKL observer
        z_np = data.KKL_observer_data(self.z_system.M, self.z_system.K, y, self.a, self.b, ic_z, self.N)
        z = torch.from_numpy(z_np).view(self.N+1, z_size).float().to(device)

        # Step 3: Apply decoder (T^*) to estimate x
        try:
            if hasattr(self.T_inv, 'predict'):
                z_np = z.detach().cpu().numpy() if isinstance(z, torch.Tensor) else np.array(z)
                x_hat_pred = self.T_inv.predict(z_np)

                if isinstance(x_hat_pred, np.ndarray):
                    x_hat = torch.from_numpy(x_hat_pred).float().to(device)
                elif isinstance(x_hat_pred, torch.Tensor):
                    x_hat = x_hat_pred.float().to(device)
                else:
                    raise TypeError(f"Unexpected output type from predict: {type(x_hat_pred)}")
            elif hasattr(self.T_inv, 'flow'):
                with torch.no_grad():
                    x_hat = self.T_inv.inverse(z.to(device)).to(device)  # z -> x, torch model returns (output, norm_output)
            else:
                with torch.no_grad():
                    x_hat = self.T_inv(z.to(device))[0].to(device)  # z -> x, torch model returns (output, norm_output)

        except Exception as e:
            raise RuntimeError(f"Decoder error: {e}")

        # Step 4: Error calculation
        error = torch.abs(x - x_hat)
        return x, x_hat, t, error



    def sim_multi(self, ic_samples, add_noise=False, perturb_std=0.0):
        avr_error = 0
        errors = []
        x_traj = []
        x_hat_traj = []
        for idx, ic in enumerate(ic_samples):
            x, x_hat, time, error = self.simulate(ic, add_noise=add_noise, perturb_std=perturb_std)
            avr_error += error
            errors.append(error.cpu().numpy())
            x_traj.append(x.cpu().numpy())
            x_hat_traj.append(x_hat.cpu().numpy())
        avr_error = avr_error / idx

        return np.array(x_traj), np.array(x_hat_traj), np.array(errors), avr_error, time

    def calc_gen_metric(self, train_ic, test_ic):
        GE = []
        GE_matrix = []
        p = len(train_ic)
        tau = self.N
        train_ic = np.expand_dims(train_ic, axis=1)
        x_train, x_hat_train, _, _, time1 = self.sim_multi(train_ic)

        train_error = 0
        for true, est in zip(x_train, x_hat_train):
            sum = 0
            for x, x_hat in zip(true, est):
                error_norm = np.linalg.norm(x - x_hat)**2
                true_norm = np.linalg.norm(x)**2
                sum += error_norm / true_norm
            train_error += sum / tau

        train_error_av = train_error / p

        if self.system.x_size == 2:
            for circle in test_ic:
                x_test, x_hat_test, _, _, time2 = self.sim_multi(circle)
                sum_circle = 0
                GE_matrix_col = []
                for true, est in zip(x_test, x_hat_test):
                    error_norm = np.linalg.norm(true - est, axis=1)**2
                    true_norm = np.linalg.norm(true, axis=1)**2
                    trajectory_average = np.sum(error_norm / true_norm) / tau
                    sum_circle += trajectory_average
                    GE_matrix_col.append(trajectory_average)
                metric = np.abs((sum_circle / len(circle)) - train_error_av)
                GE.append(metric)
                GE_matrix.append(GE_matrix_col)
        elif self.system.x_size == 3:
            for sphere in test_ic:
                sum_sphere = 0
                GE_matrix_col = []
                ic = sphere.reshape(-1, 1, 3)
                x_test, x_hat_test, _, _, _ = self.sim_multi(ic)
                for true, est in zip(x_test, x_hat_test):
                    error_norm = np.linalg.norm(true - est, axis=1)**2
                    true_norm = np.linalg.norm(true, axis=1)**2
                    trajectory_average = np.sum(error_norm / true_norm) / tau
                    sum_sphere += trajectory_average
                    GE_matrix_col.append(trajectory_average)
                metric = np.abs((sum_sphere / len(ic)) - train_error_av)
                GE.append(metric)
                GE_matrix.append(GE_matrix_col)

        return GE, np.array(GE_matrix)