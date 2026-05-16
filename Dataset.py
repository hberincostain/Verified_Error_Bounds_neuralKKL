"""
Portions of this file were adapted from the MIT-licensed repository:

    Learning-based Design of Luenberger Observers for Nonlinear Systems
    Umar Niazi, John Cao, Xudong Sun, Amritam Das, and Karl Johansson
    Version: 1.0.0
    Released: 2022-10-04
    Repository: https://github.com/Mudhdhoo/ACC_KKL_Observer
"""

import torch
import data_generation as data
import numpy as np
from Systems import System
from smt.sampling_methods import FullFactorial
from bound_cond import generat_Tx_on_traj
from integral_data import generate_T_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DataSet(torch.utils.data.Dataset):
    """
    Dataset class to generate synthetic x, y and z data.
    The set is split up into data for normal loss and data for physics loss.

    ---------------- Parameters ----------------
    system: Systems
        A system instance created from classes within Systems.py

    M: ndarray
        M matrix in the z dynamics.

    K: ndarray
        K matrix in the z dynamics.

    a: int
        Start time of ODE solver.

    b: int
        End time of ODE solver.
    
    N: int
        Number of intervals in [a,b] with step size (b-a)/N

    samples: int
        Numer of initial conditions to be sampled for data generation.

    limits_normal: ndarray
        Limits on the state sample space of initial conditions used to generate
        data to compute the normal loss.

    PINN_sample_mode: int
        Either 'split set' or 'split traj. 
        If 'split set', the samples for the physics datapoints will be generated from a separate set of
        initial conditions. To set a different limit on the state sample space for the physics
        points, use the set_physics_limit method. 
        If 'split traj', These points will be generated from the same initial conditions,
        with every other sample in a given trajectory being assigned as a physics datapoint.
        Default set to 2.

    data_gen_mode: str
        Either 'negative forward', 'backward sim', 'integral', or 'boundary_cond'
        If 'negative forward, start simulation from a given negative time untill 0. Use the outputs of
        the simuluation to obtain z(0).
        If 'backward sim', the system is simulated backwards from 0 to a given negative time. The outputs are
        used to simulate z system forward to obtain z(0).
    """

    def __init__(self, system, M, K, a, b, N, samples, limits_normal, PINN_sample_mode: str = 'split traj', data_gen_mode: str = 'negative forward', ics = None) -> None:
        super().__init__()
        self.M = M
        self.K = K
        self.system = system
        self.a = a
        self.b = b
        self.N = N
        self.samples = samples
        self.limits_normal = limits_normal
        self.PINN_sample_mode = PINN_sample_mode
        self.data_gen_mode = data_gen_mode

        if data_gen_mode == "integral":
            ic = FullFactorial(xlimits = self.limits_normal)(self.samples)
            z, x = generate_T_data(ic, M, K, system, self.N, T = b)
            z = z.to(device)
            y = []
            for x0 in x:
                y.append(system.output(x0))
            y = np.array(y)
            y = torch.from_numpy(y).to(device)
            x = torch.from_numpy(x).to(device)
            self.train_data = x, z, y, np.array([0]), x
            self.data_length = x.shape[0]

        elif data_gen_mode=="boundary_cond":
            if ics == None: x0_list = torch.from_numpy(self.system.sample_ic(self.limits_normal, self.samples, seed=888))
            else: x0_list = ics
            x = []
            z = []
            y = []
            for x0 in x0_list:
                x0 = x0.float().cpu()
                T0 = x0[0].float().cpu()*np.ones(M.shape[0],)
                Tx, trajx, trajy = generat_Tx_on_traj(x0, T0, self.M, self.K, N, b, self.system)
                x.append(torch.tensor(trajx).to(device))
                z.append(torch.tensor(Tx).to(device))
                y.append(torch.tensor(trajy).to(device))

            x = torch.cat(x, dim=0).to(device)
            z = torch.cat(z, dim=0).to(device)
            y = torch.cat(y, dim=0).to(device)
            self.train_data = x, z, y, np.array([0]), x
            self.data_length = x.shape[0]

        else:
            # generate training data
            self.train_data = self.generate_data(seed=800)                              # Generate synthetic data x, z, y
            self.data_length = self.train_data[0].shape[0]*self.train_data[0].shape[1]  # Total number of samples
            x = torch.from_numpy(self.train_data[0]).view(self.data_length, system.x_size).to(device)  # Convert to tensors and reshape
            z = torch.from_numpy(self.train_data[1]).view(self.data_length, system.z_size).to(device)
            
            if self.train_data[2].ndim > 2:
                y = torch.from_numpy(self.train_data[2]).view(self.train_data[2].shape[0]*self.train_data[2].shape[1], self.train_data[2].shape[2]).to(device)     # y data
            else:
                y = torch.from_numpy(self.train_data[2]).view(self.train_data[2].shape[0]*self.train_data[2].shape[1]).to(device)

        if PINN_sample_mode == 'split set':
            # ----------------------- Normal loss data ----------------------- 
            self.x_data = x
            self.z_data = z
            self.output_data = y
            self.ic_normal = self.train_data[4]

            # ----------------------- Physics loss data ----------------------- 
            self.train_data_ph = self.generate_data(seed = 8888)
            self.data_length_ph = self.train_data_ph[0].shape[0]*self.train_data_ph[0].shape[1]         # Total number of samples
            self.x_data_ph = torch.from_numpy(self.train_data_ph[0]).view(self.data_length_ph, system.x_size)
            self.z_data_ph = torch.from_numpy(self.train_data_ph[1]).view(self.data_length_ph, system.z_size)
            self.ic_ph = self.train_data_ph[4]
            
            # check if output is vector or scalar    
            if self.train_data_ph[2].ndim > 2:
                self.output_data_ph = torch.from_numpy(self.train_data_ph[2]).view(self.train_data_ph[2].shape[0]*self.train_data_ph[2].shape[1], self.train_data_ph[2].shape[2])     # y data
            else:
                self.output_data_ph = torch.from_numpy(self.train_data_ph[2]).view(self.train_data_ph[2].shape[0]*self.train_data_ph[2].shape[1])
        
        elif PINN_sample_mode == 'split traj':
            self.data_length = int(self.data_length / 2)
            self.x_data = x[::2]
            self.z_data = z[::2]
            self.output_data = y[::2]
            self.x_data_ph = x[1::2]
            self.z_data_ph = z[1::2]
            self.output_data_ph = y[1::2]
            self.ic = self.train_data[4]
        elif PINN_sample_mode == 'more pde':
            self.x_data = x[::4]
            self.z_data = z[::4]
            self.output_data = y[::4]

            mask = torch.ones_like(x, dtype=torch.bool)
            mask[::4] = False
            self.x_data_ph = x[mask]
            mask = torch.ones_like(z, dtype=torch.bool)
            mask[::4] = False
            self.z_data_ph = z[mask]
            mask = torch.ones_like(y, dtype=torch.bool)
            mask[::4] = False
            self.output_data_ph = y[mask]
            self.ic = self.train_data[4]
        elif PINN_sample_mode == 'no physics':
            self.x_data = x
            self.z_data = z
            self.output_data = y
            self.x_data_ph = x
            self.z_data_ph = y
            self.output_data_ph = y
            self.ic = self.train_data[4]
        
        else:
            raise Exception('Sample mode must be either ''split set'', ''split traj'' or ''no physics''.')

        # ----------------------- Mean and standard deviation -----------------------     
        self.mean_x = torch.mean(self.x_data, dim = 0).to(device)
        self.mean_z = torch.mean(self.z_data, dim = 0).to(device)
        self.mean_output = torch.mean(self.output_data, dim = 0).to(device)
        self.std_x = torch.std(self.x_data, dim = 0).to(device)
        self.std_z = torch.std(self.z_data, dim = 0).to(device)
        self.std_output = torch.std(self.output_data, dim = 0).to(device)

        self.mean_x_ph = torch.mean(self.x_data_ph, dim = 0).to(device)
        self.mean_z_ph = torch.mean(self.z_data_ph, dim = 0).to(device)
        self.mean_output_ph = torch.mean(self.output_data_ph, dim = 0).to(device)
        self.std_x_ph = torch.std(self.x_data_ph, dim = 0).to(device)
        self.std_z_ph = torch.std(self.z_data_ph, dim = 0).to(device)
        self.std_output_ph = torch.std(self.output_data, dim = 0).to(device)
        #-----------------------------------------------------------------------------
        self.time = torch.from_numpy(self.train_data[3]).to(device)
        
    def __len__(self) -> None:
        return self.data_length
    
    
    def generate_data(self, seed: int) -> tuple:
        """
        Generate synthetic state, observer, and output trajectories.
        Handles both 'negative forward' and 'backward sim' modes for z(0) estimation.
        """

        # calculate required backward simulation time based on contraction theory
        t_back = data.calc_neg_t(self.M, 10, 1e-12)
        h = (self.b - self.a) / self.N

        # sample initial conditions using LHS sampling
        ic = self.system.sample_ic(self.limits_normal, self.samples, seed=seed)
        x_data_fw, output_fw, t_fw = self.system.generate_data(ic, self.a, self.b, self.N)

        # simulate output trajectory used to estimate z(0)
        if self.data_gen_mode == 'backward sim':
            N_bw = int(np.round((t_back - self.a) / h))  
            _, output_bw, _ = self.system.generate_data(ic, self.a, t_back, N_bw)
            output_bw = np.flip(output_bw, axis=1)  
        elif self.data_gen_mode == 'negative forward':
            N_bw = int(np.round((self.a - t_back) / h))  
            _, output_bw, _ = self.system.generate_data(ic, t_back, self.a, N_bw)
        else:
            raise Exception("Invalid data_gen_mode. Must be 'backward sim' or 'negative forward'.")

        ic_z_bw = np.random.rand(self.samples, self.system.z_size)  # initial z values
        z_data_fw1 = data.KKL_observer_data(self.M, self.K, output_bw, t_back, self.a, ic_z_bw, N_bw)

        ic_z = z_data_fw1[:, -1, :]

        z_data_fw2 = data.KKL_observer_data(self.M, self.K, output_fw, self.a, self.b, ic_z, self.N)

        return x_data_fw, z_data_fw2, output_fw, t_fw, ic


    def set_physics_limits(self, limits: np.ndarray) -> None:
        """
        Sets the limits of the state sample space for the physics datapoints.
        Can only be used if sample_mode is 2.
        """
        if self.PINN_sample_mode == 'split traj':
            self.limit_physics = limits
        else:
            raise Exception('Can only set limits if sample mode is 1.')  
    
    def normalize(self) -> None:
        """
        Old method to normalize all the data before training.
        Use the normalizer class instead.
        """
        self.x_data = (self.x_data - self.mean_x) / self.std_x
        self.z_data = (self.z_data - self.mean_z) / self.std_z
        self.output_data = (self.output_data - self.mean_output) / self.std_output       
        
    def __getitem__(self, idx: int) -> None:
        x = self.x_data[idx]
        z = self.z_data[idx]
        y = self.output_data[idx]
        x_ph = self.x_data_ph[idx]
        y_ph = self.output_data_ph[idx]
        return [x.float(), z.float(), y.float(), x_ph.float(), y_ph.float()]

class ZToXDataset(torch.utils.data.Dataset):
    def __init__(self, z, x):
        """
        z: tensor of shape (N, z_dim)
        x: tensor of shape (N, x_dim)
        """
        self.z = z
        self.x = x

    def __len__(self):
        return self.z.shape[0]

    def __getitem__(self, idx):
        return self.z[idx], self.x[idx]
