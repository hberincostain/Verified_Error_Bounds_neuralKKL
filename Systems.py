"""
Portions of this file were from the MIT-licensed repository:

    Learning-based Design of Luenberger Observers for Nonlinear Systems
    Umar Niazi, John Cao, Xudong Sun, Amritam Das, and Karl Johansson
    Version: 1.0.0
    Released: 2022-10-04
    Repository: https://github.com/Mudhdhoo/ACC_KKL_Observer
"""

import numpy as np
from smt.sampling_methods import LHS
from data_generation import RK4
import torch

"""
Systems are implemented by defining 6 essential parameters:

function: The system function describing its dynamics.

output: The measureable outputs of the system.

input: Input to the system for non-autonomous systems. If the system is autonomous, input is None.

x_size: Dimension of the system.

y_size: Dimension of the output.

z_size: Dimension of the transformed system.

"""

class System:
    def __init__(self, function, output):
        self.function = function
        self.output = output
    
    # LHS Sampling
    def sample_ic(self, sample_space, samples, seed):
        return LHS(xlimits = sample_space, random_state = seed)(samples)
    
    def simulate(self, a, b, N, v):
        x,t = RK4(self.function, a, b, N, v, self.input)
        return np.array(x), t
    
    def generate_data(self, ic, a, b, N):
        data = []
        output = []
        for i in range(0, np.size(ic, axis=0)):
            x, t = self.simulate(a, b, N, ic[i]) 
            temp = []
            for j in x:
                temp.append(self.output(j)) 
            data.append(x)
            output.append(np.array(temp))
        
        return np.array(data), np.array(output), t

    def gen_noise(self, mean, std):
        x_noise = np.random.normal(mean, std, (self.x_size))
        y_noise = np.random.normal(mean, std, (self.y_size))
        if self.y_size == 1:
            y_noise = y_noise[0]
        return x_noise, y_noise
    
    def toggle_noise(self):
        if self.add_noise:
            self.add_noise = False
        else:
            self.add_noise = True

# --------------- Autonomous Systems --------------- 

# Reverse Duffing Oscilator
class RevDuff(System):
    def __init__(self, zdim, specify_zdim=False, add_noise=False, noise_mean=0, noise_std=0.01):
        self.y_size = 1
        self.x_size = 2
        
        if not specify_zdim:
            if zdim == 5:
                self.z_size = self.y_size * (2 * self.x_size + 1)
            if zdim == 3:
                self.z_size = self.y_size * (1 * self.x_size + 1)
        else:
            self.z_size = zdim
        
        self.input = None
        self.add_noise = add_noise
        self.noise = 0
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        super().__init__(self.function, self.output)
    
    def function(self, u, x):
        x1 = x[0]
        x2 = x[1]

        x1_dot = x2**3
        x2_dot = -x1

        if self.add_noise:
            self.noise = self.gen_noise(self.noise_mean, self.noise_std)[0]
        
        return np.array([x1_dot, x2_dot]) + self.noise
    
    def torch_function(self, x):
        # x is a tensor: shape (..., 2) or length-2
        x1 = x[0]
        x2 = x[1]

        x1_dot = x2 ** 3
        x2_dot = -x1

        return torch.stack([x1_dot, x2_dot])
    def torch_function_batch(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2)
        x1 = x[:, 0]   # shape (B,)
        x2 = x[:, 1]   # shape (B,)

        x1_dot = x2 ** 3
        x2_dot = -x1

        return torch.stack([x1_dot, x2_dot], dim=1)  # shape (B, 2)
    def output(self, x):
        y = x[0]

        if self.add_noise:
            self.noise = self.gen_noise(self.noise_mean, self.noise_std)[1]
        
        return y + self.noise
    
    def output_batch(self, x):
        # x: shape (B, d) or (d,)
        if x.ndim == 1:
            y = x[0]  # scalar
        else:
            y = x[:, 0]  # shape (B,)

        if self.add_noise:
            # Make sure noise shape matches y
            self.noise = self.gen_noise(self.noise_mean, self.noise_std, size=y.shape)

        return y + self.noise
    
# Van der Pol Oscillator
class VdP(System):
    def __init__(self, zdim, my = 3, add_noise = False, noise_mean = 0, noise_std = 0.01):
        self.x_size = 2
        self.y_size = 1
        if zdim == 5:
            self.z_size = self.y_size*(2*self.x_size + 1)
        if zdim == 3:
            self.z_size = self.y_size*(1*self.x_size + 1) 
        self.my = my
        self.input = None
        self.add_noise = add_noise
        self.noise = 0  
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        super().__init__(self.function, self.output)
        
    def function(self, u, x):
        x1 = x[0]
        x2 = x[1]
            
        x1_dot = x2
        x2_dot = self.my*(1 - x1**2)*x2 - x1

        if self.add_noise:
            self.noise = self.gen_noise(self.noise_mean, self.noise_std)[0]
            
        return np.array([x1_dot, x2_dot]) + self.noise
    
    def torch_function(self, x):
        x1 = x[0]
        x2 = x[1]
            
        x1_dot = x2
        x2_dot = self.my*(1 - x1**2)*x2 - x1
            
        return torch.stack([x1_dot, x2_dot])
    
    def torch_function_batch(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2)
        x1 = x[:, 0]   # shape (B,)
        x2 = x[:, 1]   # shape (B,)

        x1_dot = x2
        x2_dot = self.my * (1 - x1**2) * x2 - x1

        return torch.stack([x1_dot, x2_dot], dim=1)  # shape (B, 2)
    
    def output(self, x):
        y = x[0]

        if self.add_noise:
            self.noise = self.gen_noise(self.noise_mean, self.noise_std)[1]

        return y + self.noise
    def output_batch(self, x):
        # x: shape (B, d) or (d,)
        if x.ndim == 1:
            y = x[0]  # scalar
        else:
            y = x[:, 0]  # shape (B,)

        if self.add_noise:
            # Make sure noise shape matches y
            self.noise = self.gen_noise(self.noise_mean, self.noise_std, size=y.shape)

        return y + self.noise
    
# 3D cyclic Lotka--Volterra system
class Volterra3(System):
    def __init__(self, zdim=7, specify_zdim=True, add_noise=False, noise_mean=0.0, noise_std=0.01):
        self.x_size = 3
        self.y_size = 1

        if specify_zdim:
            self.z_size = int(zdim)
        else:
            self.z_size = self.y_size * (2 * self.x_size + 1)

        self.input = None
        self.add_noise = add_noise
        self.noise = 0
        self.noise_mean = noise_mean
        self.noise_std = noise_std

        super().__init__(self.function, self.output)

    def function(self, u, x):
        x = np.asarray(x)
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]

        x1_dot = x1 * (x2 - x3)
        x2_dot = x2 * (x3 - x1)
        x3_dot = x3 * (x1 - x2)

        if self.add_noise:
            self.noise = self.gen_noise(self.noise_mean, self.noise_std)[0]
        else:
            self.noise = 0

        return np.array([x1_dot, x2_dot, x3_dot]) + self.noise

    def torch_function(self, x):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]

        x1_dot = x1 * (x2 - x3)
        x2_dot = x2 * (x3 - x1)
        x3_dot = x3 * (x1 - x2)

        return torch.stack([x1_dot, x2_dot, x3_dot])

    def torch_function_batch(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]

        x1_dot = x1 * (x2 - x3)
        x2_dot = x2 * (x3 - x1)
        x3_dot = x3 * (x1 - x2)

        return torch.stack([x1_dot, x2_dot, x3_dot], dim=1)

    def output(self, x):
        y = x[0]

        if self.add_noise:
            self.noise = self.gen_noise(self.noise_mean, self.noise_std)[1]
        else:
            self.noise = 0

        return y + self.noise

    def output_batch(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            y = x[0]
        else:
            y = x[:, 0]

        if self.add_noise:
            noise = self.noise_std * torch.randn_like(y) + self.noise_mean
            return y + noise

        return y