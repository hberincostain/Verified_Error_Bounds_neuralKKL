import numpy as np
import torch 
import torch.nn as nn
from auto_LiRPA.jacobian import JacobianOP
import sys
import pickle
from smt.sampling_methods import FullFactorial
import os, contextlib
from utils import fuse_normalizer_into_nn, load_net
import time
from Systems import RevDuff, VdP, Volterra3
from integral_data import generate_T_data

from api import (ABCrownSolver, ConfigBuilder, VerificationSpec, input_vars, output_vars)

def refine_sampling_points(original_points, grid_length, A, B, sys, N=36):
    A = A.detach().numpy()
    B = B.detach().numpy()
    more_points = np.empty((0, 2), int)
    index = 0
    for point in original_points:
        x = point[0].item()
        y = point[1].item()
        limits = np.array([[x - grid_length, x + grid_length], [y - grid_length, y + grid_length]])
        ics = FullFactorial(xlimits = limits)(N)
        _, in_cycle = generate_T_data(ics, A, B, sys, 1400, 80)
        more_points = np.concatenate((more_points, in_cycle), axis=0)
        print(index, in_cycle.shape[0])
        index += 1
    return torch.from_numpy(more_points), grid_length/np.sqrt(N)

def remove_box(center, eps, sample_points, grid_length):
    remaining_points = []
    for point in sample_points:
        if point[0]<=center[0]+eps-grid_length and point[1]<=center[1]+eps-grid_length and point[0]>=center[0]-eps+grid_length and point[1]>=center[1]-eps+grid_length:
            continue
        remaining_points.append([point[0], point[1]])
    return torch.tensor(remaining_points)

class revduff(nn.Module):
    def __init__(self):
        super().__init__()
        # maybe a small conv/MLP, or just a simple operation
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        x1 = x[:, 0]   # shape (B,)
        x2 = x[:, 1]   # shape (B,)

        x1_dot = x2 ** 3
        x2_dot = -x1

        return torch.stack([x1_dot, x2_dot], dim=1)  # shape (B, 2)

class vdp(nn.Module):
    def __init__(self):
        super().__init__()
        # maybe a small conv/MLP, or just a simple operation
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        x1 = x[:, 0]   # shape (B,)
        x2 = x[:, 1]   # shape (B,)

        x1_dot = x2
        x2_dot = (1 - x1**2)*x2 - x1

        return torch.stack([x1_dot, x2_dot], dim=1)  # shape (B, 2)
    
class volterra3(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]

        x1_dot = x1 * (x2 - x3)
        x2_dot = x2 * (x3 - x1)
        x3_dot = x3 * (x1 - x2)

        return torch.stack([x1_dot, x2_dot, x3_dot], dim=1)

class Measurement(nn.Module):
    """
    Example measurement h(x). Adjust to match your design.
    Here: h(x) = x1, shaped as [B, 1].
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x: [B, 2]
        return x[:, 0]   # [B, 1]


class FixedDot(nn.Module):
    """Computes <x, w> for x: [B, d] and fixed w: [d]. Returns [B, 1]."""
    def __init__(self, w: torch.Tensor):
        super().__init__()
        w = w.detach().view(1, -1)          # [1, d]
        self.register_buffer("w", w)

    def forward(self, x):
        x = x.view(x.size(0), -1)           # [B, d]
        return (x * self.w).sum(dim=1, keepdim=True)  # [B, 1]

class Rx_computation_graph(nn.Module):
    def __init__(self, T_net, f_module, A, B=None, h_module=None, dtype=torch.float32, device='cuda'):
        super().__init__()
        self.T_net = T_net
        self.f_module = f_module
        self.h_module = h_module
        self.device = device

        # ensure A and B are the right dtype (float32 typically)
        self.register_buffer("A", A.detach().to(dtype=dtype))
        if B is not None:
            self.register_buffer("B", B.detach().to(dtype=dtype))
 
    def forward(self, x):
        if x.device != self.A.device:
            raise RuntimeError(f"x.device={x.device} but A.device={self.A.device}. Move inputs/model before calling.")
        if x.dtype != self.A.dtype:
            raise RuntimeError(f"x.dtype={x.dtype} but A.dtype={self.A.dtype}. Make A/B float32 like the network.")

        y = self.T_net(x)                           # T(x)
        J = JacobianOP.apply(y, x)                  # Jacobian of T wrt x
        f_x = self.f_module(x)                      # f(x)
        Jf = (J * f_x.unsqueeze(1)).sum(dim=2)      # [B, z_dim]
        AT = y @ self.A.t()                         # AT(x)

        # Bh(x)
        if (self.B is not None) and (self.h_module is not None):
            h_x = self.h_module(x)
            h_x = h_x.view(h_x.size(0), -1)

            B = self.B
            B = B.view(B.size(0), -1)

            Bh = h_x @ B.t()

        else:
            Bh = torch.zeros_like(Jf)

        R = Jf - AT - Bh                            # residual
        R_sq = (R * R).sum(dim=1, keepdim=True)     # squared L2 norm
        return R_sq
    
def verify_Rx_bound(ub, model, box_size, xdim, center, verbose=True):
    x = input_vars(2)
    y = output_vars(1)  # y[0] = V(x), y[1] = V_dot

    # define ab-CROWN input constraint
    input_constraint = (x >= [x - y for x, y in zip(center, [box_size]*xdim)]) & (x <= [x+y for x, y in zip(center, [box_size]*xdim)])
    # define output constraint
    output_constraint = (y[0] < ub) 
    spec = VerificationSpec.build_spec(input_vars=x, output_vars=y, input_constraint=input_constraint, output_constraint=output_constraint)

    cfg = ConfigBuilder.from_defaults()
    cfg = cfg.set(model__with_jacobian=True)
    cfg = cfg.set(attack__pgd_order="skip")
    cfg = cfg.set(bab__timeout=360)
    # cfg = cfg.set(solver__batch_size=512)
    cfg = cfg.set(solver__auto_enlarge_batch_size=True)
    solver = ABCrownSolver(spec, model, config=cfg)
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        result = solver.solve()
    verified = bool(getattr(result, "success", False))
    if verbose:
        print(f"[info] verifying ||R(x)|| < {ub}")
        print(f"status={getattr(result, 'status', None)}, success={verified}")

    return verified, result
    
def bisect_verified_ub(model, box_size, xdim, center, lo=0.0, hi=1.0, tol=1e-3, max_iter=30, max_hi=1e6, verbose=True, verify_high_only=False):
    """
    Find a small ub such that verify_Rx_bound(ub) succeeds.
    Requires (approximately) monotonicity: if it verifies at ub, it verifies for larger ub.

    Returns: (ub_star, history)
      ub_star: smallest verified ub found (up to tol)
      history: list of dicts with ub and verified flag
    """
    history = []

    # ensure hi is verified by expanding upward
    ok, _ = verify_Rx_bound(hi, model, box_size, xdim, center, verbose=verbose)
    history.append({"ub": hi, "verified": ok})
    if verify_high_only: return hi, history
    while not ok:
        lo = hi
        hi *= 2.0
        if hi > max_hi:
            raise RuntimeError(f"Could not find a verifying upper bound up to max_hi={max_hi}.")
        ok, _ = verify_Rx_bound(hi, model, box_size, xdim, verbose=verbose)
        history.append({"ub": hi, "verified": ok})

    # If lo verifies too, shrink downward.
    if lo > 0:
        ok_lo, _ = verify_Rx_bound(lo, model, box_size, xdim, verbose=verbose)
        history.append({"ub": lo, "verified": ok_lo})
        while ok_lo and lo > 0:
            hi = lo
            lo *= 0.5
            ok_lo, _ = verify_Rx_bound(lo, model, box_size, xdim, verbose=verbose)
            history.append({"ub": lo, "verified": ok_lo})
        # now ideally: lo fails, hi succeeds

    best = hi
    for it in range(max_iter):
        mid = 0.5 * (lo + hi)
        ok_mid, _ = verify_Rx_bound(mid, model, box_size, xdim, verbose=verbose)
        history.append({"ub": mid, "verified": ok_mid})

        if ok_mid:
            best = mid
            hi = mid
        else:
            lo = mid

        # stop when interval small (relative)
        if (hi - lo) <= tol * max(1.0, abs(best)):
            break

        if verbose:
            print(f"[bisect] iter={it:02d} lo={lo:.6g} hi={hi:.6g} best={best:.6g}")

    return best, history

if __name__ == '__main__':
    torch.manual_seed(0)
    
    z_dim = 5
    device = "cuda"
    T, T_inv = load_net("/home/hiboy/control_usra/NeuralKKL-main_train_elm/saved_models_vdp", "vdp_N=14000_num_ic=100000_box_size=2.7_w=128_d=7_diag=[-2.0, -4.0, -6.0, -8.0, -10.0]_inv_w=128_inv_d=6_pde_loss=0.007737489_lmbda=0.1_B_magni=1.0_supervised_PINN_encoder.pt", 2, 5, [128]*7, [128]*6)
    T.net = fuse_normalizer_into_nn(T.net, T.net.normalizer.mean_x, T.net.normalizer.std_x, T.net.normalizer.mean_z, T.net.normalizer.std_z)
    T.net = T.net.to(device)
    T_inv.net = T_inv.net.to(device)
    
    system = VdP(5, 1)
    A = np.diag([-2, -4, -6, -8, -10])
    A = torch.from_numpy(A).float().to(device)
    B = torch.ones(z_dim, 1).to(device)
    f_module = vdp().to(device)
    h_module = Measurement().to(device)

    model = Rx_computation_graph(T.net, f_module, A, B, h_module=h_module)

    ub = 0.0077
    with open("/home/hiboy/control_usra/NeuralKKL-main_train_elm/saved_models_vdp/vdp_pde_err=0.0077_remaining_verification_points", "rb") as f:
        dataset = pickle.load(f)
    sample_points = dataset.x_data
    N = sample_points.shape[0]
    print("Total sample points:", N)
    
    grid_lenth = 0.018
    ub_target = 0.0077
    sample_points, grid_lenth = refine_sampling_points(sample_points, grid_lenth, A.cpu(), B.cpu(), system)
    print("sample_points size: ", sample_points.shape[0])
    print("grid_length", grid_lenth)
    
    remaining_points = []
    index = 0
    for center in sample_points:
        print(index)
        index+=1
        print("verfying for: ", center)
        start_time = time.time()
        _, his = bisect_verified_ub(model, grid_lenth, 2, center.tolist(), hi=ub_target, verify_high_only=True)
        print(his)
        print("took: ", start_time-time.time())
        if his[0]["verified"]==False:
            print("verification failed for: ", center)
            remaining_points.append([center[0].item(), center[1].item()])
        else:
            print("verification passed for: ", center)
    dataset.x_data = torch.tensor(remaining_points)
    print(dataset.x_data.shape[0], "points remaining")
    with open("/home/hiboy/control_usra/NeuralKKL-main_train_elm/saved_models_vdp/vdp_pde_err=0.0077_remaining_verification_points", "wb") as f:
        pickle.dump(dataset, f)        
    exit()