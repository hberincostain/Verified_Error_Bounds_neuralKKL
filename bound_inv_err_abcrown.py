import os, sys, time, pickle, contextlib
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np

from utils import fuse_normalizer_into_nn, load_net

from api import ABCrownSolver, ConfigBuilder, VerificationSpec, input_vars, output_vars

def abs_relu(v: torch.Tensor) -> torch.Tensor:
    return torch.relu(v) + torch.relu(-v)

class InverseApproxErrorAbsVec(nn.Module):
    """
    forward(x) returns [|e0|, |e1|] as shape [B,2],
    where e = Tinv(T(x)) - x.
    """
    def __init__(self, T_net: nn.Module, Tinv_net: nn.Module):
        super().__init__()
        self.T_net = T_net
        self.Tinv_net = Tinv_net

    def forward(self, x):
        z = self.T_net(x)            # [B,5]
        x_hat = self.Tinv_net(z)     # [B,2]
        e = x_hat - x                # [B,2]
        a0 = abs_relu(e[:, 0])
        a1 = abs_relu(e[:, 1])
        return torch.stack([a0, a1], dim=1)   # [B,2]

# a,b-CROWN config + verify helper
def make_cfg(batch_size=2048, timeout_s=60.0):

    cfg = (
        ConfigBuilder.from_defaults()
        .set(general__device="cuda")
        .set(general__complete_verifier="auto")          
        .set(attack__pgd_order="skip")
        .set(solver__batch_size=int(batch_size))
        .set(bab__timeout=float(timeout_s))

        # Force input split BaB
        .set(bab__branching__input_split__enable=True)   
        .set(bab__branching__input_split__input_dim_threshold=100)

        # Avoid nonlinear/activation splitting 
        .set(bab__branching__nonlinear_split__disable=True)
    )
    return cfg

def verify_box_linf_lt_ub(model, cfg, x_lower, x_upper, ub, *, quiet=True):
    """
    Verify: for all x in the box, |e0(x)| < ub AND |e1(x)| < ub
    """
    x = input_vars(2)
    y = output_vars(2)

    input_constraint = (x >= x_lower) & (x <= x_upper)
    output_constraint = (y[0] < float(ub)) & (y[1] < float(ub))

    spec = VerificationSpec.build_spec(input_vars=x, output_vars=y, input_constraint=input_constraint, output_constraint=output_constraint)

    solver = ABCrownSolver(spec, model, config=cfg)

    if quiet:
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            res = solver.solve()
    else:
        res = solver.solve()

    ok = bool(getattr(res, "success", False))
    return ok, res


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("torch device:", device)

    ROOT = Path("~/hannah_projects/NeuralKKL_2026").expanduser()
    SAVE_DIR = ROOT / "KKL_error_bound" / "saved_models_vdp"

    model_filename = (
        "vdp_N=14000_num_ic=100000_box_size=2.7_w=128_d=7_diag=[-2.0, -4.0, -6.0, -8.0, -10.0]"
        "_inv_w=128_inv_d=6_pde_loss=0.007737489_lmbda=0.1_B_magni=1.0_supervised_PINN_encoder.pt"
    )
    dataset_filename = (
        "vdp_dataset_N=14000_num_ic=100000_zdim=5_box_size=2.7_end_time=20_A_diag=[-2.0, -4.0, -6.0, -8.0, -10.0]"
        "_B_magnitude=1.0_mode=mixed_merged_mu=1"
    )
    dataset_path = SAVE_DIR / dataset_filename

    # --- Load nets
    T, T_inv = load_net(str(SAVE_DIR), model_filename, 2, 5, [128] * 7, [128] * 6)

    T.net = fuse_normalizer_into_nn(T.net, T.net.normalizer.mean_x, T.net.normalizer.std_x, T.net.normalizer.mean_z, T.net.normalizer.std_z)

    T_net = T.net.to(device).eval()
    Tinv_net = T_inv.net.to(device).eval()

    model = InverseApproxErrorAbsVec(T_net, Tinv_net).to(device).eval()

    # --- Load sample points
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)

    sample_points = dataset.x_data[:-56000]
    print("Using sample points:", sample_points.shape[0])

    # --- Verification parameters
    grid_length = 0.018
    ub_target = 0.023
    print("grid_length:", grid_length)
    print("ub_target:", ub_target)

    cfg = make_cfg(batch_size=2048, timeout_s=60.0)

    remaining_points = []
    for idx, center in enumerate(sample_points):
        c = center.tolist()
        x_lower = [c[0] - grid_length, c[1] - grid_length]
        x_upper = [c[0] + grid_length, c[1] + grid_length]

        print(f"\n[{idx+1}/{len(sample_points)}] center={c}")
        t0 = time.time()

        ok, res = verify_box_linf_lt_ub(model, cfg, x_lower, x_upper, ub_target, quiet=True)

        dt = time.time() - t0
        print("elapsed:", dt, "status:", getattr(res, "status", None), "success:", ok)

        if not ok:
            remaining_points.append(c)

    dataset.x_data = torch.tensor(remaining_points, dtype=torch.float32)
    out_path = SAVE_DIR / f"vdp_remaining_inv_err_verification_points_grid_length={grid_length:.6g}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(dataset, f)

    print("\npoints remaining:", dataset.x_data.shape[0])
    print("saved remaining points to:", out_path)