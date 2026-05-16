from __future__ import annotations

import argparse
import itertools
import sys
import pickle
import time
from pathlib import Path
from typing import Callable, Optional, Sequence
from utils import sample_uniform_box

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import besselap
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from integral_data import integral_approx

# Make this script runnable from Simulations/ or from the project root.
try:
    THIS_DIR = Path(__file__).resolve().parent
    for candidate in (THIS_DIR, THIS_DIR.parent, THIS_DIR.parents[1]):
        if str(candidate) not in sys.path:
            sys.path.append(str(candidate))
except Exception:
    pass

from Dataset import DataSet  # project-local
from Normalizer import Normalizer  # project-local
from NN import Main_Network, NeuralODEEncoder  # project-local
from Systems import Volterra3


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_csv_floats(s: str) -> list[float]:
    return [float(v.strip()) for v in s.split(",") if v.strip()]


def parse_csv_ints(s: str) -> list[int]:
    return [int(v.strip()) for v in s.split(",") if v.strip()]


def bessel_poles(order: int, pole_scale: float = 1.0, norm: str = "phase") -> np.ndarray:
    """Return stable continuous-time Bessel poles.

    scipy.signal.besselap gives the poles of an analog Bessel prototype.
    Multiplying by pole_scale moves the poles farther left if pole_scale > 1.
    """
    if order <= 0:
        raise ValueError("Bessel order / zdim must be positive.")
    if pole_scale <= 0:
        raise ValueError("--bessel-pole-scale must be positive.")
    _, poles, _ = besselap(order, norm=norm)
    poles = np.asarray(poles, dtype=np.complex128) * float(pole_scale)
    if np.max(np.real(poles)) >= 0:
        raise ValueError(f"Bessel poles are not Hurwitz after scaling: {poles}")
    return poles


def real_block_diag_from_poles(poles: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Build a real block-diagonal matrix with the given real/complex poles.

    A complex conjugate pair a ± ib is represented by the real 2x2 block
        [[a, -b], [b, a]].
    A real pole a is represented by the 1x1 block [a].
    """
    poles = np.asarray(poles, dtype=np.complex128)
    blocks: list[np.ndarray] = []
    complex_poles = [p for p in poles if abs(p.imag) > tol and p.imag > 0]
    real_poles = [p for p in poles if abs(p.imag) <= tol]
    for p in sorted(complex_poles, key=lambda z: (z.real, abs(z.imag))):
        a, b = float(p.real), float(abs(p.imag))
        blocks.append(np.array([[a, -b], [b, a]], dtype=np.float32))
    for p in sorted(real_poles, key=lambda z: z.real):
        blocks.append(np.array([[float(p.real)]], dtype=np.float32))
    dim = sum(block.shape[0] for block in blocks)
    A = np.zeros((dim, dim), dtype=np.float32)
    cursor = 0
    for block in blocks:
        k = block.shape[0]
        A[cursor:cursor + k, cursor:cursor + k] = block
        cursor += k
    if dim != len(poles):
        raise RuntimeError(f"Internal pole/block mismatch: built dim {dim} from {len(poles)} poles.")
    return A


def bessel_block_B(A: np.ndarray, B_magnitude: float = 1.0, tol: float = 1e-8) -> np.ndarray:
    """Build B compatible with the real Bessel block matrix.

    For each 2x2 rotation/dilation block [[a,-b],[b,a]], use [1,0]^T.
    For each real 1x1 block, use [1].
    """
    n = A.shape[0]
    B = np.zeros((n, 1), dtype=np.float32)
    i = 0
    while i < n:
        is_complex_block = (
            i + 1 < n
            and abs(float(A[i, i] - A[i + 1, i + 1])) < tol
            and abs(float(A[i, i + 1] + A[i + 1, i])) < tol
            and abs(float(A[i + 1, i])) > tol
        )
        B[i, 0] = float(B_magnitude)
        if is_complex_block:
            B[i + 1, 0] = 0.0
            i += 2
        else:
            i += 1
    return B


def build_kkl_pair(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, str]:
    """Build the KKL pair (A, B) and a compact filename tag."""
    if args.A_type == "diag":
        diag = parse_csv_floats(args.diag)
        if len(diag) != args.zdim:
            raise ValueError(f"--diag has length {len(diag)}, but --zdim is {args.zdim}.")
        if max(diag) >= 0:
            raise ValueError("All diagonal entries of A should be negative for a stable KKL filter.")
        A = np.diag(diag).astype(np.float32)
        B = (args.B_magnitude * np.ones((args.zdim, 1), dtype=np.float32))
        A_tag = "diag=" + "_".join(f"{float(v):g}" for v in diag)
        return A, B, A_tag
    if args.A_type == "bessel":
        poles = bessel_poles(args.zdim, pole_scale=args.bessel_pole_scale, norm=args.bessel_norm)
        A = real_block_diag_from_poles(poles)
        B = bessel_block_B(A, args.B_magnitude)
        A_tag = f"bessel{args.zdim}_scale={args.bessel_pole_scale:g}_norm={args.bessel_norm}"
        return A.astype(np.float32), B.astype(np.float32), A_tag
    raise ValueError(f"Unknown A type: {args.A_type}")


def summarize_kkl_pair(A: np.ndarray, B: np.ndarray) -> None:
    eigs = np.linalg.eigvals(A)
    eigs_str = ", ".join(
        f"{z.real:.6g}{z.imag:+.6g}j" if abs(z.imag) > 1e-8 else f"{z.real:.6g}"
        for z in eigs
    )
    print("A matrix:")
    print(A)
    print(f"A eigenvalues: [{eigs_str}]")
    print(f"B.T: {B.T}")


def softmax_max(v: torch.Tensor, tau: float = 50.0) -> torch.Tensor:
    """Smooth approximation to max(v), with tau controlling sharpness."""
    return torch.logsumexp(tau * v, dim=0) / tau


def as_network_output(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Project networks return either y or (y, aux). Normalize to y."""
    out = model(x)
    return out[0] if isinstance(out, tuple) else out


def pde_residual_norms(
    x: torch.Tensor,
    T_net: nn.Module,
    f: Callable[[torch.Tensor], torch.Tensor],
    h: Callable[[torch.Tensor], torch.Tensor],
    A: np.ndarray | torch.Tensor,
    B: np.ndarray | torch.Tensor,
) -> torch.Tensor:
    """Return || DT(x) f(x) - A T(x) - B h(x) ||_2 for each batch point."""
    x = x.requires_grad_(True)
    A_t = torch.as_tensor(A, dtype=x.dtype, device=x.device)
    B_t = torch.as_tensor(B, dtype=x.dtype, device=x.device)

    T_val = as_network_output(T_net, x)  # [batch, zdim]
    f_val = f(x)  # [batch, xdim]
    h_val = h(x).reshape(-1, 1)  # [batch, 1]

    jac_rows = []
    for i in range(T_val.shape[1]):
        grad_i = torch.autograd.grad(
            T_val[:, i].sum(),
            x,
            create_graph=True,
            retain_graph=True,
        )[0]
        jac_rows.append(grad_i)
    J = torch.stack(jac_rows, dim=1)  # [batch, zdim, xdim]

    dTfx = torch.bmm(J, f_val.unsqueeze(-1)).squeeze(-1)  # [batch, zdim]
    residual = dTfx - (T_val @ A_t.T) - (h_val @ B_t.T)
    return torch.linalg.vector_norm(residual, dim=1)


def pde_loss(
    x: torch.Tensor,
    T_net: nn.Module,
    system: Volterra3,
    A: np.ndarray,
    B: np.ndarray,
    tau: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    norms = pde_residual_norms(
        x,
        T_net,
        system.torch_function_batch,
        system.output_batch,
        A,
        B,
    )
    return softmax_max(norms, tau=tau), norms.mean()

def pde_loss_max(
    x: torch.Tensor,
    T_net: nn.Module,
    system: Volterra3,
    A: np.ndarray,
    B: np.ndarray,
    tau: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    norms = pde_residual_norms(
        x,
        T_net,
        system.torch_function_batch,
        system.output_batch,
        A,
        B,
    )
    return norms.max()


def flatten_dataset_tensors(dataset: DataSet, include_ph_data: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect supervised pairs (x, z) from DataSet, robust to trajectory-shaped tensors."""
    xs = [dataset.x_data.float().reshape(-1, dataset.x_data.shape[-1])]
    zs = [dataset.z_data.float().reshape(-1, dataset.z_data.shape[-1])]

    if include_ph_data and hasattr(dataset, "x_data_ph") and hasattr(dataset, "z_data_ph"):
        xs.append(dataset.x_data_ph.float().reshape(-1, dataset.x_data_ph.shape[-1]))
        zs.append(dataset.z_data_ph.float().reshape(-1, dataset.z_data_ph.shape[-1]))

    return torch.cat(xs, dim=0), torch.cat(zs, dim=0)

def limits_tag(limits: np.ndarray) -> str:
    limits = np.asarray(limits, dtype=float)
    return "_".join(f"{a:g}to{b:g}" for a, b in limits)

def make_dataset(
    args: argparse.Namespace,
    system: Volterra3,
    A: np.ndarray,
    B: np.ndarray,
    limits: np.ndarray,
) -> DataSet:
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    A_tag = getattr(args, "A_tag", None)
    if A_tag is None:
        eig_tag = "_".join(f"{float(np.real(v)):g}" for v in np.linalg.eigvals(A))
        A_tag = f"eigreal={eig_tag}"
    B_tag = f"{float(args.B_magnitude):g}"

    cache_name = (
        f"Volterra_dataset_n{system.x_size}_N{args.N}_ic{args.num_ic}_z{args.zdim}"
        f"_data={limits_tag(limits)}_end{float(args.end_time):g}"
        f"_A={A_tag}_B={B_tag}"
        f"_{args.data_gen_mode}.pkl"
    )
    cache_path = save_dir / cache_name
    # print(cache_path)
    # print(args.no_cache)
    if args.no_cache or not cache_path.exists():
        dataset = DataSet(
            system,
            A,
            B,
            args.start_time,
            args.end_time,
            args.N,
            args.num_ic,
            limits,
            PINN_sample_mode=args.pinn_sample_mode,
            data_gen_mode=args.data_gen_mode,
        )
        if not args.no_cache:
            with cache_path.open("wb") as f:
                pickle.dump(dataset, f)
        print("dataset saved to: ", cache_name)
        check_integral_labels_fd(system,A,B,dataset.x_data,dataset.z_data,T=args.end_time,num_points=args.N,eps=1e-3,)
    else:
        with cache_path.open("rb") as f:
            dataset = pickle.load(f)
        print("dataset loaded from: ", cache_name)

    return dataset


def build_networks(
    args: argparse.Namespace,
    dataset: DataSet,
    x_size: int,
    z_size: int,
) -> tuple[nn.Module, nn.Module, Normalizer]:
    normalizer = Normalizer(dataset)
    activation = F.tanh

    if args.method == "Neural_ODE":
        encoder = NeuralODEEncoder(
            x_dim=x_size,
            z_dim=z_size,
            hidden_size=args.hidden_sizes[0],
            normalizer=normalizer,
        )
    else:
        encoder = Main_Network(
            x_size,
            z_size,
            args.hidden_sizes,
            activation,
            normalizer=normalizer,
        )

    decoder = Main_Network(
        z_size,
        x_size,
        args.inverse_sizes,
        activation,
        normalizer=None,
    )
    return encoder.to(DEVICE), decoder.to(DEVICE), normalizer

def check_integral_labels_fd(
    system,
    A,
    B,
    x,
    T_of_x,
    T=30.0,
    num_points=2000,
    eps=1e-3,
    max_points=200,
):
    device = x.device
    dtype = x.dtype

    A_t = torch.as_tensor(A, dtype=dtype, device=device)
    B_t = torch.as_tensor(B, dtype=dtype, device=device)

    x = x.detach().to(device=device, dtype=dtype)
    T_of_x = T_of_x.detach().to(device=device, dtype=dtype)

    n = min(max_points, x.shape[0])
    x = x[:n]
    T_of_x = T_of_x[:n]

    with torch.no_grad():
        f_val = system.torch_function_batch(x)
        h_val = system.output_batch(x).reshape(-1, 1)

        # First-order approximation to phi_eps(x).
        # For a stricter check, replace this by solve_ivp forward flow.
        x_eps = x + eps * f_val

    T_eps_list = []
    T_used_list = []
    h_used_list = []

    for k in range(n):
        x_eps_k = x_eps[k].detach().cpu().numpy().reshape(-1)

        T_eps_k = integral_approx(
            x0=x_eps_k,
            A=A,
            B=B,
            system=system,
            T=T,
            num_points=num_points,
            reject_nonpositive=True,
        )

        if T_eps_k is None:
            continue

        T_eps_list.append(np.asarray(T_eps_k, dtype=np.float32).reshape(-1))
        T_used_list.append(T_of_x[k])
        h_used_list.append(h_val[k])

    if len(T_eps_list) == 0:
        print("[FD check] no valid finite-difference points")
        return

    T_eps = torch.as_tensor(np.stack(T_eps_list), dtype=dtype, device=device)
    T_used = torch.stack(T_used_list).to(device)
    h_used = torch.stack(h_used_list).to(device)

    dTfx_fd = (T_eps - T_used) / eps
    rhs = T_used @ A_t.T + h_used @ B_t.T

    res = dTfx_fd - rhs
    norms = torch.linalg.vector_norm(res, dim=1)

    print("[FD check for integral labels]")
    print("num checked:", len(T_eps_list))
    print("mean residual:", norms.mean().item())
    print("max residual:", norms.max().item())
    print("median residual:", norms.median().item())

def make_pde_loader_from_points(points_cpu: torch.Tensor, batch_size: int) -> DataLoader:
    """
    points_cpu: tensor on CPU, shape [N, xdim].
    """
    return DataLoader(
        TensorDataset(points_cpu.float().detach().cpu()),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )


def mine_hard_pde_points(
    args: argparse.Namespace,
    encoder: nn.Module,
    system: Volterra3,
    A: np.ndarray,
    B: np.ndarray,
    limits: np.ndarray,
) -> torch.Tensor:
    """
    Sample a large pool of uniform points, compute PDE residuals, and return
    the top-k worst points as a CPU tensor.

    These points are unlabeled PDE collocation points, not supervised data.
    """
    encoder.eval()

    # Temporarily disable parameter grads. We still need gradients with respect
    # to x, but not with respect to the network parameters during mining.
    old_requires_grad = [p.requires_grad for p in encoder.parameters()]
    for p in encoder.parameters():
        p.requires_grad_(False)

    pool_size = int(args.hard_pool_size)
    top_k = min(int(args.hard_top_k), pool_size)
    chunk = int(args.hard_chunk_size)

    x_pool = sample_uniform_box(pool_size, limits, DEVICE).detach()
    residual_chunks = []

    for start in range(0, pool_size, chunk):
        xb = x_pool[start:start + chunk].detach().clone().requires_grad_(True)

        norms = pde_residual_norms(
            xb,
            encoder,
            system.torch_function_batch,
            system.output_batch,
            A,
            B,
        )

        residual_chunks.append(norms.detach().cpu())

        del xb, norms

    residuals = torch.cat(residual_chunks, dim=0)

    vals, idx = torch.topk(residuals, k=top_k, largest=True)
    hard_points = x_pool.detach().cpu()[idx]

    print("[hard mining]")
    print("pool size:", pool_size)
    print("top_k:", top_k)
    print("pool residual max:", vals[0].item())
    print("top-k cutoff residual:", vals[-1].item())
    print("hard x min:", hard_points.min(dim=0).values)
    print("hard x max:", hard_points.max(dim=0).values)

    # Restore original requires_grad flags.
    for p, flag in zip(encoder.parameters(), old_requires_grad):
        p.requires_grad_(flag)

    return hard_points

def train_encoder(
    args: argparse.Namespace,
    encoder: nn.Module,
    dataset: DataSet,
    system: Volterra3,
    A: np.ndarray,
    B: np.ndarray,
    limits: np.ndarray,
) -> tuple[nn.Module, float]:
    if args.load_encoder_path is not None:
        load_checkpoint_into_model(encoder, args.load_encoder_path)
        encoder = encoder.to(DEVICE).float().eval()
        print(f"encoder loaded from {args.load_encoder_path}")
        if args.skip_encoder_training:
            return encoder, evaluate_pde(args, encoder, system, A, B, limits)

    x_data, z_data = flatten_dataset_tensors(dataset, include_ph_data=args.include_ph_data)
    print("x_data: ", x_data)
    print("z_data: ", z_data)
    print("check with integral method: ", integral_approx(x_data[0].cpu(), A, B, system))
    count = 0
    a = 1.2
    for x0, z0 in zip(x_data, z_data):
        if torch.norm(x0)>a: count+=1
    print(f"there are {count} inital conditions with norm larger than{a}:")
    
    
    with torch.no_grad():
        x = x_data
        z = z_data

        perm = torch.randperm(x.shape[0], device=x.device)
        i = perm[:x.shape[0]//2]
        j = perm[x.shape[0]//2:min(x.shape[0], 2*x.shape[0]//2)]

        dx = torch.linalg.vector_norm(x[i] - x[j], dim=1)
        dz = torch.linalg.vector_norm(z[i] - z[j], dim=1)

        mask = dx > 1e-8
        ratio = dz[mask] / dx[mask]

        print("min dz/dx:", ratio.min().item())
        print("median dz/dx:", ratio.median().item())
        print("std dz/dx:", ratio.std().item())
        print("max dz/dx:", ratio.max().item())

    data_loader = DataLoader(
        TensorDataset(x_data, z_data),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    uniform_pde_points = sample_uniform_box(args.pde_points, limits, DEVICE).detach().cpu()
    hard_pde_points = None

    pde_points_all = uniform_pde_points
    pde_loader = make_pde_loader_from_points(pde_points_all, args.batch_size)

    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=args.lr_patience)
    mse = nn.MSELoss(reduction="mean")

    for epoch in range(args.adam_epochs):
        encoder.train()
        totals = {"loss": 0.0, "data": 0.0, "pde_max": 0.0, "pde_mean": 0.0}
        steps = 0
        
        # Periodically mine high-residual PDE points.
        if (
            args.hard_mining
            and epoch >= args.hard_start_epoch
            and (epoch - args.hard_start_epoch) % args.hard_mining_every == 0
        ):
            print(f"[hard mining] epoch={epoch}")

            if args.refresh_uniform_pde:
                uniform_pde_points = sample_uniform_box(
                    args.pde_points,
                    limits,
                    DEVICE,
                ).detach().cpu()

            hard_pde_points = mine_hard_pde_points(
                args=args,
                encoder=encoder,
                system=system,
                A=A,
                B=B,
                limits=limits,
            )

            # Mix uniform PDE points with hard PDE points.
            # Do not train only on hard points, or the residual can get worse elsewhere.
            pde_points_all = torch.cat(
                [uniform_pde_points, hard_pde_points],
                dim=0,
            )

            pde_loader = make_pde_loader_from_points(
                pde_points_all,
                args.batch_size,
            )
        
        n_steps = max(len(data_loader), len(pde_loader))
        data_iter = itertools.cycle(data_loader)
        pde_iter = itertools.cycle(pde_loader)

        for _ in range(n_steps):
            x_batch, z_batch = next(data_iter)
            (x_pde_batch,) = next(pde_iter)

            x_batch = x_batch.to(DEVICE)
            z_batch = z_batch.to(DEVICE)
            x_pde_batch = x_pde_batch.to(DEVICE).requires_grad_(True)              

            z_hat = as_network_output(encoder, x_batch)

            if args.data_loss == "mse":
                data_term = mse(z_hat, z_batch)
            else:
                data_term = torch.linalg.vector_norm(z_hat - z_batch, dim=1).mean()

            pde_max, pde_mean = pde_loss(x_pde_batch, encoder, system, A, B, tau=args.tau)

            loss = data_term + args.lambda_mean * pde_mean + args.lambda_max * pde_max

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            totals["loss"] += float(loss.detach().cpu())
            totals["data"] += float(data_term.detach().cpu())
            totals["pde_max"] += float(pde_max.detach().cpu())
            totals["pde_mean"] += float(pde_mean.detach().cpu())
            steps += 1

        mean_loss = totals["loss"] / max(steps, 1)
        scheduler.step(mean_loss)
        if epoch % args.log_every == 0 or epoch == args.adam_epochs - 1:
            print(
                f"[Encoder Adam] epoch={epoch:04d} "
                f"loss={mean_loss:.6e} "
                f"data={totals['data']/steps:.6e} "
                f"pde_softmax={totals['pde_max']/steps:.6e} "
                f"pde_mean={totals['pde_mean']/steps:.6e}"
            )

    final_pde = evaluate_pde(args, encoder, system, A, B, limits)

    if args.lbfgs_steps > 0:
        print("Starting L-BFGS PDE fine-tuning...")
        x_lbfgs = sample_uniform_box(args.lbfgs_points, limits, DEVICE).requires_grad_(True)
        lbfgs = torch.optim.LBFGS(
            encoder.parameters(),
            lr=args.lbfgs_lr,
            max_iter=args.lbfgs_max_iter,
            history_size=50,
            tolerance_grad=1e-8,
            tolerance_change=1e-9,
            line_search_fn="strong_wolfe",
        )

        def closure():
            lbfgs.zero_grad(set_to_none=True)
            max_term, _ = pde_loss(x_lbfgs, encoder, system, A, B, tau=args.tau)
            max_term.backward()
            return max_term

        for step in range(args.lbfgs_steps):
            loss = lbfgs.step(closure)
            print(f"[Encoder L-BFGS] step={step:03d} pde_softmax={loss.item():.6e}")

        final_pde = evaluate_pde(args, encoder, system, A, B, limits)

    return encoder, final_pde


def evaluate_pde(
    args: argparse.Namespace,
    encoder: nn.Module,
    system: Volterra3,
    A: np.ndarray,
    B: np.ndarray,
    limits: np.ndarray,
) -> float:
    encoder.eval()
    x_eval = sample_uniform_box(args.eval_points, limits, DEVICE).requires_grad_(True)
    max_term, mean_term = pde_loss(x_eval, encoder, system, A, B, tau=args.tau)
    pde_max = pde_loss_max(x_eval, encoder, system, A, B, tau=args.tau)
    print(f"[PDE eval] softmax={max_term.item():.6e} mean={mean_term.item():.6e} max={pde_max.item()}")

    return float(pde_max.detach().cpu())


def train_decoder(args, encoder, decoder, limits, dataset):
    if args.load_decoder_path is not None:
        load_checkpoint_into_model(decoder, args.load_decoder_path)
        decoder = decoder.to(DEVICE).eval()
        print(f"decoder loaded from {args.load_decoder_path}")
        return 0
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    x_flat = sample_uniform_box(args.decoder_num_points, limits)
    
    # Put x_flat on GPU once.
    x_flat = x_flat.to(DEVICE)

    # Compute z_flat on GPU and keep it there.
    zs = []
    with torch.no_grad():
        for start in range(0, x_flat.shape[0], args.batch_size):
            x_batch = x_flat[start:start + args.batch_size]
            z_batch = as_network_output(encoder, x_batch)
            zs.append(z_batch.detach())

    z_flat = torch.cat(zs, dim=0)
    
    # x_flat, z_flat = flatten_dataset_tensors(dataset, include_ph_data=args.include_ph_data)
    with torch.no_grad():
        x = x_flat
        z = z_flat

        perm = torch.randperm(x.shape[0], device=x.device)
        i = perm[:x.shape[0]//2]
        j = perm[x.shape[0]//2:x.shape[0]]

        dx = torch.linalg.vector_norm(x[i] - x[j], dim=1)
        dz = torch.linalg.vector_norm(z[i] - z[j], dim=1)

        mask = dx > 1e-8
        ratio = dz[mask] / dx[mask]

        print("min dz/dx:", ratio.min().item())
        print("median dz/dx:", ratio.median().item())
        print("std dz/dx:", ratio.std().item())
        print("max dz/dx:", ratio.max().item())


    optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr_decoder)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.lr_patience
    )
    mse = nn.MSELoss(reduction="mean")

    n = x_flat.shape[0]
    final_loss = float("nan")

    for epoch in range(args.decoder_epochs):
        decoder.train()
        perm = torch.randperm(n, device=DEVICE)

        loss_sum = torch.zeros((), device=DEVICE)
        steps = 0

        for start in range(0, n, args.batch_size):
            idx = perm[start:start + args.batch_size]
            z_batch = z_flat[idx]
            x_batch = x_flat[idx]

            x_hat = as_network_output(decoder, z_batch)
            loss = mse(x_hat, x_batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loss_sum += loss.detach()
            steps += 1

        final_loss = (loss_sum / max(steps, 1)).item()
        scheduler.step(final_loss)

        if epoch % args.log_every == 0 or epoch == args.decoder_epochs - 1:
            print(f"[Decoder] epoch={epoch:04d} mse={final_loss:.6e}")

    return final_loss

def cast_normalizer(normalizer, device=DEVICE, dtype=torch.float32):
    if normalizer is None:
        return None

    for name, value in vars(normalizer).items():
        if torch.is_tensor(value):
            setattr(normalizer, name, value.to(device=device, dtype=dtype))
    return normalizer

def save_models(
    args: argparse.Namespace,
    encoder: nn.Module,
    decoder: nn.Module,
    normalizer: Optional[Normalizer],
    pde_err: float,
    decoder_err: float,
    A: np.ndarray,
    data_limits
) -> tuple[Path, Path]:
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    A_tag = getattr(args, "A_tag", None)
    if A_tag is None:
        eig_tag = "_".join(f"{float(np.real(v)):g}" for v in np.linalg.eigvals(A))
        A_tag = f"eigreal={eig_tag}"
    base = (
        f"Volterra_z={args.zdim}_box={limits_tag(data_limits)}_N={args.N}_ic={args.num_ic}"
        f"_w={args.hidden_sizes[0]}_d={len(args.hidden_sizes)}"
        f"_A={A_tag}_lambda={args.lambda_max}"
        f"_pde={pde_err:.5g}_dec={decoder_err:.5g}_{args.method}"
    )
    enc_path = save_dir / f"{base}_encoder.pt"
    dec_path = save_dir / f"{base}_decoder.pt"

    torch.save({"model_state_dict": encoder.state_dict(), "normalizer": normalizer}, enc_path)
    torch.save({"model_state_dict": decoder.state_dict(), "normalizer": normalizer}, dec_path)

    print(f"Saved encoder: {enc_path}")
    print(f"Saved decoder: {dec_path}")
    return enc_path, dec_path

def compare_decoder_train_vs_uniform(args, encoder, decoder, dataset, limits):
    encoder.eval()
    decoder.eval()

    x_train, _ = flatten_dataset_tensors(dataset, include_ph_data=False)
    n = min(args.eval_points, x_train.shape[0])
    idx = torch.randperm(x_train.shape[0])[:n]
    x_train = x_train[idx].to(DEVICE)

    x_uniform = sample_uniform_box(n, limits, DEVICE).float().to(DEVICE)

    with torch.no_grad():
        err_train = torch.linalg.vector_norm(
            as_network_output(decoder, as_network_output(encoder, x_train)) - x_train,
            dim=1,
        )
        err_uniform = torch.linalg.vector_norm(
            as_network_output(decoder, as_network_output(encoder, x_uniform)) - x_uniform,
            dim=1,
        )

    print("[Decoder inverse error]")
    print("train-support max/mean:", err_train.max().item(), err_train.mean().item())
    print("uniform-box max/mean:", err_uniform.max().item(), err_uniform.mean().item())


def load_checkpoint_into_model(model: nn.Module, path: str, strict: bool = True) -> None:
    """Load either {'model_state_dict': ...} or a raw state_dict into an existing model."""
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

        if "normalizer" in checkpoint and hasattr(model, "normalizer"):
            normalizer = cast_normalizer(checkpoint["normalizer"], DEVICE, torch.float32)
            model.normalizer = normalizer

            if hasattr(model, "net") and hasattr(model.net, "normalizer"):
                model.net.normalizer = normalizer

    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise TypeError(f"Unsupported checkpoint type at {path}: {type(checkpoint)}")

    model.load_state_dict(state_dict, strict=strict)
    model.to(DEVICE).float()


def parse_initial_conditions(s: str, state_dim: int = 4) -> list[list[float]]:
    """Parse one or more state_dim-dimensional initial conditions.

    Format examples for state_dim=4:
        "1.2,0,0,0"
        "1.2,0,0,0; 0,1.2,0,0; -1,0.5,0.2,0.1"
    """
    ics: list[list[float]] = []
    for block in s.split(";"):
        block = block.strip()
        if not block:
            continue
        ic = parse_csv_floats(block)
        if len(ic) != state_dim:
            raise ValueError(
                f"Each initial condition must have exactly {state_dim} comma-separated entries. "
                "Use semicolons to separate multiple ICs, e.g. "
                "--plot-ic '1,0,0,0; 0,1,0,0; -1,0.5,0.2,0.1'."
            )
        ics.append(ic)
    if not ics:
        raise ValueError("At least one initial condition is required.")
    return ics

def sample_hard_pde_points(args, encoder, system, A, B, limits, pool_size=200000, top_k=20000, chunk=5000):
    encoder.eval()
    xs = sample_uniform_box(pool_size, limits, DEVICE)

    all_norms = []
    with torch.enable_grad():
        for start in range(0, pool_size, chunk):
            xb = xs[start:start+chunk].detach().clone().requires_grad_(True)
            nb = pde_residual_norms(
                xb,
                encoder,
                system.torch_function_batch,
                system.output_batch,
                A,
                B,
            ).detach()
            all_norms.append(nb)

    norms = torch.cat(all_norms)
    idx = torch.topk(norms, k=min(top_k, pool_size)).indices
    x_hard = xs[idx].detach().cpu()

    print("[hard mining]")
    print("pool max:", norms.max().item())
    print("top-k min:", norms[idx].min().item())
    return x_hard

def plot_solution_curve_first3(
    system: Volterra3,
    x0s: Sequence[Sequence[float]] | Sequence[float] = ((1.2, 0.0, 0.0, 0.0),),
    t_span: tuple[float, float] = (0.0, 40.0),
    n_points: int = 4000,
    save_path: Optional[str] = None,
    show: bool = True,
    mark_start_end: bool = True,
):
    """Integrate the 4D Volterra system and plot the first 3 components.

    Args:
        system: Volterra system object.
        x0s: Either a single shape-(system.x_size,) initial condition or a sequence
            of shape-(system.x_size,) initial conditions. The plot uses components
            x1, x2, x3 and ignores x4 in the 3D projection.
        t_span: Integration interval.
        n_points: Number of time samples per trajectory.
        save_path: Optional output image path.
        show: Whether to call plt.show().
        mark_start_end: Whether to mark start/end points for each trajectory.

    Returns:
        sols, fig, ax, where sols is the list of solve_ivp results.
    """
    import matplotlib.pyplot as plt

    if torch.is_tensor(x0s):
        x0s_arr = x0s.detach().cpu().numpy().astype(float)
    else:
        x0s_arr = np.asarray(x0s, dtype=float)

    state_dim = getattr(system, "x_size", 4)
    if x0s_arr.shape == (state_dim,):
        x0s_arr = x0s_arr.reshape(1, state_dim)
    if x0s_arr.ndim != 2 or x0s_arr.shape[1] != state_dim:
        raise ValueError(
            f"x0s must have shape ({state_dim},) or (num_trajectories, {state_dim})."
        )
    if state_dim < 3:
        raise ValueError("Need at least 3 state components to make a 3D projection plot.")

    t_eval = np.linspace(t_span[0], t_span[1], int(n_points))
    sols = []

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    for k, x0_arr in enumerate(x0s_arr, start=1):
        sol = solve_ivp(
            lambda t, x: -system.function(t, x),
            t_span,
            x0_arr,
            t_eval=t_eval,
            rtol=1e-9,
            atol=1e-11,
        )
        if not sol.success:
            raise RuntimeError(f"solve_ivp failed for IC #{k} {tuple(x0_arr)}: {sol.message}")
        sols.append(sol)

        ic_label = ", ".join(f"{v:.3g}" for v in x0_arr)
        label = f"IC {k}: ({ic_label})"
        ax.plot(sol.y[0], sol.y[1], sol.y[2], linewidth=1.5, label=label)

        if mark_start_end:
            ax.scatter([sol.y[0, 0]], [sol.y[1, 0]], [sol.y[2, 0]], marker="o", s=24)
            ax.scatter([sol.y[0, -1]], [sol.y[1, -1]], [sol.y[2, -1]], marker="^", s=28)

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_zlabel(r"$x_3$")
    ax.set_title(
        "System trajectories\n"
    )
    # ax.legend(loc="best")
    fig.tight_layout()

    if save_path is not None:
        save_path = str(save_path)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"Saved first-3-components solution plot to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return sols, fig, ax

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a KKL observer for the 4D Volterra example from the paper.")
    p.add_argument("save_dir", type=str)
    p.add_argument("method", choices=["supervised_PINN", "supervised_NN", "Neural_ODE"])

    # System and KKL setup.
    p.add_argument("--zdim", type=int, default=7, help="KKL latent dimension. The paper's Volterra example used a 5D Bessel-pole filter.")
    p.add_argument("--A-type", choices=["bessel", "diag"], default="diag", help="Use a real Bessel-pole block matrix A by default; diag is kept for comparison.")
    p.add_argument("--diag", type=str, default="-1,-2,-3,-4,-5,-6,-7", help="Diagonal entries if --A-type diag; length must equal zdim.")
    p.add_argument("--bessel-pole-scale", type=float, default=1.0, help="Positive multiplier for all Bessel poles. Larger values make A faster.")
    p.add_argument("--bessel-norm", choices=["phase", "delay", "mag"], default="phase", help="Normalization passed to scipy.signal.besselap.")
    p.add_argument("--B-magnitude", type=float, default=1.0)

    # Data generation.
    p.add_argument("--state-lower", type=float, default=0.1)
    p.add_argument("--state-upper", type=float, default=4.6)
    p.add_argument("--data-lower", type=float, default=0.1)
    p.add_argument("--data-upper", type=float, default=4.6)
    p.add_argument("--cutoff-lower", type=float, default=1e-5)
    p.add_argument("--cutoff-upper", type=float, default=8.0)
    p.add_argument("--start-time", type=float, default=0.0)
    p.add_argument("--end-time", type=float, default=30.0)
    p.add_argument("--N", type=int, default=4000, help="Number of time samples used by DataSet.")
    p.add_argument("--num-ic", "--num_ic", dest="num_ic", type=int, default=200, help="Number of initial conditions used by DataSet.")
    p.add_argument("--data-gen-mode", type=str, default="integral")
    p.add_argument("--pinn-sample-mode", type=str, default="no physics")
    p.add_argument("--include-ph-data", type=bool, default=False)
    p.add_argument("--no_cache", type=bool, default=0)

    # Networks.
    p.add_argument("--hidden-sizes", type=parse_csv_ints, default=parse_csv_ints("256,256,256,256"))
    p.add_argument("--inverse-sizes", type=parse_csv_ints, default=parse_csv_ints("256,256,256,256"))
    p.add_argument("--load-encoder-path", type=str, default=None, help="Optional encoder checkpoint path to load before training/evaluation.")
    p.add_argument("--load-decoder-path", type=str, default=None, help="Optional decoder checkpoint path to load before training/evaluation.")
    p.add_argument("--skip-encoder-training", action="store_true", help="Use the loaded encoder as-is.")
    p.add_argument("--skip-decoder-training", action="store_true", help="Use the loaded decoder as-is.")
    p.set_defaults(skip_encoder_training=True)
    p.set_defaults(skip_decoder_training=True)

    # Optimization.
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--pde_points", type=int, default=10000)
    p.add_argument("--eval-points", "--eval_points", dest="eval_points", type=int, default=20000)
    p.add_argument("--adam-epochs", "--adam_epochs", dest="adam_epochs", type=int, default=200)
    p.add_argument("--decoder-epochs", "--decoder_epochs", dest="decoder_epochs", type=int, default=800)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-decoder", type=float, default=1e-3)
    p.add_argument("--lr-patience", type=int, default=20)
    p.add_argument("--lambda_mean", type=float, default=0.2)
    p.add_argument("--lambda_max", type=float, default=0.2)
    p.add_argument("--tau", type=float, default=100.0, help="Softmax-max sharpness for PDE residual norms.")
    p.add_argument("--data-loss", choices=["norm", "mse"], default="norm")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument(
        "--decoder-num-points", "--decoder_num_points",
        dest="decoder_num_points",
        type=int,
        default=800000,
        help="Number of x_data points used to train decoder. Use -1 for all.",
    )
    p.add_argument(
        "--decoder-subsample-mode",
        choices=["random", "stride", "first"],
        default="random",
    )
    p.add_argument("--hard-mining", action="store_true")
    p.add_argument("--hard-start-epoch", type=int, default=500)
    p.add_argument("--hard-mining-every", type=int, default=100)
    p.add_argument("--hard-pool-size", type=int, default=200000)
    p.add_argument("--hard-top-k", type=int, default=20000)
    p.add_argument("--hard-chunk-size", type=int, default=8000)
    p.add_argument("--refresh-uniform-pde", action="store_true")

    # L-BFGS fine-tuning.
    p.add_argument("--lbfgs-steps", type=int, default=8)
    p.add_argument("--lbfgs-points", type=int, default=50_000)
    p.add_argument("--lbfgs-lr", type=float, default=1.0)
    p.add_argument("--lbfgs-max-iter", type=int, default=100)

    # Plotting.
    p.add_argument("--plot-solution", action="store_true", help="Plot a 3D projection using the first three components of the 4D Volterra system.")
    p.add_argument("--plot-only", action="store_true", help="Only make the solution plot, then exit before dataset/training.")
    p.add_argument(
        "--plot-ic",
        type=str,
        default="1.2,0.0,0.0,0.0; 0.0,1.2,0.0,0.0; -1.2,0.4,0.2,0.1",
        help="4D initial condition(s) for --plot-solution. Use semicolons for multiple ICs, e.g. '1,0,0,0; 0,1,0,0'.",
    )
    p.add_argument("--plot-time", type=float, default=40.0, help="Final time for --plot-solution.")
    p.add_argument("--plot-points", type=int, default=4000, help="Number of plotted time samples.")
    p.add_argument("--plot-save-path", type=str, default=None, help="Optional path to save the 3D trajectory plot.")
    p.add_argument("--no-plot-show", action="store_true", help="Save the plot but do not call plt.show().")

    return p

def build_limits(args: argparse.Namespace):
    data_limits = np.array([[args.data_lower, args.data_upper]] * 3, dtype=np.float32)
    pde_limits = np.array([[args.state_lower, args.state_upper]] * 3, dtype=np.float32)
    cutoff_limits = np.array([[args.cutoff_lower, args.cutoff_upper]] * 3, dtype=np.float32)
    verification_limits = np.array([[args.state_lower+(args.state_upper-args.state_lower)/4, args.state_upper-(args.state_upper-args.state_lower)/4]] * 3, dtype=np.float32)
    return data_limits, pde_limits, cutoff_limits, verification_limits


def main() -> None:
    args = build_arg_parser().parse_args()
    start = time.time()

    system = Volterra3(zdim=args.zdim)

    data_limits, pde_limits, cutoff_limits, verification_limits = build_limits(args)
    
    if args.plot_solution:
        # x0s_plot = parse_initial_conditions(args.plot_ic, system.x_size)
        x0s_plot = sample_uniform_box(8, cutoff_limits)
        plot_solution_curve_first3(
            system,
            x0s=x0s_plot,
            t_span=(0.0, args.plot_time),
            n_points=args.plot_points,
            save_path=args.plot_save_path,
            show=not args.no_plot_show,
        )
        if args.plot_only:
            return

    A, B, A_tag = build_kkl_pair(args)
    args.A_tag = A_tag

    print(f"Device: {DEVICE}")
    print(f"System: {system.name}")
    print(f"A type/tag: {A_tag}")
    print(f"B magnitude: {args.B_magnitude}")
    summarize_kkl_pair(A, B)
    print(f"data Sampling box: {data_limits.tolist()}")

    dataset = make_dataset(args, system, A, B, data_limits)
    print(f"Dataset x_data shape: {tuple(dataset.x_data.shape)}")
    print(f"Dataset z_data shape: {tuple(dataset.z_data.shape)}")
    print(f"Data preparation took {time.time() - start:.2f} s")

    encoder, decoder, normalizer = build_networks(args, dataset, system.x_size, system.z_size)

    training_start = time.time()
    encoder, pde_err = train_encoder(args, encoder, dataset, system, A, B, pde_limits)
    print(f"forward training took {time.time() - training_start:.2f} s")
    
    training_start = time.time()
    decoder_err = train_decoder(args, encoder, decoder, data_limits, dataset)
    print(f"inverse training took {time.time() - training_start:.2f} s")
    
    compare_decoder_train_vs_uniform(args, encoder, decoder, dataset, data_limits)

    normalizer_to_save = getattr(encoder, "normalizer", normalizer)
    save_models(args, encoder, decoder, normalizer_to_save, pde_err, decoder_err, A, data_limits)

    print(f"Done. Total wall time: {time.time() - start:.2f} s")


if __name__ == "__main__":
    main()
