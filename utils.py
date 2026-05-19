import os
import sys as _sys
import copy
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from NN import Main_Network

# ------------------------------------------------------------------
# Path setup (portable across laptop / hybrid cluster)
# ------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent  # KKL_error_bound/ lives inside the repo

# Make sure local project imports resolve first
for _p in [_THIS_DIR, _REPO_ROOT]:
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))


@torch.no_grad()
def _normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

def spectral_norm_power_iteration(W: torch.Tensor, u: torch.Tensor | None = None, n_iters: int = 5, eps: float = 1e-12):
    """
    Estimate ||W||_2 with power iteration.
    W: (out, in)
    u: optional cached vector of shape (out,)
    Returns: sigma (scalar tensor), u_new (for caching)
    """
    Wm = W
    if u is None or u.numel() != Wm.size(0):
        u = torch.randn(Wm.size(0), device=Wm.device, dtype=Wm.dtype)
    u = _normalize(u, eps)

    for _ in range(n_iters):
        v = _normalize(Wm.t().mv(u), eps)
        u = _normalize(Wm.mv(v), eps)

    sigma = (u @ (Wm @ v)).abs()
    return sigma, u

def lipschitz_penalty_mlp(model: nn.Module, n_iters: int = 5, eps: float = 1e-12, use_logsum: bool = True):
    """
    Penalize global Lipschitz upper bound for MLP with 1-Lipschitz activations (ReLU, tanh, etc).
    If use_logsum=True: penalty = sum log ||W||_2  (stable, preferred)
    Else: penalty = prod ||W||_2 (closer to bound, can be huge)
    """
    penalty = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            sigma, _ = spectral_norm_power_iteration(m.weight, n_iters=n_iters, eps=eps)
            term = torch.log(sigma + eps) if use_logsum else sigma
            penalty = term if penalty is None else (penalty + term if use_logsum else penalty * term)

    if penalty is None:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    return penalty


def order_curve_by_angle(boundary_xy: np.ndarray) -> np.ndarray:
    """
    Heuristic ordering by angle around centroid.
    Works well if the curve is star-shaped around its centroid (often true for VdP limit cycle).
    If your boundary points are already time-ordered along the trajectory, skip this.
    """
    c = boundary_xy.mean(axis=0)
    ang = np.arctan2(boundary_xy[:, 1] - c[1], boundary_xy[:, 0] - c[0])
    return boundary_xy[np.argsort(ang)]

def points_in_polygon(points_xy: np.ndarray, poly_xy: np.ndarray) -> np.ndarray:
    """
    Vectorized ray casting point-in-polygon.
    points_xy: (M,2)
    poly_xy: (N,2) polygon vertices (closed or not; we will close it)
    Returns: (M,) boolean
    """
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    poly = poly_xy
    if not np.allclose(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])

    x0, y0 = poly[:-1, 0], poly[:-1, 1]
    x1, y1 = poly[1:, 0], poly[1:, 1]

    denom = (y1 - y0)
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)

    cond = ((y0 > y[:, None]) != (y1 > y[:, None]))
    x_int = x0 + (y[:, None] - y0) * (x1 - x0) / denom
    crossings = cond & (x[:, None] < x_int)
    inside = np.sum(crossings, axis=1) % 2 == 1
    return inside

def _nearest_boundary_distance(interior: torch.Tensor, boundary: torch.Tensor, use_kdtree: bool = True, chunk: int = 4096):
    """
    Returns (dist, nn_idx) where dist shape (M,), nn_idx shape (M,)
    Uses KDTree if available and use_kdtree=True, otherwise torch cdist in chunks.
    """
    interior_cpu = interior.detach().cpu().float()
    boundary_cpu = boundary.detach().cpu().float()

    if use_kdtree:
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(boundary_cpu.numpy())
            dist, idx = tree.query(interior_cpu.numpy(), k=1)
            return torch.from_numpy(dist), torch.from_numpy(idx)
        except Exception:
            pass  # fall back

    # Torch fallback (chunked)
    M = interior_cpu.shape[0]
    dists = []
    idxs = []
    for i in range(0, M, chunk):
        x = interior_cpu[i:i+chunk]  # (m,2)
        # (m,N)
        D = torch.cdist(x, boundary_cpu)
        dist_i, idx_i = torch.min(D, dim=1)
        dists.append(dist_i)
        idxs.append(idx_i)
    return torch.cat(dists, dim=0), torch.cat(idxs, dim=0)
        

def augment_near_limit_cycle(boundary_xy, interior_xy, eps: float = 0.01, n_new_per_point: int = 20, inward_step: float = 0.01, perp_scale: float = 0.3,
                            ensure_inside: bool = True, boundary_already_ordered: bool = True, use_kdtree: bool = True, max_tries_per_point: int = 200,
                            seed: int | None = None):
    """
    boundary_xy: torch.Tensor or np.ndarray, shape (Nb,2) limit cycle points (preferably ordered)
    interior_xy: torch.Tensor or np.ndarray, shape (Ni,2) known interior points
    eps: threshold distance to boundary to select "near-boundary interior" points
    n_new_per_point: number of new samples to add around each selected interior point
    inward_step: scale for how far inward to move (in same units as x)
    perp_scale: lateral jitter relative to inward_step
    ensure_inside: if True, reject any samples outside the boundary polygon
    boundary_already_ordered: if False, we will order boundary points by angle heuristic
    use_kdtree: try scipy cKDTree for nearest neighbor distances
    Returns:
        augmented_interior (same type as input interior_xy if torch)
        new_points (torch.Tensor)
        near_mask (torch.BoolTensor)
    """
    if seed is not None:
        np.random.seed(seed)

    # Convert inputs to torch tensors (for NN distance), but keep numpy copies for polygon tests
    if isinstance(boundary_xy, np.ndarray):
        boundary_t = torch.from_numpy(boundary_xy).float()
    else:
        boundary_t = boundary_xy.detach().float()

    if isinstance(interior_xy, np.ndarray):
        interior_t = torch.from_numpy(interior_xy).float()
    else:
        interior_t = interior_xy.detach().float()

    device = interior_t.device
    boundary_t = boundary_t.to(device)
    interior_t = interior_t.to(device)

    # Order boundary for polygon test if needed
    boundary_np = boundary_t.detach().cpu().numpy()
    if not boundary_already_ordered:
        boundary_np = order_curve_by_angle(boundary_np)

    # Nearest boundary distances for selecting near-boundary interior points
    dist, nn_idx = _nearest_boundary_distance(interior_t, boundary_t, use_kdtree=use_kdtree)
    near_mask = dist <= eps
    near_idx = torch.nonzero(near_mask).squeeze(1)

    if near_idx.numel() == 0:
        # nothing to do
        if isinstance(interior_xy, torch.Tensor):
            return interior_xy, torch.empty((0, 2), device=device), near_mask.to(device)
        else:
            return interior_xy, np.empty((0, 2)), near_mask.numpy()

    # Prepare for sampling
    interior_np = interior_t.detach().cpu().numpy()
    nn_idx_np = nn_idx.detach().cpu().numpy()

    new_pts = []

    for i in near_idx.detach().cpu().numpy():
        p = interior_np[i]                   # (2,)
        b = boundary_np[nn_idx_np[i]]        # nearest boundary point (2,)

        v = p - b
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            # fallback direction
            v = np.array([1.0, 0.0], dtype=np.float32)
            nv = 1.0
        inward = v / nv                      # points "inward" (boundary -> interior)

        perp = np.array([-inward[1], inward[0]], dtype=np.float32)

        accepted = 0
        tries = 0
        while accepted < n_new_per_point and tries < max_tries_per_point:
            tries += 1
            # Move further inward by alpha >= 0, plus lateral jitter
            alpha = np.random.rand() * inward_step
            beta = (2.0*np.random.rand() - 1.0) * (perp_scale * inward_step)

            cand = p + alpha * inward + beta * perp

            if ensure_inside:
                inside = points_in_polygon(cand[None, :], boundary_np)[0]
                if not inside:
                    continue

            new_pts.append(cand)
            accepted += 1

        # If acceptance is low, you can reduce inward_step or perp_scale.

    new_pts = np.array(new_pts, dtype=np.float32)
    new_pts_t = torch.from_numpy(new_pts).to(device)

    # Return augmented
    if isinstance(interior_xy, torch.Tensor):
        augmented = torch.cat([interior_xy.to(device), new_pts_t], dim=0)
        return augmented, new_pts_t, near_mask.to(device)
    else:
        augmented = np.vstack([interior_xy, new_pts])
        return augmented, new_pts, near_mask.numpy()
    

def sample_uniform_box(n: int, limits: np.ndarray, device: torch.device = 'cpu') -> torch.Tensor:
    lower = torch.as_tensor(limits[:, 0], dtype=torch.float32, device=device)
    upper = torch.as_tensor(limits[:, 1], dtype=torch.float32, device=device)
    return lower + (upper - lower) * torch.rand(n, limits.shape[0], device=device)


def fuse_normalizer_into_nn(nn_with_norm, input_mean, input_std, output_mean, output_std):
    # Make a copy so you don't mess up the training network
    net = copy.deepcopy(nn_with_norm)

    with torch.no_grad():
        # ---- Fuse input normalization into first layer ----
        first = net.layers[0]
        W1 = first.weight.data
        b1 = first.bias.data

        mu_in = input_mean.view(-1).to(W1.device)
        std_in = input_std.view(-1).to(W1.device)
        inv_std = 1.0 / std_in

        # W1' and b1'
        W1_prime = W1 * inv_std.view(1, -1)
        b1_prime = b1 - torch.mv(W1_prime, mu_in)

        first.weight.copy_(W1_prime)
        first.bias.copy_(b1_prime)

        # ---- Fuse output denorm into last layer ----
        last = net.layers[-1]
        W_L = last.weight.data
        b_L = last.bias.data

        mu_out = output_mean.view(-1).to(W_L.device)
        std_out = output_std.view(-1).to(W_L.device)

        W_L_prime = W_L * std_out.view(-1, 1)
        b_L_prime = b_L * std_out + mu_out

        last.weight.copy_(W_L_prime)
        last.bias.copy_(b_L_prime)

    # Disable normalizer in forward
    net.normalizer = None
    return net

def load_net(model_path, model_name, x_dim, z_dim, hidden_sizes, inverse_size, load_decoder = True):
    encoder_checkpoint = torch.load(f"{model_path}/{model_name}", map_location=device, weights_only=False)
    decoder_name = model_name.replace('encoder', 'decoder')
    if load_decoder: decoder_checkpoint = torch.load(f"{model_path}/{decoder_name}", map_location=device, weights_only=False)
    else: decoder_checkpoint = None
    normalizer = encoder_checkpoint['normalizer']
    normalizer.mean_x = normalizer.mean_x.float()
    normalizer.std_x = normalizer.std_x.float()
    normalizer.mean_z = normalizer.mean_z.float()
    normalizer.std_z = normalizer.std_z.float()

    normalizer.mean_x_ph = normalizer.mean_x_ph.float()
    normalizer.std_x_ph = normalizer.std_x_ph.float()
    normalizer.mean_z_ph = normalizer.mean_z_ph.float()
    normalizer.std_z_ph = normalizer.std_z_ph.float()
    activation = torch.tanh
    # activation = torch.selu
    encoder = Main_Network(x_dim, z_dim, hidden_sizes=hidden_sizes,
    activation=activation, normalizer=normalizer)
    encoder.load_state_dict(encoder_checkpoint['model_state_dict'])
    encoder.eval()
    encoder.to(device)
    decoder = Main_Network(z_dim, x_dim, hidden_sizes=inverse_size,
    activation=activation, normalizer=None)
    if load_decoder:
        decoder.load_state_dict(decoder_checkpoint['model_state_dict'])
        decoder.eval()
        decoder.to(device)
    else:
        decoder = None
    
    return encoder, decoder

def _strip_module_prefix(state_dict):
    """
    Handles checkpoints saved from DataParallel-like wrappers.
    """
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state[k[len("module."):]] = v
        else:
            new_state[k] = v
    return new_state

def _resolve_activation(activation):
    """
    Convert a saved activation entry into a callable.
    Handles callables, strings, and missing values.
    """
    if activation is None:
        return torch.tanh

    if callable(activation):
        return activation

    if isinstance(activation, str):
        name = activation.lower()
        if name in {"tanh", "torch.tanh"}:
            return torch.tanh
        if name in {"relu", "torch.relu"}:
            return torch.relu
        if name in {"sigmoid", "torch.sigmoid"}:
            return torch.sigmoid
        if name in {"selu", "torch.selu"}:
            return torch.selu

    raise ValueError(f"Unsupported activation stored in checkpoint: {activation!r}")

def _infer_hidden_sizes(config, *, kind, hidden_sizes):
    """
    Infer hidden_sizes from the checkpoint config or from the argument.
    Returns a list like [100, 100, 100].
    """
    if hidden_sizes is not None:
        return list(hidden_sizes)

    # Most likely format
    if kind == "forward":
        for key in ["hidden_sizes", "encoder_hidden_sizes", "forward_hidden_sizes"]:
            if key in config and config[key] is not None:
                return list(config[key])

    if kind == "inverse":
        for key in ["inverse_size", "inverse_hidden_sizes", "decoder_hidden_sizes", "hidden_sizes"]:
            if key in config and config[key] is not None:
                return list(config[key])

    # Older format: num_hidden + hidden_size
    num_hidden = config.get("num_hidden", None)
    hidden_size = config.get("hidden_size", None)

    if num_hidden is not None and hidden_size is not None:
        return [int(hidden_size)] * int(num_hidden)

    raise ValueError(
        "Could not infer hidden_sizes from checkpoint config. "
        "Please pass hidden_sizes explicitly."
    )

def _extract_state_dict(checkpoint):
    """
    Accept a few common checkpoint formats.
    """
    for key in ["model", "model_state_dict", "state_dict"]:
        if key in checkpoint:
            state = checkpoint[key]
            break
    else:
        raise ValueError(
            "Checkpoint does not contain a model state dict. "
            "Expected one of: 'model', 'model_state_dict', or 'state_dict'."
        )

    if isinstance(state, nn.Module):
        state = state.state_dict()

    if not isinstance(state, dict):
        raise TypeError(f"Expected state dict, got {type(state)}.")

    return _strip_module_prefix(state)


def _float_normalizer_tensors(normalizer):
    """
    Make normalizer tensors float32 if the normalizer object exists.
    This matches what your legacy load_net function does.
    """
    if normalizer is None:
        return None

    for attr in [
        "mean_x", "std_x", "mean_z", "std_z",
        "mean_x_ph", "std_x_ph", "mean_z_ph", "std_z_ph",
    ]:
        if hasattr(normalizer, attr):
            val = getattr(normalizer, attr)
            if torch.is_tensor(val):
                setattr(normalizer, attr, val.float())

    return normalizer


def _load_single_saved_model(path, *, kind, x_dim, z_dim, hidden_sizes=None, device="cpu"):
    """
    Load one saved Main_Network checkpoint.
    """
    if kind not in {"forward", "inverse"}:
        raise ValueError("kind must be either 'forward' or 'inverse'.")

    path = Path(path).expanduser()
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if not isinstance(checkpoint, dict):
        raise TypeError(f"{path} did not load to a dict checkpoint. Got {type(checkpoint)}.")

    config = checkpoint.get("config", {})

    activation = _resolve_activation(config.get("activation", torch.tanh))
    normalizer = _float_normalizer_tensors(config.get("normalizer", checkpoint.get("normalizer", None)))

    hidden_sizes = _infer_hidden_sizes(
        config,
        kind=kind,
        hidden_sizes=hidden_sizes,
    )

    if kind == "forward":
        input_size = config.get("input_size", config.get("x_size", x_dim))
        output_size = config.get("output_size", config.get("z_size", z_dim))
    else:
        input_size = config.get(
            "inverse_input_size",
            config.get("input_size", config.get("z_size", z_dim)),
        )
        output_size = config.get(
            "inverse_output_size",
            config.get("output_size", config.get("x_size", x_dim)),
        )

    input_size = int(input_size)
    output_size = int(output_size)

    model = Main_Network(input_size, output_size, hidden_sizes=hidden_sizes, activation=activation, normalizer=normalizer).to(device)

    state_dict = _extract_state_dict(checkpoint)

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        raise RuntimeError(
            f"Failed to load checkpoint strictly from {path}.\n"
            f"This usually means the inferred architecture is wrong.\n"
            f"kind={kind}, input_size={input_size}, output_size={output_size}, "
            f"hidden_sizes={hidden_sizes}\n\n"
            f"Original error:\n{e}"
        ) from e

    model.eval()
    return model

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
