import argparse
import contextlib
import csv
import importlib
import inspect
import itertools
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from NN import NN

from utils import _load_single_saved_model, fuse_normalizer_into_nn

class VerificationFailure(RuntimeError):
    """A certification failure that preserves the solver status."""
    def __init__(self, message: str, solver_status: str = "unknown"):
        super().__init__(message)
        self.solver_status = str(solver_status)


class TeeStream:
    """
    Mirror writes to multiple streams.
    """
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def setup_log_file(log_path: Optional[str], results_path: Optional[str]):
    """
    If log_path is provided, or if results_path is provided, mirror stdout/stderr to a text log file.
    """
    if log_path is None:
        if results_path is None:
            return None
        log_path = str(Path(results_path).expanduser().with_suffix(".log"))

    path = Path(log_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    log_file = open(path, "w", buffering=1)
    sys.stdout = TeeStream(sys.__stdout__, log_file)
    sys.stderr = TeeStream(sys.__stderr__, log_file)

    print("==================== INVERSE LIPSCHITZ VERIFICATION LOG ====================")
    print("Log path:", path)
    print("Command:", " ".join(sys.argv))
    print("=" * 80)

    return log_file


def set_verification_mode(model: nn.Module, enabled: bool = True) -> None:
    """
    Turn on verification mode for every submodule that supports it.
    """
    for module in model.modules():
        if hasattr(module, "verification_mode"):
            module.verification_mode = enabled


def cast_normalizer_to_float32(normalizer, device: torch.device):
    """Cast tensor attributes of a normalizer object to float32 on device."""
    if normalizer is None:
        return None
    try:
        items = list(vars(normalizer).items())
    except TypeError:
        return normalizer
    for name, value in items:
        if torch.is_tensor(value):
            setattr(normalizer, name, value.to(device=device, dtype=torch.float32))
    return normalizer


def import_abcrown(abcrown_root: str):
    """
    Import alpha-beta-CROWN modules after adding the repo paths to sys.path.
    """
    root = Path(abcrown_root).expanduser().resolve()
    complete_verifier = root / "complete_verifier"

    if not root.exists():
        raise FileNotFoundError(f"alpha-beta-CROWN root does not exist: {root}")
    if not complete_verifier.exists():
        raise FileNotFoundError(f"complete_verifier directory does not exist: {complete_verifier}")

    for p in [str(complete_verifier), str(root)]:
        if p not in sys.path:
            sys.path.insert(0, p)

    sys.modules.pop("utils", None)

    api = importlib.import_module("api")
    jacobian_mod = importlib.import_module("auto_LiRPA.jacobian")

    return {"ABCrownSolver": api.ABCrownSolver, "ConfigBuilder": api.ConfigBuilder, "VerificationSpec": api.VerificationSpec,
            "input_vars": api.input_vars, "output_vars": api.output_vars, "JacobianOP": jacobian_mod.JacobianOP}

def parse_float_list(s: str, expected_len: Optional[int] = None) -> List[float]:
    vals = [float(v.strip()) for v in s.split(",") if v.strip()]
    if expected_len is not None and len(vals) != expected_len:
        raise ValueError(f"Expected {expected_len} floats, got {len(vals)} from {s!r}.")
    return vals


def parse_int_list(s: str) -> List[int]:
    return [int(v.strip()) for v in s.split(",") if v.strip()]


def parse_splits(s: str, dim: int) -> List[int]:
    vals = parse_int_list(s)
    if len(vals) == 1:
        vals = vals * dim
    if len(vals) != dim:
        raise ValueError(f"Expected either one split count or {dim} split counts, got {vals}.")
    if any(v < 1 for v in vals):
        raise ValueError(f"All split counts must be >= 1, got {vals}.")
    return vals


def resolve_path(path_or_none: Optional[str], default_dir: Path, default_filename: Optional[str]) -> Path:
    if path_or_none is not None:
        return Path(path_or_none).expanduser().resolve()
    if default_filename is None:
        raise ValueError("No path or default filename provided.")
    return (default_dir / default_filename).expanduser().resolve()


def _cfg_try_set(cfg, **kwargs):
    """Set config keys when supported by your alpha-beta-CROWN version."""
    for k, v in kwargs.items():
        try:
            cfg = cfg.set(**{k: v})
        except Exception as e:
            print(f"[cfg] Warning: could not set {k}={v} ({type(e).__name__}: {e})")
    return cfg


def make_cfg(ConfigBuilder, *, timeout_s: float, solver_batch_size: int, with_jacobian: bool, skip_attacks: bool, 
             complete_verifier: str, branching_method: str, disable_nonlinear_split: bool, enable_input_split: bool):
    """
    Build the alpha-beta-CROWN config used by ABCrownSolver.

    Notes:
        * complete_verifier="skip" runs incomplete verification only.
        * complete_verifier="bab" runs complete branch-and-bound after the incomplete stage if the incomplete stage cannot prove the property.
        * branching_method controls activation/nonlinear split heuristics, e.g. random, intercept, babsr, fsb, kfsb, nonlinear.
        * enable_input_split is kept False by default because JacobianOP graphs can crash under input splitting in some alpha-beta-CROWN versions.
    """
    cfg = ConfigBuilder.from_defaults()

    cfg = _cfg_try_set(cfg, model__with_jacobian=bool(with_jacobian))
    cfg = _cfg_try_set(cfg, solver__batch_size=int(solver_batch_size))
    cfg = _cfg_try_set(cfg, bab__timeout=float(timeout_s))

    # CROWN-only bound propagation has been more stable for this JacobianOP graph.
    cfg = _cfg_try_set(cfg, solver__init_bound_prop_method="crown")
    cfg = _cfg_try_set(cfg, solver__bound_prop_method="crown")

    if complete_verifier == "skip":
        cfg = _cfg_try_set(cfg, general__complete_verifier="skip")
        cfg = _cfg_try_set(cfg, general__enable_incomplete_verification=True)
    elif complete_verifier == "bab":
        cfg = _cfg_try_set(cfg, general__complete_verifier="bab")
        cfg = _cfg_try_set(cfg, general__enable_incomplete_verification=True)
    else:
        raise ValueError(f"Unknown complete_verifier={complete_verifier!r}")

    cfg = _cfg_try_set(cfg, bab__branching__method=str(branching_method))
    cfg = _cfg_try_set(cfg, bab__branching__nonlinear_split__disable=bool(disable_nonlinear_split))
    cfg = _cfg_try_set(cfg, bab__branching__input_split__enable=bool(enable_input_split))
    cfg = _cfg_try_set(cfg, bab__branching__input_split__enabled=bool(enable_input_split))
    cfg = _cfg_try_set(cfg, input_split__enable=bool(enable_input_split))
    cfg = _cfg_try_set(cfg, input_split__enabled=bool(enable_input_split))

    if not enable_input_split:
        cfg = _cfg_try_set(cfg, bab__branching__input_split__method="naive")
        cfg = _cfg_try_set(cfg, input_split__branching_method="naive")
    else:
        cfg = _cfg_try_set(cfg, bab__branching__input_split__method="sb")
        cfg = _cfg_try_set(cfg, input_split__branching_method="sb")

    if skip_attacks:
        cfg = _cfg_try_set(cfg, attack__pgd_order="skip")
        cfg = _cfg_try_set(cfg, bab__attack__enabled=False)
        cfg = _cfg_try_set(cfg, input_split__attack__enabled=False)

    return cfg


def verify_scalar_upper(*, abcrown, ub: float, model: nn.Module, input_dim: int, lower: List[float], upper: List[float], 
                        with_jacobian: bool, timeout_s: float, batch_size: int, strict_eps: float, quiet: bool, verbose: bool, 
                        complete_verifier: str, branching_method: str, disable_nonlinear_split: bool, enable_input_split: bool) -> Tuple[bool, Optional[str]]:
    """
    Verify model(input)[0] < ub - strict_eps for all input in the box [lower, upper].
    """
    ABCrownSolver = abcrown["ABCrownSolver"]
    ConfigBuilder = abcrown["ConfigBuilder"]
    VerificationSpec = abcrown["VerificationSpec"]
    input_vars = abcrown["input_vars"]
    output_vars = abcrown["output_vars"]

    x = input_vars(input_dim)
    y = output_vars(1)

    input_constraint = (x >= lower) & (x <= upper)
    output_constraint = (y[0] < float(ub) - float(strict_eps))

    spec = VerificationSpec.build_spec(input_vars=x, output_vars=y, input_constraint=input_constraint, output_constraint=output_constraint)

    cfg = make_cfg(ConfigBuilder, timeout_s=timeout_s, solver_batch_size=batch_size, with_jacobian=with_jacobian, skip_attacks=True,
            complete_verifier=complete_verifier, branching_method=branching_method, disable_nonlinear_split=disable_nonlinear_split,
            enable_input_split=enable_input_split)

    solver = ABCrownSolver(spec, model, config=cfg)

    try:
        if quiet:
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                res = solver.solve()
        else:
            res = solver.solve()
    except AssertionError as e:
        print(f"[verify] alpha-beta-CROWN assertion error; treating as not verified: {e}")
        del solver, spec
        return False, "abcrown-assertion-error"
    except NotImplementedError as e:
        print(f"[verify] alpha-beta-CROWN unsupported op; treating as not verified: {e}")
        del solver, spec
        return False, "abcrown-not-implemented"
    except Exception as e:
        print(f"[verify] alpha-beta-CROWN error; treating as not verified: {type(e).__name__}: {e}")
        del solver, spec
        return False, "abcrown-error"

    status = getattr(res, "status", None)
    ok = bool(getattr(res, "success", False))

    if status in ("unsafe-pgd", "unsafe-bab"):
        ok = False

    if verbose:
        print(f"[verify] output < {float(ub) - float(strict_eps):.8g}  status={status}  success={ok}")

    del solver, spec, res
    return ok, status


def verify_lipschitz_bound(*, abcrown, L: float, sq_jac_model: nn.Module, input_dim: int, lower: List[float], upper: List[float], timeout_s: float,
                            batch_size: int, strict_eps: float, quiet: bool, verbose: bool, complete_verifier: str, branching_method: str,
                            disable_nonlinear_split: bool, enable_input_split: bool) -> Tuple[bool, Optional[str]]:
    """
    Verify ||J||_F < L by checking ||J||_F^2 < L^2.
    """
    return verify_scalar_upper(abcrown=abcrown, ub=float(L) ** 2, model=sq_jac_model, input_dim=input_dim, lower=lower, upper=upper, with_jacobian=True,
                                timeout_s=timeout_s, batch_size=batch_size, strict_eps=strict_eps, quiet=quiet, verbose=verbose, complete_verifier=complete_verifier,
                                branching_method=branching_method, disable_nonlinear_split=disable_nonlinear_split, enable_input_split=enable_input_split)


def find_certified_lipschitz_upper_bound(*, abcrown, sq_jac_model: nn.Module, input_dim: int, lower: List[float], upper: List[float], initial_hi: float,
                                        tol: float, max_bisect_iters: int, max_bracket_iters: int, timeout_s: float, batch_size: int, strict_eps: float,
                                        quiet: bool, complete_verifier: str, branching_method: str, disable_nonlinear_split: bool, enable_input_split: bool,
                                        verify_high_only: bool = False, bisect_mode: str = "bracket", bisect_failed_fixed_bound: bool = False) -> Tuple[float, float, str, List[dict]]:
    
    lo = 0.0
    hi = float(initial_hi)
    last_status = None
    history = []

    print("\n[bracket] Checking initial upper bound.")
    ok, status = verify_lipschitz_bound(abcrown=abcrown, L=hi, sq_jac_model=sq_jac_model, input_dim=input_dim, lower=lower, upper=upper, timeout_s=timeout_s,
                    batch_size=batch_size, strict_eps=strict_eps, quiet=quiet, verbose=True, complete_verifier=complete_verifier, branching_method=branching_method,
                    disable_nonlinear_split=disable_nonlinear_split, enable_input_split=enable_input_split)
    last_status = status
    history.append({"L": hi, "verified": bool(ok), "status": status})

    error_statuses = {"abcrown-error", "abcrown-assertion-error", "abcrown-not-implemented"}

    if verify_high_only:
        if ok:
            # Fixed target succeeded. Return immediately.
            return hi, float("nan"), str(last_status), history

        if not bisect_failed_fixed_bound:
            # Fixed target failed and user did not request fallback bisection.
            raise VerificationFailure(f"Fixed target L={hi:.8g} was not verified. status={status}", solver_status=str(status))

        if status in error_statuses:
            raise VerificationFailure(
                f"Fixed target L={hi:.8g} hit an alpha-beta-CROWN error, so fallback "
                f"bisection is not meaningful. status={status}",
                solver_status=str(status),
            )

        print(
            f"\n[fixed-bound fallback] L={hi:.8g} was not verified "
            f"(status={status}); switching this box to bracketing + bisection."
        )
        verify_high_only = False

    if status in error_statuses:
        raise VerificationFailure(f"alpha-beta-CROWN failed before bisection. status={status}", solver_status=str(status))

    if bisect_mode not in {"simple", "bracket"}:
        raise ValueError(f"Unknown bisect_mode={bisect_mode!r}; expected 'simple' or 'bracket'.")

    if not ok:
        if bisect_mode == "simple":
            print(
                "\n[simple bracket] Initial upper bound was not verified; "
                "expanding hi as in the residual verifier."
            )
        else:
            print("\n[bracket] Searching for a verified upper bracket.")

        for _ in range(max_bracket_iters):
            lo = hi
            hi *= 2.0
            ok, status = verify_lipschitz_bound(abcrown=abcrown, L=hi, sq_jac_model=sq_jac_model, input_dim=input_dim, lower=lower, upper=upper,
                            timeout_s=timeout_s, batch_size=batch_size, strict_eps=strict_eps, quiet=quiet, verbose=True, complete_verifier=complete_verifier,
                            branching_method=branching_method, disable_nonlinear_split=disable_nonlinear_split, enable_input_split=enable_input_split)
            last_status = status
            history.append({"L": hi, "verified": bool(ok), "status": status})

            if ok:
                print(f"[bracket] Found verified hi={hi:.8g}")
                break

            if status in error_statuses:
                raise VerificationFailure(f"alpha-beta-CROWN failed while bracketing. status={status}", solver_status=str(status))

            print(f"[bracket] Not verified at L={hi:.8g} (status={status}); trying next hi.")
        else:
            raise VerificationFailure(
                f"Could not find a verified upper bracket after {max_bracket_iters} expansions. "
                f"Last hi={hi:.8g}. Last status={last_status}.",
                solver_status=str(last_status),
            )

    print("\n[bisect] Refining certified upper bound.")
    for it in range(max_bisect_iters):
        mid = 0.5 * (lo + hi)

        ok, status = verify_lipschitz_bound(abcrown=abcrown, L=mid, sq_jac_model=sq_jac_model, input_dim=input_dim, lower=lower, upper=upper, 
                        timeout_s=timeout_s, batch_size=batch_size, strict_eps=strict_eps, quiet=quiet, verbose=False, 
                        complete_verifier=complete_verifier, branching_method=branching_method, disable_nonlinear_split=disable_nonlinear_split,
                        enable_input_split=enable_input_split)
        last_status = status
        history.append({"L": mid, "verified": bool(ok), "status": status})

        if status in error_statuses:
            print(f"[bisect] Stopping early due to alpha-beta-CROWN error: {status}")
            break

        if ok:
            hi = mid
        else:
            lo = mid

        print(f"[bisect {it + 1:02d}] lo={lo:.8g}, hi={hi:.8g}, width={hi - lo:.3g}, status={status}")

        if hi - lo <= tol:
            break

    return hi, lo, str(last_status), history


# =============================================================================
# Verification model wrappers
# =============================================================================

class TinvJacobianFroSqFromZ(nn.Module):
    """
    forward(z) = || d T_inv(z) / dz ||_F^2, returned as shape [B, 1].
    """
    def __init__(self, T_inv: nn.Module, JacobianOP):
        super().__init__()
        self.T_inv = T_inv
        self.JacobianOP = JacobianOP

    def forward(self, z):
        x_hat = self.T_inv(z)
        J = self.JacobianOP.apply(x_hat, z)  # [B, x_dim, z_dim]
        sq = (J * J).sum(dim=(1, 2))
        return sq.unsqueeze(1)


class TinvJacobianFroSqFromXImage(nn.Module):
    """
    forward(x) = || d T_inv(z) / dz |_{z=T(x)} ||_F^2, returned as [B, 1].
    """
    def __init__(self, T: nn.Module, T_inv: nn.Module, JacobianOP):
        super().__init__()
        self.T = T
        self.T_inv = T_inv
        self.JacobianOP = JacobianOP

    def forward(self, x):
        z = self.T(x)
        x_hat = self.T_inv(z)
        J = self.JacobianOP.apply(x_hat, z)  # [B, x_dim, z_dim]
        sq = (J * J).sum(dim=(1, 2))
        return sq.unsqueeze(1)


class TinvJacobianFroSqRemark2(nn.Module):
    """
    forward([x, dz]) = || d T_inv(z) / dz ||_F^2 at z = T(x) + dz.

    The input dimension is x_dim + z_dim.  We verify over
        x in X, dz in [-delta_z, delta_z]^z_dim.

    This is a conservative box over-approximation of the Remark 2 set
        T(X) \oplus B_{delta_z},
    because the Euclidean ball B_{delta_z} is contained in the coordinate box.
    """
    def __init__(self, T: nn.Module, T_inv: nn.Module, JacobianOP, x_dim: int, z_dim: int):
        super().__init__()
        self.T = T
        self.T_inv = T_inv
        self.JacobianOP = JacobianOP
        self.x_dim = int(x_dim)
        self.z_dim = int(z_dim)

    def forward(self, u):
        x = u[:, :self.x_dim]
        dz = u[:, self.x_dim:self.x_dim + self.z_dim]
        z = self.T(x) + dz
        x_hat = self.T_inv(z)
        J = self.JacobianOP.apply(x_hat, z)  # [B, x_dim, z_dim]
        sq = (J * J).sum(dim=(1, 2))
        return sq.unsqueeze(1)



# =============================================================================
# Direct checkpoint loading helpers
# =============================================================================

def torch_load_checkpoint(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def infer_activation(saved_activation, fallback: str = "relu"):
    if callable(saved_activation):
        return saved_activation

    name = str(saved_activation or fallback).lower()
    if "relu" in name and "leaky" not in name:
        return F.relu
    if "tanh" in name:
        return torch.tanh
    if "sigmoid" in name:
        return torch.sigmoid
    if "leaky" in name:
        return F.leaky_relu

    print(f"[loader] Warning: unknown activation {saved_activation!r}; using relu.")
    return F.relu


def normalize_state_dict_keys_for_model(state_dict, model):
    model_keys = set(model.state_dict().keys())
    candidates = [state_dict]

    candidates.append({
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    })

    for prefix in ["net.", "model.", "T_inv_net.", "inverse.", "encoder."]:
        candidates.append({
            (k[len(prefix):] if k.startswith(prefix) else k): v
            for k, v in state_dict.items()
        })

    best = candidates[0]
    best_overlap = -1
    for cand in candidates:
        overlap = len(set(cand.keys()) & model_keys)
        if overlap > best_overlap:
            best = cand
            best_overlap = overlap

    print(f"[loader] state_dict key overlap: {best_overlap}/{len(model_keys)}")
    return best


def load_nn_checkpoint_direct(checkpoint_path: Path, *, kind: str, x_dim: int, z_dim: int, fallback_hidden_sizes: List[int], device: torch.device) -> nn.Module:
    """
    Direct loader for checkpoints saved as {"model": state_dict, "config": config}.
    """

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    ckpt = torch_load_checkpoint(checkpoint_path, device=torch.device("cpu"))

    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError(f"{checkpoint_path} is not a direct {'{model, config}'} checkpoint.")

    config = ckpt.get("config", {})
    state_dict = ckpt["model"]

    if kind == "inverse":
        in_size = int(config.get("inverse_input_size", config.get("z_size", z_dim)))
        out_size = int(config.get("inverse_output_size", config.get("x_size", x_dim)))
    elif kind == "forward":
        in_size = int(config.get("forward_input_size", config.get("x_size", x_dim)))
        out_size = int(config.get("forward_output_size", config.get("z_size", z_dim)))
    else:
        raise ValueError(f"Unknown kind={kind!r}")

    if "num_hidden" in config and "hidden_size" in config:
        num_hidden = int(config["num_hidden"])
        hidden_size = int(config["hidden_size"])
    else:
        if len(set(fallback_hidden_sizes)) != 1:
            raise ValueError(
                "Fallback hidden sizes are nonuniform, but NN expects num_hidden/hidden_size. "
                f"Got {fallback_hidden_sizes}"
            )
        num_hidden = len(fallback_hidden_sizes)
        hidden_size = int(fallback_hidden_sizes[0])

    activation = infer_activation(config.get("activation", None), fallback="relu")
    normalizer = config.get("normalizer", None)

    print(f"[loader] Direct-loading {kind} checkpoint:")
    print(f"         path={checkpoint_path}")
    print(f"         architecture: input={in_size}, output={out_size}, hidden={hidden_size} x {num_hidden}")

    model = NN(num_hidden, hidden_size, in_size, out_size, activation, normalizer=normalizer, dropout_prob=0.0)

    state_dict = normalize_state_dict_keys_for_model(state_dict, model)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()
    set_verification_mode(model, True)
    return model


def import_legacy_calc_module_from_path(calc_path: str, model_dir_for_env: Path):

    calc_path = Path(calc_path).expanduser()
    if not calc_path.is_absolute():
        calc_path = Path.cwd() / calc_path
    calc_path = calc_path.resolve()

    if not calc_path.exists():
        raise FileNotFoundError(f"Legacy calc_lipschizt_const.py not found: {calc_path}")

    legacy_dir = calc_path.parent
    repo_root = legacy_dir.parent

    candidate_paths = [legacy_dir / "NeuralKKL-main", repo_root / "NeuralKKL-main", legacy_dir, repo_root, Path.cwd()]

    conflict_names = ["NN", "Dataset", "Systems", "Normalizer", "Observer", "bound_Rx", "integral_data"]
    saved_modules = {name: sys.modules.get(name) for name in conflict_names}

    for p in reversed(candidate_paths):
        if p.exists():
            sp = str(p.resolve())
            if sp in sys.path:
                sys.path.remove(sp)
            sys.path.insert(0, sp)

    for name in conflict_names:
        if name in sys.modules:
            del sys.modules[name]

    model_dir_for_env = Path(model_dir_for_env).expanduser()
    if not model_dir_for_env.is_absolute():
        model_dir_for_env = Path.cwd() / model_dir_for_env
    os.environ["MODEL_DIR"] = str(model_dir_for_env.resolve())

    module_name = "_legacy_calc_lipschizt_const_for_inverse_verifier"
    if module_name in sys.modules:
        del sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, str(calc_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {calc_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
        nn_mod = sys.modules.get("NN", None)
        print(f"[legacy encoder loader] legacy NN module: {getattr(nn_mod, '__file__', None)}")
        if nn_mod is not None and not hasattr(nn_mod, "Main_Network"):
            raise ImportError(
                "The NN module imported for the legacy loader does not define Main_Network. "
                f"Imported NN from {getattr(nn_mod, '__file__', None)}"
            )
    finally:
        for name, mod in saved_modules.items():
            if mod is not None:
                sys.modules[name] = mod
            elif name in sys.modules:
                del sys.modules[name]

    if not hasattr(module, "load_net"):
        raise AttributeError(f"{calc_path} does not define load_net(...)")
    if not hasattr(module, "fuse_normalizer_into_nn"):
        raise AttributeError(f"{calc_path} does not define fuse_normalizer_into_nn(...)")

    return module


def load_legacy_encoder_checkpoint(encoder_path: Path, *, legacy_calc_path: str, x_dim: int, z_dim: int, encoder_hidden: List[int], device: torch.device, no_fuse_normalizers: bool) -> nn.Module:
    """
    Load old-format forward encoder
    """
    encoder_path = Path(encoder_path).expanduser().resolve()
    if not encoder_path.exists():
        raise FileNotFoundError(f"Legacy encoder checkpoint does not exist: {encoder_path}")

    legacy_calc = import_legacy_calc_module_from_path(
        legacy_calc_path,
        model_dir_for_env=encoder_path.parent,
    )
    legacy_calc.device = str(device)

    dummy_inverse_hidden = [100] * 6

    print("[legacy encoder loader] Loading old-format encoder:")
    print(f"    model_dir:  {encoder_path.parent}")
    print(f"    model_file: {encoder_path.name}")
    print(f"    hidden:     {encoder_hidden}")

    sig = inspect.signature(legacy_calc.load_net)
    if "load_decoder" in sig.parameters:
        encoder, _ = legacy_calc.load_net(
            str(encoder_path.parent),
            encoder_path.name,
            int(x_dim),
            int(z_dim),
            list(encoder_hidden),
            dummy_inverse_hidden,
            load_decoder=False,
        )
    else:
        print(
            "[legacy encoder loader] Warning: legacy load_net has no load_decoder parameter; "
            "it may try to load a decoder."
        )
        loaded = legacy_calc.load_net(str(encoder_path.parent), encoder_path.name, int(x_dim), int(z_dim), list(encoder_hidden), dummy_inverse_hidden)
        encoder = loaded[0] if isinstance(loaded, tuple) else loaded

    model = encoder.net if hasattr(encoder, "net") else encoder

    if not no_fuse_normalizers:
        n = getattr(model, "normalizer", None)
        if n is not None and all(hasattr(n, attr) for attr in ["mean_x", "std_x", "mean_z", "std_z"]):
            print("[legacy encoder loader] Fusing legacy encoder normalizer.")
            model = legacy_calc.fuse_normalizer_into_nn(model, n.mean_x, n.std_x, n.mean_z, n.std_z)
        else:
            print("[legacy encoder loader] No compatible encoder normalizer found; skipping fusion.")

    model = model.to(device).eval()
    set_verification_mode(model, True)
    print("[legacy encoder loader] Loaded legacy encoder successfully.")
    return model

def fuse_inverse_normalizer_if_present(model: nn.Module, device: torch.device) -> nn.Module:
    """Fuse inverse normalizer for T_inv: z -> x."""
    if not hasattr(model, "normalizer"):
        return model.to(device).eval()

    n = cast_normalizer_to_float32(model.normalizer, device)
    model.normalizer = n
    required = ["mean_z", "std_z", "mean_x", "std_x"]
    if not all(hasattr(n, attr) for attr in required):
        print("[normalizer] Inverse model has normalizer, but not mean_z/std_z/mean_x/std_x. Skipping fusion.")
        return model.to(device).eval()

    print("[normalizer] Fusing inverse normalizer into T_inv.")
    model = fuse_normalizer_into_nn(model, n.mean_z, n.std_z, n.mean_x, n.std_x).to(device).eval()

    set_verification_mode(model, True)
    return model


def fuse_encoder_normalizer_if_present(model: nn.Module, device: torch.device) -> nn.Module:
    """Fuse encoder normalizer for T: x -> z."""
    if not hasattr(model, "normalizer"):
        return model.to(device).eval()

    n = cast_normalizer_to_float32(model.normalizer, device)
    model.normalizer = n
    required = ["mean_x", "std_x", "mean_z", "std_z"]
    if not all(hasattr(n, attr) for attr in required):
        print("[normalizer] Encoder model has normalizer, but not mean_x/std_x/mean_z/std_z. Skipping fusion.")
        return model.to(device).eval()

    print("[normalizer] Fusing encoder normalizer into T.")
    model = fuse_normalizer_into_nn(model, n.mean_x, n.std_x, n.mean_z, n.std_z).to(device).eval()

    set_verification_mode(model, True)
    return model


def import_legacy_main_network_for_checkpoint(checkpoint_path: Path):
    """
    Import Main_Network from the NN.py that belongs to the checkpoint directory.
    """
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()

    candidates = [
        checkpoint_path.parent.parent / "NN.py",  # e.g. KKL_error_bound2/NN.py
        checkpoint_path.parent / "NN.py",
        Path.cwd() / "KKL_error_bound2" / "NN.py",
        Path.cwd() / "KKL_error_bound" / "NN.py",
    ]

    for nn_path in candidates:
        if nn_path.exists():
            module_name = f"_legacy_nn_for_{abs(hash(str(nn_path)))}"
            spec = importlib.util.spec_from_file_location(module_name, str(nn_path))
            if spec is None or spec.loader is None:
                continue

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            if hasattr(module, "Main_Network"):
                print(f"[loader] Imported legacy Main_Network from {nn_path}")
                return module.Main_Network

    raise ImportError(
        "Could not find a legacy NN.py defining Main_Network. Tried: "
        + ", ".join(str(p) for p in candidates)
    )


def load_legacy_decoder_checkpoint(decoder_path: Path, *, x_dim: int, z_dim: int, inverse_hidden: List[int], device: torch.device, activation_name: str = "tanh") -> nn.Module:
    """
    Load old-format decoder checkpoints saved as:
        {"model_state_dict": ..., "normalizer": ...}
    """
    decoder_path = Path(decoder_path).expanduser().resolve()
    Main_Network = import_legacy_main_network_for_checkpoint(decoder_path)

    ckpt = torch_load_checkpoint(decoder_path, device=torch.device("cpu"))

    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(
            f"{decoder_path} is not an old-format decoder checkpoint. "
            "Expected key 'model_state_dict'."
        )

    activation_name = str(activation_name).lower()
    if activation_name == "relu":
        activation = F.relu
    elif activation_name == "tanh":
        activation = torch.tanh
    elif activation_name == "sigmoid":
        activation = torch.sigmoid
    else:
        raise ValueError(f"Unknown legacy decoder activation: {activation_name!r}")

    print("[loader] Loading old-format Main_Network decoder checkpoint:")
    print("         path:", decoder_path)
    print("         architecture: z_dim -> x_dim")
    print("         z_dim:", z_dim)
    print("         x_dim:", x_dim)
    print("         inverse_hidden:", inverse_hidden)
    print("         activation:", activation_name)

    decoder = Main_Network(z_dim, x_dim, hidden_sizes=list(inverse_hidden), activation=activation, normalizer=None)

    state_dict = {
        k: (v.float() if torch.is_tensor(v) else v)
        for k, v in ckpt["model_state_dict"].items()
    }
    decoder.load_state_dict(state_dict, strict=True)
    decoder = decoder.float().to(device).eval()

    model = decoder.net if hasattr(decoder, "net") else decoder
    model = model.float().to(device).eval()
    set_verification_mode(model, True)
    return model


def load_inverse_model(args, save_dir: Path, inverse_hidden: List[int], device: torch.device) -> nn.Module:
    inverse_path = resolve_path(args.inverse_model_path, save_dir, args.model_filename)
    if not inverse_path.exists():
        raise FileNotFoundError(f"Inverse checkpoint does not exist: {inverse_path}")

    print("Loading inverse model from:", inverse_path)

    loaded_by_legacy_decoder = False
    try:
        model = load_nn_checkpoint_direct(inverse_path, kind="inverse", x_dim=args.x_dim, z_dim=args.z_dim, fallback_hidden_sizes=inverse_hidden, device=device)
    except Exception as direct_exc:
        print(f"[loader] Direct inverse loader failed: {type(direct_exc).__name__}: {direct_exc}")

        try:
            print("[loader] Trying old-format Main_Network decoder loader.")
            model = load_legacy_decoder_checkpoint(inverse_path, x_dim=args.x_dim, z_dim=args.z_dim, inverse_hidden=inverse_hidden, device=device, activation_name=args.legacy_inverse_activation)
            loaded_by_legacy_decoder = True
        except Exception as legacy_exc:
            print(f"[loader] Old-format decoder loader failed: {type(legacy_exc).__name__}: {legacy_exc}")
            print("[loader] Falling back to calc_lipschitz_const._load_single_saved_model.")
            model = _load_single_saved_model(inverse_path, kind="inverse", x_dim=args.x_dim, z_dim=args.z_dim, hidden_sizes=inverse_hidden, device=str(device))
            model = model.to(device).eval()

    if (not loaded_by_legacy_decoder) and (not args.no_fuse_normalizers):
        model = fuse_inverse_normalizer_if_present(model, device)

    set_verification_mode(model, True)
    return model


def load_encoder_model(args, save_dir: Path, encoder_hidden: List[int], device: torch.device) -> nn.Module:
    if args.encoder_model_path is None and args.encoder_model_filename is None:
        raise ValueError(
            "ximage mode requires an encoder checkpoint. Provide --encoder_model_path or --encoder_model_filename."
        )

    encoder_path = resolve_path(args.encoder_model_path, save_dir, args.encoder_model_filename)
    if not encoder_path.exists():
        raise FileNotFoundError(f"Encoder checkpoint does not exist: {encoder_path}")

    print("Loading encoder model from:", encoder_path)

    try:
        model = load_nn_checkpoint_direct(encoder_path, kind="forward", x_dim=args.x_dim, z_dim=args.z_dim, fallback_hidden_sizes=encoder_hidden, device=device)
    except Exception as direct_exc:
        print(f"[loader] Direct encoder loader failed: {type(direct_exc).__name__}: {direct_exc}")
        print("[loader] Falling back to exact legacy encoder loader.")
        model = load_legacy_encoder_checkpoint(encoder_path, legacy_calc_path=args.legacy_calc_path, x_dim=args.x_dim, z_dim=args.z_dim, encoder_hidden=encoder_hidden, device=device, no_fuse_normalizers=args.no_fuse_normalizers)
        set_verification_mode(model, True)
        return model

    if not args.no_fuse_normalizers:
        model = fuse_encoder_normalizer_if_present(model, device)

    set_verification_mode(model, True)
    return model



# Box construction / splitting

def sample_uniform_box(lower: List[float], upper: List[float], n: int, device: torch.device) -> torch.Tensor:
    lower_t = torch.tensor(lower, dtype=torch.float32, device=device)
    upper_t = torch.tensor(upper, dtype=torch.float32, device=device)
    u = torch.rand(n, len(lower), device=device)
    return lower_t + (upper_t - lower_t) * u


def make_subboxes(lower: List[float], upper: List[float], splits_per_dim: List[int]) -> List[Tuple[List[float], List[float]]]:
    """Split [lower, upper] into a Cartesian grid of sub-boxes."""
    intervals_per_dim = []
    for lo, hi, n in zip(lower, upper, splits_per_dim):
        pts = torch.linspace(float(lo), float(hi), int(n) + 1).tolist()
        intervals_per_dim.append([(pts[i], pts[i + 1]) for i in range(int(n))])

    boxes = []
    for combo in itertools.product(*intervals_per_dim):
        box_lower = [c[0] for c in combo]
        box_upper = [c[1] for c in combo]
        boxes.append((box_lower, box_upper))
    return boxes


def split_each_box(boxes: List[Tuple[List[float], List[float]]], splits_per_dim: List[int]) -> List[Tuple[List[float], List[float]]]:
    """Apply Cartesian splitting to each box in a list of boxes."""
    if all(int(s) == 1 for s in splits_per_dim):
        return boxes

    out = []
    for lower, upper in boxes:
        out.extend(make_subboxes(lower, upper, splits_per_dim))
    return out


def _as_numpy_xy(obj, key: Optional[str] = None) -> "np.ndarray":
    """
    Extract an N x 2 array of centers from a loaded object.
    """
    import numpy as np

    if key:
        for part in key.split("."):
            if isinstance(obj, dict):
                obj = obj[part]
            else:
                obj = getattr(obj, part)

    if torch.is_tensor(obj):
        arr = obj.detach().cpu().numpy()
    elif isinstance(obj, np.ndarray):
        arr = obj
    elif isinstance(obj, dict):
        for k in ["centers", "points", "x_data", "x", "X", "data"]:
            if k in obj:
                return _as_numpy_xy(obj[k], key=None)
        # Fallback: first array-like value with last dimension at least 2.
        for v in obj.values():
            try:
                arr = _as_numpy_xy(v, key=None)
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    return arr[:, :2]
            except Exception:
                pass
        raise ValueError(f"Could not find an N x 2 array in dict keys {list(obj.keys())}.")
    else:
        for attr in ["x_data", "centers", "points", "data"]:
            if hasattr(obj, attr):
                return _as_numpy_xy(getattr(obj, attr), key=None)
        raise TypeError(f"Unsupported center object type: {type(obj)}")

    arr = np.asarray(arr)
    if arr.ndim > 2:
        arr = arr.reshape(-1, arr.shape[-1])
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Expected an array with shape N x >=2, got {arr.shape}.")
    return arr[:, :2].astype(np.float64)


def load_x_centers(path: str, key: Optional[str] = None) -> "np.ndarray":
    """
    Load VdP local-box centers from .pt/.pth/.pkl/.pickle/.npy/.npz/.csv.
    """
    import numpy as np
    import pickle

    p = Path(path).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    p = p.resolve()

    if not p.exists():
        raise FileNotFoundError(f"Could not find x-centers file: {p}")

    suffix = p.suffix.lower()

    if suffix == ".csv":
        import csv as _csv
        rows = []
        with open(p, "r", newline="") as f:
            reader = _csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"CSV {p} has no header.")
            names = set(reader.fieldnames)
            for row in reader:
                if {"center_x", "center_y"} <= names:
                    rows.append([float(row["center_x"]), float(row["center_y"])])
                elif {"x1", "x2"} <= names:
                    rows.append([float(row["x1"]), float(row["x2"])])
                elif {"x", "y"} <= names:
                    rows.append([float(row["x"]), float(row["y"])])
                else:
                    raise ValueError(
                        f"CSV {p} must contain center_x,center_y or x1,x2 or x,y columns. "
                        f"Got {reader.fieldnames}."
                    )
        return np.asarray(rows, dtype=np.float64)

    if suffix == ".npy":
        obj = np.load(p, allow_pickle=True)
        return _as_numpy_xy(obj, key=key)

    if suffix == ".npz":
        obj = np.load(p, allow_pickle=True)
        if key is None:
            key = list(obj.keys())[0]
        return _as_numpy_xy(obj[key], key=None)

    # Try torch.load first, then pickle. 
    try:
        obj = torch.load(p, map_location="cpu", weights_only=False)
        return _as_numpy_xy(obj, key=key)
    except Exception as torch_exc:
        try:
            with open(p, "rb") as f:
                obj = pickle.load(f)
            return _as_numpy_xy(obj, key=key)
        except Exception as pickle_exc:
            raise RuntimeError(
                f"Could not load centers from {p}. "
                f"torch.load error: {type(torch_exc).__name__}: {torch_exc}; "
                f"pickle error: {type(pickle_exc).__name__}: {pickle_exc}"
            )


def _parse_radius(radius: str, x_dim: int) -> List[float]:
    vals = parse_float_list(str(radius))
    if len(vals) == 1:
        vals = vals * x_dim
    if len(vals) != x_dim:
        raise ValueError(f"Expected one radius or {x_dim} radii, got {vals}.")
    return vals


def make_x_boxes_from_centers(centers: "np.ndarray", *, radius: str, x_dim: int, x_lower_clip: Optional[List[float]] = None, x_upper_clip: Optional[List[float]] = None, stride: int = 1, max_centers: Optional[int] = None,
                                unique_decimals: int = 10) -> List[Tuple[List[float], List[float]]]:
    """
    Build a union of local x-boxes centered at sampled/retained VdP points.
    """
    import numpy as np

    centers = np.asarray(centers, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] < x_dim:
        raise ValueError(f"Expected centers of shape N x >= {x_dim}, got {centers.shape}.")

    centers = centers[:, :x_dim]
    if stride > 1:
        centers = centers[::int(stride)]
    if max_centers is not None:
        centers = centers[: int(max_centers)]

    # Remove exact/near duplicate centers to avoid verifying identical boxes.
    if unique_decimals is not None and unique_decimals >= 0:
        centers = np.unique(np.round(centers, int(unique_decimals)), axis=0)

    rad = np.asarray(_parse_radius(radius, x_dim), dtype=np.float64)

    lower = centers - rad
    upper = centers + rad

    if x_lower_clip is not None:
        lower = np.maximum(lower, np.asarray(x_lower_clip, dtype=np.float64))
    if x_upper_clip is not None:
        upper = np.minimum(upper, np.asarray(x_upper_clip, dtype=np.float64))

    keep = np.all(upper > lower, axis=1)
    lower = lower[keep]
    upper = upper[keep]

    return [(lo.tolist(), hi.tolist()) for lo, hi in zip(lower, upper)]


def load_x_boxes_from_csv(path: str, *, x_dim: int, verified_only: bool = True) -> List[Tuple[List[float], List[float]]]:
    """
    Load a union of x-boxes from a CSV.
    """
    import ast

    p = Path(path).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    p = p.resolve()
    if not p.exists():
        raise FileNotFoundError(f"Could not find x-box CSV: {p}")

    boxes = []
    with open(p, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV {p} has no header.")

        for row in reader:
            if verified_only and "status" in row and str(row["status"]).lower() != "verified":
                continue

            if "lower" in row and "upper" in row:
                lo = list(ast.literal_eval(row["lower"]))
                hi = list(ast.literal_eval(row["upper"]))
                # If this CSV came from remark2 mode, keep only x coordinates.
                lo = lo[:x_dim]
                hi = hi[:x_dim]
            elif all(f"x_lower_{i}" in row for i in range(x_dim)):
                lo = [float(row[f"x_lower_{i}"]) for i in range(x_dim)]
                hi = [float(row[f"x_upper_{i}"]) for i in range(x_dim)]
            elif x_dim == 2 and {"lower_x1", "lower_x2", "upper_x1", "upper_x2"} <= set(row):
                lo = [float(row["lower_x1"]), float(row["lower_x2"])]
                hi = [float(row["upper_x1"]), float(row["upper_x2"])]
            else:
                raise ValueError(
                    f"Unsupported x-box CSV format. Columns are {reader.fieldnames}."
                )
            boxes.append((lo, hi))

    return boxes


def attach_dz_box_to_x_boxes(x_boxes: List[Tuple[List[float], List[float]]], *, delta_z: float, z_dim: int) -> List[Tuple[List[float], List[float]]]:
    dz_lower = [-float(delta_z)] * z_dim
    dz_upper = [ float(delta_z)] * z_dim
    return [(list(xlo) + dz_lower, list(xhi) + dz_upper) for xlo, xhi in x_boxes]



def should_refine_failed_box(last_status: str, args) -> bool:
    """
    Decide whether a failed parent box should be split into smaller boxes.
    """
    error_statuses = {"abcrown-error", "abcrown-assertion-error", "abcrown-not-implemented"}
    if str(last_status) in error_statuses and not args.refine_abcrown_errors:
        return False
    return True


def empirical_zbox_from_encoder(T_net: nn.Module, x_lower: List[float], x_upper: List[float], *, n_samples: int, batch_size: int, margin_fraction: float, device: torch.device) -> Tuple[List[float], List[float]]:
    """
    Estimate a rectangular z-box containing T(X) from random x samples.
    """
    T_net.eval()
    z_mins = None
    z_maxs = None

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            n = min(batch_size, n_samples - start)
            x = sample_uniform_box(x_lower, x_upper, n, device)
            z = T_net(x)
            cur_min = z.min(dim=0).values
            cur_max = z.max(dim=0).values
            z_mins = cur_min if z_mins is None else torch.minimum(z_mins, cur_min)
            z_maxs = cur_max if z_maxs is None else torch.maximum(z_maxs, cur_max)

    widths = z_maxs - z_mins
    margin = margin_fraction * torch.clamp(widths, min=1e-6)
    z_lower = (z_mins - margin).detach().cpu().tolist()
    z_upper = (z_maxs + margin).detach().cpu().tolist()
    return z_lower, z_upper

def empirical_fro_lipschitz_estimate_autograd_zbox(Tinv_net: nn.Module, lower: List[float], upper: List[float], n_samples: int, batch_size: int, device: torch.device) -> Optional[float]:
    """
    Empirical max ||dT_inv/dz||_F over random z samples using ordinary PyTorch autograd.
    """
    if n_samples <= 0:
        return None

    Tinv_net.eval()
    max_val = 0.0

    for start in range(0, n_samples, batch_size):
        n = min(batch_size, n_samples - start)
        z = sample_uniform_box(lower, upper, n, device).detach().clone().requires_grad_(True)
        x_hat = Tinv_net(z)

        rows = []
        for k in range(x_hat.shape[1]):
            grad_k = torch.autograd.grad(x_hat[:, k].sum(), z, retain_graph=True, create_graph=False)[0]
            rows.append(grad_k)

        J = torch.stack(rows, dim=1)
        vals = torch.linalg.matrix_norm(J, ord="fro", dim=(1, 2))
        max_val = max(max_val, float(vals.max().item()))

    return max_val


def empirical_fro_lipschitz_estimate_autograd_ximage(T_net: nn.Module, Tinv_net: nn.Module, lower: List[float], upper: List[float], n_samples: int, batch_size: int, device: torch.device) -> Optional[float]:
    """
    Empirical max ||dT_inv/dz||_F at z=T(x), x sampled from an x-box.
    """
    if n_samples <= 0:
        return None

    T_net.eval()
    Tinv_net.eval()
    max_val = 0.0

    for start in range(0, n_samples, batch_size):
        n = min(batch_size, n_samples - start)
        x = sample_uniform_box(lower, upper, n, device)
        z = T_net(x).detach().clone().requires_grad_(True)
        x_hat = Tinv_net(z)

        rows = []
        for k in range(x_hat.shape[1]):
            grad_k = torch.autograd.grad(x_hat[:, k].sum(), z, retain_graph=True, create_graph=False)[0]
            rows.append(grad_k)

        J = torch.stack(rows, dim=1)
        vals = torch.linalg.matrix_norm(J, ord="fro", dim=(1, 2))
        max_val = max(max_val, float(vals.max().item()))

    return max_val


def empirical_fro_lipschitz_estimate_autograd_remark2(T_net: nn.Module, Tinv_net: nn.Module, lower: List[float], upper: List[float], x_dim: int, z_dim: int, n_samples: int, batch_size: int, device: torch.device) -> Optional[float]:
    """Empirical sanity check for Remark 2 mode."""
    if n_samples <= 0:
        return None

    T_net.eval()
    Tinv_net.eval()
    max_val = 0.0

    for start in range(0, n_samples, batch_size):
        n = min(batch_size, n_samples - start)
        u = sample_uniform_box(lower, upper, n, device)
        x = u[:, :x_dim]
        dz = u[:, x_dim:x_dim + z_dim]
        z = (T_net(x) + dz).detach().clone().requires_grad_(True)
        x_hat = Tinv_net(z)

        rows = []
        for k in range(x_hat.shape[1]):
            grad_k = torch.autograd.grad(x_hat[:, k].sum(), z, retain_graph=True, create_graph=False)[0]
            rows.append(grad_k)

        J = torch.stack(rows, dim=1)
        vals = torch.linalg.matrix_norm(J, ord="fro", dim=(1, 2))
        max_val = max(max_val, float(vals.max().item()))

    return max_val


def run_shape_check(model: nn.Module, input_dim: int, lower: List[float], upper: List[float], device: torch.device) -> None:
    """Run one forward pass before launching alpha-beta-CROWN."""
    print("\n[shape check] Running one ordinary PyTorch forward pass.")
    x = sample_uniform_box(lower, upper, n=2, device=device)
    x.requires_grad_(True)

    with torch.enable_grad():
        y = model(x)

    print("[shape check] input shape:", tuple(x.shape))
    print("[shape check] output shape:", tuple(y.shape))
    print("[shape check] output sample:", y.detach().flatten()[:5].cpu().numpy())

    if y.ndim != 2 or y.shape[0] != 2 or y.shape[1] != 1:
        raise RuntimeError(f"Expected model output shape [2, 1], got {tuple(y.shape)}")
    if not torch.isfinite(y).all():
        raise RuntimeError("Shape check output contains NaN or Inf.")



# Split-box verification driver

def verify_boxes(*, abcrown, sq_jac_model: nn.Module, input_dim: int, boxes: List[Tuple[List[float], List[float]]], args) -> Tuple[float, List[dict]]:
    """
    Verify all boxes and return the maximum certified upper bound over verified
    leaf boxes.
    """
    results: List[dict] = []
    global_hi = 0.0
    next_box_id = 0
    refine_splits = parse_splits(args.refine_splits_per_dim, input_dim)

    def write_incremental():
        if getattr(args, "results_path", None) is not None:
            save_results(results, args.results_path)

    def attempt_one_box(box_lower, box_upper, *, box_id, parent_id, depth, path):
        nonlocal global_hi

        print("\n" + "=" * 80)
        print(f"Verifying box id={box_id}, depth={depth}, path={path}")
        print("lower:", box_lower)
        print("upper:", box_upper)
        print("=" * 80)

        t_box = time.time()
        try:
            certified_hi, last_failed_lo, last_status, history = find_certified_lipschitz_upper_bound(
                abcrown=abcrown,
                sq_jac_model=sq_jac_model,
                input_dim=input_dim,
                lower=box_lower,
                upper=box_upper,
                initial_hi=args.initial_hi,
                tol=args.tol,
                max_bisect_iters=args.max_bisect_iters,
                max_bracket_iters=args.max_bracket_iters,
                timeout_s=args.timeout_s,
                batch_size=args.batch_size,
                strict_eps=args.strict_eps,
                quiet=args.quiet,
                complete_verifier=args.complete_verifier,
                branching_method=args.branching_method,
                disable_nonlinear_split=args.disable_nonlinear_split,
                enable_input_split=args.enable_input_split,
                verify_high_only=args.verify_high_only,
                bisect_mode=args.bisect_mode,
                bisect_failed_fixed_bound=args.bisect_failed_fixed_bound,
            )
            status = "verified"
            error = ""
        except VerificationFailure as e:
            print(f"[box] Could not certify this box: {e}")
            certified_hi = float("nan")
            last_failed_lo = float("nan")
            last_status = getattr(e, "solver_status", "unknown")
            history = [{"error": str(e), "solver_status": last_status}]
            status = "failed"
            error = str(e)
        except RuntimeError as e:
            print(f"[box] Could not certify this box: {e}")
            certified_hi = float("nan")
            last_failed_lo = float("nan")
            msg = str(e)
            if "abcrown-not-implemented" in msg:
                last_status = "abcrown-not-implemented"
            elif "abcrown-assertion-error" in msg:
                last_status = "abcrown-assertion-error"
            elif "abcrown-error" in msg:
                last_status = "abcrown-error"
            else:
                last_status = "unknown"
            history = [{"error": msg, "solver_status": last_status}]
            status = "failed"
            error = msg

        elapsed = time.time() - t_box

        if status == "verified" and certified_hi == certified_hi:
            global_hi = max(global_hi, certified_hi)

        row = {
            "box_index": box_id,
            "parent_index": parent_id,
            "depth": depth,
            "path": path,
            "lower": box_lower,
            "upper": box_upper,
            "certified_hi": certified_hi,
            "last_failed_lo": last_failed_lo,
            "status": status,
            "last_solver_status": last_status,
            "elapsed_seconds": elapsed,
            "is_leaf": True,
            "num_children": 0,
            "error": error,
            "history": repr(history),
        }
        results.append(row)
        write_incremental()

        print(f"[box result] id={box_id}, depth={depth}, status={status}, certified_hi={certified_hi}, elapsed={elapsed:.2f}s")
        print(f"[running result] current global max certified_hi over verified leaves={global_hi}")

        return row

    def refine_recursive(box_lower, box_upper, *, parent_id, depth, path):
        nonlocal next_box_id, global_hi

        box_id = next_box_id
        next_box_id += 1

        row = attempt_one_box(box_lower, box_upper, box_id=box_id, parent_id=parent_id, depth=depth, path=path)

        if row["status"] == "verified":
            return [row]

        can_refine = (bool(args.auto_refine_failed_boxes) and depth < int(args.max_refine_depth) and should_refine_failed_box(str(row["last_solver_status"]), args))

        if not can_refine:
            return [row]

        print(
            f"\n[refine] Box id={box_id} failed at depth={depth}; "
            f"splitting into {refine_splits} children and retrying."
        )

        # Mark the failed parent as non-leaf since it will be replaced by children.
        row["is_leaf"] = False
        child_boxes = make_subboxes(box_lower, box_upper, refine_splits)
        row["num_children"] = len(child_boxes)
        write_incremental()

        leaf_rows = []
        for child_idx, (child_lower, child_upper) in enumerate(child_boxes):
            child_path = f"{path}.{child_idx}" if path else str(child_idx)
            leaf_rows.extend(refine_recursive(child_lower, child_upper, parent_id=box_id, depth=depth + 1, path=child_path))

        return leaf_rows

    all_leaf_rows = []
    for i, (box_lower, box_upper) in enumerate(boxes, start=1):
        print("\n" + "#" * 80)
        print(f"Initial box {i}/{len(boxes)}")
        print("#" * 80)
        all_leaf_rows.extend(refine_recursive(box_lower, box_upper, parent_id="", depth=0, path=str(i)))

    # Recompute global bound and final statuses from leaf rows only.
    verified_leaf_rows = [
        r for r in results
        if bool(r.get("is_leaf", True)) and r.get("status") == "verified"
    ]
    failed_leaf_rows = [
        r for r in results
        if bool(r.get("is_leaf", True)) and r.get("status") != "verified"
    ]

    global_hi = 0.0
    for r in verified_leaf_rows:
        val = r.get("certified_hi", float("nan"))
        if val == val:
            global_hi = max(global_hi, float(val))

    print("\n[leaf summary]")
    print(f"Verified leaf boxes: {len(verified_leaf_rows)}")
    print(f"Failed leaf boxes: {len(failed_leaf_rows)}")
    print(f"Active leaf global bound: L <= {global_hi}")

    write_incremental()
    return global_hi, results


def save_results(results: List[dict], output_path: Optional[str]) -> None:
    if output_path is None:
        return

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".json":
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
    else:
        # CSV by default.
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "box_index",
                    "parent_index",
                    "depth",
                    "path",
                    "lower",
                    "upper",
                    "certified_hi",
                    "last_failed_lo",
                    "status",
                    "last_solver_status",
                    "elapsed_seconds",
                    "is_leaf",
                    "num_children",
                    "error",
                    "history",
                ],
            )
            writer.writeheader()
            for row in results:
                writer.writerow(row)

    print(f"Saved per-box results to {path}")


# =============================================================================
# CLI
# =============================================================================

def get_args():
    p = argparse.ArgumentParser()

    # alpha-beta-CROWN path.
    p.add_argument("--abcrown_root", type=str, default="~/hannah_projects/src/alpha-beta-CROWN")

    # Verification domain.
    p.add_argument(
        "--verify_domain",
        choices=["zbox", "ximage", "remark2"],
        default="zbox",
        help=(
            "zbox verifies sup_{z in Z} ||dT_inv/dz||_F. "
            "ximage verifies sup_{x in X} ||dT_inv/dz at z=T(x)||_F. "
            "remark2 verifies over z=T(x)+dz with x in X and dz in [-delta_z,delta_z]^z_dim."
        ),
    )

    # Project/model locations.
    p.add_argument("--root", type=str)
    p.add_argument("--save_subdir", type=str, default=".")
    p.add_argument(
        "--legacy_calc_path",
        type=str,
        default="KKL_error_bound/calc_lipschizt_const.py",
        help="Path to legacy calc_lipschizt_const.py for old-format encoder checkpoints.",
    )

    # Inverse checkpoint.
    p.add_argument(
        "--model_filename",
        type=str,
        default="KKL_error_bound2/saved_models_revduff/revduff_fixed_encoder_N=800_num_ic=1000_box_size=3.0_w=100_d=7_diag=[-1.0, -2.0, -3.0, -4.0, -5.0]_inv_w=128_inv_d=3_lmbda=0.1_B_magni=1.0_supervised_PINN_decoder_direct.pt",
        help="Inverse checkpoint filename under root/save_subdir.",
    )
    p.add_argument("--inverse_model_path", type=str, default=None)

    # Encoder checkpoint. Needed for ximage mode and empirical-zbox mode.
    p.add_argument("--encoder_model_filename", type=str, default=None)
    p.add_argument("--encoder_model_path", type=str, 
                   default="KKL_error_bound2/saved_models_revduff/revduff_N=800_num_ic=1000_box_size=3.0_hidden_sizes=[100, 100, 100, 100, 100, 100, 100]_diag=[-1.0, -2.0, -3.0, -4.0, -5.0]_inverse_size=[100, 100, 100, 100, 100, 100]_pde_loss=0.00074600836_lmbda=10.0_B_magni=1.0_supervised_PINN_encoder.pt")

    # Architecture fallback values. Checkpoint config is used when available.
    p.add_argument("--x_dim", type=int, default=2)
    p.add_argument("--z_dim", type=int, default=5)
    p.add_argument("--encoder_hidden", type=str, default="100,100,100,100,100,100,100")
    p.add_argument("--inverse_hidden", type=str, default="128,128,128")
    p.add_argument(
        "--legacy_inverse_activation",
        type=str,
        default="tanh",
        choices=["tanh", "relu", "sigmoid"],
        help="Activation used by old-format Main_Network decoder checkpoints.",
    )

    # X-box for ximage mode, empirical z-box construction, or Remark 2 mode.
    p.add_argument("--center_x", type=float, default=0.0)
    p.add_argument("--center_y", type=float, default=0.0)
    p.add_argument("--box_size", type=float, default=2.7)
    p.add_argument("--x_lower", type=str, default=None, help="Optional comma-separated lower corner for X.")
    p.add_argument("--x_upper", type=str, default=None, help="Optional comma-separated upper corner for X.")
    p.add_argument(
        "--delta_z",
        type=float,
        default=0.0,
        help=(
            "Radius delta_z from Remark 2. In --verify_domain remark2, the script "
            "certifies on T(X) plus the coordinate box [-delta_z, delta_z]^z_dim, "
            "which contains the Euclidean ball B_delta_z."
        ),
    )

    # Non-rectangular/union-of-boxes x regions. This is mainly for VdP, where
    # the verified region is represented by many small boxes near the limit cycle.
    p.add_argument(
        "--x_region_mode",
        choices=["box", "centers", "boxes_csv"],
        default="box",
        help=(
            "box: use one rectangular X box. "
            "centers: build local x-boxes around centers loaded from --x_centers_path. "
            "boxes_csv: load local x-boxes from --x_boxes_csv."
        ),
    )
    p.add_argument(
        "--x_centers_path",
        type=str,
        default=None,
        help="Path to a tensor/pickle/numpy/csv file containing N x 2 VdP verification centers.",
    )
    p.add_argument(
        "--x_centers_key",
        type=str,
        default=None,
        help="Optional key/attribute path inside --x_centers_path, e.g. x_data or centers.",
    )
    p.add_argument(
        "--x_box_radius",
        type=str,
        default="0.01",
        help="Half-width of each local x-box around a center. Use scalar or comma-list.",
    )
    p.add_argument(
        "--x_center_stride",
        type=int,
        default=1,
        help="Use every k-th center from --x_centers_path.",
    )
    p.add_argument(
        "--max_x_centers",
        type=int,
        default=None,
        help="Optional cap on the number of centers/boxes for debugging.",
    )
    p.add_argument(
        "--x_center_unique_decimals",
        type=int,
        default=10,
        help="Round centers to this many decimals before removing duplicates. Use -1 to disable.",
    )
    p.add_argument(
        "--x_boxes_csv",
        type=str,
        default=None,
        help="CSV containing x boxes to verify, or a previous verifier CSV whose lower/upper columns will be reused.",
    )
    p.add_argument(
        "--x_boxes_csv_all_statuses",
        action="store_true",
        default=False,
        help="When loading --x_boxes_csv, use all rows instead of only rows with status=verified.",
    )

    # Z-box for zbox mode.
    p.add_argument("--z_lower", type=str, default=None)
    p.add_argument("--z_upper", type=str, default=None)
    p.add_argument("--z_radius", type=float, default=2.0)

    # Optional empirical z-box from encoder samples.
    p.add_argument("--zbox_from_encoder_samples", type=int, default=0)
    p.add_argument("--zbox_encoder_batch_size", type=int, default=4096)
    p.add_argument("--zbox_margin_fraction", type=float, default=0.05)

    # Manual box splitting.
    p.add_argument(
        "--splits_per_dim",
        type=str,
        default="8",
        help=(
            "Number of splits per dimension. Use one integer, e.g. '2', or a comma-separated "
            "list, e.g. '2,2,1,1,1'."
        ),
    )
    p.add_argument("--max_boxes", type=int, default=None, help="Optional cap on number of sub-boxes for debugging.")

    # Automatic domain refinement for hard/failed boxes.
    p.add_argument(
        "--auto_refine_failed_boxes",
        action="store_true",
        default=False,
        help=(
            "If a box cannot be certified, split that box into smaller boxes and "
            "retry automatically. This refines the domain, not the bound value."
        ),
    )
    p.add_argument(
        "--max_refine_depth",
        type=int,
        default=0,
        help=(
            "Maximum recursive refinement depth for failed boxes. "
            "0 means no recursive refinement beyond the initial grid."
        ),
    )
    p.add_argument(
        "--refine_splits_per_dim",
        type=str,
        default="2",
        help=(
            "Number of child splits per dimension when refining a failed box. "
            "Use one integer, e.g. '2', or a comma-separated list matching the input dimension."
        ),
    )
    p.add_argument(
        "--refine_abcrown_errors",
        action="store_true",
        default=False,
        help=(
            "Also refine boxes that fail due to alpha-beta-CROWN internal errors. "
            "By default, verifier/config errors are not refined because smaller boxes may not fix them."
        ),
    )

    # Bound search.
    p.add_argument(
        "--initial_hi",
        type=float,
        default=240.0,
        help="Initial upper bound L. In bisection mode this is the starting hi; in --verify_high_only mode this is the fixed target.",
    )
    p.add_argument(
        "--verify_high_only",
        action="store_true",
        default=False,
        help="Only check L < initial_hi on each box. If omitted, run bisection to find a smaller certified L.",
    )
    p.add_argument(
        "--bisect_failed_fixed_bound",
        action="store_true",
        default=False,
        help=(
            "Use with --verify_high_only. First check the fixed bound initial_hi. "
            "If a box does not verify at that target, rerun that box with bracketing "
            "and bisection to find a larger certified upper bound instead of just failing."
        ),
    )
    p.add_argument(
        "--bisect_mode",
        choices=["simple", "bracket"],
        default="bracket",
        help=(
            "Bisection mode used when --verify_high_only is not set. "
            "'simple' first checks initial_hi; if it is unknown, it expands hi "
            "like the residual verifier, then bisects. "
            "'bracket' also expands initial_hi upward until it verifies, then bisects."
        ),
    )
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--max_bisect_iters", type=int, default=16)
    p.add_argument("--max_bracket_iters", type=int, default=8)
    p.add_argument("--strict_eps", type=float, default=1e-9)

    # alpha-beta-CROWN budget.
    p.add_argument("--timeout_s", type=float, default=120.0)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--quiet", action="store_true", default=False)
    p.add_argument(
        "--complete_verifier",
        choices=["skip", "bab"],
        default="skip",
        help="Use 'skip' for incomplete-only verification. Use 'bab' only if you want to try complete BaB.",
    )

    p.add_argument(
        "--branching_method",
        type=str,
        default="kfsb",
        choices=["random", "intercept", "nonlinear", "babsr", "fsb", "kfsb"],
        help=(
            "Activation/nonlinear split branching heuristic used when "
            "--complete_verifier bab is enabled."
        ),
    )
    p.add_argument(
        "--disable_nonlinear_split",
        action="store_true",
        default=False,
        help=(
            "Disable GenBaB nonlinear splitting. Leave this off for tanh/sigmoid "
            "networks if your alpha-beta-CROWN version supports nonlinear splitting."
        ),
    )
    p.add_argument(
        "--enable_input_split",
        action="store_true",
        default=False,
        help=(
            "Enable input-space splitting. Keep this off by default for JacobianOP "
            "graphs unless activation/nonlinear splitting is not enough."
        ),
    )

    # Normalizer handling.
    p.add_argument("--no_fuse_normalizers", action="store_true", default=False)

    # Debugging / sanity checks.
    p.add_argument("--skip_shape_check", action="store_true", default=False)
    p.add_argument("--empirical_samples", type=int, default=0)
    p.add_argument("--empirical_batch_size", type=int, default=1024)

    # Output.
    p.add_argument("--results_path", type=str, default=None)
    p.add_argument(
        "--log_path",
        type=str,
        default=None,
        help=(
            "Optional text log path. If omitted and --results_path is set, "
            "the log is written next to the CSV using the same base name."
        ),
    )

    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = get_args()

    log_file = setup_log_file(args.log_path, args.results_path)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run this in your abcrown/CUDA environment.")

    device = torch.device("cuda")
    torch.manual_seed(0)

    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", None))
    print("torch cuda device name:", torch.cuda.get_device_name(0))
    print("verify_domain:", args.verify_domain)
    print("complete_verifier:", args.complete_verifier)
    print("branching_method:", args.branching_method)
    print("disable_nonlinear_split:", args.disable_nonlinear_split)
    print("enable_input_split:", args.enable_input_split)
    print("delta_z:", args.delta_z)
    print("x_region_mode:", args.x_region_mode)
    print("x_centers_path:", args.x_centers_path)
    print("x_box_radius:", args.x_box_radius)
    print("x_boxes_csv:", args.x_boxes_csv)
    print("verify_high_only:", args.verify_high_only)
    print("bisect_failed_fixed_bound:", args.bisect_failed_fixed_bound)
    print("bisect_mode:", args.bisect_mode)
    print("results_path:", args.results_path)
    print("log_path:", args.log_path if args.log_path is not None else (str(Path(args.results_path).expanduser().with_suffix(".log")) if args.results_path else None))
    print("auto_refine_failed_boxes:", args.auto_refine_failed_boxes)
    print("max_refine_depth:", args.max_refine_depth)
    print("refine_splits_per_dim:", args.refine_splits_per_dim)
    print("refine_abcrown_errors:", args.refine_abcrown_errors)

    abcrown = import_abcrown(args.abcrown_root)
    JacobianOP = abcrown["JacobianOP"]

    root = Path(args.root).expanduser().resolve()
    save_dir = (root / args.save_subdir).resolve()
    encoder_hidden = parse_int_list(args.encoder_hidden)
    inverse_hidden = parse_int_list(args.inverse_hidden)

    # Load inverse model. Required in both modes.
    Tinv_net = load_inverse_model(args, save_dir, inverse_hidden, device)

    T_net = None

    # x-box used for ximage mode, empirical z-box construction, and Remark 2 mode.
    if args.x_lower is not None or args.x_upper is not None:
        if args.x_lower is None or args.x_upper is None:
            raise ValueError("Provide both --x_lower and --x_upper, or neither.")
        x_lower = parse_float_list(args.x_lower, expected_len=args.x_dim)
        x_upper = parse_float_list(args.x_upper, expected_len=args.x_dim)
    else:
        cx, cy = args.center_x, args.center_y
        r = args.box_size
        if args.x_dim != 2:
            raise ValueError("For x_dim != 2, provide --x_lower and --x_upper.")
        x_lower = [cx - r, cy - r]
        x_upper = [cx + r, cy + r]

    # Optional encoder loading.
    need_encoder = args.verify_domain in {"ximage", "remark2"} or args.zbox_from_encoder_samples > 0
    if need_encoder:
        T_net = load_encoder_model(args, save_dir, encoder_hidden, device)

    # Build verification domain and scalar squared-Jacobian model.
    if args.verify_domain == "zbox":
        if args.zbox_from_encoder_samples > 0:
            if T_net is None:
                raise RuntimeError("Internal error: T_net should have been loaded for empirical z-box construction.")
            print("\n[z-box] Building empirical z-box from encoder samples.")
            lower, upper = empirical_zbox_from_encoder(T_net, x_lower, x_upper, n_samples=args.zbox_from_encoder_samples, batch_size=args.zbox_encoder_batch_size, margin_fraction=args.zbox_margin_fraction, device=device)
            print("[z-box] Empirical z lower:", lower)
            print("[z-box] Empirical z upper:", upper)
            print("[z-box] WARNING: this z-box is empirical, not certified to contain all T(X).")
        elif args.z_lower is not None and args.z_upper is not None:
            lower = parse_float_list(args.z_lower, expected_len=args.z_dim)
            upper = parse_float_list(args.z_upper, expected_len=args.z_dim)
        elif args.z_lower is None and args.z_upper is None:
            lower = [-args.z_radius] * args.z_dim
            upper = [args.z_radius] * args.z_dim
        else:
            raise ValueError("Provide both --z_lower and --z_upper, or neither.")

        input_dim = args.z_dim
        sq_jac_model = TinvJacobianFroSqFromZ(Tinv_net, JacobianOP).to(device).eval()

        print("\nVerification domain: zbox")
        print("Box lower:", lower)
        print("Box upper:", upper)
        print("Quantity: sup_{z in Z} ||d T_inv(z) / dz||_F")

    elif args.verify_domain == "ximage":
        lower = x_lower
        upper = x_upper
        input_dim = args.x_dim
        sq_jac_model = TinvJacobianFroSqFromXImage(T_net, Tinv_net, JacobianOP).to(device).eval()

        print("\nVerification domain: ximage")
        print("Box lower:", lower)
        print("Box upper:", upper)
        print("Quantity: sup_{x in X} ||d T_inv(z) / dz at z=T(x)||_F")

    else:
        if args.delta_z <= 0:
            raise ValueError("--verify_domain remark2 requires --delta_z > 0.")

        dz_lower = [-float(args.delta_z)] * args.z_dim
        dz_upper = [ float(args.delta_z)] * args.z_dim
        lower = list(x_lower) + dz_lower
        upper = list(x_upper) + dz_upper
        input_dim = args.x_dim + args.z_dim
        sq_jac_model = TinvJacobianFroSqRemark2(T_net, Tinv_net, JacobianOP, x_dim=args.x_dim, z_dim=args.z_dim).to(device).eval()

        print("\nVerification domain: remark2")
        print("X lower:", x_lower)
        print("X upper:", x_upper)
        print("delta_z:", args.delta_z)
        print("Combined input lower [x,dz]:", lower)
        print("Combined input upper [x,dz]:", upper)
        print("Quantity: sup_{x in X, dz in [-delta_z,delta_z]^z_dim} ||d T_inv/dz at z=T(x)+dz||_F")
        print("This certifies a conservative superset of T(X) \\oplus B_delta_z.")

    # For the rectangular case, this is the whole box. For a union-of-boxes
    # VdP region, this is just the first local box; the full box list is built below.
    shape_lower = lower
    shape_upper = upper

    if not args.skip_shape_check:
        run_shape_check(sq_jac_model, input_dim, shape_lower, shape_upper, device)

    if args.empirical_samples > 0:
        print("\n[empirical] Running ordinary-autograd empirical sanity check.")
        t_emp = time.time()
        if args.verify_domain == "zbox":
            empirical_est = empirical_fro_lipschitz_estimate_autograd_zbox(Tinv_net, lower, upper, n_samples=args.empirical_samples, batch_size=args.empirical_batch_size, device=device)
        elif args.verify_domain == "ximage":
            empirical_est = empirical_fro_lipschitz_estimate_autograd_ximage(T_net, Tinv_net, lower, upper, n_samples=args.empirical_samples, batch_size=args.empirical_batch_size, device=device)
        else:
            empirical_est = empirical_fro_lipschitz_estimate_autograd_remark2(T_net, Tinv_net, lower, upper, x_dim=args.x_dim, z_dim=args.z_dim, n_samples=args.empirical_samples, batch_size=args.empirical_batch_size, device=device)
        print(f"[empirical] Max ||J||_F over {args.empirical_samples} random samples: {empirical_est:.8g}")
        print(f"[empirical] Time: {time.time() - t_emp:.2f} seconds")

    splits = parse_splits(args.splits_per_dim, input_dim)

    if args.verify_domain == "zbox" or args.x_region_mode == "box":
        # Original behavior: one rectangular verification box, split by splits_per_dim.
        boxes = make_subboxes(lower, upper, splits)
        print("\n[region] Using one rectangular verification box.")
    else:
        if args.verify_domain not in {"ximage", "remark2"}:
            raise ValueError("--x_region_mode centers/boxes_csv is only valid for ximage or remark2 modes.")

        if args.x_region_mode == "centers":
            if args.x_centers_path is None:
                raise ValueError("--x_region_mode centers requires --x_centers_path.")
            centers = load_x_centers(args.x_centers_path, key=args.x_centers_key)
            x_boxes = make_x_boxes_from_centers(centers, radius=args.x_box_radius, x_dim=args.x_dim, x_lower_clip=x_lower, x_upper_clip=x_upper, stride=args.x_center_stride, max_centers=args.max_x_centers, unique_decimals=args.x_center_unique_decimals)
            print("\n[region] Loaded center-based VdP region.")
            print("x_centers_path:", args.x_centers_path)
            print("raw centers shape:", tuple(centers.shape))
            print("local x-box radius:", args.x_box_radius)
            print("number of local x-boxes before optional splitting:", len(x_boxes))
        elif args.x_region_mode == "boxes_csv":
            if args.x_boxes_csv is None:
                raise ValueError("--x_region_mode boxes_csv requires --x_boxes_csv.")
            x_boxes = load_x_boxes_from_csv(
                args.x_boxes_csv,
                x_dim=args.x_dim,
                verified_only=(not args.x_boxes_csv_all_statuses),
            )
            print("\n[region] Loaded x-box region from CSV.")
            print("x_boxes_csv:", args.x_boxes_csv)
            print("number of x-boxes before optional splitting:", len(x_boxes))
        else:
            raise ValueError(f"Unsupported x_region_mode={args.x_region_mode!r}.")

        if args.verify_domain == "remark2":
            base_boxes = attach_dz_box_to_x_boxes(x_boxes, delta_z=args.delta_z, z_dim=args.z_dim)
        else:
            base_boxes = x_boxes

        boxes = split_each_box(base_boxes, splits)

    if args.max_boxes is not None:
        boxes = boxes[: args.max_boxes]

    if len(boxes) == 0:
        raise RuntimeError("No verification boxes were constructed.")

    print("\n[verify] Starting certified split-box bound search.")
    print("Number of boxes:", len(boxes))
    print("splits_per_dim:", splits)
    if args.x_region_mode != "box":
        print("NOTE: boxes are a union of local VdP boxes, not one full rectangle.")

    t0 = time.time()
    global_hi, results = verify_boxes(abcrown=abcrown, sq_jac_model=sq_jac_model, input_dim=input_dim, boxes=boxes, args=args)
    elapsed = time.time() - t0

    save_results(results, args.results_path)

    leaf_results = [r for r in results if bool(r.get("is_leaf", True))]
    failed = [r for r in leaf_results if r["status"] != "verified"]
    verified = [r for r in leaf_results if r["status"] == "verified"]
    refined_parents = [r for r in results if not bool(r.get("is_leaf", True))]

    print("\n==================== RESULT ====================")
    print(f"Verified active leaf boxes: {len(verified)} / {len(leaf_results)}")
    print(f"Failed/inconclusive active leaf boxes: {len(failed)} / {len(leaf_results)}")
    print(f"Refined parent boxes: {len(refined_parents)}")
    print(f"Certified upper bound over verified active leaves: L <= {global_hi:.10g}")
    if failed:
        print("WARNING: At least one active leaf box failed to certify, so the global result is not a full certificate over the whole domain.")
    else:
        print("FULL DOMAIN CERTIFIED over the active refined leaf partition.")
    print(f"Total verification time: {elapsed:.2f} seconds")
    print("This certifies a Frobenius-Jacobian bound. It also bounds the spectral-norm Lipschitz constant because ||J||_2 <= ||J||_F.")

    if log_file is not None:
        print("Log saved.")
        log_file.flush()


if __name__ == "__main__":
    main()