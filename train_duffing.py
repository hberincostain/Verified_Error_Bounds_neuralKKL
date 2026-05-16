import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
from pathlib import Path
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from Dataset import DataSet, ZToXDataset
from Normalizer import Normalizer
from NN import Main_Network
from Systems import RevDuff


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


DEFAULT_ENCODER_PATH = (
    "saved_models_revduff/"
    "revduff_N=800_num_ic=1000_box_size=3.0_"
    "hidden_sizes=[100, 100, 100, 100, 100, 100, 100]_"
    "diag=[-1.0, -2.0, -3.0, -4.0, -5.0]_"
    "inverse_size=[100, 100, 100, 100, 100, 100]_"
    "pde_loss=0.00074600836_lmbda=10.0_B_magni=1.0_"
    "supervised_PINN_encoder.pt"
)

def cast_normalizer_to_float32(normalizer, device):
    if normalizer is None:
        return None

    for name, value in vars(normalizer).items():
        if torch.is_tensor(value):
            setattr(normalizer, name, value.to(device=device, dtype=torch.float32))

    return normalizer

def _resolve_path(path):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _first_output(model, x):
    out = model(x)
    if isinstance(out, tuple):
        return out[0]
    return out


def load_fixed_encoder(encoder_path, x_dim, z_dim, hidden_sizes):
    """
    Load the old-format encoder checkpoint:
        {"model_state_dict": ..., "normalizer": ...}

    This intentionally matches the original code's Main_Network/tanh setup.
    """
    encoder_path = _resolve_path(encoder_path)
    if not encoder_path.exists():
        raise FileNotFoundError(f"Could not find encoder checkpoint: {encoder_path}")

    print("Loading fixed encoder from:")
    print("   ", encoder_path)

    checkpoint = torch.load(encoder_path, map_location=device, weights_only=False)

    if "model_state_dict" not in checkpoint:
        raise ValueError(
            "Expected encoder checkpoint with key 'model_state_dict'. "
            f"Got keys: {list(checkpoint.keys())}"
        )

    normalizer = checkpoint.get("normalizer", None)
    
    if normalizer is None:
        raise ValueError("Encoder checkpoint does not contain a normalizer.")
    
    normalizer = cast_normalizer_to_float32(normalizer, device)

    # Match the original legacy encoder architecture: tanh hidden activations.
    activation = F.tanh
    encoder = Main_Network(
        x_dim,
        z_dim,
        hidden_sizes=hidden_sizes,
        activation=activation,
        normalizer=normalizer,
    )

    encoder.load_state_dict(checkpoint["model_state_dict"], strict=True)
    encoder.to(device)
    encoder.eval()

    for p in encoder.parameters():
        p.requires_grad_(False)

    print("Fixed encoder loaded and frozen.")
    return encoder, normalizer


def save_decoder_old_style(decoder, normalizer, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": decoder.state_dict(),
            "normalizer": normalizer,
        },
        save_path,
    )
    print("Saved decoder to:", save_path)


def save_decoder_direct_for_verifier(decoder, normalizer, save_path, x_size, z_size, inverse_size, activation_name):
    """
    Optional direct format for the newer verifier:
        {"model": state_dict, "config": config}
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if len(set(inverse_size)) != 1:
        print("[warning] Direct verifier format assumes uniform hidden size. Skipping direct save.")
        return

    config = {
        "num_hidden": len(inverse_size),
        "hidden_size": inverse_size[0],
        "x_size": x_size,
        "z_size": z_size,
        "normalizer": normalizer,
        "activation": activation_name,
        "inverse_input_size": z_size,
        "inverse_output_size": x_size,
        "inverse_hidden_sizes": inverse_size,
    }

    torch.save(
        {
            "model": decoder.net.state_dict() if hasattr(decoder, "net") else decoder.state_dict(),
            "config": config,
        },
        save_path,
    )
    print("Saved direct verifier-format decoder to:", save_path)


def main():
    start_time = time.time()

    # ------------------------------------------------------------------
    # Default settings.
    # These are used when you simply run:
    #
    #     python3 train_duffing_less_data.py
    #
    # You can still override them using the old positional command format.
    # ------------------------------------------------------------------
    save_dir = "saved_models_revduff"
    method = "supervised_PINN"
    hidden_sizes = [100, 100, 100, 100, 100, 100, 100]
    lip_scale = 0.0  # kept only for backward compatibility
    inverse_size = [128, 128, 128]
    num_ic = 1000
    batch_size = 1024
    lmbda = 0.1
    diag = [-1.0, -2.0, -3.0, -4.0, -5.0]
    B_magni = 1.0
    encoder_path = DEFAULT_ENCODER_PATH

    if len(sys.argv) > 1:
        if len(sys.argv) < 11:
            print(
                "Usage: python train_duffing_less_data.py "
                "<save_dir> <method> <hidden_sizes> <lip_scale_or_unused> "
                "<inverse_size> <num_ic> <batch_size> <lambda> <A_diag> <B_magni> "
                "[encoder_path]"
            )
            return

        save_dir = sys.argv[1]
        method = sys.argv[2]
        hidden_sizes = list(map(int, sys.argv[3].split(',')))
        lip_scale = float(sys.argv[4])
        inverse_size = list(map(int, sys.argv[5].split(',')))
        num_ic = int(sys.argv[6])
        batch_size = int(sys.argv[7])
        lmbda = float(sys.argv[8])
        diag = list(map(float, sys.argv[9].split(',')))
        B_magni = float(sys.argv[10])
        encoder_path = sys.argv[11] if len(sys.argv) >= 12 else DEFAULT_ENCODER_PATH

    print("Using settings:")
    print("  save_dir:", save_dir)
    print("  method:", method)
    print("  hidden_sizes:", hidden_sizes)
    print("  inverse_size:", inverse_size)
    print("  num_ic:", num_ic)
    print("  batch_size:", batch_size)
    print("  lambda:", lmbda)
    print("  diag:", diag)
    print("  B_magni:", B_magni)
    print("  encoder_path:", encoder_path)

    methods = ['supervised_NN', 'unsupervised_AE', 'supervised_PINN', 'Neural_ODE']
    if method not in methods:
        raise Exception(f"Invalid method {method}. Use one of {methods}. ELM branch was removed.")

    print("Device:", device)
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))

    # ------------------ System Setup ------------------
    zdim = 5
    revduff = RevDuff(zdim=zdim, specify_zdim=True)

    x_size = revduff.x_size
    z_size = revduff.z_size

    # Match the saved dataset and encoder.
    box_size = 3.0
    limits = np.array([[-box_size, box_size], [-box_size, box_size]])
    a, b = 0, 2000
    N = 800

    A = np.diag(diag)
    B = B_magni * np.ones([zdim, 1])

    # ------------------ Load or generate x-data ------------------
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)

    dataset_path = save_dir_path / (
        "revduff_dataset_N={}_num_ic={}_zdim={}_box_size={}_end_time={}_"
        "A_diag={}_B_magnitude={}_mode={}"
    ).format(N, num_ic, zdim, box_size, b, diag, B_magni, "boundary_cond")

    print("Trying to load dataset from:", dataset_path)

    try:
        with open(dataset_path, "rb") as f:
            dataset = pickle.load(f)
        print("Loaded dataset from:", dataset_path)
    except Exception as e:
        print("Could not load dataset; generating a new one.")
        print("Reason:", repr(e))
        dataset = DataSet(revduff, A, B, a, b, N, num_ic, limits, PINN_sample_mode='split traj', data_gen_mode='boundary_cond')
        with open(dataset_path, "wb") as f:
            pickle.dump(dataset, f)
        print("Saved dataset to:", dataset_path)

    print("xdata size:", dataset.x_data.shape[0])
    print("data generation/loading took:", time.time() - start_time)

    # ------------------ Load fixed encoder ------------------
    encoder, encoder_normalizer = load_fixed_encoder(encoder_path, x_size, z_size, hidden_sizes)

    # ------------------ Build decoder ------------------
    decoder_activation = F.tanh
    decoder_activation_name = "tanh"

    normalizer = encoder_normalizer
    decoder = Main_Network(z_size, x_size, inverse_size, decoder_activation, None).to(device)

    # ------------------ Generate z = T(x) using fixed encoder ------------------
    x, _, _, _, _ = dataset[:]
    x_flat = x.reshape(-1, x.shape[-1]).float().to(device)

    with torch.no_grad():
        z_pred = _first_output(encoder, x_flat)

    z_to_x_dataset = ZToXDataset(z_pred.detach().cpu(), x_flat.detach().cpu())
    z_loader = DataLoader(z_to_x_dataset, batch_size=batch_size, shuffle=True)

    # ------------------ Train decoder only ------------------
    learning_rate = 0.001
    optimizer_d = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
    scheduler_d = ReduceLROnPlateau(optimizer_d, mode='min', factor=0.5, patience=10)
    loss_fn_d = nn.MSELoss(reduction='mean')

    decoder.train()
    epochs = 100

    print("Beginning decoder/inverse training only.")
    for epoch in range(epochs):
        loss_sum = 0.0
        num_steps = 0

        for z_batch, x_batch in z_loader:
            z_batch = z_batch.to(device)
            x_batch = x_batch.to(device)

            optimizer_d.zero_grad()
            x_hat = _first_output(decoder, z_batch)
            loss = loss_fn_d(x_hat, x_batch)
            loss.backward()
            optimizer_d.step()

            loss_sum += loss.item()
            num_steps += 1

        avg_loss = loss_sum / max(num_steps, 1)
        scheduler_d.step(avg_loss)

        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"[Decoder] Epoch {epoch + 1}, Loss: {avg_loss:.6e}")

    decoder.eval()

    # ------------------ Quick training-set error check ------------------
    with torch.no_grad():
        x_hat = _first_output(decoder, z_pred.to(device))
        err = x_hat - x_flat
        l2 = torch.norm(err, dim=1)
        linf = torch.linalg.vector_norm(err, ord=float("inf"), dim=1)
        print("Training-set max L2 inverse error:", l2.max().item())
        print("Training-set mean L2 inverse error:", l2.mean().item())
        print("Training-set max Linf inverse error:", linf.max().item())
        print("Point of max Linf error:", x_flat[linf.argmax()].detach().cpu())

    # ------------------ Save decoder ------------------
    base_name = (
        f"revduff_fixed_encoder_N={N}_num_ic={num_ic}_box_size={box_size}_"
        f"w={hidden_sizes[0]}_d={len(hidden_sizes)}_"
        f"diag={[round(x, 3) for x in diag]}_"
        f"inv_w={inverse_size[0]}_inv_d={len(inverse_size)}_"
        f"lmbda={lmbda}_B_magni={B_magni}_{method}"
    )

    old_style_decoder_path = save_dir_path / f"{base_name}_decoder.pt"
    direct_decoder_path = save_dir_path / f"{base_name}_decoder_direct.pt"

    save_decoder_old_style(decoder, normalizer, old_style_decoder_path)
    save_decoder_direct_for_verifier(decoder, normalizer, direct_decoder_path, x_size, z_size, inverse_size, decoder_activation_name)

    print("Training complete.")
    print("Total time:", time.time() - start_time)


if __name__ == '__main__':
    main()