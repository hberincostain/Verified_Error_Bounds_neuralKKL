import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
from smt.sampling_methods import FullFactorial
import torch.nn.functional as F
import pickle
from Dataset import DataSet, ZToXDataset
from Normalizer import Normalizer
from Trainer import Trainer
from NN import Main_Network
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from Systems import VdP
import time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


"""
Tunable parameters include: tau in softmax, whether to use softmax for top lmbda %, lmbda, size of network, system parameter(mu), data generation accuracy(b, N), number of data points(num_ic), A matrix
batch_size, pde&data weights, whether to mine hard points or not, whether to use lbfgs or not, whether to add more pde coallocation points on the boundary or not, how many lbfgs training points should we use, what loss function should we use for lbfgs
"""

def softmax_max(v, tau=50.0):
    # v: (batch,)
    return torch.logsumexp(tau*v, dim=0) / tau
def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch

def main():
    if len(sys.argv) < 4:
        print("Usage: python train_vdp.py <save_dir> <method> <hidden_sizes>")
        return
    start_time = time.time()
    save_dir = sys.argv[1]
    method = sys.argv[2]
    hidden_sizes = list(map(int, sys.argv[3].split(',')))
    inverse_size = list(map(int, sys.argv[5].split(',')))
    lmbda = float(sys.argv[8])

    methods = ['supervised_NN', 'unsupervised_AE', 'supervised_PINN']
    if method not in methods:
        raise Exception("Invalid choice of method")

    # ------------------ System Setup ------------------
    zdim = 5
    mu = 1
    vdp = VdP(zdim = zdim, my = mu)

    x_size = vdp.x_size
    z_size = vdp.z_size
    y_size = vdp.y_size

    box_size = 2.7
    limits = np.array([[-2.1, 2.1], [-box_size, box_size]])  # data points sampling range
    a, b = 0, 20                                            # b is the cut-off time for evaluating the integral expression of T(x) 20 is enough since e^-20~10^-17
    N = 14000                                               # the number of grid points for evaluating the integral
    num_ic = int(sys.argv[6])                               # The number of sampling points x for which we calculate T(x), not all sample points will be within the limit cycle

    diag = list(map(float, sys.argv[9].split(',')))          # We'll only use diagonal A for convenience
    A = np.diag(diag)

    B_magni = float(sys.argv[10])
    B = B_magni*np.ones([zdim,1])

    try:#check if the dataset is save
        with open(save_dir+"/vdp_dataset_N={}_num_ic={}_zdim={}_box_size={}_end_time={}_A_diag={}_B_magnitude={}_mode={}_mu={}".format(N, num_ic, zdim, box_size, b, diag, B_magni, "mixed_merged", mu), "rb") as f:
            dataset = pickle.load(f)
        print("/vdp_dataset_N={}_num_ic={}_zdim={}_box_size={}_end_time={}_A_diag={}_B_magnitude={}_mode={}_mu={}".format(N, num_ic, zdim, box_size, b, diag, B_magni, "mixed_merged", mu))
    except:
        #Generate data points on the boundary by solving the pde using method of characteristic curves(the characteriztic curves are the solution curves of the system. For vdp, they all converge to the limit cycle. 
        # The solution along characteristic curves have a term that depends on B.C., but that term decays exponentially, so we only take T(x(t)) computed using this method for the x(t) with large t)
        dataset_bc = DataSet(vdp, A, B, a, 200, N, 8, limits,
            PINN_sample_mode='no physics', data_gen_mode='boundary_cond')
        print("bc points: ", dataset_bc.x_data.shape[0])
        # Generate data points in interior by sampling uniformly in a box that contains the limit cycle, then compute the integral. If the integral is inf, that means that point is outside the limit cycle and we get rid of it.
        dataset = DataSet(vdp, A, B, a, b, N, num_ic, limits, PINN_sample_mode='no physics', data_gen_mode='integral')
        dataset.x_data = torch.cat([dataset.x_data, dataset_bc.x_data]) #We merge the two datasets (boundary and interior)
        dataset.x_data_ph = torch.cat([dataset.x_data_ph, dataset_bc.x_data_ph])
        dataset.z_data = torch.cat([dataset.z_data, dataset_bc.z_data])
        dataset.z_data_ph = torch.cat([dataset.z_data_ph, dataset_bc.z_data_ph])
        dataset.output_data = torch.cat([dataset.output_data, dataset_bc.output_data])
        dataset.output_data_ph = torch.cat([dataset.output_data_ph, dataset_bc.output_data_ph])
        dataset.data_length = dataset.x_data.shape[0]
        with open(save_dir+"/vdp_dataset_N={}_num_ic={}_zdim={}_box_size={}_end_time={}_A_diag={}_B_magnitude={}_mode={}_mu={}".format(N, num_ic, zdim, box_size, b, diag, B_magni, "mixed_merged", mu), "wb") as f:
            pickle.dump(dataset, f)
    print("Data from: ", "/vdp_dataset_N={}_num_ic={}_zdim={}_box_size={}_end_time={}_A_diag={}_B_magnitude={}_mode={}".format(N, num_ic, zdim, box_size, b, diag, B_magni, "mixed_merged"))
    print("xdata size: ", dataset.x_data.shape[0])
    print("xdata_ph size: ", dataset.x_data.shape[0])
    print("data generation took: ", time.time()-start_time)
    # plot_traj(dataset.x_data[:, 0].cpu(), dataset.x_data[:, 1].cpu())

    print("Dataset generated.\n")

    activation = F.tanh     # Use tanh to reduce the lipschits const of the inverse
    normalizer = Normalizer(dataset)
    encoder = Main_Network(x_size, z_size, hidden_sizes, activation, normalizer=normalizer)
    decoder = Main_Network(z_size, x_size, inverse_size, activation, None)

    # ------------------ Train Encoder ------------------
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-8)   
    loss_fn = nn.MSELoss(reduction='mean')
    data_batch_size = 1028
    pde_batch_size = 512
    def pde_residual(x, T_net, f, h, A, b, tau, return_norms = False):
        A = torch.as_tensor(A, dtype=x.dtype, device=device)
        b = torch.as_tensor(b, dtype=x.dtype, device=device)
        T_val, _ = T_net(x)        # [batch, n]
        f_val = f(x)            # [batch, dim]
        h_val = h(x).unsqueeze(1)
        jac_list = []
        for i in range(T_val.shape[1]):
            grad_i = torch.autograd.grad(
                T_val[:, i].sum(), x, create_graph=True
            )[0]  # [batch, dim]
            jac_list.append(grad_i)
        J = torch.stack(jac_list, dim=1)  # [batch, n, dim]

        dTfx = torch.bmm(J, f_val.unsqueeze(-1)).squeeze(-1)  # [batch, n]

        residual = dTfx - (T_val @ A.T) - (h_val * b.T)
        norms = torch.norm(residual, dim=1)
        if return_norms: return norms
        mean_norm = norms.mean()
        # max_norm = norms.max()

        # k = max(1, int(lmbda * norms.shape[0]))  # top 20%
        # # print(k)
        # topk_vals, _ = torch.topk(norms, k)
        # max_norm = topk_vals.mean()
        max_norm = softmax_max(norms, tau=tau)
        return max_norm, mean_norm
    
    def mine_hard_points(points, K=8192, chunk=2048):
        encoder.eval()
        all_norms = []
        with torch.enable_grad():
            for i in range(0, points.shape[0], chunk):
                x = points[i:i+chunk].to(device).detach().requires_grad_(True)
                norms = pde_residual(x, encoder, vdp.torch_function_batch, vdp.output_batch, A, B, 1, return_norms=True)
                all_norms.append(norms.detach().cpu())
        all_norms = torch.cat(all_norms)
        idx = torch.topk(all_norms, k=min(K, all_norms.numel())).indices
        return points[idx]

    epochs = 200
    
    dataset_f = TensorDataset(dataset.x_data_ph.float())
    loader_pde = DataLoader(dataset_f, batch_size=500, shuffle=True)

    dataset_b = TensorDataset(dataset.x_data[::].float(), dataset.z_data[::].float())
    loader_data = DataLoader(dataset_b, batch_size=int(sys.argv[7]), shuffle=True)
    data_iter = infinite_loader(loader_data)

    for epoch in range(epochs):
        encoder.train()
        loss_sum = 0
        pde_loss_sum = 0
        data_loss_sum = 0
        bc_loss_sum = 0
        num_steps = 0
        tau  = 50 * (150/50)**min(1.0, epoch / 500)  # geometric
        w_pde = min(1.0, epoch / 500)   # linear
        for (x_ph_batch,) in loader_pde:
            x_data_batch, z_data_batch = next(data_iter)
            u_pred, _ = encoder(x_data_batch)
            residual = (u_pred - z_data_batch)
            norms = torch.norm(residual, dim=1)
            data_loss = norms.mean()

            # ---- PDE loss (sample a minibatch of PDE points) ----
            x_ph_batch.requires_grad = True
            
            # x_bc_pde = x_bc_mix.detach().requires_grad_(True)
            
            norms = pde_residual(x_ph_batch, encoder, vdp.torch_function_batch, vdp.output_batch, A, B, 1, return_norms=True)
            pde_loss = torch.mean(norms)

            # max_norm, mean_norm = pde_residual(x_ph_batch, encoder, vdp.torch_function_batch, vdp.output_batch, A, B, tau)
            # pde_loss = lmbda * mean_norm + (1-lmbda) * max_norm

            # ---- Combined loss ----
            
            loss = data_loss+pde_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum+=loss.item()
            pde_loss_sum+=pde_loss.item()
            data_loss_sum+=data_loss.item()
            num_steps += 1
        
        scheduler.step(loss_sum / num_steps)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: "
                    f"Loss = {(loss_sum/num_steps):.6e}, "
                    f"PDE = {(pde_loss_sum/num_steps):.6e}, "
                    f"data = {(data_loss_sum/num_steps):.6e}, ")
        if epoch % 34 == 0:
            norms = pde_residual(dataset.x_data.float().requires_grad_(), encoder, vdp.torch_function_batch, vdp.output_batch, A, B, 0.00001, return_norms=True)
            pde_err = torch.max(norms)
            max_id = torch.argmax(norms)
            print("max pde error: ", pde_err)
            print("max pde error point: ", dataset.x_data[max_id])

    print("training took: ", time.time()-start_time)
    norms = pde_residual(dataset.x_data.float().requires_grad_(), encoder, vdp.torch_function_batch, vdp.output_batch, A, B, 0.00001, return_norms=True)
    pde_err = torch.max(norms)
    max_id = torch.argmax(norms)
    print("max pde error: ", pde_err)
    print("max pde error point: ", dataset.x_data[max_id])
    torch.save({'model_state_dict': encoder.state_dict(), 'normalizer': normalizer}, save_dir+f"/vdp_N={N}_num_ic={num_ic}_box_size={box_size}_w={hidden_sizes[0]}_d={len(hidden_sizes)}_diag={[round(x, 3) for x in diag]}_inv_w={inverse_size[0]}_inv_d={len(inverse_size)}_pde_loss={pde_err:.8g}_lmbda={lmbda}_B_magni={B_magni}_" + method + '_encoder.pt')
    print(f"/vdp_N={N}_num_ic={num_ic}_box_size={box_size}_w={hidden_sizes[0]}_d={len(hidden_sizes)}_diag={[round(x, 3) for x in diag]}_inv_w={inverse_size[0]}_inv_d={len(inverse_size)}_pde_loss={pde_err:.8g}_lmbda={lmbda}_B_magni={B_magni}_" + method + '_encoder.pt')
    
    # ------------------ Train LBFGS ------------------
    x_ph = dataset.x_data_ph[::].float()
    x_ph.requires_grad = True

    # Fine-tuning optimizer
    lbfgs = torch.optim.LBFGS(encoder.parameters(), lr=1.0, max_iter=500, history_size=50, tolerance_grad=1e-8, 
                              tolerance_change=1e-9, line_search_fn="strong_wolfe")

    start_time = time.time()
    K = 50000
    for i in range(1, 10):
        tau = 0.0001
        x_lbfgs = mine_hard_points(dataset.x_data[::].float(), K=K)
        
        def closure():
            lbfgs.zero_grad()
            x_interior = x_lbfgs.to(device).detach().requires_grad_(True)
            norms = pde_residual(x_interior, encoder, vdp.torch_function_batch, vdp.output_batch, A, B, tau, return_norms=True)
            pde_loss = torch.mean(norms)
            loss = pde_loss
            loss.backward()
            return loss    
           
        print("Starting L-BFGS...")
        print("tau", tau, " K: ", K)

        for step in range(int(sys.argv[4])):  # each step does up to max_iter LBFGS iters
            loss = lbfgs.step(closure)

        # max_pde_err, mean_pde_err = pde_residual(dataset.x_data.float().requires_grad_(), encoder, vdp.torch_function_batch, vdp.output_batch, A, B, 0.00001)
        norms = pde_residual(dataset.x_data.float().requires_grad_(), encoder, vdp.torch_function_batch, vdp.output_batch, A, B, 0.00001, return_norms=True)
        max_norm = torch.max(norms)
        max_id = torch.argmax(norms)
        print("max pde error: ", max_norm)
        print("max pde error point: ", dataset.x_data[max_id])
        # print("pde loss: ", max_pde_err, mean_pde_err)

    
    print(f"[L-BFGS] Final Loss {loss.item():.3e}")
    
    norms = pde_residual(dataset.x_data.float().requires_grad_(), encoder, vdp.torch_function_batch, vdp.output_batch, A, B, 0.00001, return_norms=True)
    pde_err = torch.max(norms).item()
    max_id = torch.argmax(norms)

    print("max pde error: ", pde_err)
    print("max pde error point: ", dataset.x_data[max_id])

    # Freeze encoder
    encoder.eval()
    print("training took: ", time.time()-start_time)
    start_time = time.time()
    torch.save({'model_state_dict': encoder.state_dict(), 'normalizer': normalizer}, save_dir+f"/vdp_N={N}_num_ic={num_ic}_box_size={box_size}_w={hidden_sizes[0]}_d={len(hidden_sizes)}_diag={[round(x, 3) for x in diag]}_inv_w={inverse_size[0]}_inv_d={len(inverse_size)}_pde_loss={pde_err:.8g}_lmbda={lmbda}_B_magni={B_magni}_" + method + '_encoder.pt')
    print(f"/vdp_N={N}_num_ic={num_ic}_box_size={box_size}_w={hidden_sizes[0]}_d={len(hidden_sizes)}_diag={[round(x, 3) for x in diag]}_inv_w={inverse_size[0]}_inv_d={len(inverse_size)}_pde_loss={pde_err:.8g}_lmbda={lmbda}_B_magni={B_magni}_" + method + '_encoder.pt')
    
    # ------------------ Train Decoder ------------------
    data_batch_size = 500
    learning_rate = 1e-3
    x, _, _, _, _ = dataset[:]
    x.to(device)
    x_flat = x.reshape(-1, x.shape[-1])
    with torch.no_grad():
        z_pred, _ = encoder(x_flat)

    z_to_x_dataset = ZToXDataset(z_pred, x_flat)
    z_loader = DataLoader(z_to_x_dataset, batch_size=data_batch_size, shuffle=True)

    optimizer_d = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
    scheduler_d = ReduceLROnPlateau(optimizer_d, mode='min', factor=0.5, patience=10)
    loss_fn_d = nn.MSELoss(reduction='mean')

    decoder.train()
    epochs = 80
    for epoch in range(epochs):
        loss_sum = 0
        lip_loss_sum = 0
        for z_batch, x_batch in z_loader:
            optimizer_d.zero_grad()
            x_hat, _ = decoder(z_batch.to(device))
            # lip_loss = spectral_norm_penalty(decoder)
            loss = loss_fn_d(x_hat.to(device), x_batch.to(device))#+lip_scale*lip_loss
            loss.backward()
            optimizer_d.step()
            loss_sum += loss.item()
            #lip_loss_sum+=lip_loss.item()
        scheduler_d.step(loss_sum)
        if epoch % 10 == 0:
            print(f"[Decoder] Epoch {epoch+1}, Loss: {loss_sum/len(z_loader):.6f}")
            print(f"[Decoder] Epoch {epoch+1}, Lipschizt Loss: {lip_loss_sum/len(z_loader):.6f}")

    torch.save({'model_state_dict': decoder.state_dict(), 'normalizer': normalizer}, save_dir+f"/vdp_N={N}_num_ic={num_ic}_box_size={box_size}_w={hidden_sizes[0]}_d={len(hidden_sizes)}_diag={[round(x, 3) for x in diag]}_inv_w={inverse_size[0]}_inv_d={len(inverse_size)}_pde_loss={pde_err:.8g}_lmbda={lmbda}_B_magni={B_magni}_" + method + '_decoder.pt')

    print("Training complete.\n")
    print("inverse training took: ", time.time()-start_time)
    decoder.eval()



if __name__ == '__main__':
    main()
