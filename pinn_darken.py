#!/usr/bin/env python3
"""
pinn_darken.py (Final version with all fixes and enhancements)

Non‑linear diffusion (Cahn‑Hilliard‑type) problem where the
inter‑diffusion coefficient is given by **Darken's model** :

    ∂C/∂t = ∂/∂x [  D̃(C) ∂C/∂x  ]

    D̃(C) = ( C_B·D_A(C) + C_A·D_B(C) )          # mobility (mechanical) term
           + (RT/Ω)·∂lnγ/∂C                    # thermodynamic term

C_A (=C) : mole fraction of component A  (0 ≤ C ≤ 1)
C_B      : 1 – C

The script does the following
   1. Generate a "ground‑truth" solution with a simple explicit FDM.
   2. Build a PINN that simultaneously learns C(t,x), D_A(C), D_B(C), and lnγ(C).
   3. Train the PINN with data, PDE, and BC losses, using a progress bar and LR scheduler.
   4. Visualise the results, including a loss history plot.

Required packages
-----------------
    torch >= 1.9
    numpy
    matplotlib
    tqdm
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

if torch.cuda.is_available():
    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU is available. Using device: {device} ({gpu_name})\n")
else:
    device = torch.device('cpu')
    print("GPU not available. Using device: CPU\n")


def true_diffusion_coefficient(C):
    return 0.01 + 0.04 * C


def solve_nonlinear_diffusion_fdm(C_left, C_right,
                                 L=1.0, T_end=10.0,
                                 Nx=101, Nt=2001):
    print('Step 1: Generating true FDM data...')
    x = np.linspace(0, L, Nx)
    dx = x[1] - x[0]
    
    D_max = true_diffusion_coefficient(1.0)  # Maximum diffusion coefficient
    dt_stable = 0.4 * dx**2 / D_max  # Safety factor of 0.4 < 0.5
    Nt_stable = int(T_end / dt_stable) + 1
    dt = T_end / (Nt_stable - 1)
    t = np.linspace(0, T_end, Nt_stable)
    
    print(f'  Using stable time step: dt = {dt:.6f}, Nt = {Nt_stable}')
    
    C = np.zeros((Nt_stable, Nx))
    C[0, :] = np.where(x <= L / 2, C_left, C_right)

    for n in range(Nt_stable - 1):
        C_old = C[n, :]
        D_vals = true_diffusion_coefficient(C_old)
        
        alpha_max = np.max(D_vals) * dt / (dx ** 2)
        if alpha_max > 0.5 and n == 0:
            print(f'  Warning: α_max = {alpha_max:.3f} > 0.5  (scheme may be unstable)')

        for i in range(1, Nx - 1):
            D_ip = (D_vals[i] + D_vals[i + 1]) / 2.0
            D_im = (D_vals[i] + D_vals[i - 1]) / 2.0
            flux_ip = D_ip * (C_old[i + 1] - C_old[i]) / dx
            flux_im = D_im * (C_old[i] - C_old[i - 1]) / dx
            C[n + 1, i] = C_old[i] + dt / dx * (flux_ip - flux_im)
        
        C[n + 1, 0] = C[n + 1, 1]
        C[n + 1, -1] = C[n + 1, -2]
        
        C[n + 1, :] = np.clip(C[n + 1, :], 0.0, 1.0)

    print('FDM generation finished.\n')
    return x, t, C


class NonlinearDiffusionPINN(nn.Module):
    def __init__(self, layers_C, layers_DA, layers_DB, layers_gamma,
                 C_left, C_right, L=1.0,
                 R=8.314, T=300.0, Omega=25000.0):
        super().__init__()
        self.net_C      = self._build_net(layers_C, final_activation=nn.Sigmoid())
        self.net_DA     = self._build_net(layers_DA)
        self.net_DB     = self._build_net(layers_DB)
        self.net_gamma  = self._build_net(layers_gamma)

        self.C_left = C_left
        self.C_right = C_right
        self.L = L
        self.R = R
        self.T = T
        self.Omega = Omega
        self.RT_Omega = (R * T) / Omega
        
        self.gamma_scale = 0.5
        self.thermo_clip_val = 0.1

        with torch.no_grad():
            self.net_gamma[-1].weight.fill_(0.0)
            self.net_gamma[-1].bias.fill_(0.0)

    def _build_net(self, layers, final_activation=None):
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                modules.append(nn.Tanh())
        if final_activation is not None:
            modules.append(final_activation)
        return nn.Sequential(*modules)

    def _D_self(self, net, C):
        raw = net(C)
        D_max = 0.05
        return D_max * torch.sigmoid(raw)

    def _ln_gamma(self, C):
        raw = self.net_gamma(C)
        return self.gamma_scale * torch.tanh(raw)

    def mutual_diffusion(self, C):
        if not C.requires_grad:
            C.requires_grad_(True)
        C_A = torch.clamp(C, 1e-6, 1.0 - 1e-6)
        C_B = 1.0 - C_A
        
        D_A = self._D_self(self.net_DA, C_A)
        D_B = self._D_self(self.net_DB, C_A)
        mobility = C_B * D_A + C_A * D_B
        
        ln_gamma = self._ln_gamma(C_A)
        dln_gamma_dC = torch.autograd.grad(ln_gamma, C_A, grad_outputs=torch.ones_like(ln_gamma), create_graph=True)[0]
        thermo = self.RT_Omega * dln_gamma_dC
        
        thermo = torch.clamp(thermo, -self.thermo_clip_val, self.thermo_clip_val)
        
        return mobility + thermo

    def forward(self, t, x):
        C_pred = self.net_C(torch.cat([t, x], dim=1))
        return C_pred

    def loss(self, t_data, x_data, C_data, t_pde, x_pde, t_ic, x_ic, t_bc,
             lambda_pde, lambda_ic, lambda_bc, lambda_Dbc):
        C_pred_data = self.forward(t_data, x_data)
        loss_data = torch.mean((C_pred_data - C_data) ** 2)

        t_pde.requires_grad_(True)
        x_pde.requires_grad_(True)
        C_pde = self.forward(t_pde, x_pde)
        
        dC_outputs = torch.autograd.grad(
            outputs=C_pde, 
            inputs=(t_pde, x_pde), 
            grad_outputs=torch.ones_like(C_pde), 
            create_graph=True
        )
        dC_dt, dC_dx = dC_outputs[0], dC_outputs[1]

        D_tilde = self.mutual_diffusion(C_pde)
        flux = D_tilde * dC_dx
        dflux_dx = torch.autograd.grad(flux, x_pde, grad_outputs=torch.ones_like(flux), create_graph=True)[0]
        pde_res = dC_dt - dflux_dx
        loss_pde = torch.mean(pde_res ** 2)

        C_pred_ic = self.forward(t_ic, x_ic)
        ic_mask = (x_ic <= self.L / 2).float()
        C_true_ic = self.C_left * ic_mask + self.C_right * (1.0 - ic_mask)
        loss_ic = torch.mean((C_pred_ic - C_true_ic) ** 2)

        x_bc_0 = torch.zeros_like(t_bc, requires_grad=True)
        x_bc_L = torch.full_like(t_bc, self.L, requires_grad=True)
        C_bc_0 = self.forward(t_bc, x_bc_0)
        dC_dx_bc_0 = torch.autograd.grad(C_bc_0, x_bc_0, grad_outputs=torch.ones_like(C_bc_0), create_graph=True)[0]
        C_bc_L = self.forward(t_bc, x_bc_L)
        dC_dx_bc_L = torch.autograd.grad(C_bc_L, x_bc_L, grad_outputs=torch.ones_like(C_bc_L), create_graph=True)[0]
        loss_bc = torch.mean(dC_dx_bc_0**2) + torch.mean(dC_dx_bc_L**2)

        D_A0 = self._D_self(self.net_DA, torch.zeros(1, 1, device=device))
        D_B1 = self._D_self(self.net_DB, torch.ones(1, 1, device=device))
        loss_D_bc = ((D_A0 - 0.0) ** 2 + (D_B1 - 0.0) ** 2).squeeze()

        total = loss_data + lambda_pde * loss_pde + lambda_ic * loss_ic + lambda_bc * loss_bc + lambda_Dbc * loss_D_bc
        return total, loss_data, loss_pde, loss_ic, loss_bc, loss_D_bc


def run_darken_pinn_training():
    C_left_true, C_right_true = 0.0, 1.0
    L_domain, T_domain = 1.0, 10.0
    Nx_fdm, Nt_fdm = 101, 2001

    x_fdm, t_fdm, C_fdm = solve_nonlinear_diffusion_fdm(
        C_left=C_left_true, C_right=C_right_true,
        L=L_domain, T_end=T_domain, Nx=Nx_fdm, Nt=Nt_fdm)

    print('Step 2: Preparing training data...')
    T_grid, X_grid = np.meshgrid(t_fdm, x_fdm, indexing='ij')
    mask = T_grid.flatten() > 1e-6
    t_flat, x_flat, c_flat = T_grid.flatten()[mask], X_grid.flatten()[mask], C_fdm.flatten()[mask]

    N_data, N_pde = 4000, 8000
    idx = np.random.choice(t_flat.size, N_data, replace=False)

    t_data = torch.from_numpy(t_flat[idx]).float().view(-1, 1).to(device)
    x_data = torch.from_numpy(x_flat[idx]).float().view(-1, 1).to(device)
    C_data = torch.from_numpy(c_flat[idx]).float().view(-1, 1).to(device)
    print(f'  - Training data (t, x, C) tensors moved to {t_data.device}')

    eps = 1e-4
    t_pde = eps + (T_domain - eps) * torch.rand(N_pde, 1, device=device)
    x_pde = torch.rand(N_pde, 1, device=device) * L_domain
    print(f'  - PDE collocation (t, x) tensors created on {t_pde.device}')
    
    N_ic = N_data // 2
    t_ic = torch.zeros(N_ic, 1, device=device)
    x_ic = torch.rand(N_ic, 1, device=device) * L_domain
    print(f'  - Initial Condition (t, x) tensors created on {t_ic.device}')

    N_bc = N_pde // 2
    t_bc = eps + (T_domain - eps) * torch.rand(N_bc, 1, device=device)
    print(f'  - Boundary collocation (t) tensor created on {t_bc.device}')
    print('Data preparation finished.\n')

    print('Step 3: Building PINN model...')
    layers_C = [2, 64, 64, 64, 64, 1]
    layers_DA, layers_DB, layers_gamma = [1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1]

    pinn = NonlinearDiffusionPINN(
        layers_C, layers_DA, layers_DB, layers_gamma,
        C_left=C_left_true, C_right=C_right_true, L=L_domain
    ).to(device)

    model_device = next(pinn.parameters()).device
    print(f'PINN model moved to {model_device}.')
    print(f'Using Omega = {pinn.Omega:.1f} to stabilize training.')
    print('Model building finished.\n')

    epochs = 30000
    learning_rate = 2e-4
    lambda_pde, lambda_ic, lambda_bc, lambda_Dbc = 1.0, 2.0, 0.5, 10.0

    optimizer = torch.optim.Adam(pinn.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99995)

    print('Step 4: Starting PINN training...')
    C_eval_point = torch.tensor([[0.5]], device=device, dtype=torch.float32)
    true_D_tilde_eval = true_diffusion_coefficient(0.5)

    loss_history = {'total': [], 'data': [], 'pde': [], 'ic':[], 'bc': [], 'D_bc': []}

    start = time.time()
    pbar = tqdm(range(1, epochs + 1), desc="Training Progress")
    for epoch in pbar:
        pinn.train()
        optimizer.zero_grad()

        total, loss_data, loss_pde, loss_ic, loss_bc, loss_D_bc = pinn.loss(
            t_data, x_data, C_data, t_pde, x_pde, t_ic, x_ic, t_bc,
            lambda_pde, lambda_ic, lambda_bc, lambda_Dbc)
        
        if torch.isnan(total):
            print(f"\nNaN detected at epoch {epoch}. Stopping training.")
            break

        total.backward()
        
        torch.nn.utils.clip_grad_norm_(pinn.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()

        loss_history['total'].append(total.item())
        loss_history['data'].append(loss_data.item())
        loss_history['pde'].append(loss_pde.item())
        loss_history['ic'].append(loss_ic.item())
        loss_history['bc'].append(loss_bc.item())
        loss_history['D_bc'].append(loss_D_bc.item())

        pbar.set_postfix({
            'Loss': f'{total.item():.3e}',
            'LR': f'{scheduler.get_last_lr()[0]:.3e}'
        })

        if epoch % 5000 == 0:
            pinn.eval()
            with torch.enable_grad():
                D_tilde_val = pinn.mutual_diffusion(C_eval_point).detach().item()
            print(f'\nEpoch {epoch:5d} | Total Loss: {total.item():.3e} '
                  f'| D̃(0.5)={D_tilde_val:.4f} (true≈{true_D_tilde_eval:.4f})')

    print(f'\nTraining finished in {time.time() - start:.2f} s.\n')

    print('Step 5: Visualizing results...')
    pinn.eval()

    plt.figure(figsize=(10, 6))
    if loss_history['total']:
        plt.plot(loss_history['total'], label='Total Loss')
        plt.plot(loss_history['data'], label='Data Loss', alpha=0.7)
        plt.plot(loss_history['pde'], label='PDE Loss', alpha=0.7)
        plt.plot(loss_history['ic'], label='IC Loss', alpha=0.7)
        plt.plot(loss_history['bc'], label='BC Loss', alpha=0.7)
        plt.plot(loss_history['D_bc'], label='D-BC Loss', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.title('Training Loss History')
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.savefig('/home/ubuntu/repos/machine-learning/darken_loss_history.png', dpi=150, bbox_inches='tight')
        plt.show()
    else:
        print("No loss history to plot.")

    C_plot = torch.linspace(0, 1, 200, device=device).view(-1, 1)
    with torch.enable_grad():
        D_tilde_pred = pinn.mutual_diffusion(C_plot).cpu().detach().numpy()
    plt.figure(figsize=(7, 5))
    C_plot_np = C_plot.cpu().detach().numpy()
    plt.plot(C_plot_np, true_diffusion_coefficient(C_plot_np), 'k-', lw=2, label='True D̃(C)')
    plt.plot(C_plot_np, D_tilde_pred, 'r--', lw=2, label='PINN D̃(C)')
    plt.xlabel('C (mole fraction of A)')
    plt.ylabel('Inter‑diffusion coefficient D̃')
    plt.title('Darken Model – True vs. PINN')
    plt.legend()
    plt.grid(True)
    plt.savefig('/home/ubuntu/repos/machine-learning/darken_diffusion_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    with torch.no_grad():
        D_A_pred = pinn._D_self(pinn.net_DA, C_plot).cpu().numpy()
        D_B_pred = pinn._D_self(pinn.net_DB, C_plot).cpu().numpy()
    plt.figure(figsize=(7, 5))
    plt.plot(C_plot_np, D_A_pred, 'b-', lw=2, label='PINN D_A(C)')
    plt.plot(C_plot_np, D_B_pred, 'g-', lw=2, label='PINN D_B(C)')
    plt.xlabel('C')
    plt.ylabel('Self‑diffusion coefficient')
    plt.title('Learned self‑diffusion coefficients')
    plt.legend()
    plt.grid(True)
    plt.savefig('/home/ubuntu/repos/machine-learning/darken_self_diffusion.png', dpi=150, bbox_inches='tight')
    plt.show()

    with torch.no_grad():
        ln_gamma_pred = pinn._ln_gamma(C_plot).cpu().numpy()
    plt.figure(figsize=(7, 5))
    plt.plot(C_plot_np, ln_gamma_pred, 'm-', lw=2, label='PINN lnγ(C)')
    plt.xlabel('C')
    plt.ylabel('lnγ')
    plt.title('Learned activity‑coefficient (log)')
    plt.legend()
    plt.grid(True)
    plt.savefig('/home/ubuntu/repos/machine-learning/darken_activity_coefficient.png', dpi=150, bbox_inches='tight')
    plt.show()

    T_grid_plot, X_grid_plot = np.meshgrid(t_fdm, x_fdm)
    t_grid_tensor = torch.from_numpy(T_grid_plot.flatten()).float().view(-1, 1).to(device)
    x_grid_tensor = torch.from_numpy(X_grid_plot.flatten()).float().view(-1, 1).to(device)
    with torch.no_grad():
        C_pred_flat = pinn.forward(t_grid_tensor, x_grid_tensor)
    C_pred = C_pred_flat.cpu().numpy().reshape(X_grid_plot.shape)
    C_fdm_T = C_fdm.T
    err = np.abs(C_fdm_T - C_pred)

    fig, axs = plt.subplots(1, 3, figsize=(22, 6), sharey=True)
    vmin, vmax = C_fdm.min(), C_fdm.max()
    im0 = axs[0].pcolormesh(t_fdm, x_fdm, C_fdm_T, cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)
    axs[0].set_title('True (FDM)')
    axs[0].set_xlabel('t')
    axs[0].set_ylabel('x')
    fig.colorbar(im0, ax=axs[0])
    im1 = axs[1].pcolormesh(t_fdm, x_fdm, C_pred, cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)
    axs[1].set_title('PINN')
    axs[1].set_xlabel('t')
    fig.colorbar(im1, ax=axs[1])
    im2 = axs[2].pcolormesh(t_fdm, x_fdm, err, cmap='coolwarm', shading='auto')
    axs[2].set_title('Absolute error')
    axs[2].set_xlabel('t')
    fig.colorbar(im2, ax=axs[2])
    plt.tight_layout()
    plt.savefig('/home/ubuntu/repos/machine-learning/darken_concentration_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()

    times_to_plot = [0.0, 5.0, 9.9]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for i, t_target in enumerate(times_to_plot):
        t_idx = np.argmin(np.abs(t_fdm - t_target))
        t_slice_tensor = torch.full((len(x_fdm), 1), t_fdm[t_idx], device=device, dtype=torch.float32)
        x_slice_tensor = torch.from_numpy(x_fdm).view(-1, 1).float().to(device)
        with torch.no_grad():
            C_pred_slice = pinn.forward(t_slice_tensor, x_slice_tensor).cpu().numpy()
        axes[i].plot(x_fdm, C_fdm[t_idx, :], 'k-', lw=2, label='FDM (true)')
        axes[i].plot(x_fdm, C_pred_slice, 'r--', lw=2, label='PINN')
        axes[i].set_title(f't = {t_fdm[t_idx]:.2f}')
        axes[i].set_xlabel('x')
        axes[i].grid(True)
        axes[i].legend()
    axes[0].set_ylabel('Concentration C')
    plt.tight_layout()
    plt.savefig('/home/ubuntu/repos/machine-learning/darken_concentration_profiles.png', dpi=150, bbox_inches='tight')
    plt.show()

    print('Visualization finished.')
    
    return {
        'pinn': pinn,
        'loss_history': loss_history,
        'x_fdm': x_fdm,
        't_fdm': t_fdm,
        'C_fdm': C_fdm,
        'final_loss': loss_history['total'][-1] if loss_history['total'] else float('inf')
    }


if __name__ == '__main__':
    results = run_darken_pinn_training()
    print(f"Training completed with final loss: {results['final_loss']:.3e}")
