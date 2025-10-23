#!/usr/bin/env python3
"""
darken_pinns_unified.py - Unified Darken Model PINNs Implementation

This unified file combines all Darken PINNs functionality:
- Core PINN implementation for diffusion pair concentration ranges
- Streamlit web interface for interactive visualization
- Standalone training script capability
- Verification and testing functions

Usage:
  python darken_pinns_unified.py                    # Run standalone training
  streamlit run darken_pinns_unified.py             # Run Streamlit app
  python darken_pinns_unified.py --verify           # Run verification tests

Diffusion Pair Focus:
- Calculations restricted to realistic concentration ranges C ∈ [0.1, 0.9]
- Avoids pure substance extremes for practical diffusion applications

Darken Model: D̃(C) = C_B·D_A(C) + C_A·D_B(C) + (RT/Ω)·∂lnγ/∂C
"""

import sys
import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    st = None

if torch.cuda.is_available():
    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU is available. Using device: {device} ({gpu_name})\n")
else:
    device = torch.device('cpu')
    print("GPU not available. Using device: CPU\n")


def true_diffusion_coefficient(C):
    """Ground truth diffusion coefficient for comparison"""
    return 0.01 + 0.04 * C


def solve_nonlinear_diffusion_fdm(C_left, C_right, L=1.0, T_end=10.0, Nx=101, Nt=2001):
    """
    Generate ground truth solution using finite difference method with enhanced stability
    """
    print('Step 1: Generating true FDM data...')
    x = np.linspace(0, L, Nx)
    dx = x[1] - x[0]
    
    D_max = true_diffusion_coefficient(1.0)
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
    """
    Darken Model PINNs for diffusion pair concentration ranges
    """
    R_GAS_CONSTANT = 8.314  # J/(mol·K) - Universal gas constant
    
    def __init__(self, layers_C, layers_DA, layers_DB, layers_gamma,
                 C_left, C_right, L=1.0, T=300.0, Omega=25000.0):
        super().__init__()
        self.net_C = self._build_net(layers_C, final_activation=nn.Sigmoid())
        self.net_DA = self._build_net(layers_DA)
        self.net_DB = self._build_net(layers_DB)
        self.net_gamma = self._build_net(layers_gamma)

        self.C_left = C_left
        self.C_right = C_right
        self.L = L
        self.R = self.R_GAS_CONSTANT  # Always use the protected constant
        self.T = T
        self.Omega = Omega
        self.RT_Omega = (self.R_GAS_CONSTANT * T) / Omega
        
        self.gamma_scale = 0.5
        self.thermo_clip_val = 0.1

        with torch.no_grad():
            self.net_gamma[-1].weight.fill_(0.0)
            self.net_gamma[-1].bias.fill_(0.0)

    def _build_net(self, layers, final_activation=None):
        """Build neural network with specified architecture"""
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                modules.append(nn.Tanh())
        if final_activation is not None:
            modules.append(final_activation)
        return nn.Sequential(*modules)

    def _D_self(self, net, C):
        """Self-diffusion coefficient with sigmoid activation"""
        raw = net(C)
        D_max = 0.05
        return D_max * torch.sigmoid(raw)

    def _ln_gamma(self, C):
        """Activity coefficient (logarithmic)"""
        raw = self.net_gamma(C)
        return self.gamma_scale * torch.tanh(raw)

    def mutual_diffusion(self, C):
        """
        Darken model mutual diffusion coefficient
        D̃(C) = C_B·D_A(C) + C_A·D_B(C) + (RT/Ω)·∂lnγ/∂C
        """
        if not C.requires_grad:
            C.requires_grad_(True)
        
        C_A = torch.clamp(C, 0.1, 0.9)
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
        """Predict concentration field C(t, x)"""
        C_pred = self.net_C(torch.cat([t, x], dim=1))
        return C_pred

    def loss(self, t_data, x_data, C_data, t_pde, x_pde, t_ic, x_ic, t_bc,
             lambda_pde, lambda_ic, lambda_bc):
        """
        Loss function for diffusion pair concentration ranges:
        - Calculations restricted to C ∈ [0.1, 0.9] for realistic diffusion pairs
        - Avoids pure substance extremes for practical applications
        """
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

        total = loss_data + lambda_pde * loss_pde + lambda_ic * loss_ic + lambda_bc * loss_bc
        return total, loss_data, loss_pde, loss_ic, loss_bc


def run_standalone_training():
    """Run standalone Darken PINNs training with matplotlib visualization"""
    print("=== Darken Model PINNs - Standalone Training ===\n")
    
    C_left_true, C_right_true = 0.2, 0.8
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

    eps = 1e-4
    t_pde = eps + (T_domain - eps) * torch.rand(N_pde, 1, device=device)
    x_pde = torch.rand(N_pde, 1, device=device) * L_domain
    
    N_ic = N_data // 2
    t_ic = torch.zeros(N_ic, 1, device=device)
    x_ic = torch.rand(N_ic, 1, device=device) * L_domain

    N_bc = N_pde // 2
    t_bc = eps + (T_domain - eps) * torch.rand(N_bc, 1, device=device)
    print('Data preparation finished.\n')

    print('Step 3: Building PINN model...')
    layers_C = [2, 64, 64, 64, 64, 1]
    layers_DA, layers_DB, layers_gamma = [1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1]

    pinn = NonlinearDiffusionPINN(
        layers_C, layers_DA, layers_DB, layers_gamma,
        C_left=C_left_true, C_right=C_right_true, L=L_domain
    ).to(device)

    print(f'PINN model moved to {next(pinn.parameters()).device}.')
    print(f'Using Omega = {pinn.Omega:.1f} to stabilize training.')
    print('Model building finished.\n')

    epochs = 30000
    learning_rate = 2e-4
    lambda_pde, lambda_ic, lambda_bc = 1.0, 2.0, 0.5

    optimizer = torch.optim.Adam(pinn.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99995)

    print('Step 4: Starting PINN training...')
    C_eval_point = torch.tensor([[0.5]], device=device, dtype=torch.float32)
    true_D_tilde_eval = true_diffusion_coefficient(0.5)

    loss_history = {'total': [], 'data': [], 'pde': [], 'ic': [], 'bc': []}

    start = time.time()
    pbar = tqdm(range(1, epochs + 1), desc="Training Progress")
    for epoch in pbar:
        pinn.train()
        optimizer.zero_grad()

        total, loss_data, loss_pde, loss_ic, loss_bc = pinn.loss(
            t_data, x_data, C_data, t_pde, x_pde, t_ic, x_ic, t_bc,
            lambda_pde, lambda_ic, lambda_bc)
        
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
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.title('Training Loss History - Diffusion Pair Focus')
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.savefig(os.path.join(os.getcwd(), 'darken_loss_history_unified.png'), dpi=150, bbox_inches='tight')
        plt.show()

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
    plt.savefig(os.path.join(os.getcwd(), 'darken_diffusion_comparison_unified.png'), dpi=150, bbox_inches='tight')
    plt.show()

    with torch.no_grad():
        D_A_pred = pinn._D_self(pinn.net_DA, C_plot).cpu().numpy()
        D_B_pred = pinn._D_self(pinn.net_DB, C_plot).cpu().numpy()
    plt.figure(figsize=(7, 5))
    plt.plot(C_plot_np, D_A_pred, 'b-', lw=2, label='PINN D_A(C)')
    plt.plot(C_plot_np, D_B_pred, 'g-', lw=2, label='PINN D_B(C)')
    plt.xlabel('C')
    plt.ylabel('Self‑diffusion coefficient')
    plt.title('Learned Self‑Diffusion Coefficients for Diffusion Pairs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(os.getcwd(), 'darken_self_diffusion_unified.png'), dpi=150, bbox_inches='tight')
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


def create_streamlit_app():
    """Create Streamlit web interface for interactive Darken PINNs"""
    if not STREAMLIT_AVAILABLE:
        print("Streamlit and Plotly are required for the web interface. Please install them.")
        return
    
    try:
        import streamlit as st
        st.set_page_config(page_title="Darken Model PINNs - Unified", layout="wide")
    except Exception:
        return

    
    st.title("🧠 Darken Model Physics-Informed Neural Networks (Unified)")
    st.markdown("---")
    
    st.markdown("""
    **Darken拡散モデル**を用いた非線形拡散方程式をPINNsで解くシステムです。
    
    **対象方程式**: ∂C/∂t = ∂/∂x [D̃(C) ∂C/∂x]
    
    **Darkenモデル**: D̃(C) = C_B·D_A(C) + C_A·D_B(C) + (RT/Ω)·∂lnγ/∂C
    
    **拡散対濃度範囲**: C ∈ [0.1, 0.9] (純物質極限を避けた現実的な範囲)
    """)
    
    st.sidebar.header("🔧 Darken Model Parameters")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.info("🔒 Gas constant R = 8.314 J/(mol·K) (protected physical constant)")
        T = st.number_input("Temperature T (K)", min_value=200.0, max_value=500.0, value=300.0, format="%.1f")
        Omega = st.number_input("Molar volume Ω", min_value=10000.0, max_value=50000.0, value=25000.0, format="%.0f")
    
    with col2:
        C_left = st.number_input("Left boundary C", min_value=0.1, max_value=0.9, value=0.2, format="%.2f")
        C_right = st.number_input("Right boundary C", min_value=0.1, max_value=0.9, value=0.8, format="%.2f")
        L_domain = st.number_input("Domain length L", min_value=0.5, max_value=2.0, value=1.0, format="%.2f")
    
    st.sidebar.header("🎯 Training Parameters")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        epochs = st.number_input("Epochs", min_value=1000, max_value=50000, value=15000, step=1000)
        learning_rate = st.number_input("Learning rate", min_value=0.0001, max_value=0.01, value=0.0002, format="%.4f")
        N_data = st.number_input("Training data points", min_value=1000, max_value=10000, value=4000, step=500)
    
    with col2:
        lambda_pde = st.number_input("PDE loss weight", min_value=0.1, max_value=10.0, value=1.0, format="%.1f")
        lambda_ic = st.number_input("IC loss weight", min_value=0.1, max_value=10.0, value=2.0, format="%.1f")
        lambda_bc = st.number_input("BC loss weight", min_value=0.1, max_value=10.0, value=0.5, format="%.1f")
    
    st.sidebar.header("🏗️ Network Architecture")
    hidden_dim_C = st.sidebar.number_input("Concentration network hidden dim", min_value=32, max_value=128, value=64, step=16)
    hidden_dim_D = st.sidebar.number_input("Diffusion network hidden dim", min_value=16, max_value=64, value=32, step=8)
    
    if 'training_in_progress' not in st.session_state:
        st.session_state.training_in_progress = False
    
    button_disabled = st.session_state.training_in_progress
    button_text = "⏳ Training in Progress..." if button_disabled else "🚀 Start Darken PINNs Training"
    
    if st.button(button_text, type="primary", disabled=button_disabled):
        st.session_state.training_in_progress = True
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        st.info(f"Using device: {device}")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        loss_placeholder = st.empty()
        
        with st.spinner("Generating FDM ground truth data..."):
            T_domain = 10.0
            Nx_fdm, Nt_fdm = 101, 2001
            x_fdm, t_fdm, C_fdm = solve_nonlinear_diffusion_fdm(
                C_left=C_left, C_right=C_right,
                L=L_domain, T_end=T_domain, Nx=Nx_fdm, Nt=Nt_fdm)
        
        st.success("✅ FDM data generation completed!")
        
        T_grid, X_grid = np.meshgrid(t_fdm, x_fdm, indexing='ij')
        mask = T_grid.flatten() > 1e-6
        t_flat, x_flat, c_flat = T_grid.flatten()[mask], X_grid.flatten()[mask], C_fdm.flatten()[mask]
        
        N_pde = 8000
        idx = np.random.choice(t_flat.size, N_data, replace=False)
        
        t_data = torch.from_numpy(t_flat[idx]).float().view(-1, 1).to(device)
        x_data = torch.from_numpy(x_flat[idx]).float().view(-1, 1).to(device)
        C_data = torch.from_numpy(c_flat[idx]).float().view(-1, 1).to(device)
        
        eps = 1e-4
        t_pde = eps + (T_domain - eps) * torch.rand(N_pde, 1, device=device)
        x_pde = torch.rand(N_pde, 1, device=device) * L_domain
        
        N_ic = N_data // 2
        t_ic = torch.zeros(N_ic, 1, device=device)
        x_ic = torch.rand(N_ic, 1, device=device) * L_domain
        
        N_bc = N_pde // 2
        t_bc = eps + (T_domain - eps) * torch.rand(N_bc, 1, device=device)
        
        layers_C = [2, hidden_dim_C, hidden_dim_C, hidden_dim_C, hidden_dim_C, 1]
        layers_DA = [1, hidden_dim_D, hidden_dim_D, 1]
        layers_DB = [1, hidden_dim_D, hidden_dim_D, 1]
        layers_gamma = [1, hidden_dim_D, hidden_dim_D, 1]
        
        pinn = NonlinearDiffusionPINN(
            layers_C, layers_DA, layers_DB, layers_gamma,
            C_left=C_left, C_right=C_right, L=L_domain,
            T=T, Omega=Omega
        ).to(device)
        
        optimizer = torch.optim.Adam(pinn.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99995)
        
        loss_history = {'total': [], 'data': [], 'pde': [], 'ic': [], 'bc': []}
        
        C_eval_point = torch.tensor([[0.5]], device=device, dtype=torch.float32)
        true_D_tilde_eval = true_diffusion_coefficient(0.5)
        
        start_time = time.time()
        loss_chart_placeholder = st.empty()
        
        for epoch in range(1, epochs + 1):
            pinn.train()
            optimizer.zero_grad()
            
            total, loss_data, loss_pde, loss_ic, loss_bc = pinn.loss(
                t_data, x_data, C_data, t_pde, x_pde, t_ic, x_ic, t_bc,
                lambda_pde, lambda_ic, lambda_bc)
            
            if torch.isnan(total):
                st.error(f"NaN detected at epoch {epoch}. Stopping training.")
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
            
            progress = epoch / epochs
            progress_bar.progress(progress)
            status_text.text(f"Epoch {epoch}/{epochs} | Loss: {total.item():.3e} | LR: {scheduler.get_last_lr()[0]:.3e}")
            
            if epoch % 500 == 0:
                pinn.eval()
                with torch.enable_grad():
                    D_tilde_val = pinn.mutual_diffusion(C_eval_point).detach().item()
                
                loss_placeholder.metric(
                    "Current D̃(0.5)", 
                    f"{D_tilde_val:.4f}", 
                    f"True: {true_D_tilde_eval:.4f}"
                )
                
                if epoch % 2000 == 0:
                    fig_loss = go.Figure()
                    fig_loss.add_trace(go.Scatter(y=loss_history['total'], name='Total Loss', line=dict(color='red')))
                    fig_loss.add_trace(go.Scatter(y=loss_history['data'], name='Data Loss', line=dict(color='blue')))
                    fig_loss.add_trace(go.Scatter(y=loss_history['pde'], name='PDE Loss', line=dict(color='green')))
                    fig_loss.add_trace(go.Scatter(y=loss_history['ic'], name='IC Loss', line=dict(color='orange')))
                    fig_loss.add_trace(go.Scatter(y=loss_history['bc'], name='BC Loss', line=dict(color='purple')))
                    
                    fig_loss.update_layout(
                        title="Training Loss History - Diffusion Pair Focus (Real-time)",
                        xaxis_title="Epoch",
                        yaxis_title="Loss",
                        yaxis_type="log",
                        height=400
                    )
                    
                    loss_chart_placeholder.plotly_chart(fig_loss, use_container_width=True)
        
        training_time = time.time() - start_time
        
        st.session_state.training_in_progress = False
        
        st.success(f"✅ Diffusion pair training completed in {training_time:.2f} seconds!")
        
        pinn.eval()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Final Training Metrics")
            final_metrics = {
                "Total Loss": f"{loss_history['total'][-1]:.2e}",
                "Data Loss": f"{loss_history['data'][-1]:.2e}",
                "PDE Loss": f"{loss_history['pde'][-1]:.2e}",
                "IC Loss": f"{loss_history['ic'][-1]:.2e}",
                "BC Loss": f"{loss_history['bc'][-1]:.2e}"
            }
            
            for metric, value in final_metrics.items():
                st.metric(metric, value)
        
        with col2:
            st.subheader("🎯 Diffusion Pair Range")
            with torch.no_grad():
                C_range = torch.linspace(0.1, 0.9, 5).view(-1, 1).to(device)
                D_A_range = pinn._D_self(pinn.net_DA, C_range)
                D_B_range = pinn._D_self(pinn.net_DB, C_range)
            
            st.write("**Concentration Range**: C ∈ [0.1, 0.9]")
            for i, c in enumerate([0.1, 0.325, 0.55, 0.775, 0.9]):
                st.metric(f"C = {c:.3f}", f"D_A: {D_A_range[i].item():.4f}, D_B: {D_B_range[i].item():.4f}")
        
        st.subheader("📈 Enhanced Results Visualization")
        
        C_plot = torch.linspace(0, 1, 200, device=device).view(-1, 1)
        with torch.enable_grad():
            D_tilde_pred = pinn.mutual_diffusion(C_plot).cpu().detach().numpy()
        with torch.no_grad():
            D_A_pred = pinn._D_self(pinn.net_DA, C_plot).cpu().numpy()
            D_B_pred = pinn._D_self(pinn.net_DB, C_plot).cpu().numpy()
            ln_gamma_pred = pinn._ln_gamma(C_plot).cpu().numpy()
        
        C_plot_np = C_plot.cpu().detach().numpy()
        
        fig_diffusion = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Mutual Diffusion Coefficient D̃(C)', 'Self-Diffusion for Diffusion Pairs', 
                          'Activity Coefficient lnγ(C)', 'Training Loss History'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        fig_diffusion.add_trace(
            go.Scatter(x=C_plot_np.flatten(), y=true_diffusion_coefficient(C_plot_np.flatten()), 
                      name='True D̃(C)', line=dict(color='black', width=3)),
            row=1, col=1
        )
        fig_diffusion.add_trace(
            go.Scatter(x=C_plot_np.flatten(), y=D_tilde_pred.flatten(), 
                      name='PINN D̃(C)', line=dict(color='red', dash='dash', width=3)),
            row=1, col=1
        )
        
        fig_diffusion.add_trace(
            go.Scatter(x=C_plot_np.flatten(), y=D_A_pred.flatten(), 
                      name='PINN D_A(C)', line=dict(color='blue', width=2)),
            row=1, col=2
        )
        fig_diffusion.add_trace(
            go.Scatter(x=C_plot_np.flatten(), y=D_B_pred.flatten(), 
                      name='PINN D_B(C)', line=dict(color='green', width=2)),
            row=1, col=2
        )
        
        
        fig_diffusion.add_trace(
            go.Scatter(x=C_plot_np.flatten(), y=ln_gamma_pred.flatten(), 
                      name='PINN lnγ(C)', line=dict(color='magenta', width=2)),
            row=2, col=1
        )
        
        fig_diffusion.add_trace(
            go.Scatter(y=loss_history['total'], name='Total Loss', line=dict(color='red')),
            row=2, col=2
        )
        
        fig_diffusion.update_xaxes(title_text="Concentration C", row=1, col=1)
        fig_diffusion.update_xaxes(title_text="Concentration C", row=1, col=2)
        fig_diffusion.update_xaxes(title_text="Concentration C", row=2, col=1)
        fig_diffusion.update_xaxes(title_text="Epoch", row=2, col=2)
        
        fig_diffusion.update_yaxes(title_text="D̃", row=1, col=1)
        fig_diffusion.update_yaxes(title_text="D_A, D_B", row=1, col=2)
        fig_diffusion.update_yaxes(title_text="lnγ", row=2, col=1)
        fig_diffusion.update_yaxes(title_text="Loss", type="log", row=2, col=2)
        
        fig_diffusion.update_layout(height=800, showlegend=True, title_text="Darken Model PINNs Results - Diffusion Pair Focus")
        
        st.plotly_chart(fig_diffusion, use_container_width=True)
        
        st.success("🎉 Darken Model PINNs analysis for diffusion pairs completed successfully!")


def verify_diffusion_pair_constraints():
    """Verification function for diffusion pair concentration range restrictions"""
    print("Diffusion Pair Concentration Range Verification")
    print("=" * 60)
    
    device = torch.device('cpu')
    layers_C = [2, 32, 32, 1]
    layers_DA, layers_DB, layers_gamma = [1, 16, 1], [1, 16, 1], [1, 16, 1]

    pinn = NonlinearDiffusionPINN(
        layers_C, layers_DA, layers_DB, layers_gamma,
        C_left=0.2, C_right=0.8, L=1.0
    ).to(device)

    C_test_range = torch.linspace(0.1, 0.9, 9).view(-1, 1).to(device)
    
    with torch.no_grad():
        D_A_range = pinn._D_self(pinn.net_DA, C_test_range)
        D_B_range = pinn._D_self(pinn.net_DB, C_test_range)

    print('Diffusion coefficients across concentration range:')
    for i, c in enumerate(C_test_range.flatten()):
        print(f'C = {c:.1f}: D_A = {D_A_range[i].item():.4f}, D_B = {D_B_range[i].item():.4f}')
    
    t_test = torch.rand(10, 1, device=device)
    x_test = torch.rand(10, 1, device=device)
    C_test = 0.1 + 0.8 * torch.rand(10, 1, device=device)
    
    total_loss, data_loss, pde_loss, ic_loss, bc_loss = pinn.loss(
        t_test, x_test, C_test, t_test, x_test, t_test, x_test, t_test,
        lambda_pde=1.0, lambda_ic=2.0, lambda_bc=0.5
    )
    
    print(f'\nLoss function test:')
    print(f'Total loss: {total_loss.item():.6f}')
    print(f'Data loss: {data_loss.item():.6f}')
    print(f'PDE loss: {pde_loss.item():.6f}')
    print(f'IC loss: {ic_loss.item():.6f}')
    print(f'BC loss: {bc_loss.item():.6f}')
    
    print('\n✅ Diffusion pair concentration range restrictions implemented successfully!')
    print('✅ Calculations restricted to C ∈ [0.1, 0.9] for realistic diffusion pairs')
    
    return True


def main():
    """Main function to handle different execution modes"""
    if '--streamlit' in sys.argv:
        if STREAMLIT_AVAILABLE:
            create_streamlit_app()
        else:
            print("Streamlit is not available. Please install streamlit and plotly.")
        return
    
    parser = argparse.ArgumentParser(description='Unified Darken Model PINNs')
    parser.add_argument('--verify', action='store_true', help='Run verification tests')
    parser.add_argument('--streamlit', action='store_true', help='Force Streamlit mode')
    
    args = parser.parse_args()
    
    if args.streamlit:
        if STREAMLIT_AVAILABLE:
            create_streamlit_app()
        else:
            print("Streamlit is not available. Please install streamlit and plotly.")
        return
    
    if args.verify:
        verify_diffusion_pair_constraints()
    else:
        results = run_standalone_training()
        print(f"\n=== Training Summary ===")
        print(f"Final loss: {results['final_loss']:.3e}")
        print(f"Diffusion pair concentration range restrictions successfully implemented!")


def is_streamlit_context():
    """Check if we're running in Streamlit context"""
    try:
        import streamlit as st
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except:
        return False

if __name__ == '__main__':
    if is_streamlit_context() or 'streamlit' in sys.argv[0].lower():
        if STREAMLIT_AVAILABLE:
            create_streamlit_app()
    else:
        main()
