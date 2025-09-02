#!/usr/bin/env python3
"""
Streamlit app for Darken model PINNs visualization
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from pinn_darken import NonlinearDiffusionPINN, solve_nonlinear_diffusion_fdm, true_diffusion_coefficient

st.set_page_config(page_title="Darken Model PINNs", layout="wide")

def create_darken_pinns_app():
    st.title("🧠 Darken Model Physics-Informed Neural Networks")
    st.markdown("---")
    
    st.markdown("""
    **Darken拡散モデル**を用いた非線形拡散方程式をPINNsで解くシステムです。
    
    **対象方程式**: ∂C/∂t = ∂/∂x [D̃(C) ∂C/∂x]
    
    **Darkenモデル**: D̃(C) = C_B·D_A(C) + C_A·D_B(C) + (RT/Ω)·∂lnγ/∂C
    """)
    
    st.sidebar.header("🔧 Darken Model Parameters")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        R = st.number_input("Gas constant R", min_value=1.0, max_value=20.0, value=8.314, format="%.3f")
        T = st.number_input("Temperature T (K)", min_value=200.0, max_value=500.0, value=300.0, format="%.1f")
        Omega = st.number_input("Molar volume Ω", min_value=10000.0, max_value=50000.0, value=25000.0, format="%.0f")
    
    with col2:
        C_left = st.number_input("Left boundary C", min_value=0.0, max_value=1.0, value=0.0, format="%.2f")
        C_right = st.number_input("Right boundary C", min_value=0.0, max_value=1.0, value=1.0, format="%.2f")
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
        lambda_Dbc = st.number_input("純物質効果重み λ_Dbc", min_value=0.1, max_value=50.0, value=20.0, step=0.5, 
                                    help="D_A(0)=0, D_B(1)=0, D_A(1)=0.05, D_B(0)=0.05の純物質境界条件の重み")
    
    st.sidebar.header("🏗️ Network Architecture")
    hidden_dim_C = st.sidebar.number_input("Concentration network hidden dim", min_value=32, max_value=128, value=64, step=16)
    hidden_dim_D = st.sidebar.number_input("Diffusion network hidden dim", min_value=16, max_value=64, value=32, step=8)
    
    if st.button("🚀 Start Darken PINNs Training", type="primary"):
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
            R=R, T=T, Omega=Omega
        ).to(device)
        
        optimizer = torch.optim.Adam(pinn.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99995)
        
        loss_history = {'total': [], 'data': [], 'pde': [], 'ic': [], 'bc': [], 'D_bc': []}
        
        C_eval_point = torch.tensor([[0.5]], device=device, dtype=torch.float32)
        true_D_tilde_eval = true_diffusion_coefficient(0.5)
        
        start_time = time.time()
        
        loss_chart_placeholder = st.empty()
        
        for epoch in range(1, epochs + 1):
            pinn.train()
            optimizer.zero_grad()
            
            total, loss_data, loss_pde, loss_ic, loss_bc, loss_D_bc = pinn.loss(
                t_data, x_data, C_data, t_pde, x_pde, t_ic, x_ic, t_bc,
                lambda_pde, lambda_ic, lambda_bc, lambda_Dbc)
            
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
            loss_history['D_bc'].append(loss_D_bc.item())
            
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
                    fig_loss.add_trace(go.Scatter(y=loss_history['D_bc'], name='D-BC Loss', line=dict(color='brown')))
                    
                    fig_loss.update_layout(
                        title="Training Loss History (Real-time)",
                        xaxis_title="Epoch",
                        yaxis_title="Loss",
                        yaxis_type="log",
                        height=400
                    )
                    
                    loss_chart_placeholder.plotly_chart(fig_loss, use_container_width=True)
        
        training_time = time.time() - start_time
        st.success(f"✅ Training completed in {training_time:.2f} seconds!")
        
        pinn.eval()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Final Training Metrics")
            final_metrics = {
                "Total Loss": f"{loss_history['total'][-1]:.2e}",
                "Data Loss": f"{loss_history['data'][-1]:.2e}",
                "PDE Loss": f"{loss_history['pde'][-1]:.2e}",
                "IC Loss": f"{loss_history['ic'][-1]:.2e}",
                "BC Loss": f"{loss_history['bc'][-1]:.2e}",
                "D-BC Loss": f"{loss_history['D_bc'][-1]:.2e}"
            }
            
            for metric, value in final_metrics.items():
                st.metric(metric, value)
        
        with col2:
            st.subheader("🎯 Learned Parameters")
            with torch.enable_grad():
                D_tilde_final = pinn.mutual_diffusion(C_eval_point).detach().item()
            
            st.metric("D̃(0.5) Learned", f"{D_tilde_final:.4f}")
            st.metric("D̃(0.5) True", f"{true_D_tilde_eval:.4f}")
            error_percent = abs(D_tilde_final - true_D_tilde_eval) / true_D_tilde_eval * 100
            st.metric("Relative Error", f"{error_percent:.2f}%")
        
        st.subheader("📈 Comprehensive Results Visualization")
        
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
            subplot_titles=('Mutual Diffusion Coefficient D̃(C)', 'Self-Diffusion Coefficients', 
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
        
        fig_diffusion.update_layout(height=800, showlegend=True, title_text="Darken Model PINNs Results")
        
        st.plotly_chart(fig_diffusion, use_container_width=True)
        
        T_grid_plot, X_grid_plot = np.meshgrid(t_fdm, x_fdm)
        t_grid_tensor = torch.from_numpy(T_grid_plot.flatten()).float().view(-1, 1).to(device)
        x_grid_tensor = torch.from_numpy(X_grid_plot.flatten()).float().view(-1, 1).to(device)
        with torch.no_grad():
            C_pred_flat = pinn.forward(t_grid_tensor, x_grid_tensor)
        C_pred = C_pred_flat.cpu().numpy().reshape(X_grid_plot.shape)
        C_fdm_T = C_fdm.T
        err = np.abs(C_fdm_T - C_pred)
        
        fig_heatmap = make_subplots(
            rows=1, cols=3,
            subplot_titles=('True (FDM)', 'PINN Prediction', 'Absolute Error'),
            specs=[[{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}]]
        )
        
        fig_heatmap.add_trace(
            go.Heatmap(z=C_fdm_T, x=t_fdm, y=x_fdm, colorscale='Viridis', name='True'),
            row=1, col=1
        )
        fig_heatmap.add_trace(
            go.Heatmap(z=C_pred, x=t_fdm, y=x_fdm, colorscale='Viridis', name='PINN'),
            row=1, col=2
        )
        fig_heatmap.add_trace(
            go.Heatmap(z=err, x=t_fdm, y=x_fdm, colorscale='RdBu', name='Error'),
            row=1, col=3
        )
        
        fig_heatmap.update_layout(height=500, title_text="Concentration Field C(t,x)")
        fig_heatmap.update_xaxes(title_text="Time t")
        fig_heatmap.update_yaxes(title_text="Space x")
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        st.subheader("📊 Concentration Profiles at Different Times")
        times_to_plot = [0.0, 5.0, 9.9]
        
        fig_profiles = go.Figure()
        
        for t_target in times_to_plot:
            t_idx = np.argmin(np.abs(t_fdm - t_target))
            t_slice_tensor = torch.full((len(x_fdm), 1), t_fdm[t_idx], device=device, dtype=torch.float32)
            x_slice_tensor = torch.from_numpy(x_fdm).view(-1, 1).float().to(device)
            with torch.no_grad():
                C_pred_slice = pinn.forward(t_slice_tensor, x_slice_tensor).cpu().numpy()
            
            fig_profiles.add_trace(go.Scatter(
                x=x_fdm, y=C_fdm[t_idx, :], 
                name=f'FDM t={t_fdm[t_idx]:.1f}', 
                line=dict(width=3)
            ))
            fig_profiles.add_trace(go.Scatter(
                x=x_fdm, y=C_pred_slice.flatten(), 
                name=f'PINN t={t_fdm[t_idx]:.1f}', 
                line=dict(dash='dash', width=2)
            ))
        
        fig_profiles.update_layout(
            title="Concentration Profiles at Different Times",
            xaxis_title="Space x",
            yaxis_title="Concentration C",
            height=400
        )
        
        st.plotly_chart(fig_profiles, use_container_width=True)
        
        st.success("🎉 Darken Model PINNs analysis completed successfully!")


if __name__ == "__main__":
    create_darken_pinns_app()
