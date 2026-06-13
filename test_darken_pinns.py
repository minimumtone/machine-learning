#!/usr/bin/env python3
"""
Test script for Darken model PINNs system
"""

import torch
import numpy as np
from pinn_darken import NonlinearDiffusionPINN, solve_nonlinear_diffusion_fdm, true_diffusion_coefficient

def test_darken_fdm_generation():
    """Test FDM data generation for Darken model"""
    print('Testing FDM data generation...')
    
    x, t, C = solve_nonlinear_diffusion_fdm(
        C_left=0.0, C_right=1.0, L=1.0, T_end=5.0, Nx=51, Nt=101
    )
    
    assert x.shape[0] == 51, f"Expected 51 spatial points, got {x.shape[0]}"
    assert t.shape[0] == C.shape[0], "Time and concentration arrays should have consistent time dimension"
    assert C.shape[1] == 51, f"Expected 51 spatial points in C, got {C.shape[1]}"
    
    assert np.all(C >= 0) and np.all(C <= 1), "Concentration values should be between 0 and 1"
    
    left_indices = x <= 0.5
    right_indices = x > 0.5
    
    left_values = C[0, left_indices]
    right_values = C[0, right_indices]
    
    assert np.allclose(left_values, 0.0, atol=1e-6), f"Left half should start at 0, got range [{left_values.min():.6f}, {left_values.max():.6f}]"
    assert np.allclose(right_values, 1.0, atol=1e-6), f"Right half should start at 1, got range [{right_values.min():.6f}, {right_values.max():.6f}]"
    
    print(f'  ✅ FDM data shape: {C.shape}')
    print(f'  ✅ Time steps adjusted for stability: Nt = {t.shape[0]}')
    print(f'  ✅ Concentration range: [{C.min():.4f}, {C.max():.4f}]')
    print('  ✅ Initial condition verified')
    
    return True

def test_darken_pinn_initialization():
    """Test Darken PINN model initialization"""
    print('\nTesting Darken PINN initialization...')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'  Using device: {device}')
    
    layers_C = [2, 32, 32, 1]
    layers_DA = [1, 16, 16, 1]
    layers_DB = [1, 16, 16, 1]
    layers_gamma = [1, 16, 16, 1]
    
    pinn = NonlinearDiffusionPINN(
        layers_C, layers_DA, layers_DB, layers_gamma,
        C_left=0.0, C_right=1.0, L=1.0
    ).to(device)
    
    t_test = torch.rand(10, 1, device=device)
    x_test = torch.rand(10, 1, device=device)
    
    C_pred = pinn.forward(t_test, x_test)
    assert C_pred.shape == (10, 1), f"Expected C prediction shape (10, 1), got {C_pred.shape}"
    assert torch.all(C_pred >= 0) and torch.all(C_pred <= 1), "Concentration predictions should be between 0 and 1"
    
    C_test = torch.rand(10, 1, device=device)
    D_tilde = pinn.mutual_diffusion(C_test)
    assert D_tilde.shape == (10, 1), f"Expected D_tilde shape (10, 1), got {D_tilde.shape}"
    assert torch.all(D_tilde > 0), "Mutual diffusion coefficient should be positive"
    
    D_A0 = pinn._D_self(pinn.net_DA, torch.zeros(1, 1, device=device))
    D_B1 = pinn._D_self(pinn.net_DB, torch.ones(1, 1, device=device))
    print(f'  ✅ D_A(0) = {D_A0.item():.6f} (should be ≈ 0)')
    print(f'  ✅ D_B(1) = {D_B1.item():.6f} (should be ≈ 0)')
    
    print(f'  ✅ Model initialized successfully on {device}')
    print(f'  ✅ Forward pass working: C_pred shape {C_pred.shape}')
    print(f'  ✅ Mutual diffusion working: D_tilde shape {D_tilde.shape}')
    
    return True

def test_darken_pinn_training():
    """Test Darken PINN training convergence"""
    print('\nTesting Darken PINN training...')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    x, t, C = solve_nonlinear_diffusion_fdm(
        C_left=0.0, C_right=1.0, L=1.0, T_end=2.0, Nx=21, Nt=41
    )
    
    T_grid, X_grid = np.meshgrid(t, x, indexing='ij')
    mask = T_grid.flatten() > 1e-6
    t_flat, x_flat, c_flat = T_grid.flatten()[mask], X_grid.flatten()[mask], C.flatten()[mask]
    
    N_data = 200
    idx = np.random.choice(t_flat.size, N_data, replace=False)
    
    t_data = torch.from_numpy(t_flat[idx]).float().view(-1, 1).to(device)
    x_data = torch.from_numpy(x_flat[idx]).float().view(-1, 1).to(device)
    C_data = torch.from_numpy(c_flat[idx]).float().view(-1, 1).to(device)
    
    N_pde = 400
    eps = 1e-4
    t_pde = eps + (2.0 - eps) * torch.rand(N_pde, 1, device=device)
    x_pde = torch.rand(N_pde, 1, device=device)
    
    N_ic = 100
    t_ic = torch.zeros(N_ic, 1, device=device)
    x_ic = torch.rand(N_ic, 1, device=device)
    
    N_bc = 200
    t_bc = eps + (2.0 - eps) * torch.rand(N_bc, 1, device=device)
    
    layers_C = [2, 32, 32, 1]
    layers_DA = [1, 16, 16, 1]
    layers_DB = [1, 16, 16, 1]
    layers_gamma = [1, 16, 16, 1]
    
    pinn = NonlinearDiffusionPINN(
        layers_C, layers_DA, layers_DB, layers_gamma,
        C_left=0.0, C_right=1.0, L=1.0
    ).to(device)
    
    optimizer = torch.optim.Adam(pinn.parameters(), lr=0.001)
    
    lambda_pde, lambda_ic, lambda_bc, lambda_Dbc = 1.0, 2.0, 0.5, 10.0
    
    initial_loss = 0.0
    final_loss = 0.0
    
    for epoch in range(1, 501):
        pinn.train()
        optimizer.zero_grad()
        
        total, loss_data, loss_pde, loss_ic, loss_bc, loss_D_bc = pinn.loss(
            t_data, x_data, C_data, t_pde, x_pde, t_ic, x_ic, t_bc,
            lambda_pde, lambda_ic, lambda_bc, lambda_Dbc
        )
        
        if torch.isnan(total):
            print(f"  ❌ NaN detected at epoch {epoch}")
            return False
        
        if epoch == 1:
            initial_loss = total.item()
        
        total.backward()
        torch.nn.utils.clip_grad_norm_(pinn.parameters(), max_norm=1.0)
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f'  Epoch {epoch:3d} | Total Loss: {total.item():.3e} | Data: {loss_data.item():.3e} | PDE: {loss_pde.item():.3e}')
        
        final_loss = total.item()
    
    improvement_ratio = initial_loss / final_loss if final_loss > 0 else float('inf')
    
    print('  ✅ Training completed without NaN')
    print(f'  ✅ Initial loss: {initial_loss:.3e}')
    print(f'  ✅ Final loss: {final_loss:.3e}')
    print(f'  ✅ Improvement ratio: {improvement_ratio:.2f}x')
    
    C_eval = torch.tensor([[0.5]], device=device, dtype=torch.float32)
    with torch.enable_grad():
        D_tilde_learned = pinn.mutual_diffusion(C_eval).detach().item()
    D_tilde_true = true_diffusion_coefficient(0.5)
    
    print(f'  ✅ D̃(0.5) learned: {D_tilde_learned:.4f}')
    print(f'  ✅ D̃(0.5) true: {D_tilde_true:.4f}')
    
    return improvement_ratio > 2.0 and final_loss < 1.0

def test_darken_physics_constraints():
    """Test physics constraints in Darken model"""
    print('\nTesting Darken model physics constraints...')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    layers_C = [2, 32, 32, 1]
    layers_DA = [1, 16, 16, 1]
    layers_DB = [1, 16, 16, 1]
    layers_gamma = [1, 16, 16, 1]
    
    pinn = NonlinearDiffusionPINN(
        layers_C, layers_DA, layers_DB, layers_gamma,
        C_left=0.0, C_right=1.0, L=1.0
    ).to(device)
    
    C_range = torch.linspace(0.01, 0.99, 50, device=device).view(-1, 1)
    
    with torch.enable_grad():
        D_tilde_values = pinn.mutual_diffusion(C_range)
    
    assert torch.all(D_tilde_values > 0), "Mutual diffusion coefficient should be positive"
    
    D_A_values = pinn._D_self(pinn.net_DA, C_range)
    D_B_values = pinn._D_self(pinn.net_DB, C_range)
    
    assert torch.all(D_A_values >= 0), "Self-diffusion D_A should be non-negative"
    assert torch.all(D_B_values >= 0), "Self-diffusion D_B should be non-negative"
    
    ln_gamma_values = pinn._ln_gamma(C_range)
    assert torch.all(torch.abs(ln_gamma_values) < 10), "Activity coefficient should be reasonable"
    
    print('  ✅ D̃(C) > 0 for all C ∈ [0.01, 0.99]')
    print('  ✅ D_A(C) ≥ 0 for all C')
    print('  ✅ D_B(C) ≥ 0 for all C')
    print('  ✅ |lnγ(C)| < 10 for all C')
    print('  ✅ Physics constraints satisfied')
    
    return True

if __name__ == "__main__":
    print("Testing Darken Model PINNs System")
    print("=" * 50)
    
    fdm_success = test_darken_fdm_generation()
    init_success = test_darken_pinn_initialization()
    training_success = test_darken_pinn_training()
    physics_success = test_darken_physics_constraints()
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"FDM Data Generation: {'PASSED' if fdm_success else 'FAILED'}")
    print(f"PINN Initialization: {'PASSED' if init_success else 'FAILED'}")
    print(f"PINN Training: {'PASSED' if training_success else 'FAILED'}")
    print(f"Physics Constraints: {'PASSED' if physics_success else 'FAILED'}")
    
    all_passed = fdm_success and init_success and training_success and physics_success
    
    if all_passed:
        print("✅ All Darken PINNs tests passed!")
    else:
        print("❌ Some Darken PINNs tests failed. Check implementation.")
    
    print(f"\nGPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
