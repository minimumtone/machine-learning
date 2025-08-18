"""
Test script for PINNs PDE discovery system
"""
import torch
import numpy as np
from pinns_discovery import PINNsHeatSolver, PINNsBurgersSolver

def test_pinns_heat_solver():
    """Test PINNs heat equation solver"""
    print('Testing PINNs Heat Solver...')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    solver = PINNsHeatSolver(alpha=0.01, hidden_dim=30, num_layers=3)
    
    def progress_callback(epoch, total_epochs, loss):
        if epoch % 200 == 0:
            print(f'Epoch {epoch}/{total_epochs}, Loss: {loss:.6f}')
    
    results = solver.train(epochs=1000, lr=0.001, n_points=500, progress_callback=progress_callback)
    
    print(f'Final training loss: {results["final_loss"]:.6f}')
    print(f'PDE loss: {results["pde_loss"]:.6f}')
    print(f'IC loss: {results["ic_loss"]:.6f}')
    print(f'BC loss: {results["bc_loss"]:.6f}')
    
    x_test = np.linspace(0, 1, 20)
    t_test = np.linspace(0, 1, 20)
    X_test, T_test = np.meshgrid(x_test, t_test)
    u_pred = solver.predict(X_test, T_test)
    
    print(f'Prediction shape: {u_pred.shape}')
    print(f'Prediction range: [{u_pred.min():.4f}, {u_pred.max():.4f}]')
    
    return results["final_loss"] < 0.1

def test_pinns_burgers_solver():
    """Test PINNs Burgers equation solver"""
    print('\nTesting PINNs Burgers Solver...')
    
    solver = PINNsBurgersSolver(nu=0.01, hidden_dim=30, num_layers=3)
    
    def progress_callback(epoch, total_epochs, loss):
        if epoch % 200 == 0:
            print(f'Epoch {epoch}/{total_epochs}, Loss: {loss:.6f}')
    
    results = solver.train(epochs=1000, lr=0.001, n_points=500, progress_callback=progress_callback)
    
    print(f'Final training loss: {results["final_loss"]:.6f}')
    print(f'PDE loss: {results["pde_loss"]:.6f}')
    print(f'IC loss: {results["ic_loss"]:.6f}')
    print(f'BC loss: {results["bc_loss"]:.6f}')
    
    x_test = np.linspace(0, 1, 20)
    t_test = np.linspace(0, 0.5, 20)
    X_test, T_test = np.meshgrid(x_test, t_test)
    u_pred = solver.predict(X_test, T_test)
    
    print(f'Prediction shape: {u_pred.shape}')
    print(f'Prediction range: [{u_pred.min():.4f}, {u_pred.max():.4f}]')
    
    return results["final_loss"] < 0.1

if __name__ == "__main__":
    print("Testing PINNs PDE Discovery System")
    print("=" * 50)
    
    heat_success = test_pinns_heat_solver()
    burgers_success = test_pinns_burgers_solver()
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"Heat Equation PINNs: {'PASSED' if heat_success else 'FAILED'}")
    print(f"Burgers Equation PINNs: {'PASSED' if burgers_success else 'FAILED'}")
    
    if heat_success and burgers_success:
        print("✅ All PINNs tests passed!")
    else:
        print("❌ Some PINNs tests failed. Consider adjusting hyperparameters.")
