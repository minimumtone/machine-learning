"""
Unified test script for both FDM and PINNs PDE discovery systems
"""
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pde_discovery import HeatConductionFDM, BurgersFDM, PDESymbolicRegression

try:
    import torch
    from pinns_discovery import PINNsHeatSolver, PINNsBurgersSolver
    PINNS_AVAILABLE = True
except ImportError:
    PINNS_AVAILABLE = False
    print("Warning: PyTorch not available, skipping PINNs tests")

def test_fdm_heat_solver():
    """Test FDM heat equation solver"""
    print('Testing FDM Heat Solver...')
    
    fdm = HeatConductionFDM(nx=30, nt=50, alpha=0.01, T_final=0.5)
    u_numerical = fdm.solve()
    
    print(f'Solution shape: {u_numerical.shape}')
    print(f'Solution range: [{u_numerical.min():.4f}, {u_numerical.max():.4f}]')
    
    pde_regression = PDESymbolicRegression(u_numerical, fdm.x, fdm.t)
    results = pde_regression.discover_heat_equation()
    
    error_percent = abs(results['best_alpha'] - 0.01) / 0.01 * 100
    print(f'Heat equation discovery error: {error_percent:.2f}%')
    
    return error_percent < 10

def test_fdm_burgers_solver():
    """Test FDM Burgers equation solver"""
    print('\nTesting FDM Burgers Solver...')
    
    fdm = BurgersFDM(nx=40, nt=80, nu=0.01, T_final=0.3)
    u_numerical = fdm.solve()
    
    print(f'Solution shape: {u_numerical.shape}')
    print(f'Solution range: [{u_numerical.min():.4f}, {u_numerical.max():.4f}]')
    
    pde_regression = PDESymbolicRegression(u_numerical, fdm.x, fdm.t)
    results = pde_regression.discover_burgers_equation()
    
    error_percent = abs(results['best_nu'] - 0.01) / 0.01 * 100
    print(f'Burgers equation discovery error: {error_percent:.2f}%')
    
    return error_percent < 20

def test_pinns_heat_solver():
    """Test PINNs heat equation solver"""
    if not PINNS_AVAILABLE:
        return True
    
    print('\nTesting PINNs Heat Solver...')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    solver = PINNsHeatSolver(alpha=0.01, hidden_dim=30, num_layers=3)
    
    def progress_callback(epoch, total_epochs, loss):
        if epoch % 200 == 0:
            print(f'Epoch {epoch}/{total_epochs}, Loss: {loss:.6f}')
    
    results = solver.train(epochs=1500, lr=0.0008, n_points=600, progress_callback=progress_callback)
    
    print(f'Final training loss: {results["final_loss"]:.6f}')
    
    x_test = np.linspace(0, 1, 20)
    t_test = np.linspace(0, 1, 20)
    X_test, T_test = np.meshgrid(x_test, t_test)
    u_pred = solver.predict(X_test, T_test)
    
    print(f'Prediction shape: {u_pred.shape}')
    print(f'Prediction range: [{u_pred.min():.4f}, {u_pred.max():.4f}]')
    
    return results["final_loss"] < 0.1

def test_pinns_burgers_solver():
    """Test PINNs Burgers equation solver"""
    if not PINNS_AVAILABLE:
        return True
    
    print('\nTesting PINNs Burgers Solver...')
    
    solver = PINNsBurgersSolver(nu=0.01, hidden_dim=40, num_layers=4)
    
    def progress_callback(epoch, total_epochs, loss):
        if epoch % 500 == 0:
            print(f'Epoch {epoch}/{total_epochs}, Loss: {loss:.6f}')
    
    results = solver.train(epochs=2000, lr=0.0005, n_points=1000, progress_callback=progress_callback)
    
    print(f'Final training loss: {results["final_loss"]:.6f}')
    
    x_test = np.linspace(0, 1, 20)
    t_test = np.linspace(0, 0.5, 20)
    X_test, T_test = np.meshgrid(x_test, t_test)
    u_pred = solver.predict(X_test, T_test)
    
    print(f'Prediction shape: {u_pred.shape}')
    print(f'Prediction range: [{u_pred.min():.4f}, {u_pred.max():.4f}]')
    
    return results["final_loss"] < 1.0

def test_epoch_controls():
    """Test epoch control functionality"""
    if not PINNS_AVAILABLE:
        return True
    
    print('\nTesting Epoch Controls...')
    
    solver = PINNsHeatSolver(alpha=0.01, hidden_dim=20, num_layers=3)
    
    epoch_counts = [500, 1000]
    losses = []
    
    for epochs in epoch_counts:
        results = solver.train(epochs=epochs, lr=0.001, n_points=300)
        losses.append(results["final_loss"])
        print(f'Epochs: {epochs}, Final Loss: {results["final_loss"]:.6f}')
    
    improvement = (losses[0] - losses[1]) / losses[0] * 100
    print(f'Loss improvement with more epochs: {improvement:.2f}%')
    
    return improvement > 0

if __name__ == "__main__":
    print("Testing Unified PDE Discovery System")
    print("=" * 50)
    
    fdm_heat_success = test_fdm_heat_solver()
    fdm_burgers_success = test_fdm_burgers_solver()
    
    if PINNS_AVAILABLE:
        pinns_heat_success = test_pinns_heat_solver()
        pinns_burgers_success = test_pinns_burgers_solver()
        epoch_control_success = test_epoch_controls()
    else:
        pinns_heat_success = True
        pinns_burgers_success = True
        epoch_control_success = True
        print("\nSkipping PINNs tests (PyTorch not available)")
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"FDM Heat Equation: {'PASSED' if fdm_heat_success else 'FAILED'}")
    print(f"FDM Burgers Equation: {'PASSED' if fdm_burgers_success else 'FAILED'}")
    
    if PINNS_AVAILABLE:
        print(f"PINNs Heat Equation: {'PASSED' if pinns_heat_success else 'FAILED'}")
        print(f"PINNs Burgers Equation: {'PASSED' if pinns_burgers_success else 'FAILED'}")
        print(f"Epoch Controls: {'PASSED' if epoch_control_success else 'FAILED'}")
    
    all_tests = [fdm_heat_success, fdm_burgers_success, pinns_heat_success, 
                 pinns_burgers_success, epoch_control_success]
    
    if all(all_tests):
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed. Check individual results above.")
