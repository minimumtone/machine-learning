"""
Test script for Burgers equation PDE discovery system
"""
from pde_discovery import BurgersFDM, PDESymbolicRegression
import numpy as np

def test_burgers_fdm_solver():
    """Test Burgers FDM solver accuracy"""
    print('Testing Burgers FDM solver...')
    fdm = BurgersFDM(nx=50, nt=100, nu=0.01, T_final=0.5)
    print(f'Diffusion stability parameter r = {fdm.r:.3f} (should be <= 0.5)')
    print(f'CFL parameter = {fdm.cfl:.3f} (should be <= 1.0)')

    u = fdm.solve()
    print(f'Solution shape: {u.shape}')
    print(f'Initial max velocity: {u[0,:].max():.3f}')
    print(f'Final max velocity: {u[-1,:].max():.3f}')
    print(f'Shock formation check: min value = {u.min():.3f}')
    
    return u, fdm

def test_burgers_pde_discovery(u, fdm):
    """Test Burgers PDE discovery accuracy"""
    print('\nTesting Burgers PDE discovery...')
    pde_reg = PDESymbolicRegression(u, fdm.x, fdm.t)
    results = pde_reg.discover_burgers_equation()

    best_formula = min(results.items(), key=lambda x: x[1]['mse'])
    print(f'Best discovered PDE: {best_formula[0]}')
    print(f'MSE: {best_formula[1]["mse"]:.2e}')
    print(f'Discovered parameters: {best_formula[1]["params"]}')
    print(f'Theoretical nu: {fdm.nu:.4f}')
    
    print('\nAll results:')
    sorted_results = sorted(results.items(), key=lambda x: x[1]['mse'])
    for name, result in sorted_results[:3]:
        print(f'  {name}: MSE={result["mse"]:.2e}, params={result["params"]}')
    
    if "u × ∂u/∂x" in best_formula[0] and "∂²u/∂x²" in best_formula[0]:
        discovered_nu = abs(best_formula[1]["params"][1]) if len(best_formula[1]["params"]) > 1 else None
        if discovered_nu:
            error = abs(discovered_nu - fdm.nu) / fdm.nu * 100
            print(f'Viscosity error: {error:.2f}%')
            return error < 20.0
    
    return best_formula[1]["mse"] < 0.1

if __name__ == "__main__":
    u, fdm = test_burgers_fdm_solver()
    success = test_burgers_pde_discovery(u, fdm)
    print(f'\nBurgers PDE Discovery Test: {"PASSED" if success else "FAILED"}')
