"""
Test script for diffusion equation implementation
"""
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_diffusion_fdm():
    """Test DiffusionFDM class"""
    print("Testing DiffusionFDM...")
    
    from pde_discovery import DiffusionFDM
    
    fdm = DiffusionFDM(L=0.02, T_final=1000, nx=30, nt=50, D=1e-11)
    u_numerical = fdm.solve()
    
    print(f"Solution shape: {u_numerical.shape}")
    print(f"Solution range: [{u_numerical.min():.3f}, {u_numerical.max():.3f}]")
    print(f"Initial condition check: {u_numerical[0, :15]}")
    
    return u_numerical.shape == (50, 30) and 0 <= u_numerical.min() <= u_numerical.max() <= 1

def test_diffusion_pinns():
    """Test PINNsDiffusionSolver import"""
    print("\nTesting PINNsDiffusionSolver import...")
    
    try:
        from pinns_discovery import PINNsDiffusionSolver
        solver = PINNsDiffusionSolver(D=1e-11, L=0.02, T_final=3600)
        print("PINNsDiffusionSolver created successfully")
        return True
    except ImportError as e:
        print(f"Import failed: {e}")
        return False
    except Exception as e:
        print(f"Creation failed: {e}")
        return False

def test_diffusion_discovery():
    """Test diffusion equation discovery"""
    print("\nTesting diffusion equation discovery...")
    
    try:
        from pde_discovery import DiffusionFDM, PDESymbolicRegression
        
        fdm = DiffusionFDM(L=0.02, T_final=1000, nx=30, nt=40, D=1e-11)
        u_numerical = fdm.solve()
        
        pde_regression = PDESymbolicRegression(u_numerical, fdm.x, fdm.t)
        results = pde_regression.discover_diffusion_equation()
        
        print(f"Discovery completed. Best D: {results['best_D']:.2e}")
        print(f"Number of formulas tested: {len(results['all_results'])}")
        
        return results['best_D'] > 0
    except Exception as e:
        print(f"Discovery test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Diffusion Equation Implementation")
    print("=" * 50)
    
    fdm_success = test_diffusion_fdm()
    pinns_success = test_diffusion_pinns()
    discovery_success = test_diffusion_discovery()
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"DiffusionFDM: {'PASSED' if fdm_success else 'FAILED'}")
    print(f"PINNsDiffusionSolver: {'PASSED' if pinns_success else 'FAILED'}")
    print(f"Diffusion Discovery: {'PASSED' if discovery_success else 'FAILED'}")
    
    if all([fdm_success, pinns_success, discovery_success]):
        print("✅ All diffusion tests passed!")
    else:
        print("❌ Some tests failed.")
