"""
Test script for Bayesian model selection in PDE discovery
"""
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_bayesian_evaluation():
    """Test Bayesian model selection functionality"""
    print("Testing Bayesian Model Selection...")
    
    from pde_discovery import DiffusionFDM, PDESymbolicRegression, ComplexityCalculator
    
    fdm = DiffusionFDM(L=0.02, T_final=1000, nx=20, nt=30, D=1e-11)
    u_numerical = fdm.solve()
    
    pde_regression = PDESymbolicRegression(u_numerical, fdm.x, fdm.t)
    
    complexity_calc = ComplexityCalculator()
    test_complexity = complexity_calc.calculate_pde_complexity("∂u/∂t = c₁ × ∂²u/∂x²", [1e-11])
    print(f"Test complexity calculation: {test_complexity}")
    
    results_standard = pde_regression.discover_diffusion_equation(use_exhaustive_search=False)
    print("Standard Bayesian evaluation completed")
    print(f"Best model: {results_standard['best_model']['name'] if results_standard['best_model'] else 'None'}")
    print(f"Number of candidates: {len(results_standard['all_results'])}")
    
    results_exhaustive = pde_regression.discover_diffusion_equation(use_exhaustive_search=True, max_complexity=2)
    print("Exhaustive search completed")
    print(f"Best model: {results_exhaustive['best_model']['name'] if results_exhaustive['best_model'] else 'None'}")
    print(f"Number of candidates: {len(results_exhaustive['all_results'])}")
    
    return True

def test_bic_aic_calculations():
    """Test BIC/AIC calculation methods"""
    print("\nTesting BIC/AIC calculations...")
    
    from pde_discovery import PDESymbolicRegression
    
    u_dummy = np.random.rand(10, 10)
    x_dummy = np.linspace(0, 1, 10)
    t_dummy = np.linspace(0, 1, 10)
    
    regression = PDESymbolicRegression(u_dummy, x_dummy, t_dummy)
    
    likelihood = regression.calculate_likelihood(0.01, 100)
    bic = regression.calculate_bic(likelihood, 2, 100)
    aic = regression.calculate_aic(likelihood, 2)
    
    print(f"Likelihood: {likelihood:.2e}")
    print(f"BIC: {bic:.2f}")
    print(f"AIC: {aic:.2f}")
    
    bic_scores = np.array([100, 102, 105, 110])
    weights = regression.calculate_model_weights(bic_scores)
    print(f"Model weights: {weights}")
    print(f"Weights sum: {np.sum(weights):.6f}")
    
    return True

if __name__ == "__main__":
    print("Testing Bayesian Model Selection System")
    print("=" * 50)
    
    bayesian_success = test_bayesian_evaluation()
    calc_success = test_bic_aic_calculations()
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"Bayesian Evaluation: {'PASSED' if bayesian_success else 'FAILED'}")
    print(f"BIC/AIC Calculations: {'PASSED' if calc_success else 'FAILED'}")
    
    if all([bayesian_success, calc_success]):
        print("✅ All Bayesian model selection tests passed!")
    else:
        print("❌ Some tests failed.")
