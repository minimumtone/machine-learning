"""
Boys function F_n(t) for evaluation of nuclear attraction and electron repulsion integrals.

F_n(t) = ∫₀¹ u^(2n) exp(-t u²) du

Implements multiple evaluation methods for numerical stability:
- Small t: Taylor series expansion
- Large t: Asymptotic expansion
- Intermediate t: Numerical integration or continued fraction
"""

import numpy as np
from scipy import special
from typing import Union


def boys_function(n: int, t: float) -> float:
    """
    Compute Boys function F_n(t).
    
    F_n(t) = ∫₀¹ u^(2n) exp(-t u²) du
    
    Related to incomplete gamma function:
    F_n(t) = (1/2) * t^(-n-1/2) * γ(n+1/2, t)
    
    Args:
        n: Order of Boys function (n >= 0)
        t: Argument (t >= 0)
    
    Returns:
        F_n(t) value
    """
    if t < 1e-10:
        return 1.0 / (2.0 * n + 1.0)
    
    if t > 30.0:
        return boys_asymptotic(n, t)
    
    return boys_gamma(n, t)


def boys_gamma(n: int, t: float) -> float:
    """
    Compute Boys function using incomplete gamma function.
    
    F_n(t) = (1/2) * t^(-n-1/2) * γ(n+1/2, t)
           = (1/2) * Γ(n+1/2) * gammainc(n+1/2, t) * t^(-n-1/2)
    
    where gammainc is the regularized lower incomplete gamma function.
    """
    if t < 1e-10:
        return 1.0 / (2.0 * n + 1.0)
    
    nu = n + 0.5
    
    gamma_nu = special.gamma(nu)
    
    gammainc_val = special.gammainc(nu, t)
    
    result = 0.5 * gamma_nu * gammainc_val * (t ** (-nu))
    
    return result


def boys_asymptotic(n: int, t: float) -> float:
    """
    Asymptotic expansion for large t.
    
    F_n(t) ≈ (2n-1)!! / (2^(n+1)) * sqrt(π/t^(2n+1))
    
    More accurate: F_n(t) ≈ (1/2) * sqrt(π/t) * (2n-1)!! / (2t)^n
    """
    from math import sqrt, pi
    
    def double_factorial(n):
        if n <= 0:
            return 1
        result = 1
        for i in range(n, 0, -2):
            result *= i
        return result
    
    df = double_factorial(2 * n - 1)
    
    result = 0.5 * sqrt(pi / t) * df / ((2.0 * t) ** n)
    
    return result


def boys_taylor(n: int, t: float, max_terms: int = 50) -> float:
    """
    Taylor series expansion for small t.
    
    F_n(t) = sum_{k=0}^∞ (-t)^k / (k! (2n + 2k + 1))
    
    Converges rapidly for small t.
    """
    result = 0.0
    term = 1.0 / (2.0 * n + 1.0)
    result += term
    
    for k in range(1, max_terms):
        term *= -t / k
        denom = 2.0 * n + 2.0 * k + 1.0
        result += term / denom
        
        if abs(term / denom) < 1e-15:
            break
    
    return result


def boys_function_array(n_max: int, t: float) -> np.ndarray:
    """
    Compute Boys functions F_0(t), F_1(t), ..., F_n_max(t) efficiently.
    
    Uses downward recursion:
    F_n(t) = (2t * F_{n+1}(t) + exp(-t)) / (2n + 1)
    
    Args:
        n_max: Maximum order
        t: Argument
    
    Returns:
        Array of F_n(t) values for n = 0, 1, ..., n_max
    """
    F = np.zeros(n_max + 1)
    
    if t < 1e-10:
        for n in range(n_max + 1):
            F[n] = 1.0 / (2.0 * n + 1.0)
        return F
    
    if t > 30.0:
        for n in range(n_max + 1):
            F[n] = boys_asymptotic(n, t)
        return F
    
    n_start = n_max + 10
    F_high = boys_gamma(n_start, t)
    
    exp_t = np.exp(-t)
    
    F_temp = [0.0] * (n_start + 1)
    F_temp[n_start] = F_high
    
    for n in range(n_start - 1, -1, -1):
        F_temp[n] = ((2.0 * t * F_temp[n + 1] + exp_t) / (2.0 * n + 1.0))
    
    for n in range(n_max + 1):
        F[n] = F_temp[n]
    
    return F


def test_boys_function():
    """Test Boys function implementation."""
    print("Testing Boys function...")
    
    test_cases = [
        (0, 0.0, 1.0),
        (0, 1.0, 0.7468241328124271),
        (1, 0.0, 1.0/3.0),
        (1, 1.0, 0.22579325293837912),
        (2, 0.0, 1.0/5.0),
        (0, 10.0, 0.08862269254527579),
    ]
    
    print("\n  n      t        F_n(t)      Expected    Error")
    print("-" * 60)
    
    for n, t, expected in test_cases:
        result = boys_function(n, t)
        error = abs(result - expected)
        print(f"  {n}   {t:6.2f}   {result:.10f}   {expected:.10f}   {error:.2e}")
    
    print("\nTesting array evaluation...")
    t = 1.0
    F_array = boys_function_array(5, t)
    print(f"F_0({t}) to F_5({t}):")
    for n, val in enumerate(F_array):
        print(f"  F_{n}({t}) = {val:.10f}")


if __name__ == "__main__":
    test_boys_function()
