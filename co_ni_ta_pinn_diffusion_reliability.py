
"""
Streamlit app: Fig.11-style Co / Ni-0.10Ta ternary diffusion-couple PINN
with likelihood-based reliability visualization.

Run:
    pip install streamlit torch numpy pandas plotly
    streamlit run fig11_co_ni_ta_pinn_reliability_v2.py

Purpose:
    - Reproduce a Fig.11-style Co / Ni-0.10Ta diffusion-couple profile.
    - Infer effective Ni-Ta interdiffusion interaction coefficients from experimental-like points.
    - Visualize reliability using:
        1. Low-cost Laplace approximation.
        2. Higher-cost FDM-based random-walk Metropolis MCMC.
    - Show likelihood/posterior uncertainty as BANDS, not only lines.

Important:
    This is an educational prototype, not a full DICTRA/CALPHAD reproduction.
    Co is dependent: x_Co = 1 - x_Ni - x_Ta.
    Independent variables are x_Ni and x_Ta.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# OpenMP runtime workaround for Windows/Anaconda environments
# ---------------------------------------------------------------------------
# Some Windows Python environments load multiple Intel OpenMP runtimes through
# combinations of PyTorch, NumPy, MKL, SciPy, or scikit-learn.
# Clean solution: use a fresh consistent environment.
# Prototype workaround: set these before importing numpy/torch/scipy.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Optional

import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# UI style
# =============================================================================

st.set_page_config(
    page_title="Fig.11 Co/Ni-Ta PINN Reliability",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.stApp {
    background: linear-gradient(135deg, #f8fafc 0%, #eef4f8 52%, #f7f2f8 100%);
    color: #243041;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff 0%, #f3f7fb 100%);
    border-right: 1px solid rgba(148, 163, 184, 0.28);
}
section[data-testid="stSidebar"] * { color: #334155; }
.hero {
    padding: 1.55rem 1.8rem;
    border-radius: 24px;
    background: linear-gradient(135deg, rgba(224,242,254,0.95), rgba(250,245,255,0.92));
    border: 1px solid rgba(148, 163, 184, 0.26);
    box-shadow: 0 14px 34px rgba(100,116,139,0.14);
    margin-bottom: 1.0rem;
}
.hero-title {
    font-size: 2.0rem;
    font-weight: 850;
    letter-spacing: -0.035em;
    color: #1e293b;
}
.hero-sub {
    color: #475569;
    font-size: 1.0rem;
    line-height: 1.7;
    max-width: 1120px;
}
.note {
    padding: 0.95rem 1.15rem;
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.78);
    border: 1px solid rgba(148, 163, 184, 0.24);
    color: #334155;
    line-height: 1.7;
    box-shadow: 0 10px 22px rgba(100,116,139,0.10);
}
div[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.72);
    border: 1px solid rgba(148, 163, 184, 0.20);
    border-radius: 16px;
    padding: 0.75rem 0.9rem;
    box-shadow: 0 8px 20px rgba(100,116,139,0.09);
}
div[data-testid="stMetricValue"] { color: #1e293b; }
div[data-testid="stMetricLabel"] { color: #64748b; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 999px;
    padding: 8px 16px;
    background-color: rgba(255,255,255,0.72);
    border: 1px solid rgba(148,163,184,0.22);
    color: #475569;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #dbeafe, #f3e8ff);
    color: #1e293b;
    border: 1px solid rgba(129,140,248,0.32);
}
</style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# Robustness changelog (merged from v25_robust audit)
# =============================================================================
# Theoretical:
#   T1. D matrix symmetry: FORCE_SYMMETRIC_D=False (default) → 4-param
#       non-symmetric D (rho12, rho21 independent).  Both PINN classes,
#       FDM, reliability/MCMC, UI toggle all support 4-param mode.
#   T2. fdm_ternary_regular_solution: staggered finite-volume scheme.
#   T3. Mass conservation: ad-hoc rescale tracked via _DIAG_COUNTERS.
#   T4. Gaussian NLL (n/2)log(2π) omitted — constant; documented.
# Numerical:
#   N1. _robust_cov_from_hessian: eps_rel eigen-floor (no two-stage clipping).
#   N2. bilinear_sample_xt: vectorized (legacy preserved).
#   N3. MCMC: per-dim proposal AND Hessian-informed proposal_cov via
#       _mcmc_proposal_from_cov (Roberts-Gelman-Gilks 2.38^2/d optimal).
#   N4. FDM NLL failures: logged to _DIAG_COUNTERS.
#   N5. numerical_hessian_adaptive: per-dimension step sizing.
# Statistical:
#   S1. chi2_misspecification_diagnosis: Wald-style misspecification check.
#   S2. geweke_diagnostic: MCMC convergence z-score.
#   S3. PSIS (Pareto-k̂) diagnostic: quantitative Laplace quality check.
#       k̂ < 0.5 = good, 0.5-0.7 = marginal, > 0.7 = unreliable.
#   S4. σ marginalization: half-Cauchy(0, 0.1) prior on σ, joint θ-σ MCMC.
#       Addresses the assumption that observation noise scale is known.
# Training robustness:
#   R1. NaN/Inf guard + clip_grad_norm_(max_norm=10) in train_pinn/train_pinn_rs.
#   R2. Self-adaptive loss weighting (RBA): gradient-norm rebalancing
#       w_i = w_i^base × mean(‖∇L_j‖) / ‖∇L_i‖ every N epochs.
# Output:
#   O1. Direct [Ni, Ta] output mode: MLP outputs 2 values via sigmoid,
#       Co = 1 − Ni − Ta.  Avoids normalization Jacobian in PDE residuals.
# =============================================================================

# =============================================================================
# Constants and helpers
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

COMPONENTS = ["Co", "Ni", "Ta"]

# ---------------------------------------------------------------------------
# CPU / GPU benchmark and device selection
# ---------------------------------------------------------------------------
_CUDA_AVAILABLE = torch.cuda.is_available()
_GPU_NAME = torch.cuda.get_device_name(0) if _CUDA_AVAILABLE else "N/A"


def _set_device(dev_str: str) -> torch.device:
    """Update the module-level DEVICE variable and return the new device."""
    global DEVICE
    DEVICE = torch.device(dev_str)
    return DEVICE


def run_device_benchmark(
    epochs: int = 30,
    width: int = 64,
    depth: int = 4,
    nx: int = 80,
) -> dict:
    """Run a short PINN-like forward+backward benchmark on CPU and (if available) GPU.

    Returns dict with keys: cpu_ms, gpu_ms (None if no GPU), recommended, speedup.
    The benchmark mimics actual PINN training: forward pass + PDE residual + backward.
    Designed to complete within ~10-30 seconds total.
    """
    import torch.nn as nn

    class _BenchNet(nn.Module):
        """Minimal MLP matching the PINN architecture for benchmarking."""
        def __init__(self, w, d, dev):
            super().__init__()
            layers = [nn.Linear(2, w), nn.Tanh()]
            for _ in range(d - 1):
                layers += [nn.Linear(w, w), nn.Tanh()]
            layers.append(nn.Linear(w, 3))
            self.net = nn.Sequential(*layers).to(dev)

        def forward(self, x, t):
            xt = torch.cat([x, t], dim=1)
            return self.net(xt)

    def _bench_device(dev, ep, w, d, n):
        device = torch.device(dev)
        model = _BenchNet(w, d, device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = torch.randn(n, 1, device=device, requires_grad=True)
        t = torch.randn(n, 1, device=device, requires_grad=True)
        c_target = torch.rand(n, 3, device=device)

        # Warmup (2 iters)
        for _ in range(2):
            opt.zero_grad()
            c = model(x, t)
            loss = ((c - c_target) ** 2).mean()
            # Simulate PDE residual: compute grad
            dc_dx = torch.autograd.grad(c.sum(), x, create_graph=True)[0]
            loss = loss + (dc_dx ** 2).mean()
            loss.backward()
            opt.step()

        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(ep):
            opt.zero_grad()
            c = model(x, t)
            loss = ((c - c_target) ** 2).mean()
            dc_dx = torch.autograd.grad(c.sum(), x, create_graph=True)[0]
            loss = loss + (dc_dx ** 2).mean()
            loss.backward()
            opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000  # ms
        return elapsed / ep  # ms per epoch

    result = {}
    # CPU benchmark
    cpu_ms = _bench_device("cpu", epochs, width, depth, nx)
    result["cpu_ms"] = round(cpu_ms, 2)
    result["gpu_ms"] = None
    result["gpu_name"] = _GPU_NAME
    result["speedup"] = 1.0
    result["recommended"] = "cpu"

    if _CUDA_AVAILABLE:
        gpu_ms = _bench_device("cuda", epochs, width, depth, nx)
        result["gpu_ms"] = round(gpu_ms, 2)
        result["speedup"] = round(cpu_ms / max(gpu_ms, 0.01), 2)
        result["recommended"] = "cuda" if gpu_ms < cpu_ms else "cpu"

    return result

# --- D-matrix symmetry flag (T1) ----------------------------------------------
# When True: 3-parameter symmetric D  (log D11, log D22, rho_raw → D12=D21).
# When False: 4-parameter non-symmetric D (log D11, log D22, rho12_raw, rho21_raw).
# Onsager symmetry holds for L_ij = L_ji, but D = L * Phi is generally
# non-symmetric; e.g. D_NiTa != D_TaNi in real interdiffusion.
# Left/right mode: 6 params (symmetric) or 8 params (non-symmetric).
# Default False: the physically correct model.  UI checkbox overrides per-run.
FORCE_SYMMETRIC_D: bool = False
THETA_DIM_SINGLE: int = 3 if FORCE_SYMMETRIC_D else 4
THETA_DIM_LR: int = 2 * THETA_DIM_SINGLE

# --- Diagnostic counters (T3, N4) --------------------------------------------
# Mutable dict recording runtime events for post-hoc analysis.
# P9: also track magnitude of mass-conservation corrections.
# C7 note: Streamlit @st.cache_data on run_fdm_teacher means counters are NOT
# updated on cache hits — values may undercount when caching is active.
_DIAG_COUNTERS: Dict[str, float] = {
    "fdm_clip_events": 0,       # mass-conservation ad-hoc rescale count
    "fdm_clip_max_delta": 0.0,  # P9: largest single-step mass correction magnitude
    "fdm_nll_failures": 0,      # FDM forward-solve failures inside NLL
}


def diag_reset() -> None:
    """Reset all diagnostic counters to zero."""
    for k in _DIAG_COUNTERS:
        _DIAG_COUNTERS[k] = 0


def diag_snapshot() -> Dict[str, int]:
    """Return a snapshot copy of current diagnostic counters."""
    return dict(_DIAG_COUNTERS)

COLORS = {"Co": "black", "Ni": "#2b83ba", "Ta": "#4daf4a"}
SYMBOLS = {"Co": "circle-open", "Ni": "square-open", "Ta": "triangle-up-open"}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def safe_streamlit_rerun():
    """Rerun Streamlit after storing results.

    Newer Streamlit uses st.rerun(); older versions used st.experimental_rerun().
    """
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()


def to_tensor(a: np.ndarray) -> torch.Tensor:
    return torch.tensor(a, dtype=DTYPE, device=DEVICE)


def distance_um_from_x(x: np.ndarray, span_um: float = 800.0) -> np.ndarray:
    return (x - 0.5) * span_um


def real_time_hours_from_tau(tau: float, tau_max: float, annealing_time_h: float) -> float:
    """Map normalized diffusion time tau to physical time in hours."""
    tau_max = max(float(tau_max), 1.0e-14)
    return float(tau) / tau_max * float(annealing_time_h)


def format_time_label(tau: float, tau_max: float, annealing_time_h: float) -> str:
    """Readable label for normalized and physical time."""
    th = real_time_hours_from_tau(tau, tau_max, annealing_time_h)
    if th < 1.0:
        return f"τ={tau:.4g}, real t={th*60:.2f} min"
    if th < 48.0:
        return f"τ={tau:.4g}, real t={th:.2f} h"
    return f"τ={tau:.4g}, real t={th/24:.2f} d"


def make_spd_matrix_np(log_d11: float, log_d22: float, rho_raw: float) -> np.ndarray:
    """Symmetric positive-definite 2x2 diffusion matrix (FORCE_SYMMETRIC_D=True).

    D12 = rho * sqrt(D11*D22), |rho| < 0.95.  D12 = D21.
    """
    d11 = float(np.exp(log_d11))
    d22 = float(np.exp(log_d22))
    rho = float(0.95 * np.tanh(rho_raw))
    d12 = rho * np.sqrt(d11 * d22)
    return np.array([[d11, d12], [d12, d22]], dtype=float)


def make_nonsym_d_matrix_np(
    log_d11: float, log_d22: float, rho12_raw: float, rho21_raw: float,
) -> np.ndarray:
    """Non-symmetric 2x2 diffusion matrix (FORCE_SYMMETRIC_D=False).

    D12 = rho12 * sqrt(D11*D22), D21 = rho21 * sqrt(D11*D22).
    Each |rho| < 0.95, ensuring det(D) > 0 when rho12*rho21 < 1.

    **P3 note**: When rho12 and rho21 have opposite signs, the discriminant
    (d11-d22)^2 + 4*d12*d21 can become negative, yielding complex eigenvalues.
    Physical interdiffusion matrices should have real positive eigenvalues.
    The function warns but does not prevent this — callers should verify the
    CFL condition with ``max(abs(eigvals))`` using complex modulus.
    """
    d11 = float(np.exp(log_d11))
    d22 = float(np.exp(log_d22))
    scale = np.sqrt(d11 * d22)
    d12 = float(0.95 * np.tanh(rho12_raw)) * scale
    d21 = float(0.95 * np.tanh(rho21_raw)) * scale
    D = np.array([[d11, d12], [d21, d22]], dtype=float)
    # P3: warn if eigenvalues are complex (discriminant < 0)
    disc = (d11 - d22) ** 2 + 4.0 * d12 * d21
    if disc < 0:
        import warnings
        warnings.warn(
            f"make_nonsym_d_matrix_np: complex eigenvalues (disc={disc:.3e}). "
            "CFL bound uses |eigmax| (complex modulus) but von Neumann stability "
            "analysis for the explicit scheme is not guaranteed.",
            stacklevel=2,
        )
    return D


def make_d_matrix_from_theta(theta: np.ndarray) -> np.ndarray:
    """Dispatch to symmetric (3 params) or non-symmetric (4 params) D constructor."""
    theta = np.asarray(theta, dtype=float).ravel()
    if theta.size == 3:
        return make_spd_matrix_np(theta[0], theta[1], theta[2])
    elif theta.size == 4:
        return make_nonsym_d_matrix_np(theta[0], theta[1], theta[2], theta[3])
    else:
        raise ValueError(f"theta must have 3 or 4 elements, got {theta.size}")


def theta_from_D_matrix(D: np.ndarray, force_symmetric: Optional[bool] = None) -> np.ndarray:
    """Extract theta from D matrix. Returns 3 params (symmetric) or 4 (non-symmetric)."""
    if force_symmetric is None:
        force_symmetric = FORCE_SYMMETRIC_D
    d11 = max(float(D[0, 0]), 1.0e-14)
    d22 = max(float(D[1, 1]), 1.0e-14)
    scale = max(np.sqrt(d11 * d22), 1.0e-14)
    if force_symmetric:
        rho = float(D[0, 1]) / scale
        rho_scaled = np.clip(rho / 0.95, -0.999999, 0.999999)
        return np.array([np.log(d11), np.log(d22), np.arctanh(rho_scaled)], dtype=float)
    else:
        rho12 = float(D[0, 1]) / scale
        rho21 = float(D[1, 0]) / scale
        rho12_s = np.clip(rho12 / 0.95, -0.999999, 0.999999)
        rho21_s = np.clip(rho21 / 0.95, -0.999999, 0.999999)
        return np.array([np.log(d11), np.log(d22),
                         np.arctanh(rho12_s), np.arctanh(rho21_s)], dtype=float)


def zero_interaction_theta_from_D(D: np.ndarray) -> np.ndarray:
    """Return theta with off-diagonal interaction forced to zero."""
    d11 = max(float(D[0, 0]), 1.0e-14)
    d22 = max(float(D[1, 1]), 1.0e-14)
    if FORCE_SYMMETRIC_D:
        return np.array([np.log(d11), np.log(d22), 0.0], dtype=float)
    else:
        return np.array([np.log(d11), np.log(d22), 0.0, 0.0], dtype=float)


def compute_zero_interaction_reference(
    x_query: np.ndarray,
    t_query: np.ndarray,
    t_max: float,
    nx: int,
    nt_save: int,
    D_source: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute FDM reference with ideal zero cross interaction.

    The zero-interaction FDM may save a different number of time frames because
    its stable time step can differ from the original FDM. Therefore the result
    is interpolated onto the caller's x_query and t_query grids. This prevents
    C_zero[idx] from using indices from another time grid.
    """
    theta_zero = zero_interaction_theta_from_D(D_source)
    rho21 = float(theta_zero[3]) if len(theta_zero) > 3 else None
    xz, tz, Cz, Dz = run_fdm_teacher(
        float(theta_zero[0]),
        float(theta_zero[1]),
        float(theta_zero[2]),
        float(t_max),
        int(nx),
        int(nt_save),
        rho21_raw=rho21,
    )
    Xq, Tq = np.meshgrid(x_query, t_query)
    C_interp = bilinear_sample_xt(xz, tz, Cz, Xq.reshape(-1, 1), Tq.reshape(-1, 1))
    C_interp = C_interp.reshape(len(t_query), len(x_query), 3)
    return np.asarray(t_query), C_interp, Dz, theta_zero


def initial_diffusion_couple(x: np.ndarray, interface: float = 0.5) -> np.ndarray:
    """Sharp Co / Ni-0.10Ta diffusion couple."""
    C = np.zeros((len(x), 3), dtype=float)
    left = x < interface
    right = ~left
    C[left, 0] = 1.0
    C[right, 1] = 0.90
    C[right, 2] = 0.10
    return C


# =============================================================================
# Regular-solution thermodynamics (chemical potential approach)
# =============================================================================

def pair_indices_rs(n_components: int) -> List[Tuple[int, int]]:
    """Return ordered pair indices (i,j) with i<j for N components."""
    return [(i, j) for i in range(n_components) for j in range(i + 1, n_components)]


def omega_matrix_from_pairs_np(theta: Sequence[float], n_components: int) -> np.ndarray:
    """Build symmetric Omega matrix with zero diagonal from pair parameters."""
    theta = np.asarray(theta, dtype=float).reshape(-1)
    pairs = pair_indices_rs(n_components)
    if theta.size != len(pairs):
        raise ValueError(f"Expected {len(pairs)} Omega pair values, got {theta.size}.")
    Omega = np.zeros((n_components, n_components), dtype=float)
    for val, (i, j) in zip(theta, pairs):
        Omega[i, j] = Omega[j, i] = float(val)
    return Omega


def omega_matrix_from_pairs_torch(theta: torch.Tensor, n_components: int) -> torch.Tensor:
    """Torch version of omega_matrix_from_pairs_np."""
    theta = theta.reshape(-1)
    pairs = pair_indices_rs(n_components)
    if theta.numel() != len(pairs):
        raise ValueError(f"Expected {len(pairs)} Omega pair values, got {theta.numel()}.")
    Omega = torch.zeros((n_components, n_components), dtype=theta.dtype, device=theta.device)
    for k, (i, j) in enumerate(pairs):
        Omega[i, j] = theta[k]
        Omega[j, i] = theta[k]
    return Omega


def blend_pairs_np(theta_left: np.ndarray, theta_right: np.ndarray, x: np.ndarray,
                   x_interface: float = 0.5, width: float = 0.02) -> np.ndarray:
    """Smoothly blend left/right pair-interaction vectors as a function of x."""
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    w = max(float(width), 1.0e-12)
    s = 0.5 * (1.0 + np.tanh((x - float(x_interface)) / w))
    return ((1.0 - s) * np.asarray(theta_left, dtype=float).reshape(1, -1)
            + s * np.asarray(theta_right, dtype=float).reshape(1, -1))


def blend_pairs_torch(theta_left: torch.Tensor, theta_right: torch.Tensor,
                      x: torch.Tensor, x_interface: float = 0.5,
                      width: float = 0.02) -> torch.Tensor:
    """Torch version of blend_pairs_np."""
    x = x.reshape(-1, 1)
    w = max(float(width), 1.0e-12)
    s = 0.5 * (1.0 + torch.tanh((x - float(x_interface)) / w))
    return (1.0 - s) * theta_left.reshape(1, -1) + s * theta_right.reshape(1, -1)


def complete_composition_np(c_ind: np.ndarray) -> np.ndarray:
    """Append dependent component c_ref = 1 - sum(c_ind) as the last column."""
    c_dep = 1.0 - np.sum(c_ind, axis=1, keepdims=True)
    return np.concatenate([c_ind, c_dep], axis=1)


def diffusion_potentials_regular_solution_np(
    c_full: np.ndarray,
    x: np.ndarray,
    theta_left: Sequence[float],
    theta_right: Optional[Sequence[float]] = None,
    RT: float = 1.0,
    x_interface: float = 0.5,
    width: float = 0.02,
    eps: float = 1.0e-12,
) -> np.ndarray:
    """Return diffusion potentials mu_i - mu_ref for i=0..N-2 (NumPy).

    Regular solution free-energy:
        g = RT sum_a c_a ln c_a + 1/2 sum_ab Omega_ab c_a c_b

    With the last component as dependent reference r:
        mu_i - mu_r = RT ln(c_i/c_r) + sum_b (Omega_ib - Omega_rb) c_b
    """
    c = np.clip(c_full, eps, 1.0)
    c = c / np.sum(c, axis=1, keepdims=True)
    n_components = c.shape[1]
    n_ind = n_components - 1
    ref = n_components - 1

    theta_left = np.asarray(theta_left, dtype=float)
    if theta_right is None:
        theta_right_arr = theta_left
    else:
        theta_right_arr = np.asarray(theta_right, dtype=float)
    theta_x = blend_pairs_np(theta_left, theta_right_arr, x, x_interface, width)
    pairs = pair_indices_rs(n_components)

    ideal = float(RT) * np.log(c[:, :n_ind] / c[:, ref:ref + 1])
    mu_cols = []
    for i in range(n_ind):
        excess = np.zeros((c.shape[0], 1), dtype=float)
        for k, (a, b) in enumerate(pairs):
            coeff_i = np.zeros((c.shape[0], 1), dtype=float)
            coeff_r = np.zeros((c.shape[0], 1), dtype=float)
            if a == i:
                coeff_i += theta_x[:, k:k + 1] * c[:, b:b + 1]
            if b == i:
                coeff_i += theta_x[:, k:k + 1] * c[:, a:a + 1]
            if a == ref:
                coeff_r += theta_x[:, k:k + 1] * c[:, b:b + 1]
            if b == ref:
                coeff_r += theta_x[:, k:k + 1] * c[:, a:a + 1]
            excess += coeff_i - coeff_r
        mu_cols.append(ideal[:, i:i + 1] + excess)
    return np.concatenate(mu_cols, axis=1)


def diffusion_potentials_regular_solution_torch(
    c_full: torch.Tensor,
    x: torch.Tensor,
    theta_left: torch.Tensor,
    theta_right: Optional[torch.Tensor] = None,
    RT: float = 1.0,
    x_interface: float = 0.5,
    width: float = 0.02,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Torch diffusion potentials mu_i - mu_ref for multicomponent regular solution."""
    c = torch.clamp(c_full, eps, 1.0)
    c = c / torch.sum(c, dim=1, keepdim=True)
    n_components = c.shape[1]
    n_ind = n_components - 1
    ref = n_components - 1

    if theta_right is None:
        theta_right = theta_left
    theta_x = blend_pairs_torch(theta_left, theta_right, x, x_interface, width)
    pairs = pair_indices_rs(n_components)

    ideal = float(RT) * torch.log(c[:, :n_ind] / c[:, ref:ref + 1])
    mu_cols = []
    for i in range(n_ind):
        excess = torch.zeros((c.shape[0], 1), dtype=c.dtype, device=c.device)
        for k, (a, b) in enumerate(pairs):
            coeff_i = torch.zeros((c.shape[0], 1), dtype=c.dtype, device=c.device)
            coeff_r = torch.zeros((c.shape[0], 1), dtype=c.dtype, device=c.device)
            if a == i:
                coeff_i = coeff_i + theta_x[:, k:k + 1] * c[:, b:b + 1]
            if b == i:
                coeff_i = coeff_i + theta_x[:, k:k + 1] * c[:, a:a + 1]
            if a == ref:
                coeff_r = coeff_r + theta_x[:, k:k + 1] * c[:, b:b + 1]
            if b == ref:
                coeff_r = coeff_r + theta_x[:, k:k + 1] * c[:, a:a + 1]
            excess = excess + coeff_i - coeff_r
        mu_cols.append(ideal[:, i:i + 1] + excess)
    return torch.cat(mu_cols, dim=1)


def sanitize_independent(c_ind: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    """Clip independent compositions and ensure the dependent component is non-negative."""
    c_ind = np.clip(c_ind, eps, 1.0 - eps)
    total = np.sum(c_ind, axis=1, keepdims=True)
    over = total > (1.0 - eps)
    if np.any(over):
        c_ind = np.where(over, c_ind / (total + eps) * (1.0 - eps), c_ind)
    return c_ind


# =============================================================================
# Composition-dependent mobility (CALPHAD style)
# =============================================================================

def mobility_matrix_from_endmembers_np(
    c_full: np.ndarray,
    log_M_endmembers: np.ndarray,
    eps: float = 1.0e-12,
) -> np.ndarray:
    """Composition-dependent mobility matrix M(c) using end-member mixing.

    CALPHAD model (simplified, fixed T):
        ln M_i(c) = Σ_j x_j * ln(M_i^j)

    Parameters
    ----------
    c_full : (Nx, n_components) array of full compositions.
    log_M_endmembers : (n_ind, n_components) array of ln(M_i^j) values.
        Row i = independent component i, column j = end-member j.

    Returns
    -------
    M : (Nx, n_ind, n_ind) diagonal mobility matrix at each grid point.
    """
    c = np.clip(c_full, eps, 1.0)
    c = c / np.sum(c, axis=1, keepdims=True)
    n_ind = log_M_endmembers.shape[0]
    Nx = c.shape[0]
    M = np.zeros((Nx, n_ind, n_ind), dtype=float)
    for i in range(n_ind):
        log_Mi = np.sum(c * log_M_endmembers[i:i + 1, :], axis=1)
        M[:, i, i] = np.exp(log_Mi)
    return M


def _mobility_diag_from_endmembers_np(
    c_full: np.ndarray,
    log_M_endmembers: np.ndarray,
    eps: float = 1.0e-12,
) -> np.ndarray:
    """Fast diagonal-only mobility computation: returns (Nx, n_ind) array.

    **P5 note**: When ``use_comp_dep_M=True``, mobility is **diagonal only**
    (off-diagonal M_ij = 0).  This means flux_i = M_ii * dmu_i with no
    cross-coupling.  When ``use_comp_dep_M=False`` (the default constant-M path),
    the full (n_ind, n_ind) mobility matrix is used.  These represent
    fundamentally different physical models — document the choice explicitly.
    The mixing rule ``ln M_i(c) = Σ_j x_j ln M_i^j`` is a thermodynamic
    average approximation; CALPHAD standard uses Redlich-Kister expansion.
    """
    c = np.clip(c_full, eps, 1.0)
    c = c / np.sum(c, axis=1, keepdims=True)
    log_M = c @ log_M_endmembers.T
    return np.exp(log_M)


def mobility_matrix_from_endmembers_torch(
    c_full: torch.Tensor,
    log_M_endmembers: torch.Tensor,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Composition-dependent mobility M(c) for PyTorch (auto-diff compatible).

    Parameters
    ----------
    c_full : (N, n_components) tensor.
    log_M_endmembers : (n_ind, n_components) tensor of ln(M_i^j).

    Returns
    -------
    M : (N, n_ind, n_ind) diagonal mobility tensor.
    """
    c = torch.clamp(c_full, eps, 1.0)
    c = c / torch.sum(c, dim=1, keepdim=True)
    n_ind = log_M_endmembers.shape[0]
    N = c.shape[0]
    M = torch.zeros((N, n_ind, n_ind), dtype=c.dtype, device=c.device)
    for i in range(n_ind):
        log_Mi = torch.sum(c * log_M_endmembers[i:i + 1, :], dim=1)
        M[:, i, i] = torch.exp(log_Mi)
    return M


def make_initial_profile_ternary_rs(
    x: np.ndarray,
    c_left: np.ndarray,
    c_right: np.ndarray,
    x0: float = 0.5,
    width: float = 0.02,
) -> np.ndarray:
    """Create smooth initial composition profile for all 3 components."""
    x = np.asarray(x, dtype=float).reshape(-1)
    c_left = np.asarray(c_left, dtype=float).reshape(-1)
    c_right = np.asarray(c_right, dtype=float).reshape(-1)
    w = max(float(width), 1.0e-12)
    s = 0.5 * (1.0 + np.tanh((x - float(x0)) / w))
    n_components = len(c_left)
    c0 = np.zeros((len(x), n_components), dtype=float)
    for a in range(n_components):
        c0[:, a] = (1.0 - s) * float(c_left[a]) + s * float(c_right[a])
    return c0


def _rs_interdiffusion_matrix_np(
    c_full: np.ndarray,
    theta_pairs: np.ndarray,
    RT: float,
    mobility: np.ndarray,
    use_comp_dep_M: bool = False,
    log_M_endmembers: Optional[np.ndarray] = None,
    eps: float = 1.0e-14,
) -> np.ndarray:
    """Compute interdiffusion coefficient matrix D̃(c) from Onsager coefficients.

    For diagonal L_kj = M_kk δ_kj (independent-component mobility):
        D̃_km = M_kk × ∂(μ_k - μ_ref)/∂c_m

    The thermodynamic factor for regular solution:
        k = m:  RT/c_k + RT/c_ref - 2Ω_{k,ref}
        k ≠ m:  RT/c_ref + Ω_km - Ω_{k,ref} - Ω_{ref,m}

    Unlike the Onsager form (J = M∂μ/∂z), the Fick form (J = -D̃∂c/∂z)
    avoids artificial flux from eps-clipping of log(c).  When c_k → 0,
    D̃_kk → ∞ but ∂c_k/∂z ∝ c_k, so the product stays bounded.

    Parameters
    ----------
    c_full : (Nx, n_components) compositions at half-grid points.
    theta_pairs : (Nx, n_pairs) spatially-varying Ω parameters.
    RT : scalar.
    mobility : (n_ind, n_ind) constant mobility matrix.
    """
    Nx, n_components = c_full.shape
    n_ind = n_components - 1
    ref = n_components - 1
    c = np.clip(c_full, eps, 1.0)
    c = c / np.sum(c, axis=1, keepdims=True)

    Omega = np.zeros((Nx, n_components, n_components), dtype=float)
    pairs = pair_indices_rs(n_components)
    for p_idx, (a, b) in enumerate(pairs):
        Omega[:, a, b] = theta_pairs[:, p_idx]
        Omega[:, b, a] = theta_pairs[:, p_idx]

    if use_comp_dep_M and log_M_endmembers is not None:
        M_diag = _mobility_diag_from_endmembers_np(c, log_M_endmembers)
    else:
        M_diag = np.zeros((Nx, n_ind), dtype=float)
        for i in range(n_ind):
            M_diag[:, i] = mobility[i, i]

    c_ref = c[:, ref]
    D_tilde = np.zeros((Nx, n_ind, n_ind), dtype=float)
    for k in range(n_ind):
        for m in range(n_ind):
            if k == m:
                thermo_factor = RT / c[:, k] + RT / c_ref - 2.0 * Omega[:, k, ref]
            else:
                thermo_factor = (RT / c_ref
                                 + Omega[:, k, m]
                                 - Omega[:, k, ref]
                                 - Omega[:, ref, m])
            D_tilde[:, k, m] = M_diag[:, k] * thermo_factor

    return D_tilde


def _rs_compute_div_flux(
    c_full: np.ndarray,
    x: np.ndarray,
    dx: float,
    mobility: np.ndarray,
    theta_left: np.ndarray,
    theta_right: np.ndarray,
    RT: float,
    x_interface: float,
    omega_width: float,
    use_comp_dep_M: bool,
    log_M_endmembers: Optional[np.ndarray],
) -> np.ndarray:
    """Compute div(flux) for the RS FDM scheme (Onsager form).

    Uses J_k = Σ_j M_kj ∂(μ_j - μ_ref)/∂z with zero-flux Neumann BC
    (DICTRA default for closed-system diffusion couples).

    Boundary cells participate in the flux balance with one-sided
    differences, ensuring exact discrete mass conservation:
    Σ_j div_flux[j,k] = 0 for each component k.
    """
    n_components = c_full.shape[1]
    n_ind = n_components - 1
    Nx = len(x)

    c_ind = sanitize_independent(c_full[:, :n_ind])
    c_full_safe = complete_composition_np(c_ind)

    mu = diffusion_potentials_regular_solution_np(
        c_full_safe, x, theta_left, theta_right, RT=RT,
        x_interface=x_interface, width=omega_width,
    )

    dmu_half = (mu[1:] - mu[:-1]) / dx

    if use_comp_dep_M:
        M_full = _mobility_diag_from_endmembers_np(c_full_safe, log_M_endmembers)
        M_half = 0.5 * (M_full[:-1] + M_full[1:])
        flux_half = M_half * dmu_half
    else:
        flux_half = np.zeros((Nx - 1, n_ind), dtype=float)
        for i_comp in range(n_ind):
            for j_comp in range(n_ind):
                flux_half[:, i_comp] += mobility[i_comp, j_comp] * dmu_half[:, j_comp]

    div_flux = np.zeros((Nx, n_ind), dtype=float)
    div_flux[1:-1] = (flux_half[1:] - flux_half[:-1]) / dx
    # Closed system (zero-flux Neumann BC, DICTRA default):
    # Boundary flux = 0, so boundary cells see one-sided flux balance.
    # This ensures exact mass conservation: Σ div_flux × dx = 0.
    div_flux[0] = flux_half[0] / dx          # left:  J_{1/2} - 0
    div_flux[-1] = -flux_half[-1] / dx       # right: 0 - J_{N-3/2}
    return div_flux


def fdm_ternary_regular_solution(
    c0_full: np.ndarray,
    x: np.ndarray,
    dt: float,
    nsteps: int,
    mobility: np.ndarray,
    theta_left: np.ndarray,
    theta_right: Optional[np.ndarray] = None,
    RT: float = 1.0,
    x_interface: float = 0.5,
    omega_width: float = 0.02,
    save_every: int = 100,
    log_M_endmembers: Optional[np.ndarray] = None,
    cfl_safety: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """DICTRA-style finite-volume solver for ternary RS diffusion.

    Onsager-form FDM solver: J_k = Σ_j M_kj ∂(μ_j - μ_ref)/∂z.

    Boundary conditions: closed system (zero-flux Neumann BC), matching
    DICTRA default for diffusion-couple simulations.  Boundary cells
    participate in the flux balance via one-sided differences, ensuring
    exact discrete mass conservation: Σ div_flux × dx = 0.

    Adaptive sub-stepping with composition-aware CFL: the sub-step size
    is limited both by ``cfl_safety / max|div_flux|`` (standard CFL) and
    by a positivity constraint that prevents any composition from going
    below ``eps_floor``.

    If ``log_M_endmembers`` is provided (shape ``(n_ind, n_components)``),
    the mobility matrix is **composition-dependent** at each grid point:
        ln M_i(c) = Σ_j x_j ln(M_i^j)
    Otherwise ``mobility`` is used as a constant (n_ind, n_ind) matrix.

    Returns (t_grid, C_history) where C_history has shape (n_saved, Nx, n_components).
    """
    use_comp_dep_M = log_M_endmembers is not None
    Nx = len(x)
    dx = float(x[1] - x[0])
    n_components = c0_full.shape[1]
    n_ind = n_components - 1

    c_full = c0_full.copy()
    # Closed system (DICTRA default): no Dirichlet enforcement needed.
    # Zero-flux Neumann BC is handled inside _rs_compute_div_flux.
    snapshots = [c_full.copy()]
    t_saved = [0.0]
    t = 0.0
    total_substeps = 0

    if theta_right is None:
        theta_right = theta_left.copy()

    # Spinodal warning: if max(Ω)/RT > 2, system may be thermodynamically unstable
    max_omega = float(np.max(np.abs(theta_left)))
    if theta_right is not None:
        max_omega = max(max_omega, float(np.max(np.abs(theta_right))))
    if RT > 0 and max_omega / RT > 2.0:
        print(f"[RS-FDM] Warning: max(Ω)/RT = {max_omega / RT:.1f} > 2 "
              f"(spinodal region). Adaptive sub-stepping active.")

    for step in range(1, nsteps + 1):
        remaining = dt
        while remaining > 1.0e-18:
            div_flux = _rs_compute_div_flux(
                c_full, x, dx, mobility, theta_left, theta_right,
                RT, x_interface, omega_width, use_comp_dep_M, log_M_endmembers,
            )

            max_div = float(np.max(np.abs(div_flux)))
            if max_div > 1.0e-30:
                dt_safe = cfl_safety / max_div
            else:
                dt_safe = remaining

            # Composition-aware CFL: prevent any c_i from going negative.
            # For each grid point where div_flux < 0 (composition decreasing),
            # limit dt so that c_i + dt*div_flux_i >= eps_floor.
            eps_floor = 1.0e-14
            c_all = np.column_stack([c_full[:, :n_ind],
                                     c_full[:, -1:]])
            div_all = np.column_stack([div_flux,
                                       -np.sum(div_flux, axis=1, keepdims=True)])
            neg_mask = div_all < -1.0e-30
            if np.any(neg_mask):
                ratios = (c_all[neg_mask] - eps_floor) / (-div_all[neg_mask])
                dt_comp = float(np.min(ratios))
                if dt_comp > 0:
                    dt_safe = min(dt_safe, dt_comp)

            sub_dt = min(dt_safe, remaining)
            c_full[:, :n_ind] += sub_dt * div_flux
            c_full[:, -1] = 1.0 - np.sum(c_full[:, :n_ind], axis=1)

            remaining -= sub_dt
            total_substeps += 1

        t += dt
        if step % save_every == 0 or step == nsteps:
            snapshots.append(c_full.copy())
            t_saved.append(t)

    if total_substeps > nsteps:
        print(f"[RS-FDM] Adaptive sub-stepping: {total_substeps} sub-steps "
              f"for {nsteps} macro-steps (avg {total_substeps / nsteps:.1f}x)")

    return np.array(t_saved, dtype=float), np.stack(snapshots, axis=0)


# =============================================================================
# FDM teacher and uncached forward solver
# =============================================================================

def _run_fdm_teacher_core(
    log_d11: float,
    log_d22: float,
    rho_raw: float,
    t_max: float,
    nx: int,
    nt_save: int,
    rho21_raw: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Uncached conservative FDM for coupled ternary interdiffusion.

    This function is intentionally uncached and is used by likelihood/MCMC to
    avoid polluting Streamlit cache with thousands of random theta values.
    When rho21_raw is provided, builds a non-symmetric D matrix.
    """
    # Defensive: force all parameters to native Python types to prevent object dtype
    log_d11 = float(log_d11); log_d22 = float(log_d22); rho_raw = float(rho_raw)
    t_max = float(t_max); nx = int(nx); nt_save = int(nt_save)
    if rho21_raw is not None:
        rho21_raw = float(rho21_raw)
    if rho21_raw is not None:
        Dmat = make_nonsym_d_matrix_np(log_d11, log_d22, rho_raw, rho21_raw)
    else:
        Dmat = make_spd_matrix_np(log_d11, log_d22, rho_raw)

    eigmax = float(np.max(np.abs(np.linalg.eigvals(Dmat))))
    x = np.linspace(0.0, 1.0, nx)
    dx = x[1] - x[0]

    C0 = initial_diffusion_couple(x)
    U = C0[:, 1:3].copy()
    left_u = np.array([0.0, 0.0], dtype=float)
    right_u = np.array([0.90, 0.10], dtype=float)
    U[0] = left_u
    U[-1] = right_u

    dt_stable = float(0.18 * dx * dx / max(eigmax, 1.0e-14))
    n_steps = int(max(10, int(np.ceil(float(t_max) / dt_stable))))
    dt = float(t_max) / n_steps

    save_ids = set(np.unique(np.linspace(0.0, float(n_steps), max(2, int(nt_save))).astype(int)).tolist())
    t_save = []
    C_save = []

    for step in range(n_steps + 1):
        if step in save_ids:
            C = np.zeros((nx, 3), dtype=float)
            C[:, 1:3] = U
            C[:, 0] = 1.0 - C[:, 1] - C[:, 2]
            C = np.clip(C, 0.0, 1.0)
            C = C / np.maximum(C.sum(axis=1, keepdims=True), 1.0e-14)
            t_save.append(step * dt)
            C_save.append(C)

        if step == n_steps:
            break

        grad_half = (U[1:] - U[:-1]) / dx
        # J_i = -sum_j D_ij * dc_j/dx  →  flux = -grad @ D^T
        # For symmetric D, D^T = D; for non-symmetric D, the transpose is required.
        flux_half = -grad_half @ Dmat.T
        U_new = U.copy()
        U_new[1:-1] = U[1:-1] - dt * (flux_half[1:] - flux_half[:-1]) / dx
        U_new[0] = left_u
        U_new[-1] = right_u
        U_new = np.clip(U_new, 0.0, 1.0)

        total_solute = np.sum(U_new, axis=1)
        bad = total_solute > 0.999
        # Fix #2: exclude boundary nodes from clip — BC values are exact
        bad[0] = False
        bad[-1] = False
        if np.any(bad):
            U_new[bad] = U_new[bad] / total_solute[bad, None] * 0.999
            _DIAG_COUNTERS["fdm_clip_events"] += 1
            _DIAG_COUNTERS["fdm_clip_max_delta"] = max(
                _DIAG_COUNTERS["fdm_clip_max_delta"],
                float(np.max(total_solute[bad] - 0.999)))

        U = U_new

    return x, np.asarray(t_save), np.stack(C_save, axis=0), Dmat


def smooth_step_indicator_np(x: np.ndarray, interface: float = 0.5, width: float = 0.02) -> np.ndarray:
    """Smooth left-to-right indicator used for left/right D blending."""
    x = np.asarray(x, dtype=float)
    w = max(float(width), 1.0e-8)
    return 0.5 * (1.0 + np.tanh((x - float(interface)) / w))


def _run_fdm_teacher_core_two_region(
    log_d11_left: float,
    log_d22_left: float,
    rho_raw_left: float,
    log_d11_right: float,
    log_d22_right: float,
    rho_raw_right: float,
    t_max: float,
    nx: int,
    nt_save: int,
    phase_width: float = 0.02,
    rho21_raw_left: Optional[float] = None,
    rho21_raw_right: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Uncached conservative FDM with smoothly blended left/right diffusion matrices."""
    # Defensive: force all parameters to native Python types to prevent object dtype
    log_d11_left = float(log_d11_left); log_d22_left = float(log_d22_left); rho_raw_left = float(rho_raw_left)
    log_d11_right = float(log_d11_right); log_d22_right = float(log_d22_right); rho_raw_right = float(rho_raw_right)
    t_max = float(t_max); nx = int(nx); nt_save = int(nt_save); phase_width = float(phase_width)
    if rho21_raw_left is not None:
        rho21_raw_left = float(rho21_raw_left)
    if rho21_raw_right is not None:
        rho21_raw_right = float(rho21_raw_right)
    if rho21_raw_left is not None:
        D_left = make_nonsym_d_matrix_np(log_d11_left, log_d22_left, rho_raw_left, rho21_raw_left)
    else:
        D_left = make_spd_matrix_np(log_d11_left, log_d22_left, rho_raw_left)
    if rho21_raw_right is not None:
        D_right = make_nonsym_d_matrix_np(log_d11_right, log_d22_right, rho_raw_right, rho21_raw_right)
    else:
        D_right = make_spd_matrix_np(log_d11_right, log_d22_right, rho_raw_right)
    D_avg = 0.5 * (D_left + D_right)

    eigmax = float(max(np.max(np.abs(np.linalg.eigvals(D_left))),
                       np.max(np.abs(np.linalg.eigvals(D_right)))))
    x = np.linspace(0.0, 1.0, nx)
    dx = x[1] - x[0]

    C0 = initial_diffusion_couple(x)
    U = C0[:, 1:3].copy()
    left_u = np.array([0.0, 0.0], dtype=float)
    right_u = np.array([0.90, 0.10], dtype=float)
    U[0] = left_u
    U[-1] = right_u

    x_half = 0.5 * (x[1:] + x[:-1])
    s_half = smooth_step_indicator_np(x_half, interface=0.5, width=float(phase_width))
    D_half = np.empty((len(x_half), 2, 2), dtype=float)
    for k, s in enumerate(s_half):
        D_half[k] = (1.0 - s) * D_left + s * D_right

    dt_stable = float(0.18 * dx * dx / max(eigmax, 1.0e-14))
    n_steps = int(max(10, int(np.ceil(float(t_max) / dt_stable))))
    dt = float(t_max) / n_steps

    save_ids = set(np.unique(np.linspace(0.0, float(n_steps), max(2, int(nt_save))).astype(int)).tolist())
    t_save = []
    C_save = []

    for step in range(n_steps + 1):
        if step in save_ids:
            C = np.zeros((nx, 3), dtype=float)
            C[:, 1:3] = U
            C[:, 0] = 1.0 - C[:, 1] - C[:, 2]
            C = np.clip(C, 0.0, 1.0)
            C = C / np.maximum(C.sum(axis=1, keepdims=True), 1.0e-14)
            t_save.append(step * dt)
            C_save.append(C)

        if step == n_steps:
            break

        grad_half = (U[1:] - U[:-1]) / dx
        flux_half = np.empty_like(grad_half)
        for k in range(len(grad_half)):
            flux_half[k] = -grad_half[k] @ D_half[k].T

        U_new = U.copy()
        U_new[1:-1] = U[1:-1] - dt * (flux_half[1:] - flux_half[:-1]) / dx
        U_new[0] = left_u
        U_new[-1] = right_u
        U_new = np.clip(U_new, 0.0, 1.0)

        total_solute = np.sum(U_new, axis=1)
        bad = total_solute > 0.999
        # Fix #2: exclude boundary nodes from clip — BC values are exact
        bad[0] = False
        bad[-1] = False
        if np.any(bad):
            U_new[bad] = U_new[bad] / total_solute[bad, None] * 0.999
            _DIAG_COUNTERS["fdm_clip_events"] += 1
            _DIAG_COUNTERS["fdm_clip_max_delta"] = max(
                _DIAG_COUNTERS["fdm_clip_max_delta"],
                float(np.max(total_solute[bad] - 0.999)))
        U = U_new

    return x, np.asarray(t_save), np.stack(C_save, axis=0), D_avg, D_left, D_right


@st.cache_data(show_spinner=False, max_entries=64)
def run_fdm_teacher_two_region(
    log_d11_left: float,
    log_d22_left: float,
    rho_raw_left: float,
    log_d11_right: float,
    log_d22_right: float,
    rho_raw_right: float,
    t_max: float,
    nx: int,
    nt_save: int,
    phase_width: float,
    rho21_raw_left: Optional[float] = None,
    rho21_raw_right: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cached two-region FDM teacher/profile call."""
    return _run_fdm_teacher_core_two_region(
        log_d11_left,
        log_d22_left,
        rho_raw_left,
        log_d11_right,
        log_d22_right,
        rho_raw_right,
        t_max,
        nx,
        nt_save,
        phase_width,
        rho21_raw_left,
        rho21_raw_right,
    )


@st.cache_data(show_spinner=False, max_entries=64)
def run_fdm_teacher(
    log_d11: float,
    log_d22: float,
    rho_raw: float,
    t_max: float,
    nx: int,
    nt_save: int,
    rho21_raw: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cached FDM only for deterministic teacher/profile calls."""
    return _run_fdm_teacher_core(log_d11, log_d22, rho_raw, t_max, nx, nt_save, rho21_raw)


def bilinear_sample_xt_legacy(
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    C: np.ndarray,
    xq: np.ndarray,
    tq: np.ndarray,
) -> np.ndarray:
    """Bilinear interpolation in x,t (original Python-loop version, N2 legacy)."""
    xq = xq.ravel()
    tq = tq.ravel()
    out = np.empty((len(xq), 3), dtype=float)

    for k, (xx, tt) in enumerate(zip(xq, tq)):
        ix = int(np.clip(np.searchsorted(x_grid, xx) - 1, 0, len(x_grid) - 2))
        it = int(np.clip(np.searchsorted(t_grid, tt) - 1, 0, len(t_grid) - 2))
        x0, x1 = x_grid[ix], x_grid[ix + 1]
        t0, t1 = t_grid[it], t_grid[it + 1]
        wx = 0.0 if x1 == x0 else (xx - x0) / (x1 - x0)
        wt = 0.0 if t1 == t0 else (tt - t0) / (t1 - t0)
        out[k] = (
            (1 - wx) * (1 - wt) * C[it, ix]
            + wx * (1 - wt) * C[it, ix + 1]
            + (1 - wx) * wt * C[it + 1, ix]
            + wx * wt * C[it + 1, ix + 1]
        )

    return out


def bilinear_sample_xt(
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    C: np.ndarray,
    xq: np.ndarray,
    tq: np.ndarray,
) -> np.ndarray:
    """Vectorized bilinear interpolation in x,t for composition array (N2)."""
    xq = np.asarray(xq).ravel()
    tq = np.asarray(tq).ravel()

    ix = np.clip(np.searchsorted(x_grid, xq) - 1, 0, len(x_grid) - 2)
    it = np.clip(np.searchsorted(t_grid, tq) - 1, 0, len(t_grid) - 2)

    x0 = x_grid[ix]
    x1 = x_grid[ix + 1]
    t0 = t_grid[it]
    t1 = t_grid[it + 1]

    dx = x1 - x0
    dt_arr = t1 - t0
    wx = np.where(dx > 0, (xq - x0) / dx, 0.0)
    wt = np.where(dt_arr > 0, (tq - t0) / dt_arr, 0.0)

    wx = wx[:, None]
    wt = wt[:, None]

    out = (
        (1 - wx) * (1 - wt) * C[it, ix]
        + wx * (1 - wt) * C[it, ix + 1]
        + (1 - wx) * wt * C[it + 1, ix]
        + wx * wt * C[it + 1, ix + 1]
    )
    return out


def _alr_forward(c: np.ndarray, ref: int = 0) -> np.ndarray:
    """Additive log-ratio transform: c (N, K) → alr (N, K-1)."""
    c_safe = np.maximum(c, 1.0e-14)
    return np.log(np.delete(c_safe, ref, axis=1) / c_safe[:, ref:ref + 1])


def _alr_inverse(alr: np.ndarray, ref: int = 0) -> np.ndarray:
    """Inverse ALR: alr (N, K-1) → c (N, K) on simplex."""
    exp_alr = np.exp(alr)
    denom = 1.0 + exp_alr.sum(axis=1, keepdims=True)
    c_non_ref = exp_alr / denom
    c_ref = 1.0 / denom
    return np.insert(c_non_ref, ref, c_ref.ravel(), axis=1)


def make_pseudo_experiment(
    x: np.ndarray,
    C_final: np.ndarray,
    noise: float,
    seed: int,
    n_points: int = 64,
    noise_model: str = "gaussian",
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate pseudo experimental points from the final FDM profile.

    If noise=0, the pseudo experimental data are exactly on the FDM final profile.

    P8 fix: ``noise_model`` controls the noise generation:
    - "gaussian" (default): add Gaussian noise in composition space, then
      simplex-project.  Fast and backward-compatible but misspecified.
    - "alr": add Gaussian noise in ALR (additive log-ratio) space, then
      transform back.  Self-consistent with independent Gaussian assumption
      in ALR-transformed residuals.
    """
    rng = np.random.default_rng(seed + 1120)
    x_exp = np.linspace(float(x.min()), float(x.max()), n_points)
    c_clean = np.column_stack([np.interp(x_exp, x, C_final[:, j]) for j in range(3)])
    if noise_model == "alr" and float(noise) > 0.0:
        alr_clean = _alr_forward(c_clean, ref=0)
        alr_noisy = alr_clean + rng.normal(0.0, float(noise), size=alr_clean.shape)
        noisy = _alr_inverse(alr_noisy, ref=0)
    else:
        noisy = c_clean + rng.normal(0.0, float(noise), size=c_clean.shape)
        noisy = np.clip(noisy, 0.0, 1.0)
        noisy = noisy / np.maximum(noisy.sum(axis=1, keepdims=True), 1.0e-14)
    return x_exp, noisy


def make_pseudo_experiment_multitime(
    x: np.ndarray,
    t_grid: np.ndarray,
    C_fdm: np.ndarray,
    noise: float,
    seed: int,
    n_points_per_time: int = 64,
    n_time_slices: int = 4,
    t_start: float = 0.0,
    noise_model: str = "gaussian",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate pseudo experimental points from multiple FDM time slices.

    P8 fix: ``noise_model="alr"`` adds noise in ALR space for
    self-consistent likelihood evaluation.
    """
    rng = np.random.default_rng(seed + 2219)
    valid = np.where(np.asarray(t_grid, dtype=float) >= float(t_start))[0]
    if len(valid) == 0:
        valid = np.arange(len(t_grid))
    idxs = np.unique(np.linspace(valid[0], valid[-1], int(max(n_time_slices, 1))).astype(int))
    if idxs[-1] != len(t_grid) - 1:
        idxs = np.unique(np.append(idxs, len(t_grid) - 1)).astype(int)

    x_base = np.linspace(float(x.min()), float(x.max()), int(n_points_per_time))
    x_list, t_list, c_list = [], [], []
    for idx in idxs:
        c_clean = np.column_stack([np.interp(x_base, x, C_fdm[idx, :, j]) for j in range(3)])
        if noise_model == "alr" and float(noise) > 0.0:
            alr_clean = _alr_forward(c_clean, ref=0)
            alr_noisy = alr_clean + rng.normal(0.0, float(noise), size=alr_clean.shape)
            noisy = _alr_inverse(alr_noisy, ref=0)
        else:
            noisy = c_clean + rng.normal(0.0, float(noise), size=c_clean.shape)
            noisy = np.clip(noisy, 0.0, 1.0)
            noisy = noisy / np.maximum(noisy.sum(axis=1, keepdims=True), 1.0e-14)
        x_list.append(x_base.reshape(-1, 1))
        t_list.append(np.full((len(x_base), 1), float(t_grid[idx])))
        c_list.append(noisy)

    return np.vstack(x_list), np.vstack(t_list), np.vstack(c_list), idxs


@dataclass
class TrainingData:
    x_obs: np.ndarray
    t_obs: np.ndarray
    c_obs: np.ndarray
    x_ic: np.ndarray
    t_ic: np.ndarray
    c_ic: np.ndarray
    x_bc: np.ndarray
    t_bc: np.ndarray
    c_bc: np.ndarray
    x_f: np.ndarray
    t_f: np.ndarray
    x_grid: np.ndarray
    t_grid: np.ndarray
    C_fdm: np.ndarray
    D_true: np.ndarray
    D_true_left: Optional[np.ndarray]
    D_true_right: Optional[np.ndarray]
    t_start: float
    x_exp: np.ndarray
    c_exp: np.ndarray
    x_exp_all: np.ndarray
    t_exp_all: np.ndarray
    c_exp_all: np.ndarray
    exp_time_indices: np.ndarray
    rs_t_max_physical: float = 0.0


def make_training_data(
    log_d11: float,
    log_d22: float,
    rho_raw: float,
    t_max: float,
    nx_fdm: int,
    nt_fdm: int,
    n_obs: int,
    n_ic: int,
    n_bc_each: int,
    n_f: int,
    noise: float,
    seed: int,
    t_start_fraction: float,
    n_exp_points: int,
    pseudo_exp_time_mode: str = "final only",
    pseudo_exp_time_slices: int = 4,
    append_pseudo_exp_to_training: bool = True,
    fdm_teacher_mode: str = "single D",
    log_d11_right: Optional[float] = None,
    log_d22_right: Optional[float] = None,
    rho_raw_right: Optional[float] = None,
    phase_width: float = 0.02,
    rho21_raw: Optional[float] = None,
    rho21_raw_right: Optional[float] = None,
    noise_model: str = "gaussian",
) -> TrainingData:
    rng = np.random.default_rng(seed)
    D_true_left = None
    D_true_right = None
    if str(fdm_teacher_mode).lower().startswith("left/right"):
        x_grid, t_grid, C_fdm, D_true, D_true_left, D_true_right = run_fdm_teacher_two_region(
            log_d11,
            log_d22,
            rho_raw,
            float(log_d11_right if log_d11_right is not None else log_d11),
            float(log_d22_right if log_d22_right is not None else log_d22),
            float(rho_raw_right if rho_raw_right is not None else rho_raw),
            t_max,
            nx_fdm,
            nt_fdm,
            float(phase_width),
            rho21_raw_left=rho21_raw,
            rho21_raw_right=rho21_raw_right,
        )
    else:
        x_grid, t_grid, C_fdm, D_true = run_fdm_teacher(
            log_d11, log_d22, rho_raw, t_max, nx_fdm, nt_fdm, rho21_raw=rho21_raw
        )
        D_true_left = D_true
        D_true_right = D_true
    # C16 note: t_grid[1] depends on nt_fdm and can vary substantially.
    # Using it as a floor means t_start may be unexpectedly large for coarse grids.
    t_start = max(float(t_start_fraction * t_max), float(t_grid[1]))

    x_obs = rng.uniform(0.02, 0.98, size=(n_obs, 1))
    t_obs = rng.uniform(t_start, t_max, size=(n_obs, 1))
    c_clean = bilinear_sample_xt(x_grid, t_grid, C_fdm, x_obs, t_obs)
    if noise_model == "alr" and float(noise) > 0.0:
        alr_clean = _alr_forward(c_clean, ref=0)
        alr_noisy = alr_clean + rng.normal(0.0, float(noise), size=alr_clean.shape)
        c_obs = _alr_inverse(alr_noisy, ref=0)
    else:
        c_obs = np.clip(c_clean + rng.normal(0.0, noise, size=c_clean.shape), 0.0, 1.0)
        c_obs = c_obs / np.maximum(c_obs.sum(axis=1, keepdims=True), 1.0e-14)

    x_ic = rng.uniform(0.0, 1.0, size=(n_ic, 1))
    t_ic = np.full_like(x_ic, t_start)
    c_ic = bilinear_sample_xt(x_grid, t_grid, C_fdm, x_ic, t_ic)

    t_left = rng.uniform(t_start, t_max, size=(n_bc_each, 1))
    t_right = rng.uniform(t_start, t_max, size=(n_bc_each, 1))
    x_bc = np.vstack([np.zeros_like(t_left), np.ones_like(t_right)])
    t_bc = np.vstack([t_left, t_right])
    c_bc = np.vstack(
        [
            np.tile(np.array([[1.0, 0.0, 0.0]]), (n_bc_each, 1)),
            np.tile(np.array([[0.0, 0.90, 0.10]]), (n_bc_each, 1)),
        ]
    )

    x_f = rng.uniform(0.0, 1.0, size=(n_f, 1))
    t_f = rng.uniform(t_start, t_max, size=(n_f, 1))

    x_exp, c_exp = make_pseudo_experiment(
        x_grid, C_fdm[-1], noise=noise, seed=seed, n_points=n_exp_points,
        noise_model=noise_model,
    )

    if str(pseudo_exp_time_mode).lower().startswith("multi"):
        x_exp_all, t_exp_all, c_exp_all, exp_time_indices = make_pseudo_experiment_multitime(
            x_grid,
            t_grid,
            C_fdm,
            noise=noise,
            seed=seed,
            n_points_per_time=n_exp_points,
            n_time_slices=int(pseudo_exp_time_slices),
            t_start=t_start,
            noise_model=noise_model,
        )
    else:
        x_exp_all = x_exp.reshape(-1, 1)
        t_exp_all = np.full((len(x_exp), 1), float(t_grid[-1]))
        c_exp_all = c_exp
        exp_time_indices = np.array([len(t_grid) - 1], dtype=int)

    if bool(append_pseudo_exp_to_training):
        x_obs = np.vstack([x_obs, x_exp_all])
        t_obs = np.vstack([t_obs, t_exp_all])
        c_obs = np.vstack([c_obs, c_exp_all])

    return TrainingData(
        x_obs=x_obs,
        t_obs=t_obs,
        c_obs=c_obs,
        x_ic=x_ic,
        t_ic=t_ic,
        c_ic=c_ic,
        x_bc=x_bc,
        t_bc=t_bc,
        c_bc=c_bc,
        x_f=x_f,
        t_f=t_f,
        x_grid=x_grid,
        t_grid=t_grid,
        C_fdm=C_fdm,
        D_true=D_true,
        D_true_left=D_true_left,
        D_true_right=D_true_right,
        t_start=t_start,
        x_exp=x_exp,
        c_exp=c_exp,
        x_exp_all=x_exp_all,
        t_exp_all=t_exp_all,
        c_exp_all=c_exp_all,
        exp_time_indices=exp_time_indices,
    )


def make_training_data_rs(
    theta_left: np.ndarray,
    theta_right: np.ndarray,
    mobility: np.ndarray,
    RT: float,
    x_interface: float,
    omega_width: float,
    phase_width: float,
    dt: float,
    nsteps: int,
    save_every: int,
    nx_fdm: int,
    n_obs: int,
    n_ic: int,
    n_bc_each: int,
    n_f: int,
    noise: float,
    seed: int,
    t_start_fraction: float,
    n_exp_points: int,
    pseudo_exp_time_mode: str = "final only",
    pseudo_exp_time_slices: int = 4,
    append_pseudo_exp_to_training: bool = True,
    learn_lr_omega: bool = False,
    noise_model: str = "gaussian",
    log_M_endmembers: Optional[np.ndarray] = None,
) -> TrainingData:
    """Generate training data using the RS (chemical-potential) FDM solver.

    This ensures model consistency: when the PINN uses the RS PDE
    (c_t = div(M grad(mu))), the teacher data is also generated from
    the same RS model — not from a Fickian D-matrix FDM.

    theta_left / theta_right are in **display** order [CoNi, CoTa, NiTa].
    Internally reordered to [NiTa, NiCo, TaCo] for fdm_ternary_regular_solution.
    """
    rng = np.random.default_rng(seed)

    theta_left_int = _reorder_theta_display_to_internal(theta_left)
    theta_right_int = _reorder_theta_display_to_internal(theta_right)

    x_grid = np.linspace(0.0, 1.0, nx_fdm)

    # Initial profile in internal order [Ni, Ta, Co]
    # eps_guard prevents log(0) in chemical potential; 5e-3 avoids
    # extreme mu values at boundaries while keeping profiles sharp.
    eps_guard = 5.0e-3
    c_left_disp = np.array([1.0 - eps_guard, eps_guard / 2, eps_guard / 2], dtype=float)
    c_left_disp /= c_left_disp.sum()
    c_right_disp = np.array([eps_guard / 2, 0.9, 0.1], dtype=float)
    c_right_disp /= c_right_disp.sum()

    c0_disp = make_initial_profile_ternary_rs(
        x_grid, c_left_disp, c_right_disp,
        x0=x_interface, width=float(phase_width),
    )
    c0_int = _reorder_c_display_to_internal_np(c0_disp)

    t_grid_rs, C_fdm_int = fdm_ternary_regular_solution(
        c0_int, x_grid, dt, nsteps, mobility,
        theta_left_int, theta_right_int,
        RT=RT, x_interface=x_interface, omega_width=omega_width,
        save_every=save_every,
        log_M_endmembers=log_M_endmembers,
    )

    C_fdm = _reorder_c_internal_to_display_np(C_fdm_int)

    # --- Time normalization for PINN input ---
    # RS FDM produces small t_max (e.g. 0.04) while x ∈ [0, 1].
    # Without normalization the network cannot distinguish different times.
    # Normalize t to [0, 1]; compensate in train_pinn_rs by M_eff = M * t_max_physical.
    t_max_physical = float(t_grid_rs[-1])
    if t_max_physical > 0:
        t_grid_rs = t_grid_rs / t_max_physical
    t_max = float(t_grid_rs[-1])  # = 1.0 after normalization
    t_start = max(float(t_start_fraction * t_max), float(t_grid_rs[1]) if len(t_grid_rs) > 1 else 0.0)

    x_obs = rng.uniform(0.02, 0.98, size=(n_obs, 1))
    t_obs = rng.uniform(t_start, t_max, size=(n_obs, 1))
    c_clean = bilinear_sample_xt(x_grid, t_grid_rs, C_fdm, x_obs, t_obs)
    c_obs = np.clip(c_clean + rng.normal(0.0, noise, size=c_clean.shape), 0.0, 1.0)
    c_obs = c_obs / np.maximum(c_obs.sum(axis=1, keepdims=True), 1.0e-14)

    x_ic = rng.uniform(0.0, 1.0, size=(n_ic, 1))
    t_ic = np.full_like(x_ic, t_start)
    c_ic = bilinear_sample_xt(x_grid, t_grid_rs, C_fdm, x_ic, t_ic)

    # RS FDM uses closed-system (zero-flux Neumann) BC, matching DICTRA.
    # No Dirichlet BC loss for the PINN — the physics PDE residual and
    # data/IC losses are sufficient to constrain the solution.
    # Empty BC arrays → has_bc=False in train_pinn_rs → BC loss skipped.
    n_ind_disp = c_left_disp.shape[0]
    x_bc = np.zeros((0, 1), dtype=float)
    t_bc = np.zeros((0, 1), dtype=float)
    c_bc = np.zeros((0, n_ind_disp), dtype=float)

    x_f = rng.uniform(0.0, 1.0, size=(n_f, 1))
    t_f = rng.uniform(t_start, t_max, size=(n_f, 1))

    x_exp, c_exp = make_pseudo_experiment(
        x_grid, C_fdm[-1], noise=noise, seed=seed, n_points=n_exp_points,
        noise_model=noise_model,
    )

    if str(pseudo_exp_time_mode).lower().startswith("multi"):
        x_exp_all, t_exp_all, c_exp_all, exp_time_indices = make_pseudo_experiment_multitime(
            x_grid, t_grid_rs, C_fdm,
            noise=noise, seed=seed,
            n_points_per_time=n_exp_points,
            n_time_slices=int(pseudo_exp_time_slices),
            t_start=t_start,
            noise_model=noise_model,
        )
    else:
        x_exp_all = x_exp.reshape(-1, 1)
        t_exp_all = np.full((len(x_exp), 1), float(t_grid_rs[-1]))
        c_exp_all = c_exp
        exp_time_indices = np.array([len(t_grid_rs) - 1], dtype=int)

    if bool(append_pseudo_exp_to_training):
        x_obs = np.vstack([x_obs, x_exp_all])
        t_obs = np.vstack([t_obs, t_exp_all])
        c_obs = np.vstack([c_obs, c_exp_all])

    return TrainingData(
        x_obs=x_obs,
        t_obs=t_obs,
        c_obs=c_obs,
        x_ic=x_ic,
        t_ic=t_ic,
        c_ic=c_ic,
        x_bc=x_bc,
        t_bc=t_bc,
        c_bc=c_bc,
        x_f=x_f,
        t_f=t_f,
        x_grid=x_grid,
        t_grid=t_grid_rs,
        C_fdm=C_fdm,
        D_true=np.zeros((2, 2)),
        D_true_left=None,
        D_true_right=None,
        t_start=t_start,
        x_exp=x_exp,
        c_exp=c_exp,
        x_exp_all=x_exp_all,
        t_exp_all=t_exp_all,
        c_exp_all=c_exp_all,
        exp_time_indices=exp_time_indices,
        rs_t_max_physical=t_max_physical,
    )


# =============================================================================
# PINN model
# =============================================================================

class MLP(nn.Module):
    """Ternary composition network.

    ``direct_output=False`` (legacy): 3 outputs → softplus → normalize to simplex.
    ``direct_output=True``: 2 outputs [Ni, Ta] via sigmoid; Co = 1 - Ni - Ta.
    Direct mode avoids the normalization Jacobian leaking into PDE residuals.
    """

    def __init__(self, width: int, depth: int, activation: str, direct_output: bool = False):
        super().__init__()
        self.direct_output = direct_output
        out_dim = 2 if direct_output else 3

        def make_activation():
            if activation == "silu":
                return nn.SiLU()
            if activation == "gelu":
                return nn.GELU()
            return nn.Tanh()

        layers = []
        for i in range(depth):
            layers.append(nn.Linear(2 if i == 0 else width, width))
            layers.append(make_activation())
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raw = self.net(torch.cat([x, t], dim=1))
        if self.direct_output:
            # Fix #1: simplex-safe projection via clamp-based rescale.
            # sigmoid → [0,1] per component, then rescale if Ni+Ta > 1.
            ni_ta = torch.sigmoid(raw)
            total = ni_ta.sum(dim=1, keepdim=True)
            ni_ta = ni_ta / torch.clamp(total, min=1.0)
            co = torch.clamp(1.0 - ni_ta.sum(dim=1, keepdim=True), min=0.0)
            return torch.cat([co, ni_ta], dim=1)
        else:
            positive = F.softplus(raw) + 1.0e-8
            return positive / torch.sum(positive, dim=1, keepdim=True)


class TernaryDiffusionPINN(nn.Module):
    def __init__(
        self,
        log_d11_init: float,
        log_d22_init: float,
        rho_raw_init: float,
        width: int,
        depth: int,
        activation: str,
        rho21_raw_init: Optional[float] = None,
        force_symmetric: bool = FORCE_SYMMETRIC_D,
        direct_output: bool = False,
    ):
        super().__init__()
        self.net = MLP(width, depth, activation, direct_output=direct_output)
        self.force_symmetric = force_symmetric
        self.log_d11 = nn.Parameter(torch.tensor([log_d11_init], dtype=DTYPE, device=DEVICE))
        self.log_d22 = nn.Parameter(torch.tensor([log_d22_init], dtype=DTYPE, device=DEVICE))
        self.rho_raw = nn.Parameter(torch.tensor([rho_raw_init], dtype=DTYPE, device=DEVICE))
        if not force_symmetric:
            rho21_val = rho21_raw_init if rho21_raw_init is not None else rho_raw_init
            self.rho21_raw = nn.Parameter(torch.tensor([rho21_val], dtype=DTYPE, device=DEVICE))
        else:
            self.rho21_raw = None

    def diffusion_matrix(self) -> torch.Tensor:
        d11 = torch.exp(self.log_d11[0])
        d22 = torch.exp(self.log_d22[0])
        scale = torch.sqrt(d11 * d22)
        rho12 = 0.95 * torch.tanh(self.rho_raw[0])
        d12 = rho12 * scale
        if self.force_symmetric or self.rho21_raw is None:
            d21 = d12
        else:
            rho21 = 0.95 * torch.tanh(self.rho21_raw[0])
            d21 = rho21 * scale
        return torch.stack([torch.stack([d11, d12]), torch.stack([d21, d22])])

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(x, t)

    def residual_train(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Training residual with full graph for backpropagation."""
        return self._residual_impl(x, t, second_derivative_graph=True, output_graph=True)

    def residual_eval(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Evaluation residual.

        Second derivatives still require the first x-derivative to carry a local
        graph, but the final derivative and outputs do not retain a persistent
        higher-order graph for parameter backpropagation.
        """
        return self._residual_impl(x, t, second_derivative_graph=True, output_graph=False)

    def _residual_impl(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        second_derivative_graph: bool,
        output_graph: bool,
    ) -> torch.Tensor:
        x = x.clone().detach().requires_grad_(True)
        t = t.clone().detach().requires_grad_(True)

        C = self.forward(x, t)
        ni = C[:, 1:2]
        ta = C[:, 2:3]

        ni_t = torch.autograd.grad(
            ni, t, torch.ones_like(ni), create_graph=output_graph, retain_graph=True
        )[0]
        ta_t = torch.autograd.grad(
            ta, t, torch.ones_like(ta), create_graph=output_graph, retain_graph=True
        )[0]

        ni_x = torch.autograd.grad(
            ni, x, torch.ones_like(ni), create_graph=second_derivative_graph, retain_graph=True
        )[0]
        ta_x = torch.autograd.grad(
            ta, x, torch.ones_like(ta), create_graph=second_derivative_graph, retain_graph=True
        )[0]

        D = self.diffusion_matrix()
        q_ni = D[0, 0] * ni_x + D[0, 1] * ta_x
        q_ta = D[1, 0] * ni_x + D[1, 1] * ta_x

        q_ni_x = torch.autograd.grad(
            q_ni, x, torch.ones_like(q_ni), create_graph=output_graph, retain_graph=True
        )[0]
        q_ta_x = torch.autograd.grad(
            q_ta, x, torch.ones_like(q_ta), create_graph=output_graph, retain_graph=True
        )[0]

        return torch.cat([ni_t - q_ni_x, ta_t - q_ta_x], dim=1)



class TwoRegionTernaryDiffusionPINN(TernaryDiffusionPINN):
    """PINNs model with left/right diffusion matrices smoothly blended at the interface.

    This is a fixed-interface approximation for diffusion couples whose two
    sides have different crystal structures or different diffusion kinetics:

        D(x) = (1 - s(x)) D_left + s(x) D_right
        s(x) = 0.5 * (1 + tanh((x - x_interface) / width))

    It is not a full DICTRA moving-boundary/local-equilibrium model, but it is
    useful as a practical two-region extension of the present PINNs prototype.
    """

    def __init__(
        self,
        log_d11_left_init: float,
        log_d22_left_init: float,
        rho_raw_left_init: float,
        log_d11_right_init: float,
        log_d22_right_init: float,
        rho_raw_right_init: float,
        width: int,
        depth: int,
        activation: str,
        phase_interface: float = 0.5,
        phase_width: float = 0.02,
        rho21_raw_left_init: Optional[float] = None,
        rho21_raw_right_init: Optional[float] = None,
        force_symmetric: bool = FORCE_SYMMETRIC_D,
        direct_output: bool = False,
    ):
        super().__init__(
            log_d11_left_init, log_d22_left_init, rho_raw_left_init,
            width, depth, activation,
            rho21_raw_init=rho21_raw_left_init, force_symmetric=force_symmetric,
            direct_output=direct_output,
        )
        self.log_d11_left = self.log_d11
        self.log_d22_left = self.log_d22
        self.rho_raw_left = self.rho_raw
        self.rho21_raw_left = self.rho21_raw  # None when force_symmetric=True
        self.log_d11_right = nn.Parameter(torch.tensor([float(log_d11_right_init)], dtype=DTYPE, device=DEVICE))
        self.log_d22_right = nn.Parameter(torch.tensor([float(log_d22_right_init)], dtype=DTYPE, device=DEVICE))
        self.rho_raw_right = nn.Parameter(torch.tensor([float(rho_raw_right_init)], dtype=DTYPE, device=DEVICE))
        if not force_symmetric:
            rho21_r = rho21_raw_right_init if rho21_raw_right_init is not None else rho_raw_right_init
            self.rho21_raw_right = nn.Parameter(torch.tensor([float(rho21_r)], dtype=DTYPE, device=DEVICE))
        else:
            self.rho21_raw_right = None
        self.phase_interface = float(phase_interface)
        self.phase_width = float(phase_width)

    def _matrix_from_params(
        self, log_d11: torch.Tensor, log_d22: torch.Tensor,
        rho_raw: torch.Tensor, rho21_raw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        d11 = torch.exp(log_d11[0])
        d22 = torch.exp(log_d22[0])
        scale = torch.sqrt(d11 * d22)
        rho12 = 0.95 * torch.tanh(rho_raw[0])
        d12 = rho12 * scale
        if self.force_symmetric or rho21_raw is None:
            d21 = d12
        else:
            rho21 = 0.95 * torch.tanh(rho21_raw[0])
            d21 = rho21 * scale
        return torch.stack([torch.stack([d11, d12]), torch.stack([d21, d22])])

    def diffusion_matrix_left(self) -> torch.Tensor:
        return self._matrix_from_params(self.log_d11_left, self.log_d22_left, self.rho_raw_left, self.rho21_raw_left)

    def diffusion_matrix_right(self) -> torch.Tensor:
        return self._matrix_from_params(self.log_d11_right, self.log_d22_right, self.rho_raw_right, self.rho21_raw_right)

    def diffusion_matrix(self) -> torch.Tensor:
        return 0.5 * (self.diffusion_matrix_left() + self.diffusion_matrix_right())

    def _residual_impl(self, x: torch.Tensor, t: torch.Tensor, second_derivative_graph: bool, output_graph: bool) -> torch.Tensor:
        x = x.clone().detach().requires_grad_(True)
        t = t.clone().detach().requires_grad_(True)

        C = self.forward(x, t)
        ni = C[:, 1:2]
        ta = C[:, 2:3]

        ni_t = torch.autograd.grad(ni, t, torch.ones_like(ni), create_graph=output_graph, retain_graph=True)[0]
        ta_t = torch.autograd.grad(ta, t, torch.ones_like(ta), create_graph=output_graph, retain_graph=True)[0]
        ni_x = torch.autograd.grad(ni, x, torch.ones_like(ni), create_graph=second_derivative_graph, retain_graph=True)[0]
        ta_x = torch.autograd.grad(ta, x, torch.ones_like(ta), create_graph=second_derivative_graph, retain_graph=True)[0]

        Dl = self.diffusion_matrix_left()
        Dr = self.diffusion_matrix_right()
        w = max(float(self.phase_width), 1.0e-8)
        s = 0.5 * (1.0 + torch.tanh((x - float(self.phase_interface)) / w))

        d11 = (1.0 - s) * Dl[0, 0] + s * Dr[0, 0]
        d12 = (1.0 - s) * Dl[0, 1] + s * Dr[0, 1]
        d21 = (1.0 - s) * Dl[1, 0] + s * Dr[1, 0]
        d22 = (1.0 - s) * Dl[1, 1] + s * Dr[1, 1]

        q_ni = d11 * ni_x + d12 * ta_x
        q_ta = d21 * ni_x + d22 * ta_x
        q_ni_x = torch.autograd.grad(q_ni, x, torch.ones_like(q_ni), create_graph=output_graph, retain_graph=True)[0]
        q_ta_x = torch.autograd.grad(q_ta, x, torch.ones_like(q_ta), create_graph=output_graph, retain_graph=True)[0]
        return torch.cat([ni_t - q_ni_x, ta_t - q_ta_x], dim=1)


# =============================================================================
# Regular-solution PINN model (chemical potential approach)
# =============================================================================

def _reorder_display_to_internal(c: torch.Tensor) -> torch.Tensor:
    """Reorder composition from display [Co,Ni,Ta] to internal [Ni,Ta,Co].

    The regular-solution chemical potential functions expect the dependent
    (reference) component to be the last column.  In the fig11 convention
    Co is dependent, so we move it to the end.
    """
    return torch.cat([c[:, 1:2], c[:, 2:3], c[:, 0:1]], dim=1)


def _reorder_theta_display_to_internal(theta: np.ndarray) -> np.ndarray:
    """Reorder Omega pair vector from display order to internal order.

    Display [Co=0,Ni=1,Ta=2] pairs:  (0,1)=CoNi, (0,2)=CoTa, (1,2)=NiTa
    Internal [Ni=0,Ta=1,Co=2] pairs: (0,1)=NiTa, (0,2)=NiCo, (1,2)=TaCo

    Omega is symmetric, so NiCo=CoNi and TaCo=CoTa.
    Mapping: internal[0]=NiTa=display[2], internal[1]=CoNi=display[0], internal[2]=CoTa=display[1]
    """
    return np.array([theta[2], theta[0], theta[1]], dtype=float)


def _reorder_theta_internal_to_display(theta: np.ndarray) -> np.ndarray:
    """Inverse of _reorder_theta_display_to_internal."""
    return np.array([theta[1], theta[2], theta[0]], dtype=float)


class TernaryRegularSolutionPINN(nn.Module):
    """Ternary PINN using regular-solution chemical potentials instead of direct D matrix.

    Uses the same MLP architecture as TernaryDiffusionPINN (fig11-equivalent).
    Output order is [Co, Ni, Ta] matching fig11; internally reorders to
    [Ni, Ta, Co] for chemical potential calculations (Co as reference).

    Trainable parameters: Omega pair-interaction terms stored in internal
    [Ni,Ta,Co] order.  The PDE is: c_t = div(M grad(mu)).
    """

    def __init__(
        self,
        width: int,
        depth: int,
        activation: str,
        theta_left_init: Optional[np.ndarray] = None,
        theta_right_init: Optional[np.ndarray] = None,
        learn_left_right_omega: bool = True,
        x_interface: float = 0.5,
        omega_width: float = 0.02,
        RT: float = 1.0,
        train_omega: bool = True,
        log_M_endmembers_init: Optional[np.ndarray] = None,
        train_mobility: bool = False,
        direct_output: bool = False,
    ):
        super().__init__()
        self.n_components = 3
        self.n_ind = 2
        self.n_pairs = 3
        self.learn_left_right_omega = learn_left_right_omega
        self.x_interface = x_interface
        self.omega_width = omega_width
        self.RT = RT
        self.use_comp_dep_mobility = log_M_endmembers_init is not None

        self.net = MLP(width, depth, activation, direct_output=direct_output)

        if theta_left_init is None:
            theta_left_init = np.ones(self.n_pairs, dtype=float)
        if theta_right_init is None:
            theta_right_init = theta_left_init.copy()
        theta_left_int = _reorder_theta_display_to_internal(theta_left_init)
        theta_right_int = _reorder_theta_display_to_internal(theta_right_init)

        self.theta_left_raw = nn.Parameter(
            torch.tensor(theta_left_int, dtype=torch.float32, device=DEVICE),
            requires_grad=train_omega,
        )
        if learn_left_right_omega:
            self.theta_right_raw = nn.Parameter(
                torch.tensor(theta_right_int, dtype=torch.float32, device=DEVICE),
                requires_grad=train_omega,
            )
        else:
            self.register_buffer(
                "theta_right_raw",
                torch.tensor(theta_left_int, dtype=torch.float32, device=DEVICE),
            )

        if log_M_endmembers_init is not None:
            self.log_M_endmembers = nn.Parameter(
                torch.tensor(log_M_endmembers_init, dtype=torch.float32, device=DEVICE),
                requires_grad=train_mobility,
            )
        else:
            self.log_M_endmembers = None

    def theta_vectors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (theta_left, theta_right) in internal [Ni,Ta,Co] pair order."""
        if self.learn_left_right_omega:
            return self.theta_left_raw, self.theta_right_raw
        return self.theta_left_raw, self.theta_left_raw

    def theta_display(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (theta_left, theta_right) in display [Co,Ni,Ta] pair order."""
        tl, tr = self.theta_vectors()
        return (
            _reorder_theta_internal_to_display(tl.detach().cpu().numpy()),
            _reorder_theta_internal_to_display(tr.detach().cpu().numpy()),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return full composition [Co, Ni, Ta] (same as fig11 MLP)."""
        return self.net(x, t)

    def residual_train(self, x: torch.Tensor, t: torch.Tensor,
                       mobility: torch.Tensor, t_scale: float = 1.0) -> torch.Tensor:
        return self._residual_impl(x, t, mobility, t_scale=t_scale, second_derivative_graph=True, output_graph=True)

    def residual_eval(self, x: torch.Tensor, t: torch.Tensor,
                      mobility: torch.Tensor, t_scale: float = 1.0) -> torch.Tensor:
        return self._residual_impl(x, t, mobility, t_scale=t_scale, second_derivative_graph=True, output_graph=False)

    def _residual_impl(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        mobility: torch.Tensor,
        t_scale: float,
        second_derivative_graph: bool,
        output_graph: bool,
    ) -> torch.Tensor:
        """PDE residual: c_t - div(M grad(mu)) for independent components.

        Forward output is [Co, Ni, Ta].  For chemical potential computation
        we reorder to [Ni, Ta, Co] so that Co (dependent) is the reference
        component (index 2) as expected by diffusion_potentials_regular_solution_torch.

        If ``self.use_comp_dep_mobility`` is True, M(c) is computed from
        end-member log-mobilities at each collocation point.  Otherwise the
        constant ``mobility`` tensor is used.
        """
        x = x.clone().detach().requires_grad_(True)
        t = t.clone().detach().requires_grad_(True)

        C = self.forward(x, t)
        C_int = _reorder_display_to_internal(C)

        theta_l, theta_r = self.theta_vectors()
        mu = diffusion_potentials_regular_solution_torch(
            C_int, x, theta_l, theta_r, RT=self.RT,
            x_interface=self.x_interface, width=self.omega_width,
        )

        ind0 = C[:, 1:2]  # first independent component
        ind1 = C[:, 2:3]  # second independent component

        ind0_t = torch.autograd.grad(ind0, t, torch.ones_like(ind0), create_graph=output_graph, retain_graph=True)[0]
        ind1_t = torch.autograd.grad(ind1, t, torch.ones_like(ind1), create_graph=output_graph, retain_graph=True)[0]

        mu0 = mu[:, 0:1]
        mu1 = mu[:, 1:2]

        mu0_x = torch.autograd.grad(mu0, x, torch.ones_like(mu0), create_graph=second_derivative_graph, retain_graph=True)[0]
        mu1_x = torch.autograd.grad(mu1, x, torch.ones_like(mu1), create_graph=second_derivative_graph, retain_graph=True)[0]

        if self.use_comp_dep_mobility and self.log_M_endmembers is not None:
            M_local = mobility_matrix_from_endmembers_torch(C_int, self.log_M_endmembers)
            q0 = t_scale * (M_local[:, 0, 0:1] * mu0_x + M_local[:, 0, 1:2] * mu1_x)
            q1 = t_scale * (M_local[:, 1, 0:1] * mu0_x + M_local[:, 1, 1:2] * mu1_x)
        else:
            q0 = mobility[0, 0] * mu0_x + mobility[0, 1] * mu1_x
            q1 = mobility[1, 0] * mu0_x + mobility[1, 1] * mu1_x

        q0_x = torch.autograd.grad(q0, x, torch.ones_like(q0), create_graph=output_graph, retain_graph=True)[0]
        q1_x = torch.autograd.grad(q1, x, torch.ones_like(q1), create_graph=output_graph, retain_graph=True)[0]

        return torch.cat([ind0_t - q0_x, ind1_t - q1_x], dim=1)


@dataclass
class TrainResultRS:
    """Training result for regular-solution mode."""
    model: TernaryRegularSolutionPINN
    data: TrainingData
    history: pd.DataFrame
    train_time: float
    mobility: np.ndarray


def train_pinn_rs(
    data: TrainingData,
    model: TernaryRegularSolutionPINN,
    mobility: np.ndarray,
    epochs: int,
    lr: float,
    weights: Dict[str, float],
    progress=None,
    status=None,
    n_collocation: int = 2000,
    w_omega_prior: float = 0.0,
    omega_prior_left: Optional[np.ndarray] = None,
    omega_prior_right: Optional[np.ndarray] = None,
    adaptive_weights: bool = False,
    rba_update_every: int = 50,
    compile_model: bool = False,
) -> Tuple[TernaryRegularSolutionPINN, pd.DataFrame]:
    """Train a TernaryRegularSolutionPINN model using fig11-standard TrainingData."""
    device = DEVICE
    model = model.to(device)
    if compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)
    # Time normalization: training data uses τ = t/t_max ∈ [0,1].
    # PDE in normalized time: dc/dτ = t_max * div(M ∇μ).
    # Effective mobility for the PINN residual: M_eff = M * t_max_physical.
    t_scale = float(data.rs_t_max_physical) if data.rs_t_max_physical > 0 else 1.0
    mobility_scaled = mobility * t_scale
    mobility_t = torch.tensor(mobility_scaled, dtype=torch.float32, device=device)

    x_obs = to_tensor(data.x_obs.reshape(-1, 1)).to(device)
    t_obs = to_tensor(data.t_obs.reshape(-1, 1)).to(device)
    c_obs = to_tensor(data.c_obs).to(device)
    x_ic = to_tensor(data.x_ic.reshape(-1, 1)).to(device)
    t_ic = to_tensor(data.t_ic.reshape(-1, 1)).to(device)
    c_ic = to_tensor(data.c_ic).to(device)
    # P7 fix: load BC data for explicit boundary enforcement
    x_bc = to_tensor(data.x_bc.reshape(-1, 1)).to(device)
    t_bc = to_tensor(data.t_bc.reshape(-1, 1)).to(device)
    c_bc = to_tensor(data.c_bc).to(device)
    has_bc = x_bc.shape[0] > 0

    w_data = float(weights.get("data", 25.0))
    w_ic = float(weights.get("ic", 12.0))
    w_bc = float(weights.get("bc", 12.0))
    w_phys = float(weights.get("phys", 10.0))

    omega_prior_left_int = _reorder_theta_display_to_internal(omega_prior_left) if omega_prior_left is not None else None
    omega_prior_right_int = _reorder_theta_display_to_internal(omega_prior_right) if omega_prior_right is not None else None

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=lr * 0.03)

    history_rows = []
    rng = np.random.default_rng(42)
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        pred_obs = model(x_obs, t_obs)
        loss_data = F.mse_loss(pred_obs, c_obs)

        pred_ic = model(x_ic, t_ic)
        loss_ic = F.mse_loss(pred_ic, c_ic)

        x_col = torch.tensor(
            rng.uniform(float(data.x_grid[0]), float(data.x_grid[-1]), size=(n_collocation, 1)),
            dtype=torch.float32, device=device,
        )
        t_col = torch.tensor(
            rng.uniform(float(data.t_grid[0]), float(data.t_grid[-1]), size=(n_collocation, 1)),
            dtype=torch.float32, device=device,
        )
        res = model.residual_train(x_col, t_col, mobility_t, t_scale=t_scale)
        loss_phys = torch.mean(res ** 2)

        # P7 fix: explicit BC enforcement
        loss_bc_val = torch.tensor(0.0, device=device)
        if has_bc:
            pred_bc = model(x_bc, t_bc)
            loss_bc_val = F.mse_loss(pred_bc, c_bc)

        # RBA: rebalance weights for RS training
        # C5 fix: on RBA epochs, reuse per-loss gradients to accumulate the
        # weighted total gradient directly, eliminating the (N+1)th backward pass.
        _rba_epoch = adaptive_weights and epoch > 1 and epoch % rba_update_every == 0
        if _rba_epoch:
            params = [p for p in model.parameters() if p.requires_grad]
            losses_rba = [loss_data, loss_ic, loss_phys]
            if has_bc:
                losses_rba.insert(2, loss_bc_val)
            all_grads = []
            gnorms = []
            for loss_i in losses_rba:
                grads = torch.autograd.grad(loss_i, params, retain_graph=True, allow_unused=True)
                all_grads.append(grads)
                gn = torch.sqrt(sum(g.norm() ** 2 for g in grads if g is not None))
                gnorms.append(max(float(gn), 1.0e-12))
            mean_gn = sum(gnorms) / len(gnorms)
            base_w_list = [weights.get("data", 25.0), weights.get("ic", 12.0)]
            if has_bc:
                base_w_list.append(weights.get("bc", 12.0))
            base_w_list.append(weights.get("phys", 10.0))
            new_w = [b * mean_gn / gn for b, gn in zip(base_w_list, gnorms)]
            if has_bc:
                w_data, w_ic, w_bc, w_phys = new_w
            else:
                w_data, w_ic, w_phys = new_w

        loss = w_data * loss_data + w_ic * loss_ic + w_phys * loss_phys
        if has_bc:
            loss = loss + w_bc * loss_bc_val

        if w_omega_prior > 0.0 and omega_prior_left_int is not None:
            theta_l, theta_r = model.theta_vectors()
            prior_l = torch.tensor(omega_prior_left_int, dtype=torch.float32, device=device)
            omega_reg = torch.mean((theta_l - prior_l) ** 2)
            if omega_prior_right_int is not None and model.learn_left_right_omega:
                prior_r = torch.tensor(omega_prior_right_int, dtype=torch.float32, device=device)
                omega_reg = omega_reg + torch.mean((theta_r - prior_r) ** 2)
            loss = loss + w_omega_prior * omega_reg

        # R1: bail early on non-finite loss to protect parameters.
        if not torch.isfinite(loss):
            if status is not None:
                status.text(
                    f"Training aborted at epoch {epoch}: non-finite loss "
                    f"(data={loss_data.item():.3e}, phys={loss_phys.item():.3e})."
                )
            break

        if _rba_epoch:
            # Accumulate weighted gradients from per-loss grads (no extra backward)
            optimizer.zero_grad()
            for p in params:
                if p.grad is None:
                    p.grad = torch.zeros_like(p.data)
            for w_i, grads_i in zip(new_w, all_grads):
                for p, g in zip(params, grads_i):
                    if g is not None:
                        p.grad.add_(w_i * g)
            # Add omega prior gradient separately
            if w_omega_prior > 0.0 and omega_prior_left_int is not None:
                omega_loss = w_omega_prior * omega_reg
                og = torch.autograd.grad(omega_loss, params, allow_unused=True)
                for p, g in zip(params, og):
                    if g is not None:
                        p.grad.add_(g)
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=10.0,
        )
        # --- Diagnostic: print loss breakdown & gradient norms at early epochs ---
        if epoch in (1, 2, 5, 20, 50, 100):
            _gn_net_rs, _gn_omega = 0.0, 0.0
            for _n, _p in model.named_parameters():
                if _p.grad is None:
                    continue
                _gnorm = float(_p.grad.norm())
                if "theta" in _n or "log_M" in _n:
                    _gn_omega += _gnorm ** 2
                elif "net." in _n:
                    _gn_net_rs += _gnorm ** 2
                else:
                    _gn_omega += _gnorm ** 2
            print(
                f"[RS-DIAG ep={epoch:3d}] loss={loss.item():.3e} | "
                f"d={loss_data.item():.3e} ic={loss_ic.item():.3e} "
                f"bc={loss_bc_val.item():.3e} phys={loss_phys.item():.3e} | "
                f"||g_net||={np.sqrt(_gn_net_rs):.3e} ||g_Ω||={np.sqrt(_gn_omega):.3e} | "
                f"lr={scheduler.get_last_lr()[0]:.2e}",
                flush=True,
            )
        # --- END DIAGNOSTIC ---

        optimizer.step()
        scheduler.step()

        theta_l_disp, theta_r_disp = model.theta_display()
        row = {
            "epoch": epoch,
            "loss": float(loss.item()),
            "data": float(loss_data.item()),
            "ic": float(loss_ic.item()),
            "bc": float(loss_bc_val.item()),
            "physics": float(loss_phys.item()),
            "Omega_CoNi_left": float(theta_l_disp[0]),
            "Omega_CoTa_left": float(theta_l_disp[1]),
            "Omega_NiTa_left": float(theta_l_disp[2]),
            "Omega_CoNi_right": float(theta_r_disp[0]),
            "Omega_CoTa_right": float(theta_r_disp[1]),
            "Omega_NiTa_right": float(theta_r_disp[2]),
        }
        history_rows.append(row)

        if progress is not None and epoch % max(1, epochs // 100) == 0:
            progress.progress(epoch / epochs)
        if status is not None and epoch % max(1, epochs // 20) == 0:
            status.text(
                f"Epoch {epoch}/{epochs}  loss={float(loss.item()):.4e}  "
                f"data={float(loss_data.item()):.4e}  phys={float(loss_phys.item()):.4e}"
            )

    train_time = time.time() - t0
    model.eval()
    if progress is not None:
        progress.progress(1.0)
    return model, pd.DataFrame(history_rows), train_time


def predict_rs(model: TernaryRegularSolutionPINN, x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Predict compositions using a regular-solution PINN."""
    model.eval()
    with torch.no_grad():
        xt = to_tensor(x.reshape(-1, 1)).to(next(model.parameters()).device)
        tt = to_tensor(t.reshape(-1, 1)).to(next(model.parameters()).device)
        return model(xt, tt).cpu().numpy()


def evaluate_model_on_grid_rs(
    model: TernaryRegularSolutionPINN,
    x: np.ndarray,
    t_grid: np.ndarray,
) -> np.ndarray:
    """Evaluate model on full x-t grid, returning (n_t, Nx, 3) array."""
    Nx = len(x)
    n_t = len(t_grid)
    C_pinn = np.zeros((n_t, Nx, 3), dtype=float)
    for ti in range(n_t):
        x_flat = x.reshape(-1)
        t_flat = np.full_like(x_flat, t_grid[ti])
        C_pinn[ti] = predict_rs(model, x_flat, t_flat)
    return C_pinn


# =============================================================================
# Omega-based reliability (regular-solution mode)
# =============================================================================

def _reorder_c_display_to_internal_np(c: np.ndarray) -> np.ndarray:
    """Reorder composition array [Co,Ni,Ta] → [Ni,Ta,Co] (numpy, last axis)."""
    return c[..., [1, 2, 0]]


def _reorder_c_internal_to_display_np(c: np.ndarray) -> np.ndarray:
    """Reorder composition array [Ni,Ta,Co] → [Co,Ni,Ta] (numpy, last axis)."""
    return c[..., [2, 0, 1]]


def gaussian_nll_multitime_rs(  # T4: (n/2)log(2π) omitted — see gaussian_nll_from_experiment docstring
    theta: np.ndarray,
    n_components: int,
    left_right: bool,
    c0_full: np.ndarray,
    x_grid: np.ndarray,
    x_exp: np.ndarray,
    t_exp: np.ndarray,
    c_exp: np.ndarray,
    sigma: float,
    dt: float,
    nsteps: int,
    save_every: int,
    mobility: np.ndarray,
    RT: float,
    x_interface: float,
    omega_width: float,
    prior_mean: Optional[np.ndarray] = None,
    prior_std: Optional[float] = None,
    log_M_endmembers: Optional[np.ndarray] = None,
) -> float:
    """Gaussian NLL for multi-time Omega estimation using FDM forward model.

    theta and c0_full are in display [Co,Ni,Ta] order.  Internally reorders
    to [Ni,Ta,Co] for the regular-solution FDM solver, then converts back.
    """
    n_pairs = len(pair_indices_rs(n_components))
    if left_right:
        theta_left_disp = theta[:n_pairs]
        theta_right_disp = theta[n_pairs:2 * n_pairs]
    else:
        theta_left_disp = theta[:n_pairs]
        theta_right_disp = theta_left_disp

    theta_left_int = _reorder_theta_display_to_internal(theta_left_disp)
    theta_right_int = _reorder_theta_display_to_internal(theta_right_disp)
    c0_int = _reorder_c_display_to_internal_np(c0_full)

    n_ind = n_components - 1
    try:
        t_grid, C_fdm_int = fdm_ternary_regular_solution(
            c0_int, x_grid, dt, nsteps, mobility, theta_left_int, theta_right_int,
            RT=RT, x_interface=x_interface, omega_width=omega_width,
            save_every=save_every,
            log_M_endmembers=log_M_endmembers,
        )
    except Exception:
        _DIAG_COUNTERS["fdm_nll_failures"] += 1
        return 1.0e12

    C_fdm = _reorder_c_internal_to_display_np(C_fdm_int)

    t_exp_1d = np.asarray(t_exp).ravel()
    x_exp_1d = np.asarray(x_exp).ravel()

    sigma_eff = max(float(sigma), 1.0e-8)
    total_nll = 0.0
    # C18 fix: map experimental time values to nearest FDM time indices,
    # then group by integer index instead of floating-point comparison.
    t_exp_indices = np.array(
        [int(np.argmin(np.abs(t_grid - t))) for t in t_exp_1d], dtype=int
    )
    for ti in np.unique(t_exp_indices):
        mask = t_exp_indices == ti
        x_pts = x_exp_1d[mask]
        c_pts = c_exp[mask]

        c_pred = np.column_stack([
            np.interp(x_pts, x_grid, C_fdm[ti, :, j])
            for j in range(n_components)
        ])

        residual = c_pts[:, 1:n_ind + 1] - c_pred[:, 1:n_ind + 1]
        total_nll += 0.5 * np.sum((residual / sigma_eff) ** 2)

    n_total = len(t_exp_1d) * n_ind
    total_nll += n_total * np.log(sigma_eff)

    if prior_mean is not None and prior_std is not None:
        prior = 0.5 * np.sum(((theta - prior_mean) / float(prior_std)) ** 2)
        total_nll += prior

    return float(total_nll) if np.isfinite(total_nll) else 1.0e12


def refine_omega_by_fdm_likelihood(
    nll_fun,
    theta_hat: np.ndarray,
    maxiter: int = 180,
    verbose: bool = False,
    progress_status=None,
    progress_bar=None,
    pair_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, Optional[Dict]]:
    """Refine Omega estimate by minimizing FDM-based NLL using Powell method."""
    from scipy.optimize import minimize
    nll_before = float(nll_fun(theta_hat))
    _eval_count = [0]
    _best_nll = [nll_before]
    _best_theta = [theta_hat.copy()]
    _t0 = time.time()
    _nll_history: List[float] = [nll_before]
    _pair_names = pair_names or [f"p{k}" for k in range(len(theta_hat))]
    _n_dim = len(theta_hat)
    _n_pairs = len(_pair_names)
    _has_lr = _n_dim >= 2 * _n_pairs  # True when left/right Omega are separate
    _expected_evals = max(1, maxiter * (2 * _n_dim + 1))

    def _nll_with_progress(th):
        _eval_count[0] += 1
        val = nll_fun(th)
        _nll_history.append(float(val))
        if val < _best_nll[0]:
            _best_nll[0] = float(val)
            _best_theta[0] = th.copy()
        if progress_status is not None and _eval_count[0] % 3 == 0:
            elapsed = time.time() - _t0
            delta_nll = nll_before - _best_nll[0]
            if _has_lr:
                param_str = "  ".join(
                    f"{_pair_names[k % _n_pairs]}={'L' if k < _n_pairs else 'R'}:{th[k]:+.3f}"
                    for k in range(min(len(th), 2 * _n_pairs))
                )
            else:
                param_str = "  ".join(
                    f"{_pair_names[k]}:{th[k]:+.3f}"
                    for k in range(min(len(th), _n_pairs))
                )
            progress_status.markdown(
                f"**FDM refinement**  eval {_eval_count[0]}  |  "
                f"NLL: {nll_before:.2f} → **{_best_nll[0]:.2f}** (Δ={delta_nll:+.2f})  |  "
                f"elapsed {elapsed:.0f}s\n\n"
                f"`{param_str}`"
            )
        if progress_bar is not None and _eval_count[0] % 3 == 0:
            frac = min(1.0, _eval_count[0] / _expected_evals)
            progress_bar.progress(frac)
        return val

    result = minimize(_nll_with_progress, theta_hat, method="Powell",
                      options={"maxiter": maxiter, "disp": verbose})
    elapsed_total = time.time() - _t0
    if progress_bar is not None:
        progress_bar.progress(1.0)
    info = {
        "nll_before": nll_before,
        "nll_after": float(result.fun),
        "success": bool(result.success),
        "nfev": int(result.nfev),
        "message": str(result.message),
        "elapsed_s": elapsed_total,
        "nll_history": _nll_history,
    }
    return np.asarray(result.x, dtype=float), info


def overwrite_model_omega_from_theta(
    model: TernaryRegularSolutionPINN,
    theta: np.ndarray,
    left_right: bool,
) -> None:
    """Overwrite model Omega parameters from a flat theta vector (display order)."""
    n_pairs = model.n_pairs
    theta_left_disp = theta[:n_pairs]
    theta_left_int = _reorder_theta_display_to_internal(theta_left_disp)
    with torch.no_grad():
        model.theta_left_raw.copy_(torch.tensor(theta_left_int, dtype=torch.float32))
        if left_right and hasattr(model, "theta_right_raw") and isinstance(model.theta_right_raw, nn.Parameter):
            theta_right_disp = theta[n_pairs:2 * n_pairs]
            theta_right_int = _reorder_theta_display_to_internal(theta_right_disp)
            model.theta_right_raw.copy_(torch.tensor(theta_right_int, dtype=torch.float32))


def laplace_reliability_rs(
    nll_fun,
    theta_hat: np.ndarray,
    hessian_step: float,
    n_samples: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Laplace approximation reliability for Omega parameters."""
    dim = len(theta_hat)
    step = np.ones(dim, dtype=float) * float(hessian_step)
    H_raw = numerical_hessian(nll_fun, theta_hat, step)
    cov, diag = _robust_cov_from_hessian(H_raw)

    rng = np.random.default_rng(seed)
    samples = rng.multivariate_normal(theta_hat, cov, size=int(n_samples), method="svd")

    nll_at_hat = float(nll_fun(theta_hat))
    return {
        "theta_hat": np.asarray(theta_hat, dtype=float),
        "cov": cov,
        "samples": samples,
        "nll_at_hat": np.array([nll_at_hat]),
        **diag,
    }


def adapt_proposal_scale(
    proposal_scale: np.ndarray,
    acceptance_rate: float,
    target_rate: float = 0.234,
    adapt_factor: float = 1.1,
) -> np.ndarray:
    """Multiplicative adaptation of proposal scale toward target acceptance rate."""
    if acceptance_rate > target_rate:
        return proposal_scale * adapt_factor
    else:
        return proposal_scale / adapt_factor


def _mcmc_proposal_from_cov(
    proposal_cov: Optional[np.ndarray],
    proposal_std,
    dim: int,
) -> Tuple[Optional[np.ndarray], np.ndarray, str]:
    """Resolve MCMC proposal: prefer multivariate cov over per-dim std.

    Returns (L, scale, mode) where L is a Cholesky factor (or None),
    scale is a (dim,) marginal std array, and mode is a diagnostic string.
    """
    if proposal_cov is not None:
        C = np.asarray(proposal_cov, dtype=float)
        if C.shape != (dim, dim):
            raise ValueError(f"proposal_cov must have shape ({dim},{dim})")
        C = 0.5 * (C + C.T)
        w, V = np.linalg.eigh(C)
        eig_floor = max(1.0e-14, 1.0e-12 * float(np.max(np.abs(w))))
        w_pd = np.maximum(w, eig_floor)
        C_pd = (V * w_pd) @ V.T
        L = np.linalg.cholesky(0.5 * (C_pd + C_pd.T))
        scale = np.sqrt(np.diag(C_pd))
        return L, scale, "informed_mv"

    proposal_arr = np.asarray(proposal_std, dtype=float).ravel()
    if proposal_arr.size == 1:
        scale = np.full(dim, float(proposal_arr[0]))
        return None, scale, "scalar"
    if proposal_arr.size == dim:
        return None, proposal_arr.copy(), "per_dim"
    raise ValueError(f"proposal_std must be scalar or length-{dim} array")


# ---------------------------------------------------------------------------
# Diagnostics (S1, S2)
# ---------------------------------------------------------------------------

def chi2_misspecification_diagnosis(
    chi2: float,
    dof: int,
    alpha: float = 0.05,
) -> Dict[str, object]:
    """Structured chi-square misspecification verdict.

    Reduced chi^2 = chi2 / dof:
      << 1 ==> sigma overestimated or overfit
      ~ 1  ==> consistent fit
      >> 1 ==> sigma underestimated, model misspecified, or correlated residuals
    """
    dof_eff = max(int(dof), 1)
    reduced = float(chi2) / dof_eff
    std = np.sqrt(2.0 * dof_eff)
    z = (float(chi2) - float(dof_eff)) / max(std, 1.0e-14)
    z_crit = 1.96 if abs(alpha - 0.05) < 1.0e-9 else float(np.sqrt(2.0))
    if z > z_crit:
        verdict = "underdispersed (sigma may be too small or model misspecified)"
    elif z < -z_crit:
        verdict = "overdispersed (sigma may be too large or overfit)"
    else:
        verdict = "consistent"
    return {
        "chi2": float(chi2),
        "dof": int(dof_eff),
        "reduced_chi2": float(reduced),
        "z_score": float(z),
        "verdict": verdict,
        "alpha": float(alpha),
    }


def geweke_diagnostic(
    samples: np.ndarray,
    first_frac: float = 0.1,
    last_frac: float = 0.5,
) -> Dict[str, np.ndarray]:
    """Geweke (1992) z-score for MCMC convergence.

    Compares the mean of the first ``first_frac`` of samples with the last
    ``last_frac``. For a converged chain z ~ N(0, 1); |z| > 2 suggests
    non-convergence.
    """
    s = np.asarray(samples, dtype=float)
    if s.ndim == 1:
        s = s.reshape(-1, 1)
    n = s.shape[0]
    if n < 20:
        return {"z": np.full(s.shape[1], np.nan), "converged": np.array([False])}
    n_a = max(int(first_frac * n), 5)
    n_b = max(int(last_frac * n), 5)
    a = s[:n_a]
    b = s[-n_b:]
    mean_a = a.mean(axis=0)
    mean_b = b.mean(axis=0)
    var_a = a.var(axis=0, ddof=1) / n_a if n_a > 1 else np.full(s.shape[1], np.inf)
    var_b = b.var(axis=0, ddof=1) / n_b if n_b > 1 else np.full(s.shape[1], np.inf)
    z = (mean_a - mean_b) / np.sqrt(np.maximum(var_a + var_b, 1.0e-30))
    converged = np.all(np.abs(z) < 2.0)
    return {"z": z, "converged": np.array([bool(converged)])}


def mcmc_reliability_rs(
    nll_fun,
    theta_hat: np.ndarray,
    n_steps: int,
    burn_in: int,
    proposal_std,
    seed: int,
    progress_bar=None,
    proposal_cov: Optional[np.ndarray] = None,
    proposal_cov_scale: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """Random-walk Metropolis MCMC for Omega parameters.

    ``proposal_std`` accepts scalar or length-d vector (d = len(theta_hat)).
    ``proposal_cov`` (if supplied) overrides and gives a multivariate Gaussian
    proposal scaled by 2.38^2/d (Roberts-Gelman-Gilks optimal).  Pass the
    Laplace covariance from ``laplace_reliability_rs`` for informed sampling.
    """
    rng = np.random.default_rng(seed + 909)
    dim = len(theta_hat)
    if proposal_cov_scale is None:
        proposal_cov_scale = 2.38 ** 2 / max(dim, 1)

    if proposal_cov is not None:
        L, proposal_scale, mode = _mcmc_proposal_from_cov(
            np.asarray(proposal_cov, dtype=float) * float(proposal_cov_scale),
            proposal_std, dim=dim,
        )
    else:
        L, proposal_scale, mode = _mcmc_proposal_from_cov(None, proposal_std, dim=dim)

    current = theta_hat.copy()
    current_lp = -nll_fun(current)

    samples = []
    accepted = 0

    for i in range(n_steps):
        if L is not None:
            proposal = current + L @ rng.standard_normal(dim)
        else:
            proposal = current + rng.normal(0.0, proposal_scale, size=dim)
        proposal_lp = -nll_fun(proposal)
        if np.log(rng.uniform()) < proposal_lp - current_lp:
            current = proposal
            current_lp = proposal_lp
            accepted += 1
        if i >= burn_in:
            samples.append(current.copy())
        if progress_bar is not None and (i % max(1, n_steps // 100) == 0 or i == n_steps - 1):
            progress_bar.progress((i + 1) / n_steps)

    samples_arr = np.asarray(samples, dtype=float)
    return {
        "theta_hat": theta_hat.copy(),
        "samples": samples_arr,
        "acceptance_rate": np.array([accepted / max(n_steps, 1)]),
        "proposal_scale": proposal_scale.copy(),
        "proposal_mode": np.array([mode]),
    }


def posterior_band_from_samples_rs(
    theta_samples: np.ndarray,
    n_components: int,
    left_right: bool,
    c0_full: np.ndarray,
    x_grid: np.ndarray,
    dt: float,
    nsteps: int,
    save_every: int,
    mobility: np.ndarray,
    RT: float,
    x_interface: float,
    omega_width: float,
    target_time: float,
    max_samples: int = 50,
    progress_bar=None,
    log_M_endmembers: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Compute posterior credible band from Omega samples via FDM forward solves.

    theta_samples and c0_full are in display [Co,Ni,Ta] order.
    Internally reorders to [Ni,Ta,Co] for the FDM solver, then converts back.
    """
    if theta_samples is None or len(theta_samples) == 0:
        nan = np.full((len(x_grid), n_components), np.nan)
        return {"q025": nan, "q500": nan, "q975": nan}

    n_pairs = len(pair_indices_rs(n_components))
    c0_int = _reorder_c_display_to_internal_np(c0_full)
    idx = np.linspace(0, len(theta_samples) - 1, min(max_samples, len(theta_samples))).astype(int)

    profiles = []
    for k_idx, i in enumerate(idx):
        th = theta_samples[i]
        if left_right:
            theta_l_disp = th[:n_pairs]
            theta_r_disp = th[n_pairs:2 * n_pairs]
        else:
            theta_l_disp = th[:n_pairs]
            theta_r_disp = theta_l_disp
        theta_l_int = _reorder_theta_display_to_internal(theta_l_disp)
        theta_r_int = _reorder_theta_display_to_internal(theta_r_disp)
        try:
            t_grid, C_fdm_int = fdm_ternary_regular_solution(
                c0_int, x_grid, dt, nsteps, mobility, theta_l_int, theta_r_int,
                RT=RT, x_interface=x_interface, omega_width=omega_width,
                save_every=save_every,
                log_M_endmembers=log_M_endmembers,
            )
            C_fdm = _reorder_c_internal_to_display_np(C_fdm_int)
            ti_closest = int(np.argmin(np.abs(t_grid - target_time)))
            profiles.append(C_fdm[ti_closest])
        except Exception:
            profiles.append(np.full((len(x_grid), n_components), np.nan))
        if progress_bar is not None:
            progress_bar.progress((k_idx + 1) / len(idx))

    profiles_arr = np.stack(profiles, axis=0)
    return {
        "q025": np.nanquantile(profiles_arr, 0.025, axis=0),
        "q500": np.nanquantile(profiles_arr, 0.500, axis=0),
        "q975": np.nanquantile(profiles_arr, 0.975, axis=0),
    }


@dataclass
class TrainResult:
    model: TernaryDiffusionPINN
    data: TrainingData
    history: pd.DataFrame
    train_time: float


def train_pinn(
    data: TrainingData,
    log_d11_init: float,
    log_d22_init: float,
    rho_raw_init: float,
    width: int,
    depth: int,
    activation: str,
    epochs: int,
    lr: float,
    weights: Dict[str, float],
    progress=None,
    status=None,
    diag_prior_log: Optional[np.ndarray] = None,
    diag_prior_weight: float = 0.0,
    fix_diagonal_from_prior: bool = False,
    diffusion_model_mode: str = "single D",
    log_d11_right_init: Optional[float] = None,
    log_d22_right_init: Optional[float] = None,
    rho_raw_right_init: Optional[float] = None,
    phase_interface: float = 0.5,
    phase_width: float = 0.02,
    rho21_raw_init: Optional[float] = None,
    rho21_raw_right_init: Optional[float] = None,
    force_symmetric: bool = FORCE_SYMMETRIC_D,
    adaptive_weights: bool = False,
    rba_update_every: int = 50,
    direct_output: bool = False,
    compile_model: bool = False,
) -> TrainResult:
    two_region = str(diffusion_model_mode).lower().startswith("left/right")
    if two_region:
        model = TwoRegionTernaryDiffusionPINN(
            log_d11_init,
            log_d22_init,
            rho_raw_init,
            float(log_d11_right_init if log_d11_right_init is not None else log_d11_init),
            float(log_d22_right_init if log_d22_right_init is not None else log_d22_init),
            float(rho_raw_right_init if rho_raw_right_init is not None else rho_raw_init),
            width,
            depth,
            activation,
            phase_interface=phase_interface,
            phase_width=phase_width,
            rho21_raw_left_init=rho21_raw_init,
            rho21_raw_right_init=rho21_raw_right_init,
            force_symmetric=force_symmetric,
            direct_output=direct_output,
        ).to(DEVICE)
    else:
        model = TernaryDiffusionPINN(
            log_d11_init, log_d22_init, rho_raw_init, width, depth, activation,
            rho21_raw_init=rho21_raw_init, force_symmetric=force_symmetric,
            direct_output=direct_output,
        ).to(DEVICE)

    if compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)

    diag_prior_tensor = None
    if diag_prior_log is not None:
        diag_prior_arr = np.asarray(diag_prior_log, dtype=float).reshape(-1)
        diag_prior_tensor = torch.tensor(diag_prior_arr, dtype=DTYPE, device=DEVICE)
        if fix_diagonal_from_prior:
            with torch.no_grad():
                if two_region and diag_prior_tensor.numel() >= 4:
                    model.log_d11_left.copy_(diag_prior_tensor[0:1])
                    model.log_d22_left.copy_(diag_prior_tensor[1:2])
                    model.log_d11_right.copy_(diag_prior_tensor[2:3])
                    model.log_d22_right.copy_(diag_prior_tensor[3:4])
                else:
                    model.log_d11.copy_(diag_prior_tensor[0:1])
                    model.log_d22.copy_(diag_prior_tensor[1:2])
            if two_region and diag_prior_tensor.numel() >= 4:
                model.log_d11_left.requires_grad_(False)
                model.log_d22_left.requires_grad_(False)
                model.log_d11_right.requires_grad_(False)
                model.log_d22_right.requires_grad_(False)
            else:
                model.log_d11.requires_grad_(False)
                model.log_d22.requires_grad_(False)

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(epochs, 1), eta_min=lr * 0.03
    )
    mse = nn.MSELoss()

    x_obs, t_obs, c_obs = to_tensor(data.x_obs), to_tensor(data.t_obs), to_tensor(data.c_obs)
    x_ic, t_ic, c_ic = to_tensor(data.x_ic), to_tensor(data.t_ic), to_tensor(data.c_ic)
    x_bc, t_bc, c_bc = to_tensor(data.x_bc), to_tensor(data.t_bc), to_tensor(data.c_bc)
    x_f, t_f = to_tensor(data.x_f), to_tensor(data.t_f)

    hist = []
    report_every = max(1, epochs // 160)
    t0 = time.time()

    # RBA adaptive weights (mutable copies of base weights)
    w_data = float(weights["data"])
    w_ic = float(weights["ic"])
    w_bc = float(weights["bc"])
    w_phys = float(weights["phys"])

    for ep in range(1, epochs + 1):
        opt.zero_grad(set_to_none=True)

        loss_data = mse(model(x_obs, t_obs), c_obs)
        loss_ic = mse(model(x_ic, t_ic), c_ic)
        loss_bc = mse(model(x_bc, t_bc), c_bc)
        res = model.residual_train(x_f, t_f)
        loss_phys = torch.mean(res * res)

        loss_diag_prior = torch.tensor(0.0, dtype=DTYPE, device=DEVICE)
        if diag_prior_tensor is not None and float(diag_prior_weight) > 0.0:
            if two_region and diag_prior_tensor.numel() >= 4:
                loss_diag_prior = (
                    (model.log_d11_left[0] - diag_prior_tensor[0]) ** 2
                    + (model.log_d22_left[0] - diag_prior_tensor[1]) ** 2
                    + (model.log_d11_right[0] - diag_prior_tensor[2]) ** 2
                    + (model.log_d22_right[0] - diag_prior_tensor[3]) ** 2
                )
            else:
                loss_diag_prior = (
                    (model.log_d11[0] - diag_prior_tensor[0]) ** 2
                    + (model.log_d22[0] - diag_prior_tensor[1]) ** 2
                )

        # RBA: rebalance weights every rba_update_every epochs
        # C5 fix: reuse per-loss gradients to accumulate weighted total gradient
        # directly, eliminating the (N+1)th backward pass.
        _rba_epoch = adaptive_weights and ep > 1 and ep % rba_update_every == 0
        if _rba_epoch:
            params = [p for p in model.parameters() if p.requires_grad]
            all_grads = []
            gnorms = []
            for loss_i in [loss_data, loss_ic, loss_bc, loss_phys]:
                grads = torch.autograd.grad(loss_i, params, retain_graph=True, allow_unused=True)
                all_grads.append(grads)
                gn = torch.sqrt(sum(g.norm() ** 2 for g in grads if g is not None))
                gnorms.append(max(float(gn), 1.0e-12))
            mean_gn = sum(gnorms) / len(gnorms)
            base = [weights["data"], weights["ic"], weights["bc"], weights["phys"]]
            new_w = [b * mean_gn / gn for b, gn in zip(base, gnorms)]
            w_data, w_ic, w_bc, w_phys = new_w

        loss = (
            w_data * loss_data
            + w_ic * loss_ic
            + w_bc * loss_bc
            + w_phys * loss_phys
            + float(diag_prior_weight) * loss_diag_prior
        )

        # R1: detect NaN/Inf loss before it corrupts parameters.
        if not torch.isfinite(loss):
            if status is not None:
                status.markdown(
                    f"⚠ Training aborted at epoch {ep}: non-finite loss "
                    f"(data={loss_data.item():.3e}, phys={loss_phys.item():.3e})."
                )
            break

        if _rba_epoch:
            # Accumulate weighted gradients from per-loss grads (no extra backward)
            opt.zero_grad()
            for p in params:
                if p.grad is None:
                    p.grad = torch.zeros_like(p.data)
            for w_i, grads_i in zip(new_w, all_grads):
                for p, g in zip(params, grads_i):
                    if g is not None:
                        p.grad.add_(w_i * g)
            # Add diag prior gradient separately if non-zero
            if diag_prior_tensor is not None and float(diag_prior_weight) > 0.0:
                dp_loss = float(diag_prior_weight) * loss_diag_prior
                dp_grads = torch.autograd.grad(dp_loss, params, allow_unused=True)
                for p, g in zip(params, dp_grads):
                    if g is not None:
                        p.grad.add_(g)
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=10.0,
        )
        # --- Diagnostic: print loss breakdown & gradient norms at early epochs ---
        if ep in (1, 2, 5, 20, 50, 100):
            _gn_net, _gn_D = 0.0, 0.0
            for _n, _p in model.named_parameters():
                if _p.grad is None:
                    continue
                _gnorm = float(_p.grad.norm())
                if "log_d" in _n or "rho_raw" in _n:
                    _gn_D += _gnorm ** 2
                else:
                    _gn_net += _gnorm ** 2
            print(
                f"[DIAG ep={ep:3d}] loss={loss.item():.3e} | "
                f"d={loss_data.item():.3e} ic={loss_ic.item():.3e} "
                f"bc={loss_bc.item():.3e} phys={loss_phys.item():.3e} | "
                f"||g_net||={np.sqrt(_gn_net):.3e} ||g_D||={np.sqrt(_gn_D):.3e} | "
                f"lr={scheduler.get_last_lr()[0]:.2e}",
                flush=True,
            )
        # --- END DIAGNOSTIC ---

        opt.step()
        scheduler.step()

        if ep == 1 or ep % report_every == 0 or ep == epochs:
            D = model.diffusion_matrix().detach().cpu().numpy()
            D_left_hist = D
            D_right_hist = D
            if two_region and hasattr(model, "diffusion_matrix_left"):
                D_left_hist = model.diffusion_matrix_left().detach().cpu().numpy()
                D_right_hist = model.diffusion_matrix_right().detach().cpu().numpy()
            rho = D[0, 1] / max(np.sqrt(D[0, 0] * D[1, 1]), 1.0e-14)
            row = {
                "epoch": ep,
                "loss": float(loss.detach().cpu()),
                "data": float(loss_data.detach().cpu()),
                "ic": float(loss_ic.detach().cpu()),
                "bc": float(loss_bc.detach().cpu()),
                "physics": float(loss_phys.detach().cpu()),
                "diag_prior": float(loss_diag_prior.detach().cpu()),
                "D_NiNi": D[0, 0],
                "D_NiTa": D[0, 1],
                "D_TaNi": D[1, 0],
                "D_TaTa": D[1, 1],
                "D_NiNi_left": D_left_hist[0, 0],
                "D_NiTa_left": D_left_hist[0, 1],
                "D_TaNi_left": D_left_hist[1, 0],
                "D_TaTa_left": D_left_hist[1, 1],
                "D_NiNi_right": D_right_hist[0, 0],
                "D_NiTa_right": D_right_hist[0, 1],
                "D_TaNi_right": D_right_hist[1, 0],
                "D_TaTa_right": D_right_hist[1, 1],
                "rho": rho,
                "lr": scheduler.get_last_lr()[0],
            }
            hist.append(row)
            if progress is not None:
                progress.progress(min(ep / epochs, 1.0))
            if status is not None:
                status.markdown(
                    f"**Training on {DEVICE}** `{ep:,}/{epochs:,}` | "
                    f"loss `{row['loss']:.3e}` | "
                    f"D=[[{D[0,0]:.3e}, {D[0,1]:+.3e}], "
                    f"[{D[1,0]:+.3e}, {D[1,1]:.3e}]]"
                )

    return TrainResult(model=model, data=data, history=pd.DataFrame(hist), train_time=time.time() - t0)


@torch.no_grad()
def predict(model: TernaryDiffusionPINN, x: np.ndarray, t: np.ndarray) -> np.ndarray:
    return model(to_tensor(x), to_tensor(t)).detach().cpu().numpy()


def residual_grid(
    model: TernaryDiffusionPINN,
    t_start: float,
    t_max: float,
    nx: int = 100,
    nt: int = 70,
    chunk_size: int = 1024,
):
    """Evaluate residual map in chunks.

    The second derivative still requires a local autograd graph, but chunking
    prevents one large graph from being built for the full residual grid.
    """
    x = np.linspace(0.0, 1.0, nx).reshape(-1, 1)
    t = np.linspace(t_start, t_max, nt).reshape(-1, 1)
    X, T = np.meshgrid(x.ravel(), t.ravel())
    Xf = X.reshape(-1, 1)
    Tf = T.reshape(-1, 1)

    chunks = []
    for start in range(0, len(Xf), int(chunk_size)):
        end = min(start + int(chunk_size), len(Xf))
        R_chunk = model.residual_eval(to_tensor(Xf[start:end]), to_tensor(Tf[start:end]))
        chunks.append(R_chunk.detach().cpu().numpy())
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    R = np.vstack(chunks)
    return x.ravel(), t.ravel(), R.reshape(nt, nx, 2)


# =============================================================================
# Likelihood and reliability
# =============================================================================

def predict_final_profile_from_theta(
    theta: np.ndarray,
    x_query: np.ndarray,
    t_max: float,
    nx: int,
    nt_save: int,
    use_cache: bool = False,
    symmetric: Optional[bool] = None,
) -> np.ndarray:
    """Predict final FDM profile.

    C10 fix: ``symmetric`` explicitly controls parameter interpretation.
    If None, inferred from ``len(theta)`` (3→symmetric, 4→non-symmetric).
    """
    theta = np.asarray(theta, dtype=float).ravel()
    if symmetric is None:
        symmetric = (theta.size <= 3)
    solver = run_fdm_teacher if use_cache else _run_fdm_teacher_core
    rho21 = None if symmetric else float(theta[3])
    xg, _, Cg, _ = solver(
        float(theta[0]), float(theta[1]), float(theta[2]),
        float(t_max), int(nx), int(nt_save), rho21_raw=rho21,
    )
    return np.column_stack([np.interp(x_query, xg, Cg[-1, :, j]) for j in range(3)])


def gaussian_nll_from_experiment(
    theta: np.ndarray,
    x_exp: np.ndarray,
    c_exp: np.ndarray,
    sigma: float,
    t_max: float,
    nx: int,
    nt_save: int,
) -> float:
    """Gaussian NLL using independent Ni and Ta components.

    **T4 note**: The (n/2)log(2π) normalisation constant is omitted.
    This does not affect posterior inference (MAP, Laplace, MCMC) because
    the constant cancels in all likelihood ratios and gradients.
    """
    sigma_eff = max(float(sigma), 1.0e-8)
    pred = predict_final_profile_from_theta(theta, x_exp, t_max, nx, nt_save, use_cache=False)
    residual = c_exp[:, 1:3] - pred[:, 1:3]
    n = residual.size
    return float(0.5 * np.sum((residual / sigma_eff) ** 2) + n * np.log(sigma_eff))


def gaussian_nll_multitime(
    theta: np.ndarray,
    x_exp_all: np.ndarray,
    t_exp_all: np.ndarray,
    c_exp_all: np.ndarray,
    sigma: float,
    t_max: float,
    nx: int,
    nt_save: int,
    symmetric: Optional[bool] = None,
) -> float:
    """Gaussian NLL using multi-time experimental data for single D mode (P4 fix).

    Unlike ``gaussian_nll_from_experiment`` which uses only the final time
    slice, this function matches predictions at every observed time point,
    improving cross-interdiffusion term identifiability.

    C10 fix: ``symmetric`` explicitly controls parameter interpretation.
    **T4 note**: The (n/2)log(2π) normalisation constant is omitted.
    """
    sigma_eff = max(float(sigma), 1.0e-8)
    theta = np.asarray(theta, dtype=float).ravel()
    if symmetric is None:
        symmetric = (theta.size <= 3)
    rho21 = None if symmetric else float(theta[3])
    try:
        xg, t_grid_fdm, Cg, _ = _run_fdm_teacher_core(
            float(theta[0]), float(theta[1]), float(theta[2]),
            float(t_max), int(nx), int(nt_save), rho21_raw=rho21,
        )
    except Exception:
        _DIAG_COUNTERS["fdm_nll_failures"] += 1
        return 1.0e12

    t_exp_1d = np.asarray(t_exp_all).ravel()
    x_exp_1d = np.asarray(x_exp_all).ravel()

    # Use bilinear interpolation in both x and t to avoid
    # nearest-neighbor time mismatch when FDM grid varies with θ.
    pred = bilinear_sample_xt(xg, t_grid_fdm, Cg, x_exp_1d, t_exp_1d)
    residual = c_exp_all[:, 1:3] - pred[:, 1:3]
    total_nll = 0.5 * np.sum((residual / sigma_eff) ** 2)

    n_total = len(t_exp_1d) * 2
    total_nll += n_total * np.log(sigma_eff)
    return float(total_nll) if np.isfinite(total_nll) else 1.0e12


def gaussian_chi2_from_experiment(
    theta: np.ndarray,
    x_exp: np.ndarray,
    c_exp: np.ndarray,
    sigma: float,
    t_max: float,
    nx: int,
    nt_save: int,
) -> Tuple[float, float]:
    sigma_eff = max(float(sigma), 1.0e-8)
    pred = predict_final_profile_from_theta(theta, x_exp, t_max, nx, nt_save, use_cache=False)
    residual = c_exp[:, 1:3] - pred[:, 1:3]
    chi2 = float(np.sum((residual / sigma_eff) ** 2))
    dof = max(int(residual.size - len(theta)), 1)  # P10: use len(theta) instead of hardcoded 3
    return chi2, chi2 / dof


def neg_log_posterior(
    theta: np.ndarray,
    x_exp: np.ndarray,
    c_exp: np.ndarray,
    sigma: float,
    t_max: float,
    nx: int,
    nt_save: int,
    prior_mean: np.ndarray,
    prior_std: np.ndarray,
    x_exp_all: Optional[np.ndarray] = None,
    t_exp_all: Optional[np.ndarray] = None,
    c_exp_all: Optional[np.ndarray] = None,
) -> float:
    # P4 fix: use multitime NLL when multitime data is available
    if x_exp_all is not None and t_exp_all is not None and c_exp_all is not None:
        nll = gaussian_nll_multitime(theta, x_exp_all, t_exp_all, c_exp_all,
                                     sigma, t_max, nx, nt_save)
    else:
        nll = gaussian_nll_from_experiment(theta, x_exp, c_exp, sigma, t_max, nx, nt_save)
    prior = 0.5 * np.sum(((theta - prior_mean) / prior_std) ** 2)
    return float(nll + prior)


# ---------------------------------------------------------------------------
#  σ-marginalised NLL / posterior  (sigma as free parameter in MCMC)
# ---------------------------------------------------------------------------
def _half_cauchy_log_prior(sigma: float, scale: float = 0.1) -> float:
    """Log-density of a half-Cauchy(0, scale) prior on σ > 0.

    p(σ) = 2 / (π * scale * (1 + (σ/scale)²))
    Uninformative but proper; scale ≈ expected noise order.
    """
    if sigma <= 0.0:
        return -np.inf
    # C9 fix: include log(2) for correct half-Cauchy normalisation
    return float(np.log(2.0) - np.log(np.pi * scale) - np.log(1.0 + (sigma / scale) ** 2))


def neg_log_posterior_marginal_sigma(
    theta_sigma: np.ndarray,
    x_exp: np.ndarray,
    c_exp: np.ndarray,
    t_max: float,
    nx: int,
    nt_save: int,
    prior_mean: np.ndarray,
    prior_std: np.ndarray,
    sigma_prior_scale: float = 0.1,
    x_exp_all: Optional[np.ndarray] = None,
    t_exp_all: Optional[np.ndarray] = None,
    c_exp_all: Optional[np.ndarray] = None,
) -> float:
    """Neg-log-posterior where the last element of theta_sigma is log(σ).

    θ_ext = [θ₁, ..., θ_d, log σ]
    σ = exp(log σ)  (Jacobian correction: +log σ cancels with change of variable)

    Prior on θ: isotropic Gaussian with ``prior_mean`` and ``prior_std``.
    Prior on σ: half-Cauchy(0, sigma_prior_scale).
    """
    ts = np.asarray(theta_sigma, dtype=float).ravel()
    theta = ts[:-1]
    log_sigma = float(ts[-1])
    sigma = np.exp(log_sigma)

    # P4 fix: use multitime NLL when multitime data is available
    if x_exp_all is not None and t_exp_all is not None and c_exp_all is not None:
        nll = gaussian_nll_multitime(theta, x_exp_all, t_exp_all, c_exp_all,
                                     sigma, t_max, nx, nt_save)
    else:
        nll = gaussian_nll_from_experiment(theta, x_exp, c_exp, sigma, t_max, nx, nt_save)
    prior_theta = 0.5 * np.sum(((theta - prior_mean) / prior_std) ** 2)
    prior_sigma = -_half_cauchy_log_prior(sigma, sigma_prior_scale)
    # Jacobian: sampling log σ so add -log σ to undo the implicit uniform in log space
    jacobian = -log_sigma
    return float(nll + prior_theta + prior_sigma + jacobian)


def neg_log_posterior_marginal_sigma_lr(
    theta_sigma_lr: np.ndarray,
    x_exp: np.ndarray,
    c_exp: np.ndarray,
    t_max: float,
    nx: int,
    nt_save: int,
    phase_width: float,
    prior_mean_lr: np.ndarray,
    prior_std_lr: np.ndarray,
    sigma_prior_scale: float = 0.1,
) -> float:
    """Neg-log-posterior with σ marginalised (left/right D version).

    θ_ext = [θ_L (dim), θ_R (dim), log σ]
    """
    ts = np.asarray(theta_sigma_lr, dtype=float).ravel()
    theta_lr = ts[:-1]
    log_sigma = float(ts[-1])
    sigma = np.exp(log_sigma)

    nll = gaussian_nll_from_experiment_lr(
        theta_lr, x_exp, c_exp, sigma, t_max, nx, nt_save, phase_width,
    )
    prior_theta = 0.5 * np.sum(((theta_lr - prior_mean_lr) / prior_std_lr) ** 2)
    prior_sigma = -_half_cauchy_log_prior(sigma, sigma_prior_scale)
    jacobian = -log_sigma
    return float(nll + prior_theta + prior_sigma + jacobian)


def adaptive_hessian_step(
    theta0: np.ndarray,
    base_step: float = 0.05,
    floor: float = 1.0e-3,
) -> np.ndarray:
    """Per-dimension step size for finite-difference Hessian.

    Uses ``max(|theta_i| * base_step, floor)`` so that parameters near zero
    (e.g. rho_raw ~ 0) still get a usable step.  The floor also matters for
    log-D parameters at typical magnitudes around 1.

    C8 fix: removed redundant ``min_step`` parameter (was dominated by
    ``floor`` in all practical cases).
    """
    theta = np.asarray(theta0, dtype=float).reshape(-1)
    return np.maximum(np.abs(theta) * float(base_step), float(floor))


def numerical_hessian_adaptive(
    fun,
    theta0: np.ndarray,
    base_step: float = 0.05,
    floor: float = 1.0e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Numerical Hessian with per-parameter step sizing.

    Returns (H, step) so callers can record the step used.
    """
    step = adaptive_hessian_step(theta0, base_step=base_step, floor=floor)
    H = numerical_hessian(fun, theta0, step)
    return H, step


def numerical_hessian(fun, theta0: np.ndarray, step: np.ndarray) -> np.ndarray:
    n = len(theta0)
    H = np.zeros((n, n), dtype=float)
    f0 = fun(theta0)
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = step[i]
        H[i, i] = (fun(theta0 + ei) - 2.0 * f0 + fun(theta0 - ei)) / (step[i] ** 2)
        for j in range(i + 1, n):
            ej = np.zeros(n)
            ej[j] = step[j]
            H[i, j] = (
                fun(theta0 + ei + ej)
                - fun(theta0 + ei - ej)
                - fun(theta0 - ei + ej)
                + fun(theta0 - ei - ej)
            ) / (4.0 * step[i] * step[j])
            H[j, i] = H[i, j]
    # C13: enforce symmetry via averaging (guards against numerical asymmetry)
    return 0.5 * (H + H.T)


def _robust_cov_from_hessian(
    H_raw: np.ndarray,
    eps_rel: float = 1.0e-8,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Convert a (possibly indefinite) numerical Hessian to a PD covariance.

    Project H onto the PD cone in the spectral sense by flooring eigenvalues
    at ``eps_floor = max(eps_rel, eps_rel * max|w|)``, then invert directly
    via the eigendecomposition.  This is more theoretically sound than
    the two-stage scheme of (a) regularizing H with a tiny scalar and
    (b) clipping eigenvalues of cov afterwards, which can silently distort
    the covariance when H is strongly indefinite.

    Returns ``(cov, diag_dict)`` where *diag_dict* holds diagnostic arrays.
    """
    H_sym = 0.5 * (H_raw + H_raw.T)
    w_raw, V = np.linalg.eigh(H_sym)
    max_abs = float(np.max(np.abs(w_raw))) if w_raw.size > 0 else 1.0
    eig_floor = max(float(eps_rel), float(eps_rel) * max_abs)
    non_pd = bool(np.any(w_raw <= 0.0))
    w_pd = np.maximum(w_raw, eig_floor)
    cov = (V * (1.0 / w_pd)) @ V.T
    cov = 0.5 * (cov + cov.T)
    cov_eigvals = np.linalg.eigvalsh(cov)
    # C1 fix: align keys with UI expectations
    diag = {
        "hessian_eigval_raw": w_raw,
        "hessian_eigval_regularized": w_pd,
        "hessian_min_eig": np.array([float(np.min(w_raw)) if w_raw.size > 0 else 0.0]),
        "hessian_non_pd": np.array([non_pd]),
        "covariance_was_clipped": np.array([non_pd]),
        "hessian_inverse_method": np.array(["eigen-floor" if non_pd else "direct"]),
        "cov_eigval_raw": cov_eigvals,
        "cov_eigval_clipped": cov_eigvals,
        "eig_floor": float(eig_floor),
    }
    return cov, diag


def laplace_reliability(
    theta_hat: np.ndarray,
    x_exp: np.ndarray,
    c_exp: np.ndarray,
    sigma: float,
    t_max: float,
    nx: int,
    nt_save: int,
    prior_mean: np.ndarray,
    prior_std_scalar: float,
    hessian_step: float,
    n_samples: int,
    seed: int,
    x_exp_all: Optional[np.ndarray] = None,
    t_exp_all: Optional[np.ndarray] = None,
    c_exp_all: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Low-cost local posterior approximation around the PINN estimate.

    The prior mean is intentionally supplied by the user/config, not set to theta_hat.
    This avoids circular confidence where the prior forces the posterior back to
    the PINN estimate.

    P4: When ``x_exp_all``, ``t_exp_all``, ``c_exp_all`` are supplied, the NLL
    evaluates all observed time points (not just the final profile).
    """
    dim = len(theta_hat)
    prior_std = np.ones(dim, dtype=float) * float(prior_std_scalar)
    fun = lambda th: neg_log_posterior(
        th, x_exp, c_exp, sigma, t_max, nx, nt_save, prior_mean, prior_std,
        x_exp_all=x_exp_all, t_exp_all=t_exp_all, c_exp_all=c_exp_all,
    )

    step = np.ones(dim, dtype=float) * float(hessian_step)
    H_raw = numerical_hessian(fun, theta_hat, step)
    cov, diag = _robust_cov_from_hessian(H_raw)

    rng = np.random.default_rng(seed)
    samples = rng.multivariate_normal(theta_hat, cov, size=int(n_samples), method="svd")

    chi2, red_chi2 = gaussian_chi2_from_experiment(theta_hat, x_exp, c_exp, sigma, t_max, nx, nt_save)
    return {
        "theta_hat": theta_hat,
        "cov": cov,
        "samples": samples,
        "chi2": np.array([chi2]),
        "reduced_chi2": np.array([red_chi2]),
        **diag,
    }


@st.cache_data(show_spinner=False, max_entries=24)
def cached_laplace_reliability(
    theta_hat: np.ndarray,
    x_exp: np.ndarray,
    c_exp: np.ndarray,
    sigma: float,
    t_max: float,
    nx: int,
    nt_save: int,
    prior_mean: np.ndarray,
    prior_std_scalar: float,
    hessian_step: float,
    n_samples: int,
    seed: int,
    x_exp_all: Optional[np.ndarray] = None,
    t_exp_all: Optional[np.ndarray] = None,
    c_exp_all: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Cached low-cost reliability calculation.

    prior_mean is supplied explicitly by the UI/config and is not forced to
    theta_hat. This avoids circular confidence where the prior pulls the
    posterior back to the PINN estimate.
    """
    return laplace_reliability(
        theta_hat=theta_hat,
        x_exp=x_exp,
        c_exp=c_exp,
        sigma=sigma,
        t_max=t_max,
        nx=nx,
        nt_save=nt_save,
        prior_mean=prior_mean,
        prior_std_scalar=prior_std_scalar,
        hessian_step=hessian_step,
        n_samples=n_samples,
        seed=seed,
        x_exp_all=x_exp_all,
        t_exp_all=t_exp_all,
        c_exp_all=c_exp_all,
    )


def mcmc_reliability(
    theta_start: np.ndarray,
    x_exp: np.ndarray,
    c_exp: np.ndarray,
    sigma: float,
    t_max: float,
    nx: int,
    nt_save: int,
    prior_mean: np.ndarray,
    prior_std_scalar: float,
    n_steps: int,
    burn_in: int,
    proposal_std,
    seed: int,
    progress_bar=None,
    proposal_cov: Optional[np.ndarray] = None,
    proposal_cov_scale: Optional[float] = None,
    marginalize_sigma: bool = False,
    sigma_prior_scale: float = 0.1,
    x_exp_all: Optional[np.ndarray] = None,
    t_exp_all: Optional[np.ndarray] = None,
    c_exp_all: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Higher-cost random-walk Metropolis posterior sampling.

    ``proposal_cov`` (if supplied) takes precedence: a multivariate Gaussian
    proposal is constructed from a Cholesky factor of
    ``proposal_cov_scale * proposal_cov``.  The default scale 2.38^2/d is the
    Roberts-Gelman-Gilks optimal RW Metropolis scaling.

    When ``marginalize_sigma`` is True, σ is jointly sampled.  The state
    vector is extended: [θ₁, ..., θ_d, log(σ)].  sigma becomes the initial
    value.  Results include ``sigma_samples``.
    """
    rng = np.random.default_rng(seed + 404)
    theta_start = np.asarray(theta_start, dtype=float).ravel()
    dim_theta = len(theta_start)
    prior_mean_arr = np.asarray(prior_mean, dtype=float).ravel()
    prior_std = np.ones(dim_theta, dtype=float) * float(prior_std_scalar)

    if marginalize_sigma:
        current = np.append(theta_start, np.log(max(float(sigma), 1.0e-10)))
        dim = dim_theta + 1
    else:
        current = theta_start.copy()
        dim = dim_theta

    if proposal_cov_scale is None:
        proposal_cov_scale = 2.38 ** 2 / max(dim, 1)

    if proposal_cov is not None:
        _cov = np.asarray(proposal_cov, dtype=float)
        # Fix #3: augment proposal_cov with sigma block when marginalize_sigma=True
        if marginalize_sigma and _cov.shape == (dim_theta, dim_theta):
            _cov_aug = np.zeros((dim, dim), dtype=float)
            _cov_aug[:dim_theta, :dim_theta] = _cov
            _cov_aug[dim_theta, dim_theta] = 0.01  # σ proposal variance
            _cov = _cov_aug
        L, proposal_scale, mode = _mcmc_proposal_from_cov(
            _cov * float(proposal_cov_scale),
            proposal_std, dim=dim,
        )
    else:
        ps_arr = np.asarray(proposal_std, dtype=float).ravel()
        if ps_arr.size == 1:
            ps_arr = np.full(dim_theta, float(ps_arr[0]))
        if marginalize_sigma:
            ps = np.append(ps_arr, 0.1)
        else:
            ps = ps_arr
        L, proposal_scale, mode = _mcmc_proposal_from_cov(None, ps, dim=dim)

    def _lp(state):
        if marginalize_sigma:
            return -neg_log_posterior_marginal_sigma(
                state, x_exp, c_exp, t_max, nx, nt_save,
                prior_mean_arr, prior_std, sigma_prior_scale,
                x_exp_all=x_exp_all, t_exp_all=t_exp_all, c_exp_all=c_exp_all,
            )
        else:
            return -neg_log_posterior(
                state, x_exp, c_exp, sigma, t_max, nx, nt_save,
                prior_mean_arr, prior_std,
                x_exp_all=x_exp_all, t_exp_all=t_exp_all, c_exp_all=c_exp_all,
            )

    current_lp = _lp(current)

    samples = []
    accepted = 0
    n_steps = int(n_steps)
    burn_in = int(burn_in)

    for i in range(n_steps):
        if L is not None:
            proposal = current + L @ rng.standard_normal(dim)
        else:
            proposal = current + rng.normal(0.0, proposal_scale, size=dim)
        proposal_lp = _lp(proposal)

        if np.log(rng.uniform()) < proposal_lp - current_lp:
            current = proposal
            current_lp = proposal_lp
            accepted += 1

        if i >= burn_in:
            samples.append(current.copy())

        if progress_bar is not None and (i % max(1, n_steps // 100) == 0 or i == n_steps - 1):
            progress_bar.progress((i + 1) / n_steps)

    samples = np.asarray(samples, dtype=float)
    if samples.ndim == 1:
        samples = samples.reshape(0, dim_theta + (1 if marginalize_sigma else 0))
    sigma_eff = sigma
    if marginalize_sigma and samples.shape[0] > 0:
        sigma_eff = float(np.exp(np.median(samples[:, -1])))

    # P12 fix: evaluate chi2 at posterior median theta (not starting point)
    # C17 fix: include theta_hat in result dict for symmetry with mcmc_reliability_lr
    theta_for_chi2 = theta_start
    if samples.shape[0] > 5:
        theta_for_chi2 = np.median(samples[:, :dim_theta], axis=0)
    chi2, red_chi2 = gaussian_chi2_from_experiment(
        theta_for_chi2, x_exp, c_exp, sigma_eff, t_max, nx, nt_save,
    )
    result = {
        "theta_hat": np.asarray(theta_start, dtype=float),
        "samples": samples[:, :dim_theta] if marginalize_sigma else samples,
        "acceptance_rate": np.array([accepted / max(n_steps, 1)]),
        "chi2": np.array([chi2]),
        "reduced_chi2": np.array([red_chi2]),
    }
    if marginalize_sigma and samples.shape[0] > 0:
        result["sigma_samples"] = np.exp(samples[:, -1])
        result["sigma_median"] = np.array([sigma_eff])
    return result


def D_metrics_from_theta_samples(samples: np.ndarray) -> pd.DataFrame:
    if samples is None or len(samples) == 0:
        return pd.DataFrame(columns=["parameter", "mean", "std", "2.5%", "50%", "97.5%"])

    mats = np.asarray([make_d_matrix_from_theta(s) for s in samples])
    vals = {
        "D_NiNi": mats[:, 0, 0],
        "D_NiTa": mats[:, 0, 1],
        "D_TaNi": mats[:, 1, 0],
        "D_TaTa": mats[:, 1, 1],
    }

    rows = []
    for name, arr in vals.items():
        rows.append(
            {
                "parameter": name,
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                "2.5%": float(np.quantile(arr, 0.025)),
                "50%": float(np.quantile(arr, 0.500)),
                "97.5%": float(np.quantile(arr, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def reliability_summary_table(name: str, result_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
    table = D_metrics_from_theta_samples(result_dict["samples"])
    table.insert(0, "method", name)
    return table


# ---------------------------------------------------------------------------
#  PSIS (Pareto Smoothed Importance Sampling) diagnostic for Laplace
# ---------------------------------------------------------------------------
def psis_diagnostic(
    laplace_samples: np.ndarray,
    x_exp: np.ndarray,
    c_exp: np.ndarray,
    sigma: float,
    t_max: float,
    nx: int,
    nt_save: int,
    prior_mean: np.ndarray,
    prior_std_scalar: float,
    theta_hat: np.ndarray,
    cov: np.ndarray,
    max_eval: int = 200,
    x_exp_all: Optional[np.ndarray] = None,
    t_exp_all: Optional[np.ndarray] = None,
    c_exp_all: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Pareto-k diagnostic for Laplace approximation quality.

    Compute importance ratios  w_s = p(θ_s | data) / q(θ_s)
    where q is the Laplace Gaussian and p is the true (unnormalized) posterior.
    Fit a Generalized Pareto Distribution to the upper tail of log(w).

    Returns dict with:
      pareto_k:  shape parameter (< 0.5 good, 0.5-0.7 ok, > 0.7 poor)
      ess_psis:  effective sample size after PSIS
    """
    dim = len(theta_hat)
    prior_std = np.ones(dim) * float(prior_std_scalar)
    theta_hat = np.asarray(theta_hat, dtype=float).ravel()
    cov = np.asarray(cov, dtype=float)

    n_use = min(len(laplace_samples), max_eval)
    samples = laplace_samples[:n_use]

    try:
        L = np.linalg.cholesky(cov)
        log_det_cov = 2.0 * np.sum(np.log(np.diag(L)))
    except np.linalg.LinAlgError:
        log_det_cov = np.log(max(np.linalg.det(cov), 1e-300))

    log_ratios = np.empty(n_use, dtype=float)
    for i, th in enumerate(samples):
        log_p = -neg_log_posterior(th, x_exp, c_exp, sigma, t_max, nx, nt_save, prior_mean, prior_std,
                                   x_exp_all=x_exp_all, t_exp_all=t_exp_all, c_exp_all=c_exp_all)
        diff = th - theta_hat
        try:
            sol = np.linalg.solve(cov, diff)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(cov, diff, rcond=None)[0]
        log_q = -0.5 * (dim * np.log(2 * np.pi) + log_det_cov + diff @ sol)
        log_ratios[i] = log_p - log_q

    log_ratios -= np.max(log_ratios)

    # Fit GPD to upper tail (top 20%)
    M = max(int(0.2 * n_use), 5)
    sorted_lr = np.sort(log_ratios)
    tail = sorted_lr[-M:]
    threshold = sorted_lr[-M - 1] if M < n_use else sorted_lr[0]
    exceedances = tail - threshold
    exceedances = exceedances[exceedances > 0]

    if len(exceedances) < 3:
        k_hat = 0.0
    else:
        # GPD shape ξ via scipy MLE (robust for all ξ).
        # Vehtari et al. (2017) PSIS thresholds: 0.5 (marginal), 0.7 (poor).
        from scipy.stats import genpareto
        try:
            xi_fit, _loc, _scale = genpareto.fit(exceedances, floc=0.0)
            k_hat = float(xi_fit)
        except Exception:
            k_hat = 0.0
        k_hat = max(min(k_hat, 2.0), -0.5)

    weights = np.exp(log_ratios)
    w_sum = np.sum(weights)
    w_normed = weights / max(w_sum, 1e-300)
    ess = float(1.0 / max(np.sum(w_normed ** 2), 1e-300))

    return {
        "pareto_k": float(k_hat),
        "ess_psis": ess,
        "n_evaluated": n_use,
    }


def posterior_band_from_samples(
    theta_samples: np.ndarray,
    x: np.ndarray,
    t_max: float,
    nx: int,
    nt_save: int,
    max_samples: int = 80,
    progress_bar=None,
) -> Dict[str, np.ndarray]:
    # Defensive: force scalar args to native Python types
    t_max = float(t_max); nx = int(nx); nt_save = int(nt_save)
    if theta_samples is None or len(theta_samples) == 0:
        nan = np.full((len(x), 3), np.nan)
        return {"q025": nan, "q500": nan, "q975": nan}

    idx = np.linspace(0, len(theta_samples) - 1, min(int(max_samples), len(theta_samples))).astype(int)
    profiles = []
    for k, i in enumerate(idx):
        profiles.append(predict_final_profile_from_theta(theta_samples[i], x, t_max, nx, nt_save, use_cache=False))
        if progress_bar is not None:
            progress_bar.progress((k + 1) / len(idx))

    profiles = np.stack(profiles, axis=0)
    return {
        "q025": np.quantile(profiles, 0.025, axis=0),
        "q500": np.quantile(profiles, 0.500, axis=0),
        "q975": np.quantile(profiles, 0.975, axis=0),
    }



# =============================================================================
# Left/right 6-parameter reliability
# =============================================================================

def theta_lr_from_matrices(D_left: np.ndarray, D_right: np.ndarray,
                           force_symmetric: Optional[bool] = None) -> np.ndarray:
    """Concatenated parameter representation for two-region D matrices."""
    th_l = theta_from_D_matrix(D_left, force_symmetric=force_symmetric)
    th_r = theta_from_D_matrix(D_right, force_symmetric=force_symmetric)
    return np.concatenate([th_l, th_r]).astype(float)


def matrices_from_theta_lr(theta_lr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return D_left, D_right from theta_lr (6 symmetric or 8 non-symmetric)."""
    theta_lr = np.asarray(theta_lr, dtype=float).reshape(-1)
    dim1 = len(theta_lr) // 2
    if dim1 not in (3, 4) or theta_lr.size != 2 * dim1:
        raise ValueError(f"theta_lr must have 6 or 8 elements, got {theta_lr.size}")
    D_left = make_d_matrix_from_theta(theta_lr[:dim1])
    D_right = make_d_matrix_from_theta(theta_lr[dim1:])
    return D_left, D_right


def predict_final_profile_from_theta_lr(
    theta_lr: np.ndarray,
    x_query: np.ndarray,
    t_max: float,
    nx: int,
    nt_save: int,
    phase_width: float,
    use_cache: bool = False,
) -> np.ndarray:
    """Final profile from left/right FDM."""
    theta_lr = np.asarray(theta_lr, dtype=float).ravel()
    dim1 = len(theta_lr) // 2
    th_l = theta_lr[:dim1]
    th_r = theta_lr[dim1:]
    rho21_l = float(th_l[3]) if dim1 > 3 else None
    rho21_r = float(th_r[3]) if dim1 > 3 else None
    solver = run_fdm_teacher_two_region if use_cache else _run_fdm_teacher_core_two_region
    xg, _, Cg, _, _, _ = solver(
        float(th_l[0]), float(th_l[1]), float(th_l[2]),
        float(th_r[0]), float(th_r[1]), float(th_r[2]),
        float(t_max), int(nx), int(nt_save), float(phase_width),
        rho21_raw_left=rho21_l, rho21_raw_right=rho21_r,
    )
    return np.column_stack([np.interp(x_query, xg, Cg[-1, :, j]) for j in range(3)])


def gaussian_nll_from_experiment_lr(  # T4: (n/2)log(2π) omitted — see gaussian_nll_from_experiment docstring
    theta_lr: np.ndarray,
    x_exp: np.ndarray,
    c_exp: np.ndarray,
    sigma: float,
    t_max: float,
    nx: int,
    nt_save: int,
    phase_width: float,
) -> float:
    """Gaussian NLL for left/right two-region model using independent Ni and Ta components."""
    sigma_eff = max(float(sigma), 1.0e-8)
    pred = predict_final_profile_from_theta_lr(theta_lr, x_exp, t_max, nx, nt_save, phase_width, use_cache=False)
    residual = c_exp[:, 1:3] - pred[:, 1:3]
    n = residual.size
    return float(0.5 * np.sum((residual / sigma_eff) ** 2) + n * np.log(sigma_eff))


def gaussian_chi2_from_experiment_lr(
    theta_lr: np.ndarray,
    x_exp: np.ndarray,
    c_exp: np.ndarray,
    sigma: float,
    t_max: float,
    nx: int,
    nt_save: int,
    phase_width: float,
) -> Tuple[float, float]:
    sigma_eff = max(float(sigma), 1.0e-8)
    pred = predict_final_profile_from_theta_lr(theta_lr, x_exp, t_max, nx, nt_save, phase_width, use_cache=False)
    residual = c_exp[:, 1:3] - pred[:, 1:3]
    chi2 = float(np.sum((residual / sigma_eff) ** 2))
    dof = max(int(residual.size - len(theta_lr)), 1)  # P10: dynamic DOF
    return chi2, chi2 / dof


def neg_log_posterior_lr(
    theta_lr: np.ndarray,
    x_exp: np.ndarray,
    c_exp: np.ndarray,
    sigma: float,
    t_max: float,
    nx: int,
    nt_save: int,
    phase_width: float,
    prior_mean_lr: np.ndarray,
    prior_std_lr: np.ndarray,
) -> float:
    nll = gaussian_nll_from_experiment_lr(theta_lr, x_exp, c_exp, sigma, t_max, nx, nt_save, phase_width)
    prior = 0.5 * np.sum(((theta_lr - prior_mean_lr) / prior_std_lr) ** 2)
    return float(nll + prior)


def _posterior_result_from_hessian(
    theta_hat: np.ndarray,
    H_raw: np.ndarray,
    n_samples: int,
    seed: int,
    chi2: float,
    red_chi2: float,
) -> Dict[str, np.ndarray]:
    """Common Hessian-to-samples conversion with diagnostics (uses N1 eigen-floor)."""
    cov, diag = _robust_cov_from_hessian(H_raw)

    rng = np.random.default_rng(seed)
    samples = rng.multivariate_normal(theta_hat, cov, size=int(n_samples), method="svd")

    return {
        "theta_hat": np.asarray(theta_hat, dtype=float),
        "cov": cov,
        "samples": samples,
        "chi2": np.array([chi2]),
        "reduced_chi2": np.array([red_chi2]),
        **diag,
    }


def laplace_reliability_lr(
    theta_hat_lr: np.ndarray,
    x_exp: np.ndarray,
    c_exp: np.ndarray,
    sigma: float,
    t_max: float,
    nx: int,
    nt_save: int,
    phase_width: float,
    prior_mean_lr: np.ndarray,
    prior_std_scalar: float,
    hessian_step: float,
    n_samples: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Laplace reliability for two-region left/right D (dim-aware)."""
    theta_hat_lr = np.asarray(theta_hat_lr, dtype=float)
    dim_lr = len(theta_hat_lr)
    prior_mean_lr = np.asarray(prior_mean_lr, dtype=float)
    prior_std_lr = np.ones(dim_lr, dtype=float) * float(prior_std_scalar)

    fun = lambda th: neg_log_posterior_lr(
        th, x_exp, c_exp, sigma, t_max, nx, nt_save, phase_width, prior_mean_lr, prior_std_lr
    )
    step = np.ones(dim_lr, dtype=float) * float(hessian_step)
    H_raw = numerical_hessian(fun, theta_hat_lr, step)
    chi2, red_chi2 = gaussian_chi2_from_experiment_lr(
        theta_hat_lr, x_exp, c_exp, sigma, t_max, nx, nt_save, phase_width
    )
    return _posterior_result_from_hessian(theta_hat_lr, H_raw, n_samples, seed, chi2, red_chi2)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_laplace_reliability_lr(
    theta_hat_lr: np.ndarray,
    x_exp: np.ndarray,
    c_exp: np.ndarray,
    sigma: float,
    t_max: float,
    nx: int,
    nt_save: int,
    phase_width: float,
    prior_mean_lr: np.ndarray,
    prior_std_scalar: float,
    hessian_step: float,
    n_samples: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    return laplace_reliability_lr(
        theta_hat_lr=theta_hat_lr,
        x_exp=x_exp,
        c_exp=c_exp,
        sigma=sigma,
        t_max=t_max,
        nx=nx,
        nt_save=nt_save,
        phase_width=phase_width,
        prior_mean_lr=prior_mean_lr,
        prior_std_scalar=prior_std_scalar,
        hessian_step=hessian_step,
        n_samples=n_samples,
        seed=seed,
    )


def mcmc_reliability_lr(
    theta_start_lr: np.ndarray,
    x_exp: np.ndarray,
    c_exp: np.ndarray,
    sigma: float,
    t_max: float,
    nx: int,
    nt_save: int,
    phase_width: float,
    prior_mean_lr: np.ndarray,
    prior_std_scalar: float,
    n_steps: int,
    burn_in: int,
    proposal_std,
    seed: int,
    progress_bar=None,
    proposal_cov: Optional[np.ndarray] = None,
    proposal_cov_scale: Optional[float] = None,
    marginalize_sigma: bool = False,
    sigma_prior_scale: float = 0.1,
) -> Dict[str, np.ndarray]:
    """Random-walk Metropolis for two-region left/right D.

    ``proposal_cov`` (if supplied) overrides and uses a multivariate Gaussian
    proposal scaled by Roberts-Gelman-Gilks optimal 2.38^2/d.

    When ``marginalize_sigma`` is True, σ is jointly sampled.  The state
    vector is extended: [θ_L, θ_R, log(σ)].  Results include ``sigma_samples``.
    """
    rng = np.random.default_rng(seed + 909)
    prior_mean_lr = np.asarray(prior_mean_lr, dtype=float)
    theta_start_lr = np.asarray(theta_start_lr, dtype=float).ravel()
    dim_theta = len(theta_start_lr)
    prior_std_lr = np.ones(dim_theta, dtype=float) * float(prior_std_scalar)

    if marginalize_sigma:
        current = np.append(theta_start_lr, np.log(max(float(sigma), 1.0e-10)))
        dim = dim_theta + 1
    else:
        current = theta_start_lr.copy()
        dim = dim_theta

    if proposal_cov_scale is None:
        proposal_cov_scale = 2.38 ** 2 / max(dim, 1)

    if proposal_cov is not None:
        _cov = np.asarray(proposal_cov, dtype=float)
        # Fix #3: augment proposal_cov with sigma block when marginalize_sigma=True
        if marginalize_sigma and _cov.shape == (dim_theta, dim_theta):
            _cov_aug = np.zeros((dim, dim), dtype=float)
            _cov_aug[:dim_theta, :dim_theta] = _cov
            _cov_aug[dim_theta, dim_theta] = 0.01
            _cov = _cov_aug
        L, proposal_scale, mode = _mcmc_proposal_from_cov(
            _cov * float(proposal_cov_scale),
            proposal_std, dim=dim,
        )
    else:
        ps_arr = np.asarray(proposal_std, dtype=float).ravel()
        if ps_arr.size == 1:
            ps_arr = np.full(dim_theta, float(ps_arr[0]))
        if marginalize_sigma:
            ps = np.append(ps_arr, 0.1)
        else:
            ps = ps_arr
        L, proposal_scale, mode = _mcmc_proposal_from_cov(None, ps, dim=dim)

    def _lp(state):
        if marginalize_sigma:
            return -neg_log_posterior_marginal_sigma_lr(
                state, x_exp, c_exp, t_max, nx, nt_save, phase_width,
                prior_mean_lr, prior_std_lr, sigma_prior_scale,
            )
        else:
            return -neg_log_posterior_lr(
                state, x_exp, c_exp, sigma, t_max, nx, nt_save, phase_width,
                prior_mean_lr, prior_std_lr,
            )

    current_lp = _lp(current)

    samples = []
    accepted = 0
    n_steps = int(n_steps)
    burn_in = int(burn_in)

    for i in range(n_steps):
        if L is not None:
            proposal = current + L @ rng.standard_normal(dim)
        else:
            proposal = current + rng.normal(0.0, proposal_scale, size=dim)
        proposal_lp = _lp(proposal)
        if np.log(rng.uniform()) < proposal_lp - current_lp:
            current = proposal
            current_lp = proposal_lp
            accepted += 1

        if i >= burn_in:
            samples.append(current.copy())

        if progress_bar is not None and (i % max(1, n_steps // 100) == 0 or i == n_steps - 1):
            progress_bar.progress((i + 1) / n_steps)

    samples = np.asarray(samples, dtype=float)
    if samples.ndim == 1:
        samples = samples.reshape(0, dim_theta + (1 if marginalize_sigma else 0))
    sigma_eff = sigma
    if marginalize_sigma and samples.shape[0] > 0:
        sigma_eff = float(np.exp(np.median(samples[:, -1])))

    # P12 fix: evaluate chi2 at posterior median theta (not starting point)
    theta_for_chi2 = theta_start_lr
    if samples.shape[0] > 5:
        theta_for_chi2 = np.median(samples[:, :dim_theta], axis=0)
    chi2, red_chi2 = gaussian_chi2_from_experiment_lr(
        theta_for_chi2, x_exp, c_exp, sigma_eff, t_max, nx, nt_save, phase_width
    )
    result = {
        "theta_hat": np.asarray(theta_start_lr, dtype=float),
        "samples": samples[:, :dim_theta] if marginalize_sigma else samples,
        "acceptance_rate": np.array([accepted / max(n_steps, 1)]),
        "chi2": np.array([chi2]),
        "reduced_chi2": np.array([red_chi2]),
    }
    if marginalize_sigma and samples.shape[0] > 0:
        result["sigma_samples"] = np.exp(samples[:, -1])
        result["sigma_median"] = np.array([sigma_eff])
    return result


def posterior_band_from_samples_lr(
    theta_samples_lr: np.ndarray,
    x: np.ndarray,
    t_max: float,
    nx: int,
    nt_save: int,
    phase_width: float,
    max_samples: int = 80,
    progress_bar=None,
) -> Dict[str, np.ndarray]:
    # Defensive: force scalar args to native Python types
    t_max = float(t_max); nx = int(nx); nt_save = int(nt_save); phase_width = float(phase_width)
    if theta_samples_lr is None or len(theta_samples_lr) == 0:
        nan = np.full((len(x), 3), np.nan)
        return {"q025": nan, "q500": nan, "q975": nan}

    idx = np.linspace(0, len(theta_samples_lr) - 1, min(int(max_samples), len(theta_samples_lr))).astype(int)
    profiles = []
    for k, i in enumerate(idx):
        profiles.append(
            predict_final_profile_from_theta_lr(
                theta_samples_lr[i], x, t_max, nx, nt_save, phase_width, use_cache=False
            )
        )
        if progress_bar is not None:
            progress_bar.progress((k + 1) / len(idx))

    profiles = np.stack(profiles, axis=0)
    return {
        "q025": np.quantile(profiles, 0.025, axis=0),
        "q500": np.quantile(profiles, 0.500, axis=0),
        "q975": np.quantile(profiles, 0.975, axis=0),
    }


def _lr_param_names(dim: int) -> list:
    if dim >= 8:
        return [
            "logD_NiNi_left", "logD_TaTa_left", "rho12_raw_left", "rho21_raw_left",
            "logD_NiNi_right", "logD_TaTa_right", "rho12_raw_right", "rho21_raw_right",
        ][:dim]
    return [
        "logD_NiNi_left", "logD_TaTa_left", "rho_raw_left",
        "logD_NiNi_right", "logD_TaTa_right", "rho_raw_right",
    ][:dim]


def reliability_summary_table_lr(label: str, rel: Dict[str, np.ndarray]) -> pd.DataFrame:
    samples = np.asarray(rel["samples"], dtype=float)
    dim = samples.shape[1] if samples.ndim == 2 else 6
    names = _lr_param_names(dim)
    rows = []
    for j, name in enumerate(names):
        vals = samples[:, j]
        rows.append(
            {
                "method": label,
                "parameter": name,
                "q025": float(np.quantile(vals, 0.025)),
                "median": float(np.quantile(vals, 0.5)),
                "q975": float(np.quantile(vals, 0.975)),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
            }
        )
    return pd.DataFrame(rows)


def mcmc_trace_plot_lr(theta_samples_lr: np.ndarray):
    fig = go.Figure()
    if theta_samples_lr is None or len(theta_samples_lr) == 0:
        return clean_layout(fig, "Left/right MCMC trace plot", 430)
    dim = theta_samples_lr.shape[1] if theta_samples_lr.ndim == 2 else 6
    names = _lr_param_names(dim)
    steps = np.arange(len(theta_samples_lr))
    for j, name in enumerate(names):
        if j >= theta_samples_lr.shape[1]:
            break
        fig.add_trace(go.Scatter(x=steps, y=theta_samples_lr[:, j], mode="lines", name=name, line=dict(width=1.5)))
    fig.update_xaxes(title="saved MCMC sample index")
    fig.update_yaxes(title="theta value")
    return clean_layout(fig, f"Left/right {dim}-parameter MCMC trace", 450)


def likelihood_contour_grid(
    theta_center: np.ndarray,
    x_exp: np.ndarray,
    c_exp: np.ndarray,
    sigma: float,
    t_max: float,
    nx: int,
    nt_save: int,
    axis_i: int,
    axis_j: int,
    half_width: float,
    n_grid: int,
    prior_mean: Optional[np.ndarray] = None,
    prior_std_scalar: Optional[float] = None,
    progress_bar=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a 2D NLL or negative log posterior contour grid.

    The third parameter is fixed at theta_center. If prior_mean is supplied,
    the grid shows negative log posterior; otherwise negative log likelihood.
    """
    a = np.linspace(theta_center[axis_i] - half_width, theta_center[axis_i] + half_width, n_grid)
    b = np.linspace(theta_center[axis_j] - half_width, theta_center[axis_j] + half_width, n_grid)
    Z = np.zeros((n_grid, n_grid), dtype=float)

    total = int(n_grid * n_grid)
    done = 0
    for iy, bv in enumerate(b):
        for ix, av in enumerate(a):
            th = theta_center.copy()
            th[axis_i] = av
            th[axis_j] = bv
            if prior_mean is None or prior_std_scalar is None:
                Z[iy, ix] = gaussian_nll_from_experiment(th, x_exp, c_exp, sigma, t_max, nx, nt_save)
            else:
                Z[iy, ix] = neg_log_posterior(
                    th, x_exp, c_exp, sigma, t_max, nx, nt_save,
                    prior_mean, np.ones(len(theta_center)) * prior_std_scalar
                )
            done += 1
            if progress_bar is not None:
                progress_bar.progress(done / max(total, 1))

    Z = Z - np.nanmin(Z)
    return a, b, Z


# =============================================================================
# Plot helpers
# =============================================================================

MPL_COLORS = {"Co": "black", "Ni": "#2b83ba", "Ta": "#4daf4a"}
MPL_MARKERS = {"Co": "o", "Ni": "s", "Ta": "^"}
_PUB_FONTSIZE = 18


def _pub_fig_to_bytes(fig: plt.Figure, dpi: int = 200) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    return buf.read()


def pub_profile_figure(
    dist: np.ndarray,
    C_fdm: np.ndarray,
    C_pinn: np.ndarray,
    dist_exp: np.ndarray,
    c_exp: np.ndarray,
    title: str = "",
    figsize: Tuple[float, float] = (10, 6),
) -> plt.Figure:
    """Publication-quality single-time profile figure."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    for j, comp in enumerate(COMPONENTS):
        ax.plot(dist_exp, c_exp[:, j], MPL_MARKERS[comp],
                color=MPL_COLORS[comp], markersize=6, markerfacecolor="none",
                markeredgewidth=1.5, label=f"Exp. {comp}", zorder=3)
    for j, comp in enumerate(COMPONENTS):
        ax.plot(dist, C_fdm[:, j], "-", color=MPL_COLORS[comp], linewidth=2.5,
                label=f"FDM {comp}")
        ax.plot(dist, C_pinn[:, j], "--", color=MPL_COLORS[comp], linewidth=2.0,
                label=f"PINNs {comp}")
    ax.set_xlabel("Distance (µm)", fontsize=_PUB_FONTSIZE)
    ax.set_ylabel("Mole fraction", fontsize=_PUB_FONTSIZE)
    ax.set_ylim(-0.04, 1.04)
    ax.tick_params(labelsize=_PUB_FONTSIZE - 2)
    ax.legend(fontsize=_PUB_FONTSIZE - 4, ncol=3, loc="upper right", framealpha=0.8)
    if title:
        ax.set_title(title, fontsize=_PUB_FONTSIZE + 2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def pub_multitime_figure(
    dist: np.ndarray,
    t_grid: np.ndarray,
    C_fdm: np.ndarray,
    C_pinn: np.ndarray,
    n_slices: int = 5,
    n_cols: int = 3,
    tau_max: float = 1.0,
    annealing_time_h: float = 160.0,
    figsize_per_panel: Tuple[float, float] = (5.5, 4.0),
) -> plt.Figure:
    """Publication-quality multi-time panel figure."""
    indices = sorted(set(np.linspace(0, len(t_grid) - 1, n_slices).astype(int).tolist()))
    n_panels = len(indices)
    n_rows = max(1, (n_panels + n_cols - 1) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
                             squeeze=False)
    for idx_panel, idx_time in enumerate(indices):
        r, c = divmod(idx_panel, n_cols)
        ax = axes[r][c]
        for j, comp in enumerate(COMPONENTS):
            ax.plot(dist, C_fdm[idx_time, :, j], "-", color=MPL_COLORS[comp],
                    linewidth=2.0, label=f"FDM {comp}" if idx_panel == 0 else "")
            ax.plot(dist, C_pinn[idx_time, :, j], "--", color=MPL_COLORS[comp],
                    linewidth=1.8, label=f"PINNs {comp}" if idx_panel == 0 else "")
        tlabel = format_time_label(float(t_grid[idx_time]), tau_max, annealing_time_h)
        ax.set_title(tlabel, fontsize=_PUB_FONTSIZE - 2)
        ax.set_ylim(-0.04, 1.04)
        ax.set_xlabel("Distance (µm)", fontsize=_PUB_FONTSIZE - 4)
        ax.set_ylabel("Mole fraction", fontsize=_PUB_FONTSIZE - 4)
        ax.tick_params(labelsize=_PUB_FONTSIZE - 5)
        ax.grid(True, alpha=0.3)
    for idx_panel in range(n_panels, n_rows * n_cols):
        r, c = divmod(idx_panel, n_cols)
        axes[r][c].set_visible(False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", fontsize=_PUB_FONTSIZE - 4,
                   ncol=len(COMPONENTS) * 2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return fig


def pub_omega_convergence_figure(
    history: pd.DataFrame,
    pair_names: List[str],
    figsize: Tuple[float, float] = (10, 5),
) -> plt.Figure:
    """Publication-quality Omega convergence figure (loss + Omega)."""
    fig, (ax_loss, ax_omega) = plt.subplots(1, 2, figsize=figsize)

    for col in ["loss", "data", "ic", "physics"]:
        if col in history.columns:
            ax_loss.semilogy(history["epoch"], history[col], linewidth=1.8, label=col)
    ax_loss.set_xlabel("Epoch", fontsize=_PUB_FONTSIZE - 2)
    ax_loss.set_ylabel("Loss", fontsize=_PUB_FONTSIZE - 2)
    ax_loss.set_title("Training loss", fontsize=_PUB_FONTSIZE)
    ax_loss.legend(fontsize=_PUB_FONTSIZE - 5)
    ax_loss.tick_params(labelsize=_PUB_FONTSIZE - 4)
    ax_loss.grid(True, alpha=0.3)

    omega_cols_l = [f"{pn}_left" for pn in ["Omega_CoNi", "Omega_CoTa", "Omega_NiTa"]]
    omega_cols_r = [f"{pn}_right" for pn in ["Omega_CoNi", "Omega_CoTa", "Omega_NiTa"]]
    for k, pname in enumerate(pair_names):
        if omega_cols_l[k] in history.columns:
            ax_omega.plot(history["epoch"], history[omega_cols_l[k]], linewidth=1.8,
                          label=f"{pname} left")
        if omega_cols_r[k] in history.columns:
            ax_omega.plot(history["epoch"], history[omega_cols_r[k]], linewidth=1.8,
                          linestyle="--", label=f"{pname} right")
    ax_omega.set_xlabel("Epoch", fontsize=_PUB_FONTSIZE - 2)
    ax_omega.set_ylabel("Ω value", fontsize=_PUB_FONTSIZE - 2)
    ax_omega.set_title("Ω convergence", fontsize=_PUB_FONTSIZE)
    ax_omega.legend(fontsize=_PUB_FONTSIZE - 5)
    ax_omega.tick_params(labelsize=_PUB_FONTSIZE - 4)
    ax_omega.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def pub_nll_convergence_figure(
    nll_history: List[float],
    figsize: Tuple[float, float] = (8, 4),
) -> plt.Figure:
    """Publication-quality NLL convergence during FDM refinement."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(range(len(nll_history)), nll_history, "-", linewidth=1.5, color="#2b83ba")
    ax.set_xlabel("Function evaluation", fontsize=_PUB_FONTSIZE - 2)
    ax.set_ylabel("NLL", fontsize=_PUB_FONTSIZE - 2)
    ax.set_title("FDM likelihood refinement convergence", fontsize=_PUB_FONTSIZE)
    ax.tick_params(labelsize=_PUB_FONTSIZE - 4)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def pub_credible_band_figure(
    dist: np.ndarray,
    C_pinn_final: np.ndarray,
    band: Dict[str, np.ndarray],
    title: str = "",
    figsize: Tuple[float, float] = (10, 6),
) -> plt.Figure:
    """Publication-quality credible band figure."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    alpha_fill = [0.15, 0.20, 0.18]
    for j, comp in enumerate(COMPONENTS):
        ax.fill_between(dist, band["q025"][:, j], band["q975"][:, j],
                        color=MPL_COLORS[comp], alpha=alpha_fill[j], label=f"95% CI {comp}")
        ax.plot(dist, band["q500"][:, j], "-", color=MPL_COLORS[comp],
                linewidth=2.0, label=f"Median {comp}")
        ax.plot(dist, C_pinn_final[:, j], "--", color=MPL_COLORS[comp],
                linewidth=1.8, label=f"PINNs {comp}")
    ax.set_xlabel("Distance (µm)", fontsize=_PUB_FONTSIZE)
    ax.set_ylabel("Mole fraction", fontsize=_PUB_FONTSIZE)
    ax.set_ylim(-0.04, 1.04)
    ax.tick_params(labelsize=_PUB_FONTSIZE - 2)
    ax.legend(fontsize=_PUB_FONTSIZE - 5, ncol=3, loc="upper right", framealpha=0.8)
    if title:
        ax.set_title(title, fontsize=_PUB_FONTSIZE + 2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig

def clean_layout(fig: go.Figure, title: str, height: int = 430, legend_y: float = -0.25) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left", y=0.98, yanchor="top"),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.68)",
        font=dict(color="#334155"),
        legend=dict(
            orientation="h",
            x=0.0,
            y=legend_y,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.60)",
            bordercolor="rgba(148,163,184,0.25)",
            borderwidth=1,
        ),
        margin=dict(l=12, r=12, t=75, b=130),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.25)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.25)", zeroline=False)
    return fig



def fdm_teacher_preview_plot(
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    C_fdm: np.ndarray,
    x_exp: np.ndarray,
    C_exp: np.ndarray,
    span_um: float = 800.0,
    n_time_lines: int = 4,
    annealing_time_h: float = 160.0,
):
    """Preview FDM teacher profiles and pseudo experimental points before PINNs training."""
    fig = go.Figure()
    dist = distance_um_from_x(x_grid, span_um)
    exp_dist = distance_um_from_x(x_exp, span_um)

    if len(t_grid) <= 1:
        idxs = [0]
    else:
        idxs = np.unique(np.linspace(0, len(t_grid) - 1, int(max(n_time_lines, 2))).astype(int)).tolist()

    for idx in idxs:
        line_width = 1.4 if idx != idxs[-1] else 3.0
        dash = "dot" if idx != idxs[-1] else "solid"
        label_time = format_time_label(float(t_grid[idx]), float(t_grid[-1]), float(annealing_time_h))
        for j, comp in enumerate(COMPONENTS):
            fig.add_trace(
                go.Scatter(
                    x=dist,
                    y=C_fdm[idx, :, j],
                    mode="lines",
                    name=f"FDM {comp} {label_time}",
                    line=dict(width=line_width, dash=dash, color=COLORS[comp]),
                    opacity=0.45 if idx != idxs[-1] else 1.0,
                    showlegend=(idx == idxs[-1]),
                )
            )

    for j, comp in enumerate(COMPONENTS):
        fig.add_trace(
            go.Scatter(
                x=exp_dist,
                y=C_exp[:, j],
                mode="markers",
                name=f"pseudo-exp {comp}",
                marker=dict(
                    symbol=SYMBOLS[comp],
                    size=7,
                    color=COLORS[comp],
                    line=dict(width=1.2),
                ),
            )
        )

    fig.update_xaxes(title="Distance from initial interface (µm)")
    fig.update_yaxes(title="Mole fraction", range=[-0.04, 1.04])
    return clean_layout(
        fig,
        f"Preview after FDM: teacher profiles and pseudo experimental points; final = {format_time_label(float(t_grid[-1]), float(t_grid[-1]), float(annealing_time_h))}",
        520,
    )


def fdm_teacher_preview_difference_plot(
    x_grid: np.ndarray,
    C_fdm_final: np.ndarray,
    x_exp: np.ndarray,
    C_exp: np.ndarray,
    span_um: float = 800.0,
):
    """Preview pseudo-experimental residual against FDM final profile."""
    dist_exp = distance_um_from_x(x_exp, span_um)
    fig = go.Figure()
    for j, comp in enumerate(COMPONENTS):
        interp_final = np.interp(x_exp.ravel(), x_grid.ravel(), C_fdm_final[:, j])
        diff = C_exp[:, j] - interp_final
        fig.add_trace(
            go.Scatter(
                x=dist_exp,
                y=diff,
                mode="markers",
                name=f"{comp}: pseudo-exp - FDM",
                marker=dict(symbol=SYMBOLS[comp], size=7, color=COLORS[comp]),
            )
        )
    fig.add_hline(y=0.0, line_dash="dash", opacity=0.5)
    fig.update_xaxes(title="Distance from initial interface (µm)")
    fig.update_yaxes(title="Mole fraction residual")
    return clean_layout(fig, "Preview residual: pseudo experimental points - FDM final profile", 360)



def fig11_profile_plot(
    x,
    C_fdm_final,
    C_pinn_final,
    x_exp,
    C_exp,
    span_um: float = 800.0,
    C_zero_final: Optional[np.ndarray] = None,
):
    dist = distance_um_from_x(x, span_um)
    dist_exp = distance_um_from_x(x_exp, span_um)
    fig = go.Figure()

    for j, comp in enumerate(COMPONENTS):
        fig.add_trace(
            go.Scatter(
                x=dist_exp,
                y=C_exp[:, j],
                mode="markers",
                name=f"Exp. {comp}",
                marker=dict(symbol=SYMBOLS[comp], size=7, color=COLORS[comp], line=dict(width=1.4)),
            )
        )

    for j, comp in enumerate(COMPONENTS):
        fig.add_trace(
            go.Scatter(
                x=dist,
                y=C_fdm_final[:, j],
                mode="lines",
                name=f"FDM {comp}",
                line=dict(width=3, color=COLORS[comp]),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=dist,
                y=C_pinn_final[:, j],
                mode="lines",
                name=f"PINNs {comp}",
                line=dict(width=2.5, dash="dash", color=COLORS[comp]),
            )
        )
        if C_zero_final is not None:
            fig.add_trace(
                go.Scatter(
                    x=dist,
                    y=C_zero_final[:, j],
                    mode="lines",
                    name=f"Zero-interaction {comp}",
                    line=dict(width=2.2, dash="dot", color=COLORS[comp]),
                )
            )

    fig.update_xaxes(title="Distance (µm)", range=[-0.52 * span_um, 0.52 * span_um])
    fig.update_yaxes(title="Mole Fraction", range=[-0.04, 1.04])
    fig.add_annotation(x=0, y=1.0, text="1200 °C for 160 h style", showarrow=False, xanchor="left")
    return clean_layout(fig, "Fig.11-style Co / Ni-0.10Ta diffusion-couple profile", 560, legend_y=-0.24)


def multi_time_profile_plot(x, t, C_fdm, C_pinn, indices, span_um: float = 800.0, annealing_time_h: float = 160.0):
    dist = distance_um_from_x(x, span_um)
    fig = go.Figure()

    for j, comp in enumerate(COMPONENTS):
        for idx in indices:
            label = format_time_label(float(t[idx]), float(t[-1]), float(annealing_time_h))
            fig.add_trace(
                go.Scatter(
                    x=dist,
                    y=C_fdm[idx, :, j],
                    mode="lines",
                    name=f"FDM {comp} {label}",
                    line=dict(width=2.2, color=COLORS[comp]),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=dist,
                    y=C_pinn[idx, :, j],
                    mode="lines",
                    name=f"PINNs {comp} {label}",
                    line=dict(width=2.0, dash="dash", color=COLORS[comp]),
                )
            )

    fig.update_xaxes(title="Distance (µm)")
    fig.update_yaxes(title="Mole fraction")
    return clean_layout(fig, "Multi-time profile check: Co, Ni, and Ta", 540, legend_y=-0.42)


def single_time_profile_plot(
    x,
    t,
    C_fdm,
    C_pinn,
    idx: int,
    span_um: float = 800.0,
    x_exp: Optional[np.ndarray] = None,
    C_exp: Optional[np.ndarray] = None,
    C_zero_time: Optional[np.ndarray] = None,
    annealing_time_h: float = 160.0,
):
    """One profile plot for one selected time.

    This is useful for ternary diffusion-couple checks because a single combined
    multi-time plot becomes visually crowded. Each figure shows Co, Ni, and Ta
    for one time slice. Experimental-like markers are overlaid only for the
    final-time plot because the pseudo experimental data are final profiles.
    """
    dist = distance_um_from_x(x, span_um)
    fig = go.Figure()

    for j, comp in enumerate(COMPONENTS):
        fig.add_trace(
            go.Scatter(
                x=dist,
                y=C_fdm[idx, :, j],
                mode="lines",
                name=f"FDM {comp}",
                line=dict(width=3, color=COLORS[comp]),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=dist,
                y=C_pinn[idx, :, j],
                mode="lines",
                name=f"PINNs {comp}",
                line=dict(width=2.5, dash="dash", color=COLORS[comp]),
            )
        )
        if C_zero_time is not None:
            fig.add_trace(
                go.Scatter(
                    x=dist,
                    y=C_zero_time[:, j],
                    mode="lines",
                    name=f"Zero-interaction {comp}",
                    line=dict(width=2.2, dash="dot", color=COLORS[comp]),
                )
            )

    # Experimental-like points exist only for the final annealing profile.
    if idx == len(t) - 1 and x_exp is not None and C_exp is not None:
        dist_exp = distance_um_from_x(x_exp, span_um)
        for j, comp in enumerate(COMPONENTS):
            fig.add_trace(
                go.Scatter(
                    x=dist_exp,
                    y=C_exp[:, j],
                    mode="markers",
                    name=f"Exp. {comp}",
                    marker=dict(
                        symbol=SYMBOLS[comp],
                        size=6,
                        color=COLORS[comp],
                        line=dict(width=1.2),
                    ),
                )
            )

    fig.update_xaxes(title="Distance (µm)")
    fig.update_yaxes(title="Mole fraction", range=[-0.04, 1.04])
    return clean_layout(
        fig,
        f"Profiles at {format_time_label(float(t[idx]), float(t[-1]), float(annealing_time_h))}",
        height=430,
        legend_y=-0.32,
    )



def classify_region_from_x(x_vals: np.ndarray, interface: float = 0.5, width: float = 0.03) -> np.ndarray:
    """Classify points into left/interface/right for two-region diagnostics."""
    x_vals = np.asarray(x_vals, dtype=float).reshape(-1)
    w = max(float(width), 0.0)
    region = np.full(x_vals.shape, "interface", dtype=object)
    region[x_vals < interface - w] = "left"
    region[x_vals > interface + w] = "right"
    return region


def predicted_vs_experiment_by_component_plot(
    C_pred_final: np.ndarray,
    x_grid: np.ndarray,
    x_exp: np.ndarray,
    C_exp: np.ndarray,
    component_index: int,
    component_name: str,
    region_width: float = 0.03,
):
    """Predicted-vs-experiment plot for one component with region labels."""
    pred = np.interp(x_exp.ravel(), x_grid.ravel(), C_pred_final[:, component_index])
    exp = C_exp[:, component_index]
    regions = classify_region_from_x(x_exp.ravel(), width=region_width)

    fig = go.Figure()
    region_symbols = {"left": "circle", "interface": "diamond", "right": "square"}
    for region_name in ["left", "interface", "right"]:
        mask = regions == region_name
        if np.any(mask):
            fig.add_trace(
                go.Scatter(
                    x=pred[mask],
                    y=exp[mask],
                    mode="markers",
                    name=f"{region_name}",
                    marker=dict(
                        symbol=region_symbols[region_name],
                        size=8,
                        color=COLORS[component_name],
                        line=dict(width=1.1),
                    ),
                )
            )

    lim_max = max(float(np.nanmax(pred)), float(np.nanmax(exp)), 1.0e-6)
    fig.add_trace(
        go.Scatter(
            x=[0.0, lim_max],
            y=[0.0, lim_max],
            mode="lines",
            name="ideal y=x",
            line=dict(dash="dash", color="gray"),
        )
    )
    fig.update_xaxes(title=f"PINNs predicted {component_name} mole fraction")
    fig.update_yaxes(title=f"Experiment-like {component_name} mole fraction")
    return clean_layout(fig, f"Predicted vs experiment by region: {component_name}", 420)


def residual_summary_by_component_region(
    C_pred_final: np.ndarray,
    x_grid: np.ndarray,
    x_exp: np.ndarray,
    C_exp: np.ndarray,
    region_width: float = 0.03,
) -> pd.DataFrame:
    """Residual summary split by component and left/interface/right region."""
    regions = classify_region_from_x(x_exp.ravel(), width=region_width)
    rows = []
    for j, comp in enumerate(COMPONENTS):
        pred = np.interp(x_exp.ravel(), x_grid.ravel(), C_pred_final[:, j])
        exp = C_exp[:, j]
        res = pred - exp
        for region_name in ["left", "interface", "right", "all"]:
            if region_name == "all":
                mask = np.ones_like(res, dtype=bool)
            else:
                mask = regions == region_name
            if not np.any(mask):
                continue
            rr = res[mask]
            rows.append(
                {
                    "component": comp,
                    "region": region_name,
                    "n": int(np.sum(mask)),
                    "mean residual": float(np.mean(rr)),
                    "MAE": float(np.mean(np.abs(rr))),
                    "RMSE": float(np.sqrt(np.mean(rr**2))),
                    "max abs residual": float(np.max(np.abs(rr))),
                }
            )
    return pd.DataFrame(rows)



def final_difference_plot(x, C_diff_final, span_um: float = 800.0):
    dist = distance_um_from_x(x, span_um)
    fig = go.Figure()
    for j, comp in enumerate(COMPONENTS):
        fig.add_trace(
            go.Scatter(
                x=dist,
                y=C_diff_final[:, j],
                mode="lines",
                name=f"{comp}: PINN-FDM",
                line=dict(width=2.5, color=COLORS[comp]),
            )
        )
    fig.add_hline(y=0.0, line_dash="dash", opacity=0.6)
    fig.update_xaxes(title="Distance (µm)")
    fig.update_yaxes(title="Mole fraction difference")
    return clean_layout(fig, "Final-profile difference: PINN - FDM", 430)


def zero_interaction_difference_plot(x, C_pinn_final, C_zero_final, span_um: float = 800.0):
    """Difference between optimized PINN profile and zero-interaction reference."""
    dist = distance_um_from_x(x, span_um)
    fig = go.Figure()
    for j, comp in enumerate(COMPONENTS):
        fig.add_trace(
            go.Scatter(
                x=dist,
                y=C_pinn_final[:, j] - C_zero_final[:, j],
                mode="lines",
                name=f"{comp}: PINN - zero-interaction",
                line=dict(width=2.5, color=COLORS[comp]),
            )
        )
    fig.add_hline(y=0.0, line_dash="dash", opacity=0.6)
    fig.update_xaxes(title="Distance (µm)")
    fig.update_yaxes(title="Mole fraction difference")
    return clean_layout(fig, "Effect of interaction term: optimized PINN - zero-interaction reference", 430)


def heatmap_diff_plot(x, t, C_diff, comp_index: int, span_um: float = 800.0):
    dist = distance_um_from_x(x, span_um)
    comp = COMPONENTS[comp_index]
    fig = go.Figure(
        data=[
            go.Heatmap(
                x=dist,
                y=t,
                z=C_diff[:, :, comp_index],
                colorscale="RdBu",
                zmid=0,
                colorbar=dict(title="diff"),
            )
        ]
    )
    fig.update_xaxes(title="Distance (µm)")
    fig.update_yaxes(title="normalized time")
    return clean_layout(fig, f"Difference map: {comp} PINN - FDM", 430)


def loss_plot(hist: pd.DataFrame):
    fig = go.Figure()
    for col in ["loss", "data", "ic", "bc", "physics"]:
        fig.add_trace(go.Scatter(x=hist["epoch"], y=hist[col], mode="lines", name=col))
    fig.update_xaxes(title="epoch")
    fig.update_yaxes(title="loss", type="log")
    return clean_layout(fig, "Loss functions", 430)


def D_history_plot(hist: pd.DataFrame, D_true: np.ndarray):
    fig = go.Figure()
    entries = [("D_NiNi", D_true[0, 0]), ("D_NiTa", D_true[0, 1]), ("D_TaTa", D_true[1, 1])]
    y_values = []
    for name, true_val in entries:
        fig.add_trace(go.Scatter(x=hist["epoch"], y=hist[name], mode="lines", name=f"PINN {name}"))
        fig.add_hline(y=float(true_val), line_dash="dash", opacity=0.45)
        y_values.extend(np.asarray(hist[name], dtype=float).tolist())
        y_values.append(float(true_val))

    y_values = np.asarray([v for v in y_values if np.isfinite(v)], dtype=float)
    if len(y_values) > 4:
        lo, hi = np.quantile(y_values, [0.02, 0.98])
        if np.isclose(lo, hi):
            pad = max(abs(hi), 1.0) * 0.1
            lo, hi = lo - pad, hi + pad
        else:
            pad = 0.12 * (hi - lo)
            lo, hi = lo - pad, hi + pad
        fig.update_yaxes(range=[lo, hi])
    fig.add_annotation(
        xref="paper", yref="paper", x=0.01, y=0.98,
        text="robust y-range: 2–98% quantile",
        showarrow=False,
        font=dict(size=11, color="#64748b"),
    )
    fig.update_xaxes(title="epoch")
    fig.update_yaxes(title="diffusion matrix value, linear scale")
    return clean_layout(fig, "Estimated diffusion-matrix parameters", 430)


def predicted_vs_experiment_plot(C_pinn_final: np.ndarray, x: np.ndarray, x_exp: np.ndarray, C_exp: np.ndarray):
    C_pred_at_exp = np.column_stack([np.interp(x_exp, x, C_pinn_final[:, j]) for j in range(3)])
    fig = go.Figure()
    for j, comp in enumerate(COMPONENTS):
        fig.add_trace(
            go.Scatter(
                x=C_pred_at_exp[:, j],
                y=C_exp[:, j],
                mode="markers",
                name=comp,
                marker=dict(symbol="circle-open", size=7, color=COLORS[comp], line=dict(width=1.4)),
            )
        )
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="y=x", line=dict(dash="dash", color="gray")))
    fig.update_xaxes(title="PINN predicted mole fraction", range=[-0.03, 1.03])
    fig.update_yaxes(title="experiment-like mole fraction", range=[-0.03, 1.03])
    return clean_layout(fig, "Experiment-like data vs PINN prediction", 460)


def diffusion_matrix_table(D_true: np.ndarray, D_pinn: np.ndarray) -> pd.DataFrame:
    rows = []
    labels = [["D_NiNi", "D_NiTa"], ["D_TaNi", "D_TaTa"]]
    for i in range(2):
        for j in range(2):
            rows.append(
                {
                    "parameter": labels[i][j],
                    "FDM true": D_true[i, j],
                    "PINN estimated": D_pinn[i, j],
                    "absolute error": abs(D_pinn[i, j] - D_true[i, j]),
                }
            )
    return pd.DataFrame(rows)




# =============================================================================
# CALPHAD / DICTRA comparison helpers
# =============================================================================

def physical_to_normalized_D(D_phys: np.ndarray, length_um: float, time_h: float) -> np.ndarray:
    """Convert physical diffusivity [m^2/s] to normalized diffusivity.

    x_norm = x / L, t_norm = t / t_scale:
        D_norm = D_phys * t_scale / L^2
    """
    L = max(float(length_um), 1.0e-12) * 1.0e-6
    ts = max(float(time_h), 1.0e-12) * 3600.0
    return np.asarray(D_phys, dtype=float) * ts / (L * L)


def normalized_to_physical_D(D_norm: np.ndarray, length_um: float, time_h: float) -> np.ndarray:
    """Convert normalized diffusivity to physical diffusivity [m^2/s]."""
    L = max(float(length_um), 1.0e-12) * 1.0e-6
    ts = max(float(time_h), 1.0e-12) * 3600.0
    return np.asarray(D_norm, dtype=float) * (L * L) / ts


def normalized_scalar_to_physical_D(value_norm: float, length_um: float, time_h: float) -> float:
    """Convert one normalized diffusivity scalar to physical [m^2/s]."""
    L = max(float(length_um), 1.0e-12) * 1.0e-6
    ts = max(float(time_h), 1.0e-12) * 3600.0
    return float(value_norm) * (L * L) / ts


def safe_log_from_positive(value: float, fallback: float) -> float:
    value = float(value)
    if np.isfinite(value) and value > 0:
        return float(np.log(value))
    return float(fallback)


def calphad_required_columns(kind: str) -> list:
    if kind == "matrix":
        return ["D_NiNi", "D_NiTa", "D_TaNi", "D_TaTa"]
    if kind == "profile":
        return ["distance_um", "Co", "Ni", "Ta"]
    return []


def validate_columns(df: pd.DataFrame, required: list) -> Tuple[bool, str]:
    missing = [c for c in required if c not in df.columns]
    if missing:
        return False, "Missing columns: " + ", ".join(missing)
    return True, ""


def representative_calphad_matrix(df: pd.DataFrame) -> np.ndarray:
    """Use median values as representative CALPHAD/DICTRA D matrix."""
    return np.array(
        [
            [float(np.nanmedian(df["D_NiNi"])), float(np.nanmedian(df["D_NiTa"]))],
            [float(np.nanmedian(df["D_TaNi"])), float(np.nanmedian(df["D_TaTa"]))],
        ],
        dtype=float,
    )


def matrix_rows_for_comparison(
    D_pinn_norm: np.ndarray,
    D_true_norm: np.ndarray,
    D_zero_norm: Optional[np.ndarray],
    D_calphad_phys: Optional[np.ndarray],
    length_um: float,
    time_h: float,
) -> pd.DataFrame:
    labels = [["D_NiNi", "D_NiTa"], ["D_TaNi", "D_TaTa"]]
    D_pinn_phys = normalized_to_physical_D(D_pinn_norm, length_um, time_h)
    D_true_phys = normalized_to_physical_D(D_true_norm, length_um, time_h)
    D_zero_phys = None if D_zero_norm is None else normalized_to_physical_D(D_zero_norm, length_um, time_h)

    rows = []
    for i in range(2):
        for j in range(2):
            row = {
                "parameter": labels[i][j],
                "PINNs normalized": D_pinn_norm[i, j],
                "PINNs physical [m2/s]": D_pinn_phys[i, j],
                "FDM true physical [m2/s]": D_true_phys[i, j],
            }
            if D_zero_phys is not None:
                row["zero-interaction physical [m2/s]"] = D_zero_phys[i, j]
            if D_calphad_phys is not None:
                c = D_calphad_phys[i, j]
                p = D_pinn_phys[i, j]
                row["CALPHAD/DICTRA physical [m2/s]"] = c
                row["PINNs - CALPHAD [m2/s]"] = p - c
                row["relative error vs CALPHAD [%]"] = np.nan if abs(c) < 1.0e-300 else 100.0 * (p - c) / abs(c)
                row["sign agreement"] = "yes" if np.sign(p) == np.sign(c) or abs(p) < 1.0e-300 or abs(c) < 1.0e-300 else "no"
            rows.append(row)
    return pd.DataFrame(rows)


def D_matrix_bar_plot(compare_df: pd.DataFrame):
    fig = go.Figure()
    params = compare_df["parameter"].tolist()
    if "PINNs physical [m2/s]" in compare_df:
        fig.add_trace(go.Bar(x=params, y=compare_df["PINNs physical [m2/s]"], name="PINNs"))
    if "CALPHAD/DICTRA physical [m2/s]" in compare_df:
        fig.add_trace(go.Bar(x=params, y=compare_df["CALPHAD/DICTRA physical [m2/s]"], name="CALPHAD/DICTRA"))
    if "zero-interaction physical [m2/s]" in compare_df:
        fig.add_trace(go.Bar(x=params, y=compare_df["zero-interaction physical [m2/s]"], name="zero-interaction"))
    fig.update_yaxes(title="D [m²/s]", type="linear")
    return clean_layout(fig, "Physical diffusion matrix comparison", 450)


def calphad_D_composition_plot(df: pd.DataFrame, D_pinn_phys: np.ndarray):
    """Plot CALPHAD D values over composition/row index with PINNs constant D as lines."""
    fig = go.Figure()
    xcol = None
    for cand in ["x_Ta", "x_Ni", "distance_um", "index"]:
        if cand in df.columns:
            xcol = cand
            break
    if xcol is None:
        xvals = np.arange(len(df))
        xtitle = "row index"
    else:
        xvals = df[xcol].to_numpy()
        xtitle = xcol

    mapping = {
        "D_NiNi": (0, 0),
        "D_NiTa": (0, 1),
        "D_TaNi": (1, 0),
        "D_TaTa": (1, 1),
    }
    for name, (i, j) in mapping.items():
        if name in df.columns:
            fig.add_trace(go.Scatter(x=xvals, y=df[name], mode="lines+markers", name=f"CALPHAD {name}"))
            fig.add_trace(
                go.Scatter(
                    x=[np.nanmin(xvals), np.nanmax(xvals)],
                    y=[D_pinn_phys[i, j], D_pinn_phys[i, j]],
                    mode="lines",
                    name=f"PINNs {name}",
                    line=dict(dash="dash"),
                )
            )
    fig.update_xaxes(title=xtitle)
    fig.update_yaxes(title="D [m²/s]")
    return clean_layout(fig, "CALPHAD/DICTRA D(x) with PINNs constant-D reference", 520)


def dictra_profile_overlay_plot(
    profile_df: pd.DataFrame,
    x: np.ndarray,
    C_pinn_final: np.ndarray,
    x_exp: np.ndarray,
    C_exp: np.ndarray,
    C_zero_final: Optional[np.ndarray],
    span_um: float,
):
    dist = distance_um_from_x(x, span_um)
    fig = go.Figure()

    for j, comp in enumerate(COMPONENTS):
        fig.add_trace(
            go.Scatter(
                x=dist,
                y=C_pinn_final[:, j],
                mode="lines",
                name=f"PINNs {comp}",
                line=dict(width=2.8, dash="dash", color=COLORS[comp]),
            )
        )
        if C_zero_final is not None:
            fig.add_trace(
                go.Scatter(
                    x=dist,
                    y=C_zero_final[:, j],
                    mode="lines",
                    name=f"Zero {comp}",
                    line=dict(width=2.0, dash="dot", color=COLORS[comp]),
                )
            )
        fig.add_trace(
            go.Scatter(
                x=profile_df["distance_um"],
                y=profile_df[comp],
                mode="lines",
                name=f"DICTRA/CALPHAD {comp}",
                line=dict(width=2.5, color=COLORS[comp]),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=distance_um_from_x(x_exp, span_um),
                y=C_exp[:, j],
                mode="markers",
                name=f"Exp-like {comp}",
                marker=dict(symbol=SYMBOLS[comp], size=6, color=COLORS[comp], line=dict(width=1.1)),
            )
        )

    fig.update_xaxes(title="Distance from interface / Matano plane (µm)")
    fig.update_yaxes(title="Mole fraction", range=[-0.04, 1.04])
    return clean_layout(fig, "Profile overlay: PINNs / zero-interaction / DICTRA-CALPHAD / experiment", 560)


def mobility_from_D_and_phi(D_phys: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Effective mobility-like matrix M_eff = D * inv(Phi).

    This is a diagnostic, not a direct DICTRA mobility database parameter.
    """
    phi = np.asarray(phi, dtype=float)
    return np.asarray(D_phys, dtype=float) @ np.linalg.pinv(phi)



def mcmc_trace_plot(theta_samples: np.ndarray):
    """Trace plot for MCMC theta samples."""
    fig = go.Figure()
    if theta_samples is None or len(theta_samples) == 0:
        return clean_layout(fig, "MCMC trace plot", 430)

    names = ["log_d11", "log_d22", "rho_raw"]
    steps = np.arange(len(theta_samples))
    for j, name in enumerate(names):
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=theta_samples[:, j],
                mode="lines",
                name=name,
                line=dict(width=1.8),
            )
        )
    fig.update_xaxes(title="saved MCMC sample index")
    fig.update_yaxes(title="theta value")
    return clean_layout(fig, "MCMC trace plot: sampled internal parameters", 430)


def posterior_parameter_plot(samples_low: np.ndarray, samples_high: Optional[np.ndarray] = None):
    """Posterior intervals on a linear y-axis.

    Off-diagonal diffusion terms can be negative. Therefore this plot must not
    use an ordinary log scale.
    """
    fig = go.Figure()

    low_df = D_metrics_from_theta_samples(samples_low)
    for k, (_, row) in enumerate(low_df.iterrows()):
        fig.add_trace(
            go.Scatter(
                x=[row["parameter"]],
                y=[row["50%"]],
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=[row["97.5%"] - row["50%"]],
                    arrayminus=[row["50%"] - row["2.5%"]],
                ),
                mode="markers",
                name="Laplace low-cost",
                showlegend=(k == 0),
                marker=dict(size=10),
            )
        )

    if samples_high is not None and len(samples_high) > 5:
        high_df = D_metrics_from_theta_samples(samples_high)
        for k, (_, row) in enumerate(high_df.iterrows()):
            fig.add_trace(
                go.Scatter(
                    x=[row["parameter"]],
                    y=[row["50%"]],
                    error_y=dict(
                        type="data",
                        symmetric=False,
                        array=[row["97.5%"] - row["50%"]],
                        arrayminus=[row["50%"] - row["2.5%"]],
                    ),
                    mode="markers",
                    name="MCMC high-cost",
                    showlegend=(k == 0),
                    marker=dict(size=10, symbol="diamond"),
                )
            )

    fig.add_hline(y=0.0, line_dash="dash", opacity=0.45)
    fig.update_yaxes(title="diffusion matrix parameter, linear scale")
    fig.update_xaxes(title="parameter")
    return clean_layout(fig, "Posterior intervals of inferred interaction coefficients", 470)



# =============================================================================
# Abstract / conference validation helpers
# =============================================================================

def flatten_D_region_rows(
    D_left_true: np.ndarray,
    D_right_true: np.ndarray,
    D_left_pred: np.ndarray,
    D_right_pred: np.ndarray,
    D_avg_true: np.ndarray,
    D_avg_pred: np.ndarray,
) -> pd.DataFrame:
    """Compare true and estimated left/right diffusion matrices."""
    rows = []
    labels = [["D_NiNi", "D_NiTa"], ["D_TaNi", "D_TaTa"]]
    for region, Dt, Dp in [
        ("left", D_left_true, D_left_pred),
        ("right", D_right_true, D_right_pred),
        ("average", D_avg_true, D_avg_pred),
    ]:
        for i in range(2):
            for j in range(2):
                true_val = float(Dt[i, j])
                pred_val = float(Dp[i, j])
                rows.append(
                    {
                        "region": region,
                        "parameter": labels[i][j],
                        "true": true_val,
                        "PINNs_estimate": pred_val,
                        "absolute_error": pred_val - true_val,
                        "relative_error_percent": np.nan if abs(true_val) < 1.0e-300 else 100.0 * (pred_val - true_val) / abs(true_val),
                        "sign_agreement": bool(np.sign(true_val) == np.sign(pred_val) or abs(true_val) < 1.0e-300 or abs(pred_val) < 1.0e-300),
                    }
                )
    return pd.DataFrame(rows)


def profile_rmse_against_exp(
    C_pred_final: np.ndarray,
    x_grid: np.ndarray,
    x_exp: np.ndarray,
    C_exp: np.ndarray,
    label: str,
    region_width: float = 0.03,
) -> pd.DataFrame:
    """RMSE/MAE summary for a final profile against experiment-like points."""
    regions = classify_region_from_x(x_exp.ravel(), width=region_width)
    rows = []
    for j, comp in enumerate(COMPONENTS):
        pred = np.interp(x_exp.ravel(), x_grid.ravel(), C_pred_final[:, j])
        exp = C_exp[:, j]
        res = pred - exp
        for region_name in ["left", "interface", "right", "all"]:
            mask = np.ones_like(res, dtype=bool) if region_name == "all" else (regions == region_name)
            if not np.any(mask):
                continue
            rr = res[mask]
            rows.append(
                {
                    "model": label,
                    "component": comp,
                    "region": region_name,
                    "n": int(np.sum(mask)),
                    "MAE": float(np.mean(np.abs(rr))),
                    "RMSE": float(np.sqrt(np.mean(rr**2))),
                    "bias": float(np.mean(rr)),
                    "max_abs": float(np.max(np.abs(rr))),
                }
            )
    return pd.DataFrame(rows)


def abstract_validation_metrics(
    x_grid: np.ndarray,
    C_pinn_final: np.ndarray,
    C_fdm_final: np.ndarray,
    x_exp: np.ndarray,
    C_exp: np.ndarray,
    D_true: np.ndarray,
    D_true_left: np.ndarray,
    D_true_right: np.ndarray,
    D_pinn: np.ndarray,
    D_pinn_left: np.ndarray,
    D_pinn_right: np.ndarray,
    C_zero_final: Optional[np.ndarray],
    region_width: float,
) -> Dict[str, pd.DataFrame]:
    """Build validation tables for abstract-level reporting."""
    D_compare = flatten_D_region_rows(
        D_true_left, D_true_right, D_pinn_left, D_pinn_right, D_true, D_pinn
    )

    rmse_tables = [
        profile_rmse_against_exp(C_pinn_final, x_grid, x_exp, C_exp, "PINNs", region_width=region_width),
        profile_rmse_against_exp(C_fdm_final, x_grid, x_exp, C_exp, "FDM_teacher", region_width=region_width),
    ]
    if C_zero_final is not None:
        rmse_tables.append(profile_rmse_against_exp(C_zero_final, x_grid, x_exp, C_exp, "zero_interaction", region_width=region_width))
    rmse_df = pd.concat(rmse_tables, ignore_index=True)

    all_rows = rmse_df[rmse_df["region"] == "all"].copy()
    pivot = all_rows.pivot_table(index="component", columns="model", values="RMSE", aggfunc="first").reset_index()
    if "PINNs" in pivot.columns and "zero_interaction" in pivot.columns:
        pivot["RMSE_improvement_vs_zero_percent"] = 100.0 * (pivot["zero_interaction"] - pivot["PINNs"]) / np.maximum(pivot["zero_interaction"], 1.0e-300)
    if "PINNs" in pivot.columns and "FDM_teacher" in pivot.columns:
        pivot["PINNs_over_FDM_teacher_RMSE_ratio"] = pivot["PINNs"] / np.maximum(pivot["FDM_teacher"], 1.0e-300)

    return {
        "D_compare": D_compare,
        "profile_rmse": rmse_df,
        "rmse_improvement": pivot,
    }


def abstract_validation_summary_text(metrics: Dict[str, pd.DataFrame]) -> str:
    """Generate a compact conference-abstract style summary."""
    ddf = metrics["D_compare"]
    rdf = metrics["rmse_improvement"]

    cross = ddf[ddf["parameter"].isin(["D_NiTa", "D_TaNi"]) & ddf["region"].isin(["left", "right"])]
    sign_ok = int(cross["sign_agreement"].sum()) if len(cross) else 0
    sign_total = int(len(cross))

    msg = []
    msg.append("Synthetic diffusion-couple validation was performed using known left/right interdiffusion matrices.")
    if sign_total:
        msg.append(f"The inferred cross terms recovered the correct sign in {sign_ok}/{sign_total} left/right cross-term entries.")
    if "RMSE_improvement_vs_zero_percent" in rdf.columns:
        vals = rdf["RMSE_improvement_vs_zero_percent"].replace([np.inf, -np.inf], np.nan).dropna()
        if len(vals):
            msg.append(f"Compared with the zero-interaction reference, the PINNs profile reduced the component-wise RMSE by a median of {float(np.nanmedian(vals)):.1f}%.")
    msg.append("These results support the use of physics-informed inverse analysis as a screening tool for CALPHAD/DICTRA mobility-description validation.")
    return " ".join(msg)


def multi_time_pseudo_exp_rmse_table(
    model,
    x_exp_all: np.ndarray,
    t_exp_all: np.ndarray,
    c_exp_all: np.ndarray,
) -> pd.DataFrame:
    """RMSE of PINNs prediction against multi-time pseudo-exp points."""
    pred = predict(model, x_exp_all.reshape(-1, 1), t_exp_all.reshape(-1, 1))
    rows = []
    unique_t = np.unique(t_exp_all.reshape(-1))
    for tv in unique_t:
        mask = np.isclose(t_exp_all.reshape(-1), tv)
        for j, comp in enumerate(COMPONENTS):
            res = pred[mask, j] - c_exp_all[mask, j]
            rows.append(
                {
                    "tau": float(tv),
                    "component": comp,
                    "n": int(np.sum(mask)),
                    "MAE": float(np.mean(np.abs(res))),
                    "RMSE": float(np.sqrt(np.mean(res**2))),
                    "bias": float(np.mean(res)),
                }
            )
    all_res = pred - c_exp_all
    for j, comp in enumerate(COMPONENTS):
        res = all_res[:, j]
        rows.append(
            {
                "tau": "all",
                "component": comp,
                "n": int(len(res)),
                "MAE": float(np.mean(np.abs(res))),
                "RMSE": float(np.sqrt(np.mean(res**2))),
                "bias": float(np.mean(res)),
            }
        )
    return pd.DataFrame(rows)


def multi_time_pseudo_exp_rmse_plot(mt_df: pd.DataFrame, annealing_time_h: float, tau_max: float):
    fig = go.Figure()
    plot_df = mt_df[mt_df["tau"] != "all"].copy()
    if len(plot_df) == 0:
        return clean_layout(fig, "Multi-time pseudo-exp RMSE", 420)
    plot_df["real_time_h"] = plot_df["tau"].astype(float) / max(float(tau_max), 1.0e-14) * float(annealing_time_h)
    for comp in COMPONENTS:
        sub = plot_df[plot_df["component"] == comp]
        fig.add_trace(
            go.Scatter(
                x=sub["real_time_h"],
                y=sub["RMSE"],
                mode="lines+markers",
                name=comp,
                line=dict(width=2.4, color=COLORS[comp]),
                marker=dict(symbol=SYMBOLS[comp], size=8),
            )
        )
    fig.update_xaxes(title="Real time equivalent (h)")
    fig.update_yaxes(title="RMSE vs multi-time pseudo-exp")
    return clean_layout(fig, "Multi-time pseudo-exp validation", 420)



def abstract_validation_plot(metrics: Dict[str, pd.DataFrame]):
    """Compact plot of RMSE improvement over zero-interaction."""
    fig = go.Figure()
    df = metrics["rmse_improvement"]
    if "RMSE_improvement_vs_zero_percent" in df.columns:
        fig.add_trace(
            go.Bar(
                x=df["component"],
                y=df["RMSE_improvement_vs_zero_percent"],
                name="RMSE improvement vs zero-interaction",
            )
        )
    fig.add_hline(y=0.0, line_dash="dash", opacity=0.5)
    fig.update_xaxes(title="Component")
    fig.update_yaxes(title="RMSE improvement [%]")
    return clean_layout(fig, "Abstract validation: improvement over zero-interaction", 420)



def posterior_band_fig11_plot(
    x: np.ndarray,
    x_exp: np.ndarray,
    C_exp: np.ndarray,
    C_pinn_final: np.ndarray,
    band: Dict[str, np.ndarray],
    span_um: float,
    title: str,
):
    dist = distance_um_from_x(x, span_um)
    dist_exp = distance_um_from_x(x_exp, span_um)

    fig = go.Figure()
    for j, comp in enumerate(COMPONENTS):
        fig.add_trace(
            go.Scatter(
                x=dist,
                y=band["q975"][:, j],
                mode="lines",
                line=dict(width=0, color=COLORS[comp]),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=dist,
                y=band["q025"][:, j],
                mode="lines",
                line=dict(width=0, color=COLORS[comp]),
                fill="tonexty",
                name=f"95% band {comp}",
                opacity=0.18,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=dist,
                y=C_pinn_final[:, j],
                mode="lines",
                name=f"PINN fit {comp}",
                line=dict(width=2.5, dash="dash", color=COLORS[comp]),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=dist_exp,
                y=C_exp[:, j],
                mode="markers",
                name=f"Exp. {comp}",
                marker=dict(symbol=SYMBOLS[comp], size=7, color=COLORS[comp], line=dict(width=1.2)),
            )
        )

    fig.update_xaxes(title="Distance (µm)")
    fig.update_yaxes(title="Mole Fraction", range=[-0.04, 1.04])
    return clean_layout(fig, title, 560, legend_y=-0.30)


def likelihood_contour_plot(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    Z: np.ndarray,
    x_label: str,
    y_label: str,
    theta_hat: np.ndarray,
    axis_i: int,
    axis_j: int,
):
    fig = go.Figure(
        data=[
            go.Contour(
                x=grid_x,
                y=grid_y,
                z=Z,
                contours=dict(showlabels=True),
                colorbar=dict(title="ΔNLL"),
                colorscale="Viridis",
            )
        ]
    )
    fig.add_trace(
        go.Scatter(
            x=[theta_hat[axis_i]],
            y=[theta_hat[axis_j]],
            mode="markers",
            name="PINN estimate",
            marker=dict(size=11, symbol="x"),
        )
    )
    fig.update_xaxes(title=x_label)
    fig.update_yaxes(title=y_label)
    return clean_layout(fig, "Likelihood / posterior contour around inferred parameters", 500)



def render_mcmc_explanation_expander(location: str = "main"):
    """Render an explanatory MCMC block in the Streamlit GUI."""
    with st.expander("MCMC信頼度評価の計算内容を表示", expanded=False):
        st.markdown(
            """
### MCMCで何をしているか

このアプリのMCMCは、PINNsを毎回再学習する処理ではありません。  
PINNsで得た推定値を出発点として、候補となる拡散パラメータを少しずつ動かし、そのたびにFDMを解いて実験点との尤度を評価します。

| 項目 | 内容 |
|---|---|
| 出発点 | PINNs推定値 `theta_hat` |
| サンプリング対象 | `theta = [log_d11, log_d22, rho_raw]` |
| forward model | FDM、`_run_fdm_teacher_core()` |
| 比較対象 | `x_exp`, `c_exp` |
| 尤度 | Gaussian likelihood |
| 事前分布 | `prior_mean`, `prior_std` |
| サンプラー | Random-walk Metropolis |
| 出力 | posterior samples, acceptance rate, credible band |

#### 1ステップで行う計算

| 順序 | 処理 |
|---:|---|
| 1 | 現在の `theta` から候補 `theta_proposal` を作る |
| 2 | 候補 `theta_proposal` を拡散行列 `D` に変換する |
| 3 | その `D` でFDMを解き、最終プロファイルを計算する |
| 4 | 実験点位置に補間する |
| 5 | Ni/Ta成分の残差からGaussian NLLを計算する |
| 6 | 事前分布ペナルティを加えて負の対数事後分布を作る |
| 7 | Metropolis基準で候補を採択または棄却する |
| 8 | burn-in後のサンプルを保存する |

#### 内部パラメータと拡散行列

```text
theta = [log_d11, log_d22, rho_raw]

D_NiNi = exp(log_d11)
D_TaTa = exp(log_d22)
rho    = 0.95 tanh(rho_raw)
D_NiTa = D_TaNi = rho sqrt(D_NiNi D_TaTa)
```

#### 採択率の見方

| acceptance rate | 解釈 | 調整 |
|---:|---|---|
| `< 0.1` | 候補が遠すぎて棄却されやすい | `proposal std` を下げる |
| `0.2〜0.5` | まずは妥当な範囲 | そのまま様子を見る |
| `> 0.6` | 候補が近すぎて探索が遅い | `proposal std` を上げる |

#### 計算コスト

MCMCは各ステップでFDMを解くため、計算時間は概ね次のように増えます。

```text
計算時間 ≈ MCMC steps × FDM 1回分の計算時間
```

そのため、まずは `MCMC steps = 700〜1000` 程度で挙動を確認し、必要に応じて `2000〜3000` に増やすのが安全です。

#### 注意点

現在の実装は、3成分すべてに同じ `proposal_std` を使う等方的なrandom-walk Metropolisです。  
研究用途では、パラメータごとに異なるproposal幅、複数チェーン、trace plot、自己相関、R-hatなどの収束診断を追加すると、より信頼性の高いMCMC解析になります。
"""
        )


def render_mcmc_quick_hint(run_high_cost_mcmc: bool, mcmc_steps: int, mcmc_burn: int, mcmc_proposal: float):
    """Small always-visible MCMC status hint."""
    if run_high_cost_mcmc:
        effective = max(int(mcmc_steps) - int(mcmc_burn), 0)
        st.info(
            f"MCMC is ON: {int(mcmc_steps):,} steps, burn-in {int(mcmc_burn):,}, "
            f"saved samples {effective:,}, proposal std {float(mcmc_proposal):.3f}. "
            "Each step solves an FDM forward problem."
        )
    else:
        st.caption(
            "High-cost MCMC is OFF. Turn it on only after the PINNs/Laplace result looks reasonable, "
            "because MCMC repeatedly solves the FDM forward model."
        )



# =============================================================================
# Main UI
# =============================================================================

st.markdown(
    """
<div class="hero">
  <div class="hero-title">Fig.11-style Co / Ni-0.10Ta Diffusion Couple</div>
  <div class="hero-sub">
    Co / Ni-0.10Ta 三元拡散対の濃度プロファイルを FDM・PINN・FDMによる疑似実験データで比較します。
    PINN は Ni/Ta の有効相互拡散行列を推定し、尤度に基づく信頼度をバンドとして可視化します。
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="note">
<b>Initial diffusion couple</b>: left = pure Co, right = Ni-0.10Ta.<br>
<b>Independent components</b>: Ni and Ta. Co is computed as 1 - Ni - Ta.<br>
<b>Model</b>: u<sub>t</sub> = ∂<sub>x</sub>(D ∂<sub>x</sub>u), where u=[x<sub>Ni</sub>, x<sub>Ta</sub>].<br>
<b>Reliability</b>: low-cost Laplace approximation and high-cost FDM-based MCMC.
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## Controls")
    st.caption("LR schedule: CosineAnnealingLR, eta_min = 0.03 × initial LR")

    # --- Device selection with benchmark ---
    st.markdown("### Compute device")
    _bench_key = "device_benchmark_result"
    if _bench_key not in st.session_state:
        st.session_state[_bench_key] = None

    if _CUDA_AVAILABLE:
        if st.button("Run CPU / GPU benchmark"):
            with st.spinner("Benchmarking CPU vs GPU (30 epochs)…"):
                st.session_state[_bench_key] = run_device_benchmark()

        _bench = st.session_state[_bench_key]
        if _bench is not None:
            st.caption(
                f"CPU: {_bench['cpu_ms']:.1f} ms/ep — "
                f"GPU ({_bench['gpu_name']}): {_bench['gpu_ms']:.1f} ms/ep — "
                f"Speedup: {_bench['speedup']:.1f}×"
            )
            _default_idx = 0 if _bench["recommended"] == "cpu" else 1
        else:
            _default_idx = 1  # default to GPU when available

        _device_choice = st.radio(
            "Training device",
            ["cpu", f"cuda ({_GPU_NAME})"],
            index=_default_idx,
            help="ベンチマーク実行後は速い方が自動選択されます。手動で切り替え可能。",
        )
        _dev_str = "cuda" if "cuda" in _device_choice else "cpu"
        _set_device(_dev_str)
        if _bench is not None and _dev_str == _bench["recommended"]:
            st.success(f"Device: **{DEVICE}** (benchmark推奨)")
        else:
            st.info(f"Device: **{DEVICE}**")
    else:
        st.warning("CUDA is not available. Training runs on CPU.")
        st.caption(f"Device: {DEVICE}")
        # Run CPU-only benchmark for reference
        if st.button("Run CPU benchmark"):
            with st.spinner("Benchmarking CPU (30 epochs)…"):
                _bench_result = run_device_benchmark()
                st.session_state[_bench_key] = _bench_result
        _bench = st.session_state[_bench_key]
        if _bench is not None:
            st.caption(f"CPU: {_bench['cpu_ms']:.1f} ms/epoch")

    st.markdown("### Analysis mode")
    pinn_analysis_mode = st.selectbox(
        "PINN analysis mode",
        ["Fickian D matrix", "Regular-solution chemical potential"],
        index=1,
        help=(
            "Fickian D matrix: traditional approach, flux = -D grad(c). "
            "Regular-solution chemical potential: flux = M grad(mu), where mu "
            "is derived from a regular-solution free-energy model with trainable "
            "Omega pair-interaction terms."
        ),
    )
    use_chemical_potential = (pinn_analysis_mode == "Regular-solution chemical potential")
    if use_chemical_potential:
        st.info(
            "化学ポテンシャルモード: Regular-solution モデルで Omega 相互作用項を推定します。\n\n"
            "PDE: c_t = div(M grad(μ)),  μ_i - μ_ref = RT ln(c_i/c_ref) + Σ_b (Ω_ib - Ω_rb) c_b"
        )
        with st.expander("📖 理論説明 (Regular-solution chemical potential model)", expanded=False):
            st.markdown(r"""
#### 1. 化学ポテンシャルと拡散

Fickian 表記 $\mathbf{J} = -D\,\nabla c$ の代わりに、化学ポテンシャル駆動の拡散方程式を使います:

$$
\frac{\partial c_i}{\partial t} = \nabla \cdot \left( M_{ij}\, \nabla \mu_j \right)
$$

ここで $M_{ij}$ は **mobility 行列** (Onsager 係数 $L_{ij}$ に比例) です。
Fickian 拡散行列 $D$ は $D_{ij} = \sum_k M_{ik} \frac{\partial \mu_k}{\partial c_j}$ を通じて化学ポテンシャルから導出されます。

#### 2. Regular-solution モデル

三元系 $(i = 1, 2, 3)$ のモル Gibbs エネルギーを以下で近似します:

$$
G_\mathrm{mix} = RT \sum_i c_i \ln c_i + \sum_{i<j} \Omega_{ij}\, c_i\, c_j
$$

- 第1項: **ideal-solution** (理想混合エントロピー)
- 第2項: **excess** (ペア相互作用 $\Omega_{ij}$ = Redlich–Kister $L^0_{ij}$)

成分 $i$ の化学ポテンシャル (参照成分 $r$ からの差):

$$
\mu_i - \mu_r = RT \ln\frac{c_i}{c_r} + \sum_{b} (\Omega_{ib} - \Omega_{rb})\, c_b
$$

(P6 note: $\Omega_{aa} = 0$ の慣行を使用。本実装では Co を参照成分 ($r$ = Co) としています。)

#### 3. PINN による Omega 推定

PINNs は以下の損失関数を最小化し、$\Omega_{ij}$ を含むネットワークパラメータを同時推定します:

$$
\mathcal{L} = \underbrace{\lambda_d \sum \|c^\mathrm{pred} - c^\mathrm{obs}\|^2}_{\text{data}} + \underbrace{\lambda_p \sum \|c_t - \nabla\cdot(M\nabla\mu)\|^2}_{\text{physics (PDE)}} + \underbrace{\lambda_\mathrm{ic} \sum \|c(x,0) - c_0(x)\|^2}_{\text{initial condition}}
$$

#### 4. FDM 尤度再最適化

PINN 推定値 $\hat{\Omega}$ を初期値として、FDM (有限差分法) シミュレータを用いた厳密な尤度最大化を行います:

$$
\hat{\Omega}^\mathrm{refined} = \arg\min_\Omega \sum_{t_k} \frac{1}{2\sigma^2}\|c^\mathrm{FDM}(\Omega, t_k) - c^\mathrm{exp}(t_k)\|^2
$$

Powell 法で最適化し、Hessian から Laplace 近似の事後分布 $\mathcal{N}(\hat{\Omega}, H^{-1})$ を得ます。

#### 5. DICTRA との対応

| 本実装 | DICTRA/Thermo-Calc |
|---|---|
| $\Omega_{ij}$ | $L^0_{ij}$ (Redlich–Kister 0次項) |
| Mobility 行列 $M$ | Onsager 係数 $L_{ij}$ |
| $RT$ | $RT$ (物理温度) |
| FDM solver | DICTRA homogenization |
""")


    st.markdown("### True diffusion matrix for FDM")
    st.caption("Values are normalized, not physical SI units.")
    fdm_teacher_mode = st.selectbox(
        "FDM teacher diffusion model",
        ["single D", "left/right D"],
        index=1,
        help="Use left/right D for FCC/BCC-like examples with strongly different diffusivities.",
    )
    fcc_bcc_preset = st.selectbox(
        "example true-D preset",
        ["manual", "TOFA abstract validation", "FCC-left slow / BCC-right fast"],
        index=1,
        help="Preset normalized values. TOFA abstract validation is designed to be stable and visibly non-trivial.",
    )
    if fcc_bcc_preset == "TOFA abstract validation":
        # Moderate left/right contrast: large enough to show asymmetric diffusion,
        # but not so extreme that PINNs/MCMC becomes unstable for a quick abstract-level demo.
        default_l11, default_l22, default_rho_l = -5.20, -7.00, -0.20
        default_r11, default_r22, default_rho_r = -3.20, -5.00, -0.35
        st.success(
            "TOFA abstract validation preset: left/right D contrast is about 7–8×. "
            "This is intended for reproducible synthetic validation and abstract figures."
        )
    elif fcc_bcc_preset == "FCC-left slow / BCC-right fast":
        default_l11, default_l22, default_rho_l = -7.00, -10.00, -0.15
        default_r11, default_r22, default_rho_r = -2.40, -5.40, -0.35
        st.info(
            "FCC/BCC contrast preset: left side is slow, right side is fast. "
            "Approximate contrast: D_NiNi right/left ≈ 100 and D_TaTa right/left ≈ 100. "
            "PINNs initial values are aligned with these defaults unless you change them."
        )
    else:
        default_l11, default_l22, default_rho_l = -3.15, -4.00, -0.35
        default_r11, default_r22, default_rho_r = -3.15, -4.00, -0.35

    ui_force_symmetric = st.checkbox(
        "Force symmetric D (D₁₂ = D₂₁)",
        value=FORCE_SYMMETRIC_D,
        help="Onsager symmetry holds for L (mobility), not D = LΦ. "
             "Real interdiffusion matrices are generally non-symmetric. "
             "Uncheck to use 4-parameter non-symmetric D.",
    )

    log_d11_true = st.slider("left/single log D_NiNi true", -12.0, -1.0, float(default_l11), 0.05)
    log_d22_true = st.slider("left/single log D_TaTa true", -12.0, -1.0, float(default_l22), 0.05)
    rho_raw_true = st.slider("left/single coupling rho₁₂ (D₁₂) true", -2.5, 2.5, float(default_rho_l), 0.05)
    if not ui_force_symmetric:
        rho21_raw_true = st.slider("left/single coupling rho₂₁ (D₂₁) true", -2.5, 2.5, float(default_rho_l), 0.05)
    else:
        rho21_raw_true = None

    log_d11_true_right = float(log_d11_true)
    log_d22_true_right = float(log_d22_true)
    rho_raw_true_right = float(rho_raw_true)
    rho21_raw_true_right = rho21_raw_true
    if fdm_teacher_mode == "left/right D":
        st.markdown("#### Right-side true D for FDM")
        log_d11_true_right = st.slider("right log D_NiNi true", -12.0, -1.0, float(default_r11), 0.05)
        log_d22_true_right = st.slider("right log D_TaTa true", -12.0, -1.0, float(default_r22), 0.05)
        rho_raw_true_right = st.slider("right coupling rho₁₂ (D₁₂) true", -2.5, 2.5, float(default_rho_r), 0.05)
        if not ui_force_symmetric:
            rho21_raw_true_right = st.slider("right coupling rho₂₁ (D₂₁) true", -2.5, 2.5, float(default_rho_r), 0.05)
        else:
            rho21_raw_true_right = None

    st.markdown("### FDM teacher / experiment-like points")
    t_max = st.slider("annealing time, normalized", 0.02, 1.00, 0.22, 0.01)
    nx_fdm = st.select_slider("FDM spatial grid", options=[121, 161, 201, 251, 301], value=201)
    nt_fdm = st.select_slider("FDM saved frames", options=[40, 60, 80, 100, 140], value=80)
    span_um = st.slider("plot distance span, µm", 400.0, 1200.0, 800.0, 50.0)
    multi_time_count = st.slider("multi-time profile slices", 3, 9, 5, 1)
    multi_time_columns = st.selectbox("multi-time plot columns", [1, 2, 3], index=1)
    show_zero_interaction_reference = st.checkbox(
        "show zero-interaction reference",
        value=True,
        help="Force D_NiTa = D_TaNi = 0 while keeping selected diagonal terms.",
    )
    zero_interaction_source = st.selectbox(
        "zero-interaction diagonal source",
        ["PINN estimated diagonals", "FDM true diagonals"],
        index=0,
        help="Choose the diagonal terms used for the zero-interaction reference.",
    )
    noise = st.slider("pseudo experimental noise", 0.000, 0.050, 0.008, 0.001)
    # P8 fix: allow user to select noise model
    noise_model = st.selectbox(
        "noise model",
        ["gaussian (simplex-projected)", "ALR (additive log-ratio)"],
        index=0,
        help=(
            "Gaussian: adds iid noise then re-normalizes to simplex (fast, introduces "
            "correlation). ALR: adds noise in log-ratio space (self-consistent with "
            "Gaussian NLL assumption)."
        ),
    )
    noise_model_key = "alr" if "ALR" in noise_model else "gaussian"
    n_exp_points = st.slider("experimental-like points per time", 30, 120, 64, 2)
    pseudo_exp_time_mode = st.selectbox(
        "pseudo-exp time mode",
        ["final only", "multi-time"],
        index=1 if str(fcc_bcc_preset) == "TOFA abstract validation" else 0,
        help="Use multi-time pseudo experimental data to improve identifiability of cross-interdiffusion terms.",
    )
    pseudo_exp_time_slices = st.slider(
        "pseudo-exp time slices",
        1,
        8,
        4 if str(fcc_bcc_preset) == "TOFA abstract validation" else 1,
        1,
        help="Number of FDM time slices used to generate experiment-like data. Final time is always included.",
    )
    append_pseudo_exp_to_training = st.checkbox(
        "append pseudo-exp points to PINNs observations",
        value=True,
        help="If ON, pseudo experimental points are also used as supervised observation points during PINNs training.",
    )
    t_start_fraction = st.slider("PINN start time / total time", 0.005, 0.150, 0.030, 0.005)
    seed = st.number_input("random seed", min_value=1, max_value=999999, value=7, step=1)

    st.markdown("### Training points")
    n_obs = st.slider("sparse observations", 50, 2000, 500, 10)
    n_ic = st.slider("initial profile points at t_start", 30, 800, 240, 10)
    n_bc_each = st.slider("boundary points per side", 20, 500, 120, 10)
    n_f = st.slider("collocation points", 300, 12000, 3600, 100)

    st.markdown("### PINNs")
    st.caption("By default, PINNs initial D values are aligned with the FDM teacher true-D preset.")
    log_d11_init = st.slider("left/single initial log D_NiNi", -12.0, -1.0, float(default_l11), 0.05)
    log_d22_init = st.slider("left/single initial log D_TaTa", -12.0, -1.0, float(default_l22), 0.05)
    rho_raw_init = st.slider("left/single initial coupling raw rho", -2.5, 2.5, float(default_rho_l), 0.05)
    width = st.select_slider("network width", options=[24, 32, 48, 64, 96, 128], value=64)
    depth = st.select_slider("network depth", options=[2, 3, 4, 5, 6], value=4)
    activation = st.selectbox("activation", ["tanh", "silu", "gelu"], index=0)
    epochs = st.slider("epochs", 300, 15000, 6000, 100)
    lr = st.select_slider(
        "learning rate", options=[1e-4, 3e-4, 1e-3, 3e-3], value=3e-4,
        format_func=lambda v: f"{v:.0e}"
    )

    st.markdown("### Diffusion-couple / PINNs stability")
    diffusion_model_mode = st.selectbox(
        "PINNs diffusion model",
        ["single D", "left/right D"],
        index=1,
        help="Use one diffusion matrix over the domain, or separate left/right matrices smoothly blended near the interface.",
    )
    phase_interface_width = st.slider("phase interface width", 0.001, 0.080, 0.020, 0.001)

    log_d11_right_init = float(log_d11_init)
    log_d22_right_init = float(log_d22_init)
    rho_raw_right_init = float(rho_raw_init)
    if diffusion_model_mode == "left/right D":
        st.markdown("#### Right-side PINNs initial D")
        log_d11_right_init = st.slider("right initial log D_NiNi", -12.0, -1.0, float(default_r11), 0.05)
        log_d22_right_init = st.slider("right initial log D_TaTa", -12.0, -1.0, float(default_r22), 0.05)
        rho_raw_right_init = st.slider("right initial coupling raw rho", -2.5, 2.5, float(default_rho_r), 0.05)
    with st.expander("拡散対PINNsの注意", expanded=False):
        st.markdown(
            """
拡散対の初期条件は階段状で、数学的には不連続です。現在のデータ生成では `PINN start time / total time` で `t=0` を避け、FDMで少し拡散した後の滑らかなプロファイルを初期条件としてPINNsに与えています。

| 設定 | 推奨 |
|---|---|
| `PINN start time / total time` | `0.01`〜`0.05` 程度 |
| `w_ic` | 小さめ、例: `2`〜`5` |
| `diffusion model` | 左右で結晶構造が異なるなら `left/right D` |
| `phase interface width` | `0.01`〜`0.03` 程度 |

`left/right D` では、固定界面近傍で以下のように拡散行列を滑らかに接続します。

```text
D(x) = (1 - s(x)) D_left + s(x) D_right
s(x) = 0.5 * [1 + tanh((x - x_interface) / width)]
```

これは厳密なDICTRA型の移動境界・局所平衡モデルではありませんが、左右で結晶構造や拡散速度が異なる効果をPINNs/FDMの枠内で扱うための実用的な近似です。
"""
        )

    st.markdown("### Paper / CALPHAD basis")
    st.caption("論文・DICTRAと同じ基準で比較するための物理スケールと自己拡散係数です。")
    paper_T_C = st.number_input("paper temperature, °C", value=1200.0, step=10.0)
    paper_time_h = st.number_input("paper annealing time, h", value=160.0, step=10.0)
    paper_length_um = st.number_input(
        "physical length scale L, µm",
        value=float(span_um),
        min_value=1.0,
        step=50.0,
        help="Normalized x in [0,1] is mapped to a physical length scale L. D_phys = D_norm * L^2 / t.",
    )

    # TOFA abstract preset:
    # Use known diagonal entries as fixed anchors so the inverse problem focuses on
    # cross-interdiffusion terms relative to a Darken/zero-interaction reference.
    is_tofa_abstract_preset = (str(fcc_bcc_preset) == "TOFA abstract validation")
    if is_tofa_abstract_preset:
        st.success(
            "TOFA abstract validation default: diagonal terms are fixed from the FDM-teacher true diagonal values. "
            "The main inferred quantities are the cross-interdiffusion terms."
        )

    use_self_diffusion = st.checkbox(
        "use available self-diffusion coefficients in PINNs",
        value=bool(is_tofa_abstract_preset),
        help="Use literature/CALPHAD self or tracer diffusion values as diagonal anchors for D_NiNi and D_TaTa.",
    )

    # Default values. For TOFA abstract validation, use the FDM-teacher true
    # diagonal values converted from normalized units to m^2/s.
    self_D_Ni_phys = 0.0
    self_D_Ta_phys = 0.0
    self_D_Ni_left_phys = 0.0
    self_D_Ta_left_phys = 0.0
    self_D_Ni_right_phys = 0.0
    self_D_Ta_right_phys = 0.0

    if use_self_diffusion:
        if diffusion_model_mode == "single D":
            default_single_D_Ni_phys = normalized_scalar_to_physical_D(np.exp(float(log_d11_true)), paper_length_um, paper_time_h) if is_tofa_abstract_preset else 0.0
            default_single_D_Ta_phys = normalized_scalar_to_physical_D(np.exp(float(log_d22_true)), paper_length_um, paper_time_h) if is_tofa_abstract_preset else 0.0

            st.markdown("#### Single-region self-diffusion anchors")
            st.caption("全領域を1つの有効拡散行列で近似する場合に使います。")
            self_D_Ni_phys = st.number_input(
                "single-region D_NiNi prior [m²/s]",
                value=float(default_single_D_Ni_phys),
                format="%.4e",
                help="single Dモデル用のNi対角項アンカーです。",
            )
            self_D_Ta_phys = st.number_input(
                "single-region D_TaTa prior [m²/s]",
                value=float(default_single_D_Ta_phys),
                format="%.4e",
                help="single Dモデル用のTa対角項アンカーです。",
            )
        else:
            default_left_D_Ni_phys = normalized_scalar_to_physical_D(np.exp(float(log_d11_true)), paper_length_um, paper_time_h) if is_tofa_abstract_preset else 0.0
            default_left_D_Ta_phys = normalized_scalar_to_physical_D(np.exp(float(log_d22_true)), paper_length_um, paper_time_h) if is_tofa_abstract_preset else 0.0
            default_right_D_Ni_phys = normalized_scalar_to_physical_D(np.exp(float(log_d11_true_right)), paper_length_um, paper_time_h) if is_tofa_abstract_preset else 0.0
            default_right_D_Ta_phys = normalized_scalar_to_physical_D(np.exp(float(log_d22_true_right)), paper_length_um, paper_time_h) if is_tofa_abstract_preset else 0.0

            st.markdown("#### Left/right self-diffusion anchors")
            st.caption("左右で結晶構造・相・CALPHAD/DICTRAデータが異なる場合に使います。")
            if is_tofa_abstract_preset:
                st.caption("TOFA概要用プリセットでは、これら4つの対角項はFDM教師のtrue diagonalから自動設定されています。")
            lr_col1, lr_col2 = st.columns(2)
            with lr_col1:
                st.markdown("**Left region**")
                self_D_Ni_left_phys = st.number_input(
                    "left-region D_NiNi prior [m²/s]",
                    value=float(default_left_D_Ni_phys),
                    format="%.4e",
                    help="左側相/構造のNi対角拡散アンカーです。",
                )
                self_D_Ta_left_phys = st.number_input(
                    "left-region D_TaTa prior [m²/s]",
                    value=float(default_left_D_Ta_phys),
                    format="%.4e",
                    help="左側相/構造のTa対角拡散アンカーです。",
                )
            with lr_col2:
                st.markdown("**Right region**")
                self_D_Ni_right_phys = st.number_input(
                    "right-region D_NiNi prior [m²/s]",
                    value=float(default_right_D_Ni_phys),
                    format="%.4e",
                    help="右側相/構造のNi対角拡散アンカーです。",
                )
                self_D_Ta_right_phys = st.number_input(
                    "right-region D_TaTa prior [m²/s]",
                    value=float(default_right_D_Ta_phys),
                    format="%.4e",
                    help="右側相/構造のTa対角拡散アンカーです。",
                )

    diag_constraint_default_index = 3 if is_tofa_abstract_preset else 0
    diag_constraint_mode = st.selectbox(
        "self-diffusion use mode",
        ["free", "initialize only", "weak diagonal prior", "fix diagonal terms"],
        index=diag_constraint_default_index,
        help="Use available self-diffusion values as initialization, weak prior, or fixed diagonal terms.",
    )
    diag_prior_weight = st.slider("self-diffusion prior weight", 0.0, 200.0, 10.0, 1.0)

    if is_tofa_abstract_preset and diag_constraint_mode != "fix diagonal terms":
        st.warning(
            "TOFA abstract validation is designed around fixed diagonal terms. "
            "Changing this mode is useful for sensitivity analysis, but the default abstract claim assumes fixed diagonals."
        )

    with st.expander("Paper basis note", expanded=False):
        st.markdown(
            """
| 基準 | このアプリでの扱い |
|---|---|
| phase | FCC γ相を想定 |
| temperature | 既定値 1200 °C |
| annealing time | 既定値 160 h |
| composition unit | mole fraction |
| dependent component | Co |
| independent components | Ni, Ta |
| diffusion frame | CALPHAD/DICTRA比較では volume-fixed interdiffusion coefficient を推奨 |

自己拡散係数を使う場合は、物理単位 `[m²/s]` から内部の無次元Dへ変換します。

```text
D_norm = D_phys * t_scale / L_scale^2
```

`diffusion model = single D` の場合は、全領域で共有する対角項 `D_NiNi`, `D_TaTa` のアンカーとして使います。  
`diffusion model = left/right D` の場合は、左領域・右領域それぞれの対角項アンカーとして使います。  
`TOFA abstract validation` プリセットでは、これらの対角項を固定し、主に交差項 `D_NiTa`, `D_TaNi` をPINNsで推定する設計です。
"""
        )

    with st.expander("Loss weights", expanded=False):
        w_data = st.slider("w_data", 0.1, 100.0, 25.0, 0.1)
        w_ic = st.slider("w_ic", 0.1, 100.0, 12.0, 0.1)
        w_bc = st.slider("w_bc", 0.1, 100.0, 12.0, 0.1)
        w_phys = st.slider("w_physics", 0.01, 100.0, 0.1, 0.01)
        ui_adaptive_weights = st.checkbox(
            "Self-adaptive loss weighting (RBA)",
            value=False,
            help=(
                "Gradient-norm rebalancing: w_i(epoch) = w_i^base × mean(‖∇L_j‖) / ‖∇L_i‖. "
                "Prevents any single loss component from dominating training "
                "(Wang+ 2022, McClenny+ 2023)."
            ),
        )
        ui_direct_output = st.checkbox(
            "Direct [Ni, Ta] output (no simplex normalize)",
            value=False,
            help=(
                "Output Ni, Ta directly via sigmoid; Co = 1 − Ni − Ta. "
                "Avoids the normalization Jacobian that leaks into PDE residuals "
                "when using softplus + normalize."
            ),
        )
        ui_torch_compile = st.checkbox(
            "torch.compile (PyTorch 2 JIT)",
            value=False,
            help=(
                "PyTorch 2 の torch.compile でモデルの forward/backward を "
                "JIT コンパイルします。初回は数秒のコンパイル時間がかかりますが、"
                "以降の epoch が高速化されます。epoch 数が多い場合に有効。"
            ),
        )

    # --- Chemical potential (Omega) specific controls ---
    if use_chemical_potential:
        st.markdown("### Regular-solution Omega settings")

        rs_system_preset = st.selectbox(
            "System preset",
            ["Co-Ni-Ta (default)", "Fe-C-Si (Darken uphill diffusion)"],
            index=0,
            help=(
                "Co-Ni-Ta: default ternary system from fig11. "
                "Fe-C-Si: DICTRA Darken experiment (1050°C, 13 days). "
                "Uphill diffusion of C due to Si activity gradient."
            ),
        )

        if rs_system_preset == "Fe-C-Si (Darken uphill diffusion)":
            st.info(
                "Fe-C-Si Darken実験 (1050°C = 1323 K, 13 days)\n\n"
                "左: 3.80 wt% Si, 0.49 wt% C (残 Fe)\n"
                "右: 0.05 wt% Si, 0.45 wt% C (残 Fe)\n\n"
                "Cがuphill拡散 (Si濃度勾配による熱力学的カップリング)"
            )
            _omega_default = "3.0,1.0,0.5"
            _omega_right_default = "3.0,1.0,0.5"
            _rt_default = 1.0
            _mob_mode_default = 1
            _logM_defaults = (-3.5, -5.0, -3.5, -6.0, -6.0, -6.0)
        else:
            _omega_default = "1.5,0.5,2.0"
            _omega_right_default = "1.5,0.5,2.0"
            _rt_default = 1.0
            _mob_mode_default = 0
            _logM_defaults = (-4.0, -4.0, -4.0, -6.0, -6.0, -6.0)

        st.caption("Ωの順序: (comp1,comp2), (comp1,ref), (comp2,ref). RS mode時のFDM教師データはμ-Mモデルで生成（モデル整合性を保証）。")
        omega_left_init_str = st.text_input("初期Ω left (init)", value=_omega_default)
        learn_lr_omega = st.checkbox("左右別々のΩを学習", value=True)
        omega_right_init_str = st.text_input("初期Ω right (init)", value=_omega_right_default, disabled=not learn_lr_omega)
        # C11 fix: separate omega spatial blend width from initial composition profile width.
        # phase_interface_width controls the initial c(x) sigmoid; omega_blend_width controls
        # the spatial blending of left/right Omega values in the PINN and FDM NLL.
        rs_omega_blend_width = st.slider(
            "Ω spatial blend width",
            0.001, 0.150, 0.020, 0.001,
            help=(
                "Width of the tanh blending region for left/right Omega. "
                "Distinct from 'phase interface width' which controls the initial composition profile."
            ),
        )
        rs_RT = st.number_input("RT (regular-solution)", min_value=0.01, max_value=10.0, value=1.0, step=0.1)

        rs_mobility_mode = st.selectbox(
            "Mobility model",
            ["constant (scalar matrix)", "composition-dependent (CALPHAD end-member)"],
            index=_mob_mode_default,
            help=(
                "Constant: single diagonal M matrix. "
                "Composition-dependent: ln M_i(c) = Σ_j x_j ln(M_i^j) (CALPHAD end-member mixing)."
            ),
        )
        rs_use_comp_dep_mobility = (rs_mobility_mode == "composition-dependent (CALPHAD end-member)")

        if not rs_use_comp_dep_mobility:
            rs_M_diag = st.number_input("mobility diagonal M", min_value=1.0e-5, max_value=1.0, value=2.0e-2, step=1.0e-3, format="%.5f")
            rs_M_offdiag = st.number_input("mobility off-diagonal", min_value=-0.5, max_value=0.5, value=0.0, step=1.0e-3, format="%.5f")
        else:
            st.caption("ln(M_i^j): log-mobility of independent component i in end-member j")
            st.caption("行: 独立成分 (comp1, comp2), 列: end-member (comp1, comp2, ref)")
            rs_logM_00 = st.number_input("ln M_comp1^comp1", value=_logM_defaults[0], step=0.5, format="%.2f")
            rs_logM_01 = st.number_input("ln M_comp1^comp2", value=_logM_defaults[1], step=0.5, format="%.2f")
            rs_logM_02 = st.number_input("ln M_comp1^ref", value=_logM_defaults[2], step=0.5, format="%.2f")
            rs_logM_10 = st.number_input("ln M_comp2^comp1", value=_logM_defaults[3], step=0.5, format="%.2f")
            rs_logM_11 = st.number_input("ln M_comp2^comp2", value=_logM_defaults[4], step=0.5, format="%.2f")
            rs_logM_12 = st.number_input("ln M_comp2^ref", value=_logM_defaults[5], step=0.5, format="%.2f")
            rs_train_mobility = st.checkbox("Train mobility end-members", value=False)
        rs_fdm_dt = st.number_input("FDM dt for Omega teacher", min_value=1.0e-7, max_value=1.0e-3, value=1.0e-5, step=1.0e-6, format="%.2e")
        rs_fdm_nsteps = st.number_input("FDM steps for Omega teacher", min_value=10, max_value=100000, value=4000, step=500)
        rs_fdm_save_every = st.number_input("FDM save_every", min_value=1, max_value=10000, value=100, step=10)
        rs_n_collocation = st.number_input("collocation points/epoch", min_value=8, max_value=50000, value=2000, step=500)
        rs_w_omega_prior = st.slider("Omega prior regularization weight", 0.0, 50.0, 0.0, 0.5)
        do_fdm_refine = st.checkbox("PINN後にFDM順問題による尤度でΩを再最適化", value=True,
                                     help="候補Ωで RS FDM を順計算し、疑似実験点との残差（負の対数尤度）を最小化してΩを精密化します。")
        fdm_refine_maxiter = st.number_input("FDM再最適化 maxiter", min_value=10, max_value=2000, value=180, step=10)
        rs_like_sigma = st.slider("likelihood sigma (RS)", 0.001, 0.080, 0.008, 0.001)
        rs_hessian_step = st.slider("Laplace Hessian step (RS)", 0.005, 0.150, 0.030, 0.005)
        rs_laplace_samples = st.slider("Laplace posterior samples (RS)", 200, 5000, 800, 100)
        rs_band_samples = st.slider("FDM samples for credible band (RS)", 10, 150, 50, 10)
        rs_run_mcmc = st.checkbox("Run MCMC for Omega reliability", value=False)
        rs_mcmc_steps = st.slider("MCMC steps (RS)", 100, 3000, 600, 100)
        rs_mcmc_burn = st.slider("MCMC burn-in (RS)", 0, 1500, 150, 50)
        rs_mcmc_proposal = st.slider("MCMC proposal std (RS)", 0.005, 0.300, 0.035, 0.005)
        rs_skip_reliability = st.checkbox("Skip Omega reliability evaluation", value=False)

    st.markdown("### Reliability")
    st.caption("Low-cost = Laplace approximation. High-cost = FDM-based MCMC.")
    like_sigma = st.slider("likelihood sigma", 0.001, 0.080, max(float(noise), 0.008), 0.001)
    rel_nx = st.select_slider("reliability FDM grid", options=[61, 81, 101, 121, 151], value=81)
    rel_nt = st.select_slider("reliability saved frames", options=[20, 25, 35, 45], value=25)

    st.caption("Independent prior center, not automatically centered on PINN estimate.")
    prior_log_d11 = st.slider("prior mean log D_NiNi", -7.0, -1.0, -3.3, 0.05)
    prior_log_d22 = st.slider("prior mean log D_TaTa", -8.0, -1.0, -4.1, 0.05)
    prior_rho_raw = st.slider("prior mean rho₁₂_raw", -2.5, 2.5, 0.0, 0.05)
    if not ui_force_symmetric:
        prior_rho21_raw = st.slider("prior mean rho₂₁_raw", -2.5, 2.5, 0.0, 0.05)
    else:
        prior_rho21_raw = None
    prior_std = st.slider("theta prior std", 0.2, 8.0, 3.0, 0.1)
    hessian_step = st.slider("Laplace Hessian step", 0.005, 0.150, 0.035, 0.005)
    laplace_samples = st.slider("Laplace posterior samples", 200, 5000, 1200, 100)
    band_samples = st.slider("FDM samples for credible band", 10, 150, 60, 10)

    run_high_cost_mcmc = st.checkbox("Run high-cost MCMC reliability", value=False)
    mcmc_steps = st.slider("MCMC steps", 100, 3000, 700, 100)
    mcmc_burn = st.slider("MCMC burn-in", 0, 1500, 200, 50)
    mcmc_proposal = st.slider("MCMC proposal std", 0.005, 0.300, 0.045, 0.005)
    ui_marginalize_sigma = st.checkbox(
        "Marginalize σ (joint θ-σ sampling)",
        value=False,
        help=(
            "Treat observation noise σ as unknown and sample it jointly with θ. "
            "Uses half-Cauchy(0, 0.1) prior on σ.  Produces σ posterior samples "
            "in addition to D-matrix credible intervals."
        ),
    )

    with st.expander("MCMC設定の読み方", expanded=False):
        st.markdown(
            """
| 設定 | 意味 |
|---|---|
| `MCMC steps` | 提案・採択判定を行う総ステップ数 |
| `MCMC burn-in` | 初期値の影響が強い前半サンプルを捨てる数 |
| `MCMC proposal std` | 候補パラメータを動かす幅 |

`proposal std` が大きすぎると採択率が下がり、小さすぎると探索が遅くなります。  
まずは `0.03〜0.06` 程度から試すのが無難です。
"""
        )

    with st.expander("Likelihood contour", expanded=False):
        contour_n = st.slider("contour grid size", 7, 25, 11, 2)
        contour_half_width = st.slider("contour half width in theta", 0.05, 1.0, 0.35, 0.05)
        contour_axis_pair = st.selectbox(
            "contour axes",
            ["logD_NiNi vs logD_TaTa", "logD_NiNi vs rho_raw", "logD_TaTa vs rho_raw"],
        )

    run = st.button("Run Fig.11-style FDM → PINN", type="primary")


if "fig11_result_v12" not in st.session_state:
    st.session_state.fig11_result_v12 = None
    st.session_state.fig11_inputs_v12 = None


if run:
    set_seed(int(seed))

    self_D_norm = None
    self_log_diag = None
    self_D_norm_lr = None
    self_log_diag_lr = None
    if bool(use_self_diffusion) and float(self_D_Ni_phys) > 0.0 and float(self_D_Ta_phys) > 0.0:
        self_D_phys_matrix = np.array([[float(self_D_Ni_phys), 0.0], [0.0, float(self_D_Ta_phys)]], dtype=float)
        self_D_norm = physical_to_normalized_D(self_D_phys_matrix, float(paper_length_um), float(paper_time_h))
        self_log_diag = np.array([
            safe_log_from_positive(self_D_norm[0, 0], log_d11_init),
            safe_log_from_positive(self_D_norm[1, 1], log_d22_init),
        ], dtype=float)

    lr_self_available = (
        float(self_D_Ni_left_phys) > 0.0 and float(self_D_Ta_left_phys) > 0.0
        and float(self_D_Ni_right_phys) > 0.0 and float(self_D_Ta_right_phys) > 0.0
    )
    if bool(use_self_diffusion) and lr_self_available:
        D_left_phys = np.array([[float(self_D_Ni_left_phys), 0.0], [0.0, float(self_D_Ta_left_phys)]], dtype=float)
        D_right_phys = np.array([[float(self_D_Ni_right_phys), 0.0], [0.0, float(self_D_Ta_right_phys)]], dtype=float)
        D_left_norm = physical_to_normalized_D(D_left_phys, float(paper_length_um), float(paper_time_h))
        D_right_norm = physical_to_normalized_D(D_right_phys, float(paper_length_um), float(paper_time_h))
        self_D_norm_lr = {"left": D_left_norm, "right": D_right_norm}
        self_log_diag_lr = np.array([
            safe_log_from_positive(D_left_norm[0, 0], log_d11_init),
            safe_log_from_positive(D_left_norm[1, 1], log_d22_init),
            safe_log_from_positive(D_right_norm[0, 0], log_d11_init),
            safe_log_from_positive(D_right_norm[1, 1], log_d22_init),
        ], dtype=float)

    log_d11_train_init = float(log_d11_init)
    log_d22_train_init = float(log_d22_init)
    log_d11_right_train_init = float(log_d11_right_init)
    log_d22_right_train_init = float(log_d22_right_init)
    if self_log_diag_lr is not None and diffusion_model_mode == "left/right D" and diag_constraint_mode in ["initialize only", "weak diagonal prior", "fix diagonal terms"]:
        log_d11_train_init = float(self_log_diag_lr[0])
        log_d22_train_init = float(self_log_diag_lr[1])
        log_d11_right_train_init = float(self_log_diag_lr[2])
        log_d22_right_train_init = float(self_log_diag_lr[3])
    elif self_log_diag is not None and diag_constraint_mode in ["initialize only", "weak diagonal prior", "fix diagonal terms"]:
        log_d11_train_init = float(self_log_diag[0])
        log_d22_train_init = float(self_log_diag[1])
        log_d11_right_train_init = float(self_log_diag[0])
        log_d22_right_train_init = float(self_log_diag[1])

    # =========================================================================
    # FDM teacher data generation
    # =========================================================================
    if use_chemical_potential:
        # --- RS mode: use μ-M driven FDM to ensure model consistency ---
        st.info("Step 1/2: RS (μ-M) FDMでCo / Ni-0.10Ta diffusion coupleを計算しています。")

        def _parse_omega_str_early(s: str, n_pairs: int) -> np.ndarray:
            vals = np.array([float(v.strip()) for v in s.split(",")], dtype=float)
            if vals.size != n_pairs:
                raise ValueError(f"Expected {n_pairs} Omega values, got {vals.size}")
            return vals

        _n_pairs_teacher = 3
        _theta_left_teacher = _parse_omega_str_early(omega_left_init_str, _n_pairs_teacher)
        if learn_lr_omega:
            _theta_right_teacher = _parse_omega_str_early(omega_right_init_str, _n_pairs_teacher)
        else:
            _theta_right_teacher = _theta_left_teacher.copy()

        _log_M_endmembers_teacher = None
        if rs_use_comp_dep_mobility:
            _log_M_endmembers_teacher = np.array([
                [rs_logM_00, rs_logM_01, rs_logM_02],
                [rs_logM_10, rs_logM_11, rs_logM_12],
            ], dtype=float)
            _mobility_teacher = np.eye(2) * 1.0e-2
        else:
            _mobility_teacher = np.eye(2) * float(rs_M_diag) + (np.ones((2, 2)) - np.eye(2)) * float(rs_M_offdiag)

        data = make_training_data_rs(
            theta_left=_theta_left_teacher,
            theta_right=_theta_right_teacher,
            mobility=_mobility_teacher,
            RT=float(rs_RT),
            x_interface=0.5,
            omega_width=float(rs_omega_blend_width),
            phase_width=float(phase_interface_width),
            dt=float(rs_fdm_dt),
            nsteps=int(rs_fdm_nsteps),
            save_every=int(rs_fdm_save_every),
            nx_fdm=nx_fdm,
            n_obs=n_obs,
            n_ic=n_ic,
            n_bc_each=n_bc_each,
            n_f=n_f,
            noise=noise,
            seed=int(seed),
            t_start_fraction=t_start_fraction,
            n_exp_points=n_exp_points,
            pseudo_exp_time_mode=str(pseudo_exp_time_mode),
            pseudo_exp_time_slices=int(pseudo_exp_time_slices),
            append_pseudo_exp_to_training=bool(append_pseudo_exp_to_training),
            learn_lr_omega=bool(learn_lr_omega),
            noise_model=noise_model_key,
            log_M_endmembers=_log_M_endmembers_teacher,
        )
    else:
        # --- Fickian mode: D-matrix FDM (original) ---
        st.info("Step 1/2: Co / Ni-0.10Ta sharp-interface diffusion coupleをFDMで計算しています。")
        data = make_training_data(
            log_d11=log_d11_true,
            log_d22=log_d22_true,
            rho_raw=rho_raw_true,
            t_max=t_max,
            nx_fdm=nx_fdm,
            nt_fdm=nt_fdm,
            n_obs=n_obs,
            n_ic=n_ic,
            n_bc_each=n_bc_each,
            n_f=n_f,
            noise=noise,
            seed=int(seed),
            t_start_fraction=t_start_fraction,
            n_exp_points=n_exp_points,
            pseudo_exp_time_mode=str(pseudo_exp_time_mode),
            pseudo_exp_time_slices=int(pseudo_exp_time_slices),
            append_pseudo_exp_to_training=bool(append_pseudo_exp_to_training),
            fdm_teacher_mode=str(fdm_teacher_mode),
            log_d11_right=float(log_d11_true_right),
            log_d22_right=float(log_d22_true_right),
            rho_raw_right=float(rho_raw_true_right),
            phase_width=float(phase_interface_width),
            rho21_raw=float(rho21_raw_true) if rho21_raw_true is not None else None,
            rho21_raw_right=float(rho21_raw_true_right) if rho21_raw_true_right is not None else None,
            noise_model=noise_model_key,
        )

    st.success("FDM teacher calculation and pseudo experimental data generation finished.")

    with st.expander("Preview FDM teacher data before PINNs training", expanded=True):
        st.markdown(
            """
FDM教師データと疑似実験点が生成された直後の確認図です。  
この時点ではPINNs学習はまだ始まっていません。ここで初期プロファイル、最終プロファイル、疑似実験点の位置・ノイズ量を確認します。

注意：FDM教師データはsharp step初期条件から積分しているため、粗い格子や長時間設定では界面近傍に微小なkinkが残ることがあります。必要に応じて `FDM spatial grid` を増やしてください。
"""
        )
        st.plotly_chart(
            fdm_teacher_preview_plot(
                data.x_grid,
                data.t_grid,
                data.C_fdm,
                data.x_exp,
                data.c_exp,
                span_um=float(span_um),
                n_time_lines=4,
                annealing_time_h=float(paper_time_h),
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            fdm_teacher_preview_difference_plot(
                data.x_grid,
                data.C_fdm[-1],
                data.x_exp,
                data.c_exp,
                span_um=float(span_um),
            ),
            use_container_width=True,
        )

        preview_cols = st.columns(4)
        preview_cols[0].metric("FDM x grid", f"{len(data.x_grid):,}")
        preview_cols[1].metric("FDM saved frames", f"{len(data.t_grid):,}")
        preview_cols[2].metric("final pseudo-exp points", f"{len(data.x_exp):,}")
        preview_cols[3].metric("noise setting", f"{float(noise):.3g}")

        mt_preview_cols = st.columns(3)
        mt_preview_cols[0].metric("pseudo-exp time mode", str(pseudo_exp_time_mode))
        mt_preview_cols[1].metric("pseudo-exp total points", f"{len(data.x_exp_all):,}")
        mt_preview_cols[2].metric("pseudo-exp time slices", f"{len(data.exp_time_indices):,}")
        if str(pseudo_exp_time_mode) == "multi-time":
            st.info(
                "Multi-time pseudo-exp is active. These points are appended to PINNs observations, "
                "so the inverse problem uses time-evolution information, not only the final profile."
            )
        if str(fdm_teacher_mode) == "left/right D":
            st.warning(
                "FDM teacher uses left/right D. This is the intended setting for FCC/BCC-like contrast examples. "
                "Check that the fast side broadens much more rapidly than the slow side."
            )

    # =========================================================================
    # Branch: Chemical potential (RS) or Fickian D matrix
    # =========================================================================
    if use_chemical_potential:
        def _parse_omega_str(s: str, n_pairs: int) -> np.ndarray:
            vals = np.array([float(v.strip()) for v in s.split(",")], dtype=float)
            if vals.size != n_pairs:
                raise ValueError(f"Expected {n_pairs} Omega values, got {vals.size}")
            return vals

        n_pairs_rs = 3
        theta_left_init_vals = _parse_omega_str(omega_left_init_str, n_pairs_rs)
        if learn_lr_omega:
            theta_right_init_vals = _parse_omega_str(omega_right_init_str, n_pairs_rs)
        else:
            theta_right_init_vals = theta_left_init_vals.copy()

        log_M_endmembers_init = None
        rs_train_mob = False
        if rs_use_comp_dep_mobility:
            log_M_endmembers_init = np.array([
                [rs_logM_00, rs_logM_01, rs_logM_02],
                [rs_logM_10, rs_logM_11, rs_logM_12],
            ], dtype=float)
            rs_train_mob = bool(rs_train_mobility)
            mobility_rs = np.eye(2) * 1.0e-2
        else:
            mobility_rs = np.eye(2) * float(rs_M_diag) + (np.ones((2, 2)) - np.eye(2)) * float(rs_M_offdiag)

        rs_model = TernaryRegularSolutionPINN(
            width=width,
            depth=depth,
            activation=str(activation),
            theta_left_init=theta_left_init_vals,
            theta_right_init=theta_right_init_vals,
            learn_left_right_omega=bool(learn_lr_omega),
            x_interface=0.5,
            omega_width=float(rs_omega_blend_width),
            RT=float(rs_RT),
            train_omega=True,
            log_M_endmembers_init=log_M_endmembers_init,
            train_mobility=rs_train_mob,
            direct_output=bool(ui_direct_output),
        )

        st.info("Step 2/2: PINNsでΩ相互作用項を推定しています (chemical potential mode)...")
        progress = st.progress(0.0)
        status = st.empty()
        rs_model, rs_hist, rs_train_time = train_pinn_rs(
            data=data,
            model=rs_model,
            mobility=mobility_rs,
            epochs=int(epochs),
            lr=float(lr),
            weights={"data": w_data, "ic": w_ic, "bc": w_bc, "phys": w_phys},
            progress=progress,
            status=status,
            n_collocation=int(rs_n_collocation),
            w_omega_prior=float(rs_w_omega_prior),
            omega_prior_left=theta_left_init_vals,
            omega_prior_right=theta_right_init_vals if learn_lr_omega else None,
            adaptive_weights=bool(ui_adaptive_weights),
            rba_update_every=50,
            compile_model=bool(ui_torch_compile),
        )

        theta_l_disp, theta_r_disp = rs_model.theta_display()

        if learn_lr_omega:
            theta_hat_rs = np.concatenate([theta_l_disp, theta_r_disp])
            prior_mean_rs = np.concatenate([theta_left_init_vals, theta_right_init_vals])
        else:
            theta_hat_rs = theta_l_disp.copy()
            prior_mean_rs = theta_left_init_vals.copy()

        # C11 fix: phase_interface_width controls initial c(x) sigmoid width;
        # rs_omega_blend_width (separate slider) controls Omega spatial blending.
        # Guard values (5e-3) avoid log(0) in chemical potential.
        _eg2 = 5.0e-3
        c_left_rs = np.array([1.0 - _eg2, _eg2 / 2, _eg2 / 2], dtype=float)
        c_left_rs /= c_left_rs.sum()
        c_right_rs = np.array([_eg2 / 2, 0.9, 0.1], dtype=float)
        c_right_rs /= c_right_rs.sum()
        c0_full_rs = make_initial_profile_ternary_rs(
            data.x_grid, c_left_rs, c_right_rs,
            x0=0.5, width=float(phase_interface_width),
        )

        # t_exp_all is in normalized time [0,1]; FDM NLL needs physical time.
        _t_exp_physical = data.t_exp_all * data.rs_t_max_physical if data.rs_t_max_physical > 0 else data.t_exp_all
        nll_fun_rs = lambda th: gaussian_nll_multitime_rs(
            th, 3, learn_lr_omega, c0_full_rs, data.x_grid,
            data.x_exp_all, _t_exp_physical, data.c_exp_all,
            sigma=float(rs_like_sigma), dt=float(rs_fdm_dt),
            nsteps=int(rs_fdm_nsteps), save_every=int(rs_fdm_save_every),
            mobility=mobility_rs, RT=float(rs_RT),
            x_interface=0.5, omega_width=float(rs_omega_blend_width),
            prior_mean=prior_mean_rs, prior_std=5.0,
            log_M_endmembers=log_M_endmembers_init,
        )

        refine_info_rs = None
        # FDM refinement is deferred to the result display section
        # so the user can inspect profiles immediately after PINN training.

        rs_result = TrainResultRS(
            model=rs_model,
            data=data,
            history=rs_hist,
            train_time=rs_train_time,
            mobility=mobility_rs,
        )

        st.session_state.fig11_result_v12 = rs_result
        st.session_state.fig11_inputs_v12 = {
            "t_max": float(data.t_grid[-1]),
            "use_chemical_potential": True,
            "learn_lr_omega": bool(learn_lr_omega),
            "theta_hat_rs": theta_hat_rs.tolist(),
            "refine_info_rs": None,
            "do_fdm_refine": bool(do_fdm_refine),
            "fdm_refine_maxiter": int(fdm_refine_maxiter),
            "prior_mean_rs": prior_mean_rs.tolist(),
            "fdm_teacher_mode": str(fdm_teacher_mode),
            "span_um": span_um,
            "noise": noise,
            "seed": int(seed),
            "rs_RT": float(rs_RT),
            "rs_M_diag": float(rs_M_diag) if not rs_use_comp_dep_mobility else 1.0e-2,
            "rs_M_offdiag": float(rs_M_offdiag) if not rs_use_comp_dep_mobility else 0.0,
            "log_M_endmembers": log_M_endmembers_init,
            "rs_use_comp_dep_mobility": rs_use_comp_dep_mobility,
            "rs_system_preset": rs_system_preset,
            "rs_fdm_dt": float(rs_fdm_dt),
            "rs_fdm_nsteps": int(rs_fdm_nsteps),
            "rs_fdm_save_every": int(rs_fdm_save_every),
            "rs_like_sigma": float(rs_like_sigma),
            "rs_hessian_step": float(rs_hessian_step),
            "rs_laplace_samples": int(rs_laplace_samples),
            "rs_band_samples": int(rs_band_samples),
            "rs_run_mcmc": bool(rs_run_mcmc),
            "rs_mcmc_steps": int(rs_mcmc_steps),
            "rs_mcmc_burn": int(rs_mcmc_burn),
            "rs_mcmc_proposal": float(rs_mcmc_proposal),
            "rs_skip_reliability": bool(rs_skip_reliability),
            "phase_interface_width": float(phase_interface_width),
            "rs_omega_blend_width": float(rs_omega_blend_width),
            "paper_time_h": float(paper_time_h),
            "paper_length_um": float(paper_length_um),
            "show_zero_interaction_reference": show_zero_interaction_reference,
        }
        status.success("Completed. Rendering result tabs...")
        safe_streamlit_rerun()

    # =========================================================================
    # Fickian D matrix mode: original approach
    # =========================================================================
    st.info("Step 2/2: PINNsでCo/Ni/TaプロファイルとNi-Ta相互拡散行列を推定しています。")
    progress = st.progress(0.0)
    status = st.empty()
    result = train_pinn(
        data=data,
        log_d11_init=log_d11_train_init,
        log_d22_init=log_d22_train_init,
        rho_raw_init=rho_raw_init,
        width=width,
        depth=depth,
        activation=activation,
        epochs=epochs,
        lr=lr,
        weights={"data": w_data, "ic": w_ic, "bc": w_bc, "phys": w_phys},
        progress=progress,
        status=status,
        # C14 note: diag_prior_log / fix_diagonal_from_prior logic is complex
        # because it handles single D vs left/right D vs None across 3 modes.
        diag_prior_log=(
            self_log_diag_lr if (
                diffusion_model_mode == "left/right D"
                and self_log_diag_lr is not None
                and diag_constraint_mode in ["weak diagonal prior", "fix diagonal terms"]
            )
            else self_log_diag if (
                self_log_diag is not None
                and diag_constraint_mode in ["weak diagonal prior", "fix diagonal terms"]
            )
            else None
        ),
        diag_prior_weight=float(diag_prior_weight) if diag_constraint_mode == "weak diagonal prior" else 0.0,
        fix_diagonal_from_prior=bool(
            (
                diffusion_model_mode == "left/right D"
                and self_log_diag_lr is not None
                and diag_constraint_mode == "fix diagonal terms"
            )
            or (
                self_log_diag is not None
                and diag_constraint_mode == "fix diagonal terms"
            )
        ),
        diffusion_model_mode=str(diffusion_model_mode),
        log_d11_right_init=float(log_d11_right_train_init),
        log_d22_right_init=float(log_d22_right_train_init),
        rho_raw_right_init=float(rho_raw_right_init),
        phase_interface=0.5,
        phase_width=float(phase_interface_width),
        rho21_raw_init=float(rho21_raw_true) if rho21_raw_true is not None else None,
        rho21_raw_right_init=float(rho21_raw_true_right) if rho21_raw_true_right is not None else None,
        force_symmetric=ui_force_symmetric,
        adaptive_weights=bool(ui_adaptive_weights),
        direct_output=bool(ui_direct_output),
        compile_model=bool(ui_torch_compile),
    )

    st.session_state.fig11_result_v12 = result
    st.session_state.fig11_inputs_v12 = {
        "t_max": t_max,
        "fdm_teacher_mode": str(fdm_teacher_mode),
        "log_d11_true": float(log_d11_true),
        "log_d22_true": float(log_d22_true),
        "rho_raw_true": float(rho_raw_true),
        "log_d11_true_right": float(log_d11_true_right),
        "log_d22_true_right": float(log_d22_true_right),
        "rho_raw_true_right": float(rho_raw_true_right),
        "rho21_raw_true": float(rho21_raw_true) if rho21_raw_true is not None else None,
        "rho21_raw_true_right": float(rho21_raw_true_right) if rho21_raw_true_right is not None else None,
        "force_symmetric": bool(ui_force_symmetric),
        "fcc_bcc_preset": str(fcc_bcc_preset),
        "abstract_validation_preset": str(fcc_bcc_preset == "TOFA abstract validation"),
        "fixed_diagonal_abstract_default": bool(is_tofa_abstract_preset and diag_constraint_mode == "fix diagonal terms"),
        "span_um": span_um,
        "show_zero_interaction_reference": show_zero_interaction_reference,
        "zero_interaction_source": zero_interaction_source,
        "noise": noise,
        "n_exp_points": int(n_exp_points),
        "pseudo_exp_time_mode": str(pseudo_exp_time_mode),
        "pseudo_exp_time_slices": int(pseudo_exp_time_slices),
        "append_pseudo_exp_to_training": bool(append_pseudo_exp_to_training),
        "like_sigma": like_sigma,
        "rel_nx": rel_nx,
        "rel_nt": rel_nt,
        "prior_mean": (
            np.array([prior_log_d11, prior_log_d22, prior_rho_raw, prior_rho21_raw], dtype=float)
            if prior_rho21_raw is not None
            else np.array([prior_log_d11, prior_log_d22, prior_rho_raw], dtype=float)
        ),
        "prior_std": prior_std,
        "hessian_step": hessian_step,
        "laplace_samples": laplace_samples,
        "band_samples": band_samples,
        "run_high_cost_mcmc": run_high_cost_mcmc,
        "mcmc_steps": mcmc_steps,
        "mcmc_burn": mcmc_burn,
        "mcmc_proposal": mcmc_proposal,
        "marginalize_sigma": bool(ui_marginalize_sigma),
        "adaptive_weights": bool(ui_adaptive_weights),
        "direct_output": bool(ui_direct_output),
        "seed": int(seed),
        "contour_n": contour_n,
        "contour_half_width": contour_half_width,
        "contour_axis_pair": contour_axis_pair,
        "paper_T_C": float(paper_T_C),
        "paper_time_h": float(paper_time_h),
        "paper_length_um": float(paper_length_um),
        "use_self_diffusion": bool(use_self_diffusion),
        "self_D_Ni_phys": float(self_D_Ni_phys),
        "self_D_Ta_phys": float(self_D_Ta_phys),
        "diag_constraint_mode": diag_constraint_mode,
        "diag_prior_weight": float(diag_prior_weight),
        "self_D_norm": None if self_D_norm is None else self_D_norm,
        "self_log_diag": None if self_log_diag is None else self_log_diag,
        "diffusion_model_mode": str(diffusion_model_mode),
        "log_d11_right_init": float(log_d11_right_init),
        "log_d22_right_init": float(log_d22_right_init),
        "rho_raw_right_init": float(rho_raw_right_init),
        "phase_interface_width": float(phase_interface_width),
        "self_D_Ni_left_phys": float(self_D_Ni_left_phys),
        "self_D_Ta_left_phys": float(self_D_Ta_left_phys),
        "self_D_Ni_right_phys": float(self_D_Ni_right_phys),
        "self_D_Ta_right_phys": float(self_D_Ta_right_phys),
        "self_D_norm_lr": None if self_D_norm_lr is None else self_D_norm_lr,
        "self_log_diag_lr": None if self_log_diag_lr is None else self_log_diag_lr,
    }
    status.success("Completed. Rendering result tabs...")
    safe_streamlit_rerun()


result = st.session_state.fig11_result_v12
inputs = st.session_state.fig11_inputs_v12

if result is None:
    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="note"><b>1. FDM</b><br>Co / Ni-0.10Ta のsharp interfaceから連成拡散を解きます。</div>', unsafe_allow_html=True)
    c2.markdown('<div class="note"><b>2. PINNs</b><br>Co, Ni, Ta濃度場と2x2相互拡散行列を推定します。</div>', unsafe_allow_html=True)
    c3.markdown('<div class="note"><b>3. Reliability bands</b><br>尤度からプロファイル信頼帯と係数区間を出します。</div>', unsafe_allow_html=True)
    st.stop()


st.success("Loaded completed calculation from session state. Rendering result tabs below.")

# =========================================================================
# Chemical potential result display
# =========================================================================
if type(result).__name__ == "TrainResultRS":
    rs_result = result
    rs_model = rs_result.model
    rs_data = rs_result.data
    rs_hist = rs_result.history
    span_um = float(inputs["span_um"])

    x = rs_data.x_grid
    t = rs_data.t_grid
    C_pinn_rs = evaluate_model_on_grid_rs(rs_model, x, t)
    C_fdm_rs = rs_data.C_fdm
    C_diff_rs = C_pinn_rs - C_fdm_rs

    theta_l_disp, theta_r_disp = rs_model.theta_display()
    pair_names = ["Omega_CoNi", "Omega_CoTa", "Omega_NiTa"]

    all_rmse = float(np.sqrt(np.mean(C_diff_rs ** 2)))
    final_rmse_each = np.sqrt(np.mean((C_diff_rs[-1] ** 2), axis=0))

    m1, m2, m3 = st.columns(3)
    m1.metric("Total profile RMSE", f"{all_rmse:.3e}")
    m2.metric("Final RMSE Co/Ni/Ta", f"{final_rmse_each[0]:.2e}/{final_rmse_each[1]:.2e}/{final_rmse_each[2]:.2e}")
    m3.metric("Analysis mode", "Regular-solution chemical potential")

    st.markdown("### Estimated Omega interaction terms")
    omega_rows = []
    for k, pname in enumerate(pair_names):
        omega_rows.append({
            "pair": pname,
            "estimated_left": float(theta_l_disp[k]),
            "estimated_right": float(theta_r_disp[k]),
        })
    st.dataframe(pd.DataFrame(omega_rows), use_container_width=True)

    if inputs.get("refine_info_rs") is not None:
        ri = inputs["refine_info_rs"]
        st.markdown("### FDM likelihood refinement")
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("NLL before", f"{ri['nll_before']:.2f}")
        rc2.metric("NLL after", f"{ri['nll_after']:.2f}")
        rc3.metric("Evaluations", f"{ri['nfev']}")
    elif inputs.get("do_fdm_refine", False):
        st.markdown("### FDM順問題による尤度でΩを再最適化")
        st.caption(
            "PINN学習後のΩ推定値を出発点として、FDM順問題の尤度を最小化してΩを精密化します。"
        )
        if st.button("Ωを再最適化する", key="btn_fdm_refine_rs"):
            _eg_ref = 5.0e-3
            _c_left_ref = np.array([1.0 - _eg_ref, _eg_ref / 2, _eg_ref / 2])
            _c_left_ref /= _c_left_ref.sum()
            _c_right_ref = np.array([_eg_ref / 2, 0.9, 0.1])
            _c_right_ref /= _c_right_ref.sum()
            c0_full_ref = make_initial_profile_ternary_rs(
                x, _c_left_ref, _c_right_ref,
                x0=0.5, width=float(inputs["phase_interface_width"]),
            )
            _learn_lr = bool(inputs["learn_lr_omega"])
            _prior_mean = np.array(inputs["prior_mean_rs"], dtype=float)
            _t_exp_phys_ref = rs_data.t_exp_all * rs_data.rs_t_max_physical if rs_data.rs_t_max_physical > 0 else rs_data.t_exp_all
            nll_fun_ref = lambda th: gaussian_nll_multitime_rs(
                th, 3, _learn_lr, c0_full_ref, x,
                rs_data.x_exp_all, _t_exp_phys_ref, rs_data.c_exp_all,
                sigma=float(inputs["rs_like_sigma"]),
                dt=float(inputs["rs_fdm_dt"]),
                nsteps=int(inputs["rs_fdm_nsteps"]),
                save_every=int(inputs["rs_fdm_save_every"]),
                mobility=rs_result.mobility,
                RT=float(inputs["rs_RT"]),
                x_interface=0.5,
                omega_width=float(inputs.get("rs_omega_blend_width", inputs["phase_interface_width"])),
                prior_mean=_prior_mean, prior_std=5.0,
                log_M_endmembers=inputs.get("log_M_endmembers"),
            )
            _theta_hat = np.array(inputs["theta_hat_rs"], dtype=float)
            st.info("FDM順問題による尤度でΩを再最適化しています...")
            refine_status = st.empty()
            refine_bar = st.progress(0.0)
            _omega_pair_names = ["Ω_CoNi", "Ω_CoTa", "Ω_NiTa"]
            theta_refined, refine_info = refine_omega_by_fdm_likelihood(
                nll_fun_ref, _theta_hat,
                maxiter=int(inputs.get("fdm_refine_maxiter", 180)),
                verbose=False,
                progress_status=refine_status, progress_bar=refine_bar,
                pair_names=_omega_pair_names,
            )
            if np.all(np.isfinite(theta_refined)):
                overwrite_model_omega_from_theta(rs_model, theta_refined, _learn_lr)
                inputs["theta_hat_rs"] = theta_refined.tolist()
            inputs["refine_info_rs"] = refine_info
            st.session_state.fig11_inputs_v12 = inputs
            _ri = refine_info
            st.success(
                f"FDM refinement 完了: NLL {_ri['nll_before']:.2f} → {_ri['nll_after']:.2f}  "
                f"({_ri['nfev']} evals, {_ri.get('elapsed_s', 0):.1f}s, "
                f"{'converged' if _ri['success'] else 'maxiter reached'})"
            )
            safe_streamlit_rerun()

    tab_rs1, tab_rs2, tab_rs3, tab_rs4, tab_rs5 = st.tabs(
        ["Profiles", "Training", "Omega reliability", "Data", "Publication figures"]
    )

    with tab_rs1:
        st.markdown("### Final diffusion-couple profile (regular-solution)")
        dist = distance_um_from_x(x, span_um)
        dist_exp = distance_um_from_x(rs_data.x_exp_all, span_um)
        fig = go.Figure()
        for j_comp, comp in enumerate(COMPONENTS):
            fig.add_trace(go.Scatter(
                x=dist_exp, y=rs_data.c_exp_all[:, j_comp], mode="markers",
                name=f"Exp. {comp}",
                marker=dict(symbol=SYMBOLS[comp], size=7, color=COLORS[comp], line=dict(width=1.4)),
            ))
        for j_comp, comp in enumerate(COMPONENTS):
            fig.add_trace(go.Scatter(
                x=dist, y=C_fdm_rs[-1, :, j_comp], mode="lines",
                name=f"FDM {comp}", line=dict(width=3, color=COLORS[comp]),
            ))
            fig.add_trace(go.Scatter(
                x=dist, y=C_pinn_rs[-1, :, j_comp], mode="lines",
                name=f"PINNs {comp}", line=dict(width=2.5, dash="dash", color=COLORS[comp]),
            ))
        fig.update_xaxes(title="Distance (µm)")
        fig.update_yaxes(title="Mole Fraction", range=[-0.04, 1.04])
        st.plotly_chart(clean_layout(fig, "Regular-solution profile: Co / Ni-Ta", 560), use_container_width=True)

        st.markdown("### Multi-time profiles")
        indices_rs = sorted(set(np.linspace(0, len(t) - 1, 5).astype(int).tolist()))
        for idx_time in indices_rs:
            fig_t = go.Figure()
            for j_comp, comp in enumerate(COMPONENTS):
                fig_t.add_trace(go.Scatter(
                    x=dist, y=C_fdm_rs[idx_time, :, j_comp], mode="lines",
                    name=f"FDM {comp}", line=dict(width=2.5, color=COLORS[comp]),
                ))
                fig_t.add_trace(go.Scatter(
                    x=dist, y=C_pinn_rs[idx_time, :, j_comp], mode="lines",
                    name=f"PINNs {comp}", line=dict(width=2.0, dash="dash", color=COLORS[comp]),
                ))
            paper_time_h_disp = float(inputs.get("paper_time_h", 160.0))
            fig_t.update_xaxes(title="Distance (µm)")
            fig_t.update_yaxes(title="Mole fraction", range=[-0.04, 1.04])
            st.plotly_chart(
                clean_layout(fig_t, f"t = {format_time_label(float(t[idx_time]), float(t[-1]), paper_time_h_disp)}", 400),
                use_container_width=True,
            )

    with tab_rs2:
        st.markdown("### Loss history")
        fig_loss = go.Figure()
        for col in ["loss", "data", "ic", "physics"]:
            if col in rs_hist.columns:
                fig_loss.add_trace(go.Scatter(x=rs_hist["epoch"], y=rs_hist[col], mode="lines", name=col))
        fig_loss.update_xaxes(title="epoch")
        fig_loss.update_yaxes(title="loss", type="log")
        st.plotly_chart(clean_layout(fig_loss, "Loss functions (regular-solution)", 430), use_container_width=True)

        st.markdown("### Omega convergence")
        fig_omega = go.Figure()
        omega_cols_l = ["Omega_CoNi_left", "Omega_CoTa_left", "Omega_NiTa_left"]
        omega_cols_r = ["Omega_CoNi_right", "Omega_CoTa_right", "Omega_NiTa_right"]
        for k, pname in enumerate(pair_names):
            if omega_cols_l[k] in rs_hist.columns:
                fig_omega.add_trace(go.Scatter(x=rs_hist["epoch"], y=rs_hist[omega_cols_l[k]], mode="lines", name=f"{pname} left"))
            if omega_cols_r[k] in rs_hist.columns:
                fig_omega.add_trace(go.Scatter(x=rs_hist["epoch"], y=rs_hist[omega_cols_r[k]], mode="lines", name=f"{pname} right"))
        fig_omega.update_xaxes(title="epoch")
        fig_omega.update_yaxes(title="Omega value")
        st.plotly_chart(clean_layout(fig_omega, "Omega pair-interaction convergence", 450), use_container_width=True)

        st.dataframe(rs_hist, use_container_width=True)

    with tab_rs3:
        st.markdown("### Omega-based reliability (regular-solution)")
        if inputs.get("rs_skip_reliability", False):
            st.warning("Omega reliability evaluation was skipped.")
        else:
            theta_hat_rs = np.array(inputs["theta_hat_rs"], dtype=float)
            learn_lr_omega_val = bool(inputs["learn_lr_omega"])

            _eg = 5.0e-3
            c0_full_for_nll = make_initial_profile_ternary_rs(
                x, np.array([1.0 - _eg, _eg / 2, _eg / 2]),
                np.array([_eg / 2, 0.9, 0.1]),
                x0=0.5, width=float(inputs["phase_interface_width"]),
            )
            _t_exp_phys_nll = rs_data.t_exp_all * rs_data.rs_t_max_physical if rs_data.rs_t_max_physical > 0 else rs_data.t_exp_all
            nll_fun_rs = lambda th: gaussian_nll_multitime_rs(
                th, 3, learn_lr_omega_val,
                c0_full_for_nll, x,
                rs_data.x_exp_all, _t_exp_phys_nll, rs_data.c_exp_all,
                sigma=float(inputs["rs_like_sigma"]),
                dt=float(inputs["rs_fdm_dt"]),
                nsteps=int(inputs["rs_fdm_nsteps"]),
                save_every=int(inputs["rs_fdm_save_every"]),
                mobility=rs_result.mobility,
                RT=float(inputs["rs_RT"]),
                x_interface=0.5,
                omega_width=float(inputs.get("rs_omega_blend_width", inputs["phase_interface_width"])),
                prior_mean=theta_hat_rs,
                prior_std=5.0,
                log_M_endmembers=inputs.get("log_M_endmembers"),
            )

            with st.spinner("Computing Laplace reliability for Omega..."):
                low_rel_rs = laplace_reliability_rs(
                    nll_fun_rs, theta_hat_rs,
                    hessian_step=float(inputs["rs_hessian_step"]),
                    n_samples=int(inputs["rs_laplace_samples"]),
                    seed=int(inputs["seed"]),
                )

            st.markdown("#### Laplace Hessian diagnostics")
            h_min_rs = float(low_rel_rs.get("hessian_min_eig", np.array([np.nan]))[0])
            h_non_pd_rs = bool(low_rel_rs.get("hessian_non_pd", np.array([False]))[0])
            cov_clipped_rs = bool(low_rel_rs.get("covariance_was_clipped", np.array([False]))[0])
            lhd1, lhd2, lhd3 = st.columns(3)
            lhd1.metric("min eig(H)", f"{h_min_rs:.3e}")
            lhd2.metric("Hessian non-PD", "yes" if h_non_pd_rs else "no")
            lhd3.metric("cov eig clipped", "yes" if cov_clipped_rs else "no")

            # C6 fix: surface non-PD warning to user in RS mode
            if h_non_pd_rs:
                st.error(
                    "RS Laplace warning: the Hessian has non-positive eigenvalue(s). "
                    "The Omega estimate may be near a saddle/flat direction. "
                    "The covariance was numerically repaired (eigen-floor) and "
                    "should be interpreted cautiously."
                )
            elif cov_clipped_rs:
                st.warning(
                    "RS Laplace warning: covariance eigenvalues were clipped. "
                    "Intervals may be affected by regularization."
                )

            st.markdown("#### Omega posterior summary")
            samples_rs = low_rel_rs["samples"]
            n_pairs_show = rs_model.n_pairs
            summary_rows = []
            for k in range(samples_rs.shape[1]):
                vals = samples_rs[:, k]
                if k < n_pairs_show:
                    lbl = f"{pair_names[k]}_left"
                else:
                    lbl = f"{pair_names[k - n_pairs_show]}_right"
                summary_rows.append({
                    "parameter": lbl,
                    "q025": float(np.quantile(vals, 0.025)),
                    "median": float(np.quantile(vals, 0.5)),
                    "q975": float(np.quantile(vals, 0.975)),
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                })
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

            st.markdown("#### Posterior credible band")
            with st.spinner("Computing credible band from Omega samples..."):
                _eg3 = 5.0e-3
                c0_full_for_band = make_initial_profile_ternary_rs(
                    x, np.array([1.0 - _eg3, _eg3 / 2, _eg3 / 2]),
                    np.array([_eg3 / 2, 0.9, 0.1]),
                    x0=0.5, width=float(inputs["phase_interface_width"]),
                )
                band_progress = st.progress(0.0)
                band_rs = posterior_band_from_samples_rs(
                    samples_rs, 3, learn_lr_omega_val,
                    c0_full_for_band, x,
                    dt=float(inputs["rs_fdm_dt"]),
                    nsteps=int(inputs["rs_fdm_nsteps"]),
                    save_every=int(inputs["rs_fdm_save_every"]),
                    mobility=rs_result.mobility,
                    RT=float(inputs["rs_RT"]),
                    x_interface=0.5,
                    omega_width=float(inputs.get("rs_omega_blend_width", inputs["phase_interface_width"])),
                    target_time=float(t[-1]),
                    max_samples=int(inputs["rs_band_samples"]),
                    progress_bar=band_progress,
                    log_M_endmembers=inputs.get("log_M_endmembers"),
                )

            fig_band = go.Figure()
            dist = distance_um_from_x(x, span_um)
            for j_comp, comp in enumerate(COMPONENTS):
                fig_band.add_trace(go.Scatter(
                    x=np.concatenate([dist, dist[::-1]]),
                    y=np.concatenate([band_rs["q025"][:, j_comp], band_rs["q975"][::-1, j_comp]]),
                    fill="toself", fillcolor=f"rgba({','.join(str(int(c*255)) for c in [0.2, 0.5, 0.8][:j_comp+1] + [0.2]*(3-j_comp-1))},0.15)",
                    line=dict(width=0), name=f"95% CI {comp}", showlegend=True,
                ))
                fig_band.add_trace(go.Scatter(
                    x=dist, y=band_rs["q500"][:, j_comp], mode="lines",
                    name=f"Median {comp}", line=dict(width=2, color=COLORS[comp]),
                ))
                fig_band.add_trace(go.Scatter(
                    x=dist, y=C_pinn_rs[-1, :, j_comp], mode="lines",
                    name=f"PINNs {comp}", line=dict(width=2, dash="dash", color=COLORS[comp]),
                ))
            fig_band.update_xaxes(title="Distance (µm)")
            fig_band.update_yaxes(title="Mole Fraction", range=[-0.04, 1.04])
            st.plotly_chart(
                clean_layout(fig_band, "Posterior credible band (regular-solution Omega)", 560),
                use_container_width=True,
            )

            if bool(inputs.get("rs_run_mcmc", False)):
                st.markdown("#### MCMC for Omega")
                mcmc_progress_rs = st.progress(0.0)
                with st.spinner("Running MCMC for Omega parameters..."):
                    high_rel_rs = mcmc_reliability_rs(
                        nll_fun_rs, theta_hat_rs,
                        n_steps=int(inputs["rs_mcmc_steps"]),
                        burn_in=int(inputs["rs_mcmc_burn"]),
                        proposal_std=float(inputs["rs_mcmc_proposal"]),
                        seed=int(inputs["seed"]),
                        progress_bar=mcmc_progress_rs,
                    )
                acc_rs = float(high_rel_rs["acceptance_rate"][0])
                st.metric("MCMC acceptance", f"{100*acc_rs:.1f}%")
                if acc_rs < 0.10:
                    st.warning("MCMC acceptance is low. Try lowering proposal std.")

                st.markdown("##### MCMC trace")
                fig_trace = go.Figure()
                mcmc_samples = high_rel_rs["samples"]
                for k in range(mcmc_samples.shape[1]):
                    if k < n_pairs_show:
                        lbl = f"{pair_names[k]}_left"
                    else:
                        lbl = f"{pair_names[k - n_pairs_show]}_right"
                    fig_trace.add_trace(go.Scatter(
                        x=np.arange(len(mcmc_samples)), y=mcmc_samples[:, k],
                        mode="lines", name=lbl, line=dict(width=1.5),
                    ))
                fig_trace.update_xaxes(title="MCMC sample index")
                fig_trace.update_yaxes(title="Omega value")
                st.plotly_chart(
                    clean_layout(fig_trace, "MCMC trace for Omega parameters", 450),
                    use_container_width=True,
                )

    with tab_rs4:
        st.markdown("### Raw data")
        st.markdown(f"x grid: {len(x)} points, t grid: {len(t)} frames")
        st.markdown(f"Pseudo-exp points: {len(rs_data.x_exp_all)}")
        st.markdown(f"Training obs: {len(rs_data.x_obs)}")
        st.dataframe(pd.DataFrame({
            "x_exp": np.asarray(rs_data.x_exp_all[:50]).ravel(),
            "t_exp": np.asarray(rs_data.t_exp_all[:50]).ravel(),
            **{f"c_exp_{comp}": np.asarray(rs_data.c_exp_all[:50, j]).ravel() for j, comp in enumerate(COMPONENTS)},
        }), use_container_width=True)

    with tab_rs5:
        st.markdown("### 論文用 matplotlib 図 (Publication-quality figures)")
        st.caption("フォントサイズ大、高解像度 PNG で出力します。右クリックまたはダウンロードボタンで保存できます。")

        st.markdown("#### 1. Final profile (regular-solution)")
        fig_pub_profile = pub_profile_figure(
            dist, C_fdm_rs[-1], C_pinn_rs[-1], dist_exp, rs_data.c_exp_all,
            title="Regular-solution profile: Co / Ni-Ta",
        )
        st.image(_pub_fig_to_bytes(fig_pub_profile), use_container_width=True)
        buf_profile = io.BytesIO()
        fig_pub_profile.savefig(buf_profile, format="png", dpi=300, bbox_inches="tight", facecolor="white")
        st.download_button("Download profile (PNG, 300 dpi)", buf_profile.getvalue(),
                           file_name="rs_profile_final.png", mime="image/png")
        plt.close(fig_pub_profile)

        st.markdown("#### 2. Multi-time profiles")
        paper_time_h_disp = float(inputs.get("paper_time_h", 160.0))
        fig_pub_mt = pub_multitime_figure(
            dist, t, C_fdm_rs, C_pinn_rs,
            n_slices=5, n_cols=3,
            tau_max=float(t[-1]), annealing_time_h=paper_time_h_disp,
        )
        st.image(_pub_fig_to_bytes(fig_pub_mt), use_container_width=True)
        buf_mt = io.BytesIO()
        fig_pub_mt.savefig(buf_mt, format="png", dpi=300, bbox_inches="tight", facecolor="white")
        st.download_button("Download multi-time (PNG, 300 dpi)", buf_mt.getvalue(),
                           file_name="rs_multitime_profiles.png", mime="image/png")
        plt.close(fig_pub_mt)

        st.markdown("#### 3. Training loss & Ω convergence")
        fig_pub_conv = pub_omega_convergence_figure(rs_hist, pair_names)
        st.image(_pub_fig_to_bytes(fig_pub_conv), use_container_width=True)
        buf_conv = io.BytesIO()
        fig_pub_conv.savefig(buf_conv, format="png", dpi=300, bbox_inches="tight", facecolor="white")
        st.download_button("Download convergence (PNG, 300 dpi)", buf_conv.getvalue(),
                           file_name="rs_training_convergence.png", mime="image/png")
        plt.close(fig_pub_conv)

        if inputs.get("refine_info_rs") is not None and inputs["refine_info_rs"].get("nll_history"):
            st.markdown("#### 4. FDM refinement NLL convergence")
            fig_pub_nll = pub_nll_convergence_figure(inputs["refine_info_rs"]["nll_history"])
            st.image(_pub_fig_to_bytes(fig_pub_nll), use_container_width=True)
            buf_nll = io.BytesIO()
            fig_pub_nll.savefig(buf_nll, format="png", dpi=300, bbox_inches="tight", facecolor="white")
            st.download_button("Download NLL convergence (PNG, 300 dpi)", buf_nll.getvalue(),
                               file_name="rs_nll_convergence.png", mime="image/png")
            plt.close(fig_pub_nll)

        try:
            _band_rs_pub = band_rs  # noqa: F841 – set inside tab_rs3 scope
            st.markdown("#### 5. Posterior credible band")
            fig_pub_band = pub_credible_band_figure(
                dist, C_pinn_rs[-1], _band_rs_pub,
                title="Posterior 95% credible band (Regular-solution Ω)",
            )
            st.image(_pub_fig_to_bytes(fig_pub_band), use_container_width=True)
            buf_band = io.BytesIO()
            fig_pub_band.savefig(buf_band, format="png", dpi=300, bbox_inches="tight", facecolor="white")
            st.download_button("Download credible band (PNG, 300 dpi)", buf_band.getvalue(),
                               file_name="rs_credible_band.png", mime="image/png")
            plt.close(fig_pub_band)
        except NameError:
            # C12: band_rs may be undefined if reliability tab was not executed.
            # This is expected behaviour — the figure is simply skipped.
            pass

    st.stop()

# =========================================================================
# Fickian D matrix result display (original)
# =========================================================================

if type(result).__name__ == "TrainResultRS":
    st.error("Regular-solution result detected but display section was skipped. Please re-run.")
    st.stop()

model = result.model
data = result.data
hist = result.history
span_um = float(inputs["span_um"])

x = data.x_grid
t = data.t_grid
X, T = np.meshgrid(x, t)

C_pinn = predict(model, X.reshape(-1, 1), T.reshape(-1, 1)).reshape(len(t), len(x), 3)
C_fdm = data.C_fdm
C_diff = C_pinn - C_fdm

D_pinn = model.diffusion_matrix().detach().cpu().numpy()
D_pinn_left = D_pinn
D_pinn_right = D_pinn
if hasattr(model, "diffusion_matrix_left"):
    D_pinn_left = model.diffusion_matrix_left().detach().cpu().numpy()
    D_pinn_right = model.diffusion_matrix_right().detach().cpu().numpy()
D_true = data.D_true
D_true_left = data.D_true_left if data.D_true_left is not None else D_true
D_true_right = data.D_true_right if data.D_true_right is not None else D_true

C_zero = None
D_zero = None
theta_zero = None
if bool(inputs.get("show_zero_interaction_reference", True)):
    D_source = D_pinn if inputs.get("zero_interaction_source", "PINN estimated diagonals") == "PINN estimated diagonals" else D_true
    _, C_zero, D_zero, theta_zero = compute_zero_interaction_reference(
        x_query=x,
        t_query=t,
        t_max=float(inputs["t_max"]),
        nx=len(x),
        nt_save=len(t),
        D_source=D_source,
    )

final_rmse_each = np.sqrt(np.mean((C_diff[-1] ** 2), axis=0))
all_rmse = float(np.sqrt(np.mean(C_diff ** 2)))
D_rel_err = float(np.linalg.norm(D_pinn - D_true) / max(np.linalg.norm(D_true), 1.0e-14))

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total profile RMSE", f"{all_rmse:.3e}")
m2.metric("Final RMSE Co/Ni/Ta", f"{final_rmse_each[0]:.2e}/{final_rmse_each[1]:.2e}/{final_rmse_each[2]:.2e}")
m3.metric("D matrix rel. error", f"{100 * D_rel_err:.2f}%")
m4.metric("Train time", f"{result.train_time:.1f} s")
st.caption(
    f"PINNs start time fraction={data.t_start / max(float(inputs['t_max']), 1.0e-14):.3f}; "
    f"diffusion model={inputs.get('diffusion_model_mode', 'single D')}; "
    f"phase interface width={float(inputs.get('phase_interface_width', 0.0)):.3f}"
)
if inputs.get("diffusion_model_mode") == "left/right D":
    st.markdown("#### Left/right diffusion matrices estimated by PINNs")
    lr_rows = []
    for label, Dtmp in [("left", D_pinn_left), ("right", D_pinn_right), ("average", D_pinn)]:
        lr_rows.extend([
            {"region": label, "parameter": "D_NiNi", "normalized value": Dtmp[0, 0]},
            {"region": label, "parameter": "D_NiTa", "normalized value": Dtmp[0, 1]},
            {"region": label, "parameter": "D_TaNi", "normalized value": Dtmp[1, 0]},
            {"region": label, "parameter": "D_TaTa", "normalized value": Dtmp[1, 1]},
        ])
    st.dataframe(pd.DataFrame(lr_rows), use_container_width=True)

if C_zero is not None:
    z1, z2, z3 = st.columns(3)
    zero_vs_pinn = float(np.sqrt(np.mean((C_pinn[-1] - C_zero[-1]) ** 2)))
    zero_vs_fdm = float(np.sqrt(np.mean((C_fdm[-1] - C_zero[-1]) ** 2)))
    z1.metric("Zero-interaction ref.", "enabled")
    z2.metric("RMSE PINN vs zero", f"{zero_vs_pinn:.3e}")
    z3.metric("RMSE FDM vs zero", f"{zero_vs_fdm:.3e}")

paper_time_h_for_display = float(inputs.get("paper_time_h", 160.0))
st.caption(
    f"Time conversion: normalized final time τ_max = {float(t[-1]):.4g} "
    f"corresponds to physical annealing time = {paper_time_h_for_display:.3g} h. "
    f"Displayed physical times use real t = τ / τ_max × {paper_time_h_for_display:.3g} h."
)

indices = sorted(set(np.linspace(0, len(t) - 1, int(multi_time_count)).astype(int).tolist()))

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["Profiles", "Differences", "Training", "Likelihood & bands", "Abstract validation", "CALPHAD / DICTRA", "Data & export"]
)

with tab1:
    st.markdown("### Final diffusion-couple profile")
    st.caption(
        f"最終時刻の FDM / PINNs / pseudo experimental data / zero-interaction を重ねたFig.11風プロットです。"
        f" 最終時刻は {format_time_label(float(t[-1]), float(t[-1]), paper_time_h_for_display)} です。"
    )
    st.plotly_chart(
        fig11_profile_plot(
            x,
            C_fdm[-1],
            C_pinn[-1],
            data.x_exp,
            data.c_exp,
            span_um=span_um,
            C_zero_final=None if C_zero is None else C_zero[-1],
        ),
        use_container_width=True,
    )
    st.markdown("### Predicted vs experiment diagnostics")
    if inputs.get("diffusion_model_mode") == "left/right D":
        st.warning(
            "This diffusion-couple setting uses left/right D. A single predicted-vs-experiment scatter can mix "
            "left, interface, and right regions. Use the component/region-split plots below for diagnosis."
        )

    with st.expander("Show mixed predicted-vs-experiment plot", expanded=False):
        st.plotly_chart(
            predicted_vs_experiment_plot(C_pinn[-1], x, data.x_exp, data.c_exp),
            use_container_width=True,
        )

    region_diag_width = float(inputs.get("phase_interface_width", 0.03))
    st.caption(
        f"Region split uses x < 0.5 - {region_diag_width:.3f} as left, "
        f"x > 0.5 + {region_diag_width:.3f} as right, and the middle as interface."
    )

    pv_cols = st.columns(3)
    for j, comp in enumerate(COMPONENTS):
        with pv_cols[j]:
            st.plotly_chart(
                predicted_vs_experiment_by_component_plot(
                    C_pinn[-1],
                    x,
                    data.x_exp,
                    data.c_exp,
                    component_index=j,
                    component_name=comp,
                    region_width=region_diag_width,
                ),
                use_container_width=True,
            )

    with st.expander("Residual summary by component and region", expanded=False):
        st.dataframe(
            residual_summary_by_component_region(
                C_pinn[-1],
                x,
                data.x_exp,
                data.c_exp,
                region_width=region_diag_width,
            ),
            use_container_width=True,
        )

    st.markdown("### Multi-time profiles, separated by time")
    st.caption(
        "各時刻ごとに Co / Ni / Ta の FDM と PINNs を別々の図として表示します。"
        "最終時刻の図だけ pseudo experimental data も重ねます。"
    )

    mt_cols = st.columns(int(multi_time_columns))
    for k, idx_time in enumerate(indices):
        with mt_cols[k % int(multi_time_columns)]:
            st.plotly_chart(
                single_time_profile_plot(
                    x,
                    t,
                    C_fdm,
                    C_pinn,
                    idx_time,
                    span_um=span_um,
                    x_exp=data.x_exp,
                    C_exp=data.c_exp,
                    C_zero_time=None if C_zero is None else C_zero[idx_time],
                    annealing_time_h=paper_time_h_for_display,
                ),
                use_container_width=True,
            )

    with st.expander("Show combined multi-time overview", expanded=False):
        st.plotly_chart(
            multi_time_profile_plot(x, t, C_fdm, C_pinn, indices, span_um=span_um, annealing_time_h=paper_time_h_for_display),
            use_container_width=True,
        )

with tab2:
    st.plotly_chart(final_difference_plot(x, C_diff[-1], span_um=span_um), use_container_width=True)
    if C_zero is not None:
        st.plotly_chart(
            zero_interaction_difference_plot(x, C_pinn[-1], C_zero[-1], span_um=span_um),
            use_container_width=True,
        )
    h1, h2, h3 = st.columns(3)
    h1.plotly_chart(heatmap_diff_plot(x, t, C_diff, 0, span_um=span_um), use_container_width=True)
    h2.plotly_chart(heatmap_diff_plot(x, t, C_diff, 1, span_um=span_um), use_container_width=True)
    h3.plotly_chart(heatmap_diff_plot(x, t, C_diff, 2, span_um=span_um), use_container_width=True)
with tab3:
    c1, c2 = st.columns(2)
    c1.plotly_chart(loss_plot(hist), use_container_width=True)
    c2.plotly_chart(D_history_plot(hist, D_true), use_container_width=True)

    st.markdown("### Diffusion matrix")
    st.dataframe(diffusion_matrix_table(D_true, D_pinn), use_container_width=True)

    xr, tr, R = residual_grid(model, data.t_start, float(inputs["t_max"]))
    r_ni = float(np.sqrt(np.mean(R[:, :, 0] ** 2)))
    r_ta = float(np.sqrt(np.mean(R[:, :, 1] ** 2)))
    q1, q2 = st.columns(2)
    q1.metric("PDE residual RMSE Ni", f"{r_ni:.3e}")
    q2.metric("PDE residual RMSE Ta", f"{r_ta:.3e}")

    st.markdown("### Training history")
    st.dataframe(hist, use_container_width=True)

with tab4:
    st.markdown(
        """
<div class="note">
このタブでは、FDMによる疑似実験データから逆推定したNi-Ta相互拡散行列の信頼度を評価します。<br>
<b>低コスト法</b>: PINN推定値の近傍で尤度を二次近似するLaplace近似。<br>
<b>高精度・高コスト法</b>: FDM forward modelを毎回解くMetropolis MCMC。<br>
線だけでなく、尤度からサンプルした拡散係数に基づく <b>95% credible band</b> を表示します。
</div>
""",
        unsafe_allow_html=True,
    )

    with st.expander("Pseudo-experiment noise model note", expanded=False):
        st.markdown(
            """
疑似実験点は、FDM最終プロファイルに独立Gaussianノイズを加えた後、濃度を `[0,1]` にclipし、さらに総和が1になるようrenormalizeしています。  
一方、尤度評価ではNi/Taの2成分に独立Gaussian誤差を仮定しています。

| noise | 影響 |
|---:|---|
| `≤ 0.01` | clip/renormalizeの影響は通常小さい |
| `0.03〜0.05` | reduced chi-squareやposterior widthにバイアスが入り得る |

厳密な統計モデルにする場合は、組成制約つきの尤度、例えばlog-ratio空間やDirichlet型ノイズモデルを使うのが望ましいです。
"""
        )

    render_mcmc_quick_hint(
        bool(inputs["run_high_cost_mcmc"]),
        int(inputs["mcmc_steps"]),
        int(inputs["mcmc_burn"]),
        float(inputs["mcmc_proposal"]),
    )
    render_mcmc_explanation_expander()

    if inputs.get("diffusion_model_mode") == "left/right D":
        theta_hat_lr = theta_lr_from_matrices(D_pinn_left, D_pinn_right,
                                                force_symmetric=inputs.get("force_symmetric"))
        st.markdown(f"### Left/right D dedicated {len(theta_hat_lr)}-parameter reliability")
        if len(theta_hat_lr) == 8:
            st.success(
                "Using theta_lr = [logD_NiNi_L, logD_TaTa_L, rho12_L, rho21_L, "
                "logD_NiNi_R, logD_TaTa_R, rho12_R, rho21_R] (non-symmetric D) "
                "and solving the two-region FDM for each posterior sample."
            )
        else:
            st.success(
                "Using theta_lr = [logD_NiNi_L, logD_TaTa_L, rho_L, logD_NiNi_R, logD_TaTa_R, rho_R] "
                "and solving the two-region FDM for each posterior sample."
            )
        like_sigma_eff = float(inputs["like_sigma"])
        rel_nx_eff = int(inputs["rel_nx"])
        rel_nt_eff = int(inputs["rel_nt"])
        phase_width_eff = float(inputs.get("phase_interface_width", 0.02))
        prior_mean_raw = np.asarray(inputs["prior_mean"], dtype=float)
        dim_per_region = len(theta_hat_lr) // 2
        if dim_per_region == 4 and len(prior_mean_raw) == 3:
            prior_mean_base = np.array(
                [prior_mean_raw[0], prior_mean_raw[1], prior_mean_raw[2], prior_mean_raw[2]],
                dtype=float,
            )
        elif len(prior_mean_raw) == dim_per_region:
            prior_mean_base = prior_mean_raw
        else:
            prior_mean_base = prior_mean_raw[:dim_per_region] if len(prior_mean_raw) >= dim_per_region else np.pad(prior_mean_raw, (0, dim_per_region - len(prior_mean_raw)))
        prior_mean_lr = np.concatenate([prior_mean_base, prior_mean_base])
        prior_std_eff = float(inputs["prior_std"])

        if dim_per_region == 4:
            param_names = [
                "logD_NiNi_left", "logD_TaTa_left", "rho12_raw_left", "rho21_raw_left",
                "logD_NiNi_right", "logD_TaTa_right", "rho12_raw_right", "rho21_raw_right",
            ]
        else:
            param_names = [
                "logD_NiNi_left", "logD_TaTa_left", "rho_raw_left",
                "logD_NiNi_right", "logD_TaTa_right", "rho_raw_right",
            ]
        theta_lr_df = pd.DataFrame(
            {
                "parameter": param_names,
                "theta_hat_lr": theta_hat_lr,
                "prior_mean_lr": prior_mean_lr,
            }
        )
        st.dataframe(theta_lr_df, use_container_width=True)

        with st.spinner("Low-cost left/right reliability: 6D Laplace approximation..."):
            low_rel_lr = cached_laplace_reliability_lr(
                theta_hat_lr=theta_hat_lr,
                x_exp=data.x_exp,
                c_exp=data.c_exp,
                sigma=like_sigma_eff,
                t_max=float(inputs["t_max"]),
                nx=rel_nx_eff,
                nt_save=rel_nt_eff,
                phase_width=phase_width_eff,
                prior_mean_lr=prior_mean_lr,
                prior_std_scalar=prior_std_eff,
                hessian_step=float(inputs["hessian_step"]),
                n_samples=int(inputs["laplace_samples"]),
                seed=int(inputs["seed"]),
            )

        st.markdown("#### Left/right Laplace Hessian diagnostics")
        h_min_lr = float(low_rel_lr.get("hessian_min_eig", np.array([np.nan]))[0])
        h_non_pd_lr = bool(low_rel_lr.get("hessian_non_pd", np.array([False]))[0])
        cov_clipped_lr = bool(low_rel_lr.get("covariance_was_clipped", np.array([False]))[0])
        inv_method_lr = str(low_rel_lr.get("hessian_inverse_method", np.array(["unknown"]))[0])
        lhd1, lhd2, lhd3, lhd4 = st.columns(4)
        lhd1.metric("min eig(H) 6D", f"{h_min_lr:.3e}")
        lhd2.metric("Hessian non-PD", "yes" if h_non_pd_lr else "no")
        lhd3.metric("cov eig clipped", "yes" if cov_clipped_lr else "no")
        lhd4.metric("inverse method", inv_method_lr)

        with st.expander("Left/right Hessian / covariance eigenvalues", expanded=False):
            st.dataframe(
                pd.DataFrame(
                    {
                        "index": np.arange(len(low_rel_lr.get("hessian_eigval_raw", []))),
                        "eig(H raw)": low_rel_lr.get("hessian_eigval_raw", np.array([])),
                        "eig(H regularized)": low_rel_lr.get("hessian_eigval_regularized", np.array([])),
                        "eig(cov raw)": low_rel_lr.get("cov_eigval_raw", np.array([])),
                        "eig(cov clipped)": low_rel_lr.get("cov_eigval_clipped", np.array([])),
                    }
                ),
                use_container_width=True,
            )

        if h_non_pd_lr:
            st.error(
                "6D Laplace warning: the left/right Hessian has non-positive eigenvalue(s). "
                "The 6D credible band is shown, but should be interpreted cautiously."
            )
        elif cov_clipped_lr:
            st.warning(
                "6D Laplace warning: covariance eigenvalues were clipped. "
                "Intervals may be affected by regularization."
            )

        high_rel_lr = None
        if bool(inputs["run_high_cost_mcmc"]):
            st.markdown(f"#### Left/right {len(theta_hat_lr)}D MCMC")
            mcmc_lr_progress = st.progress(0.0)
            with st.spinner("High-cost left/right MCMC: each proposal solves two-region FDM..."):
                high_rel_lr = mcmc_reliability_lr(
                    theta_start_lr=theta_hat_lr,
                    x_exp=data.x_exp,
                    c_exp=data.c_exp,
                    sigma=like_sigma_eff,
                    t_max=float(inputs["t_max"]),
                    nx=rel_nx_eff,
                    nt_save=rel_nt_eff,
                    phase_width=phase_width_eff,
                    prior_mean_lr=prior_mean_lr,
                    prior_std_scalar=prior_std_eff,
                    n_steps=int(inputs["mcmc_steps"]),
                    burn_in=int(inputs["mcmc_burn"]),
                    proposal_std=float(inputs["mcmc_proposal"]),
                    seed=int(inputs["seed"]),
                    progress_bar=mcmc_lr_progress,
                    # Fix #5: pass Laplace cov as informed proposal (LR)
                    proposal_cov=low_rel_lr.get("cov"),
                    marginalize_sigma=bool(inputs.get("marginalize_sigma", False)),
                )
            acc_lr = float(high_rel_lr["acceptance_rate"][0])
            st.metric("left/right MCMC acceptance", f"{100*acc_lr:.1f}%")
            if acc_lr < 0.10:
                st.warning("Left/right MCMC acceptance is low. Try lowering MCMC proposal std.")
            elif acc_lr > 0.60:
                st.info("Left/right MCMC acceptance is high. Try increasing MCMC proposal std.")

        st.markdown("#### Left/right posterior parameter summary")
        lr_tables = [reliability_summary_table_lr(f"Laplace left/right {len(theta_hat_lr)}D", low_rel_lr)]
        if high_rel_lr is not None and len(high_rel_lr["samples"]) > 5:
            lr_tables.append(reliability_summary_table_lr(f"MCMC left/right {len(theta_hat_lr)}D", high_rel_lr))
        st.dataframe(pd.concat(lr_tables, ignore_index=True), use_container_width=True)

        if high_rel_lr is not None and len(high_rel_lr["samples"]) > 5:
            st.plotly_chart(mcmc_trace_plot_lr(high_rel_lr["samples"]), use_container_width=True)

        st.markdown("#### Left/right credible bands on profile")
        st.caption("Each posterior sample is evaluated by the two-region FDM, not by averaged single-D FDM.")
        low_lr_band_progress = st.progress(0.0)
        low_lr_band = posterior_band_from_samples_lr(
            low_rel_lr["samples"],
            x,
            float(inputs["t_max"]),
            rel_nx_eff,
            rel_nt_eff,
            phase_width_eff,
            max_samples=int(inputs["band_samples"]),
            progress_bar=low_lr_band_progress,
        )
        st.plotly_chart(
            posterior_band_fig11_plot(
                x=x,
                x_exp=data.x_exp,
                C_exp=data.c_exp,
                C_pinn_final=C_pinn[-1],
                band=low_lr_band,
                span_um=span_um,
                title=f"Left/right {len(theta_hat_lr)}D Laplace credible band on Fig.11-style profile",
            ),
            use_container_width=True,
        )

        if high_rel_lr is not None and len(high_rel_lr["samples"]) > 5:
            high_lr_band_progress = st.progress(0.0)
            high_lr_band = posterior_band_from_samples_lr(
                high_rel_lr["samples"],
                x,
                float(inputs["t_max"]),
                rel_nx_eff,
                rel_nt_eff,
                phase_width_eff,
                max_samples=int(inputs["band_samples"]),
                progress_bar=high_lr_band_progress,
            )
            st.plotly_chart(
                posterior_band_fig11_plot(
                    x=x,
                    x_exp=data.x_exp,
                    C_exp=data.c_exp,
                    C_pinn_final=C_pinn[-1],
                    band=high_lr_band,
                    span_um=span_um,
                    title=f"Left/right {len(theta_hat_lr)}D MCMC credible band on Fig.11-style profile",
                ),
                use_container_width=True,
            )

        st.info(
            "For left/right D mode, the legacy averaged-D 3-parameter reliability, contour, and band are skipped "
            "to avoid mixing physical models."
        )

    else:
        theta_hat = theta_from_D_matrix(D_pinn, force_symmetric=inputs.get("force_symmetric"))
        like_sigma_eff = float(inputs["like_sigma"])
        rel_nx_eff = int(inputs["rel_nx"])
        rel_nt_eff = int(inputs["rel_nt"])
        prior_mean_raw = np.asarray(inputs["prior_mean"], dtype=float)
        if len(theta_hat) == 4 and len(prior_mean_raw) == 3:
            prior_mean = np.array(
                [prior_mean_raw[0], prior_mean_raw[1], prior_mean_raw[2], prior_mean_raw[2]],
                dtype=float,
            )
        elif len(prior_mean_raw) == len(theta_hat):
            prior_mean = prior_mean_raw
        else:
            prior_mean = prior_mean_raw[:len(theta_hat)] if len(prior_mean_raw) >= len(theta_hat) else np.pad(prior_mean_raw, (0, len(theta_hat) - len(prior_mean_raw)))
        prior_std_eff = float(inputs["prior_std"])

        with st.spinner("Low-cost reliability: Laplace approximation from likelihood curvature..."):
            low_rel = cached_laplace_reliability(
                theta_hat=theta_hat,
                x_exp=data.x_exp,
                c_exp=data.c_exp,
                sigma=like_sigma_eff,
                t_max=float(inputs["t_max"]),
                nx=rel_nx_eff,
                nt_save=rel_nt_eff,
                prior_mean=prior_mean,
                prior_std_scalar=prior_std_eff,
                hessian_step=float(inputs["hessian_step"]),
                n_samples=int(inputs["laplace_samples"]),
                seed=int(inputs["seed"]),
                # P4 fix: pass multitime data for full-time NLL evaluation
                x_exp_all=data.x_exp_all,
                t_exp_all=data.t_exp_all,
                c_exp_all=data.c_exp_all,
            )

        st.markdown("### Laplace Hessian diagnostics")
        h_min = float(low_rel.get("hessian_min_eig", np.array([np.nan]))[0])
        h_non_pd = bool(low_rel.get("hessian_non_pd", np.array([False]))[0])
        cov_clipped = bool(low_rel.get("covariance_was_clipped", np.array([False]))[0])
        inv_method = str(low_rel.get("hessian_inverse_method", np.array(["unknown"]))[0])

        hd1, hd2, hd3, hd4 = st.columns(4)
        hd1.metric("min eig(H)", f"{h_min:.3e}")
        hd2.metric("Hessian non-PD", "yes" if h_non_pd else "no")
        hd3.metric("cov eig clipped", "yes" if cov_clipped else "no")
        hd4.metric("inverse method", inv_method)

        with st.expander("Hessian / covariance eigenvalues", expanded=False):
            hdiag_df = pd.DataFrame(
                {
                    "index": np.arange(len(low_rel.get("hessian_eigval_raw", []))),
                    "eig(H raw)": low_rel.get("hessian_eigval_raw", np.array([])),
                    "eig(H regularized)": low_rel.get("hessian_eigval_regularized", np.array([])),
                    "eig(cov raw)": low_rel.get("cov_eigval_raw", np.array([])),
                    "eig(cov clipped)": low_rel.get("cov_eigval_clipped", np.array([])),
                }
            )
            st.dataframe(hdiag_df, use_container_width=True)

        if h_non_pd:
            st.error(
                "Laplace warning: the raw Hessian of the negative log posterior has non-positive eigenvalue(s). "
                "The PINNs estimate may be near a saddle/flat direction rather than a strict local minimum. "
                "The covariance shown below was numerically repaired and should be interpreted cautiously."
            )
        elif cov_clipped:
            st.warning(
                "Laplace warning: covariance eigenvalues were clipped for numerical stability. "
                "Credible intervals may be dominated by regularization rather than likelihood curvature."
            )

        # PSIS diagnostic for Laplace quality
        if "cov" in low_rel and "theta_hat" in low_rel:
            with st.spinner("Computing PSIS diagnostic for Laplace quality..."):
                psis_result = psis_diagnostic(
                    laplace_samples=low_rel["samples"],
                    x_exp=data.x_exp,
                    c_exp=data.c_exp,
                    sigma=like_sigma_eff,
                    t_max=float(inputs["t_max"]),
                    nx=rel_nx_eff,
                    nt_save=rel_nt_eff,
                    prior_mean=prior_mean,
                    prior_std_scalar=prior_std_eff,
                    theta_hat=low_rel["theta_hat"],
                    cov=low_rel["cov"],
                    max_eval=min(200, int(inputs["laplace_samples"])),
                    # P4 fix: pass multitime data
                    x_exp_all=data.x_exp_all,
                    t_exp_all=data.t_exp_all,
                    c_exp_all=data.c_exp_all,
                )
            pk = psis_result["pareto_k"]
            psis_ess = psis_result["ess_psis"]
            pk1, pk2, pk3 = st.columns(3)
            pk1.metric("Pareto-k̂", f"{pk:.2f}")
            pk2.metric("PSIS ESS", f"{psis_ess:.0f}")
            pk3.metric("PSIS evals", f"{psis_result['n_evaluated']}")
            if pk > 0.7:
                st.error(
                    f"Pareto-k̂ = {pk:.2f} > 0.7: Laplace approximation is **unreliable**. "
                    "The posterior likely has heavy tails or multimodality. "
                    "Use MCMC results instead."
                )
            elif pk > 0.5:
                st.warning(
                    f"Pareto-k̂ = {pk:.2f} ∈ (0.5, 0.7]: Laplace approximation is **marginal**. "
                    "Consider running MCMC for confirmation."
                )
            else:
                st.success(f"Pareto-k̂ = {pk:.2f} ≤ 0.5: Laplace approximation is adequate.")

        high_rel = None
        if bool(inputs["run_high_cost_mcmc"]):
            st.warning("High-cost MCMC repeatedly solves FDM and can take a while.")
            mcmc_progress = st.progress(0.0)
            with st.spinner("High-cost reliability: FDM-based MCMC posterior sampling..."):
                high_rel = mcmc_reliability(
                    theta_start=theta_hat,
                    x_exp=data.x_exp,
                    c_exp=data.c_exp,
                    sigma=like_sigma_eff,
                    t_max=float(inputs["t_max"]),
                    nx=rel_nx_eff,
                    nt_save=rel_nt_eff,
                    prior_mean=prior_mean,
                    prior_std_scalar=prior_std_eff,
                    n_steps=int(inputs["mcmc_steps"]),
                    burn_in=int(inputs["mcmc_burn"]),
                    proposal_std=float(inputs["mcmc_proposal"]),
                    seed=int(inputs["seed"]),
                    progress_bar=mcmc_progress,
                    # Fix #5: pass Laplace cov as informed proposal
                    proposal_cov=low_rel.get("cov"),
                    marginalize_sigma=bool(inputs.get("marginalize_sigma", False)),
                    # P4 fix: pass multitime data for full-time NLL evaluation
                    x_exp_all=data.x_exp_all,
                    t_exp_all=data.t_exp_all,
                    c_exp_all=data.c_exp_all,
                )

        chi2_low = float(low_rel["chi2"][0])
        red_low = float(low_rel["reduced_chi2"][0])

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Likelihood sigma", f"{like_sigma_eff:.3f}")
        r2.metric("Chi-square", f"{chi2_low:.2f}")
        r3.metric("Reduced chi-square", f"{red_low:.2f}")
        if high_rel is not None:
            acc_rate = float(high_rel["acceptance_rate"][0])
            r4.metric("MCMC acceptance", f"{100 * acc_rate:.1f}%")
            if acc_rate < 0.10:
                st.warning("MCMC acceptance rate is low. The proposal step may be too large; try lowering `MCMC proposal std`.")
            elif acc_rate > 0.60:
                st.info("MCMC acceptance rate is high. The proposal step may be too small; samples may move slowly. Consider increasing `MCMC proposal std`.")
            else:
                st.success("MCMC acceptance rate is in a practical range for a first diagnostic.")
            if "sigma_median" in high_rel:
                sig_med = float(high_rel["sigma_median"][0])
                sig_samps = high_rel["sigma_samples"]
                st.metric("σ posterior median", f"{sig_med:.4f}")
                st.caption(
                    f"σ 95% CI: [{np.percentile(sig_samps, 2.5):.4f}, {np.percentile(sig_samps, 97.5):.4f}]"
                )
        else:
            r4.metric("MCMC", "not run")

        low_table = reliability_summary_table("Laplace low-cost", low_rel)
        if high_rel is not None:
            high_table = reliability_summary_table("MCMC high-cost", high_rel)
            rel_table = pd.concat([low_table, high_table], ignore_index=True)
        else:
            rel_table = low_table

        st.markdown("### Posterior interval of inferred interaction coefficients")
        st.dataframe(rel_table, use_container_width=True)

        st.plotly_chart(
            posterior_parameter_plot(low_rel["samples"], None if high_rel is None else high_rel["samples"]),
            use_container_width=True,
        )

        if high_rel is not None and len(high_rel["samples"]) > 5:
            st.markdown("### MCMC trace diagnostic")
            st.caption("保存されたMCMCサンプル列です。大きなドリフトが残っている場合は、burn-in不足や混合不良の可能性があります。")
            st.plotly_chart(mcmc_trace_plot(high_rel["samples"]), use_container_width=True)

        st.markdown("### Likelihood/posterior credible bands on profile")
        st.caption(
            "各サンプルの拡散行列でFDMを解き、最終プロファイルの2.5%, 50%, 97.5%分位をバンドとして表示します。"
        )

        low_band_progress = st.progress(0.0)
        low_band = posterior_band_from_samples(
            low_rel["samples"],
            x,
            float(inputs["t_max"]),
            rel_nx_eff,
            rel_nt_eff,
            max_samples=int(inputs["band_samples"]),
            progress_bar=low_band_progress,
        )
        st.plotly_chart(
            posterior_band_fig11_plot(
                x=x,
                x_exp=data.x_exp,
                C_exp=data.c_exp,
                C_pinn_final=C_pinn[-1],
                band=low_band,
                span_um=span_um,
                title="Low-cost Laplace credible band on Fig.11-style profile",
            ),
            use_container_width=True,
        )

        if high_rel is not None and len(high_rel["samples"]) > 5:
            high_band_progress = st.progress(0.0)
            high_band = posterior_band_from_samples(
                high_rel["samples"],
                x,
                float(inputs["t_max"]),
                rel_nx_eff,
                rel_nt_eff,
                max_samples=int(inputs["band_samples"]),
                progress_bar=high_band_progress,
            )
            st.plotly_chart(
                posterior_band_fig11_plot(
                    x=x,
                    x_exp=data.x_exp,
                    C_exp=data.c_exp,
                    C_pinn_final=C_pinn[-1],
                    band=high_band,
                    span_um=span_um,
                    title="High-cost MCMC credible band on Fig.11-style profile",
                ),
                use_container_width=True,
            )

        st.markdown("### Likelihood / posterior contour")
        pair = inputs["contour_axis_pair"]
        if pair == "logD_NiNi vs logD_TaTa":
            axis_i, axis_j = 0, 1
            xl, yl = "log D_NiNi", "log D_TaTa"
        elif pair == "logD_NiNi vs rho_raw":
            axis_i, axis_j = 0, 2
            xl, yl = "log D_NiNi", "rho_raw"
        else:
            axis_i, axis_j = 1, 2
            xl, yl = "log D_TaTa", "rho_raw"

        contour_evals = int(inputs["contour_n"]) ** 2
        st.caption(
            f"Contour grid cost: {int(inputs['contour_n'])} × {int(inputs['contour_n'])} = "
            f"{contour_evals:,} FDM evaluations."
        )
        if contour_evals >= 400:
            st.warning("Large contour grid: this may take a long time because each grid point solves an FDM forward problem.")

        contour_progress = st.progress(0.0)
        with st.spinner("Computing likelihood/posterior contour grid..."):
            gx, gy, gz = likelihood_contour_grid(
                theta_center=theta_hat,
                x_exp=data.x_exp,
                c_exp=data.c_exp,
                sigma=like_sigma_eff,
                t_max=float(inputs["t_max"]),
                nx=rel_nx_eff,
                nt_save=rel_nt_eff,
                axis_i=axis_i,
                axis_j=axis_j,
                half_width=float(inputs["contour_half_width"]),
                n_grid=int(inputs["contour_n"]),
                prior_mean=prior_mean,
                prior_std_scalar=prior_std_eff,
                progress_bar=contour_progress,
            )
        st.plotly_chart(
            likelihood_contour_plot(gx, gy, gz, xl, yl, theta_hat, axis_i, axis_j),
            use_container_width=True,
        )

        st.markdown(
            """
    <div class="note">
    Reduced chi-square が 1 に近い場合、仮定した測定ノイズとモデルのズレが概ね整合的です。
    かなり大きい場合は、ノイズを過小評価している、または定数拡散行列モデルが実験点を説明しきれていない可能性があります。
    Laplace近似は高速ですが局所的・ガウス近似です。MCMCは高コストですが、非対称・非ガウスな事後分布をある程度表現できます。
    </div>
    """,
            unsafe_allow_html=True,
        )



with tab5:
    st.markdown("## Abstract validation dashboard")
    st.markdown(
        """
<div class="note">
This tab is designed for preparing the TOFA abstract. It summarizes whether the synthetic diffusion-couple inverse problem is behaving as expected: recovery of known left/right D, improvement over zero-interaction, and component/region residuals.
</div>
""",
        unsafe_allow_html=True,
    )

    region_width_abs = float(inputs.get("phase_interface_width", 0.03))
    abstract_metrics = abstract_validation_metrics(
        x_grid=x,
        C_pinn_final=C_pinn[-1],
        C_fdm_final=C_fdm[-1],
        x_exp=data.x_exp,
        C_exp=data.c_exp,
        D_true=D_true,
        D_true_left=D_true_left,
        D_true_right=D_true_right,
        D_pinn=D_pinn,
        D_pinn_left=D_pinn_left,
        D_pinn_right=D_pinn_right,
        C_zero_final=None if C_zero is None else C_zero[-1],
        region_width=region_width_abs,
    )

    st.markdown("### One-paragraph abstract-ready validation summary")
    summary_text = abstract_validation_summary_text(abstract_metrics)
    if bool(inputs.get("fixed_diagonal_abstract_default", False)):
        summary_text = (
            summary_text
            + " In this validation setting, the diagonal terms were fixed using known main-diffusion information, "
            + "so the inverse problem focuses on effective cross-interdiffusion terms."
        )
    st.info(summary_text)

    st.markdown("### Known true D vs PINNs-estimated D")
    st.caption("For synthetic validation, true D is known from the FDM teacher. This table is not used for real experimental data.")
    st.dataframe(abstract_metrics["D_compare"], use_container_width=True)

    st.markdown("### Profile error and improvement over zero-interaction")
    st.plotly_chart(abstract_validation_plot(abstract_metrics), use_container_width=True)
    st.dataframe(abstract_metrics["rmse_improvement"], use_container_width=True)

    with st.expander("Component/region residual summary", expanded=False):
        st.dataframe(abstract_metrics["profile_rmse"], use_container_width=True)

    st.download_button(
        "Download abstract D comparison CSV",
        abstract_metrics["D_compare"].to_csv(index=False).encode("utf-8"),
        "tofa_abstract_validation_D_compare.csv",
        "text/csv",
    )
    st.download_button(
        "Download abstract RMSE summary CSV",
        abstract_metrics["profile_rmse"].to_csv(index=False).encode("utf-8"),
        "tofa_abstract_validation_profile_rmse.csv",
        "text/csv",
    )

    st.markdown("### Multi-time pseudo-exp validation")
    if str(inputs.get("pseudo_exp_time_mode", "final only")) == "multi-time":
        mt_rmse_df = multi_time_pseudo_exp_rmse_table(
            model,
            data.x_exp_all,
            data.t_exp_all,
            data.c_exp_all,
        )
        st.plotly_chart(
            multi_time_pseudo_exp_rmse_plot(mt_rmse_df, paper_time_h_for_display, float(t[-1])),
            use_container_width=True,
        )
        st.dataframe(mt_rmse_df, use_container_width=True)
        st.info(
            "Multi-time pseudo-exp is active. This gives the inverse problem time-evolution constraints, "
            "which is usually more informative for cross-interdiffusion terms than a final-profile-only validation."
        )
        st.download_button(
            "Download multi-time pseudo-exp RMSE CSV",
            mt_rmse_df.to_csv(index=False).encode("utf-8"),
            "tofa_multitime_pseudo_exp_rmse.csv",
            "text/csv",
        )
    else:
        st.warning(
            "Pseudo-exp time mode is final only. For TOFA abstract validation, multi-time pseudo-exp is recommended "
            "because it improves identifiability of cross-interdiffusion terms."
        )

    st.markdown("### Suggested TOFA abstract claim")
    st.code(
        summary_text
        + " The present implementation treats Co as the dependent component and Ni/Ta as independent components, "
        + "and the inferred quantities are effective interdiffusion coefficients rather than direct CALPHAD mobility parameters.",
        language="text",
    )


with tab6:
    st.markdown("## CALPHAD / DICTRA basis comparison")
    st.markdown(
        """
<div class="note">
PINNsで推定した相互作用項が妥当かどうかを、論文/CALPHAD/DICTRAと同じ基準で比較するためのタブです。<br>
推奨基準は <b>FCC γ相, 1200 °C, 160 h, mole fraction, Co従属, Ni/Ta独立, volume-fixed interdiffusion coefficient</b> です。
</div>
""",
        unsafe_allow_html=True,
    )

    paper_T_C_eff = float(inputs.get("paper_T_C", 1200.0))
    paper_time_h_eff = float(inputs.get("paper_time_h", 160.0))
    paper_length_um_eff = float(inputs.get("paper_length_um", span_um))

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("basis T", f"{paper_T_C_eff:.0f} °C")
    b2.metric("basis time", f"{paper_time_h_eff:.1f} h")
    b3.metric("length scale L", f"{paper_length_um_eff:.1f} µm")
    b4.metric("dependent", "Co; independent Ni/Ta")

    with st.expander("基準合わせの注意", expanded=True):
        st.markdown(
            """
| 項目 | 合わせるべき基準 |
|---|---|
| dependent component | `Co` |
| independent components | `Ni`, `Ta` |
| frame | `volume-fixed interdiffusion coefficient` を推奨 |
| unit | `m²/s` |
| composition | mole fraction |
| distance origin | 論文比較では Matano interface 基準が望ましい |
| phase | FCC γ相 |

現在のPINNs内部Dは無次元です。このタブでは次の変換で物理単位へ戻して比較します。

```text
D_physical = D_normalized * L_scale^2 / t_scale
```
"""
        )

    st.markdown("### Self-diffusion diagonal anchors used in PINNs")
    self_info = {
        "use_self_diffusion": inputs.get("use_self_diffusion", False),
        "mode": inputs.get("diag_constraint_mode", "free"),
        "D_NiNi_self [m2/s]": inputs.get("self_D_Ni_phys", 0.0),
        "D_TaTa_self [m2/s]": inputs.get("self_D_Ta_phys", 0.0),
        "prior weight": inputs.get("diag_prior_weight", 0.0),
    }
    st.dataframe(pd.DataFrame([self_info]), use_container_width=True)
    if (
        inputs.get("diffusion_model_mode") == "left/right D"
        and inputs.get("diag_constraint_mode") == "fix diagonal terms"
        and inputs.get("self_D_norm_lr") is None
    ):
        st.warning(
            "Two-region + fix diagonal terms is active, but left/right self-diffusion anchors were not fully provided. "
            "To fix both regions, enter all four left/right diagonal values."
        )

    if inputs.get("self_D_norm") is not None:
        st.caption("Self-diffusion values converted to normalized units used by PINNs:")
        st.dataframe(
            pd.DataFrame(
                {
                    "parameter": ["D_NiNi", "D_TaTa"],
                    "self normalized": [inputs["self_D_norm"][0, 0], inputs["self_D_norm"][1, 1]],
                    "self log normalized": [inputs["self_log_diag"][0], inputs["self_log_diag"][1]],
                }
            ),
            use_container_width=True,
        )
    else:
        st.info("Self-diffusion anchors were not used in this run. Enable them in the sidebar before running if values are available.")

    st.markdown("### PINNs diffusion matrix in physical units")
    D_pinn_phys = normalized_to_physical_D(D_pinn, paper_length_um_eff, paper_time_h_eff)
    D_true_phys = normalized_to_physical_D(D_true, paper_length_um_eff, paper_time_h_eff)
    st.dataframe(
        matrix_rows_for_comparison(
            D_pinn_norm=D_pinn,
            D_true_norm=D_true,
            D_zero_norm=D_zero,
            D_calphad_phys=None,
            length_um=paper_length_um_eff,
            time_h=paper_time_h_eff,
        ),
        use_container_width=True,
    )
    if inputs.get("diffusion_model_mode") == "left/right D":
        st.caption("Left/right region matrices in physical units:")
        lr_phys_rows = []
        for label, Dtmp in [("left", D_pinn_left), ("right", D_pinn_right), ("average", D_pinn)]:
            Dtmp_phys = normalized_to_physical_D(Dtmp, paper_length_um_eff, paper_time_h_eff)
            lr_phys_rows.extend([
                {"region": label, "parameter": "D_NiNi", "physical [m2/s]": Dtmp_phys[0, 0]},
                {"region": label, "parameter": "D_NiTa", "physical [m2/s]": Dtmp_phys[0, 1]},
                {"region": label, "parameter": "D_TaNi", "physical [m2/s]": Dtmp_phys[1, 0]},
                {"region": label, "parameter": "D_TaTa", "physical [m2/s]": Dtmp_phys[1, 1]},
            ])
        st.dataframe(pd.DataFrame(lr_phys_rows), use_container_width=True)

    st.markdown("### Upload CALPHAD / DICTRA interdiffusion matrix CSV")
    st.caption("Required columns: D_NiNi, D_NiTa, D_TaNi, D_TaTa. Optional: T_C, x_Co, x_Ni, x_Ta, frame, dependent.")
    calphad_matrix_file = st.file_uploader("CALPHAD/DICTRA D matrix CSV", type=["csv"], key="calphad_matrix_csv")

    calphad_df = None
    D_calphad_phys = None
    if calphad_matrix_file is not None:
        calphad_df = pd.read_csv(calphad_matrix_file)
        ok, msg = validate_columns(calphad_df, calphad_required_columns("matrix"))
        if not ok:
            st.error(msg)
        else:
            if "dependent" in calphad_df.columns:
                bad_dep = sorted(set(str(v) for v in calphad_df["dependent"].dropna().unique() if str(v).lower() != "co"))
                if bad_dep:
                    st.warning(f"dependent column contains values other than Co: {bad_dep}. Direct comparison may be invalid.")
            if "frame" in calphad_df.columns:
                frames = sorted(set(str(v) for v in calphad_df["frame"].dropna().unique()))
                st.caption("frame values in CSV: " + ", ".join(frames))
            D_calphad_phys = representative_calphad_matrix(calphad_df)
            compare_df = matrix_rows_for_comparison(
                D_pinn_norm=D_pinn,
                D_true_norm=D_true,
                D_zero_norm=D_zero,
                D_calphad_phys=D_calphad_phys,
                length_um=paper_length_um_eff,
                time_h=paper_time_h_eff,
            )
            st.markdown("#### Representative comparison table")
            st.dataframe(compare_df, use_container_width=True)
            st.plotly_chart(D_matrix_bar_plot(compare_df), use_container_width=True)
            if len(calphad_df) > 1:
                st.plotly_chart(calphad_D_composition_plot(calphad_df, D_pinn_phys), use_container_width=True)

            st.download_button(
                "Download PINNs vs CALPHAD D comparison CSV",
                compare_df.to_csv(index=False).encode("utf-8"),
                "pinns_vs_calphad_D_matrix_comparison.csv",
                "text/csv",
            )

            st.markdown("#### Interpretation")
            for _, row in compare_df.iterrows():
                if row["parameter"] in ["D_NiTa", "D_TaNi"] and "CALPHAD/DICTRA physical [m2/s]" in compare_df.columns:
                    st.write(
                        f"- **{row['parameter']}**: sign agreement = `{row.get('sign agreement', 'n/a')}`, "
                        f"relative error = `{row.get('relative error vs CALPHAD [%]', np.nan):.2f}%`."
                    )

    st.markdown("### Optional thermodynamic factor CSV")
    st.caption("If Phi is available, upload Phi_NiNi, Phi_NiTa, Phi_TaNi, Phi_TaTa to estimate M_eff = D_pinn Phi^-1.")
    phi_file = st.file_uploader("Thermodynamic factor Phi CSV", type=["csv"], key="phi_csv")
    if phi_file is not None:
        phi_df = pd.read_csv(phi_file)
        required_phi = ["Phi_NiNi", "Phi_NiTa", "Phi_TaNi", "Phi_TaTa"]
        ok, msg = validate_columns(phi_df, required_phi)
        if not ok:
            st.error(msg)
        else:
            Phi = np.array(
                [
                    [float(np.nanmedian(phi_df["Phi_NiNi"])), float(np.nanmedian(phi_df["Phi_NiTa"]))],
                    [float(np.nanmedian(phi_df["Phi_TaNi"])), float(np.nanmedian(phi_df["Phi_TaTa"]))],
                ],
                dtype=float,
            )
            M_eff = mobility_from_D_and_phi(D_pinn_phys, Phi)
            st.warning("M_eff = D Phi^-1 is a diagnostic effective mobility-like matrix, not a direct DICTRA database parameter.")
            st.dataframe(
                pd.DataFrame(
                    [
                        {"parameter": "M_eff_NiNi", "value": M_eff[0, 0]},
                        {"parameter": "M_eff_NiTa", "value": M_eff[0, 1]},
                        {"parameter": "M_eff_TaNi", "value": M_eff[1, 0]},
                        {"parameter": "M_eff_TaTa", "value": M_eff[1, 1]},
                    ]
                ),
                use_container_width=True,
            )

    st.markdown("### Upload DICTRA / CALPHAD profile CSV")
    st.caption("Required columns: distance_um, Co, Ni, Ta. This is used for profile-level validation.")
    dictra_profile_file = st.file_uploader("DICTRA/CALPHAD profile CSV", type=["csv"], key="dictra_profile_csv")
    if dictra_profile_file is not None:
        prof_df = pd.read_csv(dictra_profile_file)
        ok, msg = validate_columns(prof_df, calphad_required_columns("profile"))
        if not ok:
            st.error(msg)
        else:
            st.plotly_chart(
                dictra_profile_overlay_plot(
                    profile_df=prof_df,
                    x=x,
                    C_pinn_final=C_pinn[-1],
                    x_exp=data.x_exp,
                    C_exp=data.c_exp,
                    C_zero_final=None if C_zero is None else C_zero[-1],
                    span_um=span_um,
                ),
                use_container_width=True,
            )


with tab6:
    st.markdown(
        """
<div class="note">
このデモでは実際のFig.11のraw experimental dataは使っていません。
FDM最終プロファイルにノイズを加えた pseudo experimental data を open symbols として表示しています。
実論文の完全再現には、実測プロファイル、熱力学DB、mobility DB、DICTRA相当の計算が必要です。
</div>
""",
        unsafe_allow_html=True,
    )

    summary = pd.DataFrame(
        {
            "quantity": [
                "D_NiNi_true",
                "D_NiTa_true",
                "D_TaNi_true",
                "D_TaTa_true",
                "D_NiNi_PINN",
                "D_NiTa_PINN",
                "D_TaNi_PINN",
                "D_TaTa_PINN",
                "total_profile_RMSE",
                "D_matrix_relative_error",
                "train_time_s",
            ],
            "value": [
                D_true[0, 0],
                D_true[0, 1],
                D_true[1, 0],
                D_true[1, 1],
                D_pinn[0, 0],
                D_pinn[0, 1],
                D_pinn[1, 0],
                D_pinn[1, 1],
                all_rmse,
                D_rel_err,
                result.train_time,
            ],
        }
    )
    st.download_button(
        "Download summary CSV",
        summary.to_csv(index=False).encode("utf-8"),
        "fig11_summary.csv",
        "text/csv",
    )

    dist = distance_um_from_x(x, span_um)
    profile_df = pd.DataFrame(
        {
            "distance_um": dist,
            "FDM_Co": C_fdm[-1, :, 0],
            "FDM_Ni": C_fdm[-1, :, 1],
            "FDM_Ta": C_fdm[-1, :, 2],
            "PINN_Co": C_pinn[-1, :, 0],
            "PINN_Ni": C_pinn[-1, :, 1],
            "PINN_Ta": C_pinn[-1, :, 2],
            "DIFF_Co": C_diff[-1, :, 0],
            "DIFF_Ni": C_diff[-1, :, 1],
            "DIFF_Ta": C_diff[-1, :, 2],
        }
    )
    st.download_button(
        "Download final profiles CSV",
        profile_df.to_csv(index=False).encode("utf-8"),
        "fig11_profiles.csv",
        "text/csv",
    )

    exp_df = pd.DataFrame(
        {
            "distance_um": distance_um_from_x(data.x_exp, span_um),
            "Exp_Co": data.c_exp[:, 0],
            "Exp_Ni": data.c_exp[:, 1],
            "Exp_Ta": data.c_exp[:, 2],
        }
    )
    st.download_button(
        "Download pseudo experimental data CSV",
        exp_df.to_csv(index=False).encode("utf-8"),
        "fig11_pseudo_experimental.csv",
        "text/csv",
    )

    D_table = diffusion_matrix_table(D_true, D_pinn)
    physical_D_table = matrix_rows_for_comparison(
        D_pinn_norm=D_pinn,
        D_true_norm=D_true,
        D_zero_norm=D_zero,
        D_calphad_phys=None,
        length_um=float(inputs.get("paper_length_um", span_um)),
        time_h=float(inputs.get("paper_time_h", 160.0)),
    )
    if D_zero is not None:
        zero_rows = pd.DataFrame(
            {
                "parameter": ["D_NiNi", "D_NiTa", "D_TaNi", "D_TaTa"],
                "zero_interaction_reference": [D_zero[0, 0], D_zero[0, 1], D_zero[1, 0], D_zero[1, 1]],
            }
        )
        D_table = D_table.merge(zero_rows, on="parameter", how="left")

    st.download_button(
        "Download diffusion matrix CSV",
        D_table.to_csv(index=False).encode("utf-8"),
        "fig11_diffusion_matrix.csv",
        "text/csv",
    )
    st.download_button(
        "Download physical diffusion matrix CSV",
        physical_D_table.to_csv(index=False).encode("utf-8"),
        "fig11_diffusion_matrix_physical_units.csv",
        "text/csv",
    )

    if C_zero is not None:
        zero_df = pd.DataFrame(
            {
                "distance_um": distance_um_from_x(x, span_um),
                "ZeroInteraction_Co": C_zero[-1, :, 0],
                "ZeroInteraction_Ni": C_zero[-1, :, 1],
                "ZeroInteraction_Ta": C_zero[-1, :, 2],
                "PINN_minus_Zero_Co": C_pinn[-1, :, 0] - C_zero[-1, :, 0],
                "PINN_minus_Zero_Ni": C_pinn[-1, :, 1] - C_zero[-1, :, 1],
                "PINN_minus_Zero_Ta": C_pinn[-1, :, 2] - C_zero[-1, :, 2],
            }
        )
        st.download_button(
            "Download zero-interaction reference CSV",
            zero_df.to_csv(index=False).encode("utf-8"),
            "fig11_zero_interaction_reference.csv",
            "text/csv",
        )
