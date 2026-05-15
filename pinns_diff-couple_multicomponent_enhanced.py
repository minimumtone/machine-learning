#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multicomponent regular-solution diffusion-couple PINN.

This file is an enhanced, multicomponent successor of pinns_diff-couple.py.
It keeps the same thermodynamic concept used in the original script:

    binary: mu_A - mu_B = RT ln(c_A/c_B) + Omega_AB (1 - 2 c_A)

and extends it to an N-component regular-solution model by using a symmetric
pair-interaction matrix Omega.  The dependent component is the last component;
PINN residuals are imposed on the first N-1 independent components.

Main additions inspired by fig11_co_ni_ta_pinn_reliability_v23_multitime_pseudoexp.py:
    - multicomponent diffusion-couple pseudo-experimental data
    - multi-time pseudo-experimental sampling
    - trainable pair interaction terms Omega_ij, optionally left/right
    - loss-weight control, cosine LR schedule, detailed history
    - residual/error heatmaps and profile diagnostics
    - likelihood-based Laplace reliability and optional MCMC for Omega terms
    - posterior credible bands from FDM forward solves

Run examples:
    python pinns_diff-couple_multicomponent_enhanced.py --quick
    python pinns_diff-couple_multicomponent_enhanced.py --epochs 10000 --learn-left-right-omega --run-mcmc
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import subprocess
from datetime import datetime

# -----------------------------------------------------------------------------
# OpenMP runtime workaround for Windows / Anaconda / Streamlit environments
# -----------------------------------------------------------------------------
# Put these BEFORE importing numpy, scipy, torch, sklearn, etc.
# Some Windows environments load multiple Intel OpenMP runtimes through
# combinations of PyTorch, NumPy, MKL, SciPy, matplotlib, or Streamlit.
# The clean solution is a consistent fresh environment, but this keeps the
# educational prototype runnable on typical Anaconda installations.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import random
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Reproducibility and device helpers
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


DTYPE = torch.float32


# -----------------------------------------------------------------------------
# Multicomponent regular-solution thermodynamics
# -----------------------------------------------------------------------------

def pair_indices(n_components: int) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(n_components) for j in range(i + 1, n_components)]


def omega_matrix_from_pairs_np(theta: Sequence[float], n_components: int) -> np.ndarray:
    """Build a symmetric Omega matrix with zero diagonal from pair parameters."""
    theta = np.asarray(theta, dtype=float).reshape(-1)
    pairs = pair_indices(n_components)
    if theta.size != len(pairs):
        raise ValueError(f"Expected {len(pairs)} Omega pair values, got {theta.size}.")
    Omega = np.zeros((n_components, n_components), dtype=float)
    for val, (i, j) in zip(theta, pairs):
        Omega[i, j] = Omega[j, i] = float(val)
    return Omega


def omega_matrix_from_pairs_torch(theta: torch.Tensor, n_components: int) -> torch.Tensor:
    """Torch version of omega_matrix_from_pairs_np."""
    theta = theta.reshape(-1)
    pairs = pair_indices(n_components)
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
    return (1.0 - s) * np.asarray(theta_left, dtype=float).reshape(1, -1) + s * np.asarray(theta_right, dtype=float).reshape(1, -1)


def blend_pairs_torch(theta_left: torch.Tensor, theta_right: torch.Tensor, x: torch.Tensor,
                      x_interface: float = 0.5, width: float = 0.02) -> torch.Tensor:
    """Torch left/right Omega blending.  Returns [N, n_pairs]."""
    w = max(float(width), 1.0e-12)
    s = 0.5 * (1.0 + torch.tanh((x.reshape(-1, 1) - float(x_interface)) / w))
    return (1.0 - s) * theta_left.reshape(1, -1) + s * theta_right.reshape(1, -1)


def complete_composition_np(c_ind: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    """Append dependent component c_dep = 1 - sum(c_ind)."""
    c_ind = np.asarray(c_ind, dtype=float)
    c_dep = 1.0 - np.sum(c_ind, axis=1, keepdims=True)
    c_full = np.concatenate([c_ind, c_dep], axis=1)
    c_full = np.clip(c_full, eps, 1.0)
    c_full = c_full / np.maximum(np.sum(c_full, axis=1, keepdims=True), eps)
    return c_full


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
    """Return diffusion potentials mu_i - mu_ref for i=0..N-2.

    Regular solution free-energy concept:
        g = RT sum_a c_a ln c_a + 1/2 sum_ab Omega_ab c_a c_b

    With the last component as dependent reference r, the diffusion potential is:
        mu_i - mu_r = RT ln(c_i/c_r) + sum_b (Omega_ib - Omega_rb) c_b

    For a binary A/B system this exactly reduces to the original formula:
        RT ln(c_A/c_B) + Omega_AB (1 - 2 c_A)
    """
    c = np.clip(np.asarray(c_full, dtype=float), eps, 1.0)
    c = c / np.maximum(np.sum(c, axis=1, keepdims=True), eps)
    n_components = c.shape[1]
    n_ind = n_components - 1
    ref = n_components - 1

    theta_left = np.asarray(theta_left, dtype=float)
    theta_right = theta_left if theta_right is None else np.asarray(theta_right, dtype=float)
    theta_x = blend_pairs_np(theta_left, theta_right, np.asarray(x), x_interface, width)

    mu = np.zeros((c.shape[0], n_ind), dtype=float)
    ideal = float(RT) * np.log(c[:, :n_ind] / c[:, ref:ref + 1])
    mu += ideal
    pairs = pair_indices(n_components)
    for row in range(c.shape[0]):
        Omega = omega_matrix_from_pairs_np(theta_x[row], n_components)
        delta = Omega[:n_ind, :] - Omega[ref:ref + 1, :]
        mu[row, :] += delta @ c[row, :]
    return mu


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
    """Torch diffusion potentials mu_i - mu_ref for a multicomponent regular solution."""
    c = torch.clamp(c_full, eps, 1.0)
    c = c / torch.sum(c, dim=1, keepdim=True)
    n_components = c.shape[1]
    n_ind = n_components - 1
    ref = n_components - 1

    if theta_right is None:
        theta_right = theta_left
    theta_x = blend_pairs_torch(theta_left, theta_right, x, x_interface, width)
    pairs = pair_indices(n_components)

    ideal = float(RT) * torch.log(c[:, :n_ind] / c[:, ref:ref + 1])
    mu_cols = []
    for i in range(n_ind):
        excess = torch.zeros((c.shape[0], 1), dtype=c.dtype, device=c.device)
        for k, (a, b) in enumerate(pairs):
            # contribution of (Omega_i,b - Omega_ref,b) c_b without explicitly forming per-row matrices
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


# Binary-compatible wrappers from the original script -------------------------

def mu_regular_solution_np(c, x, RT=1.0, Omega_left=3.0, Omega_right=3.0,
                           x_interface=0.5, width=0.02, eps=1e-12):
    """Original binary regular-solution potential, kept for compatibility."""
    c_safe = np.clip(c, eps, 1.0 - eps)
    Omega_x = 0.5 * (Omega_left + Omega_right) + 0.5 * (Omega_right - Omega_left) * np.tanh((x - x_interface) / width)
    return RT * np.log(c_safe / (1.0 - c_safe)) + Omega_x * (1.0 - 2.0 * c_safe)


def mu_regular_solution_torch(c, x, RT=1.0, Omega_left=3.0, Omega_right=3.0,
                              x_interface=0.5, width=0.02, eps=1e-8):
    """Original binary torch potential, kept for compatibility."""
    c_safe = torch.clamp(c, eps, 1.0 - eps)
    x_in = x.expand_as(c_safe) if x.dim() != c_safe.dim() else x
    Omega_left_t = torch.as_tensor(float(Omega_left), dtype=c_safe.dtype, device=c_safe.device)
    Omega_right_t = torch.as_tensor(float(Omega_right), dtype=c_safe.dtype, device=c_safe.device)
    Omega_x = 0.5 * (Omega_left_t + Omega_right_t) + 0.5 * (Omega_right_t - Omega_left_t) * torch.tanh((x_in - x_interface) / width)
    return RT * torch.log(c_safe / (1.0 - c_safe)) + Omega_x * (1.0 - 2.0 * c_safe)


# -----------------------------------------------------------------------------
# FDM teacher for multicomponent regular-solution diffusion
# -----------------------------------------------------------------------------

def sanitize_independent(U: np.ndarray, eps: float = 1.0e-10) -> np.ndarray:
    U = np.clip(np.asarray(U, dtype=float), eps, 1.0 - eps)
    s = np.sum(U, axis=1)
    bad = s > 1.0 - eps
    if np.any(bad):
        U[bad] = U[bad] / s[bad, None] * (1.0 - eps)
    return U


def make_initial_profile_multicomponent(
    x: np.ndarray,
    c_left: Sequence[float],
    c_right: Sequence[float],
    x0: float = 0.5,
    width: float = 0.02,
) -> np.ndarray:
    """Smooth diffusion-couple profile for all components; rows sum to 1."""
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    c_left = np.asarray(c_left, dtype=float).reshape(1, -1)
    c_right = np.asarray(c_right, dtype=float).reshape(1, -1)
    c_left = c_left / np.sum(c_left)
    c_right = c_right / np.sum(c_right)
    s = 0.5 * (1.0 + np.tanh((x - float(x0)) / max(float(width), 1.0e-12)))
    c = (1.0 - s) * c_left + s * c_right
    return c / np.maximum(np.sum(c, axis=1, keepdims=True), 1.0e-14)


def fdm_multicomponent_regular_solution(
    c0_full: np.ndarray,
    x: np.ndarray,
    dt: float,
    nsteps: int,
    mobility: np.ndarray,
    theta_left: Sequence[float],
    theta_right: Optional[Sequence[float]] = None,
    RT: float = 1.0,
    x_interface: float = 0.5,
    omega_width: float = 0.02,
    save_every: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Explicit conservative FDM for independent components.

    PDE convention follows the original script's thermodynamic driving form:
        c_t = d/dx( M dmu/dx )
    where physical flux J = - M dmu/dx.
    """
    x = np.asarray(x, dtype=float)
    dx = float(x[1] - x[0])
    c0_full = np.asarray(c0_full, dtype=float)
    n_components = c0_full.shape[1]
    n_ind = n_components - 1
    U = sanitize_independent(c0_full[:, :n_ind].copy())
    M = np.asarray(mobility, dtype=float)
    if M.shape != (n_ind, n_ind):
        raise ValueError(f"mobility must have shape {(n_ind, n_ind)}, got {M.shape}")

    times: List[float] = []
    snaps: List[np.ndarray] = []
    theta_right = theta_left if theta_right is None else theta_right

    for n in range(int(nsteps) + 1):
        if n % int(save_every) == 0 or n == int(nsteps):
            snaps.append(complete_composition_np(U))
            times.append(n * dt)
        if n == int(nsteps):
            break

        c_full = complete_composition_np(U)
        mu = diffusion_potentials_regular_solution_np(
            c_full, x, theta_left, theta_right, RT=RT, x_interface=x_interface, width=omega_width
        )
        grad_mu_half = (mu[1:] - mu[:-1]) / dx
        q_half = grad_mu_half @ M.T  # q = M grad(mu); PDE c_t = div(q)
        U_new = U.copy()
        U_new[0] = U[0] + (dt / dx) * (q_half[0] - 0.0)
        U_new[1:-1] = U[1:-1] + (dt / dx) * (q_half[1:] - q_half[:-1])
        U_new[-1] = U[-1] + (dt / dx) * (0.0 - q_half[-1])
        U = sanitize_independent(U_new)

    return np.asarray(times), np.stack(snaps, axis=0)


def bilinear_sample_xt(x_grid: np.ndarray, t_grid: np.ndarray, C: np.ndarray,
                       xq: np.ndarray, tq: np.ndarray) -> np.ndarray:
    """Bilinear interpolation in x,t for composition array C[nt,nx,nc]."""
    xq = np.asarray(xq, dtype=float).reshape(-1)
    tq = np.asarray(tq, dtype=float).reshape(-1)
    out = np.empty((len(xq), C.shape[2]), dtype=float)
    for k, (xx, tt) in enumerate(zip(xq, tq)):
        ix = int(np.clip(np.searchsorted(x_grid, xx) - 1, 0, len(x_grid) - 2))
        it = int(np.clip(np.searchsorted(t_grid, tt) - 1, 0, len(t_grid) - 2))
        x0, x1 = x_grid[ix], x_grid[ix + 1]
        t0, t1 = t_grid[it], t_grid[it + 1]
        wx = 0.0 if x1 == x0 else (xx - x0) / (x1 - x0)
        wt = 0.0 if t1 == t0 else (tt - t0) / (t1 - t0)
        out[k] = (
            (1.0 - wx) * (1.0 - wt) * C[it, ix]
            + wx * (1.0 - wt) * C[it, ix + 1]
            + (1.0 - wx) * wt * C[it + 1, ix]
            + wx * wt * C[it + 1, ix + 1]
        )
    return out


def make_pseudo_experiment_multitime(
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    C_fdm: np.ndarray,
    noise: float,
    seed: int,
    n_points_per_time: int = 64,
    n_time_slices: int = 4,
    t_start: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate noisy pseudo-experimental data from multiple FDM time slices."""
    rng = np.random.default_rng(seed + 2219)
    valid = np.where(np.asarray(t_grid) >= float(t_start))[0]
    if len(valid) == 0:
        valid = np.arange(len(t_grid))
    idxs = np.unique(np.linspace(valid[0], valid[-1], int(max(n_time_slices, 1))).astype(int))
    if idxs[-1] != len(t_grid) - 1:
        idxs = np.unique(np.append(idxs, len(t_grid) - 1)).astype(int)

    x_base = np.linspace(float(x_grid.min()), float(x_grid.max()), int(n_points_per_time))
    xs, ts, cs = [], [], []
    for idx in idxs:
        c_clean = np.column_stack([np.interp(x_base, x_grid, C_fdm[idx, :, j]) for j in range(C_fdm.shape[2])])
        noisy = c_clean + rng.normal(0.0, float(noise), size=c_clean.shape)
        noisy = np.clip(noisy, 1.0e-10, 1.0)
        noisy = noisy / np.maximum(noisy.sum(axis=1, keepdims=True), 1.0e-14)
        xs.append(x_base.reshape(-1, 1))
        ts.append(np.full((len(x_base), 1), float(t_grid[idx])))
        cs.append(noisy)
    return np.vstack(xs), np.vstack(ts), np.vstack(cs), idxs


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
    exp_time_indices: np.ndarray


# -----------------------------------------------------------------------------
# PINN model
# -----------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_layers: Sequence[int] = (96, 96, 96), act: str = "tanh"):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, int(h)))
            if act == "silu":
                layers.append(nn.SiLU())
            elif act == "gelu":
                layers.append(nn.GELU())
            elif act == "relu":
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Tanh())
            prev = int(h)
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X)


class MultiComponentRegularSolutionPINN(nn.Module):
    def __init__(
        self,
        n_components: int,
        theta_left_init: Sequence[float],
        theta_right_init: Optional[Sequence[float]] = None,
        hidden_layers: Sequence[int] = (96, 96, 96),
        activation: str = "tanh",
        learn_left_right_omega: bool = False,
        x_interface: float = 0.5,
        omega_width: float = 0.02,
        RT: float = 1.0,
        train_omega: bool = True,
    ):
        super().__init__()
        self.n_components = int(n_components)
        self.n_ind = self.n_components - 1
        self.n_pairs = len(pair_indices(self.n_components))
        self.net = MLP(2, self.n_components, hidden_layers=hidden_layers, act=activation)
        theta_left_arr = torch.tensor(theta_left_init, dtype=DTYPE).reshape(-1)
        if theta_left_arr.numel() != self.n_pairs:
            raise ValueError(f"theta_left_init must have {self.n_pairs} pair values")
        theta_right_arr = theta_left_arr.clone() if theta_right_init is None else torch.tensor(theta_right_init, dtype=DTYPE).reshape(-1)
        if theta_right_arr.numel() != self.n_pairs:
            raise ValueError(f"theta_right_init must have {self.n_pairs} pair values")
        self.theta_left = nn.Parameter(theta_left_arr.clone(), requires_grad=bool(train_omega))
        self.learn_left_right_omega = bool(learn_left_right_omega)
        self.theta_right = nn.Parameter(theta_right_arr.clone(), requires_grad=bool(train_omega and learn_left_right_omega))
        self.x_interface = float(x_interface)
        self.omega_width = float(omega_width)
        self.RT = float(RT)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        logits = self.net(X)
        return F.softmax(logits, dim=1)

    def theta_vectors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        theta_r = self.theta_right if self.learn_left_right_omega else self.theta_left
        return self.theta_left, theta_r

    def omega_matrix_left(self) -> torch.Tensor:
        return omega_matrix_from_pairs_torch(self.theta_left, self.n_components)

    def omega_matrix_right(self) -> torch.Tensor:
        _, theta_r = self.theta_vectors()
        return omega_matrix_from_pairs_torch(theta_r, self.n_components)

    def residual(self, X: torch.Tensor, mobility: torch.Tensor, eval_mode: bool = False) -> torch.Tensor:
        X_req = X.clone().detach().requires_grad_(True)
        c = self.forward(X_req)
        theta_l, theta_r = self.theta_vectors()
        mu = diffusion_potentials_regular_solution_torch(
            c, X_req[:, 0:1], theta_l, theta_r, RT=self.RT,
            x_interface=self.x_interface, width=self.omega_width
        )
        res_cols = []
        for i in range(self.n_ind):
            c_i = c[:, i:i + 1]
            grad_c_i = torch.autograd.grad(
                c_i, X_req, torch.ones_like(c_i), create_graph=not eval_mode, retain_graph=True
            )[0]
            c_t = grad_c_i[:, 1:2]

            q_i = torch.zeros_like(c_t)
            for j in range(self.n_ind):
                mu_j = mu[:, j:j + 1]
                grad_mu_j = torch.autograd.grad(
                    mu_j, X_req, torch.ones_like(mu_j), create_graph=True, retain_graph=True
                )[0]
                mu_x_j = grad_mu_j[:, 0:1]
                q_i = q_i + mobility[i, j] * mu_x_j
            grad_q_i = torch.autograd.grad(
                q_i, X_req, torch.ones_like(q_i), create_graph=not eval_mode, retain_graph=True
            )[0]
            q_x = grad_q_i[:, 0:1]
            res_cols.append(c_t - q_x)
        return torch.cat(res_cols, dim=1)


@dataclass
class TrainConfig:
    w_data: float = 25.0
    w_ic: float = 12.0
    w_bc: float = 12.0
    w_phys: float = 10.0
    w_omega_prior: float = 0.0
    epochs: int = 10000
    lr: float = 1.0e-3
    n_collocation: int = 3000
    hidden: Tuple[int, ...] = (96, 96, 96)
    activation: str = "tanh"


def make_training_data_from_fdm(
    x: np.ndarray,
    t_grid: np.ndarray,
    C_fdm: np.ndarray,
    n_obs_random: int,
    n_ic: int,
    n_bc_each: int,
    n_f: int,
    noise: float,
    seed: int,
    n_exp_points: int,
    n_exp_time_slices: int,
    append_pseudo_exp_to_training: bool = True,
    t_start_fraction: float = 0.02,
) -> TrainingData:
    rng = np.random.default_rng(seed + 100)
    t_start = max(float(t_grid[-1]) * float(t_start_fraction), float(t_grid[1]))
    x_obs = rng.uniform(float(x.min()), float(x.max()), size=(int(n_obs_random), 1))
    t_obs = rng.uniform(t_start, float(t_grid[-1]), size=(int(n_obs_random), 1))
    c_clean = bilinear_sample_xt(x, t_grid, C_fdm, x_obs, t_obs)
    c_obs = c_clean + rng.normal(0.0, float(noise), size=c_clean.shape)
    c_obs = np.clip(c_obs, 1.0e-10, 1.0)
    c_obs = c_obs / np.maximum(c_obs.sum(axis=1, keepdims=True), 1.0e-14)

    x_exp, t_exp, c_exp, exp_time_indices = make_pseudo_experiment_multitime(
        x, t_grid, C_fdm, noise=noise, seed=seed,
        n_points_per_time=n_exp_points, n_time_slices=n_exp_time_slices, t_start=t_start
    )
    if append_pseudo_exp_to_training:
        x_obs = np.vstack([x_obs, x_exp])
        t_obs = np.vstack([t_obs, t_exp])
        c_obs = np.vstack([c_obs, c_exp])

    x_ic = rng.uniform(float(x.min()), float(x.max()), size=(int(n_ic), 1))
    t_ic = np.full_like(x_ic, t_start)
    c_ic = bilinear_sample_xt(x, t_grid, C_fdm, x_ic, t_ic)

    t_left = rng.uniform(t_start, float(t_grid[-1]), size=(int(n_bc_each), 1))
    t_right = rng.uniform(t_start, float(t_grid[-1]), size=(int(n_bc_each), 1))
    x_bc = np.vstack([np.zeros_like(t_left), np.ones_like(t_right)])
    t_bc = np.vstack([t_left, t_right])
    c_bc = bilinear_sample_xt(x, t_grid, C_fdm, x_bc, t_bc)

    x_f = rng.uniform(float(x.min()), float(x.max()), size=(int(n_f), 1))
    t_f = rng.uniform(t_start, float(t_grid[-1]), size=(int(n_f), 1))

    return TrainingData(x_obs, t_obs, c_obs, x_ic, t_ic, c_ic, x_bc, t_bc, c_bc, x_f, t_f, x, t_grid, C_fdm, exp_time_indices)


def train_pinn_multicomponent(
    model: MultiComponentRegularSolutionPINN,
    data: TrainingData,
    mobility_np: np.ndarray,
    cfg: TrainConfig,
    device: torch.device,
    omega_prior_left: Optional[np.ndarray] = None,
    omega_prior_right: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Tuple[MultiComponentRegularSolutionPINN, Dict[str, List[float]]]:
    model = model.to(device)
    mobility = torch.tensor(mobility_np, dtype=DTYPE, device=device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.epochs, 1), eta_min=cfg.lr * 0.03)
    mse = nn.MSELoss()

    X_obs = torch.tensor(np.hstack([data.x_obs, data.t_obs]), dtype=DTYPE, device=device)
    C_obs = torch.tensor(data.c_obs, dtype=DTYPE, device=device)
    X_ic = torch.tensor(np.hstack([data.x_ic, data.t_ic]), dtype=DTYPE, device=device)
    C_ic = torch.tensor(data.c_ic, dtype=DTYPE, device=device)
    X_bc = torch.tensor(np.hstack([data.x_bc, data.t_bc]), dtype=DTYPE, device=device)
    C_bc = torch.tensor(data.c_bc, dtype=DTYPE, device=device)
    X_f_all = torch.tensor(np.hstack([data.x_f, data.t_f]), dtype=DTYPE, device=device)

    prior_l = None if omega_prior_left is None else torch.tensor(omega_prior_left, dtype=DTYPE, device=device)
    prior_r = None if omega_prior_right is None else torch.tensor(omega_prior_right, dtype=DTYPE, device=device)

    hist: Dict[str, List[float]] = {
        "epoch": [], "loss": [], "data": [], "ic": [], "bc": [], "physics": [], "omega_prior": [], "lr": []
    }
    for k, (i, j) in enumerate(pair_indices(model.n_components)):
        hist[f"Omega_{i}{j}_left"] = []
        hist[f"Omega_{i}{j}_right"] = []

    t0 = time.time()
    report_every = max(1, cfg.epochs // 80)
    n_f_all = X_f_all.shape[0]

    for ep in range(1, cfg.epochs + 1):
        opt.zero_grad(set_to_none=True)
        loss_data = mse(model(X_obs), C_obs)
        loss_ic = mse(model(X_ic), C_ic)
        loss_bc = mse(model(X_bc), C_bc)

        if cfg.n_collocation < n_f_all:
            ids = torch.randint(0, n_f_all, (cfg.n_collocation,), device=device)
            X_f = X_f_all[ids]
        else:
            X_f = X_f_all
        res = model.residual(X_f, mobility, eval_mode=False)
        loss_phys = torch.mean(res * res)

        loss_prior = torch.tensor(0.0, dtype=DTYPE, device=device)
        if cfg.w_omega_prior > 0.0 and prior_l is not None:
            theta_l, theta_r = model.theta_vectors()
            loss_prior = loss_prior + torch.mean((theta_l - prior_l) ** 2)
            if prior_r is not None:
                loss_prior = loss_prior + torch.mean((theta_r - prior_r) ** 2)

        loss = cfg.w_data * loss_data + cfg.w_ic * loss_ic + cfg.w_bc * loss_bc + cfg.w_phys * loss_phys + cfg.w_omega_prior * loss_prior
        loss.backward()
        opt.step()
        scheduler.step()

        if ep == 1 or ep % report_every == 0 or ep == cfg.epochs:
            theta_l, theta_r = model.theta_vectors()
            hist["epoch"].append(float(ep))
            hist["loss"].append(float(loss.detach().cpu()))
            hist["data"].append(float(loss_data.detach().cpu()))
            hist["ic"].append(float(loss_ic.detach().cpu()))
            hist["bc"].append(float(loss_bc.detach().cpu()))
            hist["physics"].append(float(loss_phys.detach().cpu()))
            hist["omega_prior"].append(float(loss_prior.detach().cpu()))
            hist["lr"].append(float(scheduler.get_last_lr()[0]))
            thl = theta_l.detach().cpu().numpy()
            thr = theta_r.detach().cpu().numpy()
            for k, (i, j) in enumerate(pair_indices(model.n_components)):
                hist[f"Omega_{i}{j}_left"].append(float(thl[k]))
                hist[f"Omega_{i}{j}_right"].append(float(thr[k]))
            if verbose:
                pair_msg = ", ".join([f"O{i}{j}L={thl[k]:+.3f}/R={thr[k]:+.3f}" for k, (i, j) in enumerate(pair_indices(model.n_components))])
                print(f"ep {ep:6d}/{cfg.epochs} loss={loss.item():.3e} data={loss_data.item():.3e} phys={loss_phys.item():.3e} time={time.time()-t0:.1f}s | {pair_msg}")

    return model, hist


@torch.no_grad()
def evaluate_model_on_grid(model: MultiComponentRegularSolutionPINN, x_grid: np.ndarray, t_grid: np.ndarray,
                           device: torch.device) -> np.ndarray:
    model.eval()
    Xmesh = np.array(np.meshgrid(x_grid, t_grid, indexing="xy"))
    Xflat = np.stack([Xmesh[0].ravel(), Xmesh[1].ravel()], axis=1)
    X_tensor = torch.tensor(Xflat, dtype=DTYPE, device=device)
    C = model(X_tensor).detach().cpu().numpy().reshape(len(t_grid), len(x_grid), model.n_components)
    return C


def residual_grid(model: MultiComponentRegularSolutionPINN, mobility_np: np.ndarray, x_grid: np.ndarray, t_grid: np.ndarray,
                  device: torch.device, chunk_size: int = 1024) -> np.ndarray:
    model.eval()
    Xmesh = np.array(np.meshgrid(x_grid, t_grid, indexing="xy"))
    Xflat = np.stack([Xmesh[0].ravel(), Xmesh[1].ravel()], axis=1)
    mobility = torch.tensor(mobility_np, dtype=DTYPE, device=device)
    chunks = []
    for start in range(0, len(Xflat), int(chunk_size)):
        end = min(start + int(chunk_size), len(Xflat))
        X = torch.tensor(Xflat[start:end], dtype=DTYPE, device=device)
        R = model.residual(X, mobility, eval_mode=True).detach().cpu().numpy()
        chunks.append(R)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return np.vstack(chunks).reshape(len(t_grid), len(x_grid), model.n_ind)


# -----------------------------------------------------------------------------
# Reliability: Laplace and MCMC over Omega pair terms
# -----------------------------------------------------------------------------

def split_theta(theta: np.ndarray, n_pairs: int, left_right: bool) -> Tuple[np.ndarray, np.ndarray]:
    theta = np.asarray(theta, dtype=float).reshape(-1)
    if left_right:
        if theta.size != 2 * n_pairs:
            raise ValueError(f"Expected {2*n_pairs} theta values for left/right mode")
        return theta[:n_pairs], theta[n_pairs:]
    if theta.size != n_pairs:
        raise ValueError(f"Expected {n_pairs} theta values for single-Omega mode")
    return theta, theta


def predict_fdm_from_theta(
    theta: np.ndarray,
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
) -> Tuple[np.ndarray, np.ndarray]:
    n_pairs = len(pair_indices(n_components))
    theta_l, theta_r = split_theta(theta, n_pairs, left_right)
    return fdm_multicomponent_regular_solution(
        c0_full, x_grid, dt, nsteps, mobility, theta_l, theta_r,
        RT=RT, x_interface=x_interface, omega_width=omega_width, save_every=save_every
    )


def gaussian_nll_multitime(
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
) -> float:
    sigma_eff = max(float(sigma), 1.0e-8)
    t_pred, C_pred = predict_fdm_from_theta(theta, n_components, left_right, c0_full, x_grid, dt, nsteps, save_every, mobility, RT, x_interface, omega_width)
    pred = bilinear_sample_xt(x_grid, t_pred, C_pred, x_exp, t_exp)
    # use independent components; dependent component is constrained by sum
    residual = c_exp[:, :n_components - 1] - pred[:, :n_components - 1]
    n = residual.size
    nll = 0.5 * np.sum((residual / sigma_eff) ** 2) + n * np.log(sigma_eff)
    if prior_mean is not None and prior_std is not None and float(prior_std) > 0:
        nll += 0.5 * np.sum(((np.asarray(theta) - np.asarray(prior_mean)) / float(prior_std)) ** 2)
    return float(nll)


def numerical_hessian(fun, theta0: np.ndarray, step: float) -> np.ndarray:
    theta0 = np.asarray(theta0, dtype=float)
    n = theta0.size
    H = np.zeros((n, n), dtype=float)
    f0 = fun(theta0)
    h = np.ones(n, dtype=float) * float(step)
    for i in range(n):
        ei = np.zeros(n); ei[i] = h[i]
        H[i, i] = (fun(theta0 + ei) - 2.0 * f0 + fun(theta0 - ei)) / (h[i] ** 2)
        for j in range(i + 1, n):
            ej = np.zeros(n); ej[j] = h[j]
            H[i, j] = (fun(theta0 + ei + ej) - fun(theta0 + ei - ej) - fun(theta0 - ei + ej) + fun(theta0 - ei - ej)) / (4.0 * h[i] * h[j])
            H[j, i] = H[i, j]
    return H


def laplace_reliability(fun, theta_hat: np.ndarray, hessian_step: float, n_samples: int, seed: int) -> Dict[str, np.ndarray]:
    H_raw = numerical_hessian(fun, theta_hat, hessian_step)
    H = 0.5 * (H_raw + H_raw.T)
    eig_H_raw = np.linalg.eigvalsh(H)
    H_reg = H + 1.0e-8 * np.eye(len(theta_hat))
    try:
        cov_raw = np.linalg.inv(H_reg)
        inv_method = "inverse"
    except np.linalg.LinAlgError:
        cov_raw = np.linalg.pinv(H_reg)
        inv_method = "pinv"
    eig_cov, V = np.linalg.eigh(0.5 * (cov_raw + cov_raw.T))
    eig_cov_clip = np.clip(eig_cov, 1.0e-10, 25.0)
    cov = V @ np.diag(eig_cov_clip) @ V.T
    rng = np.random.default_rng(seed + 303)
    samples = rng.multivariate_normal(theta_hat, cov, size=int(n_samples), method="svd")
    return {
        "theta_hat": np.asarray(theta_hat, dtype=float),
        "hessian": H,
        "hessian_eig_raw": eig_H_raw,
        "hessian_non_pd": np.array([bool(np.any(eig_H_raw <= 0.0))]),
        "cov": cov,
        "cov_eig_raw": eig_cov,
        "cov_eig_clipped": eig_cov_clip,
        "covariance_was_clipped": np.array([bool(np.any(np.abs(eig_cov - eig_cov_clip) > 1.0e-14))]),
        "hessian_inverse_method": np.array([inv_method]),
        "samples": samples,
    }


def mcmc_reliability(fun, theta_start: np.ndarray, n_steps: int, burn_in: int,
                     proposal_std: float, seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed + 909)
    current = np.asarray(theta_start, dtype=float).copy()
    current_lp = -fun(current)
    samples = []
    accepted = 0
    for step in range(int(n_steps)):
        prop = current + rng.normal(0.0, float(proposal_std), size=current.shape)
        prop_lp = -fun(prop)
        if np.log(rng.uniform()) < prop_lp - current_lp:
            current = prop
            current_lp = prop_lp
            accepted += 1
        if step >= int(burn_in):
            samples.append(current.copy())
    return {"samples": np.asarray(samples), "acceptance_rate": np.array([accepted / max(int(n_steps), 1)])}


def posterior_band_from_samples(
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
    max_samples: int = 80,
) -> Dict[str, np.ndarray]:
    if theta_samples is None or len(theta_samples) == 0:
        nan = np.full((len(x_grid), n_components), np.nan)
        return {"q025": nan, "q500": nan, "q975": nan}
    idx = np.linspace(0, len(theta_samples) - 1, min(int(max_samples), len(theta_samples))).astype(int)
    profiles = []
    for i in idx:
        t_pred, C_pred = predict_fdm_from_theta(theta_samples[i], n_components, left_right, c0_full, x_grid, dt, nsteps, save_every, mobility, RT, x_interface, omega_width)
        prof = bilinear_sample_xt(x_grid, t_pred, C_pred, x_grid.reshape(-1, 1), np.full((len(x_grid), 1), float(target_time)))
        profiles.append(prof)
    P = np.stack(profiles, axis=0)
    return {"q025": np.quantile(P, 0.025, axis=0), "q500": np.quantile(P, 0.500, axis=0), "q975": np.quantile(P, 0.975, axis=0)}


# -----------------------------------------------------------------------------
# Plotting and export
# -----------------------------------------------------------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_loss_plot(hist: Dict[str, List[float]], save_dir: str) -> None:
    plt.figure(figsize=(8, 5))
    for key in ["loss", "data", "ic", "bc", "physics", "omega_prior"]:
        if key in hist and len(hist[key]):
            plt.semilogy(hist["epoch"], np.asarray(hist[key]) + 1.0e-14, label=key)
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_components.png"), dpi=160)
    plt.close()


def save_omega_trace(hist: Dict[str, List[float]], pairs: List[Tuple[int, int]], save_dir: str) -> None:
    plt.figure(figsize=(9, 5))
    for i, j in pairs:
        plt.plot(hist["epoch"], hist[f"Omega_{i}{j}_left"], label=f"Omega_{i}{j} left")
        if np.max(np.abs(np.asarray(hist[f"Omega_{i}{j}_right"]) - np.asarray(hist[f"Omega_{i}{j}_left"]))) > 1.0e-12:
            plt.plot(hist["epoch"], hist[f"Omega_{i}{j}_right"], "--", label=f"Omega_{i}{j} right")
    plt.xlabel("epoch"); plt.ylabel("Omega interaction"); plt.grid(True); plt.legend(ncol=2); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "omega_trace.png"), dpi=160)
    plt.close()


def save_profile_plots(x: np.ndarray, t: np.ndarray, C_true: np.ndarray, C_pred: np.ndarray,
                       component_names: Sequence[str], exp_x: np.ndarray, exp_t: np.ndarray, exp_c: np.ndarray,
                       exp_indices: np.ndarray, save_dir: str) -> None:
    idxs = sorted(set([0, len(t) // 2, len(t) - 1] + [int(i) for i in exp_indices]))
    for idx in idxs:
        plt.figure(figsize=(8, 5))
        for j, name in enumerate(component_names):
            plt.plot(x, C_true[idx, :, j], label=f"FDM {name}")
            plt.plot(x, C_pred[idx, :, j], "--", label=f"PINN {name}")
            mask = np.isclose(exp_t.reshape(-1), t[idx])
            if np.any(mask):
                plt.scatter(exp_x.reshape(-1)[mask], exp_c[mask, j], s=12, alpha=0.55, label=f"pseudo-exp {name}" if j == 0 else None)
        plt.xlabel("x"); plt.ylabel("mole fraction"); plt.ylim(-0.04, 1.04)
        plt.title(f"profiles at t={t[idx]:.4e}")
        plt.grid(True); plt.legend(ncol=2, fontsize=8); plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"profiles_tidx_{idx:03d}.png"), dpi=160)
        plt.close()


def save_heatmaps(x: np.ndarray, t: np.ndarray, C_true: np.ndarray, C_pred: np.ndarray,
                  residual: Optional[np.ndarray], component_names: Sequence[str], save_dir: str) -> None:
    err = np.abs(C_pred - C_true)
    for j, name in enumerate(component_names):
        plt.figure(figsize=(7, 4.5))
        im = plt.imshow(err[:, :, j], extent=(x.min(), x.max(), t.min(), t.max()), aspect="auto", origin="lower")
        plt.colorbar(im, label=f"|error {name}|")
        plt.xlabel("x"); plt.ylabel("t"); plt.title(f"absolute error heatmap: {name}")
        plt.tight_layout(); plt.savefig(os.path.join(save_dir, f"abs_error_heatmap_{name}.png"), dpi=160); plt.close()
    if residual is not None:
        for j in range(residual.shape[2]):
            plt.figure(figsize=(7, 4.5))
            im = plt.imshow(np.abs(residual[:, :, j]), extent=(x.min(), x.max(), t.min(), t.max()), aspect="auto", origin="lower")
            plt.colorbar(im, label=f"|PDE residual comp {j}|")
            plt.xlabel("x"); plt.ylabel("t"); plt.title(f"PDE residual heatmap: independent comp {j}")
            plt.tight_layout(); plt.savefig(os.path.join(save_dir, f"residual_heatmap_ind{j}.png"), dpi=160); plt.close()


def save_posterior_band_plot(x: np.ndarray, C_pinn_final: np.ndarray, band: Dict[str, np.ndarray],
                             component_names: Sequence[str], save_dir: str, filename: str) -> None:
    plt.figure(figsize=(8.5, 5))
    for j, name in enumerate(component_names):
        plt.fill_between(x, band["q025"][:, j], band["q975"][:, j], alpha=0.18)
        plt.plot(x, band["q500"][:, j], ":", label=f"posterior median {name}")
        plt.plot(x, C_pinn_final[:, j], "--", label=f"PINN {name}")
    plt.xlabel("x"); plt.ylabel("mole fraction"); plt.ylim(-0.04, 1.04)
    plt.title("posterior credible bands from FDM forward solves")
    plt.grid(True); plt.legend(ncol=2, fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=160)
    plt.close()


def save_history_csv(hist: Dict[str, List[float]], save_dir: str) -> None:
    keys = list(hist.keys())
    n = max(len(v) for v in hist.values())
    arr = np.full((n, len(keys)), np.nan)
    for k, key in enumerate(keys):
        vals = np.asarray(hist[key], dtype=float)
        arr[:len(vals), k] = vals
    np.savetxt(os.path.join(save_dir, "training_history.csv"), arr, delimiter=",", header=",".join(keys), comments="")




def refine_omega_by_fdm_likelihood(
    nll_fun,
    theta_start: np.ndarray,
    maxiter: int = 180,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Post-PINN black-box refinement of Omega using the FDM likelihood.

    A flexible PINN can fit concentration profiles while leaving thermodynamic
    parameters weakly identified.  The reference Streamlit code uses FDM-based
    likelihood/reliability calculations; here we also use that likelihood as a
    direct post-fit correction for the Omega vector.  This usually improves the
    interaction-term estimate much more than simply increasing epochs.
    """
    theta0 = np.asarray(theta_start, dtype=float).reshape(-1)
    f0 = float(nll_fun(theta0))
    try:
        from scipy.optimize import minimize  # type: ignore
    except Exception as exc:
        if verbose:
            print("Warning: scipy.optimize is unavailable; skipping FDM Omega refinement:", exc)
        return theta0, {"success": 0.0, "nll_initial": f0, "nll_refined": f0, "nit": 0.0, "message": "scipy unavailable"}

    def wrapped(th):
        val = nll_fun(np.asarray(th, dtype=float))
        if not np.isfinite(val):
            return 1.0e100
        return float(val)

    res = minimize(
        wrapped,
        theta0,
        method="Powell",
        options={"maxiter": int(maxiter), "xtol": 1.0e-4, "ftol": 1.0e-4, "disp": False},
    )
    theta_ref = np.asarray(res.x, dtype=float)
    f1 = float(wrapped(theta_ref))
    if verbose:
        print(f"FDM Omega refinement: nll {f0:.6e} -> {f1:.6e}; success={bool(res.success)}; nit={getattr(res, 'nit', 0)}")
    return theta_ref, {
        "success": float(bool(res.success)),
        "nll_initial": f0,
        "nll_refined": f1,
        "nit": float(getattr(res, "nit", 0)),
        "message": str(getattr(res, "message", "")),
    }


def overwrite_model_omega_from_theta(model: MultiComponentRegularSolutionPINN, theta: np.ndarray, left_right: bool) -> None:
    """Copy refined Omega vector back into the trained model for plotting/evaluation."""
    theta_l, theta_r = split_theta(theta, model.n_pairs, left_right)
    with torch.no_grad():
        model.theta_left.copy_(torch.tensor(theta_l, dtype=model.theta_left.dtype, device=model.theta_left.device))
        if model.learn_left_right_omega:
            model.theta_right.copy_(torch.tensor(theta_r, dtype=model.theta_right.dtype, device=model.theta_right.device))


def streamlit_display_saved_results(save_dir: str) -> None:
    """Interactive Plotly result viewer for a completed run."""
    try:
        import streamlit as st  # type: ignore
        import pandas as pd  # type: ignore
        import plotly.graph_objects as go  # type: ignore
    except Exception:
        return

    summary_path = os.path.join(save_dir, "omega_summary.json")
    x_path = os.path.join(save_dir, "x_grid.npy")
    t_path = os.path.join(save_dir, "t_grid.npy")
    cf_path = os.path.join(save_dir, "C_fdm.npy")
    cp_path = os.path.join(save_dir, "C_pinn.npy")
    if not all(os.path.exists(pth) for pth in [summary_path, x_path, t_path, cf_path, cp_path]):
        return

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    x = np.load(x_path)
    t = np.load(t_path)
    C_fdm = np.load(cf_path)
    C_pinn = np.load(cp_path)
    names = summary.get("component_names", [f"C{i}" for i in range(C_fdm.shape[2])])

    st.subheader("対話的診断ビュー")
    tab_profile, tab_time, tab_loss, tab_omega, tab_heat, tab_reliability = st.tabs(
        ["Profile", "Multi-time", "Loss", "Ω trace", "Error maps", "Reliability"]
    )

    def clean(fig, title, height=520):
        fig.update_layout(
            title=dict(text=title, x=0.01, xanchor="left"),
            height=height,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.75)",
            legend=dict(orientation="h", y=-0.22),
            margin=dict(l=10, r=10, t=60, b=105),
        )
        fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.25)")
        fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.25)")
        return fig

    with tab_profile:
        idx = st.slider("表示時刻 index", min_value=0, max_value=len(t)-1, value=len(t)-1)
        fig = go.Figure()
        for j, name in enumerate(names):
            fig.add_trace(go.Scatter(x=x, y=C_fdm[idx, :, j], mode="lines", name=f"FDM {name}", line=dict(width=3)))
            fig.add_trace(go.Scatter(x=x, y=C_pinn[idx, :, j], mode="lines", name=f"PINN {name}", line=dict(width=2.5, dash="dash")))
        fig.update_xaxes(title="x")
        fig.update_yaxes(title="mole fraction", range=[-0.04, 1.04])
        st.plotly_chart(clean(fig, f"Composition profiles at t={float(t[idx]):.4e}"), use_container_width=True)

    with tab_time:
        fig = go.Figure()
        idxs = np.unique(np.linspace(0, len(t)-1, min(5, len(t))).astype(int))
        for idx in idxs:
            for j, name in enumerate(names):
                fig.add_trace(go.Scatter(x=x, y=C_fdm[idx, :, j], mode="lines", name=f"FDM {name} t{idx}", opacity=0.55))
                fig.add_trace(go.Scatter(x=x, y=C_pinn[idx, :, j], mode="lines", name=f"PINN {name} t{idx}", line=dict(dash="dash"), opacity=0.75))
        fig.update_xaxes(title="x")
        fig.update_yaxes(title="mole fraction", range=[-0.04, 1.04])
        st.plotly_chart(clean(fig, "Multi-time profile check", 620), use_container_width=True)

    hist_path = os.path.join(save_dir, "training_history.csv")
    if os.path.exists(hist_path):
        hist = pd.read_csv(hist_path)
        with tab_loss:
            fig = go.Figure()
            for col in ["loss", "data", "ic", "bc", "physics", "omega_prior"]:
                if col in hist.columns:
                    fig.add_trace(go.Scatter(x=hist["epoch"], y=hist[col], mode="lines", name=col))
            fig.update_xaxes(title="epoch")
            fig.update_yaxes(title="loss", type="log")
            st.plotly_chart(clean(fig, "Loss components"), use_container_width=True)
        with tab_omega:
            fig = go.Figure()
            for col in hist.columns:
                if col.startswith("Omega_"):
                    fig.add_trace(go.Scatter(x=hist["epoch"], y=hist[col], mode="lines", name=col))
            fig.update_xaxes(title="epoch")
            fig.update_yaxes(title="Omega")
            st.plotly_chart(clean(fig, "Interaction-parameter trace"), use_container_width=True)
            if "omega_summary" in summary:
                st.dataframe(pd.DataFrame(summary["omega_summary"]), use_container_width=True)

    with tab_heat:
        comp = st.selectbox("component", names, index=min(1, len(names)-1))
        j = list(names).index(comp)
        err = np.abs(C_pinn[:, :, j] - C_fdm[:, :, j])
        fig = go.Figure(data=[go.Heatmap(x=x, y=t, z=err, colorbar=dict(title="abs error"))])
        fig.update_xaxes(title="x")
        fig.update_yaxes(title="t")
        st.plotly_chart(clean(fig, f"Absolute error map: {comp}"), use_container_width=True)

    with tab_reliability:
        rel_path = os.path.join(save_dir, "laplace_reliability.npz")
        if os.path.exists(rel_path):
            rel = np.load(rel_path)
            st.write("Laplace Hessian eigenvalues")
            st.write(rel["hessian_eig_raw"])
            st.write("Covariance eigenvalues before clipping")
            st.write(rel["cov_eig_raw"])
        band_png = os.path.join(save_dir, "laplace_posterior_band_final.png")
        if os.path.exists(band_png):
            st.image(band_png, caption="Laplace posterior band", use_container_width=True)
        if not os.path.exists(rel_path) and not os.path.exists(band_png):
            st.info("信頼性評価は未実行です。")

# -----------------------------------------------------------------------------
# Streamlit front-end helpers
# -----------------------------------------------------------------------------

def running_under_streamlit() -> bool:
    """Return True only when this file is executed by `streamlit run`.

    The CLI workflow is kept separate from the Streamlit front-end.  This avoids
    accidental full retraining whenever a widget value changes and also keeps
    heavy PyTorch/NumPy work in a clean child Python process on Windows.
    """
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx  # type: ignore
        return get_script_run_ctx() is not None
    except Exception:
        return False


def _append_arg(cmd: List[str], flag: str, value) -> None:
    cmd.extend([flag, str(value)])


def streamlit_app() -> None:
    """Robust Streamlit launcher for the CLI workflow.

    Design follows the reference app style: configure first, run explicitly with
    a button, preserve outputs, and surface diagnostics rather than hiding them.
    The actual training is executed as a subprocess so Streamlit widget reruns do
    not duplicate model training or keep stale autograd/OpenMP state alive.
    """
    try:
        import streamlit as st  # type: ignore
    except Exception as exc:
        raise RuntimeError("Streamlit is not installed. Use `python file.py` for CLI mode.") from exc

    st.set_page_config(
        page_title="Multicomponent Regular-Solution PINN",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .stApp {background: linear-gradient(135deg,#f8fafc 0%,#eef4f8 55%,#f7f2f8 100%);}
        section[data-testid="stSidebar"] {background: rgba(255,255,255,0.88);}
        .hero {padding:1.2rem 1.4rem;border-radius:22px;background:rgba(255,255,255,0.76);border:1px solid rgba(148,163,184,0.25);box-shadow:0 12px 28px rgba(100,116,139,0.12);}
        .hero-title {font-size:1.8rem;font-weight:850;color:#1e293b;}
        .hero-sub {color:#475569;line-height:1.7;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="hero">
          <div class="hero-title">多元系 Regular-solution PINN: 相互作用項 Ω<sub>ij</sub> 推定</div>
          <div class="hero-sub">
          化学ポテンシャルの概念は Regular solution のまま維持し、組成プロファイルから多元系のペア相互作用項を推定します。
          計算はボタンを押した時だけ、別Pythonプロセスで実行します。
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("モデル設定")
        quick = st.checkbox("軽量テスト設定にする", value=False, help="まず動作確認したい場合だけON。通常は高精度設定を推奨。")
        n_components = st.number_input("成分数", min_value=2, max_value=6, value=3, step=1)
        default_names = ",".join(default_component_names(int(n_components)))
        component_names = st.text_input("成分名", value="Co,Ni,Ta" if int(n_components) == 3 else default_names)
        left_composition = st.text_input("左側組成", value="0.98,0.01,0.01" if int(n_components) == 3 else ",".join(["0.8"] + [str(0.2/(int(n_components)-1))]*(int(n_components)-1)))
        right_composition = st.text_input("右側組成", value="0.00,0.90,0.10" if int(n_components) == 3 else ",".join([str(0.2)] + [str(0.8/(int(n_components)-1))]*(int(n_components)-1)))
        n_pairs = len(pair_indices(int(n_components)))
        st.caption("Ωの順序: " + ", ".join([f"({i},{j})" for i, j in pair_indices(int(n_components))]))
        theta_left_true = st.text_input("教師Ω left", value=",".join(["2.5", "1.0", "3.5"][:n_pairs]) if n_pairs == 3 else ",".join(["2.0"]*n_pairs))
        theta_left_init = st.text_input("初期Ω left", value=",".join(["1.5", "0.5", "2.0"][:n_pairs]) if n_pairs == 3 else ",".join(["1.0"]*n_pairs))
        learn_lr = st.checkbox("左右別々のΩを学習", value=True)
        theta_right_true = st.text_input("教師Ω right", value=theta_left_true, disabled=not learn_lr)
        theta_right_init = st.text_input("初期Ω right", value=theta_left_init, disabled=not learn_lr)

        st.header("計算設定")
        epochs = st.number_input("epochs", min_value=1, max_value=200000, value=10000 if not quick else 20, step=1000 if not quick else 1)
        lr = st.number_input("learning rate", min_value=1.0e-5, max_value=1.0e-2, value=1.0e-3, step=1.0e-4, format="%.5f")
        hidden = st.text_input("MLP hidden layers", value="128,128,128,128" if not quick else "64,64")
        n_collocation = st.number_input("collocation points/epoch", min_value=8, max_value=50000, value=8000 if not quick else 32, step=500 if not quick else 8)
        n_obs_random = st.number_input("random observation points", min_value=0, max_value=50000, value=2500 if not quick else 80, step=500 if not quick else 10)
        n_f = st.number_input("physics candidate points", min_value=8, max_value=100000, value=9000 if not quick else 80, step=500 if not quick else 10)
        M_diag = st.number_input("mobility diagonal M", min_value=1.0e-5, max_value=1.0, value=2.0e-2 if not quick else 1.0e-3, step=1.0e-3, format="%.5f")
        nx = st.number_input("FDM grid Nx", min_value=21, max_value=401, value=101 if not quick else 21, step=10)
        nsteps = st.number_input("FDM steps", min_value=10, max_value=300000, value=4000 if not quick else 40, step=500 if not quick else 10)
        save_every = st.number_input("FDM save_every", min_value=1, max_value=10000, value=100 if not quick else 20, step=10 if not quick else 1)
        n_exp_time_slices = st.number_input("pseudo-exp 時刻数", min_value=1, max_value=20, value=4 if not quick else 2, step=1)
        n_exp_points = st.number_input("pseudo-exp 点数/時刻", min_value=4, max_value=500, value=48 if not quick else 8, step=4)
        noise = st.number_input("pseudo-exp noise σ", min_value=0.0, max_value=0.1, value=0.004, step=0.001, format="%.4f")
        enable_reliability = st.checkbox("Laplace信頼性評価を実行", value=False if quick else True)
        run_mcmc = st.checkbox("MCMCも実行する", value=False, disabled=not enable_reliability)
        do_fdm_refine = st.checkbox("PINN後にFDM尤度でΩを再最適化", value=True, help="相互作用項の精度を上げるための重要ステップ。")
        fdm_refine_maxiter = st.number_input("FDM再最適化 maxiter", min_value=10, max_value=2000, value=180 if not quick else 20, step=10)
        save_dir = st.text_input("保存先フォルダ", value="pinn_multicomponent_streamlit_results")

    validation_error = None
    try:
        parse_composition(left_composition, int(n_components))
        parse_composition(right_composition, int(n_components))
        parse_theta(theta_left_true, n_pairs)
        parse_theta(theta_left_init, n_pairs)
        if learn_lr:
            parse_theta(theta_right_true, n_pairs)
            parse_theta(theta_right_init, n_pairs)
    except Exception as exc:
        validation_error = str(exc)
        st.error(validation_error)

    col1, col2, col3 = st.columns(3)
    col1.metric("成分数", int(n_components))
    col2.metric("Ωペア数", n_pairs)
    col3.metric("実行モード", "quick" if quick else "standard")

    st.info("Windows/Anacondaで出やすいOpenMP重複問題を避けるため、計算プロセスに KMP_DUPLICATE_LIB_OK=TRUE, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1 を渡します。")

    if st.button("保存済み結果を再表示", disabled=not os.path.exists(os.path.join(save_dir, "omega_summary.json"))):
        streamlit_display_saved_results(save_dir)

    if st.button("PINNでΩ相互作用項を推定する", type="primary", disabled=validation_error is not None):
        cmd: List[str] = [sys.executable, os.path.abspath(__file__)]
        _append_arg(cmd, "--save-dir", save_dir)
        _append_arg(cmd, "--n-components", int(n_components))
        _append_arg(cmd, "--component-names", component_names)
        _append_arg(cmd, "--left-composition", left_composition)
        _append_arg(cmd, "--right-composition", right_composition)
        _append_arg(cmd, "--theta-left-true", theta_left_true)
        _append_arg(cmd, "--theta-left-init", theta_left_init)
        if learn_lr:
            cmd.append("--learn-left-right-omega")
            _append_arg(cmd, "--theta-right-true", theta_right_true)
            _append_arg(cmd, "--theta-right-init", theta_right_init)
        _append_arg(cmd, "--Nx", int(nx))
        _append_arg(cmd, "--nsteps", int(nsteps))
        _append_arg(cmd, "--save-every", int(save_every))
        _append_arg(cmd, "--epochs", int(epochs))
        _append_arg(cmd, "--lr", float(lr))
        _append_arg(cmd, "--hidden", hidden)
        _append_arg(cmd, "--n-collocation", int(n_collocation))
        _append_arg(cmd, "--n-obs-random", int(n_obs_random))
        _append_arg(cmd, "--n-f", int(n_f))
        _append_arg(cmd, "--M-diag", float(M_diag))
        _append_arg(cmd, "--noise", float(noise))
        _append_arg(cmd, "--n-exp-time-slices", int(n_exp_time_slices))
        _append_arg(cmd, "--n-exp-points", int(n_exp_points))
        if quick:
            cmd.append("--quick")
        if not enable_reliability:
            cmd.append("--skip-reliability")
        if run_mcmc and enable_reliability:
            cmd.append("--run-mcmc")
        if not do_fdm_refine:
            cmd.append("--skip-fdm-refine")
        _append_arg(cmd, "--fdm-refine-maxiter", int(fdm_refine_maxiter))

        env = os.environ.copy()
        env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env["PYTHONUNBUFFERED"] = "1"

        st.code(" ".join([repr(c) if " " in c else c for c in cmd]), language="bash")
        log_box = st.empty()
        lines: List[str] = []
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, bufsize=1)
            assert proc.stdout is not None
            for line in proc.stdout:
                lines.append(line.rstrip())
                log_box.code("\n".join(lines[-120:]), language="text")
            ret = proc.wait()
            if ret != 0:
                st.error(f"計算プロセスが終了コード {ret} で停止しました。上のログを確認してください。")
                return
        except Exception as exc:
            st.exception(exc)
            return

        st.success("計算が完了しました。")
        summary_path = os.path.join(save_dir, "omega_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            st.subheader("Ω推定結果")
            st.json(summary.get("omega_summary", summary))
            st.metric("Global RMSE", f"{summary.get('global_rmse', float('nan')):.4e}")

        streamlit_display_saved_results(save_dir)
        st.caption(f"保存先: {os.path.abspath(save_dir)}")

# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------

def parse_composition(text: str, n_components: int) -> np.ndarray:
    vals = np.array([float(v.strip()) for v in text.split(",")], dtype=float)
    if vals.size != n_components:
        raise ValueError(f"Composition must contain {n_components} comma-separated values")
    vals = np.clip(vals, 1.0e-12, 1.0)
    return vals / np.sum(vals)


def parse_theta(text: str, n_pairs: int) -> np.ndarray:
    vals = np.array([float(v.strip()) for v in text.split(",")], dtype=float)
    if vals.size != n_pairs:
        raise ValueError(f"Theta must contain {n_pairs} comma-separated pair Omega values")
    return vals


def default_component_names(n: int) -> List[str]:
    base = ["A", "B", "C", "D", "E", "F"]
    return base[:n] if n <= len(base) else [f"C{i}" for i in range(n)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Multicomponent regular-solution diffusion-couple PINN")
    parser.add_argument("--save-dir", default="pinn_multicomponent_results")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n-components", type=int, default=3)
    parser.add_argument("--component-names", default="A,B,C")
    parser.add_argument("--left-composition", default="0.78,0.17,0.05")
    parser.add_argument("--right-composition", default="0.20,0.68,0.12")
    parser.add_argument("--theta-left-true", default="2.5,1.0,3.5", help="Pair Omegas in order (0,1),(0,2),(1,2), etc.")
    parser.add_argument("--theta-right-true", default=None)
    parser.add_argument("--theta-left-init", default="1.5,0.5,2.0")
    parser.add_argument("--theta-right-init", default=None)
    parser.set_defaults(learn_left_right_omega=True)
    parser.add_argument("--learn-left-right-omega", dest="learn_left_right_omega", action="store_true", help="Use separate left/right Omega vectors. Default: enabled.")
    parser.add_argument("--single-omega", dest="learn_left_right_omega", action="store_false", help="Use one shared Omega vector for the whole diffusion couple.")
    parser.add_argument("--no-train-omega", action="store_true")
    parser.add_argument("--RT", type=float, default=1.0)
    parser.add_argument("--M-diag", type=float, default=2.0e-2)
    parser.add_argument("--M-offdiag", type=float, default=0.0)
    parser.add_argument("--L", type=float, default=1.0)
    parser.add_argument("--Nx", type=int, default=101)
    parser.add_argument("--dt", type=float, default=1.0e-5)
    parser.add_argument("--nsteps", type=int, default=4000)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--interface", type=float, default=0.5)
    parser.add_argument("--interface-width", type=float, default=0.02)
    parser.add_argument("--noise", type=float, default=0.004)
    parser.add_argument("--n-obs-random", type=int, default=2500)
    parser.add_argument("--n-exp-points", type=int, default=64)
    parser.add_argument("--n-exp-time-slices", type=int, default=4)
    parser.add_argument("--n-ic", type=int, default=256)
    parser.add_argument("--n-bc-each", type=int, default=128)
    parser.add_argument("--n-f", type=int, default=9000)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--n-collocation", type=int, default=8000)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--hidden", default="128,128,128,128")
    parser.add_argument("--activation", default="tanh", choices=["tanh", "relu", "silu", "gelu"])
    parser.add_argument("--w-data", type=float, default=80.0)
    parser.add_argument("--w-ic", type=float, default=25.0)
    parser.add_argument("--w-bc", type=float, default=25.0)
    parser.add_argument("--w-phys", type=float, default=25.0)
    parser.add_argument("--w-omega-prior", type=float, default=0.0)
    parser.add_argument("--like-sigma", type=float, default=0.008)
    parser.add_argument("--laplace-samples", type=int, default=800)
    parser.add_argument("--hessian-step", type=float, default=0.03)
    parser.add_argument("--band-samples", type=int, default=50)
    parser.add_argument("--run-mcmc", action="store_true")
    parser.add_argument("--mcmc-steps", type=int, default=600)
    parser.add_argument("--mcmc-burn", type=int, default=150)
    parser.add_argument("--mcmc-proposal", type=float, default=0.035)
    parser.add_argument("--skip-residual-grid", action="store_true")
    parser.add_argument("--skip-reliability", action="store_true")
    parser.add_argument("--skip-fdm-refine", action="store_true", help="Skip post-PINN black-box FDM likelihood refinement of Omega.")
    parser.add_argument("--fdm-refine-maxiter", type=int, default=180, help="Max iterations for Powell FDM likelihood refinement.")
    parser.add_argument("--quick", action="store_true", help="Small fast run for smoke testing")
    args = parser.parse_args()

    if args.quick:
        args.Nx = 21
        args.nsteps = 40
        args.save_every = 20
        args.epochs = min(args.epochs, 20)
        args.n_collocation = min(args.n_collocation, 32)
        args.n_obs_random = min(args.n_obs_random, 80)
        args.n_exp_points = 8
        args.n_exp_time_slices = 2
        args.n_ic = 24
        args.n_bc_each = 12
        args.n_f = 40
        args.laplace_samples = 12
        args.band_samples = 2
        args.run_mcmc = False
        args.skip_residual_grid = True
        args.skip_reliability = True

    set_seed(args.seed)
    ensure_dir(args.save_dir)
    device = get_device()
    print("Device:", device)

    n_components = int(args.n_components)
    n_ind = n_components - 1
    pairs = pair_indices(n_components)
    n_pairs = len(pairs)
    component_names = [s.strip() for s in args.component_names.split(",")]
    if len(component_names) != n_components:
        component_names = default_component_names(n_components)

    c_left = parse_composition(args.left_composition, n_components)
    c_right = parse_composition(args.right_composition, n_components)
    theta_left_true = parse_theta(args.theta_left_true, n_pairs)
    theta_right_true = theta_left_true if args.theta_right_true is None else parse_theta(args.theta_right_true, n_pairs)
    theta_left_init = parse_theta(args.theta_left_init, n_pairs)
    theta_right_init = theta_left_init if args.theta_right_init is None else parse_theta(args.theta_right_init, n_pairs)

    x = np.linspace(0.0, float(args.L), int(args.Nx))
    c0_full = make_initial_profile_multicomponent(x, c_left, c_right, x0=args.interface, width=args.interface_width)
    mobility = np.eye(n_ind) * float(args.M_diag) + (np.ones((n_ind, n_ind)) - np.eye(n_ind)) * float(args.M_offdiag)

    print("Running multicomponent FDM teacher...")
    t_grid, C_fdm = fdm_multicomponent_regular_solution(
        c0_full, x, args.dt, args.nsteps, mobility, theta_left_true, theta_right_true,
        RT=args.RT, x_interface=args.interface, omega_width=args.interface_width, save_every=args.save_every
    )

    data = make_training_data_from_fdm(
        x, t_grid, C_fdm,
        n_obs_random=args.n_obs_random, n_ic=args.n_ic, n_bc_each=args.n_bc_each, n_f=args.n_f,
        noise=args.noise, seed=args.seed, n_exp_points=args.n_exp_points,
        n_exp_time_slices=args.n_exp_time_slices, append_pseudo_exp_to_training=True
    )

    cfg = TrainConfig(
        w_data=args.w_data, w_ic=args.w_ic, w_bc=args.w_bc, w_phys=args.w_phys,
        w_omega_prior=args.w_omega_prior, epochs=args.epochs, lr=args.lr,
        n_collocation=args.n_collocation,
        hidden=tuple(int(v.strip()) for v in args.hidden.split(",") if v.strip()),
        activation=args.activation,
    )

    model = MultiComponentRegularSolutionPINN(
        n_components=n_components,
        theta_left_init=theta_left_init,
        theta_right_init=theta_right_init,
        hidden_layers=cfg.hidden,
        activation=cfg.activation,
        learn_left_right_omega=args.learn_left_right_omega,
        x_interface=args.interface,
        omega_width=args.interface_width,
        RT=args.RT,
        train_omega=not args.no_train_omega,
    )

    print("Training multicomponent PINN for Omega interaction terms...")
    model, hist = train_pinn_multicomponent(
        model, data, mobility, cfg, device,
        omega_prior_left=theta_left_init, omega_prior_right=theta_right_init,
        verbose=True
    )

    C_pinn = evaluate_model_on_grid(model, x, t_grid, device)
    err_global = float(np.sqrt(np.mean((C_pinn - C_fdm) ** 2)))
    print(f"Global RMSE over all components/times: {err_global:.4e}")

    if args.skip_residual_grid:
        R = None
        print("Skipping residual grid evaluation.")
    else:
        try:
            R = residual_grid(model, mobility, x, t_grid, device, chunk_size=1024)
        except Exception as exc:
            print("Warning: residual grid failed:", exc)
            R = None

    theta_l_est, theta_r_est = model.theta_vectors()
    theta_l_np = theta_l_est.detach().cpu().numpy()
    theta_r_np = theta_r_est.detach().cpu().numpy()
    theta_hat = np.concatenate([theta_l_np, theta_r_np]) if args.learn_left_right_omega else theta_l_np.copy()
    prior_mean = np.concatenate([theta_left_init, theta_right_init]) if args.learn_left_right_omega else theta_left_init.copy()

    # Reliability uses pseudo-exp points only, not the random collocation/data points.
    x_exp = data.x_obs[-args.n_exp_points * len(data.exp_time_indices):]
    t_exp = data.t_obs[-args.n_exp_points * len(data.exp_time_indices):]
    c_exp = data.c_obs[-args.n_exp_points * len(data.exp_time_indices):]

    def nll_fun(th: np.ndarray) -> float:
        return gaussian_nll_multitime(
            th, n_components, args.learn_left_right_omega, c0_full, x, x_exp, t_exp, c_exp,
            sigma=args.like_sigma, dt=args.dt, nsteps=args.nsteps, save_every=args.save_every,
            mobility=mobility, RT=args.RT, x_interface=args.interface, omega_width=args.interface_width,
            prior_mean=prior_mean, prior_std=5.0,
        )

    refine_info = None
    theta_hat_pinn = theta_hat.copy()
    if args.skip_fdm_refine:
        print("Skipping post-PINN FDM likelihood refinement of Omega.")
    else:
        print("Refining Omega by direct FDM likelihood optimization...")
        theta_refined, refine_info = refine_omega_by_fdm_likelihood(
            nll_fun, theta_hat, maxiter=args.fdm_refine_maxiter, verbose=True
        )
        if np.all(np.isfinite(theta_refined)):
            theta_hat = theta_refined
            overwrite_model_omega_from_theta(model, theta_hat, args.learn_left_right_omega)
            C_pinn = evaluate_model_on_grid(model, x, t_grid, device)
            err_global = float(np.sqrt(np.mean((C_pinn - C_fdm) ** 2)))
            print(f"Global RMSE after Omega refinement: {err_global:.4e}")

    low_rel = None
    band = None
    high_rel = None
    if args.skip_reliability:
        print("Skipping likelihood reliability evaluation.")
    else:
        print("Computing Laplace reliability for Omega interaction terms...")
        low_rel = laplace_reliability(nll_fun, theta_hat, args.hessian_step, args.laplace_samples, args.seed)
        band = posterior_band_from_samples(
            low_rel["samples"], n_components, args.learn_left_right_omega, c0_full, x,
            args.dt, args.nsteps, args.save_every, mobility, args.RT, args.interface,
            args.interface_width, target_time=float(t_grid[-1]), max_samples=args.band_samples
        )

        if args.run_mcmc:
            print("Running high-cost FDM-based MCMC reliability for Omega interaction terms...")
            high_rel = mcmc_reliability(nll_fun, theta_hat, args.mcmc_steps, args.mcmc_burn, args.mcmc_proposal, args.seed)
            print(f"MCMC acceptance rate: {float(high_rel['acceptance_rate'][0]):.3f}")

    # Refresh estimated theta vectors after possible FDM refinement.
    theta_l_est, theta_r_est = model.theta_vectors()
    theta_l_np = theta_l_est.detach().cpu().numpy()
    theta_r_np = theta_r_est.detach().cpu().numpy()

    # Save artifacts
    np.save(os.path.join(args.save_dir, "x_grid.npy"), x)
    np.save(os.path.join(args.save_dir, "t_grid.npy"), t_grid)
    np.save(os.path.join(args.save_dir, "C_fdm.npy"), C_fdm)
    np.save(os.path.join(args.save_dir, "C_pinn.npy"), C_pinn)
    if R is not None:
        np.save(os.path.join(args.save_dir, "residual_grid.npy"), R)
    if low_rel is not None:
        np.savez_compressed(os.path.join(args.save_dir, "laplace_reliability.npz"), **low_rel)
    if high_rel is not None:
        np.savez_compressed(os.path.join(args.save_dir, "mcmc_reliability.npz"), **high_rel)

    save_history_csv(hist, args.save_dir)
    save_loss_plot(hist, args.save_dir)
    save_omega_trace(hist, pairs, args.save_dir)
    save_profile_plots(x, t_grid, C_fdm, C_pinn, component_names, x_exp, t_exp, c_exp, data.exp_time_indices, args.save_dir)
    save_heatmaps(x, t_grid, C_fdm, C_pinn, R, component_names, args.save_dir)
    if band is not None:
        save_posterior_band_plot(x, C_pinn[-1], band, component_names, args.save_dir, "laplace_posterior_band_final.png")

    pair_rows = []
    for k, (i, j) in enumerate(pairs):
        pair_rows.append({
            "pair": f"Omega_{component_names[i]}_{component_names[j]}",
            "true_left": float(theta_left_true[k]),
            "estimated_left": float(theta_l_np[k]),
            "true_right": float(theta_right_true[k]),
            "estimated_right": float(theta_r_np[k]),
            "abs_error_left": abs(float(theta_l_np[k] - theta_left_true[k])),
            "abs_error_right": abs(float(theta_r_np[k] - theta_right_true[k])),
        })
    with open(os.path.join(args.save_dir, "omega_summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "global_rmse": err_global,
            "fdm_refine": refine_info,
            "theta_hat_pinn_before_fdm_refine": theta_hat_pinn.tolist(),
            "theta_hat_used_for_reliability_and_summary": theta_hat.tolist(),
            "component_names": component_names,
            "pair_order": [(component_names[i], component_names[j]) for i, j in pairs],
            "omega_summary": pair_rows,
            "laplace_hessian_non_pd": None if low_rel is None else bool(low_rel["hessian_non_pd"][0]),
            "laplace_covariance_was_clipped": None if low_rel is None else bool(low_rel["covariance_was_clipped"][0]),
            "args": vars(args),
        }, f, indent=2, ensure_ascii=False)

    print("\nEstimated interaction terms:")
    for row in pair_rows:
        print(
            f"  {row['pair']}: left true={row['true_left']:+.4f}, est={row['estimated_left']:+.4f}; "
            f"right true={row['true_right']:+.4f}, est={row['estimated_right']:+.4f}"
        )
    print("Saved results to", args.save_dir)


if __name__ == "__main__":
    if running_under_streamlit():
        streamlit_app()
    else:
        main()
