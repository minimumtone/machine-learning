"""
Co-Ni-Ta 合金拡散フィット: 時系列濃度プロファイル + 拡散係数逆解析 + HTML レポート
==========================================================================
Regular-solution モードで FDM 教師 → PINN 学習 → 移動度 (拡散係数) の逆解析 を 1 回だけ実行し、
その結果から時系列の濃度プロファイル図・データ・学生向け HTML レポートをまとめて出力する。

計算の流れ:
    Step 1  RS FDM 順問題 (真の M, Ω) → 教師データ + FDM 疑似実験データ (多時刻)
    Step 2  PINN 順フィット (M, Ω 既知) → 時系列濃度プロファイル
    Step 3  逆解析 (Ω 既知, M 未知):
              3a  PINN に学習可能な log M を入れて同時推定
              3b  学習済み PINN の微分から M を最小二乗で推定 (事後推定)
              3c  FDM を繰り返して疑似実験データとの尤度マップ (信頼度) を作成
              3d  推定 M で FDM を再計算して真のプロファイルと比較

出力 (--outdir 以下):
    REPORT.html                     学生向けレポート (PINN の解説から逆解析まで, 図は埋め込み)
    REPORT.md                       同内容の簡易 Markdown 版
    timeseries_results.npz          全配列 (再描画用)
    timeseries_profiles.csv         long 形式 (time, x, c_FDM, c_PINN)
    pseudo_exp_points.csv           FDM による疑似実験データ点
    training_history.csv            順フィット PINN の loss 履歴
    inverse_history.csv             逆解析 PINN の loss / M 履歴
    nll_grid.csv                    FDM 尤度マップ
    fig_*.png, timeseries_animation.gif

使い方:
    python plot_diffusion_timeseries_profiles.py                       # 計算 + 描画
    python plot_diffusion_timeseries_profiles.py --epochs 6000 --device cpu
    python plot_diffusion_timeseries_profiles.py --replot results/timeseries_results.npz
                                                                      # 再計算なしで再描画
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import html
import importlib.util
import io
import json
import math
import os
import sys
import time
import types

import numpy as np

COMPONENTS = ["Co", "Ni", "Ta"]
COMP_LATEX = {"Co": r"$c_{\mathrm{Co}}$", "Ni": r"$c_{\mathrm{Ni}}$", "Ta": r"$c_{\mathrm{Ta}}$"}
COMP_COLORS = {"Co": "black", "Ni": "#2b83ba", "Ta": "#4daf4a"}
COMP_MARKERS = {"Co": "o", "Ni": "s", "Ta": "^"}
OMEGA_LABELS = [r"$\Omega_{\mathrm{CoNi}}$", r"$\Omega_{\mathrm{CoTa}}$", r"$\Omega_{\mathrm{NiTa}}$"]
IND_COMPONENTS = ["Ni", "Ta"]  # independent components (internal order); Co is the reference
M_LABELS = [r"$M_{\mathrm{Ni}}$", r"$M_{\mathrm{Ta}}$"]
FONTSIZE = 20

# Initial diffusion-couple end compositions in display order [Co, Ni, Ta]
# (must match make_training_data_rs in the core module).
_EPS_GUARD = 5.0e-3
_C_LEFT_DISP = np.array([1.0 - _EPS_GUARD, _EPS_GUARD / 2, _EPS_GUARD / 2])
_C_RIGHT_DISP = np.array([_EPS_GUARD / 2, 0.9, 0.1])
_NX_FDM = 201
_X_INTERFACE = 0.5
_OMEGA_WIDTH = 0.02
_PHASE_WIDTH = 0.02
_RT = 1.0

# Display-only settings.  They are stored in the results NPZ, so --replot keeps the
# values of the original run unless they are given explicitly on the command line.
_DEFAULT_ANNEALING_TIME_H = 160.0
_DEFAULT_SPAN_UM = 800.0
_DEFAULT_N_TIME_SLICES = 8
_LR_FLOOR_FRACTION = 0.03  # cosine decay floor, relative to each parameter group's initial lr


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Time-series concentration profiles for Co-Ni-Ta diffusion fit (RS mode)")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--epochs", type=int, default=6000)
    p.add_argument("--outdir", default="./timeseries_output")
    p.add_argument("--replot", default=None, help="Path to timeseries_results.npz; skip computation and only re-draw figures")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=0, help="CPU threads for torch (0 = all cores)")
    p.add_argument("--mu_floor", type=float, default=5.0e-3)
    p.add_argument("--w_phys", type=float, default=0.2)
    p.add_argument("--w_data", type=float, default=25.0)
    p.add_argument("--w_ic", type=float, default=12.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=float, default=0.3, help="Physics-loss warmup fraction")
    p.add_argument("--n_obs", type=int, default=1500)
    p.add_argument("--n_f", type=int, default=2000, help="Collocation points per epoch (forward and inverse PINN)")
    p.add_argument("--omega", type=float, nargs=3, default=[1.0, 1.0, 1.0], help="True Omega [CoNi, CoTa, NiTa]")
    p.add_argument("--mobility_diag", type=float, nargs="+", default=[1.0e-2, 4.0e-3],
                   help="True mobility diagonal [M_Ni, M_Ta] (one value = same for both)")
    p.add_argument("--skip_inverse", action="store_true", help="Skip the mobility inverse analysis (Step 3)")
    p.add_argument("--inverse_epochs", type=int, default=0, help="Epochs for the inverse PINN (0 = same as --epochs)")
    p.add_argument("--m_init_factor", type=float, default=3.0, help="Initial guess M_init = factor x M_true for the inverse PINN")
    p.add_argument("--lr_m", type=float, default=5e-3, help="Learning rate for log M in the inverse PINN")
    p.add_argument("--nll_grid", type=int, default=11, help="Grid points per axis for the FDM likelihood map (0 = skip)")
    p.add_argument("--nll_half_decades", type=float, default=0.5, help="Half-width of the likelihood map in log10(M)")
    p.add_argument("--dt", type=float, default=1e-5, help="RS FDM time step")
    p.add_argument("--nsteps", type=int, default=4000, help="RS FDM macro steps")
    p.add_argument("--save_every", type=int, default=100, help="RS FDM frame interval (frames = nsteps/save_every + 1)")
    p.add_argument("--noise", type=float, default=0.02, help="Pseudo-experimental noise (mole fraction)")
    p.add_argument("--n_exp_points", type=int, default=24, help="Pseudo-experimental points per time slice")
    p.add_argument("--n_time_slices", type=int, default=None,
                   help=f"Number of pseudo-experimental time slices (= panels); default {_DEFAULT_N_TIME_SLICES}, "
                        "--replot keeps the saved value")
    p.add_argument("--onsager", action="store_true", help="Use Onsager-form PDE residual instead of Fick form")
    p.add_argument("--annealing_time_h", type=float, default=None,
                   help=f"Physical annealing time mapped to final frame; default {_DEFAULT_ANNEALING_TIME_H}, "
                        "--replot keeps the saved value")
    p.add_argument("--span_um", type=float, default=None,
                   help=f"Physical distance span mapped to x in [0, 1]; default {_DEFAULT_SPAN_UM}, --replot keeps the saved value")
    p.add_argument("--no_gif", action="store_true")
    args = p.parse_args()
    if args.replot is None:
        if args.nsteps < 2 * args.save_every:
            p.error("--nsteps must be >= 2 * --save_every so that at least 3 FDM frames are saved "
                    "(needed for the time-derivative baseline of the PDE-consistency check)")
        if args.noise < 0:
            p.error("--noise must be >= 0")
        if args.noise == 0 and not args.skip_inverse:
            p.error("--noise 0 has no likelihood: the Gaussian NLL of Step 3 divides by sigma^2. "
                    "Use --noise > 0, or --skip_inverse for a noise-free forward-only run")
    return args


def _resolve_display_settings(args: argparse.Namespace, saved: dict | None) -> None:
    """Fill annealing_time_h / span_um / n_time_slices: CLI value > saved NPZ config > default."""
    saved = saved or {}
    if args.annealing_time_h is None:
        v = saved.get("annealing_time_h")
        args.annealing_time_h = float(v) if v is not None else _DEFAULT_ANNEALING_TIME_H
    if args.span_um is None:
        v = saved.get("span_um")
        args.span_um = float(v) if v is not None else _DEFAULT_SPAN_UM
    if args.n_time_slices is None:
        v = saved.get("n_time_slices")
        args.n_time_slices = int(v) if v is not None else _DEFAULT_N_TIME_SLICES


# ---------------------------------------------------------------------------
# Core-module loading (the core file is a Streamlit app; stub Streamlit out)
# ---------------------------------------------------------------------------
class _SessionStateMock(dict):
    def __getattr__(self, name):
        return self.get(name, None)

    def __setattr__(self, name, value):
        self[name] = value


class _StopCallable:
    def __call__(self, *a, **kw):
        raise SystemExit(0)


class _CallableMock:
    def __call__(self, *a, **kw):
        if len(a) == 1 and isinstance(a[0], int):
            return [_CallableMock() for _ in range(a[0])]
        return _CallableMock()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def __iter__(self):
        return iter([_CallableMock(), _CallableMock()])

    def __bool__(self):
        return False

    def __getattr__(self, name):
        return _CallableMock()

    def __float__(self):
        return 0.0

    def __int__(self):
        return 0

    def __str__(self):
        return ""

    def __len__(self):
        return 0

    def __contains__(self, item):
        return False


class _StMock:
    def __init__(self):
        object.__setattr__(self, "session_state", _SessionStateMock())

    def __getattr__(self, name):
        if name == "session_state":
            return object.__getattribute__(self, "session_state")
        if name == "stop":
            return _StopCallable()
        return _CallableMock()

    def cache_data(self, *a, **kw):
        return lambda f: f


def load_core_module():
    sys.modules["streamlit"] = _StMock()
    for m in ["streamlit.components", "streamlit.components.v1",
              "streamlit.runtime", "streamlit.runtime.scriptrunner_utils"]:
        sys.modules[m] = types.ModuleType(m)

    here = os.path.dirname(os.path.abspath(__file__))
    main_py = os.path.join(here, "co_ni_ta_pinn_diffusion_reliability.py")
    if not os.path.isfile(main_py):
        raise FileNotFoundError(f"core module not found: {main_py}")
    spec = importlib.util.spec_from_file_location("co_ni_ta_pinn_diffusion_reliability", main_py)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["co_ni_ta_pinn_diffusion_reliability"] = mod
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    return mod


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------
def _mobility_from_args(args: argparse.Namespace) -> np.ndarray:
    vals = list(args.mobility_diag)
    if len(vals) == 1:
        vals = vals * 2
    if len(vals) != 2:
        raise ValueError("--mobility_diag takes 1 or 2 values")
    return np.diag(np.asarray(vals, dtype=float))


def _fdm_forward(mod, args: argparse.Namespace, omega_true: np.ndarray, mobility: np.ndarray):
    """Run the RS FDM with the same set-up as make_training_data_rs.

    Returns (t_grid_normalized, C_fdm_display, t_max_physical).
    """
    x_grid = np.linspace(0.0, 1.0, _NX_FDM)
    c_left = _C_LEFT_DISP / _C_LEFT_DISP.sum()
    c_right = _C_RIGHT_DISP / _C_RIGHT_DISP.sum()
    c0_disp = mod.make_initial_profile_ternary_rs(x_grid, c_left, c_right, x0=_X_INTERFACE, width=_PHASE_WIDTH)
    c0_int = mod._reorder_c_display_to_internal_np(c0_disp)
    theta_int = mod._reorder_theta_display_to_internal(omega_true)
    t_grid, C_int = mod.fdm_ternary_regular_solution(
        c0_int, x_grid, args.dt, args.nsteps, mobility, theta_int, theta_int,
        RT=_RT, x_interface=_X_INTERFACE, omega_width=_OMEGA_WIDTH,
        save_every=args.save_every, mu_floor=args.mu_floor,
    )
    t_max = float(t_grid[-1])
    return t_grid / t_max if t_max > 0 else t_grid, mod._reorder_c_internal_to_display_np(C_int), t_max


def _fick_residual_trainable_M(mod, model, x, t, M_eff, theta_int_t, mu_floor: float):
    """Fick-form residual  ∂c/∂t − ∂/∂x[(M·Φ(c))·∂c/∂x]  with a *trainable* mobility M.

    The thermodynamic factor Φ(c) = ∂(μ_j − μ_Co)/∂c_m is evaluated on detached
    compositions (frozen-coefficient approach, as in the core Fick form), but the
    mobility stays in the autograd graph so gradients reach log M.
    """
    import torch
    x = x.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)
    C = model(x, t)
    C_int = mod._reorder_display_to_internal(C)
    c0, c1 = C_int[:, 0:1], C_int[:, 1:2]
    ones = torch.ones_like(c0)
    c0_t = torch.autograd.grad(c0, t, ones, create_graph=True, retain_graph=True)[0]
    c1_t = torch.autograd.grad(c1, t, ones, create_graph=True, retain_graph=True)[0]
    c0_x = torch.autograd.grad(c0, x, ones, create_graph=True, retain_graph=True)[0]
    c1_x = torch.autograd.grad(c1, x, ones, create_graph=True, retain_graph=True)[0]
    Phi = mod.interdiffusion_matrix_rs_torch(
        C_int.detach(), theta_int_t, None, x=None, RT=_RT, mobility=None, mu_floor=mu_floor,
    )
    D = torch.einsum("kj,njm->nkm", M_eff, Phi)
    q0 = D[:, 0, 0:1] * c0_x + D[:, 0, 1:2] * c1_x
    q1 = D[:, 1, 0:1] * c0_x + D[:, 1, 1:2] * c1_x
    q0_x = torch.autograd.grad(q0, x, ones, create_graph=True, retain_graph=True)[0]
    q1_x = torch.autograd.grad(q1, x, ones, create_graph=True, retain_graph=True)[0]
    return torch.cat([c0_t - q0_x, c1_t - q1_x], dim=1)


def train_inverse_pinn(mod, data, args: argparse.Namespace, omega_true: np.ndarray,
                       M_true_diag: np.ndarray, epochs: int):
    """Joint PINN inverse problem: network weights + log M (Ω known)."""
    import pandas as pd
    import torch
    import torch.nn.functional as F

    device = mod.DEVICE
    mod.set_seed(args.seed + 1)
    model = mod.TernaryRegularSolutionPINN(
        width=64, depth=4, activation="tanh",
        theta_left_init=omega_true, theta_right_init=None,
        learn_left_right_omega=False, x_interface=_X_INTERFACE,
        omega_width=_OMEGA_WIDTH, RT=_RT, train_omega=False,
        direct_output=True, n_time_fourier=4,
        mu_floor=args.mu_floor, use_fick_form=True,
    ).to(device)
    t_scale = float(data.rs_t_max_physical) if data.rs_t_max_physical > 0 else 1.0
    M_init = M_true_diag * args.m_init_factor
    log_M = torch.nn.Parameter(torch.tensor(np.log(M_init), dtype=torch.float32, device=device))
    theta_int_t = torch.tensor(mod._reorder_theta_display_to_internal(omega_true), dtype=torch.float32, device=device)

    x_obs = mod.to_tensor(data.x_obs.reshape(-1, 1)).to(device)
    t_obs = mod.to_tensor(data.t_obs.reshape(-1, 1)).to(device)
    c_obs = mod.to_tensor(data.c_obs).to(device)
    x_ic = mod.to_tensor(data.x_ic.reshape(-1, 1)).to(device)
    t_ic = mod.to_tensor(data.t_ic.reshape(-1, 1)).to(device)
    c_ic = mod.to_tensor(data.c_ic).to(device)

    net_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam([{"params": net_params, "lr": args.lr}, {"params": [log_M], "lr": args.lr_m}])
    T = max(epochs, 1)

    def _cosine_factor(step: int) -> float:
        # same relative decay (1 -> _LR_FLOOR_FRACTION) for the network and the log M group
        return _LR_FLOOR_FRACTION + (1.0 - _LR_FLOOR_FRACTION) * 0.5 * (1.0 + math.cos(math.pi * min(step, T) / T))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=[_cosine_factor, _cosine_factor])
    warmup_end = max(1, int(epochs * args.warmup))
    rng = np.random.default_rng(args.seed + 1)
    rows = []
    x_lo, x_hi = float(data.x_grid[0]), float(data.x_grid[-1])
    t_lo, t_hi = float(data.t_grid[0]), float(data.t_grid[-1])
    for epoch in range(1, epochs + 1):
        w_phys = args.w_phys * min(1.0, epoch / warmup_end)
        model.train()
        opt.zero_grad()
        loss_data = F.mse_loss(model(x_obs, t_obs), c_obs)
        loss_ic = F.mse_loss(model(x_ic, t_ic), c_ic)
        x_col = torch.tensor(rng.uniform(x_lo, x_hi, size=(args.n_f, 1)), dtype=torch.float32, device=device)
        t_col = torch.tensor(rng.uniform(t_lo, t_hi, size=(args.n_f, 1)), dtype=torch.float32, device=device)
        M_eff = torch.diag(torch.exp(log_M)) * t_scale
        res = _fick_residual_trainable_M(mod, model, x_col, t_col, M_eff, theta_int_t, args.mu_floor)
        loss_phys = torch.mean(res ** 2)
        loss = args.w_data * loss_data + args.w_ic * loss_ic + w_phys * loss_phys
        if not torch.isfinite(loss):
            print(f"  [inverse] non-finite loss at epoch {epoch}; stopping")
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net_params, max_norm=10.0)
        opt.step()
        sched.step()
        M_now = torch.exp(log_M).detach().cpu().numpy()
        rows.append({"epoch": epoch, "loss": float(loss), "data": float(loss_data), "ic": float(loss_ic),
                     "physics": float(loss_phys), "M_Ni": float(M_now[0]), "M_Ta": float(M_now[1])})
        if epoch % max(1, epochs // 10) == 0 or epoch == 1:
            print(f"  [inverse ep={epoch:5d}] loss={float(loss):.3e} data={float(loss_data):.3e} "
                  f"phys={float(loss_phys):.3e} | M_Ni={M_now[0]:.3e} M_Ta={M_now[1]:.3e} "
                  f"(true {M_true_diag[0]:.3e} {M_true_diag[1]:.3e})")
    hist = pd.DataFrame(rows)
    M_hat = np.exp(log_M.detach().cpu().numpy())
    return model, hist, M_hat, M_init


def _pinn_rate_terms(mod, model, x_np: np.ndarray, t_np: np.ndarray, omega_true: np.ndarray, mu_floor: float):
    """Autograd terms of the frozen-coefficient RS PDE on a trained PINN.

    Returns (c_t, a, cmin) as numpy arrays in normalized time τ:
      c_t (N, 2) = ∂c_k/∂τ,  a (N, 2) with a_j = Σ_m Φ_jm(c) ∂²c_m/∂x²  (internal order Ni, Ta),
      cmin (N,)  = min_i c_i at each point.
    """
    import torch
    device = mod.DEVICE
    n = len(x_np)
    x = torch.tensor(np.asarray(x_np, dtype=float).reshape(-1, 1), dtype=torch.float32, device=device, requires_grad=True)
    t = torch.tensor(np.asarray(t_np, dtype=float).reshape(-1, 1), dtype=torch.float32, device=device, requires_grad=True)
    theta_int_t = torch.tensor(mod._reorder_theta_display_to_internal(omega_true), dtype=torch.float32, device=device)
    model.eval()
    C = model(x, t)
    C_int = mod._reorder_display_to_internal(C)
    ones = torch.ones((n, 1), device=device)
    c_t, c_x = [], []
    for k in range(2):
        ck = C_int[:, k:k + 1]
        c_t.append(torch.autograd.grad(ck, t, ones, create_graph=True, retain_graph=True)[0])
        c_x.append(torch.autograd.grad(ck, x, ones, create_graph=True, retain_graph=True)[0])
    c_xx = [torch.autograd.grad(cx, x, ones, retain_graph=True)[0] for cx in c_x]
    Phi = mod.interdiffusion_matrix_rs_torch(C_int.detach(), theta_int_t, None, x=None, RT=_RT, mobility=None, mu_floor=mu_floor)
    a = [Phi[:, j, 0:1] * c_xx[0] + Phi[:, j, 1:2] * c_xx[1] for j in range(2)]
    A = torch.cat(a, dim=1).detach().cpu().numpy()
    B = torch.cat(c_t, dim=1).detach().cpu().numpy()
    cmin = C.detach().cpu().numpy().min(axis=1)
    return B, A, cmin


def estimate_mobility_least_squares(mod, model, data, omega_true: np.ndarray, mu_floor: float,
                                    n_points: int = 6000, seed: int = 0, c_min: float = 0.05, trim: float = 0.05):
    """Post-hoc estimate of M from a trained PINN.

    With Ω known, the RS PDE  ∂c_k/∂t = Σ_j M_kj a_j,  a_j = Σ_m Φ_jm(c) ∂²c_m/∂x²
    (frozen-coefficient form, identical to the PINN residual) is *linear* in M.
    Derivatives are evaluated by autograd on the trained network, so M follows
    from linear least squares.  Points with any c < c_min (where Φ ~ 1/c amplifies
    network errors) and the `trim` fraction of rows with the largest |a| are dropped.
    Returns (M_full (2,2), M_diag (2,), n_used) in physical (un-normalized) time units.
    """
    t_scale = float(data.rs_t_max_physical) if data.rs_t_max_physical > 0 else 1.0
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.02, 0.98, size=n_points)
    t = rng.uniform(float(data.t_start), float(data.t_grid[-1]), size=n_points)
    B, A, cmin = _pinn_rate_terms(mod, model, x, t, omega_true, mu_floor)
    mask = np.isfinite(A).all(axis=1) & np.isfinite(B).all(axis=1) & (cmin > c_min)
    row_norm = np.abs(A).max(axis=1)
    if mask.sum() > 20:
        mask &= row_norm <= np.quantile(row_norm[mask], 1.0 - trim)
    A, B = A[mask], B[mask]
    M_full = np.zeros((2, 2))
    M_diag = np.zeros(2)
    for k in range(2):
        coef, *_ = np.linalg.lstsq(A, B[:, k], rcond=None)
        M_full[k] = coef / t_scale
        M_diag[k] = float(A[:, k] @ B[:, k]) / max(float(A[:, k] @ A[:, k]), 1e-30) / t_scale
    return M_full, M_diag, int(mask.sum())


_PDE_CHECK_CMIN = 0.05


def pde_consistency_check(mod, models: list[tuple[str, str, object, list[tuple[str, np.ndarray]]]], data,
                          omega_true: np.ndarray, mobility_true: np.ndarray, mu_floor: float,
                          n_points: int = 6000, seed: int = 0) -> dict[str, np.ndarray]:
    """How well do the trained networks satisfy the PDE for a given mobility?

    For each (network, M) pair the RMS of the frozen-coefficient residual
    r_k = ∂c_k/∂τ − t_max Σ_j M_kj a_j is evaluated on random collocation points,
    over all points and over the interface region (min_i c_i ≥ _PDE_CHECK_CMIN).
    M = 0 gives the RMS of ∂c/∂τ itself: a network is only PDE-consistent for M if
    RMS(r; M) is clearly smaller than RMS(r; 0).  The same quantities are computed for
    the FDM teacher (finite-difference ∂c/∂t vs. the solver's own flux divergence)
    as a baseline.  A spatial profile of the terms at one intermediate frame is also
    returned for plotting.
    """
    t_scale = float(data.rs_t_max_physical) if data.rs_t_max_physical > 0 else 1.0
    x_grid = np.asarray(data.x_grid, dtype=float)
    t_grid = np.asarray(data.t_grid, dtype=float)
    C_fdm = np.asarray(data.C_fdm, dtype=float)
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=n_points)
    t = rng.uniform(float(data.t_start), float(t_grid[-1]), size=n_points)

    labels: list[str] = []
    rms_rows: list[list[float]] = []

    def _rms(r: np.ndarray, m: np.ndarray) -> list[float]:
        return [float(np.sqrt(np.mean(r[m, k] ** 2))) if m.sum() > 0 else float("nan") for k in range(2)]

    # FDM baseline: c_t by central differences in t, rate from the solver's flux divergence
    theta_int = mod._reorder_theta_display_to_internal(omega_true)
    valid = np.where(t_grid >= float(data.t_start) - 1e-12)[0]
    fr = [i for i in valid if 1 <= i <= len(t_grid) - 2]
    if not fr:  # too few frames after t_start: fall back to every interior frame
        fr = list(range(1, len(t_grid) - 1))
    if not fr:
        raise ValueError(f"PDE-consistency check needs >= 3 saved FDM frames (got {len(t_grid)}); "
                         "decrease --save_every or increase --nsteps")
    ct_f, rate_f, cmin_f = [], [], []
    dx = float(x_grid[1] - x_grid[0])
    for i in fr:
        C_int = mod._reorder_c_display_to_internal_np(C_fdm[i])
        ct_f.append((mod._reorder_c_display_to_internal_np(C_fdm[i + 1])[:, :2]
                     - mod._reorder_c_display_to_internal_np(C_fdm[i - 1])[:, :2]) / (t_grid[i + 1] - t_grid[i - 1]))
        div = mod._rs_compute_div_flux(C_int, x_grid, dx, mobility_true, theta_int, theta_int, _RT,
                                       _X_INTERFACE, _OMEGA_WIDTH, False, None, mu_floor=mu_floor)
        rate_f.append(div * t_scale)
        cmin_f.append(C_fdm[i].min(axis=1))
    ct_f, rate_f, cmin_f = np.concatenate(ct_f), np.concatenate(rate_f), np.concatenate(cmin_f)
    m_all, m_int = np.ones(len(cmin_f), dtype=bool), cmin_f >= _PDE_CHECK_CMIN
    for lab, r in [("FDM 教師 (差分) | M = 真値", ct_f - rate_f), ("FDM 教師 (差分) | M = 0", ct_f)]:
        labels.append(lab)
        rms_rows.append(_rms(r, m_all) + _rms(r, m_int))

    prof: dict[str, np.ndarray] = {}
    i_mid = int(fr[len(fr) // 2])
    prof["pde_prof_frame"] = np.array(i_mid)
    prof["pde_prof_ct_fdm"] = ct_f.reshape(len(fr), len(x_grid), 2)[fr.index(i_mid)]

    for key, name, model, m_list in models:
        B, A, cmin = _pinn_rate_terms(mod, model, x, t, omega_true, mu_floor)
        m_all = np.isfinite(A).all(axis=1) & np.isfinite(B).all(axis=1)
        m_int = m_all & (cmin >= _PDE_CHECK_CMIN)
        for m_lab, M in m_list:
            r = B - t_scale * A * np.asarray(M, dtype=float)[None, :]
            labels.append(f"{name} | M = {m_lab}")
            rms_rows.append(_rms(r, m_all) + _rms(r, m_int))
        labels.append(f"{name} | M = 0")
        rms_rows.append(_rms(B, m_all) + _rms(B, m_int))
        Bp, Ap, _ = _pinn_rate_terms(mod, model, x_grid, np.full_like(x_grid, t_grid[i_mid]), omega_true, mu_floor)
        prof[f"pde_prof_ct_{key}"] = Bp
        prof[f"pde_prof_rate_{key}"] = t_scale * Ap * np.asarray(m_list[0][1], dtype=float)[None, :]
    out = {"pde_check_labels": np.array(labels), "pde_check_rms": np.array(rms_rows, dtype=float)}
    out.update(prof)
    return out


def _nll_pseudo_exp(C_model: np.ndarray, x_grid: np.ndarray, res_exp: dict, sigma: float) -> float:
    """Gaussian NLL (constant omitted) of the pseudo-experimental points under C_model."""
    x_exp, c_exp, idx = res_exp["x"], res_exp["c"], res_exp["frame"]
    nll = 0.0
    for ti in np.unique(idx):
        sel = idx == ti
        for j in range(3):
            pred = np.interp(x_exp[sel], x_grid, C_model[ti, :, j])
            nll += float(np.sum((c_exp[sel, j] - pred) ** 2))
    return nll / (2.0 * sigma ** 2)


def fdm_nll_scan(mod, args: argparse.Namespace, omega_true: np.ndarray, M_true_diag: np.ndarray,
                 x_grid: np.ndarray, res_exp: dict):
    """2-D likelihood map over (log10 M_Ni, log10 M_Ta) using repeated FDM runs."""
    n = int(args.nll_grid)
    h = float(args.nll_half_decades)
    lg_ni = np.log10(M_true_diag[0]) + np.linspace(-h, h, n)
    lg_ta = np.log10(M_true_diag[1]) + np.linspace(-h, h, n)
    Z = np.full((n, n), np.nan)
    t0 = time.time()
    for iy, lt in enumerate(lg_ta):
        for ix, ln in enumerate(lg_ni):
            M = np.diag([10.0 ** ln, 10.0 ** lt])
            with contextlib.redirect_stdout(io.StringIO()):
                _, C, _ = _fdm_forward(mod, args, omega_true, M)
            Z[iy, ix] = _nll_pseudo_exp(C, x_grid, res_exp, args.noise)
        print(f"  [nll] row {iy + 1}/{n} (log10 M_Ta={lt:+.3f}) done, elapsed {time.time() - t0:.0f}s")
    iy, ix = np.unravel_index(np.nanargmin(Z), Z.shape)
    M_hat = np.array([10.0 ** lg_ni[ix], 10.0 ** lg_ta[iy]])
    return lg_ni, lg_ta, Z, M_hat


def _dtilde_profile(mod, C_disp: np.ndarray, omega_true: np.ndarray, mobility: np.ndarray, mu_floor: float) -> np.ndarray:
    """Interdiffusion matrix D̃(c) (Nx, 2, 2) along a composition profile (display order input)."""
    import torch
    C_int = torch.tensor(mod._reorder_c_display_to_internal_np(C_disp), dtype=torch.float32)
    theta = torch.tensor(mod._reorder_theta_display_to_internal(omega_true), dtype=torch.float32)
    with torch.no_grad():
        D = mod.interdiffusion_matrix_rs_torch(C_int, theta, None, x=None, RT=_RT,
                                               mobility=torch.tensor(mobility, dtype=torch.float32), mu_floor=mu_floor)
    return D.numpy()


def run_computation(args: argparse.Namespace) -> dict[str, np.ndarray]:
    n_threads = args.threads if args.threads > 0 else (os.cpu_count() or 1)
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    mod = load_core_module()
    import torch
    torch.set_num_threads(n_threads)

    device_str = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available; using CPU")
        device_str = "cpu"
    mod._set_device(device_str)
    device = mod.DEVICE
    print(f"[device] {device}  ({os.cpu_count()} CPU cores, torch threads={torch.get_num_threads()})")

    omega_true = np.asarray(args.omega, dtype=float)
    mobility = _mobility_from_args(args)
    M_true_diag = np.diag(mobility).copy()
    use_fick = not args.onsager

    print("=" * 72)
    print("Step 1: RS FDM forward problem (teacher + pseudo-experimental data)")
    print("=" * 72)
    mod.set_seed(args.seed)
    t0 = time.time()
    data = mod.make_training_data_rs(
        theta_left=omega_true, theta_right=omega_true,
        mobility=mobility, RT=1.0, x_interface=0.5,
        omega_width=0.02, phase_width=0.02,
        dt=args.dt, nsteps=args.nsteps, save_every=args.save_every, nx_fdm=201,
        n_obs=args.n_obs, n_ic=200, n_bc_each=0, n_f=args.n_f,
        noise=args.noise, seed=args.seed, t_start_fraction=0.1,
        n_exp_points=args.n_exp_points,
        pseudo_exp_time_mode="multi-time",
        pseudo_exp_time_slices=args.n_time_slices,
        append_pseudo_exp_to_training=True,
        noise_model="gaussian", mu_floor=args.mu_floor,
    )
    t_fdm = time.time() - t0
    t_max_phys = float(data.rs_t_max_physical) if data.rs_t_max_physical > 0 else 1.0
    print(f"  done in {t_fdm:.1f}s | frames={len(data.t_grid)} | t_max_physical={t_max_phys:.5f} | "
          f"pseudo-exp slices={len(data.exp_time_indices)} x {args.n_exp_points} pts")

    print("=" * 72)
    print(f"Step 2: PINN training ({args.epochs} epochs, {'Fick' if use_fick else 'Onsager'} form)")
    print("=" * 72)
    mod.set_seed(args.seed)
    model = mod.TernaryRegularSolutionPINN(
        width=64, depth=4, activation="tanh",
        theta_left_init=omega_true, theta_right_init=None,
        learn_left_right_omega=False, x_interface=0.5,
        omega_width=0.02, RT=1.0, train_omega=True,
        direct_output=True, n_time_fourier=4,
        mu_floor=args.mu_floor, use_fick_form=use_fick,
    )
    weights = {"data": args.w_data, "ic": args.w_ic, "bc": 0.0, "phys": args.w_phys}
    t0 = time.time()
    model, history_df, _ = mod.train_pinn_rs(
        data=data, model=model, mobility=mobility,
        epochs=args.epochs, lr=args.lr, weights=weights,
        n_collocation=args.n_f, adaptive_weights=False,
        phys_warmup_fraction=args.warmup,
    )
    t_pinn = time.time() - t0
    print(f"  done in {t_pinn:.1f}s")
    torch.save(model.state_dict(), os.path.join(args.outdir, "pinn_forward_state.pt"))

    model.eval()
    x_grid = np.asarray(data.x_grid, dtype=float)
    t_grid = np.asarray(data.t_grid, dtype=float)
    C_pinn = np.zeros_like(data.C_fdm)
    with torch.no_grad():
        x_t = torch.tensor(x_grid, dtype=torch.float32, device=device).unsqueeze(1)
        for ti, t_val in enumerate(t_grid):
            C_pinn[ti] = model(x_t, torch.full_like(x_t, float(t_val))).cpu().numpy()
        theta_l, _ = model.theta_display()
    omega_learned = np.asarray(theta_l, dtype=float)

    hist_cols = list(history_df.columns)
    C_fdm = np.asarray(data.C_fdm, dtype=float)
    exp_idx = np.asarray(data.exp_time_indices, dtype=int)
    x_exp_all = np.asarray(data.x_exp_all, dtype=float).reshape(-1)
    c_exp_all = np.asarray(data.c_exp_all, dtype=float)
    exp_frame = np.repeat(exp_idx, len(x_exp_all) // len(exp_idx))
    res: dict[str, np.ndarray] = {
        "x_grid": x_grid,
        "t_grid": t_grid,
        "t_max_physical": np.array(t_max_phys),
        "t_start": np.array(float(data.t_start)),
        "C_fdm": C_fdm,
        "C_pinn": C_pinn,
        "x_exp_all": x_exp_all,
        "t_exp_all": np.asarray(data.t_exp_all, dtype=float).reshape(-1),
        "c_exp_all": c_exp_all,
        "exp_time_indices": exp_idx,
        "exp_frame": exp_frame,
        "omega_true": omega_true,
        "omega_learned": omega_learned,
        "history": history_df.to_numpy(dtype=float),
        "history_columns": np.array(hist_cols),
        "M_true": M_true_diag,
        "D_true_final": _dtilde_profile(mod, C_fdm[-1], omega_true, mobility, args.mu_floor),
        "config": np.array(json.dumps({**vars(args), "device": str(device), "use_fick_form": use_fick})),
    }
    timing = {"fdm": t_fdm, "pinn": t_pinn}

    if args.skip_inverse:
        res["timing"] = np.array([t_fdm, t_pinn])
        return res

    print("=" * 72)
    print("Step 3: Inverse analysis of the mobility (diffusion coefficient), Ω known")
    print("=" * 72)
    # Self-check: our stand-alone FDM wrapper must reproduce the teacher data exactly.
    with contextlib.redirect_stdout(io.StringIO()):
        _, C_chk, _ = _fdm_forward(mod, args, omega_true, mobility)
    fdm_repro = float(np.max(np.abs(C_chk - C_fdm)))
    print(f"  [check] stand-alone FDM reproduces teacher data: max|Δc| = {fdm_repro:.2e}")

    inv_epochs = args.inverse_epochs if args.inverse_epochs > 0 else args.epochs
    print(f"  3a: joint PINN inverse ({inv_epochs} epochs, trainable log M, init = {args.m_init_factor} x true)")
    t0 = time.time()
    model_inv, inv_hist, M_hat_pinn, M_init = train_inverse_pinn(mod, data, args, omega_true, M_true_diag, inv_epochs)
    timing["inverse_pinn"] = time.time() - t0
    print(f"  done in {timing['inverse_pinn']:.1f}s | M_hat = {M_hat_pinn} (true {M_true_diag})")
    torch.save(model_inv.state_dict(), os.path.join(args.outdir, "pinn_inverse_state.pt"))
    model_inv.eval()
    C_pinn_inv = np.zeros_like(C_fdm)
    with torch.no_grad():
        x_t = torch.tensor(x_grid, dtype=torch.float32, device=device).unsqueeze(1)
        for ti, t_val in enumerate(t_grid):
            C_pinn_inv[ti] = model_inv(x_t, torch.full_like(x_t, float(t_val))).cpu().numpy()

    print("  3b: post-hoc least-squares estimate of M from PINN derivatives")
    M_ls_fwd_full, M_ls_fwd_diag, n_fwd = estimate_mobility_least_squares(mod, model, data, omega_true, args.mu_floor, seed=args.seed)
    M_ls_inv_full, M_ls_inv_diag, n_inv = estimate_mobility_least_squares(mod, model_inv, data, omega_true, args.mu_floor, seed=args.seed)
    print(f"      forward-fit PINN : diag {M_ls_fwd_diag}  full {M_ls_fwd_full.tolist()}  (n={n_fwd})")
    print(f"      inverse PINN     : diag {M_ls_inv_diag}  full {M_ls_inv_full.tolist()}  (n={n_inv})")

    res_exp = {"x": x_exp_all, "c": c_exp_all, "frame": exp_frame}
    if args.nll_grid > 0:
        print(f"  3c: FDM likelihood map ({args.nll_grid} x {args.nll_grid} runs)")
        t0 = time.time()
        lg_ni, lg_ta, Z, M_hat_nll = fdm_nll_scan(mod, args, omega_true, M_true_diag, x_grid, res_exp)
        timing["nll_scan"] = time.time() - t0
        print(f"  done in {timing['nll_scan']:.1f}s | argmin M = {M_hat_nll}")
        res.update({"nll_log10_M_Ni": lg_ni, "nll_log10_M_Ta": lg_ta, "nll_Z": Z, "M_hat_nll": M_hat_nll})

    print("  3e: PDE-consistency check of the trained networks (RMS residual vs. M)")
    pde_chk = pde_consistency_check(
        mod,
        [("fwd", "順フィット PINN", model, [("真値", M_true_diag), ("初期推定値", M_init)]),
         ("inv", "逆解析 PINN", model_inv, [("M̂ (3a)", M_hat_pinn), ("真値", M_true_diag)])],
        data, omega_true, mobility, args.mu_floor, seed=args.seed,
    )
    for lab, row in zip(pde_chk["pde_check_labels"], pde_chk["pde_check_rms"]):
        print(f"      {str(lab):34s} RMS r (all) Ni={row[0]:.3e} Ta={row[1]:.3e} | (c_min>={_PDE_CHECK_CMIN}) Ni={row[2]:.3e} Ta={row[3]:.3e}")

    print("  3d: forward FDM check with the PINN-estimated mobility")
    with contextlib.redirect_stdout(io.StringIO()):
        _, C_fdm_hat, _ = _fdm_forward(mod, args, omega_true, np.diag(M_hat_pinn))
        C_fdm_init = _fdm_forward(mod, args, omega_true, np.diag(M_init))[1]
    nll_true = _nll_pseudo_exp(C_fdm, x_grid, res_exp, args.noise)
    nll_hat = _nll_pseudo_exp(C_fdm_hat, x_grid, res_exp, args.noise)
    nll_init = _nll_pseudo_exp(C_fdm_init, x_grid, res_exp, args.noise)
    print(f"      NLL(pseudo-exp): true M = {nll_true:.2f}, M_hat = {nll_hat:.2f}, M_init = {nll_init:.2f}")

    res.update({
        "fdm_repro_maxdiff": np.array(fdm_repro),
        "M_init": M_init,
        "M_hat_pinn": M_hat_pinn,
        "M_ls_forward_full": M_ls_fwd_full, "M_ls_forward_diag": M_ls_fwd_diag,
        "M_ls_inverse_full": M_ls_inv_full, "M_ls_inverse_diag": M_ls_inv_diag,
        "M_ls_n_points": np.array([n_fwd, n_inv]),
        "C_pinn_inv": C_pinn_inv,
        "C_fdm_hat": C_fdm_hat,
        "D_hat_final": _dtilde_profile(mod, C_fdm[-1], omega_true, np.diag(M_hat_pinn), args.mu_floor),
        "nll_true_hat_init": np.array([nll_true, nll_hat, nll_init]),
        "inverse_history": inv_hist.to_numpy(dtype=float),
        "inverse_history_columns": np.array(list(inv_hist.columns)),
        "timing_labels": np.array(list(timing.keys())),
        "timing": np.array(list(timing.values())),
        **pde_chk,
    })
    return res


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def save_results(outdir: str, res: dict[str, np.ndarray]) -> None:
    import pandas as pd

    np.savez_compressed(os.path.join(outdir, "timeseries_results.npz"), **res)

    x, t = res["x_grid"], res["t_grid"]
    cfg = json.loads(str(res["config"]))
    t_h = t * cfg["annealing_time_h"]
    dist = (x - 0.5) * cfg["span_um"]
    T, X = np.meshgrid(t, x, indexing="ij")
    TH, D = np.meshgrid(t_h, dist, indexing="ij")
    df = pd.DataFrame({
        "tau": T.ravel(), "time_h": TH.ravel(), "t_physical_rs": (T * float(res["t_max_physical"])).ravel(),
        "x": X.ravel(), "distance_um": D.ravel(),
    })
    for j, comp in enumerate(COMPONENTS):
        df[f"c_{comp}_FDM"] = res["C_fdm"][:, :, j].ravel()
        df[f"c_{comp}_PINN"] = res["C_pinn"][:, :, j].ravel()
    df.to_csv(os.path.join(outdir, "timeseries_profiles.csv"), index=False, float_format="%.6g")

    exp = pd.DataFrame({
        "tau": res["t_exp_all"], "time_h": res["t_exp_all"] * cfg["annealing_time_h"],
        "x": res["x_exp_all"], "distance_um": (res["x_exp_all"] - 0.5) * cfg["span_um"],
    })
    for j, comp in enumerate(COMPONENTS):
        exp[f"c_{comp}_pseudo_exp"] = res["c_exp_all"][:, j]
    exp.to_csv(os.path.join(outdir, "pseudo_exp_points.csv"), index=False, float_format="%.6g")

    hist = pd.DataFrame(res["history"], columns=[str(c) for c in res["history_columns"]])
    hist.to_csv(os.path.join(outdir, "training_history.csv"), index=False)

    if "inverse_history" in res:
        inv = pd.DataFrame(res["inverse_history"], columns=[str(c) for c in res["inverse_history_columns"]])
        inv.to_csv(os.path.join(outdir, "inverse_history.csv"), index=False)
    if "nll_Z" in res:
        LN, LT = np.meshgrid(res["nll_log10_M_Ni"], res["nll_log10_M_Ta"], indexing="xy")
        pd.DataFrame({"log10_M_Ni": LN.ravel(), "log10_M_Ta": LT.ravel(), "M_Ni": 10.0 ** LN.ravel(),
                      "M_Ta": 10.0 ** LT.ravel(), "NLL": res["nll_Z"].ravel()}).to_csv(
            os.path.join(outdir, "nll_grid.csv"), index=False, float_format="%.6g")


def load_results(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(res: dict[str, np.ndarray]) -> dict[str, object]:
    C_fdm, C_pinn, t = res["C_fdm"], res["C_pinn"], res["t_grid"]
    t_start = float(res["t_start"])
    valid = np.where(t >= t_start - 1e-12)[0]
    i0, i1 = int(valid[0]), int(valid[-1])
    fdm_change = float(np.max(np.abs(C_fdm[i1] - C_fdm[i0])))
    pinn_change = float(np.max(np.abs(C_pinn[i1] - C_pinn[i0])))
    rmse_t = np.sqrt(np.mean((C_fdm - C_pinn) ** 2, axis=(1, 2)))
    rmse_comp = np.sqrt(np.mean((C_fdm[valid] - C_pinn[valid]) ** 2, axis=(0, 1)))
    return {
        "finite_fdm": bool(np.all(np.isfinite(C_fdm))),
        "finite_pinn": bool(np.all(np.isfinite(C_pinn))),
        "fdm_sum_dev_max": float(np.max(np.abs(C_fdm.sum(axis=2) - 1.0))),
        "pinn_sum_dev_max": float(np.max(np.abs(C_pinn.sum(axis=2) - 1.0))),
        "fdm_min": float(C_fdm.min()), "fdm_max": float(C_fdm.max()),
        "pinn_min": float(C_pinn.min()), "pinn_max": float(C_pinn.max()),
        "fdm_temporal_change": fdm_change,
        "pinn_temporal_change": pinn_change,
        "temporal_ratio": pinn_change / max(fdm_change, 1e-12),
        "rmse_per_time": rmse_t,
        "rmse_per_component": rmse_comp,
        "rmse_valid_mean": float(np.mean(rmse_t[valid])),
        "valid_indices": valid,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": FONTSIZE,
        "axes.titlesize": FONTSIZE,
        "axes.labelsize": FONTSIZE,
        "xtick.labelsize": FONTSIZE - 3,
        "ytick.labelsize": FONTSIZE - 3,
        "legend.fontsize": FONTSIZE - 4,
        "figure.titlesize": FONTSIZE + 2,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "savefig.facecolor": "white",
    })
    return plt


def _time_label(tau: float, hours: float) -> str:
    return f"$t$ = {hours:.1f} h  ($\\tau$ = {tau:.3f})"


def _panel_indices(res: dict[str, np.ndarray], n_panels: int) -> list[int]:
    exp_idx = sorted({int(i) for i in res["exp_time_indices"]})
    if len(exp_idx) >= 2:
        return exp_idx
    t = res["t_grid"]
    valid = np.where(t >= float(res["t_start"]) - 1e-12)[0]
    return sorted(set(np.linspace(valid[0], valid[-1], n_panels).astype(int).tolist()))


def fig_timeseries_panels(res, cfg, outdir, plt) -> str:
    x, t = res["x_grid"], res["t_grid"]
    dist = (x - 0.5) * cfg["span_um"]
    idx = _panel_indices(res, cfg["n_time_slices"])
    n_cols = 4 if len(idx) > 6 else 3
    n_rows = int(np.ceil(len(idx) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.0 * n_cols, 5.0 * n_rows), squeeze=False, sharex=True, sharey=True)
    exp_t = res["t_exp_all"]
    for k, ti in enumerate(idx):
        ax = axes[k // n_cols][k % n_cols]
        sel = np.isclose(exp_t, t[ti], atol=1e-9)
        for j, comp in enumerate(COMPONENTS):
            if sel.any():
                ax.plot((res["x_exp_all"][sel] - 0.5) * cfg["span_um"], res["c_exp_all"][sel, j],
                        COMP_MARKERS[comp], color=COMP_COLORS[comp], markersize=7, markerfacecolor="none",
                        markeredgewidth=1.6, zorder=3, label=f"FDM pseudo-exp. {comp}" if k == 0 else None)
            ax.plot(dist, res["C_fdm"][ti, :, j], "-", color=COMP_COLORS[comp], lw=2.6,
                    label=f"FDM {comp}" if k == 0 else None)
            ax.plot(dist, res["C_pinn"][ti, :, j], "--", color=COMP_COLORS[comp], lw=2.2,
                    label=f"PINN {comp}" if k == 0 else None)
        ax.set_title(_time_label(float(t[ti]), float(t[ti]) * cfg["annealing_time_h"]), fontsize=FONTSIZE - 2)
        ax.set_ylim(-0.04, 1.04)
        if k // n_cols == n_rows - 1:
            ax.set_xlabel(r"Distance ($\mu$m)")
        if k % n_cols == 0:
            ax.set_ylabel("Mole fraction")
    for k in range(len(idx), n_rows * n_cols):
        axes[k // n_cols][k % n_cols].set_visible(False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    order = [labels.index(f"{kind} {c}") for c in COMPONENTS for kind in ("FDM", "PINN", "FDM pseudo-exp.") if f"{kind} {c}" in labels]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.01), frameon=True)
    form = "Fick form" if cfg["use_fick_form"] else "Onsager form"
    fig.suptitle(f"Co–Ni–Ta diffusion couple: FDM (solid), PINN fit (dashed), FDM pseudo-experimental points (markers) — "
                 f"RS chemical potential, {form}, {cfg['epochs']} epochs", y=1.0)
    fig.tight_layout(rect=[0, 0.10, 1, 0.97])
    path = os.path.join(outdir, "fig_timeseries_panels.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_timeseries_by_component(res, cfg, outdir, plt) -> str:
    import matplotlib as mpl
    x, t = res["x_grid"], res["t_grid"]
    dist = (x - 0.5) * cfg["span_um"]
    idx = _panel_indices(res, cfg["n_time_slices"])
    hours = t * cfg["annealing_time_h"]
    cmap = plt.get_cmap("viridis")
    norm = mpl.colors.Normalize(vmin=hours[idx[0]], vmax=hours[idx[-1]])
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5), sharex=True)
    for j, comp in enumerate(COMPONENTS):
        ax = axes[j]
        for ti in idx:
            col = cmap(norm(hours[ti]))
            ax.plot(dist, res["C_fdm"][ti, :, j], "-", color=col, lw=2.6)
            ax.plot(dist, res["C_pinn"][ti, :, j], "--", color=col, lw=2.2)
        ax.set_title(COMP_LATEX[comp])
        ax.set_xlabel(r"Distance ($\mu$m)")
        if j == 0:
            ax.set_ylabel("Mole fraction")
        ax.set_ylim(-0.04, 1.04)
    from matplotlib.lines import Line2D
    style_handles = [Line2D([], [], color="gray", ls="-", lw=2.6, label="FDM"),
                     Line2D([], [], color="gray", ls="--", lw=2.2, label="PINN")]
    axes[0].legend(handles=style_handles, loc="center right")
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.025, pad=0.02)
    cbar.set_label("Annealing time $t$ (h)")
    fig.suptitle("Time evolution of concentration profiles: FDM (solid) vs PINN fit (dashed)", y=1.02)
    path = os.path.join(outdir, "fig_timeseries_by_component.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_spacetime_heatmap(res, cfg, outdir, plt) -> str:
    x, t = res["x_grid"], res["t_grid"]
    valid = np.where(t >= float(res["t_start"]) - 1e-12)[0]
    dist = (x - 0.5) * cfg["span_um"]
    hours = t[valid] * cfg["annealing_time_h"]
    C_fdm, C_pinn = res["C_fdm"][valid], res["C_pinn"][valid]
    diff = C_pinn - C_fdm
    vmax_d = max(float(np.abs(diff).max()), 1e-6)
    fig, axes = plt.subplots(3, 3, figsize=(21, 15), sharex=True, sharey=True)
    for j, comp in enumerate(COMPONENTS):
        for k, (Z, title, cm, vmin, vmax) in enumerate([
            (C_fdm[:, :, j], f"FDM {COMP_LATEX[comp]}", "viridis", 0.0, 1.0),
            (C_pinn[:, :, j], f"PINN {COMP_LATEX[comp]}", "viridis", 0.0, 1.0),
            (diff[:, :, j], f"PINN $-$ FDM ({comp})", "RdBu_r", -vmax_d, vmax_d),
        ]):
            ax = axes[j][k]
            im = ax.pcolormesh(dist, hours, Z, cmap=cm, vmin=vmin, vmax=vmax, shading="auto")
            ax.set_title(title, fontsize=FONTSIZE - 1)
            ax.grid(False)
            if j == 2:
                ax.set_xlabel(r"Distance ($\mu$m)")
            if k == 0:
                ax.set_ylabel("Annealing time $t$ (h)")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle("Space–time concentration maps (rows: Co, Ni, Ta; columns: FDM, PINN, difference)", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    path = os.path.join(outdir, "fig_spacetime_heatmap.png")
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_fixed_position_timeseries(res, cfg, outdir, plt) -> str:
    x, t = res["x_grid"], res["t_grid"]
    valid = np.where(t >= float(res["t_start"]) - 1e-12)[0]
    hours = t[valid] * cfg["annealing_time_h"]
    probe_x = [0.40, 0.45, 0.50, 0.55, 0.60]
    cmap = plt.get_cmap("plasma")
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5), sharex=True)
    for j, comp in enumerate(COMPONENTS):
        ax = axes[j]
        for q, xp in enumerate(probe_x):
            xi = int(np.argmin(np.abs(x - xp)))
            col = cmap(q / max(len(probe_x) - 1, 1))
            d_um = (x[xi] - 0.5) * cfg["span_um"]
            ax.plot(hours, res["C_fdm"][valid, xi, j], "-", color=col, lw=2.6, label=f"FDM  $x$ = {d_um:+.0f} $\\mu$m")
            ax.plot(hours, res["C_pinn"][valid, xi, j], "--", color=col, lw=2.2, label=f"PINN $x$ = {d_um:+.0f} $\\mu$m")
        ax.set_title(COMP_LATEX[comp])
        ax.set_xlabel("Annealing time $t$ (h)")
        if j == 0:
            ax.set_ylabel("Mole fraction")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, -0.02), fontsize=FONTSIZE - 6)
    fig.suptitle("Concentration vs. time at fixed positions (0 $\\mu$m = initial interface)", y=1.02)
    path = os.path.join(outdir, "fig_fixed_position_timeseries.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_loss_history(res, cfg, outdir, plt) -> tuple[str | None, str | None]:
    import pandas as pd
    hist = pd.DataFrame(res["history"], columns=[str(c) for c in res["history_columns"]])
    if len(hist) == 0:
        return None, None
    fig, ax = plt.subplots(figsize=(12, 6))
    for key, label in [("data", "data"), ("ic", "initial condition"), ("physics", "physics (PDE)"), ("loss", "total")]:
        if key in hist.columns:
            v = hist[key].to_numpy(dtype=float)
            v = np.where(np.isfinite(v) & (v > 0), v, np.nan)
            ax.semilogy(hist["epoch"], v, lw=2.2, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("PINN training loss history")
    ax.legend()
    fig.tight_layout()
    p_loss = os.path.join(outdir, "fig_loss_history.png")
    fig.savefig(p_loss, dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    for k, key in enumerate(["Omega_CoNi_left", "Omega_CoTa_left", "Omega_NiTa_left"]):
        if key in hist.columns:
            ax.plot(hist["epoch"], hist[key], lw=2.2, label=OMEGA_LABELS[k])
            ax.axhline(float(res["omega_true"][k]), color="gray", ls=":", lw=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$\Omega$ (normalized by $RT$)")
    ax.set_title(r"Learned $\Omega$ history (dotted: true values)")
    ax.legend()
    fig.tight_layout()
    p_omega = os.path.join(outdir, "fig_omega_history.png")
    fig.savefig(p_omega, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p_loss, p_omega


def fig_pinn_schematic(outdir, plt) -> str:
    """Static schematic of a PINN (inputs → network → outputs → autograd → losses)."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig, ax = plt.subplots(figsize=(20, 8.5))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 8.5)
    ax.axis("off")
    ax.grid(False)

    def box(x, y, w, h, text, fc, fs=FONTSIZE - 2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15", fc=fc, ec="black", lw=1.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1, text=None, fs=FONTSIZE - 5):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=28, lw=2.0, color="black"))
        if text:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.28, text, ha="center", va="bottom", fontsize=fs)

    box(0.3, 3.3, 2.6, 1.9, "Input\n$(x,\\ t)$", "#fde9c9")
    box(4.2, 2.6, 4.2, 3.3, "Neural network\n$\\mathcal{N}_{\\mathbf{w}}(x,t)$\n(4 layers $\\times$ 64, tanh)", "#d9e8f5")
    box(9.8, 3.3, 3.4, 1.9, "Output\n$c_{\\mathrm{Co}},\\ c_{\\mathrm{Ni}},\\ c_{\\mathrm{Ta}}$", "#dff2d8")
    arrow(2.9, 4.25, 4.2, 4.25)
    arrow(8.4, 4.25, 9.8, 4.25)

    box(14.6, 6.5, 5.1, 1.5, "Data loss  $\\mathcal{L}_{\\mathrm{data}}$\n$\\langle (c_{\\mathcal{N}} - c_{\\mathrm{obs}})^2 \\rangle$", "#f4d6d6", FONTSIZE - 4)
    box(14.6, 4.3, 5.1, 1.5, "Initial-condition loss  $\\mathcal{L}_{\\mathrm{ic}}$", "#f4d6d6", FONTSIZE - 4)
    box(14.6, 1.2, 5.1, 2.4, "Physics loss  $\\mathcal{L}_{\\mathrm{phys}}$\n$\\langle r^2 \\rangle$,  $r = \\partial_t c - \\partial_x[\\tilde{D}\\,\\partial_x c]$",
        "#f4d6d6", FONTSIZE - 4)
    arrow(13.2, 4.9, 14.6, 7.0)
    arrow(13.2, 4.4, 14.6, 5.0)
    box(9.8, 0.6, 3.4, 1.7, "Autograd\n$\\partial_t c,\\ \\partial_x c,\\ \\partial_{xx} c$", "#efe3f7", FONTSIZE - 4)
    arrow(11.5, 3.3, 11.5, 2.3)
    arrow(13.2, 1.5, 14.6, 2.0)
    ax.text(4.6, 1.0, "Trainable: network weights $\\mathbf{w}$\n(+ unknown physical parameters,\ne.g. mobility $M$, for inverse problems)",
            ha="center", va="center", fontsize=FONTSIZE - 5, style="italic")
    ax.text(17.15, 0.35, "$\\mathcal{L} = w_{\\mathrm{data}}\\mathcal{L}_{\\mathrm{data}} + w_{\\mathrm{ic}}\\mathcal{L}_{\\mathrm{ic}} + w_{\\mathrm{phys}}\\mathcal{L}_{\\mathrm{phys}}$"
            "  $\\rightarrow$ minimize", ha="center", va="center", fontsize=FONTSIZE - 3)
    path = os.path.join(outdir, "fig_pinn_schematic.png")
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_pseudo_exp_data(res, cfg, outdir, plt) -> str:
    """What the 'experiment' provides: noisy points at a few times (+ the hidden truth as thin lines)."""
    x, t = res["x_grid"], res["t_grid"]
    dist = (x - 0.5) * cfg["span_um"]
    idx = _panel_indices(res, cfg["n_time_slices"])
    n_cols = 4 if len(idx) > 6 else 3
    n_rows = int(np.ceil(len(idx) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.0 * n_cols, 5.0 * n_rows), squeeze=False, sharex=True, sharey=True)
    exp_t = res["t_exp_all"]
    for k, ti in enumerate(idx):
        ax = axes[k // n_cols][k % n_cols]
        sel = np.isclose(exp_t, t[ti], atol=1e-9)
        for j, comp in enumerate(COMPONENTS):
            ax.plot(dist, res["C_fdm"][ti, :, j], "-", color=COMP_COLORS[comp], lw=1.2, alpha=0.5)
            ax.plot((res["x_exp_all"][sel] - 0.5) * cfg["span_um"], res["c_exp_all"][sel, j], COMP_MARKERS[comp],
                    color=COMP_COLORS[comp], markersize=7, markerfacecolor="none", markeredgewidth=1.6,
                    label=f"{comp} (pseudo-exp.)" if k == 0 else None)
        ax.set_title(_time_label(float(t[ti]), float(t[ti]) * cfg["annealing_time_h"]), fontsize=FONTSIZE - 2)
        ax.set_ylim(-0.04, 1.04)
        if k // n_cols == n_rows - 1:
            ax.set_xlabel(r"Distance ($\mu$m)")
        if k % n_cols == 0:
            ax.set_ylabel("Mole fraction")
    for k in range(len(idx), n_rows * n_cols):
        axes[k // n_cols][k % n_cols].set_visible(False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Pseudo-experimental data: FDM profiles + Gaussian noise ($\\sigma$ = {cfg['noise']}), "
                 f"{cfg['n_exp_points']} points per time (thin lines: hidden FDM truth)", y=1.0)
    fig.tight_layout(rect=[0, 0.08, 1, 0.97])
    path = os.path.join(outdir, "fig_pseudo_exp_data.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_inverse_history(res, cfg, outdir, plt) -> str | None:
    import pandas as pd
    if "inverse_history" not in res:
        return None
    hist = pd.DataFrame(res["inverse_history"], columns=[str(c) for c in res["inverse_history_columns"]])
    M_true, M_init = res["M_true"], res["M_init"]
    fig, axes = plt.subplots(1, 3, figsize=(22, 6.5))
    ax = axes[0]
    for key, label in [("data", "data"), ("ic", "initial condition"), ("physics", "physics (PDE)"), ("loss", "total")]:
        v = hist[key].to_numpy(dtype=float)
        ax.semilogy(hist["epoch"], np.where(np.isfinite(v) & (v > 0), v, np.nan), lw=2.2, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Inverse PINN: loss history")
    ax.legend()
    ax = axes[1]
    cols = ["#2b83ba", "#4daf4a"]
    for k, key in enumerate(["M_Ni", "M_Ta"]):
        ax.semilogy(hist["epoch"], hist[key], lw=2.4, color=cols[k], label=M_LABELS[k] + " (PINN)")
        ax.axhline(M_true[k], color=cols[k], ls=":", lw=2.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mobility $M$ (model units)")
    ax.set_title("Trainable mobility (dotted: true)")
    ax.legend()
    ax = axes[2]
    for k, key in enumerate(["M_Ni", "M_Ta"]):
        ax.plot(hist["epoch"], hist[key] / M_true[k], lw=2.4, color=cols[k], label=M_LABELS[k] + r" / $M^{\mathrm{true}}$")
    ax.axhline(1.0, color="gray", ls=":", lw=2.0)
    ax.axhline(M_init[0] / M_true[0], color="gray", ls="--", lw=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Ratio to true value")
    ax.set_title("Convergence ratio (dashed: initial guess)")
    ax.legend()
    fig.suptitle("Inverse analysis with PINN: network weights and $\\log M$ trained jointly ($\\Omega$ known)", y=1.03)
    fig.tight_layout()
    path = os.path.join(outdir, "fig_inverse_history.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_nll_map(res, cfg, outdir, plt) -> str | None:
    if "nll_Z" not in res:
        return None
    lg_ni, lg_ta, Z = res["nll_log10_M_Ni"], res["nll_log10_M_Ta"], res["nll_Z"]
    dZ = Z - np.nanmin(Z)
    M_true, M_pinn = res["M_true"], res["M_hat_pinn"]
    fig, axes = plt.subplots(1, 3, figsize=(23, 7), gridspec_kw={"width_ratios": [1.35, 1, 1]})
    ax = axes[0]
    im = ax.pcolormesh(lg_ni, lg_ta, np.log10(dZ + 1.0), cmap="viridis_r", shading="auto")
    cs = ax.contour(lg_ni, lg_ta, dZ, levels=[1.15, 3.09, 5.91], colors=["white", "orange", "red"], linewidths=2.2)
    ax.clabel(cs, fmt={1.15: r"1$\sigma$", 3.09: r"2$\sigma$", 5.91: r"3$\sigma$"}, fontsize=FONTSIZE - 5)
    ax.plot(np.log10(M_true[0]), np.log10(M_true[1]), "w*", ms=22, mec="black", label="true $M$")
    ax.plot(np.log10(M_pinn[0]), np.log10(M_pinn[1]), "o", color="#ff7f0e", ms=14, mec="black", label="PINN estimate")
    ax.plot(np.log10(res["M_hat_nll"][0]), np.log10(res["M_hat_nll"][1]), "s", color="cyan", ms=12, mec="black", label="FDM grid argmin")
    ax.plot(np.log10(res["M_init"][0]), np.log10(res["M_init"][1]), "x", color="red", ms=14, mew=3, label="PINN initial guess")
    ax.set_xlabel(r"$\log_{10} M_{\mathrm{Ni}}$")
    ax.set_ylabel(r"$\log_{10} M_{\mathrm{Ta}}$")
    ax.set_title(r"FDM likelihood map: $\log_{10}(\Delta\mathrm{NLL}+1)$")
    ax.grid(False)
    ax.legend(loc="upper left", fontsize=FONTSIZE - 6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    iy, ix = np.unravel_index(np.nanargmin(Z), Z.shape)
    for k, (ax, lg, prof, label, m_t, m_p) in enumerate([
        (axes[1], lg_ni, dZ[iy, :], r"$\log_{10} M_{\mathrm{Ni}}$", M_true[0], M_pinn[0]),
        (axes[2], lg_ta, dZ[:, ix], r"$\log_{10} M_{\mathrm{Ta}}$", M_true[1], M_pinn[1]),
    ]):
        ax.plot(lg, prof, "o-", lw=2.4, ms=8, color="#2b83ba")
        for lvl, col, name in [(0.5, "gray", r"1$\sigma$"), (2.0, "orange", r"2$\sigma$")]:
            ax.axhline(lvl, color=col, ls=":", lw=1.8)
            ax.text(lg[0], lvl, name, va="bottom", fontsize=FONTSIZE - 6, color=col)
        ax.axvline(np.log10(m_t), color="black", ls="--", lw=2.0, label="true")
        ax.axvline(np.log10(m_p), color="#ff7f0e", ls="-.", lw=2.0, label="PINN")
        ax.set_xlabel(label)
        ax.set_ylabel(r"$\Delta\mathrm{NLL}$")
        ax.set_ylim(-0.5, min(float(np.nanmax(prof)) * 1.05, 60.0))
        ax.set_title("Profile through the minimum")
        ax.legend()
    fig.suptitle(f"Likelihood of the pseudo-experimental data as a function of the mobility "
                 f"({len(lg_ni)}$\\times${len(lg_ta)} FDM runs; NLL relative to the minimum)", y=1.03)
    fig.tight_layout()
    path = os.path.join(outdir, "fig_nll_map.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_forward_check(res, cfg, outdir, plt) -> str | None:
    if "C_fdm_hat" not in res:
        return None
    x, t = res["x_grid"], res["t_grid"]
    dist = (x - 0.5) * cfg["span_um"]
    idx = _panel_indices(res, cfg["n_time_slices"])
    pick = [idx[0], idx[len(idx) // 3], idx[(2 * len(idx)) // 3], idx[-1]]
    fig, axes = plt.subplots(1, 4, figsize=(24, 6.2), sharey=True)
    exp_t = res["t_exp_all"]
    for k, ti in enumerate(pick):
        ax = axes[k]
        sel = np.isclose(exp_t, t[ti], atol=1e-9)
        for j, comp in enumerate(COMPONENTS):
            ax.plot((res["x_exp_all"][sel] - 0.5) * cfg["span_um"], res["c_exp_all"][sel, j], COMP_MARKERS[comp],
                    color=COMP_COLORS[comp], markersize=7, markerfacecolor="none", markeredgewidth=1.5, zorder=3)
            ax.plot(dist, res["C_fdm"][ti, :, j], "-", color=COMP_COLORS[comp], lw=2.6, label=f"FDM true $M$ ({comp})" if k == 0 else None)
            ax.plot(dist, res["C_fdm_hat"][ti, :, j], "--", color=COMP_COLORS[comp], lw=2.2, label=f"FDM $\\hat{{M}}$ ({comp})" if k == 0 else None)
            ax.plot(dist, res["C_pinn_inv"][ti, :, j], ":", color=COMP_COLORS[comp], lw=2.2, label=f"inverse PINN ({comp})" if k == 0 else None)
        ax.set_title(_time_label(float(t[ti]), float(t[ti]) * cfg["annealing_time_h"]), fontsize=FONTSIZE - 2)
        ax.set_xlabel(r"Distance ($\mu$m)")
        ax.set_ylim(-0.04, 1.04)
    axes[0].set_ylabel("Mole fraction")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.16), fontsize=FONTSIZE - 5)
    rmse = float(np.sqrt(np.mean((res["C_fdm_hat"] - res["C_fdm"]) ** 2)))
    fig.suptitle(f"Forward check: FDM with true $M$ (solid) vs FDM with PINN-estimated $\\hat{{M}}$ (dashed) vs inverse PINN (dotted); "
                 f"markers = pseudo-exp.  RMSE(FDM $\\hat{{M}}$ vs true) = {rmse:.2e}", y=1.02, fontsize=FONTSIZE)
    fig.tight_layout()
    path = os.path.join(outdir, "fig_forward_check.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_pde_consistency(res, cfg, outdir, plt) -> str | None:
    """2 x 2: rows = forward-fit PINN / inverse PINN, cols = Ni / Ta.

    Each panel compares, at one intermediate frame, the FDM time derivative (finite
    differences), the network's autograd ∂c/∂τ and the PDE rate M·a implied by the
    network's second derivatives.  For a PDE-consistent network the last two coincide.
    """
    if "pde_prof_ct_fwd" not in res:
        return None
    x, t = res["x_grid"], res["t_grid"]
    dist = (x - 0.5) * cfg["span_um"]
    ti = int(res["pde_prof_frame"])
    M_true, M_hat = res["M_true"], res["M_hat_pinn"]
    fig, axes = plt.subplots(2, 2, figsize=(20, 11), sharex=True)
    rows = [("fwd", "forward-fit PINN", r"$M_{\mathrm{true}}$", M_true), ("inv", "inverse PINN", r"$\hat{M}$ (3a)", M_hat)]
    for r, (key, name, m_lab, M) in enumerate(rows):
        for k, comp in enumerate(IND_COMPONENTS):
            ax = axes[r, k]
            ax.plot(dist, res["pde_prof_ct_fdm"][:, k], "-", color="k", lw=2.6, label=r"FDM: $\partial c/\partial\tau$ (finite diff.)")
            ax.plot(dist, res[f"pde_prof_ct_{key}"][:, k], "--", color="#d7191c", lw=2.4, label=r"PINN: $\partial c/\partial\tau$ (autograd)")
            rate = res[f"pde_prof_rate_{key}"][:, k]
            ax.plot(dist, rate, ":", color="#2b83ba", lw=2.6, label=rf"PINN: $t_{{\max}}\,M\,a$ with $M$ = {m_lab}")
            ax.axhline(0, color="gray", lw=0.8)
            # the PDE rate can spike by orders of magnitude where c -> 0 (D ~ 1/c); keep the
            # axis on the scale of the time derivatives and let the spike run off the panel
            ref = np.concatenate([res["pde_prof_ct_fdm"][:, k], res[f"pde_prof_ct_{key}"][:, k],
                                  np.percentile(rate, [2, 98])])
            lo, hi = float(ref.min()), float(ref.max())
            pad = 0.15 * (hi - lo + 1e-12)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_title(f"{name}: {COMP_LATEX[comp]}  ($M$ = {M[k]:.2e})", fontsize=FONTSIZE - 2)
            if r == 1:
                ax.set_xlabel(r"Distance ($\mu$m)")
            if k == 0:
                ax.set_ylabel(r"Rate ($\partial c/\partial\tau$)")
            ax.set_xlim(dist[0], dist[-1])
    axes[0, 0].legend(fontsize=FONTSIZE - 7, loc="upper left")
    fig.suptitle("PDE-consistency check at " + _time_label(float(t[ti]), float(t[ti]) * cfg["annealing_time_h"])
                 + ": time derivative vs. PDE rate  (top: forward-fit PINN, bottom: inverse PINN; left: Ni, right: Ta)",
                 y=1.0, fontsize=FONTSIZE - 1)
    fig.tight_layout()
    path = os.path.join(outdir, "fig_pde_consistency.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_dtilde_profiles(res, cfg, outdir, plt) -> str:
    x = res["x_grid"]
    dist = (x - 0.5) * cfg["span_um"]
    D_true = res["D_true_final"]
    D_hat = res.get("D_hat_final")
    C = res["C_fdm"][-1]
    fig, axes = plt.subplots(1, 3, figsize=(23, 6.5))
    ax = axes[0]
    for j, comp in enumerate(COMPONENTS):
        ax.plot(dist, C[:, j], "-", color=COMP_COLORS[comp], lw=2.6, label=COMP_LATEX[comp])
    ax.set_xlabel(r"Distance ($\mu$m)")
    ax.set_ylabel("Mole fraction")
    ax.set_title("Composition at the final time (FDM)")
    ax.legend()
    names = [[r"$\tilde{D}_{\mathrm{NiNi}}$", r"$\tilde{D}_{\mathrm{NiTa}}$"], [r"$\tilde{D}_{\mathrm{TaNi}}$", r"$\tilde{D}_{\mathrm{TaTa}}$"]]
    cols = [["#2b83ba", "#7fbfdf"], ["#4daf4a", "#a6d96a"]]
    for k, ax in enumerate(axes[1:]):
        for m in range(2):
            ax.plot(dist, D_true[:, k, m], "-", color=cols[k][m], lw=2.6, label=names[k][m] + " (true $M$)")
            if D_hat is not None:
                ax.plot(dist, D_hat[:, k, m], "--", color=cols[k][m], lw=2.2, label=names[k][m] + r" ($\hat{M}$)")
        ax.set_xlabel(r"Distance ($\mu$m)")
        ax.set_ylabel(r"$\tilde{D}$ (model units)")
        ax.set_title(f"Interdiffusion coefficients, row {IND_COMPONENTS[k]}")
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.legend(fontsize=FONTSIZE - 6)
    fig.suptitle(r"Composition-dependent interdiffusion matrix $\tilde{D}(c) = M\,\partial(\mu_j-\mu_{\mathrm{Co}})/\partial c_m$ along the final profile", y=1.03)
    fig.tight_layout()
    path = os.path.join(outdir, "fig_dtilde_profiles.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_mobility_summary(res, cfg, outdir, plt) -> str | None:
    if "M_hat_pinn" not in res:
        return None
    M_true = res["M_true"]
    entries = [("initial guess", res["M_init"], "#bbbbbb"),
               ("PINN joint (3a)", res["M_hat_pinn"], "#ff7f0e"),
               ("LS from forward PINN (3b)", res["M_ls_forward_diag"], "#9467bd"),
               ("LS from inverse PINN (3b)", res["M_ls_inverse_diag"], "#8c564b")]
    if "M_hat_nll" in res:
        entries.append(("FDM grid argmin (3c)", res["M_hat_nll"], "#17becf"))
    fig, axes = plt.subplots(1, 2, figsize=(20, 6.5))
    for k, ax in enumerate(axes):
        vals = [float(e[1][k] / M_true[k]) for e in entries]
        x_max = max(1.2, min(max(vals) * 1.15, 6.0))
        ax.barh(range(len(entries)), np.clip(vals, 0, x_max), color=[e[2] for e in entries], edgecolor="black")
        ax.axvline(1.0, color="black", ls="--", lw=2)
        ax.set_yticks(range(len(entries)))
        ax.set_yticklabels([e[0] for e in entries], fontsize=FONTSIZE - 4)
        ax.set_xlabel(M_LABELS[k] + r" estimate / true")
        ax.set_title(f"{M_LABELS[k]}: true = {M_true[k]:.3e}")
        for i, v in enumerate(vals):
            ax.text(min(max(v, 0.0), x_max), i, f" {v:.3f}", va="center", fontsize=FONTSIZE - 5)
        ax.set_xlim(0, x_max * 1.12)
        ax.invert_yaxis()
    fig.suptitle("Mobility estimates from the different inverse methods (ratio to the true value)", y=1.03)
    fig.tight_layout()
    path = os.path.join(outdir, "fig_mobility_summary.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def make_animation(res, cfg, outdir, plt) -> str | None:
    from matplotlib.animation import FuncAnimation, PillowWriter
    x, t = res["x_grid"], res["t_grid"]
    valid = np.where(t >= float(res["t_start"]) - 1e-12)[0]
    dist = (x - 0.5) * cfg["span_um"]
    fig, ax = plt.subplots(figsize=(11, 7))
    lines = {}
    for j, comp in enumerate(COMPONENTS):
        lines[("fdm", j)], = ax.plot(dist, res["C_fdm"][valid[0], :, j], "-", color=COMP_COLORS[comp], lw=2.8, label=f"FDM {comp}")
        lines[("pinn", j)], = ax.plot(dist, res["C_pinn"][valid[0], :, j], "--", color=COMP_COLORS[comp], lw=2.4, label=f"PINN {comp}")
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlabel(r"Distance ($\mu$m)")
    ax.set_ylabel("Mole fraction")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    title = ax.set_title("")
    fig.tight_layout()

    def update(frame):
        ti = valid[frame]
        for j in range(3):
            lines[("fdm", j)].set_ydata(res["C_fdm"][ti, :, j])
            lines[("pinn", j)].set_ydata(res["C_pinn"][ti, :, j])
        title.set_text(_time_label(float(t[ti]), float(t[ti]) * cfg["annealing_time_h"]))
        return list(lines.values()) + [title]

    anim = FuncAnimation(fig, update, frames=len(valid), blit=False)
    path = os.path.join(outdir, "timeseries_animation.gif")
    anim.save(path, writer=PillowWriter(fps=4), dpi=90)
    plt.close(fig)
    return path

# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------
def _ci_from_profile(lg: np.ndarray, prof: np.ndarray, level: float) -> tuple[float, float]:
    """Interval (in log10 M) where ΔNLL <= level, with linear interpolation at the crossings."""
    inside = np.where(prof <= level)[0]
    if len(inside) == 0:
        return float("nan"), float("nan")
    lo_i, hi_i = int(inside[0]), int(inside[-1])
    lo = lg[lo_i]
    if lo_i > 0:
        lo = np.interp(level, [prof[lo_i], prof[lo_i - 1]], [lg[lo_i], lg[lo_i - 1]])
    hi = lg[hi_i]
    if hi_i < len(lg) - 1:
        hi = np.interp(level, [prof[hi_i], prof[hi_i + 1]], [lg[hi_i], lg[hi_i + 1]])
    return float(lo), float(hi)


def _inverse_summary(res) -> dict[str, object]:
    if "M_hat_pinn" not in res:
        return {}
    M_true = res["M_true"]
    rows = [("真値 (FDM 教師に使用)", M_true), ("初期推定値 (逆解析の出発点)", res["M_init"]),
            ("3a  PINN 同時推定 (log M 学習)", res["M_hat_pinn"]),
            ("3b  最小二乗 (順フィット PINN の微分から)", res["M_ls_forward_diag"]),
            ("3b  最小二乗 (逆解析 PINN の微分から)", res["M_ls_inverse_diag"])]
    if "M_hat_nll" in res:
        rows.append(("3c  FDM 尤度マップの最小点 (格子)", res["M_hat_nll"]))
    out: dict[str, object] = {
        "rows": rows,
        "rmse_inv_pinn": float(np.sqrt(np.mean((res["C_pinn_inv"] - res["C_fdm"]) ** 2))),
        "rmse_fdm_hat": float(np.sqrt(np.mean((res["C_fdm_hat"] - res["C_fdm"]) ** 2))),
        "nll_true_hat_init": res["nll_true_hat_init"],
    }
    if "nll_Z" in res:
        Z = res["nll_Z"]
        dZ = Z - np.nanmin(Z)
        iy, ix = np.unravel_index(np.nanargmin(Z), Z.shape)
        out["ci_ni"] = _ci_from_profile(res["nll_log10_M_Ni"], dZ[iy, :], 0.5)
        out["ci_ta"] = _ci_from_profile(res["nll_log10_M_Ta"], dZ[:, ix], 0.5)
    if "pde_check_rms" in res:
        labels = [str(s) for s in res["pde_check_labels"]]
        rms = res["pde_check_rms"]
        try:
            r_true = rms[labels.index("順フィット PINN | M = 真値")]
            r_zero = rms[labels.index("順フィット PINN | M = 0")]
            out["pde_consistent"] = bool(np.all(r_true[2:] < r_zero[2:]))
        except ValueError:
            pass
    return out


def _timing_dict(res) -> dict[str, float]:
    if "timing_labels" in res:
        return {str(k): float(v) for k, v in zip(res["timing_labels"], res["timing"])}
    t = res["timing"] if "timing" in res else np.array([np.nan, np.nan])
    return {"fdm": float(t[0]), "pinn": float(t[1])}


# ---------------------------------------------------------------------------
# Markdown report (short version; the HTML report is the main deliverable)
# ---------------------------------------------------------------------------
def write_report(outdir: str, res, cfg, metrics, figs: dict[str, str | None]) -> str:
    t = res["t_grid"]
    hours = t * cfg["annealing_time_h"]
    rmse_t = metrics["rmse_per_time"]
    valid = metrics["valid_indices"]
    timing = _timing_dict(res)
    inv = _inverse_summary(res)
    form_label = "Fick 形式 D̃(c)·∂c/∂x" if cfg["use_fick_form"] else "Onsager 形式 M·∂μ/∂x"
    exp_set = {int(i) for i in res["exp_time_indices"]}

    def rel(p):
        return os.path.basename(p) if p else None

    lines = [
        "# Co–Ni–Ta 合金拡散: PINN 時系列フィットと拡散係数逆解析",
        "",
        "詳細版 (PINN の解説付き, 全図表埋め込み) は `REPORT.html` を参照。",
        "",
        f"PDE 残差は **{form_label}**。device={cfg['device']}, epochs={cfg['epochs']}, "
        f"w_data/w_ic/w_phys={cfg['w_data']}/{cfg['w_ic']}/{cfg['w_phys']}, noise={cfg['noise']}.",
        "",
        "計算時間: " + ", ".join(f"{k} {v:.1f} s" for k, v in timing.items()),
        "",
        "## 妥当性チェック (順フィット)",
        "",
        "| 項目 | FDM | PINN |",
        "|---|---|---|",
        f"| 有限値 | {metrics['finite_fdm']} | {metrics['finite_pinn']} |",
        f"| 濃度範囲 | [{metrics['fdm_min']:.4f}, {metrics['fdm_max']:.4f}] | [{metrics['pinn_min']:.4f}, {metrics['pinn_max']:.4f}] |",
        f"| max \\|Σc − 1\\| | {metrics['fdm_sum_dev_max']:.2e} | {metrics['pinn_sum_dev_max']:.2e} |",
        f"| 時間変化量 max\\|c(t_end) − c(t_start)\\| | {metrics['fdm_temporal_change']:.4f} | {metrics['pinn_temporal_change']:.4f} (比 {metrics['temporal_ratio']:.3f}) |",
        f"| RMSE (t ≥ t_start 平均) | – | {metrics['rmse_valid_mean']:.3e} |",
        "",
        "成分別 RMSE: " + ", ".join(f"{c} = {v:.3e}" for c, v in zip(COMPONENTS, metrics["rmse_per_component"])),
        "",
        "## 時刻別 RMSE",
        "",
        "| frame | τ | t (h) | RMSE | 疑似実験点 |",
        "|---|---|---|---|---|",
    ]
    for ti in valid:
        lines.append(f"| {ti} | {t[ti]:.3f} | {hours[ti]:.1f} | {rmse_t[ti]:.3e} | {'●' if ti in exp_set else ''} |")
    if inv:
        lines += ["", "## 移動度 (拡散係数) 逆解析", "", "| 推定法 | M_Ni | M_Ta | M_Ni/true | M_Ta/true |", "|---|---|---|---|---|"]
        M_true = res["M_true"]
        for name, M in inv["rows"]:
            lines.append(f"| {name} | {M[0]:.4e} | {M[1]:.4e} | {M[0] / M_true[0]:.3f} | {M[1] / M_true[1]:.3f} |")
        n_t, n_h, n_i = inv["nll_true_hat_init"]
        lines += ["", f"疑似実験データの NLL: 真の M = {n_t:.2f}, PINN 推定 M̂ = {n_h:.2f}, 初期推定値 = {n_i:.2f}",
                  f"FDM(M̂) と FDM(真の M) の RMSE = {inv['rmse_fdm_hat']:.3e}; 逆解析 PINN の RMSE = {inv['rmse_inv_pinn']:.3e}"]
        if "ci_ni" in inv:
            lines += ["", f"FDM 尤度マップの 1σ 区間 (log10): M_Ni ∈ [{inv['ci_ni'][0]:.3f}, {inv['ci_ni'][1]:.3f}], "
                      f"M_Ta ∈ [{inv['ci_ta'][0]:.3f}, {inv['ci_ta'][1]:.3f}] "
                      f"(真値 {np.log10(M_true[0]):.3f}, {np.log10(M_true[1]):.3f})"]
    lines += ["", "## 図", ""]
    for key, title in [("schematic", "PINN の概念図"), ("pseudo_exp", "疑似実験データ"), ("panels", "時刻別パネル"),
                       ("by_component", "成分別時系列"), ("heatmap", "空間–時間マップ"), ("fixed", "固定位置の時間変化"),
                       ("loss", "学習損失"), ("omega", "Ω 履歴"), ("inverse", "逆解析 PINN の履歴"), ("nll", "FDM 尤度マップ"),
                       ("summary", "推定値まとめ"), ("forward_check", "推定 M による順計算検証"), ("dtilde", "相互拡散係数 D̃"),
                       ("gif", "アニメーション")]:
        if figs.get(key):
            lines += [f"### {title}", "", f"![{key}]({rel(figs[key])})", ""]
    path = os.path.join(outdir, "REPORT.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# HTML report (student-oriented, self-contained)
# ---------------------------------------------------------------------------
def _img_data_uri(path: str) -> str:
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    mime = {"png": "image/png", "gif": "image/gif", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(ext, "application/octet-stream")
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode("ascii")


def _h_fig(path: str | None, caption: str, num: list[int]) -> str:
    if not path or not os.path.isfile(path):
        return ""
    num[0] += 1
    return (f'<figure><img src="{_img_data_uri(path)}" alt="{html.escape(os.path.basename(path))}">'
            f"<figcaption><b>図 {num[0]}.</b> {caption}</figcaption></figure>\n")


def _h_table(headers: list[str], rows: list[list[str]], caption: str | None = None) -> str:
    s = "<table>"
    if caption:
        s += f"<caption>{caption}</caption>"
    s += "<thead><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr></thead><tbody>"
    for r in rows:
        s += "<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
    return s + "</tbody></table>\n"


_HTML_CSS = """
body { font-family: "Hiragino Sans", "Noto Sans JP", "Yu Gothic", Meiryo, Arial, sans-serif; max-width: 1200px;
       margin: 0 auto; padding: 24px 32px; line-height: 1.7; color: #222; background: #fafafa; }
h1 { border-bottom: 4px solid #2b83ba; padding-bottom: 8px; }
h2 { border-left: 8px solid #2b83ba; padding-left: 12px; margin-top: 56px; background: #eef5fb; padding: 6px 12px; }
h3 { margin-top: 32px; color: #1a5f8a; }
figure { margin: 24px 0; text-align: center; background: white; padding: 12px; border: 1px solid #ddd; border-radius: 6px; }
figure img { max-width: 100%; height: auto; }
figcaption { text-align: left; margin-top: 8px; font-size: 0.95em; color: #444; }
table { border-collapse: collapse; margin: 16px 0; background: white; font-size: 0.95em; }
th, td { border: 1px solid #bbb; padding: 6px 12px; text-align: left; }
th { background: #e3eef7; }
caption { caption-side: top; font-weight: bold; text-align: left; padding: 4px 0; }
.box { background: #fff8e6; border: 1px solid #f0c060; border-radius: 6px; padding: 12px 18px; margin: 18px 0; }
.note { background: #eef7ee; border: 1px solid #8cc48c; border-radius: 6px; padding: 12px 18px; margin: 18px 0; }
.warn { background: #fdecec; border: 1px solid #e08080; border-radius: 6px; padding: 12px 18px; margin: 18px 0; }
.summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin: 20px 0; }
.card { background: white; border: 1px solid #ccc; border-radius: 8px; padding: 12px 16px; }
.card .v { font-size: 1.5em; font-weight: bold; color: #1a5f8a; }
.card .k { font-size: 0.9em; color: #555; }
pre { background: #1e1e1e; color: #ddd; padding: 12px; overflow-x: auto; font-size: 0.8em; border-radius: 6px; max-height: 480px; }
code { background: #eee; padding: 1px 4px; border-radius: 3px; }
nav ol { columns: 2; }
.ok { color: #2a7f2a; font-weight: bold; } .ng { color: #b00020; font-weight: bold; }
"""


def write_html_report(outdir: str, res, cfg, metrics, figs: dict[str, str | None], log_path: str | None) -> str:
    t = res["t_grid"]
    hours = t * cfg["annealing_time_h"]
    valid = metrics["valid_indices"]
    rmse_t = metrics["rmse_per_time"]
    exp_set = {int(i) for i in res["exp_time_indices"]}
    timing = _timing_dict(res)
    inv = _inverse_summary(res)
    M_true = res["M_true"]
    num = [0]
    n_frames_valid = len(valid)
    form_label = "Fick 形式" if cfg["use_fick_form"] else "Onsager 形式"
    ok = lambda b: '<span class="ok">OK</span>' if b else '<span class="ng">NG</span>'  # noqa: E731
    f3 = lambda v: f"{v:.3e}"  # noqa: E731

    parts: list[str] = []
    P = parts.append

    # ---- header & summary -------------------------------------------------
    P(f"""<h1>Co–Ni–Ta 合金の拡散: PINN による濃度プロファイルのフィットと拡散係数の逆解析</h1>
<p><b>対象:</b> 学生向け解説レポート &nbsp;|&nbsp; <b>生成日時:</b> {time.strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp;
<b>生成スクリプト:</b> <code>plot_diffusion_timeseries_profiles.py</code> (コア: <code>co_ni_ta_pinn_diffusion_reliability.py</code>)</p>
<div class="box">このレポートは <b>Physics-Informed Neural Networks (PINNs)</b> を初めて学ぶ人を対象に,
(1) PINN とは何か, (2) 合金拡散の順問題 (FDM), (3) PINN による濃度プロファイルの時系列フィット,
(4) PINN と FDM を用いた<b>拡散係数 (移動度) の逆解析</b> を, 1 回の計算で得た結果と図表で説明します。
すべての図は実際の計算結果から自動生成されています。</div>
<nav><b>目次</b><ol>
<li><a href="#s1">PINNs とは何か</a></li><li><a href="#s2">対象とする拡散問題</a></li>
<li><a href="#s3">疑似実験データ (FDM 順問題)</a></li><li><a href="#s4">PINN 順フィット: 時系列濃度プロファイル</a></li>
<li><a href="#s5">拡散係数 (移動度) の逆解析</a></li><li><a href="#s6">考察・限界</a></li>
<li><a href="#s7">演習課題</a></li><li><a href="#s8">付録 (設定・ログ・ファイル)</a></li></ol></nav>
""")
    cards = [
        (f"{metrics['rmse_valid_mean']:.3e}", "順フィット PINN の RMSE (FDM との差, t ≥ t_start)"),
        (f"{metrics['temporal_ratio']:.2f}", "時間変化量比 PINN / FDM"),
        (f"{cfg['epochs']}", "学習 epoch 数 (順フィット)"),
        (f"{len(exp_set)} × {cfg['n_exp_points']}", "疑似実験点 (時刻 × 点)"),
    ]
    if inv:
        cards.append((f"{res['M_hat_pinn'][0] / M_true[0]:.3f} / {res['M_hat_pinn'][1] / M_true[1]:.3f}",
                      "PINN 逆解析: M̂_Ni/真値, M̂_Ta/真値"))
        cards.append((f"{inv['rmse_fdm_hat']:.2e}", "FDM(M̂) と FDM(真の M) の RMSE"))
        if "M_hat_nll" in res:
            cards.append((f"{res['M_hat_nll'][0] / M_true[0]:.3f} / {res['M_hat_nll'][1] / M_true[1]:.3f}",
                          "FDM 尤度マップの最小点: M̂_Ni/真値, M̂_Ta/真値"))
    P('<div class="summary">' + "".join(f'<div class="card"><div class="v">{v}</div><div class="k">{k}</div></div>' for v, k in cards) + "</div>")

    # ---- 1. PINN intro ----------------------------------------------------
    P(r"""<h2 id="s1">1. PINNs (Physics-Informed Neural Networks) とは何か</h2>
<h3>1.1 ニューラルネットワークを「関数」として使う</h3>
<p>拡散の問題では, 位置 \(x\) と時刻 \(t\) を与えると濃度 \(c(x,t)\) が決まります。PINN では, この未知関数
\(c(x,t)\) を<b>ニューラルネットワーク</b> \( \mathcal{N}_{\mathbf w}(x,t) \) で表します。ネットワークは重み \(\mathbf w\) を持つ
滑らかな関数で, 入力 \((x,t)\) から出力 (ここでは 3 成分のモル分率 \(c_{\mathrm{Co}}, c_{\mathrm{Ni}}, c_{\mathrm{Ta}}\)) を返します。
「データを補間する多項式」の代わりに「ネットワーク」を使う, と考えるとわかりやすいでしょう。</p>
<h3>1.2 物理法則を「損失関数」に入れる</h3>
<p>普通の機械学習は「データに合うように」重みを決めます (データ損失 \(\mathcal L_{\mathrm{data}}\))。PINN はそれに加えて,
支配方程式 (ここでは拡散方程式) を<b>どれだけ満たしていないか</b>を表す<b>残差</b> \(r\) を計算し, その二乗平均
\(\mathcal L_{\mathrm{phys}} = \langle r^2\rangle\) も同時に小さくします。初期条件の損失 \(\mathcal L_{\mathrm{ic}}\) も加えて</p>
\[ \mathcal L(\mathbf w) = w_{\mathrm{data}}\,\mathcal L_{\mathrm{data}} + w_{\mathrm{ic}}\,\mathcal L_{\mathrm{ic}} + w_{\mathrm{phys}}\,\mathcal L_{\mathrm{phys}} \]
<p>を最小化します。物理損失は<b>測定点がない場所・時刻</b> (コロケーション点) でも評価できるので,
少ないノイズ付きデータからでも物理的に自然な解が得られる, というのが PINN の狙いです。</p>
<h3>1.3 微分は「自動微分」で計算する</h3>
<p>残差 \(r = \partial c/\partial t - \partial_x[\tilde D\,\partial_x c]\) には \(c\) の時間微分・空間微分が必要です。
ネットワークは解析的な関数の合成なので, 深層学習フレームワーク (PyTorch) の<b>自動微分</b>で \(\partial c/\partial t\),
\(\partial c/\partial x\), \(\partial^2 c/\partial x^2\) を厳密に計算できます (差分近似は不要)。</p>
<h3>1.4 順問題と逆問題</h3>
<ul>
<li><b>順問題:</b> 拡散係数など物理パラメータが既知で, \(c(x,t)\) を求める (FDM でも解ける)。PINN では重み \(\mathbf w\) だけを学習します。</li>
<li><b>逆問題:</b> 観測データから<b>未知の物理パラメータ</b> (ここでは移動度 \(M\), すなわち拡散係数) を推定する。
PINN では \(M\) を<b>学習可能な変数として損失関数に含める</b>だけで, 重み \(\mathbf w\) と \(M\) を同時に最適化できます。
本レポートの第 5 章がこれに当たります。</li>
</ul>
<h3>1.5 PINN の利点と注意点</h3>
<table><thead><tr><th>利点</th><th>注意点</th></tr></thead><tbody>
<tr><td>メッシュ不要で, データと物理を同じ枠組みで扱える</td><td>学習に時間がかかる (本例: CPU で数分〜十数分)</td></tr>
<tr><td>逆問題が「パラメータを学習変数にする」だけで書ける</td><td>損失の重み \(w_{\mathrm{data}}, w_{\mathrm{phys}}\) の調整が結果に影響する</td></tr>
<tr><td>ノイズのあるデータを滑らかに補間・外挿できる</td><td>解が急峻な (界面のような) 領域は表現しにくい</td></tr>
<tr><td>自動微分で高階微分が正確に得られる</td><td>収束の保証がなく, 診断 (残差・RMSE・尤度) が重要</td></tr>
</tbody></table>
""")
    P(_h_fig(figs.get("schematic"), "PINN の概念図。入力 \\((x,t)\\) → ネットワーク → 3 成分の濃度。出力を自動微分して PDE 残差を作り, "
             "データ損失・初期条件損失・物理損失の重み付き和を最小化する。逆問題では移動度 \\(M\\) も学習変数に加える。", num))

    # ---- 2. Diffusion model -----------------------------------------------
    c_l = _C_LEFT_DISP / _C_LEFT_DISP.sum()
    c_r = _C_RIGHT_DISP / _C_RIGHT_DISP.sum()
    P(rf"""<h2 id="s2">2. 対象とする拡散問題: Co–Ni–Ta 拡散対</h2>
<h3>2.1 セットアップ</h3>
<p>左側 (Co ≈ {c_l[0]:.3f}, Ni ≈ {c_l[1]:.4f}, Ta ≈ {c_l[2]:.4f}) と右側 (Co ≈ {c_r[0]:.4f}, Ni ≈ {c_r[1]:.3f}, Ta ≈ {c_r[2]:.3f}) の
合金を \(x = 0.5\) (表示上は 0 µm) で接合した<b>拡散対</b>を考えます。系は閉じており (両端でゼロ流束), 時間とともに界面付近で
成分が混ざり合います。Co を基準成分 (従属成分) とし, Ni と Ta を独立成分として扱います。</p>
<h3>2.2 Regular-solution (正則溶体) 熱力学</h3>
<p>各成分の化学ポテンシャルは, 理想混合項と対相互作用パラメータ \(\Omega_{{ij}}\) の項からなります (\(RT\) で無次元化):</p>
\[ \mu_i = RT\ln c_i + \sum_{{j\ne i}} \Omega_{{ij}} c_j - \sum_{{j \lt k}} \Omega_{{jk}} c_j c_k \]
<p>本計算では \(\Omega_{{\mathrm{{CoNi}}}}, \Omega_{{\mathrm{{CoTa}}}}, \Omega_{{\mathrm{{NiTa}}}}\) =
{np.array2string(res['omega_true'], precision=3)} (RT 単位) を用いました。\(c_i\to 0\) での \(\ln c_i\) の発散を避けるため,
\(c_i\) を <code>mu_floor</code> = {cfg['mu_floor']} で下から抑えています。</p>
<h3>2.3 Onsager 形式の流束と相互拡散係数 \(\tilde D\)</h3>
<p>独立成分 \(k\in\{{\mathrm{{Ni}},\mathrm{{Ta}}\}}\) の流束は化学ポテンシャル勾配に比例します (移動度行列 \(M\)):</p>
\[ J_k = -\sum_j M_{{kj}}\,\frac{{\partial(\mu_j-\mu_{{\mathrm{{Co}}}})}}{{\partial x}},\qquad
   \frac{{\partial c_k}}{{\partial t}} = -\frac{{\partial J_k}}{{\partial x}} \]
<p>連鎖律で濃度勾配の形に書き直すと Fick 形式になり, <b>相互拡散係数行列</b> (interdiffusion matrix) が現れます:</p>
\[ \frac{{\partial c_k}}{{\partial t}} = \frac{{\partial}}{{\partial x}}\Big[\sum_m \tilde D_{{km}}(c)\,\frac{{\partial c_m}}{{\partial x}}\Big],\qquad
   \tilde D_{{km}}(c) = \sum_j M_{{kj}}\,\frac{{\partial(\mu_j-\mu_{{\mathrm{{Co}}}})}}{{\partial c_m}} \]
<p>つまり<b>拡散係数 \(\tilde D\) は「移動度 \(M\)」×「熱力学因子 \(\partial\mu/\partial c\)」</b>で, 組成に依存します。
熱力学因子は \(\Omega\) から計算できるので, 本レポートの逆解析では<b>移動度 \(M\) (対角成分 \(M_{{\mathrm{{Ni}}}}, M_{{\mathrm{{Ta}}}}\)) を未知パラメータ</b>とします。
これは CALPHAD/DICTRA で「熱力学データベースは既知, 移動度データベースを実験から評価する」状況に対応します。</p>
<div class="note"><b>Fick 形式と凍結係数近似:</b> PINN の物理損失は {form_label}で評価しています。Fick 形式では
\(\tilde D(c)\) の中の \(1/c\) 項をネットワーク重みで微分すると勾配が爆発しやすいため, \(\tilde D\) の値は計算グラフから切り離して
「係数を凍結」します (frozen coefficient)。その副作用として, <b>\(\tilde D\) 内の \(\Omega\) には勾配が流れず, 順フィットでは \(\Omega\) は初期値のまま</b>です
(逆解析では \(M\) だけを計算グラフに残すことで \(M\) を学習可能にしています)。</div>
<h3>2.4 無次元化と表示スケール</h3>
<p>FDM の物理時間 \(t_{{\max}}\) = {float(res['t_max_physical']):.4f} (モデル単位) は空間 \(x\in[0,1]\) に比べて小さいため, PINN の入力時刻は
\(\tau = t/t_{{\max}}\in[0,1]\) に正規化し, 移動度に \(t_{{\max}}\) を掛けて補正します。図の軸は見やすさのため
\(\tau = 1 \to\) {cfg['annealing_time_h']} h, \(x\in[0,1]\to\) {cfg['span_um']} µm (界面 0 µm) の例示スケールで表示しています。</p>
""")
    P(_h_table(["項目", "値"], [
        ["FDM 格子点数 / dt / ステップ数 / 保存間隔", f"{_NX_FDM} / {cfg['dt']} / {cfg['nsteps']} / {cfg['save_every']} (frames = {len(t)})"],
        ["真の移動度 diag(M) [M_Ni, M_Ta]", f"[{M_true[0]:.3e}, {M_true[1]:.3e}]"],
        ["Ω [CoNi, CoTa, NiTa] (RT 単位)", np.array2string(res["omega_true"], precision=3)],
        ["mu_floor", f"{cfg['mu_floor']}"],
        ["PINN 構造", "4 層 × 64 ユニット, tanh, 時間 Fourier 特徴 4, 出力は simplex 制約付き"],
        ["損失重み w_data / w_ic / w_phys", f"{cfg['w_data']} / {cfg['w_ic']} / {cfg['w_phys']} (物理損失は最初の {int(cfg['warmup'] * 100)}% の epoch で線形に立ち上げ)"],
        ["学習率 / epochs", f"{cfg['lr']} (cosine 減衰) / {cfg['epochs']}"],
        ["コロケーション点 / 観測点", f"{cfg['n_f']} 点 (毎 epoch 再サンプル) / {cfg['n_obs']} 点 + 疑似実験点"],
        ["デバイス / 計算時間", f"{cfg['device']} / " + ", ".join(f"{k}: {v:.1f} s" for k, v in timing.items())],
    ], "表 1. 計算条件"))

    # ---- 3. Pseudo-experimental data --------------------------------------
    P(f"""<h2 id="s3">3. 疑似実験データ (FDM 順問題)</h2>
<p>実験データの代わりに, 真の \\(M, \\Omega\\) で解いた <b>FDM (有限差分法, DICTRA 型の有限体積スキーム)</b> の解を「隠れた真値」とし,
{len(exp_set)} 時刻 × {cfg['n_exp_points']} 点 で標準偏差 \\(\\sigma\\) = {cfg['noise']} のガウスノイズを加えたものを<b>疑似実験データ</b>とします。
このほか, 学習用に \\(t\\ge t_{{\\mathrm{{start}}}}\\) の FDM 解からランダムに {cfg['n_obs']} 点をサンプルした観測点も用います
(ノイズは同じ)。真値が分かっているので, PINN のフィットと逆解析の<b>精度を定量的に検証</b>できるのがこの構成の利点です。</p>
<p>FDM は適応サブステップで安定化された陽解法で, 質量保存 (\\(\\sum_i c_i = 1\\), 最大偏差 {metrics['fdm_sum_dev_max']:.1e}) を満たします。</p>
""")
    P(_h_fig(figs.get("pseudo_exp"), f"疑似実験データ。各パネルが 1 時刻 ({len(exp_set)} 時刻, 左上から右下へ時刻順)。マーカーが疑似実験点 "
             "(○ Co, □ Ni, △ Ta), 細い線が隠れた FDM 真値。", num))

    # ---- 4. Forward PINN fit ----------------------------------------------
    P(f"""<h2 id="s4">4. PINN 順フィット: 時系列濃度プロファイル</h2>
<p>まず \\(M, \\Omega\\) を既知として PINN を {cfg['epochs']} epoch 学習し (順問題), 全時刻・全位置の濃度を FDM と比較します。
学習後のネットワークは任意の \\((x,t)\\) で評価できるので, FDM の {n_frames_valid} フレームすべてと比較できます。</p>
<h3>4.1 妥当性チェック</h3>""")
    P(_h_table(["項目", "FDM", "PINN", "判定"], [
        ["有限値", str(metrics["finite_fdm"]), str(metrics["finite_pinn"]), ok(metrics["finite_fdm"] and metrics["finite_pinn"])],
        ["濃度範囲", f"[{metrics['fdm_min']:.4f}, {metrics['fdm_max']:.4f}]", f"[{metrics['pinn_min']:.4f}, {metrics['pinn_max']:.4f}]",
         ok(metrics["pinn_min"] >= -1e-6 and metrics["pinn_max"] <= 1 + 1e-6)],
        ["max |Σc − 1| (simplex 制約)", f"{metrics['fdm_sum_dev_max']:.2e}", f"{metrics['pinn_sum_dev_max']:.2e}", ok(metrics["pinn_sum_dev_max"] < 1e-4)],
        ["時間変化量 max|c(t_end) − c(t_start)|", f"{metrics['fdm_temporal_change']:.4f}", f"{metrics['pinn_temporal_change']:.4f} (比 {metrics['temporal_ratio']:.3f})",
         ok(metrics["temporal_ratio"] > 0.5)],
        ["RMSE (t ≥ t_start 平均)", "–", f"{metrics['rmse_valid_mean']:.3e}", ok(metrics["rmse_valid_mean"] < 0.05)],
        ["成分別 RMSE", "–", ", ".join(f"{c} = {v:.3e}" for c, v in zip(COMPONENTS, metrics["rmse_per_component"])), ""],
        ["Ω_learned", "–", np.array2string(res["omega_learned"], precision=3) + (" (Fick 形式のため初期値のまま; 2.3 節)" if cfg["use_fick_form"] else ""), ""],
    ], "表 2. 順フィットの妥当性チェック"))
    P("<h3>4.2 時刻別 RMSE</h3>")
    rmse_hdr = ["frame", "τ", "t (h, 表示)", "RMSE (全成分)", "疑似実験点あり"]

    def rmse_row(ti: int) -> list[str]:
        return [str(ti), f"{t[ti]:.3f}", f"{hours[ti]:.1f}", f"{rmse_t[ti]:.3e}", "●" if ti in exp_set else ""]

    P(_h_table(rmse_hdr, [rmse_row(ti) for ti in valid if ti in exp_set],
               "表 3. 時刻別 RMSE (PINN − FDM), 疑似実験のある時刻"))
    P(f"<details><summary>全 {len(valid)} フレームの RMSE を展開</summary>")
    P(_h_table(rmse_hdr, [rmse_row(ti) for ti in valid], "表 3′. 全フレームの時刻別 RMSE"))
    P("</details>")
    P("<h3>4.3 図</h3>")
    P(_h_fig(figs.get("panels"), f"時刻別パネル ({len(exp_set)} 時刻, 左上から右下へ時刻順)。実線: FDM, 破線: PINN, マーカー: 疑似実験点。"
             "黒: Co, 青: Ni, 緑: Ta。", num))
    P(_h_fig(figs.get("by_component"), "成分別の時間発展 (左: Co, 中: Ni, 右: Ta)。色が時刻 (カラーバー), 実線: FDM, 破線: PINN。", num))
    P(_h_fig(figs.get("heatmap"), "空間–時間マップ。行: Co, Ni, Ta; 列: FDM, PINN, 差 (PINN − FDM)。縦軸が時間, 横軸が位置。", num))
    P(_h_fig(figs.get("fixed"), "固定位置 (−80, −40, 0, +40, +80 µm) での濃度の時間変化 (左: Co, 中: Ni, 右: Ta)。実線: FDM, 破線: PINN。", num))
    P(_h_fig(figs.get("loss"), "順フィット PINN の損失履歴 (対数軸)。物理損失は warm-up により途中から立ち上がる。", num))
    P(_h_fig(figs.get("omega"), "Ω の履歴。Fick 形式 (凍結係数) では勾配が流れないため一定 (2.3 節)。", num))
    P(_h_fig(figs.get("gif"), "時間発展アニメーション (実線: FDM, 破線: PINN)。", num))

    # ---- 5. Inverse analysis ----------------------------------------------
    if inv:
        M_init, M_pinn = res["M_init"], res["M_hat_pinn"]
        inv_hist_len = int(res["inverse_history"].shape[0])
        P(rf"""<h2 id="s5">5. 拡散係数 (移動度) の逆解析</h2>
<h3>5.1 問題設定</h3>
<p>熱力学 (\(\Omega\)) は既知, <b>移動度 \(M = \mathrm{{diag}}(M_{{\mathrm{{Ni}}}}, M_{{\mathrm{{Ta}}}})\) は未知</b>とし,
第 3 章の疑似実験データ (+観測点) から \(M\) を推定します。真値は \(M_{{\mathrm{{Ni}}}}\) = {M_true[0]:.3e}, \(M_{{\mathrm{{Ta}}}}\) = {M_true[1]:.3e},
逆解析の出発点 (初期推定値) は真値の {cfg['m_init_factor']} 倍 ({M_init[0]:.3e}, {M_init[1]:.3e}) としました。
3 つの独立な方法で推定し, 互いに比較します。</p>
<h3>5.2 方法 3a: PINN による同時推定</h3>
<p>\(\log M_{{\mathrm{{Ni}}}}, \log M_{{\mathrm{{Ta}}}}\) を学習可能な変数にし, ネットワーク重みと一緒に Adam で最適化します
(学習率: 重み {cfg['lr']}, \(\log M\) {cfg['lr_m']}; {inv_hist_len} epoch)。
物理損失 \(\langle r^2\rangle\) は \(\tilde D = M\,\Phi(c)\) を通して \(M\) に依存するので, <b>データに合う \(c(x,t)\) を保ちながら PDE 残差を最小にする \(M\)</b> が選ばれます。
熱力学因子 \(\Phi(c)=\partial(\mu_j-\mu_{{\mathrm{{Co}}}})/\partial c_m\) は凍結 (detach) し, \(M\) だけを計算グラフに残しています。</p>""")
        P(_h_fig(figs.get("inverse"), "逆解析 PINN の履歴。左: 損失, 中: 学習中の移動度 (点線: 真値), 右: 真値との比 (破線: 初期推定値)。", num))
        P(r"""<h3>5.3 方法 3b: 学習済み PINN の微分からの最小二乗 (事後推定)</h3>
<p>\(\Omega\) が既知なら, PDE は \(M\) について<b>線形</b>です:</p>
\[ \frac{\partial c_k}{\partial \tau} = \sum_j M^{\mathrm{eff}}_{kj}\, a_j,\qquad a_j = \sum_m \Phi_{jm}(c)\,\frac{\partial^2 c_m}{\partial x^2} \]
<p>学習済みネットワークの自動微分で \(\partial c_k/\partial\tau\) と \(\partial^2 c_m/\partial x^2\) をコロケーション点で評価すれば, \(M\) は通常の最小二乗で求まります
(\(\Phi\) を凍結した PINN 残差と同じ形。\(1/c\) による誤差増幅を避けるため \(c_i \ge 0.05\) の点のみ使用し, 残差の大きい 5% を除外)。順フィット PINN (\(M\) を知って学習) と逆解析 PINN (\(M\) を知らずに学習) の両方に適用しました。
順フィットの PINN から真値に近い \(M\) が復元されれば, ネットワークが PDE を「理解」している証拠になります。</p>""")
        P(r"""<h3>5.4 方法 3c: FDM を繰り返す尤度マップ (信頼度の評価)</h3>
<p>PINN を使わない古典的な方法として, \((\log_{10}M_{\mathrm{Ni}}, \log_{10}M_{\mathrm{Ta}})\) の格子上で FDM を解き,
疑似実験点との<b>負の対数尤度</b> \(\mathrm{NLL} = \sum (c_{\mathrm{exp}}-c_{\mathrm{FDM}})^2/(2\sigma^2)\) を計算しました。
最小値からの差 \(\Delta\mathrm{NLL}\) が 1.15 / 3.09 / 5.91 の等高線が 2 パラメータの 1σ / 2σ / 3σ 信頼領域に対応し,
<b>データがどの程度 \(M\) を決めているか (identifiability)</b> が読み取れます。</p>""")
        if "nll_Z" not in res:
            P('<p class="warn">この実行では尤度マップはスキップされました (<code>--nll_grid 0</code>)。</p>')
        P(_h_fig(figs.get("nll"), f"FDM 尤度マップ ({len(res['nll_log10_M_Ni']) if 'nll_Z' in res else 0}² 回の FDM 計算)。左: ΔNLL の対数表示と 1σ/2σ/3σ 等高線, "
                 "★ 真値, ● PINN 推定値, ■ 格子上の最小点, × PINN の初期推定値。中・右: 最小点を通る各軸方向の ΔNLL 断面 (点線: 1σ, 2σ 水準)。", num))
        P("<h3>5.5 推定結果のまとめ</h3>")
        rows = [[name, f"{M[0]:.4e}", f"{M[1]:.4e}", f"{M[0] / M_true[0]:.3f}", f"{M[1] / M_true[1]:.3f}",
                 f"{100 * (M[0] / M_true[0] - 1):+.1f}% / {100 * (M[1] / M_true[1] - 1):+.1f}%"] for name, M in inv["rows"]]
        P(_h_table(["推定法", r"\(M_{\mathrm{Ni}}\)", r"\(M_{\mathrm{Ta}}\)", r"\(M_{\mathrm{Ni}}\) / 真値", r"\(M_{\mathrm{Ta}}\) / 真値", "相対誤差"], rows, "表 4. 移動度の推定結果"))
        Mf, Mi = res["M_ls_forward_full"], res["M_ls_inverse_full"]
        n_ls = res["M_ls_n_points"] if "M_ls_n_points" in res else np.array([-1, -1])
        P(_h_table(["PINN", r"\(M_{\mathrm{NiNi}}\)", r"\(M_{\mathrm{NiTa}}\)", r"\(M_{\mathrm{TaNi}}\)", r"\(M_{\mathrm{TaTa}}\)", "使用点数"], [
            ["順フィット PINN (3b, 全成分 LS)", f3(Mf[0, 0]), f3(Mf[0, 1]), f3(Mf[1, 0]), f3(Mf[1, 1]), str(int(n_ls[0]))],
            ["逆解析 PINN (3b, 全成分 LS)", f3(Mi[0, 0]), f3(Mi[0, 1]), f3(Mi[1, 0]), f3(Mi[1, 1]), str(int(n_ls[1]))],
            ["真値", f3(M_true[0]), "0", "0", f3(M_true[1]), "–"],
        ], "表 5. 最小二乗で非対角成分も自由にした場合 (真の M は対角)"))
        P("<p class=\"note\">全成分 LS では説明変数 \\(a_{\\mathrm{Ni}}, a_{\\mathrm{Ta}}\\) が強く相関 (ほぼ共線) しているため, 非対角成分は不定になりやすく, "
          "対角成分だけを推定した値 (表 4) の方が信頼できます。使用点数が少ないのは, \\(\\Phi\\sim 1/c\\) による誤差増幅を避けるために 全成分 \\(c_i \\ge 0.05\\) の点だけを使っているためです "
          "(Ta は最大 0.1 なので界面右側の狭い領域に限られる)。</p>")
        if "ci_ni" in inv:
            ci_n, ci_t = inv["ci_ni"], inv["ci_ta"]
            P(_h_table(["パラメータ", "真値 (log10)", "PINN 推定 (log10)", "FDM 尤度 1σ 区間 (log10)", "真値は区間内?", "PINN 推定は区間内?"], [
                [r"\(M_{\mathrm{Ni}}\)", f"{np.log10(M_true[0]):.3f}", f"{np.log10(M_pinn[0]):.3f}", f"[{ci_n[0]:.3f}, {ci_n[1]:.3f}]",
                 ok(ci_n[0] <= np.log10(M_true[0]) <= ci_n[1]), ok(ci_n[0] <= np.log10(M_pinn[0]) <= ci_n[1])],
                [r"\(M_{\mathrm{Ta}}\)", f"{np.log10(M_true[1]):.3f}", f"{np.log10(M_pinn[1]):.3f}", f"[{ci_t[0]:.3f}, {ci_t[1]:.3f}]",
                 ok(ci_t[0] <= np.log10(M_true[1]) <= ci_t[1]), ok(ci_t[0] <= np.log10(M_pinn[1]) <= ci_t[1])],
            ], "表 6. 尤度断面から読んだ 1σ 区間 (ΔNLL ≤ 0.5, 格子間の線形補間)"))
        P(_h_fig(figs.get("summary"), "各推定法の移動度推定値 (真値との比)。破線 = 1 が真値。", num))
        n_t, n_h, n_i = inv["nll_true_hat_init"]
        P("""<h3>5.6 推定した移動度で順計算して検証する</h3>
<p>推定値 \\(\\hat M\\) (方法 3a) を FDM に入れて解き直し, 真の \\(M\\) の FDM と比較します。逆解析が正しければ 2 つのプロファイルはほぼ一致し,
疑似実験データの NLL は真の \\(M\\) の値に近づくはずです (逆に NLL が初期推定値より悪ければ, 逆解析は失敗しています)。</p>""")
        P(_h_table(["モデル", "疑似実験データの NLL", "FDM(真の M) との RMSE"], [
            ["FDM, 真の M", f"{n_t:.2f}", "0"],
            [r"FDM, PINN 推定 \(\hat M\)", f"{n_h:.2f}", f"{inv['rmse_fdm_hat']:.3e}"],
            [r"FDM, 初期推定値 \(M_{\mathrm{init}}\)", f"{n_i:.2f}", "–"],
            ["逆解析 PINN そのもの", "–", f"{inv['rmse_inv_pinn']:.3e}"],
        ], "表 7. 順計算による検証 (NLL が小さいほどデータに合う)"))
        P(_h_fig(figs.get("forward_check"), "順計算による検証 (4 時刻)。実線: FDM (真の M), 破線: FDM (PINN 推定 M̂), 点線: 逆解析 PINN, マーカー: 疑似実験点。", num))
        P(_h_fig(figs.get("dtilde"), "最終時刻の組成プロファイル (左) と, それに沿った相互拡散係数行列 D̃(c) の成分 (中: Ni 行, 右: Ta 行; symlog 軸)。"
                 "実線: 真の M, 破線: 推定 M̂。D̃ = M × 熱力学因子 なので組成とともに大きく変化する。", num))
        if "pde_check_rms" in res:
            P(r"""<h3>5.7 診断: 学習済みネットワークは PDE を満たしているか</h3>
<p>PINN 逆解析 (3a, 3b) は「ネットワークの時間微分 \(\partial c/\partial\tau\) と, 空間 2 階微分から作った PDE の右辺
\(t_{\max} M a\) が釣り合うように \(M\) を決める」方法です。したがって, <b>学習済みネットワークの微分が PDE と整合していなければ, どんな \(M\) も正しく推定できません</b>。
これを確かめるには, ネットワークを固定して \(M\) だけを変えたときの残差の大きさ
\(\mathrm{RMS}\,r(M)\), \(r_k = \partial c_k/\partial\tau - t_{\max}\sum_j M_{kj}a_j\) を比べます。
\(M = 0\) のとき \(r = \partial c/\partial\tau\) なので, <b>真値の \(M\) での \(\mathrm{RMS}\,r\) が \(M=0\) より小さくなっていなければ, ネットワークは PDE を「理解」していない</b>と判断できます。
基準として, FDM 教師データに対して同じ量を差分で評価した値も示します (ソルバ自身の流束を使うのでほぼ 0 になるはず)。</p>""")
            chk_rows = [[html.escape(str(lab)), f3(row[0]), f3(row[1]), f3(row[2]), f3(row[3])]
                        for lab, row in zip(res["pde_check_labels"], res["pde_check_rms"])]
            P(_h_table(["ネットワーク | 代入した M", r"RMS \(r_{\mathrm{Ni}}\) (全領域)", r"RMS \(r_{\mathrm{Ta}}\) (全領域)",
                        rf"RMS \(r_{{\mathrm{{Ni}}}}\) (\(\min_i c_i \ge {_PDE_CHECK_CMIN}\))", rf"RMS \(r_{{\mathrm{{Ta}}}}\) (\(\min_i c_i \ge {_PDE_CHECK_CMIN}\))"],
                       chk_rows, "表 8. PDE 整合性チェック: 固定したネットワークに対して M を変えたときの残差 RMS (残差は ∂c/∂τ の単位)"))
            r_fdm = res["pde_check_rms"]
            if np.all(r_fdm[0, :2] > r_fdm[1, :2]):
                P(r"""<p><b>FDM 教師の「全領域」の行に注意:</b> 真の \(M\) を入れても残差が \(M=0\) より大きくなっています。これは FDM が間違っているのではなく,
右側の希薄 Co 領域 (\(c_{\mathrm{Co}} \lesssim 0.01\)) で \(\tilde D \propto 1/c\) が非常に大きく, 瞬間的な PDE の右辺
\(\partial_x[\tilde D\,\partial_x c]\) が (適応サブステップで安定化された) 保存フレーム間の平均変化率 \(\Delta c/\Delta\tau\) と一致しない
(stiff な成分) ためです。界面領域 (\(\min_i c_i \ge 0.05\)) では FDM の残差はほぼ 0 になり, この列がネットワークを評価する正しい基準になります。
希薄領域では PDE の右辺そのものが不安定なので, 点ごとの PDE 残差を使う逆解析 (3a, 3b) がこの領域を含めてはいけない理由でもあります。</p>""")
            if inv.get("pde_consistent") is False:
                P(rf"""<p class="warn"><b>この実行では, 順フィット PINN に真の \(M\) を代入しても残差は \(M=0\) のときより小さくなりませんでした。</b>
すなわちネットワークはデータには合っていても, 局所的には PDE を満たしていません。この状況では 3a・3b の推定値は信頼できず,
特に 3a は「残差を最小にするには \(M\to 0\) にすればよい」という方向に引き寄せられます (説明変数の誤差による減衰バイアス)。
原因は主に 3 つです: (i) \(\tilde D \propto 1/c\) のため拡散係数が組成によって 100 倍以上変わり, 希薄な「すそ」領域
(\(c \lesssim 0.01\), ノイズ σ = {cfg['noise']} より小さいのでデータには見えない) で PDE の両辺が大きな値をとること,
(ii) 界面幅 (初期 {_PHASE_WIDTH} ≈ {_PHASE_WIDTH * cfg['span_um']:.0f} µm) が疑似実験点の間隔 (≈ {cfg['span_um'] / cfg['n_exp_points']:.0f} µm) より狭く,
2 階微分をデータから学べないこと, (iii) 凍結係数形式では \(\partial\tilde D/\partial x\cdot\partial c/\partial x\) 項が落ちること。
一方, FDM を繰り返す尤度マップ (3c) はネットワークの微分を使わないのでこの問題の影響を受けません。</p>""")
            elif inv.get("pde_consistent") is True:
                P(r"<p>順フィット PINN に真の \(M\) を代入すると残差が \(M=0\) より小さくなっており, ネットワークの微分は PDE と (少なくとも界面領域で) 整合しています。"
                  r"この場合は 3a・3b の推定値に意味がありますが, 真値との差は残差の残り (表 8) に応じたバイアスを含みます。</p>")
            P(_h_fig(figs.get("pde_check"), "PDE 整合性チェック (中間時刻, 上段: 順フィット PINN, 下段: 逆解析 PINN; 左: Ni, 右: Ta)。"
                     "黒実線: FDM の ∂c/∂τ (差分), 赤破線: ネットワークの ∂c/∂τ (自動微分), 青点線: ネットワークの 2 階微分から作った PDE の右辺 "
                     "(上段は真の M, 下段は推定 M̂)。PDE を満たすネットワークでは赤と青が重なるはず。"
                     "青は c → 0 の場所 (D̃ ∝ 1/c) で軸範囲を超えてスパイクすることがある (軸は時間微分のスケールに合わせている)。", num))
    else:
        P('<h2 id="s5">5. 拡散係数 (移動度) の逆解析</h2><p class="warn">この実行では逆解析はスキップされました (<code>--skip_inverse</code>)。</p>')
        P(_h_fig(figs.get("dtilde"), "最終時刻の組成プロファイル (左) と相互拡散係数行列 D̃(c) (中: Ni 行, 右: Ta 行)。", num))

    # ---- 6. Discussion ----------------------------------------------------
    disc = [
        f"順フィット PINN の RMSE は {metrics['rmse_valid_mean']:.3e}, 時間変化量比は {metrics['temporal_ratio']:.2f}。"
        "PINN は界面の急峻な部分をやや滑らかに表現する傾向があり (図の差分マップ参照), 時間発展を過小評価しやすい。"
        r"\(w_{\mathrm{phys}}\) を小さくするとデータ追従が良くなり, 大きくすると PDE の整合性が上がる (トレードオフ)。",
        r"Fick 形式では \(\tilde D\) を凍結するため \(\Omega\) は学習されない (順フィットの \(\Omega\) は初期値)。\(\Omega\) も推定したい場合は Onsager 形式 (<code>--onsager</code>) か, "
        "本レポートの逆解析のように「学習したいパラメータだけを計算グラフに残す」工夫が必要。",
    ]
    if inv:
        n_t, n_h, n_i = inv["nll_true_hat_init"]
        if n_h < n_i and n_h - n_t <= 3.09:
            verdict_3a = (f"初期推定値 ({cfg['m_init_factor']} 倍) から改善し, NLL は真値の 2σ 以内 (ΔNLL = {n_h - n_t:.1f}); "
                          f"FDM(\\(\\hat M\\)) のプロファイルは真値と RMSE {inv['rmse_fdm_hat']:.2e} で一致する。")
        elif n_h < n_i:
            verdict_3a = (f"初期推定値より NLL は改善したが (ΔNLL = {n_h - n_t:.1f} > 3.09), 真値の 2σ 領域には入っていない。"
                          f"FDM(\\(\\hat M\\)) と真値の RMSE は {inv['rmse_fdm_hat']:.2e}。")
        else:
            verdict_3a = (f"<b>逆解析は失敗している</b>: NLL ({n_h:.0f}) は初期推定値 ({n_i:.0f}) より悪く, 真値 ({n_t:.0f}) から遠い。"
                          + ("学習中に \\(M\\) が小さい方へ流れていったのは, 5.7 節で示した「ネットワークが PDE を満たしていない」状況では残差を最小にするのは \\(M\\to 0\\) だからである。"
                             if inv.get("pde_consistent") is False else "")
                          + f"FDM(\\(\\hat M\\)) と真値の RMSE は {inv['rmse_fdm_hat']:.2e}。")
        disc += [
            f"PINN 同時推定 (3a) の移動度は真値の {M_pinn[0] / M_true[0]:.3f} 倍 (Ni), {M_pinn[1] / M_true[1]:.3f} 倍 (Ta)。" + verdict_3a,
        ]
        if "M_hat_nll" in res:
            Mn = res["M_hat_nll"]
            disc.append(f"FDM 尤度マップ (3c) の最小点は真値の {Mn[0] / M_true[0]:.3f} 倍 (Ni), {Mn[1] / M_true[1]:.3f} 倍 (Ta) で, "
                        "この疑似実験データに対する基準解 (benchmark) として使える。古典的だがネットワークの微分に依存しないので堅牢。")
        disc += [
            r"\(M_{\mathrm{Ta}}\) は Ta の濃度変化が小さい (最大 0.1) ため \(M_{\mathrm{Ni}}\) より決めにくい。尤度マップの断面の幅 (1σ 区間) を比べると, データがどのパラメータをよく決めているかが分かる。",
            r"最小二乗 (3b) は学習後に数秒で計算できるが, ネットワークの 2 階微分の精度に依存する。順フィット PINN から復元した \(M\) が真値からずれる分は, "
            "PINN がデータに合っていても PDE を完全には満たしていないことを意味する (表 8 と図で確認)。このレポートでは 3b を「ネットワーク微分の診断」として扱い, 拡散係数の推定値としては使わない。",
            "FDM 尤度マップ (3c) は最も信頼できるが, 格子点数 × FDM 1 回の計算コストがかかる (パラメータが増えると急激に高コスト)。"
            "PINN 逆解析は 1 回の学習で済むのが利点だが, 本例のように拡散係数が組成で 100 倍以上変わる多スケール問題では, 点ごとの PDE 残差を使う逆解析は悪条件になりやすい。"
            "改善案: 疑似実験点を密にして界面を解像する (<code>--n_exp_points</code>), 残差を \\(\\min_i c_i\\) で重み付けする, "
            "\\(M\\) を入力にもつパラメトリック PINN を学習してデータ誤差で \\(M\\) を決める (尤度マップの PINN 版), など。",
        ]
    P('<h2 id="s6">6. 考察と限界</h2><ul>' + "".join(f"<li>{d}</li>" for d in disc) + "</ul>")

    # ---- 7. Exercises -----------------------------------------------------
    P(r"""<h2 id="s7">7. 演習課題 (学生向け)</h2><ol>
<li><code>--noise 0.05</code> にしてノイズを増やすと, 尤度マップの 1σ 領域と PINN 推定値はどう変わるか。</li>
<li><code>--w_phys 1.0</code> と <code>--w_phys 0.05</code> で順フィットの RMSE と時間変化量比を比較し, 物理損失の役割を説明せよ。</li>
<li><code>--m_init_factor 0.3</code> (真値より小さい初期値) から逆解析を始めても同じ \(\hat M\) に収束するか。</li>
<li><code>--n_time_slices 2</code> (時刻を減らす) で拡散係数の識別性 (identifiability) がどう悪化するか, 尤度マップで確認せよ。</li>
<li>表 5 の非対角成分は真値 0 に対してどの程度か。非対角成分を「ゼロと区別できる」ためには何が必要か考察せよ。</li>
<li><code>--onsager</code> で Onsager 形式にすると \(\Omega\) の学習が可能になる。\(\Omega\) と \(M\) を同時に推定するとき, 両者は区別できるか (\(\tilde D = M\,\Phi(\Omega)\) に注意)。</li>
</ol><p>再計算なしで図とレポートだけ作り直す: <code>python plot_diffusion_timeseries_profiles.py --replot timeseries_results.npz --outdir &lt;dir&gt;</code></p>""")

    # ---- 8. Appendix ------------------------------------------------------
    P('<h2 id="s8">8. 付録</h2><h3>8.1 設定 (JSON)</h3>')
    P(f"<pre>{html.escape(json.dumps(cfg, indent=2, ensure_ascii=False))}</pre>")
    if "fdm_repro_maxdiff" in res:
        P(f"<p>スタンドアロン FDM ラッパーが教師データを再現するかの自己チェック: max|Δc| = {float(res['fdm_repro_maxdiff']):.2e} "
          f"{ok(float(res['fdm_repro_maxdiff']) < 1e-10)}</p>")
    P("<h3>8.2 出力ファイル</h3>")
    files = sorted(f for f in os.listdir(outdir) if not f.startswith("."))
    P(_h_table(["ファイル", "サイズ"], [[html.escape(f), f"{os.path.getsize(os.path.join(outdir, f)) / 1024:.0f} KB"] for f in files]))
    if log_path and os.path.isfile(log_path):
        with open(log_path, encoding="utf-8", errors="replace") as f:
            log_txt = f.read()
        P(f"<h3>8.3 実行ログ (標準出力)</h3><pre>{html.escape(log_txt[-60000:])}</pre>")
    P("""<script>
MathJax = {tex: {inlineMath: [['\\\\(', '\\\\)']], displayMath: [['\\\\[', '\\\\]']]}};
</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>""")

    doc = ("<!DOCTYPE html><html lang=\"ja\"><head><meta charset=\"utf-8\">"
           "<title>Co–Ni–Ta 拡散: PINN フィットと拡散係数逆解析</title>"
           f"<style>{_HTML_CSS}</style></head><body>\n" + "\n".join(parts) + "\n</body></html>")
    path = os.path.join(outdir, "REPORT.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
class _Tee:
    """Duplicate stdout to a log file so the report can embed the run log."""

    def __init__(self, path: str):
        self._f = open(path, "a", encoding="utf-8")
        self._out = sys.stdout

    def write(self, s):
        self._out.write(s)
        self._f.write(s)

    def flush(self):
        self._out.flush()
        self._f.flush()

    def close(self):
        self._f.close()


def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    log_path = os.path.join(args.outdir, "run_log_internal.txt")
    tee = _Tee(log_path)
    sys.stdout = tee
    print(f"[start] {time.strftime('%Y-%m-%d %H:%M:%S')}  args: {json.dumps(vars(args))}")

    if args.replot:
        res = load_results(args.replot)
        print(f"[replot] loaded {args.replot}")
        cfg = json.loads(str(res["config"]))
        _resolve_display_settings(args, cfg)
    else:
        _resolve_display_settings(args, None)
        res = run_computation(args)
        save_results(args.outdir, res)
        print(f"[saved] results to {args.outdir}")
        cfg = json.loads(str(res["config"]))
    cfg["annealing_time_h"] = args.annealing_time_h
    cfg["span_um"] = args.span_um
    cfg["n_time_slices"] = args.n_time_slices

    metrics = compute_metrics(res)
    print("=" * 72)
    print("Validation")
    print("=" * 72)
    for k in ["finite_fdm", "finite_pinn", "fdm_sum_dev_max", "pinn_sum_dev_max", "fdm_min", "fdm_max",
              "pinn_min", "pinn_max", "fdm_temporal_change", "pinn_temporal_change", "temporal_ratio", "rmse_valid_mean"]:
        print(f"  {k:22s}: {metrics[k]}")
    print(f"  Omega_true    : {res['omega_true']}")
    print(f"  Omega_learned : {res['omega_learned']}")
    print(f"  M_true        : {res['M_true']}")
    inv = _inverse_summary(res)
    if inv:
        for name, M in inv["rows"]:
            print(f"  M[{name}] : {M}  ratio {M / res['M_true']}")
        print(f"  RMSE FDM(M_hat) vs FDM(true) : {inv['rmse_fdm_hat']:.3e}")
        print(f"  RMSE inverse PINN vs FDM     : {inv['rmse_inv_pinn']:.3e}")

    plt = _setup_mpl()
    figs: dict[str, str | None] = {}
    figs["schematic"] = fig_pinn_schematic(args.outdir, plt)
    figs["pseudo_exp"] = fig_pseudo_exp_data(res, cfg, args.outdir, plt)
    figs["panels"] = fig_timeseries_panels(res, cfg, args.outdir, plt)
    figs["by_component"] = fig_timeseries_by_component(res, cfg, args.outdir, plt)
    figs["heatmap"] = fig_spacetime_heatmap(res, cfg, args.outdir, plt)
    figs["fixed"] = fig_fixed_position_timeseries(res, cfg, args.outdir, plt)
    figs["loss"], figs["omega"] = fig_loss_history(res, cfg, args.outdir, plt)
    figs["inverse"] = fig_inverse_history(res, cfg, args.outdir, plt)
    figs["nll"] = fig_nll_map(res, cfg, args.outdir, plt)
    figs["summary"] = fig_mobility_summary(res, cfg, args.outdir, plt)
    figs["forward_check"] = fig_forward_check(res, cfg, args.outdir, plt)
    figs["pde_check"] = fig_pde_consistency(res, cfg, args.outdir, plt)
    figs["dtilde"] = fig_dtilde_profiles(res, cfg, args.outdir, plt)
    figs["gif"] = None if args.no_gif else make_animation(res, cfg, args.outdir, plt)
    for k, v in figs.items():
        if v:
            print(f"  [fig] {k:13s} -> {v}")
    report = write_report(args.outdir, res, cfg, metrics, figs)
    print(f"  [report] {report}")
    tee.flush()
    html_report = write_html_report(args.outdir, res, cfg, metrics, figs, log_path)
    print(f"  [report] {html_report}")
    sys.stdout = tee._out
    tee.close()


if __name__ == "__main__":
    main()
