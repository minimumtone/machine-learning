---
name: testing-pinn-diffusion
description: Test the fig11 PINN diffusion Streamlit app end-to-end. Use when verifying PINN training, reliability (Laplace/MCMC), or RS mode changes.
---

# Testing the Fig.11 PINN Diffusion App

## App Overview

- **Main file**: `fig11_co_ni_ta_pinn_reliability_v23_multitime_pseudoexp.py` (~7400+ lines)
- **Framework**: Streamlit + PyTorch + NumPy
- **Modes**: Fickian D (linear diffusion matrix) and RS (Regular-solution chemical potential)
- **Features**: PINN training, FDM teacher, Laplace approximation, MCMC reliability, PSIS diagnostic, σ marginalization, RBA loss weighting, direct output

## Running the App

```bash
cd /home/ubuntu/repos/machine-learning
streamlit run fig11_co_ni_ta_pinn_reliability_v23_multitime_pseudoexp.py --server.port 8502
```

- App takes 45-60s to initialize (large file + torch imports)
- Default mode: Regular-solution chemical potential
- All new features (RBA, PSIS, σ marginalization, direct output) default to OFF

## Testing Strategy: Hybrid Approach

### Why Hybrid?

Streamlit UI testing of computationally-intensive features (PSIS, MCMC, reliability) on CPU can hang for 22+ minutes due to Streamlit's rendering/caching/WebSocket overhead — even when the underlying computation takes <10s. The bottleneck is NOT the FDM solver but Streamlit's runtime.

**Solution**: Use direct Python subprocess testing for compute-heavy features, Streamlit UI for simple interactions.

### Direct Python Testing (Recommended for compute-heavy features)

1. Read the source file, truncate before the "Main UI" section (~line 4940)
2. Mock Streamlit with a minimal module that supports:
   - Context managers (`with st.sidebar:`)
   - Attribute-based session_state (`st.session_state.key = val`)
   - Widget return values (slider, checkbox, selectbox defaults)
   - Cache decorators as pass-through
3. `exec()` the truncated source into a module namespace (must register in `sys.modules` for `@dataclass` to work)
4. Extract and call functions directly

**Key mock requirements**:
- `session_state` must support `__setattr__`, `__getattr__`, `__contains__`, `get()`
- All widgets must support `__enter__`/`__exit__` for context manager use
- `columns()` and `tabs()` must return lists for unpacking
- The module must be in `sys.modules` with a valid `__name__` for `@dataclass` decorator

### Streamlit UI Testing (For simple interactions)

- Use browser tool to interact with the UI
- Sidebar may collapse on narrow viewports — the browser tool's default width might cause elements to be "offscreen"
- JavaScript `.click()` does NOT trigger Streamlit's widget protocol — native browser tool clicks are needed
- If sidebar elements are offscreen, consider using direct Python testing instead

## TrainingData Dataclass

`TrainingData` has many required fields. When constructing manually:
```python
TrainingData(
    x_obs, t_obs, c_obs,     # observation points
    x_ic, t_ic, c_ic,         # initial condition
    x_bc, t_bc, c_bc,         # boundary condition
    x_f, t_f,                 # collocation points
    x_grid, t_grid, C_fdm,   # FDM reference
    D_true, D_true_left, D_true_right,  # true D (can be dummy for RS)
    t_start,                  # start time
    x_exp, c_exp,             # final-time experiment
    x_exp_all, t_exp_all, c_exp_all,  # multitime experiment
    exp_time_indices,         # time indices
)
```

## FDM Function Signatures

`fdm_ternary_regular_solution(c0_full, x, dt, nsteps, mobility, theta_left, ...)` returns `(t_grid, C_history)` — note the order is t_grid first.

## Key Test Assertions

### RS Training (backward compatibility)
- `train_pinn_rs()` returns `(model, DataFrame)` tuple
- History DataFrame has 'loss' column with no NaN values
- `model.theta_display()` returns two arrays of length 3 (Omega_CoNi, Omega_CoTa, Omega_NiTa)
- All Omega values are finite

### PSIS Diagnostic
- `psis_diagnostic()` returns dict with keys: `pareto_k`, `ess_psis`, `n_evaluated`
- `pareto_k` is finite float; k < 0.5 = reliable
- `n_evaluated` matches the `max_eval` parameter

### σ Marginalization
- `mcmc_reliability()` with `marginalize_sigma=True` returns dict with `sigma_median` key
- `sigma_median` is positive finite float in (0, 1)
- `sigma_samples` length = mcmc_steps - burn_in

## Devin Secrets Needed

No secrets required. The app runs locally with synthetic data.

## Tips

- For CPU testing, reduce parameters: epochs=100-300, MCMC steps=10-30, PSIS evals=5, FDM grid nx=101
- RS mode with 100 epochs takes ~4s on CPU (direct Python), 5-10min via Streamlit
- Always revert temporary parameter changes after testing
- The `type(result).__name__` pattern is used instead of `isinstance()` due to Streamlit's re-execution model regenerating classes
