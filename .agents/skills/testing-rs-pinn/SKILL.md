---
name: testing-rs-pinn
description: Test the RS (Regular Solution) mode PINN diffusion module end-to-end. Use when verifying PINN residual computation, interdiffusion matrix, or training convergence.
---

## Overview

The main file `co_ni_ta_pinn_diffusion_reliability.py` (~8800 lines) is a Streamlit app that cannot be directly imported due to module-level UI code. All headless testing requires a Streamlit mock.

## Streamlit Mock Pattern

```python
import sys, os, types

class _SessionStateMock(dict):
    def __getattr__(self, n): return self.get(n, None)
    def __setattr__(self, n, v): self[n] = v
class _StopCallable:
    def __call__(self, *a, **kw): raise SystemExit(0)
class _CallableMock:
    def __call__(self, *a, **kw):
        if len(a)==1 and isinstance(a[0], int): return [_CallableMock() for _ in range(a[0])]
        return _CallableMock()
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def __iter__(self): return iter([_CallableMock(), _CallableMock()])
    def __bool__(self): return False
    def __getattr__(self, n): return _CallableMock()
    def __float__(self): return 0.0
    def __int__(self): return 0
    def __str__(self): return ""
    def __len__(self): return 0
    def __contains__(self, i): return False
class _StMock:
    def __init__(self): object.__setattr__(self, 'session_state', _SessionStateMock())
    def __getattr__(self, n):
        if n=='session_state': return object.__getattribute__(self, 'session_state')
        if n=='stop': return _StopCallable()
        return _CallableMock()
    def cache_data(self, *a, **kw): return lambda f: f

sys.modules['streamlit'] = _StMock()
for m in ['streamlit.components','streamlit.components.v1',
          'streamlit.runtime','streamlit.runtime.scriptrunner_utils']:
    sys.modules[m] = types.ModuleType(m)

import importlib.util
_spec = importlib.util.spec_from_file_location(
    'co_ni_ta_pinn_diffusion_reliability',
    '/path/to/co_ni_ta_pinn_diffusion_reliability.py')
_mod = importlib.util.module_from_spec(_spec)
sys.modules['co_ni_ta_pinn_diffusion_reliability'] = _mod
try:
    _spec.loader.exec_module(_mod)
except SystemExit:
    pass  # st.stop() halts UI code; functions/classes are already defined
```

## Key Functions and Classes

- `interdiffusion_matrix_rs_torch()`: Computes D̃(c) = M × (thermodynamic factor). Uses `torch.einsum` for full mobility matrix support.
- `TernaryRegularSolutionPINN`: Main model class. `use_fick_form=True` enables Fick residual.
- `_residual_fick()`: Fick form with frozen coefficient (D̃ detached).
- `_residual_onsager()`: Original Onsager form (M·∂μ/∂x).
- `make_training_data_rs()`: Generates FDM teacher data. Required params include `RT`, `x_interface`, `omega_width`, `phase_width`, `n_bc_each`, `t_start_fraction`, `n_exp_points`.
- `train_pinn_rs()`: Training loop. History DataFrame columns: `"data"`, `"ic"`, `"physics"` (NOT `"phys"`).

## Component Order Convention

- **Display order**: [Co, Ni, Ta] — used in UI and plots
- **Internal order**: [Ni, Ta, Co] — used in PDE residual computation
- Use `_reorder_theta_display_to_internal()` to convert Omega parameters

## Key Test Assertions

### D̃ correctness
Compare `interdiffusion_matrix_rs_torch` against finite-difference of `diffusion_potentials_regular_solution_torch`. Expected rel_err < 0.005 for all entries.

### Frozen coefficient
After `loss.backward()` on Fick residual:
- Network params (`model.net`) should have non-zero gradients
- Omega params (`model.theta_left_raw`) should have `grad=None` (by design — D̃ is detached)

### Off-diagonal mobility
With `M_offdiag != 0`, D̃ should differ from diagonal-only case. Diagonal-only case should match `M_kk * thermo_factor` exactly.

### Training convergence (short run)
With 300 epochs, `use_fick_form=True`, `direct_output=True`:
- `data_loss < 0.1`
- `physics_loss < 10.0` (Onsager without mu_floor was 7M)
- `max|pred(t=0.1) - pred(t=1.0)| > 0.02` (temporal evolution exists)

## Common Pitfalls

- History column is `"physics"` not `"phys"` — using wrong key causes KeyError
- `make_training_data_rs` requires many positional-like keyword args; check signature before calling
- `mu_floor` default is `5e-3` in both model and FDM; ensure consistency
- Omega gradient is intentionally zero in Fick form (frozen coefficient design) — this is NOT a bug
- `t_scale` is already incorporated into `mobility` by `train_pinn_rs` — do not multiply again in Fick residual

## Devin Secrets Needed

No secrets required for numerical testing. Streamlit UI testing requires no authentication.
