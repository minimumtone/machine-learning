---
name: testing-pinn-diffusion
description: Test the PINN ternary diffusion Streamlit app. Use when verifying changes to fig11_co_ni_ta_pinn_reliability_v23_multitime_pseudoexp.py.
---

# Testing the PINN Diffusion App

## Key Constraints

### Module Import Fails
The main file (`fig11_co_ni_ta_pinn_reliability_v23_multitime_pseudoexp.py`) cannot be imported as a Python module because Streamlit top-level UI code runs during import and crashes with `AttributeError: 'NoneType' object has no attribute 'model'`. 

**Workaround**: Replicate the exact logic from the source file in standalone test scripts. Copy the specific function code you need to test rather than importing it.

### CPU Training is Too Slow for E2E
Full PINN training takes hours on CPU. MCMC/Laplace/PSIS computations can take 30+ minutes even with reduced parameters.

**Workaround**: Test functions at the unit level with fast mock inputs. For MCMC, use `n_steps=10, burn_in=2`. For FDM, use `nx=81, t_max=0.5, nt_save=10`.

### Streamlit UI Testing
Streamlit UI interactions are slow and unreliable via automation. The app starts on port 8501 by default.

**Workaround**: Use `py_compile.compile()` to verify syntax, then `streamlit run <file> &` + `curl localhost:8501` to verify HTTP 200 startup. For functional testing, use direct Python function calls.

## Test Patterns

### Simplex Constraint Testing
To test that composition outputs satisfy simplex (all ≥ 0, sum = 1.0):
1. Create a model with adversarial weights that force sigmoid outputs near 1.0
2. Check `output.sum(dim=1)` is within atol=1e-6 of 1.0
3. Check `(output >= 0).all()`
4. Check Co column has some rows > 0 (gradient not dead)

### FDM Boundary Preservation Testing
To verify boundary conditions aren't drifted by the clip logic:
1. Run FDM with right BC `[Ni=0.90, Ta=0.10]` (sum=1.0 triggers `> 0.999` check)
2. Compare final `U[-1]` against initial BC — should match within 1e-10
3. Run same FDM WITHOUT the fix to demonstrate drift (validates the test itself)
4. Check `clip_events << n_steps` (boundary excluded from clip)

### MCMC proposal_cov Augment Testing
To verify `marginalize_sigma=True` + `proposal_cov` doesn't crash:
1. Create mock `proposal_cov` of shape `(dim_theta, dim_theta)`
2. Set `marginalize_sigma=True` (extends state to `dim_theta+1`)
3. Verify the augment logic produces `(dim_theta+1, dim_theta+1)` PD matrix
4. Verify Cholesky decomposition succeeds
5. Verify OLD code (without augment) produces `ValueError: proposal_cov must have shape`

## Testing Shell Commands

```bash
# Syntax check
python -c "import py_compile; py_compile.compile('fig11_co_ni_ta_pinn_reliability_v23_multitime_pseudoexp.py', doraise=True); print('OK')"

# Streamlit startup check
streamlit run fig11_co_ni_ta_pinn_reliability_v23_multitime_pseudoexp.py --server.headless true &
sleep 10
curl -s -o /dev/null -w '%{http_code}' http://localhost:8501
```

## Devin Secrets Needed
No secrets required for testing this app. All tests run locally with synthetic data.
