# MACE-MP-0 vs Jiang et al. DFT isolated point-defect energies in B2-NiAl

## Reference
- C. Jiang, L.-Q. Chen, Z.-K. Liu, *Acta Materialia* **53** (2005) 2643–2652.
  - DFT/GGA (VASP/PW91) using SQS, **Table 5** relaxed isolated point-defect formation enthalpies.
  - Reference states: fcc Al and ferromagnetic fcc Ni.
- MACE-MP-0: isolated per-defect energies from `analysis/b2_defect_energies.csv`.
  - Antisites are `n_defect=1` supercells directly (the smallest possible isolated defect).
  - Vacancies are extrapolated to `n_defect=1` from `n_defect<=10` dilute supercells.
  - All MACE values are at `x_Al = 0.492–0.508`, i.e. the same isolated-defect limit (`x_Al -> 0.5`) as the DFT reference. Composition units are therefore matched.

## Comparison table

| Defect type | Jiang DFT (eV/defect) | MACE-MP-0 (eV/defect) | MACE `n_defect` | MACE `x_Al` | `Δ = MACE - DFT` (eV) |
|---|---:|---:|---:|---:|---:|
| Ni vacancy (Al-rich, Ni sublattice) | 0.30 | 1.032 | 1 (extrap.) | 0.504 | **+0.73** |
| Al antisite on Ni site (Al-rich) | 1.90 | 1.601 | 1 | 0.508 | −0.30 |
| Ni antisite on Al site (Ni-rich) | 0.99 | 0.562 | 1 | 0.492 | −0.43 |
| Al vacancy (Ni-rich, Al sublattice) | 1.83 | 1.678 | 1 (extrap.) | 0.496 | −0.15 |

## Implications for Al-rich defect competition

The Al-rich stable-defect competition is between **Ni vacancies** (`Va_Ni`) and **Al antisites on Ni sites** (`Al_Ni`):

- **Jiang DFT:** `E(Al_Ni) − E(Va_Ni) = 1.90 − 0.30 = +1.60 eV` → Ni vacancies are overwhelmingly favored.
- **MACE-MP-0:** `E(Al_Ni) − E(Va_Ni) ≈ 1.60 − 1.03 = +0.57 eV` → vacancies are still favored, but by a much smaller margin.

The MACE-MP-0 gap is therefore **≈1.03 eV smaller** than DFT. This means:
1. At finite temperature, MACE predicts **more thermal Al antisite mixing** on the Ni sublattice than DFT would.
2. The Boltzmann hybrid `c_vac` values are at best semi-quantitative; they are likely shifted toward higher antisite fractions because MACE underestimates antisite formation energies and overestimates Ni vacancy formation energy.
3. The lower `c_vac^hybrid` values (e.g., 0.106 at `x_Al=0.60` and 1473 K) relative to the pure structural-vacancy model (0.167) mainly reflect this reduced vacancy-vs-antisite energy gap. A DFT-calibrated hybrid would be closer to the pure-vacancy line.

## MACE-MP-0 accuracy issue

- Ni vacancy formation energy is **overestimated by +0.73 eV** (vacancies look much less stable than in DFT).
- Al antisite and Ni antisite formation energies are **underestimated by ≈0.3–0.4 eV** (antisites look more stable than in DFT).

As a result, **MACE-MP-0 gets the qualitative dominant-defect type right** (vacancies on the Al-rich side, antisites on the Ni-rich side) but **fails quantitatively for the energy differences that set defect concentrations**. Quantitative defect concentrations should not be trusted from MACE alone; they require DFT calibration or a properly constructed 4SL/8SL sublattice model.

## Relation to Medasani et al. (2016)

Medasani *et al.*, *npj Comput. Mater.* **2** (2016) 1 used 4×4×4 B2 supercells and machine learning to classify dominant defect types in 100 B2 intermetallics. For B2-NiAl, their high-throughput DFT classification (vacancy on the Al-rich side, antisite on the Ni-rich side) is the same as Jiang *et al.* and the current project, and their supercell settings (128-site B2, constant-volume ionic relaxation) match the present MACE setup.
