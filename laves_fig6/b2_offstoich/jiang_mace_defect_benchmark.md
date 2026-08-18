# MACE-MP-0 vs Jiang et al. DFT isolated point-defect energies in B2-NiAl

## Reference
- C. Jiang, L.-Q. Chen, Z.-K. Liu, *Acta Materialia* **53** (2005) 2643–2652.
  - DFT/GGA (VASP/PW91) using SQS, **Table 5** relaxed isolated point-defect formation enthalpies.
  - Reference states: fcc Al and ferromagnetic fcc Ni.
- MACE-MP-0: per-defect energies from `analysis/b2_defect_energies.csv` using the smallest-`n_defect` supercells available.

## Comparison table

| Defect type | Jiang DFT (eV/defect) | MACE-MP-0 (eV/defect) | MACE `n_defect` | MACE `x_Al` | `Δ = MACE - DFT` (eV) |
|---|---:|---:|---:|---:|---:|
| Ni vacancy (Al-rich, Ni sublattice) | 0.30 | 1.065 | 3 | 0.512 | +0.77 |
| Al antisite on Ni site (Al-rich) | 1.90 | 1.601 | 1 | 0.508 | −0.30 |
| Ni antisite on Al site (Ni-rich) | 0.99 | 0.562 | 1 | 0.492 | −0.43 |
| Al vacancy (Ni-rich, Al sublattice) | 1.83 | 1.720 | 3 | 0.488 | −0.11 |

## Implications for Al-rich defect competition

The Al-rich stable-defect competition is between **Ni vacancies** (`Va_Ni`) and **Al antisites on Ni sites** (`Al_Ni`):

- **Jiang DFT:** `E(Al_Ni) − E(Va_Ni) = 1.90 − 0.30 = +1.60 eV` → Ni vacancies are overwhelmingly favored.
- **MACE-MP-0:** `E(Al_Ni) − E(Va_Ni) ≈ 1.60 − 1.07 = +0.54 eV` → vacancies are still favored, but by a much smaller margin.

The MACE-MP-0 gap is therefore **≈1.06 eV smaller** than DFT. This means:
1. At finite temperature, MACE predicts **more thermal Al antisite mixing** on the Ni sublattice than DFT.
2. The Boltzmann hybrid `c_vac` values reported in this project are **upper/lower estimates** that are likely shifted toward antisite-rich configurations relative to DFT.
3. The lower `c_vac^hybrid` values (e.g., 0.106 at `x_Al=0.60` and 1473 K) relative to the pure structural-vacancy model (0.167) mainly reflect this reduced vacancy-vs-antisite energy gap; DFT would predict even stronger vacancy dominance and a hybrid curve closer to the pure-vacancy line.

## Relation to Medasani et al. (2016)

Medasani *et al.*, *npj Comput. Mater.* **2** (2016) 1 used 4×4×4 B2 supercells and machine learning to classify dominant defect types in 100 B2 intermetallics. For B2-NiAl, their high-throughput DFT classification ( vacancy on the Al-rich side, antisite on the Ni-rich side) is the same as Jiang *et al.* and the current project, and their supercell settings (128-site B2, constant-volume ionic relaxation) match the present MACE setup.
