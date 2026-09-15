# MACE-MP-0 Voronoi excess-volume analysis

MACE-MP-0 small, float64, CPU; 128-atom BCC cells; random occupations.
Periodic Voronoi volumes use scipy.spatial.Voronoi on a 3x3x3 image set.

## 1. Pure-element volumes

| element | V_i^pure (A^3) |
|---|---:|
| Al | 16.832937 |
| Hf | 22.101820 |
| Nb | 18.316904 |
| Ta | 18.309278 |
| Ti | 17.141430 |
| V | 13.197193 |
| Zr | 23.471479 |

## 2. Binary excess volumes

| pair | element | neighbor | Delta V (A^3) | Omega_MACE |
|---|---|---|---:|---:|
| Al-Nb | Al | Nb | -0.180355 | -0.016389 |
| Al-Nb | Nb | Al | -0.395715 | -0.016389 |
| Al-Ti | Al | Ti | -0.241983 | -0.022637 |
| Al-Ti | Ti | Al | -0.527086 | -0.022637 |
| Al-V | Al | V | -1.916726 | -0.033018 |
| Al-V | V | Al | 0.925204 | -0.033018 |
| Hf-Nb | Hf | Nb | -1.296935 | 0.001538 |
| Hf-Nb | Nb | Hf | 1.359087 | 0.001538 |
| Hf-Ta | Hf | Ta | -1.483341 | 0.000971 |
| Hf-Ta | Ta | Hf | 1.522574 | 0.000971 |
| Hf-Ti | Hf | Ti | -2.041979 | 0.001802 |
| Hf-Ti | Ti | Hf | 2.112707 | 0.001802 |
| Hf-Zr | Hf | Zr | 0.584445 | -0.000202 |
| Hf-Zr | Zr | Hf | -0.593649 | -0.000202 |
| Nb-Ta | Nb | Ta | -0.119538 | -0.002301 |
| Nb-Ta | Ta | Nb | 0.035273 | -0.002301 |
| Nb-Ti | Nb | Ti | -0.561054 | 0.005740 |
| Nb-Ti | Ti | Nb | 0.764568 | 0.005740 |
| Nb-V | Nb | V | -2.097208 | 0.000167 |
| Nb-V | V | Nb | 2.102480 | 0.000167 |
| Nb-Zr | Nb | Zr | 1.646121 | -0.017387 |
| Nb-Zr | Zr | Nb | -2.372716 | -0.017387 |
| Ta-Ti | Ta | Ti | -0.469137 | 0.002281 |
| Ta-Ti | Ti | Ta | 0.550009 | 0.002281 |
| Ta-Zr | Ta | Zr | 1.706294 | -0.022099 |
| Ta-Zr | Zr | Ta | -2.629601 | -0.022099 |
| Ti-V | Ti | V | -1.618345 | -0.001009 |
| Ti-V | V | Ti | 1.587719 | -0.001009 |
| Ti-Zr | Ti | Zr | 2.605419 | -0.000426 |
| Ti-Zr | Zr | Ti | -2.622740 | -0.000426 |

## 3. HfNbTaTiZr

| element | Delta V_HEA (A^3) | seed std (A^3) | Delta V_pred (A^3) | f_i | flag |
|---|---:|---:|---:|---:|---|
| Hf | -1.902289 | 0.142698 | -1.698836 | 1.119760 |  |
| Nb | 0.945288 | 0.028905 | 0.927421 | 1.019266 |  |
| Ta | 1.199858 | 0.034577 | 1.116140 | 1.075007 |  |
| Ti | 2.110708 | 0.060694 | 2.410076 | 0.875785 |  |
| Zr | -3.133792 | 0.145269 | -3.297869 | 0.950248 |  |

Cell: V_HEA/N = 19.712137 A^3; V_Vegard = 19.861334 A^3; Alonso full-additivity denominator = -0.103375 A^3; f_cell = 1.443265.

## 3. AlNbTiV

| element | Delta V_HEA (A^3) | seed std (A^3) | Delta V_pred (A^3) | f_i | flag |
|---|---:|---:|---:|---:|---|
| Al | -0.327533 | 0.125733 | -1.169532 | 0.280055 |  |
| Nb | -1.376691 | 0.273041 | -1.526988 | 0.901573 |  |
| Ti | -0.312486 | 0.264025 | -0.690432 | 0.452596 |  |
| V | 2.499556 | 0.100969 | 2.307702 | 1.083136 |  |

Cell: V_HEA/N = 16.492827 A^3; V_Vegard = 16.372116 A^3; Alonso full-additivity denominator = -0.269813 A^3; f_cell = -0.447389.
