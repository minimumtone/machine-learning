# Comprehensive Validation Report
r_WS definition: $r_{WS} = (3 V_{atom} / 4\pi)^{1/3}$

## 1. Bootstrap Uncertainty for r_WS

### MP-B2
- Median std: 0.0208 Å
- Max std: 0.0786 Å (Si)
- Avg 95% CI width: 0.0924 Å

### MP-L12
- Median std: 0.0348 Å
- Max std: 0.2086 Å (Re)
- Avg 95% CI width: 0.1559 Å

### OQMD-B2
- Median std: 0.0180 Å
- Max std: 0.0272 Å (Li)
- Avg 95% CI width: 0.0729 Å

### OQMD-L12
- Median std: 0.0295 Å
- Max std: 0.0662 Å (Eu)
- Avg 95% CI width: 0.1134 Å

## 2. Element-Average r_WS(DFT) vs Contact Radii

### MP-B2
- RMSE: 0.1981 Å
- R²: 0.3554
- Mean r_contact/r_WS: 0.8998
- N elements: 68

### MP-L12
- RMSE: 0.2786 Å
- R²: 0.0215
- Mean r_contact/r_WS: 0.8592
- N elements: 72

### OQMD-B2
- RMSE: 0.2410 Å
- R²: 0.0882
- Mean r_contact/r_WS: 0.8718
- N elements: 72

### OQMD-L12
- RMSE: 0.1963 Å
- R²: 0.0217
- Mean r_contact/r_WS: 0.9052
- N elements: 58

## 3. Compound-Level r_WS Parity (DFT vs Predicted)

### MP-B2
- N compounds: 357
- RMSE: 0.0266 Å
- MAE: 0.0174 Å
- R²: 0.9756

### MP-L12
- N compounds: 848
- RMSE: 0.0615 Å
- MAE: 0.0428 Å
- R²: 0.9444

### OQMD-B2
- N compounds: 2539
- RMSE: 0.0551 Å
- MAE: 0.0356 Å
- R²: 0.9352

### OQMD-L12
- N compounds: 238
- RMSE: 0.0256 Å
- MAE: 0.0182 Å
- R²: 0.9731

## 4. Periodic Table: r_contact / r_WS Ratio

### MP-B2
- Mean ratio: 0.8998 ± 0.0689
- Range: 0.7806 – 1.0155

### MP-L12
- Mean ratio: 0.8592 ± 0.0865
- Range: 0.3050 – 0.9734

### OQMD-B2
- Mean ratio: 0.8718 ± 0.0676
- Range: 0.7142 – 1.0154

### OQMD-L12
- Mean ratio: 0.9052 ± 0.0692
- Range: 0.6428 – 1.1666

## 5. B2 vs L1₂ r_WS Structure Dependence

### MP
- Mean diff (B2−L1₂): -3.81%
- RMSE: 0.1180 Å
- N elements: 68

### OQMD
- Mean diff (B2−L1₂): +6.39%
- RMSE: 0.1242 Å
- N elements: 58

## 6. Element Group Analysis (r_contact / r_WS)

### MP-B2
| Group | N | Mean ratio | Std | Min | Max |
|-------|---|-----------|-----|-----|-----|
| 3d | 10 | 0.8497 | 0.0483 | 0.7806 | 0.9277 |
| 4d | 10 | 0.8758 | 0.0724 | 0.7961 | 0.9776 |
| 5d | 10 | 0.8760 | 0.0691 | 0.7867 | 0.9695 |
| Actinide | 5 | 0.9348 | 0.0437 | 0.8883 | 1.0155 |
| Lanthanide | 14 | 0.9577 | 0.0124 | 0.9416 | 0.9791 |
| Other | 4 | 0.9200 | 0.0959 | 0.8122 | 1.0155 |
| p-block | 10 | 0.8777 | 0.0477 | 0.8094 | 0.9976 |
| s-block | 5 | 0.9265 | 0.0756 | 0.8141 | 1.0155 |

### MP-L12
| Group | N | Mean ratio | Std | Min | Max |
|-------|---|-----------|-----|-----|-----|
| 3d | 10 | 0.8047 | 0.0626 | 0.7179 | 0.9262 |
| 4d | 10 | 0.8575 | 0.0493 | 0.7940 | 0.9463 |
| 5d | 10 | 0.8528 | 0.0405 | 0.7803 | 0.9155 |
| Actinide | 6 | 0.8783 | 0.0216 | 0.8367 | 0.8991 |
| Lanthanide | 14 | 0.9086 | 0.0149 | 0.8809 | 0.9293 |
| Other | 6 | 0.8009 | 0.2284 | 0.3050 | 0.9734 |
| p-block | 11 | 0.8710 | 0.0426 | 0.7876 | 0.9369 |
| s-block | 5 | 0.8676 | 0.0784 | 0.7202 | 0.9392 |

### OQMD-B2
| Group | N | Mean ratio | Std | Min | Max |
|-------|---|-----------|-----|-----|-----|
| 3d | 10 | 0.7992 | 0.0345 | 0.7579 | 0.8793 |
| 4d | 10 | 0.8367 | 0.0427 | 0.7910 | 0.9297 |
| 5d | 10 | 0.8336 | 0.0384 | 0.7910 | 0.9114 |
| Actinide | 6 | 0.9130 | 0.0256 | 0.8914 | 0.9569 |
| Lanthanide | 14 | 0.9330 | 0.0115 | 0.9145 | 0.9480 |
| Other | 6 | 0.9085 | 0.1100 | 0.7142 | 1.0154 |
| p-block | 11 | 0.8680 | 0.0368 | 0.8094 | 0.9159 |
| s-block | 5 | 0.9063 | 0.0825 | 0.7608 | 0.9836 |

### OQMD-L12
| Group | N | Mean ratio | Std | Min | Max |
|-------|---|-----------|-----|-----|-----|
| 3d | 10 | 0.8695 | 0.0630 | 0.7524 | 0.9422 |
| 4d | 8 | 0.8774 | 0.0476 | 0.8139 | 0.9363 |
| 5d | 7 | 0.9152 | 0.0522 | 0.8439 | 0.9869 |
| Actinide | 5 | 0.9520 | 0.0261 | 0.9346 | 1.0036 |
| Lanthanide | 13 | 0.9305 | 0.0169 | 0.9025 | 0.9567 |
| Other | 2 | 0.9150 | 0.0025 | 0.9125 | 0.9175 |
| p-block | 10 | 0.8657 | 0.0828 | 0.6428 | 0.9565 |
| s-block | 3 | 1.0135 | 0.1086 | 0.9271 | 1.1666 |

## 7. Summary: r_WS vs Contact Radii Across All Cases

| Case | N_compounds | N_elements | rWS med_std | rWS avg_CI | r_c/r_WS mean | r_c/r_WS std |
|------|------------|-----------|------------|-----------|--------------|-------------|
| MP-B2 | 357 | 68 | 0.0208 | 0.0924 | 0.8998 | 0.0689 |
| MP-L12 | 848 | 72 | 0.0348 | 0.1559 | 0.8592 | 0.0865 |
| OQMD-B2 | 2539 | 72 | 0.0180 | 0.0729 | 0.8718 | 0.0676 |
| OQMD-L12 | 238 | 58 | 0.0295 | 0.1134 | 0.9052 | 0.0692 |

