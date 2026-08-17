# C14-Nb(Ni,Al)2 at x=0.5: Yamanouchi weighted-average hypothesis check

- V_pure_Nb (MACE) = 18.186 Å³/atom
- V_fcc_Ni (MACE) = 10.809 Å³/atom
- V_fcc_Al (MACE) = 16.733 Å³/atom
- V_B2_NiAl (MACE) = 11.967 Å³/atom
- V_fcc_NiAl at x=0.5 (SQS average) = 12.565 Å³/atom
- V_C14_NbNiAl at x=0.5 (observed, MACE SQS average) = 14.280 Å³/atom

| model | expression | V (Å³/atom) | |V - V_C14| |
|---|---|---|---|
| pure-element average | (V_Nb + V_Ni + V_Al)/3 | 15.243 | 0.962 |
| fcc-NiAl weighted | (V_Nb + 2 V_fcc_NiAl(0.5))/3 | 14.439 | 0.158 |
| B2-NiAl weighted | (V_Nb + 2 V_B2_NiAl)/3 | 14.040 | 0.240 |

**Result**: The compound-derived weighted averages (V_1, V_3) are closer to the observed C14 volume than the pure-element average V_2. Best match: fcc-NiAl weighted (ΔV=0.158 Å³/atom), consistent with Yamanouchi & Miura's central claim.