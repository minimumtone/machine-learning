#!/usr/bin/env python3
"""
Comprehensive comparison of HEA lattice parameter predictions:
  1. This work (pairwise max-contact model with DFT-derived effective radii)
  2. Alonso2021 (volume size factor method, Eq.10)
  3. Vegard's law (simple weighted average of pure-element atomic volumes)
  4. King1966 atomic volumes (Vegard-type prediction using King's V_atom data)

Data sources:
  - Alonso2021 Table 2: 68 cubic HEAs with experimental lattice parameters
  - King1966 Table II: Volume size factors for 469 binary solid solutions
  - This work: Effective radii from four_case_comparison_study.py optimization

Output:
  - Parity plots comparing all methods
  - RMSE / MAE bar charts
  - Error distribution histograms
  - Appendix-ready data tables (CSV)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
import json, sys, os

# Japanese font setup
for fp in fm.findSystemFonts():
    if "ipag" in fp.lower() or "ipagothic" in fp.lower():
        plt.rcParams["font.family"] = fm.FontProperties(fname=fp).get_name()
        break
else:
    for fp in fm.findSystemFonts():
        if "wqy" in fp.lower():
            plt.rcParams["font.family"] = fm.FontProperties(fname=fp).get_name()
            break

# =====================================================================
# Font settings (doubled for presentation quality)
# =====================================================================
plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "figure.dpi": 150,
})

OUTDIR = Path("alonso_king_output")
OUTDIR.mkdir(exist_ok=True)

# =====================================================================
# 1. Alonso2021 Table 2 — 68 cubic HEAs
#    Columns: name, composition_dict, structure, a_exp, a_vegard, a_eq10,
#             err_vegard_pct, err_eq10_pct, ref_num
# =====================================================================
ALONSO_TABLE2 = [
    # BCC HEAs
    {"name": "W$_{0.273}$Nb$_{0.227}$Mo$_{0.256}$Ta$_{0.244}$",
     "comp": {"W":0.273,"Nb":0.227,"Mo":0.256,"Ta":0.244}, "struct":"BCC",
     "a_exp":3.2134, "a_vegard":3.2263, "a_eq10":3.2148, "ref":"[23]"},
    {"name": "W$_{0.25}$Nb$_{0.22}$Mo$_{0.26}$Ta$_{0.27}$",
     "comp": {"W":0.25,"Nb":0.22,"Mo":0.26,"Ta":0.27}, "struct":"BCC",
     "a_exp":3.24, "a_vegard":3.23, "a_eq10":3.22, "ref":"[24]"},
    {"name": "NbMoTaW",
     "comp": {"Nb":0.25,"Mo":0.25,"Ta":0.25,"W":0.25}, "struct":"BCC",
     "a_exp":3.222, "a_vegard":3.231, "a_eq10":3.217, "ref":"[25]"},
    {"name": "W$_{0.211}$Nb$_{0.206}$Mo$_{0.217}$Ta$_{0.156}$V$_{0.210}$",
     "comp": {"W":0.211,"Nb":0.206,"Mo":0.217,"Ta":0.156,"V":0.210}, "struct":"BCC",
     "a_exp":3.1832, "a_vegard":3.1849, "a_eq10":3.1804, "ref":"[23]"},
    {"name": "WNbMoTaVTi",
     "comp": {"W":1/6,"Nb":1/6,"Mo":1/6,"Ta":1/6,"V":1/6,"Ti":1/6}, "struct":"BCC",
     "a_exp":3.216, "a_vegard":3.209, "a_eq10":3.188, "ref":"[26]"},
    {"name": "NbZrHfVTi (a)",
     "comp": {"Nb":0.2,"Zr":0.2,"Hf":0.2,"V":0.2,"Ti":0.2}, "struct":"BCC",
     "a_exp":3.377, "a_vegard":3.361, "a_eq10":3.374, "ref":"[36]"},
    {"name": "Al$_{0.262}$Cr$_{0.241}$Fe$_{0.259}$Mo$_{0.235}$V$_{0.003}$",
     "comp": {"Al":0.262,"Cr":0.241,"Fe":0.259,"Mo":0.235,"V":0.003}, "struct":"BCC",
     "a_exp":3.01, "a_vegard":3.04, "a_eq10":3.02, "ref":"[37]"},
    {"name": "Al$_{0.200}$Cr$_{0.243}$Fe$_{0.232}$Mo$_{0.166}$V$_{0.158}$",
     "comp": {"Al":0.200,"Cr":0.243,"Fe":0.232,"Mo":0.166,"V":0.158}, "struct":"BCC",
     "a_exp":2.98, "a_vegard":3.02, "a_eq10":3.01, "ref":"[37]"},
    {"name": "Al$_{0.180}$Ni$_{0.189}$Cu$_{0.214}$Fe$_{0.196}$Cr$_{0.220}$",
     "comp": {"Al":0.180,"Ni":0.189,"Cu":0.214,"Fe":0.196,"Cr":0.220}, "struct":"BCC",
     "a_exp":2.894, "a_vegard":2.926, "a_eq10":2.918, "ref":"[38]"},
    {"name": "NbZrHfTi",
     "comp": {"Nb":0.25,"Zr":0.25,"Hf":0.25,"Ti":0.25}, "struct":"BCC",
     "a_exp":3.438, "a_vegard":3.435, "a_eq10":3.428, "ref":"[40]"},
    {"name": "Nb$_{0.200}$Mo$_{0.208}$Cr$_{0.187}$Ti$_{0.202}$V$_{0.203}$",
     "comp": {"Nb":0.200,"Mo":0.208,"Cr":0.187,"Ti":0.202,"V":0.203}, "struct":"BCC",
     "a_exp":3.140, "a_vegard":3.139, "a_eq10":3.138, "ref":"[41]"},
    {"name": "NbTaTiV (a)",
     "comp": {"Nb":0.25,"Ta":0.25,"Ti":0.25,"V":0.25}, "struct":"BCC",
     "a_exp":3.23, "a_vegard":3.23, "a_eq10":3.23, "ref":"[42]"},
    {"name": "NbTaTiV (b)",
     "comp": {"Nb":0.25,"Ta":0.25,"Ti":0.25,"V":0.25}, "struct":"BCC",
     "a_exp":3.2206, "a_vegard":3.2319, "a_eq10":3.2299, "ref":"[43]"},
    {"name": "Nb$_{0.27}$Mo$_{0.21}$Ta$_{0.27}$W$_{0.24}$",
     "comp": {"Nb":0.27,"Mo":0.21,"Ta":0.27,"W":0.24}, "struct":"BCC",
     "a_exp":3.218, "a_vegard":3.226, "a_eq10":3.213, "ref":"[44]"},
    {"name": "Nb$_{0.22}$Mo$_{0.18}$Ta$_{0.28}$W$_{0.31}$",
     "comp": {"Nb":0.22,"Mo":0.18,"Ta":0.28,"W":0.31}, "struct":"BCC",
     "a_exp":3.216, "a_vegard":3.222, "a_eq10":3.210, "ref":"[44]"},
    {"name": "Nb$_{0.24}$Mo$_{0.27}$Ta$_{0.25}$W$_{0.24}$",
     "comp": {"Nb":0.24,"Mo":0.27,"Ta":0.25,"W":0.24}, "struct":"BCC",
     "a_exp":3.214, "a_vegard":3.229, "a_eq10":3.215, "ref":"[44]"},
    {"name": "Nb$_{0.24}$Mo$_{0.18}$Ta$_{0.36}$W$_{0.22}$",
     "comp": {"Nb":0.24,"Mo":0.18,"Ta":0.36,"W":0.22}, "struct":"BCC",
     "a_exp":3.228, "a_vegard":3.245, "a_eq10":3.234, "ref":"[44]"},
    {"name": "Nb$_{0.216}$Mo$_{0.230}$Ta$_{0.281}$W$_{0.273}$",
     "comp": {"Nb":0.216,"Mo":0.230,"Ta":0.281,"W":0.273}, "struct":"BCC",
     "a_exp":3.2034, "a_vegard":3.2303, "a_eq10":3.2177, "ref":"[45]"},
    {"name": "Ti$_2$ZrHfV$_{0.5}$Mo$_{0.2}$",
     "comp": {"Ti":2/4.7,"Zr":1/4.7,"Hf":1/4.7,"V":0.5/4.7,"Mo":0.2/4.7}, "struct":"BCC",
     "a_exp":3.4584, "a_vegard":3.3805, "a_eq10":3.3845, "ref":"[46]"},
    {"name": "Ti$_{0.262}$Nb$_{0.255}$Ta$_{0.121}$Zr$_{0.242}$Al$_{0.120}$",
     "comp": {"Ti":0.262,"Nb":0.255,"Ta":0.121,"Zr":0.242,"Al":0.120}, "struct":"BCC",
     "a_exp":3.355, "a_vegard":3.363, "a_eq10":3.360, "ref":"[47]"},
    {"name": "NbTaTiV$_{0.333}$Cr$_{0.309}$Fe$_{0.308}$Ta$_{0.025}$W$_{0.025}$",
     "comp": {"V":0.333,"Cr":0.309,"Fe":0.308,"Ta":0.025,"W":0.025}, "struct":"BCC",
     "a_exp":2.935, "a_vegard":2.948, "a_eq10":2.930, "ref":"[49]"},
    {"name": "NbMoTaVTi (b)",
     "comp": {"Nb":0.2,"Mo":0.2,"Ta":0.2,"V":0.2,"Ti":0.2}, "struct":"BCC",
     "a_exp":3.1945, "a_vegard":3.2153, "a_eq10":3.2055, "ref":"[56]"},
    {"name": "Nb$_{0.199}$Hf$_{0.198}$Ta$_{0.175}$V$_{0.212}$Ti$_{0.217}$",
     "comp": {"Nb":0.199,"Hf":0.198,"Ta":0.175,"V":0.212,"Ti":0.217}, "struct":"BCC",
     "a_exp":3.279, "a_vegard":3.295, "a_eq10":3.290, "ref":"[57]"},
    {"name": "Nb$_{0.304}$Mo$_{0.037}$Ta$_{0.051}$Zr$_{0.290}$Ti$_{0.319}$",
     "comp": {"Nb":0.304,"Mo":0.037,"Ta":0.051,"Zr":0.290,"Ti":0.319}, "struct":"BCC",
     "a_exp":3.285, "a_vegard":3.381, "a_eq10":3.368, "ref":"[58]"},
    {"name": "Nb$_{0.255}$Mo$_{0.207}$Ta$_{0.190}$Zr$_{0.131}$Ti$_{0.217}$",
     "comp": {"Nb":0.255,"Mo":0.207,"Ta":0.190,"Zr":0.131,"Ti":0.217}, "struct":"BCC",
     "a_exp":3.24, "a_vegard":3.31, "a_eq10":3.28, "ref":"[58]"},
    {"name": "Nb$_{0.275}$Mo$_{0.095}$Ta$_{0.091}$Zr$_{0.256}$Ti$_{0.284}$",
     "comp": {"Nb":0.275,"Mo":0.095,"Ta":0.091,"Zr":0.256,"Ti":0.284}, "struct":"BCC",
     "a_exp":3.40, "a_vegard":3.47, "a_eq10":3.46, "ref":"[58]"},
    {"name": "NbZrHfVTi (b)",
     "comp": {"Nb":0.2,"Zr":0.2,"Hf":0.2,"V":0.2,"Ti":0.2}, "struct":"BCC",
     "a_exp":3.3663, "a_vegard":3.3613, "a_eq10":3.3582, "ref":"[59]"},
    {"name": "Nb$_{0.238}$V$_{0.245}$Al$_{0.266}$Ti$_{0.251}$",
     "comp": {"Nb":0.238,"V":0.245,"Al":0.266,"Ti":0.251}, "struct":"BCC",
     "a_exp":3.18, "a_vegard":3.07, "a_eq10":3.10, "ref":"[62]"},
    # FCC HEAs
    {"name": "CoCrFeNi",
     "comp": {"Co":0.25,"Cr":0.25,"Fe":0.25,"Ni":0.25}, "struct":"FCC",
     "a_exp":3.575, "a_vegard":3.579, "a_eq10":3.587, "ref":"[28]"},
    {"name": "CoCrFeNiMn",
     "comp": {"Co":0.2,"Cr":0.2,"Fe":0.2,"Mn":0.2,"Ni":0.2}, "struct":"FCC",
     "a_exp":3.597, "a_vegard":3.594, "a_eq10":3.602, "ref":"[28]"},
    {"name": "Co$_{0.204}$Cr$_{0.205}$Fe$_{0.202}$Mn$_{0.194}$Ni$_{0.195}$",
     "comp": {"Co":0.204,"Cr":0.205,"Fe":0.202,"Mn":0.194,"Ni":0.195}, "struct":"FCC",
     "a_exp":3.59, "a_vegard":3.59, "a_eq10":3.60, "ref":"[29]"},
    {"name": "Co$_{0.203}$Cr$_{0.194}$Fe$_{0.206}$Mn$_{0.201}$Ni$_{0.196}$",
     "comp": {"Co":0.203,"Cr":0.194,"Fe":0.206,"Mn":0.201,"Ni":0.196}, "struct":"FCC",
     "a_exp":3.60, "a_vegard":3.59, "a_eq10":3.60, "ref":"[30]"},
    {"name": "Cr$_{0.127}$Fe$_{0.498}$Ni$_{0.111}$Mn$_{0.264}$",
     "comp": {"Cr":0.127,"Fe":0.498,"Ni":0.111,"Mn":0.264}, "struct":"FCC",
     "a_exp":3.61, "a_vegard":3.62, "a_eq10":3.62, "ref":"[31]"},
    {"name": "Co$_{0.20}$Cr$_{0.20}$Fe$_{0.40}$Ni$_{0.10}$Mn$_{0.10}$",
     "comp": {"Co":0.20,"Cr":0.20,"Fe":0.40,"Ni":0.10,"Mn":0.10}, "struct":"FCC",
     "a_exp":3.587, "a_vegard":3.598, "a_eq10":3.605, "ref":"[32]"},
    {"name": "Co$_{0.211}$Cr$_{0.187}$Fe$_{0.342}$Ni$_{0.063}$Mn$_{0.197}$",
     "comp": {"Co":0.211,"Cr":0.187,"Fe":0.342,"Ni":0.063,"Mn":0.197}, "struct":"FCC",
     "a_exp":3.588, "a_vegard":3.605, "a_eq10":3.611, "ref":"[33]"},
    {"name": "Ru$_{0.185}$Rh$_{0.156}$Pd$_{0.182}$Os$_{0.143}$Ir$_{0.159}$Pt$_{0.174}$",
     "comp": {"Ru":0.185,"Rh":0.156,"Pd":0.182,"Os":0.143,"Ir":0.159,"Pt":0.174}, "struct":"FCC",
     "a_exp":3.8473, "a_vegard":3.8462, "a_eq10":3.8471, "ref":"[34]"},
    {"name": "CoCrCuNiZn",
     "comp": {"Co":0.2,"Cr":0.2,"Cu":0.2,"Ni":0.2,"Zn":0.2}, "struct":"BCC",
     "a_exp":2.8831, "a_vegard":2.9012, "a_eq10":2.8815, "ref":"[39]"},
    {"name": "V$_{0.098}$Co$_{0.301}$Cr$_{0.095}$Fe$_{0.455}$Ni$_{0.051}$",
     "comp": {"V":0.098,"Co":0.301,"Cr":0.095,"Fe":0.455,"Ni":0.051}, "struct":"FCC",
     "a_exp":3.582, "a_vegard":3.610, "a_eq10":3.604, "ref":"[48]"},
    {"name": "Co$_{0.286}$Al$_{0.071}$Fe$_{0.286}$Ni$_{0.286}$Mn$_{0.071}$",
     "comp": {"Co":0.286,"Al":0.071,"Fe":0.286,"Ni":0.286,"Mn":0.071}, "struct":"FCC",
     "a_exp":3.6084, "a_vegard":3.6061, "a_eq10":3.5923, "ref":"[50]"},
    {"name": "CoCrFeNiPd (a)",
     "comp": {"Co":0.2,"Cr":0.2,"Fe":0.2,"Ni":0.2,"Pd":0.2}, "struct":"FCC",
     "a_exp":3.6473, "a_vegard":3.6455, "a_eq10":3.6658, "ref":"[51]"},
    {"name": "CoCrFeNiPd (b)",
     "comp": {"Co":0.2,"Cr":0.2,"Fe":0.2,"Ni":0.2,"Pd":0.2}, "struct":"FCC",
     "a_exp":3.6803, "a_vegard":3.6455, "a_eq10":3.6658, "ref":"[52]"},
    {"name": "Co$_{0.244}$Cr$_{0.244}$Fe$_{0.244}$Ni$_{0.244}$Al$_{0.024}$",
     "comp": {"Co":0.244,"Cr":0.244,"Fe":0.244,"Ni":0.244,"Al":0.024}, "struct":"FCC",
     "a_exp":3.58, "a_vegard":3.59, "a_eq10":3.59, "ref":"[53]"},
    {"name": "Co$_{0.314}$Al$_{0.029}$Fe$_{0.318}$Ni$_{0.307}$Mn$_{0.032}$",
     "comp": {"Co":0.314,"Al":0.029,"Fe":0.318,"Ni":0.307,"Mn":0.032}, "struct":"FCC",
     "a_exp":3.5862, "a_vegard":3.5798, "a_eq10":3.5796, "ref":"[54]"},
    {"name": "Co$_{0.290}$Al$_{0.067}$Fe$_{0.288}$Ni$_{0.268}$Mn$_{0.087}$",
     "comp": {"Co":0.290,"Al":0.067,"Fe":0.288,"Ni":0.268,"Mn":0.087}, "struct":"FCC",
     "a_exp":3.600, "a_vegard":3.606, "a_eq10":3.593, "ref":"[54]"},
    {"name": "Co$_{0.2}$Al$_{0.1}$Fe$_{0.3}$Ni$_{0.4}$",
     "comp": {"Co":0.2,"Al":0.1,"Fe":0.3,"Ni":0.4}, "struct":"FCC",
     "a_exp":3.5936, "a_vegard":3.6132, "a_eq10":3.5936, "ref":"[55]"},
    {"name": "Ir$_{0.191}$Pt$_{0.195}$Pd$_{0.207}$Rh$_{0.199}$Ru$_{0.208}$",
     "comp": {"Ir":0.191,"Pt":0.195,"Pd":0.207,"Rh":0.199,"Ru":0.208}, "struct":"FCC",
     "a_exp":3.856, "a_vegard":3.849, "a_eq10":3.851, "ref":"[60]"},
    {"name": "Co$_{0.237}$Cr$_{0.232}$Fe$_{0.245}$Ni$_{0.209}$Mn$_{0.077}$",
     "comp": {"Co":0.237,"Cr":0.232,"Fe":0.245,"Ni":0.209,"Mn":0.077}, "struct":"FCC",
     "a_exp":3.58, "a_vegard":3.59, "a_eq10":3.59, "ref":"[61]"},
    {"name": "Co$_{0.269}$Cr$_{0.250}$Fe$_{0.249}$Ni$_{0.184}$Mo$_{0.048}$",
     "comp": {"Co":0.269,"Cr":0.250,"Fe":0.249,"Ni":0.184,"Mo":0.048}, "struct":"FCC",
     "a_exp":3.585, "a_vegard":3.603, "a_eq10":3.606, "ref":"[63]"},
    {"name": "Co$_{0.247}$Cr$_{0.245}$Fe$_{0.239}$Ni$_{0.246}$Mo$_{0.024}$",
     "comp": {"Co":0.247,"Cr":0.245,"Fe":0.239,"Ni":0.246,"Mo":0.024}, "struct":"FCC",
     "a_exp":3.604, "a_vegard":3.589, "a_eq10":3.594, "ref":"[64]"},
    {"name": "Co$_{0.238}$Cr$_{0.238}$Fe$_{0.238}$Ni$_{0.238}$Mo$_{0.048}$",
     "comp": {"Co":0.238,"Cr":0.238,"Fe":0.238,"Ni":0.238,"Mo":0.048}, "struct":"FCC",
     "a_exp":3.595, "a_vegard":3.599, "a_eq10":3.602, "ref":"[65]"},
    {"name": "Co$_{0.204}$Cr$_{0.197}$Fe$_{0.299}$Ni$_{0.299}$",
     "comp": {"Co":0.204,"Cr":0.197,"Fe":0.299,"Ni":0.299}, "struct":"FCC",
     "a_exp":3.5759, "a_vegard":3.5764, "a_eq10":3.5782, "ref":"[66,67]"},
    {"name": "Co$_{0.305}$Cr$_{0.208}$Fe$_{0.193}$Ni$_{0.294}$",
     "comp": {"Co":0.305,"Cr":0.208,"Fe":0.193,"Ni":0.294}, "struct":"FCC",
     "a_exp":3.5695, "a_vegard":3.5704, "a_eq10":3.5721, "ref":"[66,67]"},
    {"name": "Co$_{0.296}$Cr$_{0.213}$Fe$_{0.303}$Ni$_{0.188}$",
     "comp": {"Co":0.296,"Cr":0.213,"Fe":0.303,"Ni":0.188}, "struct":"FCC",
     "a_exp":3.5741, "a_vegard":3.5801, "a_eq10":3.5822, "ref":"[66,67]"},
    {"name": "Co$_{0.265}$Cr$_{0.205}$Fe$_{0.271}$Ni$_{0.260}$",
     "comp": {"Co":0.265,"Cr":0.205,"Fe":0.271,"Ni":0.260}, "struct":"FCC",
     "a_exp":3.573, "a_vegard":3.576, "a_eq10":3.578, "ref":"[66,67]"},
    {"name": "Co$_{0.220}$Cr$_{0.244}$Fe$_{0.230}$Ni$_{0.307}$",
     "comp": {"Co":0.220,"Cr":0.244,"Fe":0.230,"Ni":0.307}, "struct":"FCC",
     "a_exp":3.5737, "a_vegard":3.5758, "a_eq10":3.577, "ref":"[66,67]"},
    {"name": "Co$_{0.229}$Cr$_{0.236}$Fe$_{0.325}$Ni$_{0.210}$",
     "comp": {"Co":0.229,"Cr":0.236,"Fe":0.325,"Ni":0.210}, "struct":"FCC",
     "a_exp":3.5795, "a_vegard":3.5834, "a_eq10":3.5843, "ref":"[66,67]"},
    {"name": "Co$_{0.315}$Cr$_{0.245}$Fe$_{0.245}$Ni$_{0.220}$",
     "comp": {"Co":0.315,"Cr":0.245,"Fe":0.245,"Ni":0.220}, "struct":"FCC",
     "a_exp":3.5704, "a_vegard":3.6079, "a_eq10":3.609, "ref":"[66,67]"},
    {"name": "Co$_{0.216}$Cr$_{0.227}$Fe$_{0.276}$Ni$_{0.281}$",
     "comp": {"Co":0.216,"Cr":0.227,"Fe":0.276,"Ni":0.281}, "struct":"FCC",
     "a_exp":3.5752, "a_vegard":3.5779, "a_eq10":3.5792, "ref":"[66,67]"},
    {"name": "Co$_{0.277}$Cr$_{0.233}$Fe$_{0.233}$Ni$_{0.277}$",
     "comp": {"Co":0.277,"Cr":0.233,"Fe":0.233,"Ni":0.277}, "struct":"FCC",
     "a_exp":3.572, "a_vegard":3.5998, "a_eq10":3.6013, "ref":"[66,67]"},
    {"name": "Co$_{0.273}$Cr$_{0.229}$Fe$_{0.284}$Ni$_{0.214}$",
     "comp": {"Co":0.273,"Cr":0.229,"Fe":0.284,"Ni":0.214}, "struct":"FCC",
     "a_exp":3.5751, "a_vegard":3.5801, "a_eq10":3.5815, "ref":"[66,67]"},
    {"name": "Co$_{0.188}$Cr$_{0.214}$Fe$_{0.198}$Ni$_{0.400}$",
     "comp": {"Co":0.188,"Cr":0.214,"Fe":0.198,"Ni":0.400}, "struct":"FCC",
     "a_exp":3.5708, "a_vegard":3.5692, "a_eq10":3.5711, "ref":"[66,67]"},
    {"name": "Co$_{0.201}$Cr$_{0.193}$Fe$_{0.400}$Ni$_{0.205}$",
     "comp": {"Co":0.201,"Cr":0.193,"Fe":0.400,"Ni":0.205}, "struct":"FCC",
     "a_exp":3.5803, "a_vegard":3.5847, "a_eq10":3.5864, "ref":"[66,67]"},
    {"name": "Co$_{0.248}$Cr$_{0.244}$Fe$_{0.251}$Ni$_{0.257}$",
     "comp": {"Co":0.248,"Cr":0.244,"Fe":0.251,"Ni":0.257}, "struct":"FCC",
     "a_exp":3.5767, "a_vegard":3.5783, "a_eq10":3.5793, "ref":"[66,67]"},
    {"name": "Co$_{0.386}$Cr$_{0.196}$Fe$_{0.211}$Ni$_{0.208}$",
     "comp": {"Co":0.386,"Cr":0.196,"Fe":0.211,"Ni":0.208}, "struct":"FCC",
     "a_exp":3.568, "a_vegard":3.5723, "a_eq10":3.5744, "ref":"[66,67]"},
]

# =====================================================================
# 2. King1966 — Pure element atomic volumes (Å³)
#    V_atom = V_cell / N_atoms at room temperature
#    Sources: King1966 Table II (implicit), CRC Handbook, crystallographic data
# =====================================================================
KING_ATOMIC_VOLUMES = {
    # Element: V_atom (Å³) — room temperature crystallographic values
    # FCC metals: V = a³/4
    "Al": 16.602,  # a=4.0495, V=66.409/4
    "Cu": 11.810,  # a=3.6149, V=47.240/4
    "Ni": 10.941,  # a=3.5240, V=43.763/4
    "Pd": 14.716,  # a=3.8907, V=58.863/4
    "Pt": 15.095,  # a=3.9242, V=60.378/4
    "Au": 16.966,  # a=4.0782, V=67.863/4
    "Ag": 17.061,  # a=4.0862, V=68.245/4
    "Ir": 14.155,  # a=3.8394, V=56.621/4
    "Rh": 13.754,  # a=3.8034, V=55.016/4
    # HCP metals: V = a²c√3/2 / 2
    "Co": 11.073,  # a=2.5071, c=4.0695
    "Ti": 17.649,  # a=2.9506, c=4.6855
    "Zr": 23.279,  # a=3.2312, c=5.1477
    "Hf": 22.312,  # a=3.1946, c=5.0511
    "Ru": 13.571,  # a=2.7059, c=4.2815
    "Os": 13.977,  # a=2.7341, c=4.3197
    "Re": 14.712,  # a=2.7608, c=4.4580
    "Mn": 12.210,  # alpha-Mn, a=8.9139, 58 atoms, V/58
    "Zn": 15.207,  # a=2.6649, c=4.9468
    # BCC metals: V = a³/2
    "Fe": 11.776,  # a=2.8665, V=23.553/2
    "Cr": 12.008,  # a=2.8839, V=24.016/2
    "V":  13.824,  # a=3.0240, V=27.647/2
    "Nb": 17.978,  # a=3.3004, V=35.955/2
    "Mo": 15.583,  # a=3.1472, V=31.165/2
    "Ta": 18.014,  # a=3.3026, V=36.027/2
    "W":  15.850,  # a=3.1652, V=31.700/2
    # Others
    "Si": 20.024,  # diamond cubic, a=5.4309, V/8
    "Ge": 22.634,  # diamond cubic, a=5.6575, V/8
    "Be": 8.111,   # HCP a=2.2856, c=3.5832
    "Mg": 23.240,  # HCP a=3.2094, c=5.2105
    "Y":  33.018,  # HCP a=3.6482, c=5.7318
    "La": 37.168,  # DHCP a=3.7740, c=12.159, V/4
    "Ce": 34.367,  # FCC a=5.1612
    "Sc": 24.987,  # HCP a=3.3090, c=5.2733
    "B":  7.241,   # rhombohedral, ~12 atoms
    "P":  23.000,  # estimated (white phosphorus)
    "Sn": 27.053,  # beta-Sn, a=5.8318, c=3.1819
    "Pb": 30.321,  # FCC a=4.9502
}

# =====================================================================
# 3. Load our effective radii from the optimization output
# =====================================================================
def load_our_radii():
    """Load effective radii from four_case_comparison_study output."""
    radii = {}
    # Try multiple possible locations
    for csv_dir in [Path("four_case_output/figures"), Path("four_case_output"), Path("data")]:
        for src in ["MP", "OQMD"]:
            for struct in ["B2", "L12"]:
                fname = csv_dir / f"radii_{src}_{struct}.csv"
                if fname.exists():
                    df = pd.read_csv(fname)
                    key = f"{src}_{struct}"
                    if key not in radii:  # don't overwrite
                        radii[key] = dict(zip(df["element"], df["radius"]))
    return radii

# =====================================================================
# 4. Prediction functions
# =====================================================================
def predict_vegard_king(comp, struct, king_vols=KING_ATOMIC_VOLUMES):
    """Predict lattice parameter using King atomic volumes + Vegard's law.
    V_avg = sum(c_i * V_i), then a = (n_auc * V_avg)^(1/3)
    n_auc = 4 for FCC, 2 for BCC.
    """
    # Normalize compositions to sum to 1.0
    total = sum(comp.values())
    v_avg = 0.0
    for el, c in comp.items():
        if el not in king_vols:
            return None
        v_avg += (c / total) * king_vols[el]
    n_auc = 4 if struct == "FCC" else 2
    return (n_auc * v_avg) ** (1/3)

def predict_pairwise_max(comp, struct, radii_dict):
    """Predict lattice parameter using pairwise max-contact model.
    FCC: a = 2*sqrt(2) * sum_ij c_i*c_j*max(r_i, r_j)
    BCC: a = (4/sqrt(3)) * sum_ij c_i*c_j*max(r_i, r_j)
    """
    elements = list(comp.keys())
    for el in elements:
        if el not in radii_dict:
            return None
    # Normalize compositions to sum to 1.0 to avoid quadratic amplification
    total = sum(comp.values())
    r_eff = 0.0
    for i, el_i in enumerate(elements):
        for j, el_j in enumerate(elements):
            r_eff += (comp[el_i]/total) * (comp[el_j]/total) * max(radii_dict[el_i], radii_dict[el_j])
    if struct == "FCC":
        return 2 * np.sqrt(2) * r_eff
    else:  # BCC
        return 4 / np.sqrt(3) * r_eff

def predict_simple_vegard(comp, struct, radii_dict):
    """Predict lattice parameter using simple Vegard (weighted average radius).
    FCC: a = 2*sqrt(2) * r_avg
    BCC: a = (4/sqrt(3)) * r_avg
    """
    # Normalize compositions to sum to 1.0
    total = sum(comp.values())
    r_avg = 0.0
    for el, c in comp.items():
        if el not in radii_dict:
            return None
        r_avg += (c / total) * radii_dict[el]
    if struct == "FCC":
        return 2 * np.sqrt(2) * r_avg
    else:
        return 4 / np.sqrt(3) * r_avg

# =====================================================================
# 5. Main analysis
# =====================================================================
def main():
    print("=" * 70)
    print("Alonso2021 / King1966 / This Work — Comprehensive HEA Comparison")
    print("=" * 70)

    # Load our radii
    our_radii = load_our_radii()
    if not our_radii:
        print("WARNING: Could not load effective radii from four_case_output/")
        print("Attempting to load from CSV files directly...")

    print(f"\nLoaded radius sets: {list(our_radii.keys())}")
    for key, rdict in our_radii.items():
        print(f"  {key}: {len(rdict)} elements, range {min(rdict.values()):.3f}–{max(rdict.values()):.3f} Å")

    # ---------------------------------------------------------------
    # Build comparison dataframe
    # ---------------------------------------------------------------
    rows = []
    for entry in ALONSO_TABLE2:
        name = entry["name"]
        comp = entry["comp"]
        struct = entry["struct"]
        a_exp = entry["a_exp"]

        row = {
            "name": name,
            "struct": struct,
            "a_exp": a_exp,
            "a_alonso_vegard": entry["a_vegard"],
            "a_alonso_eq10": entry["a_eq10"],
            "err_alonso_vegard": abs(entry["a_vegard"] - a_exp) / a_exp * 100,
            "err_alonso_eq10": abs(entry["a_eq10"] - a_exp) / a_exp * 100,
        }

        # King atomic volume prediction
        a_king = predict_vegard_king(comp, struct)
        if a_king is not None:
            row["a_king_vegard"] = a_king
            row["err_king_vegard"] = abs(a_king - a_exp) / a_exp * 100
        else:
            missing = [el for el in comp if el not in KING_ATOMIC_VOLUMES]
            row["a_king_vegard"] = np.nan
            row["err_king_vegard"] = np.nan

        # Our predictions (multiple radius sets)
        for rset_name, rset in our_radii.items():
            a_pw = predict_pairwise_max(comp, struct, rset)
            a_sv = predict_simple_vegard(comp, struct, rset)
            if a_pw is not None:
                row[f"a_pw_{rset_name}"] = a_pw
                row[f"err_pw_{rset_name}"] = abs(a_pw - a_exp) / a_exp * 100
            else:
                row[f"a_pw_{rset_name}"] = np.nan
                row[f"err_pw_{rset_name}"] = np.nan
            if a_sv is not None:
                row[f"a_sv_{rset_name}"] = a_sv
                row[f"err_sv_{rset_name}"] = abs(a_sv - a_exp) / a_exp * 100
            else:
                row[f"a_sv_{rset_name}"] = np.nan
                row[f"err_sv_{rset_name}"] = np.nan

        rows.append(row)

    df = pd.DataFrame(rows)

    # Save full data as CSV (Appendix material)
    df.to_csv(OUTDIR / "alonso_king_comparison_full.csv", index=False)
    print(f"\nFull comparison table saved: {OUTDIR / 'alonso_king_comparison_full.csv'}")
    print(f"Total HEAs: {len(df)}  (BCC: {(df['struct']=='BCC').sum()}, FCC: {(df['struct']=='FCC').sum()})")

    # ---------------------------------------------------------------
    # Statistics summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RMSE and MAE Summary")
    print("=" * 70)

    methods = [
        ("Alonso Vegard", "a_alonso_vegard"),
        ("Alonso Eq.10", "a_alonso_eq10"),
        ("King Vegard", "a_king_vegard"),
    ]
    for rset_name in our_radii:
        methods.append((f"This work PW-max ({rset_name})", f"a_pw_{rset_name}"))
        methods.append((f"This work Vegard ({rset_name})", f"a_sv_{rset_name}"))

    stats_rows = []
    for label, col in methods:
        for struct in ["All", "BCC", "FCC"]:
            sub = df if struct == "All" else df[df["struct"] == struct]
            valid = sub.dropna(subset=[col])
            if len(valid) == 0:
                continue
            errs = valid[col] - valid["a_exp"]
            rmse = np.sqrt((errs ** 2).mean())
            mae = errs.abs().mean()
            rel_errs = (errs.abs() / valid["a_exp"] * 100)
            mean_rel = rel_errs.mean()
            max_rel = rel_errs.max()
            stats_rows.append({
                "Method": label, "Structure": struct,
                "N": len(valid), "RMSE (Å)": rmse, "MAE (Å)": mae,
                "Mean |err| (%)": mean_rel, "Max |err| (%)": max_rel
            })

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(OUTDIR / "comparison_statistics.csv", index=False)
    print(stats_df.to_string(index=False))

    # ---------------------------------------------------------------
    # Figure 1: Parity plot — all methods
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    fig.suptitle("HEA格子定数予測の比較（Alonso2021 Table 2データ, $N={}$）".format(len(df)),
                 fontsize=22, fontweight="bold")

    plot_methods = [
        ("Alonso Vegard則", "a_alonso_vegard"),
        ("Alonso Eq.10\n(体積サイズ因子)", "a_alonso_eq10"),
        ("King原子体積\nVegard則", "a_king_vegard"),
    ]
    # Add best "this work" methods
    best_pw = None
    best_pw_rmse = 999
    for rset_name in our_radii:
        col = f"a_pw_{rset_name}"
        valid = df.dropna(subset=[col])
        if len(valid) > 0:
            rmse = np.sqrt(((valid[col] - valid["a_exp"]) ** 2).mean())
            if rmse < best_pw_rmse:
                best_pw_rmse = rmse
                best_pw = rset_name
    if best_pw:
        plot_methods.append((f"本研究 PW-max\n({best_pw})", f"a_pw_{best_pw}"))
    # Also add OQMD_B2 and MP_B2 if different
    for rn in ["OQMD_B2", "MP_B2"]:
        if rn in our_radii and rn != best_pw:
            plot_methods.append((f"本研究 PW-max\n({rn})", f"a_pw_{rn}"))
            break
    # Also add simple Vegard with best set for comparison
    if best_pw:
        plot_methods.append((f"本研究 単純Vegard\n({best_pw})", f"a_sv_{best_pw}"))

    for idx, (label, col) in enumerate(plot_methods[:6]):
        ax = axes.flat[idx]
        valid = df.dropna(subset=[col])
        bcc = valid[valid["struct"] == "BCC"]
        fcc = valid[valid["struct"] == "FCC"]

        if len(bcc) > 0:
            ax.scatter(bcc["a_exp"], bcc[col], c="steelblue", marker="s", s=60,
                      edgecolors="navy", linewidth=0.5, alpha=0.8, label=f"BCC ($N$={len(bcc)})")
        if len(fcc) > 0:
            ax.scatter(fcc["a_exp"], fcc[col], c="coral", marker="o", s=60,
                      edgecolors="darkred", linewidth=0.5, alpha=0.8, label=f"FCC ($N$={len(fcc)})")

        lims = [valid["a_exp"].min() - 0.1, valid["a_exp"].max() + 0.1]
        ax.plot(lims, lims, "k--", linewidth=1.5)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("$a_\\mathrm{exp}$ (Å)")
        ax.set_ylabel("$a_\\mathrm{pred}$ (Å)")
        ax.set_title(label, fontsize=16)
        ax.legend(fontsize=11)

        # Add RMSE annotation
        errs = valid[col] - valid["a_exp"]
        rmse = np.sqrt((errs ** 2).mean())
        mae = errs.abs().mean()
        ax.text(0.05, 0.92, f"RMSE = {rmse:.4f} Å\nMAE = {mae:.4f} Å",
                transform=ax.transAxes, fontsize=12, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

        ax.set_aspect("equal")

    plt.tight_layout()
    fig.savefig(OUTDIR / "parity_all_methods.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nParity plot saved: {OUTDIR / 'parity_all_methods.png'}")

    # ---------------------------------------------------------------
    # Figure 2: RMSE bar chart comparison
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    for sidx, (stitle, sfilter) in enumerate([("全体", "All"), ("BCC", "BCC"), ("FCC", "FCC")]):
        ax = axes[sidx]
        sub_stats = stats_df[(stats_df["Structure"] == sfilter) & (stats_df["N"] > 0)].copy()
        sub_stats = sub_stats.sort_values("RMSE (Å)")
        colors = []
        for m in sub_stats["Method"]:
            if "Alonso" in m:
                colors.append("steelblue")
            elif "King" in m:
                colors.append("forestgreen")
            elif "PW-max" in m:
                colors.append("coral")
            else:
                colors.append("gray")
        bars = ax.barh(range(len(sub_stats)), sub_stats["RMSE (Å)"], color=colors,
                      edgecolor="black", linewidth=0.5)
        ax.set_yticks(range(len(sub_stats)))
        ax.set_yticklabels([m.replace("\n", " ") for m in sub_stats["Method"]], fontsize=11)
        ax.set_xlabel("RMSE (Å)", fontsize=14)
        ax.set_title(f"{stitle} (N={sub_stats['N'].max()})", fontsize=16)
        # Add value labels
        for bar, val in zip(bars, sub_stats["RMSE (Å)"]):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                   f"{val:.4f}", va="center", fontsize=10)
        ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(OUTDIR / "rmse_bar_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"RMSE bar chart saved: {OUTDIR / 'rmse_bar_comparison.png'}")

    # ---------------------------------------------------------------
    # Figure 3: Error distribution histogram
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    hist_methods = [
        ("Alonso Eq.10", "a_alonso_eq10", "steelblue"),
        ("King Vegard", "a_king_vegard", "forestgreen"),
    ]
    if best_pw:
        hist_methods.append((f"本研究 PW-max ({best_pw})", f"a_pw_{best_pw}", "coral"))

    for sidx, struct in enumerate(["All", "BCC", "FCC"]):
        ax = axes[sidx]
        sub = df if struct == "All" else df[df["struct"] == struct]
        for label, col, color in hist_methods:
            valid = sub.dropna(subset=[col])
            if len(valid) == 0:
                continue
            rel_errs = (valid[col] - valid["a_exp"]) / valid["a_exp"] * 100
            ax.hist(rel_errs, bins=20, alpha=0.5, color=color, label=label, edgecolor="black")
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("相対誤差 (%)")
        ax.set_ylabel("頻度")
        ax.set_title(f"{struct} (N={len(sub)})" if struct != "All" else f"全体 (N={len(sub)})")
        ax.legend(fontsize=10)

    plt.tight_layout()
    fig.savefig(OUTDIR / "error_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Error distribution saved: {OUTDIR / 'error_distribution.png'}")

    # ---------------------------------------------------------------
    # Figure 4: King atomic volume vs our r_WS comparison (periodic table style)
    # ---------------------------------------------------------------
    # Convert King V_atom to r_WS = (3V/4pi)^(1/3) for direct comparison
    king_rws = {}
    for el, v in KING_ATOMIC_VOLUMES.items():
        king_rws[el] = (3 * v / (4 * np.pi)) ** (1/3)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    for sidx, (rset_name, ax_title) in enumerate([
        ("OQMD_B2", "OQMD-B2 vs King $r_\\mathrm{WS}$"),
        ("OQMD_L12", "OQMD-L1$_2$ vs King $r_\\mathrm{WS}$")
    ]):
        ax = axes[sidx]
        if rset_name not in our_radii:
            ax.text(0.5, 0.5, "Data not available", transform=ax.transAxes, ha="center")
            continue
        rset = our_radii[rset_name]
        common = set(rset.keys()) & set(king_rws.keys())
        x = [king_rws[el] for el in common]
        y = [rset[el] for el in common]
        labels = list(common)

        ax.scatter(x, y, c="steelblue", s=80, edgecolors="navy", linewidth=0.5, alpha=0.8)
        # Label outliers (> 2 std from y=x)
        diffs = [yi - xi for xi, yi in zip(x, y)]
        mean_d, std_d = np.mean(diffs), np.std(diffs)
        for xi, yi, el in zip(x, y, labels):
            if abs(yi - xi - mean_d) > 2 * std_d:
                ax.annotate(el, (xi, yi), fontsize=10, fontweight="bold", color="red",
                           xytext=(5, 5), textcoords="offset points")

        lims = [min(min(x), min(y)) - 0.05, max(max(x), max(y)) + 0.05]
        ax.plot(lims, lims, "k--", linewidth=1.5)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("King $r_\\mathrm{WS}$ (Å)", fontsize=16)
        ax.set_ylabel(f"本研究 $r_\\mathrm{{eff}}$ (Å) [{rset_name}]", fontsize=16)
        ax.set_title(ax_title, fontsize=18)
        ax.set_aspect("equal")

        # R² and RMSE
        from scipy import stats as sp_stats
        slope, intercept, r_value, p_value, std_err = sp_stats.linregress(x, y)
        rmse = np.sqrt(np.mean([(yi - xi) ** 2 for xi, yi in zip(x, y)]))
        ax.text(0.05, 0.92, f"$R^2$ = {r_value**2:.4f}\nRMSE = {rmse:.4f} Å\n$N$ = {len(common)}",
                transform=ax.transAxes, fontsize=13, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    plt.tight_layout()
    fig.savefig(OUTDIR / "king_vs_our_rws.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"King vs our r_WS saved: {OUTDIR / 'king_vs_our_rws.png'}")

    # ---------------------------------------------------------------
    # Save Appendix data: Alonso Table 2 + King atomic volumes
    # ---------------------------------------------------------------
    # Alonso Table 2 reformatted for Appendix
    appendix_alonso = []
    for entry in ALONSO_TABLE2:
        comp_str = " ".join(f"{el}:{c:.3f}" for el, c in entry["comp"].items())
        appendix_alonso.append({
            "HEA": entry["name"].replace("$","").replace("{","").replace("}","").replace("_","").replace("\\",""),
            "Structure": entry["struct"],
            "Composition": comp_str,
            "a_exp (Å)": entry["a_exp"],
            "a_Vegard (Å)": entry["a_vegard"],
            "a_Eq10 (Å)": entry["a_eq10"],
            "Ref": entry["ref"],
        })
    pd.DataFrame(appendix_alonso).to_csv(OUTDIR / "appendix_alonso_table2.csv", index=False)

    # King atomic volumes + derived r_WS
    king_appendix = []
    for el, v in sorted(KING_ATOMIC_VOLUMES.items()):
        rws = (3 * v / (4 * np.pi)) ** (1/3)
        king_appendix.append({
            "Element": el,
            "V_atom (Å³)": f"{v:.3f}",
            "r_WS (Å)": f"{rws:.4f}",
        })
    pd.DataFrame(king_appendix).to_csv(OUTDIR / "appendix_king_atomic_volumes.csv", index=False)

    print(f"\nAppendix CSVs saved in {OUTDIR}/")
    print(f"  appendix_alonso_table2.csv ({len(appendix_alonso)} entries)")
    print(f"  appendix_king_atomic_volumes.csv ({len(king_appendix)} elements)")

    # ---------------------------------------------------------------
    # Save King volume size factors (Ωsf) from Alonso Table 3
    # ---------------------------------------------------------------
    alonso_omega_sf = {
        ("Al","Fe"): -47.65, ("Al","Mo"): -34.05, ("Al","Nb"): 4.48,
        ("Al","Ni"): -22.60, ("Co","V"): 4.54, ("Co","Zn"): 43.57,
        ("Cr","Cu"): 3.33, ("Cr","Ti"): 46.17, ("Cr","Zn"): 26.50,
        ("Fe","Ta"): 38.06, ("Hf","Nb"): -19.95, ("Hf","Ta"): -19.63,
        ("Hf","V"): -40.81, ("Hf","Zr"): 4.31,
        ("Ir","Re"): 0.19, ("Mo","Al"): -9.16, ("Mo","Co"): -30.11,
        ("Mo","Fe"): -26.17, ("Nb","Al"): -9.42, ("Nb","Cr"): -28.97,
        ("Os","Ir"): 1.74, ("Os","Pd"): 4.77, ("Os","Pt"): 7.80,
        ("Os","Rh"): -1.63, ("Pt","Os"): -7.12,
        ("Re","Co"): -26.46, ("Re","Fe"): -23.94, ("Re","Ir"): -5.61,
        ("Re","Rh"): -6.70,
        ("Rh","Os"): 1.58, ("Rh","Re"): 4.17, ("Rh","Ru"): 0.23,
        ("Rh","Ti"): -21.24,
        ("Ru","Co"): -18.28, ("Ru","Fe"): -11.24,
        ("Ta","Al"): -8.73, ("Ta","Fe"): -38.9, ("Ta","Hf"): 21.93,
        ("Ti","Nb"): 1.49, ("Ti","W"): -17.11,
        ("V","Hf"): 64.64, ("V","Ni"): -27.83, ("V","Ti"): 26.22,
        ("Zr","Ta"): -25.02,
    }
    omega_rows = [{"Solvent": s, "Solute": sl, "Omega_sf (%)": v}
                  for (s, sl), v in sorted(alonso_omega_sf.items())]
    pd.DataFrame(omega_rows).to_csv(OUTDIR / "appendix_volume_size_factors.csv", index=False)
    print(f"  appendix_volume_size_factors.csv ({len(omega_rows)} pairs)")

    print("\n" + "=" * 70)
    print("Analysis complete.")
    print("=" * 70)

if __name__ == "__main__":
    main()
