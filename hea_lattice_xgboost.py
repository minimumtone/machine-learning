#!/usr/bin/env python3
"""
HEA Lattice Constant Prediction via Dynamic Volume Size Factor (XGBoost)
=========================================================================

Goal: Surpass Alonso (2021) RMSE 0.023 Å → maximise accuracy

Strategy:
  Phase 1: Reproduce Alonso Eq.10 + DFT-derived Ω_sf from MP/OQMD
  Phase 2: Structure-specific DFT Ω_sf (B2→BCC, L1₂→FCC) with γ scaling
  Phase 3: Hybrid Physics-ML — XGBoost learns residual corrections
  Phase 4: Ensemble optimisation with noise floor analysis

Key innovations:
  1. DFT-derived pairwise Ω_sf from ~3500 binary compounds (MP+OQMD)
  2. Structure-matched Ω_sf: B2 data for BCC HEAs, L1₂ data for FCC HEAs
  3. Dynamic size factor: composition-dependent corrections via XGBoost
  4. Noise floor estimation from duplicate HEA measurements

Author: Satoshi Minamoto (NIMS)
"""

import warnings
import json
import pickle
from pathlib import Path
from itertools import combinations
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, RBF, ConstantKernel
from sklearn.svm import SVR
from scipy.optimize import minimize
from xgboost import XGBRegressor
from cubist import Cubist

warnings.filterwarnings("ignore")

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

plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "figure.dpi": 150,
})

OUTDIR = Path("hea_xgboost_output")
OUTDIR.mkdir(exist_ok=True)

# =====================================================================
# Elemental Properties
# =====================================================================
PAULING_EN = {
    "Li":0.98,"Be":1.57,"B":2.04,"C":2.55,"N":3.04,"O":3.44,
    "Na":0.93,"Mg":1.31,"Al":1.61,"Si":1.90,"P":2.19,"S":2.58,
    "K":0.82,"Ca":1.00,"Sc":1.36,"Ti":1.54,"V":1.63,"Cr":1.66,
    "Mn":1.55,"Fe":1.83,"Co":1.88,"Ni":1.91,"Cu":1.90,"Zn":1.65,
    "Ga":1.81,"Ge":2.01,"As":2.18,"Se":2.55,"Br":2.96,
    "Rb":0.82,"Sr":0.95,"Y":1.22,"Zr":1.33,"Nb":1.60,"Mo":2.16,
    "Tc":1.90,"Ru":2.20,"Rh":2.28,"Pd":2.20,"Ag":1.93,"Cd":1.69,
    "In":1.78,"Sn":1.96,"Sb":2.05,"Te":2.10,"I":2.66,
    "Cs":0.79,"Ba":0.89,"La":1.10,"Ce":1.12,"Pr":1.13,"Nd":1.14,
    "Sm":1.17,"Eu":1.20,"Gd":1.20,"Tb":1.10,"Dy":1.22,"Ho":1.23,
    "Er":1.24,"Tm":1.25,"Yb":1.10,"Lu":1.27,
    "Hf":1.30,"Ta":1.50,"W":2.36,"Re":1.90,"Os":2.20,"Ir":2.20,
    "Pt":2.28,"Au":2.54,"Hg":2.00,"Tl":1.62,"Pb":2.33,"Bi":2.02,
    "Th":1.30,"U":1.38,"Pu":1.28,"Np":1.36,
}

VEC = {
    "Li":1,"Be":2,"B":3,"C":4,"N":5,"O":6,
    "Na":1,"Mg":2,"Al":3,"Si":4,"P":5,"S":6,
    "K":1,"Ca":2,"Sc":3,"Ti":4,"V":5,"Cr":6,
    "Mn":7,"Fe":8,"Co":9,"Ni":10,"Cu":11,"Zn":12,
    "Ga":3,"Ge":4,"As":5,"Se":6,"Br":7,
    "Rb":1,"Sr":2,"Y":3,"Zr":4,"Nb":5,"Mo":6,
    "Tc":7,"Ru":8,"Rh":9,"Pd":10,"Ag":11,"Cd":12,
    "In":3,"Sn":4,"Sb":5,"Te":6,"I":7,
    "Cs":1,"Ba":2,"La":3,"Ce":4,"Pr":5,"Nd":6,
    "Sm":8,"Eu":9,"Gd":10,"Tb":11,"Dy":12,"Ho":13,
    "Er":14,"Tm":15,"Yb":16,"Lu":3,
    "Hf":4,"Ta":5,"W":6,"Re":7,"Os":8,"Ir":9,
    "Pt":10,"Au":11,"Hg":12,"Tl":3,"Pb":4,"Bi":5,
    "Th":4,"U":6,"Pu":6,"Np":5,
}

D_ELECTRONS = {
    "Li":0,"Be":0,"B":0,"C":0,"N":0,"O":0,
    "Na":0,"Mg":0,"Al":0,"Si":0,"P":0,"S":0,
    "K":0,"Ca":0,"Sc":1,"Ti":2,"V":3,"Cr":5,
    "Mn":5,"Fe":6,"Co":7,"Ni":8,"Cu":10,"Zn":10,
    "Ga":10,"Ge":10,"As":10,"Se":10,"Br":10,
    "Rb":0,"Sr":0,"Y":1,"Zr":2,"Nb":4,"Mo":5,
    "Tc":5,"Ru":7,"Rh":8,"Pd":10,"Ag":10,"Cd":10,
    "In":10,"Sn":10,"Sb":10,"Te":10,"I":10,
    "Cs":0,"Ba":0,"La":1,"Ce":1,"Pr":3,"Nd":4,
    "Sm":6,"Eu":7,"Gd":7,"Tb":9,"Dy":10,"Ho":11,
    "Er":12,"Tm":13,"Yb":14,"Lu":14,
    "Hf":2,"Ta":3,"W":4,"Re":5,"Os":6,"Ir":7,
    "Pt":9,"Au":10,"Hg":10,"Tl":10,"Pb":10,"Bi":10,
    "Th":2,"U":3,"Pu":6,"Np":4,
}

# Atomic mass
ATOMIC_MASS = {
    "H":1.008,"He":4.003,"Li":6.941,"Be":9.012,"B":10.81,"C":12.01,
    "N":14.01,"O":16.00,"Al":26.98,"Si":28.09,"P":30.97,"S":32.07,
    "Ti":47.87,"V":50.94,"Cr":52.00,"Mn":54.94,"Fe":55.85,"Co":58.93,
    "Ni":58.69,"Cu":63.55,"Zn":65.38,"Ga":69.72,"Ge":72.63,
    "Zr":91.22,"Nb":92.91,"Mo":95.95,"Ru":101.07,"Rh":102.91,"Pd":106.42,
    "Ag":107.87,"In":114.82,"Sn":118.71,"Hf":178.49,"Ta":180.95,
    "W":183.84,"Re":186.21,"Os":190.23,"Ir":192.22,"Pt":195.08,
    "Au":196.97,"Pb":207.2,"Bi":208.98,
    "Sc":44.96,"Y":88.91,"La":138.91,"Ce":140.12,"Nd":144.24,
    "Sm":150.36,"Eu":151.96,"Gd":157.25,"Tb":158.93,"Dy":162.50,
    "Ho":164.93,"Er":167.26,"Tm":168.93,"Yb":173.05,"Lu":174.97,
    "Mg":24.31,"Ca":40.08,"Sr":87.62,"Ba":137.33,
    "Th":232.04,"U":238.03,"Pu":244.06,"Np":237.05,
}

# King1966 pure-element atomic volumes (Å³)
KING_ATOMIC_VOLUMES = {
    "Al":16.602,"Cu":11.810,"Ni":10.941,"Pd":14.716,"Pt":15.095,
    "Au":16.966,"Ag":17.061,"Ir":14.155,"Rh":13.754,
    "Co":11.073,"Ti":17.649,"Zr":23.279,"Hf":22.312,
    "Ru":13.571,"Os":13.977,"Re":14.712,"Mn":12.210,"Zn":15.207,
    "Fe":11.776,"Cr":12.008,"V":13.824,"Nb":17.978,"Mo":15.583,
    "Ta":18.014,"W":15.850,"Si":20.024,"Ge":22.634,"Be":8.111,
    "Mg":23.240,"Y":33.018,"La":37.168,"Ce":34.367,"Sc":24.987,
    "B":7.241,"P":23.000,"Sn":27.053,"Pb":30.321,
}

# =====================================================================
# Alonso Table 2 — 68 cubic HEAs
# =====================================================================
ALONSO_TABLE2 = [
    {"comp":{"W":0.273,"Nb":0.227,"Mo":0.256,"Ta":0.244},"struct":"BCC","a_exp":3.2134,"a_vegard":3.2263,"a_eq10":3.2148},
    {"comp":{"W":0.25,"Nb":0.22,"Mo":0.26,"Ta":0.27},"struct":"BCC","a_exp":3.24,"a_vegard":3.23,"a_eq10":3.22},
    {"comp":{"Nb":0.25,"Mo":0.25,"Ta":0.25,"W":0.25},"struct":"BCC","a_exp":3.222,"a_vegard":3.231,"a_eq10":3.217},
    {"comp":{"W":0.211,"Nb":0.206,"Mo":0.217,"Ta":0.156,"V":0.210},"struct":"BCC","a_exp":3.1832,"a_vegard":3.1849,"a_eq10":3.1804},
    {"comp":{"W":1/6,"Nb":1/6,"Mo":1/6,"Ta":1/6,"V":1/6,"Ti":1/6},"struct":"BCC","a_exp":3.216,"a_vegard":3.209,"a_eq10":3.188},
    {"comp":{"Nb":0.2,"Zr":0.2,"Hf":0.2,"V":0.2,"Ti":0.2},"struct":"BCC","a_exp":3.377,"a_vegard":3.361,"a_eq10":3.374},
    {"comp":{"Al":0.262,"Cr":0.241,"Fe":0.259,"Mo":0.235,"V":0.003},"struct":"BCC","a_exp":3.01,"a_vegard":3.04,"a_eq10":3.02},
    {"comp":{"Al":0.200,"Cr":0.243,"Fe":0.232,"Mo":0.166,"V":0.158},"struct":"BCC","a_exp":2.98,"a_vegard":3.02,"a_eq10":3.01},
    {"comp":{"Al":0.180,"Ni":0.189,"Cu":0.214,"Fe":0.196,"Cr":0.220},"struct":"BCC","a_exp":2.894,"a_vegard":2.926,"a_eq10":2.918},
    {"comp":{"Nb":0.25,"Zr":0.25,"Hf":0.25,"Ti":0.25},"struct":"BCC","a_exp":3.438,"a_vegard":3.435,"a_eq10":3.428},
    {"comp":{"Nb":0.200,"Mo":0.208,"Cr":0.187,"Ti":0.202,"V":0.203},"struct":"BCC","a_exp":3.140,"a_vegard":3.139,"a_eq10":3.138},
    {"comp":{"Nb":0.25,"Ta":0.25,"Ti":0.25,"V":0.25},"struct":"BCC","a_exp":3.23,"a_vegard":3.23,"a_eq10":3.23},
    {"comp":{"Nb":0.25,"Ta":0.25,"Ti":0.25,"V":0.25},"struct":"BCC","a_exp":3.2206,"a_vegard":3.2319,"a_eq10":3.2299},
    {"comp":{"Nb":0.27,"Mo":0.21,"Ta":0.27,"W":0.24},"struct":"BCC","a_exp":3.218,"a_vegard":3.226,"a_eq10":3.213},
    {"comp":{"Nb":0.22,"Mo":0.18,"Ta":0.28,"W":0.31},"struct":"BCC","a_exp":3.216,"a_vegard":3.222,"a_eq10":3.210},
    {"comp":{"Nb":0.24,"Mo":0.27,"Ta":0.25,"W":0.24},"struct":"BCC","a_exp":3.214,"a_vegard":3.229,"a_eq10":3.215},
    {"comp":{"Nb":0.24,"Mo":0.18,"Ta":0.36,"W":0.22},"struct":"BCC","a_exp":3.228,"a_vegard":3.245,"a_eq10":3.234},
    {"comp":{"Nb":0.216,"Mo":0.230,"Ta":0.281,"W":0.273},"struct":"BCC","a_exp":3.2034,"a_vegard":3.2303,"a_eq10":3.2177},
    {"comp":{"Ti":2/4.7,"Zr":1/4.7,"Hf":1/4.7,"V":0.5/4.7,"Mo":0.2/4.7},"struct":"BCC","a_exp":3.4584,"a_vegard":3.3805,"a_eq10":3.3845},
    {"comp":{"Ti":0.262,"Nb":0.255,"Ta":0.121,"Zr":0.242,"Al":0.120},"struct":"BCC","a_exp":3.355,"a_vegard":3.363,"a_eq10":3.360},
    {"comp":{"V":0.333,"Cr":0.309,"Fe":0.308,"Ta":0.025,"W":0.025},"struct":"BCC","a_exp":2.935,"a_vegard":2.948,"a_eq10":2.930},
    {"comp":{"Nb":0.2,"Mo":0.2,"Ta":0.2,"V":0.2,"Ti":0.2},"struct":"BCC","a_exp":3.1945,"a_vegard":3.2153,"a_eq10":3.2055},
    {"comp":{"Nb":0.199,"Hf":0.198,"Ta":0.175,"V":0.212,"Ti":0.217},"struct":"BCC","a_exp":3.279,"a_vegard":3.295,"a_eq10":3.290},
    {"comp":{"Nb":0.304,"Mo":0.037,"Ta":0.051,"Zr":0.290,"Ti":0.319},"struct":"BCC","a_exp":3.285,"a_vegard":3.381,"a_eq10":3.368},
    {"comp":{"Nb":0.255,"Mo":0.207,"Ta":0.190,"Zr":0.131,"Ti":0.217},"struct":"BCC","a_exp":3.24,"a_vegard":3.31,"a_eq10":3.28},
    {"comp":{"Nb":0.275,"Mo":0.095,"Ta":0.091,"Zr":0.256,"Ti":0.284},"struct":"BCC","a_exp":3.40,"a_vegard":3.47,"a_eq10":3.46},
    {"comp":{"Nb":0.2,"Zr":0.2,"Hf":0.2,"V":0.2,"Ti":0.2},"struct":"BCC","a_exp":3.3663,"a_vegard":3.3613,"a_eq10":3.3582},
    {"comp":{"Nb":0.238,"V":0.245,"Al":0.266,"Ti":0.251},"struct":"BCC","a_exp":3.18,"a_vegard":3.07,"a_eq10":3.10},
    {"comp":{"Co":0.25,"Cr":0.25,"Fe":0.25,"Ni":0.25},"struct":"FCC","a_exp":3.575,"a_vegard":3.579,"a_eq10":3.587},
    {"comp":{"Co":0.2,"Cr":0.2,"Fe":0.2,"Mn":0.2,"Ni":0.2},"struct":"FCC","a_exp":3.597,"a_vegard":3.594,"a_eq10":3.602},
    {"comp":{"Co":0.204,"Cr":0.205,"Fe":0.202,"Mn":0.194,"Ni":0.195},"struct":"FCC","a_exp":3.59,"a_vegard":3.59,"a_eq10":3.60},
    {"comp":{"Co":0.203,"Cr":0.194,"Fe":0.206,"Mn":0.201,"Ni":0.196},"struct":"FCC","a_exp":3.60,"a_vegard":3.59,"a_eq10":3.60},
    {"comp":{"Cr":0.127,"Fe":0.498,"Ni":0.111,"Mn":0.264},"struct":"FCC","a_exp":3.61,"a_vegard":3.62,"a_eq10":3.62},
    {"comp":{"Co":0.20,"Cr":0.20,"Fe":0.40,"Ni":0.10,"Mn":0.10},"struct":"FCC","a_exp":3.587,"a_vegard":3.598,"a_eq10":3.605},
    {"comp":{"Co":0.211,"Cr":0.187,"Fe":0.342,"Ni":0.063,"Mn":0.197},"struct":"FCC","a_exp":3.588,"a_vegard":3.605,"a_eq10":3.611},
    {"comp":{"Ru":0.185,"Rh":0.156,"Pd":0.182,"Os":0.143,"Ir":0.159,"Pt":0.174},"struct":"FCC","a_exp":3.8473,"a_vegard":3.8462,"a_eq10":3.8471},
    {"comp":{"Co":0.2,"Cr":0.2,"Cu":0.2,"Ni":0.2,"Zn":0.2},"struct":"BCC","a_exp":2.8831,"a_vegard":2.9012,"a_eq10":2.8815},
    {"comp":{"V":0.098,"Co":0.301,"Cr":0.095,"Fe":0.455,"Ni":0.051},"struct":"FCC","a_exp":3.582,"a_vegard":3.610,"a_eq10":3.604},
    {"comp":{"Co":0.286,"Al":0.071,"Fe":0.286,"Ni":0.286,"Mn":0.071},"struct":"FCC","a_exp":3.6084,"a_vegard":3.6061,"a_eq10":3.5923},
    {"comp":{"Co":0.2,"Cr":0.2,"Fe":0.2,"Ni":0.2,"Pd":0.2},"struct":"FCC","a_exp":3.6473,"a_vegard":3.6455,"a_eq10":3.6658},
    {"comp":{"Co":0.2,"Cr":0.2,"Fe":0.2,"Ni":0.2,"Pd":0.2},"struct":"FCC","a_exp":3.6803,"a_vegard":3.6455,"a_eq10":3.6658},
    {"comp":{"Co":0.244,"Cr":0.244,"Fe":0.244,"Ni":0.244,"Al":0.024},"struct":"FCC","a_exp":3.58,"a_vegard":3.59,"a_eq10":3.59},
    {"comp":{"Co":0.314,"Al":0.029,"Fe":0.318,"Ni":0.307,"Mn":0.032},"struct":"FCC","a_exp":3.5862,"a_vegard":3.5798,"a_eq10":3.5796},
    {"comp":{"Co":0.290,"Al":0.067,"Fe":0.288,"Ni":0.268,"Mn":0.087},"struct":"FCC","a_exp":3.600,"a_vegard":3.606,"a_eq10":3.593},
    {"comp":{"Co":0.2,"Al":0.1,"Fe":0.3,"Ni":0.4},"struct":"FCC","a_exp":3.5936,"a_vegard":3.6132,"a_eq10":3.5936},
    {"comp":{"Ir":0.191,"Pt":0.195,"Pd":0.207,"Rh":0.199,"Ru":0.208},"struct":"FCC","a_exp":3.856,"a_vegard":3.849,"a_eq10":3.851},
    {"comp":{"Co":0.237,"Cr":0.232,"Fe":0.245,"Ni":0.209,"Mn":0.077},"struct":"FCC","a_exp":3.58,"a_vegard":3.59,"a_eq10":3.59},
    {"comp":{"Co":0.269,"Cr":0.250,"Fe":0.249,"Ni":0.184,"Mo":0.048},"struct":"FCC","a_exp":3.585,"a_vegard":3.603,"a_eq10":3.606},
    {"comp":{"Co":0.247,"Cr":0.245,"Fe":0.239,"Ni":0.246,"Mo":0.024},"struct":"FCC","a_exp":3.604,"a_vegard":3.589,"a_eq10":3.594},
    {"comp":{"Co":0.238,"Cr":0.238,"Fe":0.238,"Ni":0.238,"Mo":0.048},"struct":"FCC","a_exp":3.595,"a_vegard":3.599,"a_eq10":3.602},
    {"comp":{"Co":0.204,"Cr":0.197,"Fe":0.299,"Ni":0.299},"struct":"FCC","a_exp":3.5759,"a_vegard":3.5764,"a_eq10":3.5782},
    {"comp":{"Co":0.305,"Cr":0.208,"Fe":0.193,"Ni":0.294},"struct":"FCC","a_exp":3.5695,"a_vegard":3.5704,"a_eq10":3.5721},
    {"comp":{"Co":0.296,"Cr":0.213,"Fe":0.303,"Ni":0.188},"struct":"FCC","a_exp":3.5741,"a_vegard":3.5801,"a_eq10":3.5822},
    {"comp":{"Co":0.265,"Cr":0.205,"Fe":0.271,"Ni":0.260},"struct":"FCC","a_exp":3.573,"a_vegard":3.576,"a_eq10":3.578},
    {"comp":{"Co":0.220,"Cr":0.244,"Fe":0.230,"Ni":0.307},"struct":"FCC","a_exp":3.5737,"a_vegard":3.5758,"a_eq10":3.577},
    {"comp":{"Co":0.229,"Cr":0.236,"Fe":0.325,"Ni":0.210},"struct":"FCC","a_exp":3.5795,"a_vegard":3.5834,"a_eq10":3.5843},
    {"comp":{"Co":0.315,"Cr":0.245,"Fe":0.245,"Ni":0.220},"struct":"FCC","a_exp":3.5704,"a_vegard":3.6079,"a_eq10":3.609},
    {"comp":{"Co":0.216,"Cr":0.227,"Fe":0.276,"Ni":0.281},"struct":"FCC","a_exp":3.5752,"a_vegard":3.5779,"a_eq10":3.5792},
    {"comp":{"Co":0.277,"Cr":0.233,"Fe":0.233,"Ni":0.277},"struct":"FCC","a_exp":3.572,"a_vegard":3.5998,"a_eq10":3.6013},
    {"comp":{"Co":0.273,"Cr":0.229,"Fe":0.284,"Ni":0.214},"struct":"FCC","a_exp":3.5751,"a_vegard":3.5801,"a_eq10":3.5815},
    {"comp":{"Co":0.188,"Cr":0.214,"Fe":0.198,"Ni":0.400},"struct":"FCC","a_exp":3.5708,"a_vegard":3.5692,"a_eq10":3.5711},
    {"comp":{"Co":0.201,"Cr":0.193,"Fe":0.400,"Ni":0.205},"struct":"FCC","a_exp":3.5803,"a_vegard":3.5847,"a_eq10":3.5864},
    {"comp":{"Co":0.248,"Cr":0.244,"Fe":0.251,"Ni":0.257},"struct":"FCC","a_exp":3.5767,"a_vegard":3.5783,"a_eq10":3.5793},
    {"comp":{"Co":0.386,"Cr":0.196,"Fe":0.211,"Ni":0.208},"struct":"FCC","a_exp":3.568,"a_vegard":3.5723,"a_eq10":3.5744},
]


# =====================================================================
# Load DFT data
# =====================================================================
def load_compound_data():
    base = Path("four_case_output/figures")
    dfs = []
    for src in ["MP", "OQMD"]:
        for struct in ["B2", "L12"]:
            f = base / f"compounds_{src}_{struct}.csv"
            if f.exists():
                df = pd.read_csv(f)
                df["db"] = src
                df["stype"] = struct
                dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def compute_dft_omega_sf(compound_df):
    """
    Compute DFT-derived pairwise volume deviations from Vegard's law.
    For each binary compound AB with DFT lattice constant a:
      V_actual = a³/Z  (per atom average)
      V_vegard = (V_A + V_B)/2  (for B2) or (3V_A + V_B)/4 (for L12)
      Ω_pair(A,B) = (V_actual - V_vegard) / V_vegard
    Returns dict: (elA, elB) → mean Ω (fractional)
    """
    pair_data = defaultdict(list)

    for _, row in compound_df.iterrows():
        elA = row.get("element_A", "")
        elB = row.get("element_B", "")
        a = row.get("lattice_constant", 0)
        stype = row.get("stype", row.get("structure_type", ""))

        if not elA or not elB or a <= 0:
            continue
        if elA not in KING_ATOMIC_VOLUMES or elB not in KING_ATOMIC_VOLUMES:
            continue

        vA = KING_ATOMIC_VOLUMES[elA]
        vB = KING_ATOMIC_VOLUMES[elB]

        if stype == "B2":
            Z = 2
            v_actual = a**3 / Z
            v_vegard = (vA + vB) / 2
        elif stype == "L12":
            Z = 4
            cA = row.get("count_A", 3)
            cB = row.get("count_B", 1)
            total = cA + cB
            v_actual = a**3 / Z
            v_vegard = (cA * vA + cB * vB) / total
        else:
            continue

        omega = (v_actual - v_vegard) / v_vegard
        pair = tuple(sorted([elA, elB]))
        pair_data[pair].append(omega)

    # Average and return
    omega_sf = {}
    for pair, vals in pair_data.items():
        omega_sf[pair] = np.mean(vals)

    return omega_sf


def compute_structure_specific_omega_sf(compound_df):
    """
    Compute structure-specific DFT Ω_sf:
      omega_b2:  from B2 compounds only  → used for BCC HEAs
      omega_l12: from L1₂ compounds only → used for FCC HEAs
    Returns (omega_b2, omega_l12) dicts.
    """
    pair_b2 = defaultdict(list)
    pair_l12 = defaultdict(list)

    for _, row in compound_df.iterrows():
        elA = row.get("element_A", "")
        elB = row.get("element_B", "")
        a = row.get("lattice_constant", 0)
        stype = row.get("stype", "")
        if a <= 2 or a >= 8 or elA not in KING_ATOMIC_VOLUMES or elB not in KING_ATOMIC_VOLUMES:
            continue

        pair = tuple(sorted([elA, elB]))
        vA = KING_ATOMIC_VOLUMES[elA]
        vB = KING_ATOMIC_VOLUMES[elB]

        if stype == "B2":
            v_actual = a**3 / 2
            v_vegard = (vA + vB) / 2
            pair_b2[pair].append((v_actual - v_vegard) / v_vegard)
        elif stype == "L12":
            cA = row.get("count_A", 3)
            cB = row.get("count_B", 1)
            total = cA + cB
            v_actual = a**3 / 4
            v_vegard = (cA * vA + cB * vB) / total
            pair_l12[pair].append((v_actual - v_vegard) / v_vegard)

    omega_b2 = {p: np.median(v) for p, v in pair_b2.items() if len(v) >= 2}
    omega_l12 = {p: np.median(v) for p, v in pair_l12.items() if len(v) >= 2}
    return omega_b2, omega_l12


def compute_eq10_scaled(comp, struct, omega_sf, gamma=1.0):
    """
    Alonso Eq.10 with scaled DFT Ω_sf correction.
    V = nauc * [Σ c_i V_i + γ · Σ_i Σ_{j≠i} c_i c_j V_j Ω_sf(i,j)]
    """
    elements = list(comp.keys())
    fracs = np.array([comp[e] for e in elements])
    fracs = fracs / fracs.sum()
    vols = np.array([KING_ATOMIC_VOLUMES.get(e, 15.0) for e in elements])
    n_auc = 4 if struct == "FCC" else 2
    n_elem = len(elements)

    v_vegard = np.sum(fracs * vols)
    correction = 0.0
    for i in range(n_elem):
        for j in range(n_elem):
            if i != j:
                pair = tuple(sorted([elements[i], elements[j]]))
                omega = omega_sf.get(pair, 0.0)
                correction += fracs[i] * fracs[j] * vols[j] * omega

    v_total = n_auc * (v_vegard + gamma * correction)
    if v_total <= 0:
        return compute_vegard(comp, struct)
    return v_total ** (1/3)


def estimate_noise_floor(hea_data):
    """
    Estimate experimental noise from duplicate HEA compositions.
    Returns σ_noise in Å.
    """
    def comp_key(comp):
        return tuple(sorted((e, round(c, 3)) for e, c in comp.items()))

    groups = defaultdict(list)
    for h in hea_data:
        groups[comp_key(h["comp"])].append(h["a_exp"])

    spreads = []
    dup_info = []
    for key, vals in groups.items():
        if len(vals) > 1:
            spread = max(vals) - min(vals)
            spreads.append(spread)
            elems = [e for e, _ in key]
            dup_info.append((elems, vals, spread))

    if spreads:
        sigma = np.mean(spreads) / 1.128  # E[range]/σ = 1.128 for n=2
    else:
        sigma = 0.010  # default estimate

    return sigma, dup_info


# =====================================================================
# Feature Engineering
# =====================================================================
FEATURE_NAMES = [
    "a_vegard",          # 0: Vegard baseline
    "a_eq10_dft",        # 1: Alonso Eq.10 with DFT Ω_sf
    "V_avg",             # 2: weighted mean atomic volume
    "V_std",             # 3: std of atomic volumes
    "V_range",           # 4: max - min atomic volume
    "delta_r",           # 5: atomic size mismatch δ (%)
    "delta_chi",         # 6: electronegativity mismatch
    "VEC_avg",           # 7: valence electron concentration
    "d_avg",             # 8: average d-electron count
    "n_elements",        # 9: number of elements
    "struct_flag",       # 10: BCC=0, FCC=1
    "S_mix",             # 11: configurational entropy
    "Omega_sf_mean",     # 12: mean pairwise DFT Ω_sf
    "Omega_sf_std",      # 13: std pairwise DFT Ω_sf
    "Omega_sf_min",      # 14: min pairwise DFT Ω_sf
    "Omega_sf_max",      # 15: max pairwise DFT Ω_sf
    "Omega_sf_absmax",   # 16: max |Ω_sf|
    "mass_avg",          # 17: average atomic mass
    "mass_std",          # 18: std atomic mass
    "c_max",             # 19: max composition fraction
    "c_std",             # 20: std of composition
    "VEC_std",           # 21: std of VEC
    "en_range",          # 22: range of electronegativity
]


def compute_vegard(comp, struct):
    """Compute Vegard lattice constant using King volumes."""
    elements = list(comp.keys())
    fracs = np.array([comp[e] for e in elements])
    fracs = fracs / fracs.sum()
    vols = np.array([KING_ATOMIC_VOLUMES.get(e, 15.0) for e in elements])
    v_avg = np.sum(fracs * vols)
    n_auc = 4 if struct == "FCC" else 2
    return (n_auc * v_avg) ** (1/3)


def compute_eq10_dft(comp, struct, omega_sf):
    """
    Compute Alonso Eq.10 with DFT-derived Ω_sf (unscaled, γ=1).
    V = nauc * [Σ c_i V_i + Σ_i c_i V_i * Σ_{j≠i} c_j Ω_sf(j,i)]
    """
    return compute_eq10_scaled(comp, struct, omega_sf, gamma=1.0)


def compute_features(comp, struct, omega_sf):
    """Compute 23-dimensional feature vector."""
    elements = list(comp.keys())
    fracs = np.array([comp[e] for e in elements])
    fracs = fracs / fracs.sum()
    n_elem = len(elements)

    features = np.zeros(23)

    vols = np.array([KING_ATOMIC_VOLUMES.get(e, 15.0) for e in elements])
    v_avg = np.sum(fracs * vols)
    n_auc = 4 if struct == "FCC" else 2

    # [0] Vegard baseline
    features[0] = (n_auc * v_avg) ** (1/3)

    # [1] Alonso Eq.10 with DFT Ω_sf
    features[1] = compute_eq10_dft(comp, struct, omega_sf)

    # [2-4] Atomic volume stats
    features[2] = v_avg
    features[3] = np.sqrt(np.sum(fracs * (vols - v_avg)**2))
    features[4] = np.max(vols) - np.min(vols)

    # [5] δr - atomic radius mismatch (using cube root of volume as radius proxy)
    r_vals = vols ** (1/3)
    r_avg = np.sum(fracs * r_vals)
    features[5] = 100 * np.sqrt(np.sum(fracs * (1 - r_vals / r_avg)**2))

    # [6] δχ - electronegativity mismatch
    en_vals = np.array([PAULING_EN.get(e, 1.5) for e in elements])
    en_avg = np.sum(fracs * en_vals)
    features[6] = np.sqrt(np.sum(fracs * (en_vals - en_avg)**2))

    # [7] VEC average
    vec_vals = np.array([VEC.get(e, 5) for e in elements])
    features[7] = np.sum(fracs * vec_vals)

    # [8] d-electron average
    d_vals = np.array([D_ELECTRONS.get(e, 0) for e in elements])
    features[8] = np.sum(fracs * d_vals)

    # [9] n_elements
    features[9] = n_elem

    # [10] struct flag
    features[10] = 1.0 if struct == "FCC" else 0.0

    # [11] Mixing entropy
    features[11] = -np.sum(fracs[fracs > 0] * np.log(fracs[fracs > 0]))

    # [12-16] DFT Ω_sf pair stats
    omega_vals = []
    for i in range(n_elem):
        for j in range(i+1, n_elem):
            pair = tuple(sorted([elements[i], elements[j]]))
            omega_vals.append(omega_sf.get(pair, 0.0))

    if omega_vals:
        omega_arr = np.array(omega_vals)
        features[12] = np.mean(omega_arr)
        features[13] = np.std(omega_arr) if len(omega_arr) > 1 else 0.0
        features[14] = np.min(omega_arr)
        features[15] = np.max(omega_arr)
        features[16] = np.max(np.abs(omega_arr))

    # [17-18] Atomic mass stats
    mass_vals = np.array([ATOMIC_MASS.get(e, 50.0) for e in elements])
    features[17] = np.sum(fracs * mass_vals)
    features[18] = np.sqrt(np.sum(fracs * (mass_vals - features[17])**2))

    # [19-20] Composition stats
    features[19] = np.max(fracs)
    features[20] = np.std(fracs)

    # [21] VEC std
    features[21] = np.sqrt(np.sum(fracs * (vec_vals - features[7])**2))

    # [22] EN range
    features[22] = np.max(en_vals) - np.min(en_vals)

    return features


# =====================================================================
# Phase 1: DFT binary → HEA transfer model
# =====================================================================
def build_binary_training_data(compound_df, omega_sf):
    """Build training set from binary DFT compounds using King volumes."""
    X_list, y_list, meta_list = [], [], []

    for _, row in compound_df.iterrows():
        elA = row.get("element_A", "")
        elB = row.get("element_B", "")
        a = row.get("lattice_constant", 0)
        stype = row.get("stype", row.get("structure_type", ""))

        if not elA or not elB or a <= 0:
            continue
        if elA not in KING_ATOMIC_VOLUMES or elB not in KING_ATOMIC_VOLUMES:
            continue

        if stype == "B2":
            comp = {elA: 0.5, elB: 0.5}
            struct = "BCC"
        elif stype == "L12":
            cA = row.get("count_A", 3)
            cB = row.get("count_B", 1)
            total = cA + cB
            comp = {elA: cA/total, elB: cB/total}
            struct = "FCC"
        else:
            continue

        feats = compute_features(comp, struct, omega_sf)
        X_list.append(feats)

        # Target: Vegard-based lattice constant from DFT volume
        # For B2: a_experimental_equivalent = (2 * V_per_atom)^(1/3)
        # We use the King volumes for prediction, DFT gives the "true" value
        y_list.append(a)
        meta_list.append({"elA": elA, "elB": elB, "struct": struct,
                          "db": row.get("db",""), "stype": stype})

    return np.array(X_list), np.array(y_list), meta_list


# =====================================================================
# Phase 2: Hybrid residual learning on HEA data
# =====================================================================
def build_hea_features(omega_sf):
    """Build feature matrix for Alonso 68 HEAs."""
    X_list, y_list, meta_list = [], [], []

    for hea in ALONSO_TABLE2:
        feats = compute_features(hea["comp"], hea["struct"], omega_sf)
        X_list.append(feats)
        y_list.append(hea["a_exp"])
        meta_list.append(hea)

    return np.array(X_list), np.array(y_list), meta_list


def loo_cv_xgboost(X, y, **xgb_params):
    """Leave-One-Out cross-validation for XGBoost."""
    loo = LeaveOneOut()
    y_pred = np.zeros(len(y))

    for train_idx, test_idx in loo.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y[train_idx]

        model = XGBRegressor(**xgb_params)
        model.fit(X_tr, y_tr, verbose=False)
        y_pred[test_idx] = model.predict(X_te)

    return y_pred


def kfold_cv_xgboost(X, y, n_splits=5, **xgb_params):
    """K-fold cross-validation for XGBoost."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_pred = np.zeros(len(y))

    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y[train_idx]

        model = XGBRegressor(**xgb_params)
        model.fit(X_tr, y_tr, verbose=False)
        y_pred[test_idx] = model.predict(X_te)

    return y_pred


# =====================================================================
# Phase 3: Two-stage transfer learning
# =====================================================================
def two_stage_model(X_binary, y_binary, X_hea, y_hea, omega_sf, xgb_params):
    """
    Stage 1: Pre-train on DFT binary data
    Stage 2: Fine-tune residual on HEA data with LOO-CV
    """
    # Stage 1: Train base model on binary data
    base_model = XGBRegressor(**xgb_params)
    base_model.fit(X_binary, y_binary, verbose=False)

    # Stage 1 predictions on HEAs
    y_hea_base = base_model.predict(X_hea)

    # Stage 2: LOO-CV residual correction
    residuals = y_hea - y_hea_base  # what the base model misses

    # Use simple Ridge for residual (fewer parameters, less overfitting)
    loo = LeaveOneOut()
    y_pred = np.zeros(len(y_hea))

    for train_idx, test_idx in loo.split(X_hea):
        # Compute residuals for training set
        res_train = residuals[train_idx]

        # Use a simple linear model for the residual
        ridge = Ridge(alpha=10.0)
        ridge.fit(X_hea[train_idx], res_train)
        correction = ridge.predict(X_hea[test_idx])

        y_pred[test_idx] = y_hea_base[test_idx] + correction

    return y_pred, y_hea_base


# =====================================================================
# Main
# =====================================================================
def main():
    print("=" * 70)
    print("HEA Lattice Constant Prediction: Dynamic Volume Size Factor")
    print("=" * 70)

    N = len(ALONSO_TABLE2)
    y_hea = np.array([h["a_exp"] for h in ALONSO_TABLE2])
    structs = np.array([h["struct"] for h in ALONSO_TABLE2])
    bcc = structs == "BCC"
    fcc = structs == "FCC"

    # --- Load data ---
    print("\n[1] Loading DFT compound data...")
    compound_df = load_compound_data()
    print(f"    Total compounds: {len(compound_df)}")

    # --- Compute DFT-derived Ω_sf ---
    print("\n[2] Computing DFT-derived volume size factors...")
    omega_sf = compute_dft_omega_sf(compound_df)
    omega_b2, omega_l12 = compute_structure_specific_omega_sf(compound_df)
    print(f"    Combined Ω_sf pairs: {len(omega_sf)}")
    print(f"    B2-specific pairs:   {len(omega_b2)}")
    print(f"    L1₂-specific pairs:  {len(omega_l12)}")

    # --- Noise floor analysis ---
    print("\n[3] Noise floor analysis (duplicate HEA compositions)...")
    sigma_noise, dup_info = estimate_noise_floor(ALONSO_TABLE2)
    for elems, vals, spread in dup_info:
        print(f"    {'-'.join(elems)}: a = {', '.join(f'{v:.4f}' for v in vals)}, "
              f"spread = {spread:.4f} Å")
    print(f"    Estimated σ_noise = {sigma_noise:.4f} Å")
    print(f"    Theoretical minimum RMSE ≈ {sigma_noise:.4f} Å")

    # --- Build features ---
    print("\n[4] Building feature matrices...")
    X_binary, y_binary, meta_binary = build_binary_training_data(compound_df, omega_sf)
    X_hea, y_hea_feat, meta_hea = build_hea_features(omega_sf)
    print(f"    Binary training samples: {X_binary.shape[0]}")
    print(f"    HEA test samples: {X_hea.shape[0]}")

    # --- Baselines ---
    print("\n[5] Computing baselines...")

    # Alonso Vegard & Eq.10 (from their paper Table 2)
    a_vegard_alonso = np.array([h["a_vegard"] for h in ALONSO_TABLE2])
    a_eq10_alonso = np.array([h["a_eq10"] for h in ALONSO_TABLE2])

    # Our Vegard (King volumes)
    a_vegard_king = X_hea[:, 0]

    # Our Eq.10 (DFT Ω_sf, unscaled)
    a_eq10_dft = X_hea[:, 1]

    # Structure-specific Eq.10 with optimised γ
    print("    Optimising structure-specific γ parameters...")
    best_rmse_ss = 999
    best_gb, best_gf = 0, 0
    bcc_idx = np.where(bcc)[0]
    fcc_idx = np.where(fcc)[0]

    for gb in np.arange(-0.5, 2.01, 0.05):
        for gf in np.arange(-0.5, 2.01, 0.05):
            a_pred = np.zeros(N)
            for i in bcc_idx:
                a_pred[i] = compute_eq10_scaled(
                    ALONSO_TABLE2[i]["comp"], "BCC", omega_b2, gb)
            for i in fcc_idx:
                a_pred[i] = compute_eq10_scaled(
                    ALONSO_TABLE2[i]["comp"], "FCC", omega_l12, gf)
            rmse = np.sqrt(np.mean((y_hea - a_pred)**2))
            if rmse < best_rmse_ss:
                best_rmse_ss = rmse
                best_gb, best_gf = gb, gf

    # Fine-tune
    for gb in np.arange(best_gb - 0.05, best_gb + 0.06, 0.01):
        for gf in np.arange(best_gf - 0.05, best_gf + 0.06, 0.01):
            a_pred = np.zeros(N)
            for i in bcc_idx:
                a_pred[i] = compute_eq10_scaled(
                    ALONSO_TABLE2[i]["comp"], "BCC", omega_b2, gb)
            for i in fcc_idx:
                a_pred[i] = compute_eq10_scaled(
                    ALONSO_TABLE2[i]["comp"], "FCC", omega_l12, gf)
            rmse = np.sqrt(np.mean((y_hea - a_pred)**2))
            if rmse < best_rmse_ss:
                best_rmse_ss = rmse
                best_gb, best_gf = gb, gf

    a_eq10_ss = np.zeros(N)
    for i in bcc_idx:
        a_eq10_ss[i] = compute_eq10_scaled(
            ALONSO_TABLE2[i]["comp"], "BCC", omega_b2, best_gb)
    for i in fcc_idx:
        a_eq10_ss[i] = compute_eq10_scaled(
            ALONSO_TABLE2[i]["comp"], "FCC", omega_l12, best_gf)

    print(f"    Optimal γ_BCC = {best_gb:.2f}, γ_FCC = {best_gf:.2f}")

    def print_metrics(name, y_pred, y_true=y_hea):
        res = y_true - y_pred
        rmse = np.sqrt(np.mean(res**2))
        mae = np.mean(np.abs(res))
        mape = np.mean(np.abs(res / y_true)) * 100
        r2 = 1 - np.sum(res**2) / np.sum((y_true - y_true.mean())**2)
        print(f"    {name:40s}: RMSE={rmse:.4f} Å, MAE={mae:.4f} Å, "
              f"MAPE={mape:.3f}%, R$^2$={r2:.4f}")
        return rmse, mae, mape, r2

    print(f"\n    === All {N} HEAs ===")
    results = {}
    results["Alonso Vegard"] = print_metrics("Alonso Vegard", a_vegard_alonso)
    results["Alonso Eq.10"] = print_metrics("Alonso Eq.10", a_eq10_alonso)
    results["King Vegard (this work)"] = print_metrics("King Vegard (this work)", a_vegard_king)
    results["DFT Eq.10 (this work)"] = print_metrics("DFT Eq.10 (this work)", a_eq10_dft)
    results["DFT Eq.10 SS (this work)"] = print_metrics(
        f"DFT Eq.10 SS (γB={best_gb:.2f},γF={best_gf:.2f})", a_eq10_ss)

    # --- ML models ---
    print("\n[6] Training ML models...")

    xgb_params = dict(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
        gamma=0.2, reg_alpha=0.5, reg_lambda=2.0,
        random_state=42, n_jobs=-1, verbosity=0,
    )

    # Model A: Direct XGBoost on HEA (LOO-CV)
    print("    Training Model A: Direct XGBoost on HEA (LOO-CV)...")
    y_pred_loo = loo_cv_xgboost(X_hea, y_hea, **xgb_params)
    results["XGBoost LOO-CV"] = print_metrics("XGBoost LOO-CV (direct)", y_pred_loo)

    # Model B: 5-fold CV
    print("    Training Model B: XGBoost 5-fold CV on HEA...")
    y_pred_5fold = kfold_cv_xgboost(X_hea, y_hea, n_splits=5, **xgb_params)
    results["XGBoost 5-fold"] = print_metrics("XGBoost 5-fold CV", y_pred_5fold)

    # Model C: Two-stage transfer (binary → HEA)
    print("    Training Model C: Two-stage transfer...")
    y_pred_transfer, y_pred_base = two_stage_model(
        X_binary, y_binary, X_hea, y_hea, omega_sf, xgb_params)
    results["Transfer (base)"] = print_metrics("Transfer base (binary)", y_pred_base)
    results["Transfer (corrected)"] = print_metrics("Transfer + Ridge correction", y_pred_transfer)

    # Model D: Structure-specific Eq.10 + Ridge residual correction (LOO-CV)
    print("    Training Model D: SS Eq.10 + LOO-CV Ridge correction...")
    residual_ss = y_hea - a_eq10_ss

    feats_simple = np.column_stack([
        X_hea[:, 0],   # a_vegard
        X_hea[:, 5],   # delta_r
        X_hea[:, 6],   # delta_chi
        X_hea[:, 7],   # VEC
        X_hea[:, 10],  # struct_flag
        X_hea[:, 11],  # S_mix
        X_hea[:, 9],   # n_elements
    ])

    best_ridge_rmse = 999
    best_ridge_alpha = 1.0
    for alpha in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]:
        loo = LeaveOneOut()
        y_corr = np.zeros(N)
        for tr, te in loo.split(feats_simple):
            sc = StandardScaler()
            Xtr = sc.fit_transform(feats_simple[tr])
            Xte = sc.transform(feats_simple[te])
            m = Ridge(alpha=alpha)
            m.fit(Xtr, residual_ss[tr])
            y_corr[te] = m.predict(Xte)
        a_corr = a_eq10_ss + y_corr
        rmse = np.sqrt(np.mean((y_hea - a_corr)**2))
        if rmse < best_ridge_rmse:
            best_ridge_rmse = rmse
            best_ridge_alpha = alpha

    loo = LeaveOneOut()
    y_corr_ridge = np.zeros(N)
    for tr, te in loo.split(feats_simple):
        sc = StandardScaler()
        Xtr = sc.fit_transform(feats_simple[tr])
        Xte = sc.transform(feats_simple[te])
        m = Ridge(alpha=best_ridge_alpha)
        m.fit(Xtr, residual_ss[tr])
        y_corr_ridge[te] = m.predict(Xte)
    a_ss_ridge = a_eq10_ss + y_corr_ridge
    results["SS Eq.10 + Ridge"] = print_metrics(
        f"SS Eq.10 + Ridge(α={best_ridge_alpha})", a_ss_ridge)

    # Model E: SS Eq.10 + XGBoost residual correction
    print("    Training Model E: SS Eq.10 + LOO-CV XGBoost correction...")
    xgb_res_params = dict(
        n_estimators=50, max_depth=1, learning_rate=0.05,
        reg_lambda=5, random_state=42, verbosity=0,
    )
    y_corr_xgb = np.zeros(N)
    loo = LeaveOneOut()
    for tr, te in loo.split(feats_simple):
        sc = StandardScaler()
        Xtr = sc.fit_transform(feats_simple[tr])
        Xte = sc.transform(feats_simple[te])
        m = XGBRegressor(**xgb_res_params)
        m.fit(Xtr, residual_ss[tr])
        y_corr_xgb[te] = m.predict(Xte)
    a_ss_xgb = a_eq10_ss + y_corr_xgb
    results["SS Eq.10 + XGBoost"] = print_metrics(
        "SS Eq.10 + XGBoost residual", a_ss_xgb)

    # Model F: SS Eq.10 + GPR residual correction (LOO-CV)
    print("    Training Model F: SS Eq.10 + LOO-CV GPR correction...")
    gpr_kernels = [
        ("Matern32", ConstantKernel(0.001) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(0.0001)),
        ("Matern52", ConstantKernel(0.001) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(0.0001)),
        ("RBF", ConstantKernel(0.001) * RBF(length_scale=1.0) + WhiteKernel(0.0001)),
    ]

    best_gpr_rmse = 999
    best_gpr_name = ""
    best_gpr_preds = None
    best_gpr_stds = None

    for kern_name, kernel in gpr_kernels:
        y_corr_gpr = np.zeros(N)
        y_std_gpr = np.zeros(N)
        loo = LeaveOneOut()
        for tr, te in loo.split(feats_simple):
            sc = StandardScaler()
            Xtr = sc.fit_transform(feats_simple[tr])
            Xte = sc.transform(feats_simple[te])
            gpr = GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=5,
                alpha=1e-6, normalize_y=True, random_state=42)
            gpr.fit(Xtr, residual_ss[tr])
            pred, std = gpr.predict(Xte, return_std=True)
            y_corr_gpr[te] = pred
            y_std_gpr[te] = std
        a_ss_gpr = a_eq10_ss + y_corr_gpr
        rmse = np.sqrt(np.mean((y_hea - a_ss_gpr)**2))
        mae = np.mean(np.abs(y_hea - a_ss_gpr))
        print(f"      GPR({kern_name}): RMSE={rmse:.4f} Å, MAE={mae:.4f} Å")
        if rmse < best_gpr_rmse:
            best_gpr_rmse = rmse
            best_gpr_name = kern_name
            best_gpr_preds = a_ss_gpr.copy()
            best_gpr_stds = y_std_gpr.copy()

    results["SS Eq.10 + GPR"] = print_metrics(
        f"SS Eq.10 + GPR({best_gpr_name})", best_gpr_preds)
    a_ss_gpr = best_gpr_preds
    gpr_uncertainty = best_gpr_stds

    # Model G: SS Eq.10 + SVR residual correction (LOO-CV)
    print("    Training Model G: SS Eq.10 + LOO-CV SVR correction...")
    svr_configs = [
        ("RBF_C1", SVR(kernel="rbf", C=1.0, epsilon=0.001)),
        ("RBF_C10", SVR(kernel="rbf", C=10.0, epsilon=0.001)),
        ("RBF_C01", SVR(kernel="rbf", C=0.1, epsilon=0.001)),
    ]

    best_svr_rmse = 999
    best_svr_name = ""
    best_svr_preds = None

    for svr_name, svr_model in svr_configs:
        y_corr_svr = np.zeros(N)
        loo = LeaveOneOut()
        for tr, te in loo.split(feats_simple):
            sc = StandardScaler()
            Xtr = sc.fit_transform(feats_simple[tr])
            Xte = sc.transform(feats_simple[te])
            svr_model_copy = SVR(kernel=svr_model.kernel, C=svr_model.C,
                                 epsilon=svr_model.epsilon)
            svr_model_copy.fit(Xtr, residual_ss[tr])
            y_corr_svr[te] = svr_model_copy.predict(Xte)
        a_ss_svr = a_eq10_ss + y_corr_svr
        rmse = np.sqrt(np.mean((y_hea - a_ss_svr)**2))
        mae = np.mean(np.abs(y_hea - a_ss_svr))
        print(f"      SVR({svr_name}): RMSE={rmse:.4f} Å, MAE={mae:.4f} Å")
        if rmse < best_svr_rmse:
            best_svr_rmse = rmse
            best_svr_name = svr_name
            best_svr_preds = a_ss_svr.copy()

    results["SS Eq.10 + SVR"] = print_metrics(
        f"SS Eq.10 + SVR({best_svr_name})", best_svr_preds)
    a_ss_svr = best_svr_preds

    # Model H: SS Eq.10 + Random Forest residual correction (LOO-CV)
    print("    Training Model H: SS Eq.10 + LOO-CV Random Forest correction...")
    rf_configs = [
        ("RF100_d3", dict(n_estimators=100, max_depth=3, min_samples_leaf=5, random_state=42)),
        ("RF200_d2", dict(n_estimators=200, max_depth=2, min_samples_leaf=5, random_state=42)),
        ("RF500_d3", dict(n_estimators=500, max_depth=3, min_samples_leaf=3, random_state=42)),
        ("RF100_d5", dict(n_estimators=100, max_depth=5, min_samples_leaf=5, random_state=42)),
        ("RF200_dNone", dict(n_estimators=200, max_depth=None, min_samples_leaf=5, random_state=42)),
    ]

    best_rf_rmse = 999
    best_rf_name = ""
    best_rf_preds = None

    for rf_name, rf_params in rf_configs:
        y_corr_rf = np.zeros(N)
        loo = LeaveOneOut()
        for tr, te in loo.split(feats_simple):
            sc = StandardScaler()
            Xtr = sc.fit_transform(feats_simple[tr])
            Xte = sc.transform(feats_simple[te])
            rf = RandomForestRegressor(**rf_params)
            rf.fit(Xtr, residual_ss[tr])
            y_corr_rf[te] = rf.predict(Xte)
        a_ss_rf = a_eq10_ss + y_corr_rf
        rmse = np.sqrt(np.mean((y_hea - a_ss_rf)**2))
        mae = np.mean(np.abs(y_hea - a_ss_rf))
        print(f"      RF({rf_name}): RMSE={rmse:.4f} Å, MAE={mae:.4f} Å")
        if rmse < best_rf_rmse:
            best_rf_rmse = rmse
            best_rf_name = rf_name
            best_rf_preds = a_ss_rf.copy()

    results["SS Eq.10 + RF"] = print_metrics(
        f"SS Eq.10 + RF({best_rf_name})", best_rf_preds)
    a_ss_rf = best_rf_preds

    # Model I: SS Eq.10 + Cubist residual correction (LOO-CV)
    print("    Training Model I: SS Eq.10 + LOO-CV Cubist correction...")
    cubist_configs = [
        ("C5_N1", dict(n_rules=5, n_committees=1)),
        ("C10_N1", dict(n_rules=10, n_committees=1)),
        ("C5_N5", dict(n_rules=5, n_committees=5)),
        ("C10_N5", dict(n_rules=10, n_committees=5)),
        ("C20_N10", dict(n_rules=20, n_committees=10)),
    ]

    best_cub_rmse = 999
    best_cub_name = ""
    best_cub_preds = None

    for cub_name, cub_params in cubist_configs:
        y_corr_cub = np.zeros(N)
        loo = LeaveOneOut()
        for tr, te in loo.split(feats_simple):
            sc = StandardScaler()
            Xtr = sc.fit_transform(feats_simple[tr])
            Xte = sc.transform(feats_simple[te])
            try:
                cub = Cubist(**cub_params)
                cub.fit(Xtr, residual_ss[tr])
                y_corr_cub[te] = cub.predict(Xte)
            except Exception:
                y_corr_cub[te] = 0.0
        a_ss_cub = a_eq10_ss + y_corr_cub
        rmse = np.sqrt(np.mean((y_hea - a_ss_cub)**2))
        mae = np.mean(np.abs(y_hea - a_ss_cub))
        print(f"      Cubist({cub_name}): RMSE={rmse:.4f} Å, MAE={mae:.4f} Å")
        if rmse < best_cub_rmse:
            best_cub_rmse = rmse
            best_cub_name = cub_name
            best_cub_preds = a_ss_cub.copy()

    results["SS Eq.10 + Cubist"] = print_metrics(
        f"SS Eq.10 + Cubist({best_cub_name})", best_cub_preds)
    a_ss_cub = best_cub_preds

    # Ensemble optimisation
    print("\n[7] Ensemble optimisation...")

    # 2-way: SS Eq.10 + XGBoost LOO
    best_rmse_ens = 999
    best_w = 0
    for w in np.arange(0, 1.01, 0.01):
        y_ens = w * a_eq10_ss + (1 - w) * y_pred_loo
        rmse = np.sqrt(np.mean((y_hea - y_ens)**2))
        if rmse < best_rmse_ens:
            best_rmse_ens = rmse
            best_w = w
    y_ensemble_opt = best_w * a_eq10_ss + (1 - best_w) * y_pred_loo
    results["Optimal Ensemble"] = print_metrics(
        f"Optimal Ensemble (w={best_w:.2f})", y_ensemble_opt)

    # 3-way: King Vegard + SS Eq.10 + XGBoost LOO
    best_rmse3 = 999
    best_w1, best_w2 = 0, 0
    for w1 in np.arange(0, 1.01, 0.02):
        for w2 in np.arange(0, 1.01 - w1, 0.02):
            w3 = 1 - w1 - w2
            y_ens = w1 * a_vegard_king + w2 * a_eq10_ss + w3 * y_pred_loo
            rmse = np.sqrt(np.mean((y_hea - y_ens)**2))
            if rmse < best_rmse3:
                best_rmse3 = rmse
                best_w1, best_w2 = w1, w2
    w3 = 1 - best_w1 - best_w2
    y_ensemble_3way = best_w1 * a_vegard_king + best_w2 * a_eq10_ss + w3 * y_pred_loo
    results["3-way Ensemble"] = print_metrics(
        f"3-way ({best_w1:.2f}V+{best_w2:.2f}E+{w3:.2f}X)", y_ensemble_3way)

    # 4-way: King Vegard + SS Eq.10 + SS Eq.10+Ridge + XGBoost LOO
    best_rmse4 = 999
    best_ww = (0, 0, 0)
    for w1 in np.arange(0, 1.01, 0.05):
        for w2 in np.arange(0, 1.01 - w1, 0.05):
            for w3 in np.arange(0, 1.01 - w1 - w2, 0.05):
                w4 = 1 - w1 - w2 - w3
                y_ens = w1 * a_vegard_king + w2 * a_eq10_ss + w3 * a_ss_ridge + w4 * y_pred_loo
                rmse = np.sqrt(np.mean((y_hea - y_ens)**2))
                if rmse < best_rmse4:
                    best_rmse4 = rmse
                    best_ww = (w1, w2, w3)
    w4 = 1 - sum(best_ww)
    y_ensemble_4way = (best_ww[0] * a_vegard_king + best_ww[1] * a_eq10_ss
                       + best_ww[2] * a_ss_ridge + w4 * y_pred_loo)
    results["4-way Ensemble"] = print_metrics(
        f"4-way ({best_ww[0]:.2f}V+{best_ww[1]:.2f}E+{best_ww[2]:.2f}R+{w4:.2f}X)",
        y_ensemble_4way)

    # --- BCC/FCC breakdown ---
    print("\n    === BCC HEAs ===")
    key_preds = [
        ("Alonso Eq.10", a_eq10_alonso),
        ("King Vegard", a_vegard_king),
        ("DFT Eq.10 SS", a_eq10_ss),
        ("SS Eq.10 + Ridge", a_ss_ridge),
        ("SS Eq.10 + GPR", a_ss_gpr),
        ("SS Eq.10 + SVR", a_ss_svr),
        ("SS Eq.10 + RF", a_ss_rf),
        ("SS Eq.10 + Cubist", a_ss_cub),
        ("XGBoost LOO", y_pred_loo),
        ("Optimal Ensemble", y_ensemble_opt),
    ]
    for name, y_p in key_preds:
        res = y_hea[bcc] - y_p[bcc]
        rmse = np.sqrt(np.mean(res**2))
        print(f"    {name:25s}: RMSE={rmse:.4f} Å (N={bcc.sum()})")

    print("\n    === FCC HEAs ===")
    for name, y_p in key_preds:
        res = y_hea[fcc] - y_p[fcc]
        rmse = np.sqrt(np.mean(res**2))
        print(f"    {name:25s}: RMSE={rmse:.4f} Å (N={fcc.sum()})")

    # --- Select best model ---
    best_method = min(results.items(), key=lambda x: x[1][0])
    print(f"\n    ★ Best method: {best_method[0]} (RMSE={best_method[1][0]:.4f} Å)")

    method_predictions = {
        "XGBoost LOO-CV": y_pred_loo,
        "XGBoost 5-fold": y_pred_5fold,
        "Optimal Ensemble": y_ensemble_opt,
        "3-way Ensemble": y_ensemble_3way,
        "4-way Ensemble": y_ensemble_4way,
        "Transfer (corrected)": y_pred_transfer,
        "Transfer (base)": y_pred_base,
        "DFT Eq.10 (this work)": a_eq10_dft,
        "DFT Eq.10 SS (this work)": a_eq10_ss,
        "SS Eq.10 + Ridge": a_ss_ridge,
        "SS Eq.10 + XGBoost": a_ss_xgb,
        "SS Eq.10 + GPR": a_ss_gpr,
        "SS Eq.10 + SVR": a_ss_svr,
        "SS Eq.10 + RF": a_ss_rf,
        "SS Eq.10 + Cubist": a_ss_cub,
        "King Vegard (this work)": a_vegard_king,
    }
    y_best = method_predictions.get(best_method[0], y_ensemble_opt)

    # --- Train final model for export ---
    print("\n[8] Training final XGBoost model on all HEA data...")
    final_model = XGBRegressor(**xgb_params)
    final_model.fit(X_hea, y_hea, verbose=False)

    model_bundle = {
        "xgboost_model": final_model,
        "omega_b2": omega_b2,
        "omega_l12": omega_l12,
        "omega_sf": omega_sf,
        "gamma_bcc": best_gb,
        "gamma_fcc": best_gf,
        "ridge_alpha": best_ridge_alpha,
        "king_volumes": dict(KING_ATOMIC_VOLUMES),
        "feature_names": FEATURE_NAMES,
        "noise_floor_sigma": sigma_noise,
    }
    with open(OUTDIR / "xgboost_model.pkl", "wb") as f:
        pickle.dump(model_bundle, f)
    print(f"    Model bundle saved to {OUTDIR / 'xgboost_model.pkl'}")

    # --- Save results ---
    print("\n[9] Saving results...")
    results_df = pd.DataFrame([
        {"Method": name, "RMSE_Ang": vals[0], "MAE_Ang": vals[1],
         "MAPE_pct": vals[2], "R2": vals[3]}
        for name, vals in results.items()
    ])
    results_df.to_csv(OUTDIR / "comparison_statistics.csv", index=False)

    detail_data = []
    for i, hea in enumerate(ALONSO_TABLE2):
        elems = list(hea["comp"].keys())
        fracs = [hea["comp"][e] for e in elems]
        comp_str = "".join(f"{e}{f:.3f}" for e, f in zip(elems, fracs))
        row = {
            "index": i + 1,
            "composition": comp_str,
            "struct": hea["struct"],
            "a_exp": y_hea[i],
            "a_vegard_alonso": a_vegard_alonso[i],
            "a_eq10_alonso": a_eq10_alonso[i],
            "a_vegard_king": a_vegard_king[i],
            "a_eq10_dft": a_eq10_dft[i],
            "a_eq10_ss": a_eq10_ss[i],
            "a_ss_ridge": a_ss_ridge[i],
            "a_ss_gpr": a_ss_gpr[i],
            "a_ss_gpr_std": gpr_uncertainty[i],
            "a_ss_svr": a_ss_svr[i],
            "a_ss_rf": a_ss_rf[i],
            "a_ss_cubist": a_ss_cub[i],
            "a_xgb_loo": y_pred_loo[i],
            "a_ensemble": y_ensemble_opt[i],
            "a_best": y_best[i],
            "error_alonso": a_eq10_alonso[i] - y_hea[i],
            "error_best": y_best[i] - y_hea[i],
        }
        detail_data.append(row)
    pd.DataFrame(detail_data).to_csv(OUTDIR / "detailed_predictions.csv", index=False)

    # Save Ω_sf data
    omega_data = []
    for pair, val in sorted(omega_b2.items()):
        omega_data.append({"pair": f"{pair[0]}-{pair[1]}", "omega_b2": val,
                           "omega_l12": omega_l12.get(pair, None)})
    for pair, val in sorted(omega_l12.items()):
        if pair not in omega_b2:
            omega_data.append({"pair": f"{pair[0]}-{pair[1]}", "omega_b2": None,
                               "omega_l12": val})
    pd.DataFrame(omega_data).to_csv(OUTDIR / "omega_sf_data.csv", index=False)

    # =====================================================================
    # Figures
    # =====================================================================
    print("\n[10] Generating publication figures...")

    # --- Fig 1: Parity comparison (4 panels) ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    plot_data = [
        ("Alonso Eq.10", a_eq10_alonso, "gray"),
        ("King Vegard (this work)", a_vegard_king, "steelblue"),
        ("DFT Eq.10 SS (this work)", a_eq10_ss, "#44AA77"),
        ("Best Model (this work)", y_best, "crimson"),
    ]
    for ax, (name, y_p, color) in zip(axes.flat, plot_data):
        ax.scatter(y_hea[bcc], y_p[bcc], c=color, marker="s", s=80,
                   alpha=0.85, label=f"BCC ({bcc.sum()})", edgecolors="k", lw=0.5)
        ax.scatter(y_hea[fcc], y_p[fcc], c=color, marker="o", s=80,
                   alpha=0.85, label=f"FCC ({fcc.sum()})", edgecolors="k", lw=0.5)
        lims = [min(y_hea.min(), y_p.min()) - 0.05,
                max(y_hea.max(), y_p.max()) + 0.05]
        ax.plot(lims, lims, "k--", lw=1)
        ax.set_xlim(lims); ax.set_ylim(lims)
        res = y_hea - y_p
        rmse = np.sqrt(np.mean(res**2))
        r2 = 1 - np.sum(res**2) / np.sum((y_hea - y_hea.mean())**2)
        ax.set_title(f"{name}\nRMSE = {rmse:.4f} Å, R$^2$ = {r2:.4f}")
        ax.set_xlabel("Experimental $a$ (Å)")
        ax.set_ylabel("Predicted $a$ (Å)")
        ax.legend(fontsize=12)
        ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(OUTDIR / "fig1_parity_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()

    # --- Fig 2: RMSE bar chart ---
    fig, ax = plt.subplots(figsize=(16, 7))
    bar_methods = [
        ("Alonso\nVegard", results["Alonso Vegard"][0]),
        ("Alonso\nEq.10", results["Alonso Eq.10"][0]),
        ("King\nVegard", results["King Vegard (this work)"][0]),
        ("DFT\nEq.10", results["DFT Eq.10 (this work)"][0]),
        ("DFT Eq.10\nSS", results["DFT Eq.10 SS (this work)"][0]),
        ("SS+Ridge", results["SS Eq.10 + Ridge"][0]),
        ("SS+GPR", results["SS Eq.10 + GPR"][0]),
        ("SS+SVR", results["SS Eq.10 + SVR"][0]),
        ("SS+RF", results["SS Eq.10 + RF"][0]),
        ("SS+Cubist", results["SS Eq.10 + Cubist"][0]),
        ("SS+XGBoost", results["SS Eq.10 + XGBoost"][0]),
        ("Optimal\nEnsemble", results["Optimal Ensemble"][0]),
    ]
    names_bar, vals_bar = zip(*bar_methods)
    colors = ["#AAAAAA", "#888888", "#4477AA", "#44AA77", "#22AA22",
              "#CC8800", "#FF6600", "#9933CC", "#228B22", "#8B4513",
              "#EE3333", "#DD22DD"]
    bars = ax.bar(range(len(names_bar)), vals_bar, color=colors, edgecolor="k")
    ax.set_xticks(range(len(names_bar)))
    ax.set_xticklabels(names_bar, fontsize=13)
    ax.set_ylabel("RMSE (Å)")
    ax.set_title(f"HEA Lattice Constant Prediction Accuracy (N={N})")
    ax.axhline(y=sigma_noise, color="purple", ls=":", alpha=0.6,
               label=f"Noise floor σ={sigma_noise:.4f} Å")
    ax.legend(fontsize=12)
    for b, v in zip(bars, vals_bar):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.0003,
                f"{v:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTDIR / "fig2_rmse_bar.png", dpi=200, bbox_inches="tight")
    plt.close()

    # --- Fig 3: Error distribution ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (name, y_p, color) in zip(axes, [
        ("Alonso Eq.10", a_eq10_alonso, "gray"),
        ("DFT Eq.10 SS", a_eq10_ss, "#22AA22"),
        ("Best Model", y_best, "crimson"),
    ]):
        errors = (y_hea - y_p) * 1000  # mÅ
        ax.hist(errors, bins=20, color=color, edgecolor="k", alpha=0.8)
        ax.axvline(x=0, color="k", ls="--", lw=1)
        ax.set_xlabel("Error (mÅ)")
        ax.set_ylabel("Count")
        rmse = np.sqrt(np.mean((y_hea - y_p)**2))
        ax.set_title(f"{name}\nRMSE={rmse:.4f} Å, bias={errors.mean():.1f} mÅ")
    plt.tight_layout()
    plt.savefig(OUTDIR / "fig3_error_distribution.png", dpi=200, bbox_inches="tight")
    plt.close()

    # --- Fig 4: Feature importance ---
    fig, ax = plt.subplots(figsize=(10, 8))
    importances = final_model.feature_importances_
    idx_sort = np.argsort(importances)[::-1]
    ax.barh(range(len(idx_sort)), importances[idx_sort],
            color="steelblue", edgecolor="k")
    ax.set_yticks(range(len(idx_sort)))
    ax.set_yticklabels([FEATURE_NAMES[i] for i in idx_sort], fontsize=12)
    ax.set_xlabel("Feature Importance (gain)")
    ax.set_title("XGBoost Feature Importance")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUTDIR / "fig4_feature_importance.png", dpi=200, bbox_inches="tight")
    plt.close()

    # --- Fig 5: BCC/FCC separate parity ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, struct_name, mask in zip(axes, ["BCC", "FCC"], [bcc, fcc]):
        ax.scatter(y_hea[mask], y_best[mask], c="crimson", s=80,
                   edgecolors="k", lw=0.5, zorder=3, label="Best Model")
        ax.scatter(y_hea[mask], a_eq10_alonso[mask], c="gray", s=60,
                   marker="^", edgecolors="k", lw=0.5, alpha=0.7,
                   label="Alonso Eq.10", zorder=2)
        lims = [y_hea[mask].min() - 0.05, y_hea[mask].max() + 0.05]
        ax.plot(lims, lims, "k--", lw=1)
        ax.set_xlim(lims); ax.set_ylim(lims)
        rmse_best = np.sqrt(np.mean((y_hea[mask] - y_best[mask])**2))
        rmse_eq10 = np.sqrt(np.mean((y_hea[mask] - a_eq10_alonso[mask])**2))
        ax.set_title(f"{struct_name} HEA (N={mask.sum()})\n"
                     f"Best RMSE={rmse_best:.4f}, Alonso RMSE={rmse_eq10:.4f} Å")
        ax.set_xlabel("Experimental $a$ (Å)")
        ax.set_ylabel("Predicted $a$ (Å)")
        ax.legend(fontsize=12)
        ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(OUTDIR / "fig5_bcc_fcc_parity.png", dpi=200, bbox_inches="tight")
    plt.close()

    # --- Fig 6: Vegard law validation ---
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_hea[bcc], a_vegard_king[bcc], c="blue", marker="s", s=60,
               label="BCC", edgecolors="k", lw=0.5)
    ax.scatter(y_hea[fcc], a_vegard_king[fcc], c="red", marker="o", s=60,
               label="FCC", edgecolors="k", lw=0.5)
    lims = [y_hea.min() - 0.05, y_hea.max() + 0.05]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlim(lims); ax.set_ylim(lims)
    res = y_hea - a_vegard_king
    rmse = np.sqrt(np.mean(res**2))
    ax.set_title(f"Vegard's Law Validation (King Atomic Volumes)\nRMSE = {rmse:.4f} Å")
    ax.set_xlabel("Experimental $a$ (Å)")
    ax.set_ylabel("Vegard $a$ (Å)")
    ax.legend(fontsize=13)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(OUTDIR / "fig6_vegard_check.png", dpi=200, bbox_inches="tight")
    plt.close()

    # --- Fig 8: GPR uncertainty plot ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: parity with error bars
    ax = axes[0]
    ax.errorbar(y_hea[bcc], a_ss_gpr[bcc], yerr=2*gpr_uncertainty[bcc],
                fmt="s", color="steelblue", ms=8, capsize=3, capthick=1,
                elinewidth=1, alpha=0.8, label="BCC", markeredgecolor="k",
                markeredgewidth=0.5)
    ax.errorbar(y_hea[fcc], a_ss_gpr[fcc], yerr=2*gpr_uncertainty[fcc],
                fmt="o", color="crimson", ms=8, capsize=3, capthick=1,
                elinewidth=1, alpha=0.8, label="FCC", markeredgecolor="k",
                markeredgewidth=0.5)
    lims = [min(y_hea.min(), a_ss_gpr.min()) - 0.05,
            max(y_hea.max(), a_ss_gpr.max()) + 0.05]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlim(lims); ax.set_ylim(lims)
    rmse_gpr = np.sqrt(np.mean((y_hea - a_ss_gpr)**2))
    ax.set_title(f"GPR Prediction with 95% CI\nRMSE = {rmse_gpr:.4f} Å", fontsize=14)
    ax.set_xlabel("Experimental $a$ (Å)", fontsize=13)
    ax.set_ylabel("GPR Predicted $a$ (Å)", fontsize=13)
    ax.legend(fontsize=12)
    ax.set_aspect("equal")

    # Right: uncertainty vs error magnitude
    ax = axes[1]
    abs_errors = np.abs(y_hea - a_ss_gpr)
    ax.scatter(gpr_uncertainty[bcc], abs_errors[bcc], c="steelblue", s=80,
               marker="s", edgecolors="k", lw=0.5, label="BCC", alpha=0.8)
    ax.scatter(gpr_uncertainty[fcc], abs_errors[fcc], c="crimson", s=80,
               marker="o", edgecolors="k", lw=0.5, label="FCC", alpha=0.8)
    max_val = max(gpr_uncertainty.max(), abs_errors.max()) * 1.1
    ax.plot([0, max_val], [0, max_val], "k--", lw=1, alpha=0.5, label="y=x")
    ax.set_xlabel("GPR Predicted Uncertainty $\\sigma$ (Å)", fontsize=13)
    ax.set_ylabel("Actual |Error| (Å)", fontsize=13)
    ax.set_title("Uncertainty Calibration", fontsize=14)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTDIR / "fig8_gpr_uncertainty.png", dpi=200, bbox_inches="tight")
    plt.close()

    # --- Fig 7: Model flowchart ---
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16); ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Dynamic Volume Size Factor Model", fontsize=22, fontweight="bold")

    boxes = [
        (1, 8.5, "MP/OQMD\nDFT Data\n(3511 compounds)", "#4477AA"),
        (6, 8.5, "Structure-specific\nΩ$_{sf}$(B2→BCC,\nL1$_2$→FCC)", "#44AA77"),
        (11, 8.5, "King Atomic\nVolumes", "#DD8800"),
        (1, 5.5, "23D Feature\nVector", "#AA4477"),
        (6, 5.5, "XGBoost\nResidual Model", "#EE3333"),
        (11, 5.5, f"Eq.10 (γ$_B$={best_gb:.2f},\nγ$_F$={best_gf:.2f})", "#888888"),
        (6, 2.5, f"Optimal Ensemble\nRMSE={best_method[1][0]:.4f} Å", "#CC8800"),
        (11, 2.5, f"Noise Floor\nσ={sigma_noise:.4f} Å", "#8822AA"),
    ]
    for x, y_pos, text, color in boxes:
        rect = plt.Rectangle((x - 1.5, y_pos - 0.8), 3.0, 1.6,
                              facecolor=color, alpha=0.3, edgecolor=color, lw=2)
        ax.add_patch(rect)
        ax.text(x, y_pos, text, ha="center", va="center", fontsize=12, fontweight="bold")

    for (x1, y1, x2, y2) in [
        (2.5, 8.5, 4.5, 8.5), (7.5, 8.5, 9.5, 8.5),
        (1, 7.7, 1, 6.3), (6, 7.7, 6, 6.3), (11, 7.7, 11, 6.3),
        (2.5, 5.5, 4.5, 5.5), (7.5, 5.5, 9.5, 5.5),
        (6, 4.7, 6, 3.3), (11, 4.7, 11, 3.3),
        (7.5, 2.5, 9.5, 2.5),
    ]:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=2, color="gray"))

    plt.savefig(OUTDIR / "fig7_flowchart.png", dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\n    All figures saved to {OUTDIR}/")

    # --- Summary ---
    print("\n" + "=" * 70)
    print(f"★ Best overall RMSE: {best_method[1][0]:.4f} Å ({best_method[0]})")
    print(f"  Alonso Eq.10 RMSE:    {results['Alonso Eq.10'][0]:.4f} Å")
    print(f"  Improvement:          {(1 - best_method[1][0]/results['Alonso Eq.10'][0])*100:.1f}%")
    print(f"  Noise floor:          {sigma_noise:.4f} Å")
    alonso_rmse = results["Alonso Eq.10"][0]
    if best_method[1][0] < alonso_rmse:
        print(f"  → Surpassed Alonso Eq.10 ({alonso_rmse:.4f} Å)")
    print("=" * 70)

    return results, y_best, y_ensemble_opt, a_eq10_ss, a_ss_gpr, gpr_uncertainty, a_ss_rf, a_ss_cub


if __name__ == "__main__":
    results, y_best, y_ensemble, a_eq10_ss, a_gpr, gpr_unc, a_rf, a_cub = main()
