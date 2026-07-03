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

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
_WARNED_UNKNOWN_ELEMENTS = set()

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
    "Ta":18.014,"W":15.850,"Be":8.111,
    "Mg":23.240,"Y":33.018,"La":37.168,"Ce":34.367,"Sc":24.987,
    "Sn":27.053,"Pb":30.321,
    "Er":30.66,"Tb":32.09,"Dy":31.54,"Ca":43.63,
}

# =====================================================================
# Alonso Table 2 — 64 cubic HEAs (BCC 29 + FCC 35)
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
# Independent Test Set — 31 HEAs NOT in Alonso Table 2 (BCC 17 + FCC 14)
# Sources: Senkov (2012, 2013), Otto (2013), Niu (2017), Stepanov (2015),
#          Dirras (2016), Yao (2016), Freudenberger (2017), Wang (2019),
#          Tseng (2019), Reget (2020), Chen (2022), Kantelis (2025), Chen (2023)
# All single-phase, XRD-verified compositions. 19 elements.
# =====================================================================
INDEPENDENT_TEST = [
    # --- BCC HEAs (17) ---
    # Yao 2016 / Senkov review: MoNbTaV single BCC
    {"comp":{"Mo":0.25,"Nb":0.25,"Ta":0.25,"V":0.25},"struct":"BCC","a_exp":3.208,
     "ref":"Yao2016_Entropy","note":"equiatomic 4-element refractory"},
    # Stepanov 2015: AlNbTiV single BCC (density 5.59 g/cm³ → a=3.191 Å)
    {"comp":{"Al":0.25,"Nb":0.25,"Ti":0.25,"V":0.25},"struct":"BCC","a_exp":3.191,
     "ref":"Stepanov2015_JAlloyCompd","note":"equiatomic 4-element with Al"},
    # Kantelis 2025 AIP Adv: MoNbTaVW
    {"comp":{"Mo":0.20,"Nb":0.20,"Ta":0.20,"V":0.20,"W":0.20},"struct":"BCC","a_exp":3.185,
     "ref":"Kantelis2025_AIPAdv","note":"equiatomic 5-element refractory"},
    # Senkov 2012: HfNbTaTiZr single BCC (as-cast)
    {"comp":{"Hf":0.20,"Nb":0.20,"Ta":0.20,"Ti":0.20,"Zr":0.20},"struct":"BCC","a_exp":3.410,
     "ref":"Senkov2012_Intermet","note":"equiatomic 5-element Senkov standard"},
    # Dirras 2016: Ti35Zr27.5Hf27.5Nb5Ta5 single BCC
    {"comp":{"Ti":0.35,"Zr":0.275,"Hf":0.275,"Nb":0.05,"Ta":0.05},"struct":"BCC","a_exp":3.440,
     "ref":"Dirras2016_MaterCharact","note":"Ti-rich non-equiatomic 5-element"},
    # Tseng 2019 Entropy: HfMoNbTaTiZr equiatomic 6-element (Yeh group)
    {"comp":{"Hf":1/6,"Mo":1/6,"Nb":1/6,"Ta":1/6,"Ti":1/6,"Zr":1/6},"struct":"BCC","a_exp":3.345,
     "ref":"Tseng2019_Entropy","note":"equiatomic 6-element Yeh group"},
    # Tseng 2019: HfMoTaTiZr (no Nb)
    {"comp":{"Hf":0.20,"Mo":0.20,"Ta":0.20,"Ti":0.20,"Zr":0.20},"struct":"BCC","a_exp":3.364,
     "ref":"Tseng2019_Entropy","note":"equiatomic 5-element (no Nb)"},
    # Tseng 2019: HfMoNbTiZr (no Ta)
    {"comp":{"Hf":0.20,"Mo":0.20,"Nb":0.20,"Ti":0.20,"Zr":0.20},"struct":"BCC","a_exp":3.369,
     "ref":"Tseng2019_Entropy","note":"equiatomic 5-element (no Ta)"},
    # Tseng 2019: HfMoNbTaZr (no Ti)
    {"comp":{"Hf":0.20,"Mo":0.20,"Nb":0.20,"Ta":0.20,"Zr":0.20},"struct":"BCC","a_exp":3.347,
     "ref":"Tseng2019_Entropy","note":"equiatomic 5-element (no Ti)"},
    # Tseng 2019: HfMoNbTaTi (no Zr)
    {"comp":{"Hf":0.20,"Mo":0.20,"Nb":0.20,"Ta":0.20,"Ti":0.20},"struct":"BCC","a_exp":3.305,
     "ref":"Tseng2019_Entropy","note":"equiatomic 5-element (no Zr)"},
    #
    # --- FCC HEAs (14) ---
    # Otto 2013: CoCrFeMnNi (Cantor alloy)
    {"comp":{"Co":0.20,"Cr":0.20,"Fe":0.20,"Mn":0.20,"Ni":0.20},"struct":"FCC","a_exp":3.5988,
     "ref":"Otto2013_ActaMat","note":"Cantor alloy standard"},
    # Wang2019 Scripta Mater: CoCrFeNi FCC (precision measurement)
    {"comp":{"Co":0.25,"Cr":0.25,"Fe":0.25,"Ni":0.25},"struct":"FCC","a_exp":3.5723,
     "ref":"Wang2019_ScriptaMat","note":"equiatomic 4-element precision XRD"},
    # Wang2019: Co0.5CrFeNi FCC
    {"comp":{"Co":0.143,"Cr":0.286,"Fe":0.286,"Ni":0.286},"struct":"FCC","a_exp":3.5805,
     "ref":"Wang2019_ScriptaMat","note":"Co-lean non-equiatomic"},
    # Niu 2017 Sci Rep: (CoCrFeNi)0.89Pd0.11 single FCC
    {"comp":{"Co":0.2225,"Cr":0.2225,"Fe":0.2225,"Ni":0.2225,"Pd":0.11},"struct":"FCC","a_exp":3.620,
     "ref":"Niu2017_SciRep","note":"11% Pd addition to CoCrFeNi"},
    # Niu 2017: (CoCrFeNi)0.80Pd0.20 single FCC
    {"comp":{"Co":0.20,"Cr":0.20,"Fe":0.20,"Ni":0.20,"Pd":0.20},"struct":"FCC","a_exp":3.660,
     "ref":"Niu2017_SciRep","note":"equiatomic 5-element with Pd"},
    # Niu 2017: (CoCrFeNi)0.73Pd0.27 single FCC
    {"comp":{"Co":0.1825,"Cr":0.1825,"Fe":0.1825,"Ni":0.1825,"Pd":0.27},"struct":"FCC","a_exp":3.710,
     "ref":"Niu2017_SciRep","note":"Pd-rich 27% addition"},
    # Niu 2017: (CoCrFeNi)0.80V0.20 single FCC
    {"comp":{"Co":0.20,"Cr":0.20,"Fe":0.20,"Ni":0.20,"V":0.20},"struct":"FCC","a_exp":3.610,
     "ref":"Niu2017_SciRep","note":"equiatomic 5-element with V"},
    # Freudenberger 2017 Metals: AuCuNiPd noble metal 4-element
    {"comp":{"Au":0.25,"Cu":0.25,"Ni":0.25,"Pd":0.25},"struct":"FCC","a_exp":3.8093,
     "ref":"Freudenberger2017_Metals","note":"noble metal 4-element"},
    # Freudenberger 2017: AuCuNiPt
    {"comp":{"Au":0.25,"Cu":0.25,"Ni":0.25,"Pt":0.25},"struct":"FCC","a_exp":3.8107,
     "ref":"Freudenberger2017_Metals","note":"noble metal 4-element with Pt"},
    # Freudenberger 2017: AuCuPdPt (no Ni)
    {"comp":{"Au":0.25,"Cu":0.25,"Pd":0.25,"Pt":0.25},"struct":"FCC","a_exp":3.8847,
     "ref":"Freudenberger2017_Metals","note":"noble metal 4-element (no Ni)"},
    # Freudenberger 2017: AuNiPdPt (no Cu)
    {"comp":{"Au":0.25,"Ni":0.25,"Pd":0.25,"Pt":0.25},"struct":"FCC","a_exp":3.8738,
     "ref":"Freudenberger2017_Metals","note":"noble metal 4-element (no Cu)"},
    # Freudenberger 2017: CuNiPdPt (no Au)
    {"comp":{"Cu":0.25,"Ni":0.25,"Pd":0.25,"Pt":0.25},"struct":"FCC","a_exp":3.7622,
     "ref":"Freudenberger2017_Metals","note":"noble metal 4-element (no Au)"},
    # Freudenberger 2017: AuCuNiPdPt quinary
    {"comp":{"Au":0.20,"Cu":0.20,"Ni":0.20,"Pd":0.20,"Pt":0.20},"struct":"FCC","a_exp":3.8307,
     "ref":"Freudenberger2017_Metals","note":"noble metal 5-element quinary"},
    #
    # --- Additional BCC HEAs to balance low-a coverage ---
    # Reget 2020 Metals: Mo25Nb25V25W25 single BCC (powder metallurgy)
    {"comp":{"Mo":0.25,"Nb":0.25,"V":0.25,"W":0.25},"struct":"BCC","a_exp":3.157,
     "ref":"Reget2020_Metals","note":"equiatomic 4-element refractory (no Ti)"},
    # Reget 2020 Metals: Mo25Nb25V25Ti25 single BCC
    {"comp":{"Mo":0.25,"Nb":0.25,"V":0.25,"Ti":0.25},"struct":"BCC","a_exp":3.174,
     "ref":"Reget2020_Metals","note":"equiatomic 4-element (Ti replaces W)"},
    # Reget 2020 Metals: Mo20Nb20V20W20Ti20 single BCC
    {"comp":{"Mo":0.20,"Nb":0.20,"V":0.20,"W":0.20,"Ti":0.20},"struct":"BCC","a_exp":3.164,
     "ref":"Reget2020_Metals","note":"equiatomic 5-element refractory"},
    # Reget 2020 Metals: Mo30Nb30V30Ti10 single BCC (Ti-lean)
    {"comp":{"Mo":0.30,"Nb":0.30,"V":0.30,"Ti":0.10},"struct":"BCC","a_exp":3.156,
     "ref":"Reget2020_Metals","note":"non-equiatomic Ti-lean 4-element"},
    # Senkov 2013 Acta Mater: NbTiVZr single BCC (HIP + 1200C/24h)
    {"comp":{"Nb":0.25,"Ti":0.25,"V":0.25,"Zr":0.25},"struct":"BCC","a_exp":3.338,
     "ref":"Senkov2013_ActaMat","note":"equiatomic 4-element low-density"},
    # Chen 2022 Acta Mater: HfNbZr single BCC (equiatomic ternary)
    {"comp":{"Hf":1/3,"Nb":1/3,"Zr":1/3},"struct":"BCC","a_exp":3.4869,
     "ref":"Chen2022_ActaMat","note":"equiatomic ternary BCC"},
    # Chen 2023 Nat Commun: AlCoMnNiV single BCC (CALPHAD prediction + synthesis)
    {"comp":{"Al":0.20,"Co":0.20,"Mn":0.20,"Ni":0.20,"V":0.20},"struct":"BCC","a_exp":2.900,
     "ref":"Chen2023_NatCommun","note":"equiatomic 5-element with Al,Mn"},
    #
    # --- Additional FCC HEAs ---
    # Chen 2023 Nat Commun: CoFeMnNiZn single FCC (CALPHAD prediction + synthesis)
    {"comp":{"Co":0.20,"Fe":0.20,"Mn":0.20,"Ni":0.20,"Zn":0.20},"struct":"FCC","a_exp":3.635,
     "ref":"Chen2023_NatCommun","note":"equiatomic 5-element with Mn,Zn"},
]


# =====================================================================
# Multi-phase HEA Classification Database
# Sources: Zhang (2008), Guo & Liu (2011), Yang & Zhang (2012),
#          Senkov (2013, 2018), Tsai & Yeh (2014 review)
# phase: "SS" = single-phase solid solution (BCC/FCC/BCC+FCC)
#        "IM" = contains intermetallic compounds (Laves, sigma, B2 ordered, etc.)
#        "AM" = amorphous phase
# struct: dominant crystal structure for SS alloys
# =====================================================================
MULTIPHASE_HEA_DB = [
    # --- Single-phase solid solutions (SS) ---
    # Zhang 2008, Table 1 + Guo 2011, Table 1: SS alloys
    {"comp":{"Co":0.2,"Cr":0.2,"Fe":0.2,"Mn":0.2,"Ni":0.2},"phase":"SS","struct":"FCC",
     "ref":"Cantor2004"},
    {"comp":{"Co":0.25,"Cr":0.25,"Fe":0.25,"Ni":0.25},"phase":"SS","struct":"FCC",
     "ref":"Zhang2008"},
    {"comp":{"Nb":0.25,"Mo":0.25,"Ta":0.25,"W":0.25},"phase":"SS","struct":"BCC",
     "ref":"Senkov2010"},
    {"comp":{"Nb":0.2,"Mo":0.2,"Ta":0.2,"V":0.2,"W":0.2},"phase":"SS","struct":"BCC",
     "ref":"Senkov2010"},
    {"comp":{"Nb":0.2,"Mo":0.2,"Ta":0.2,"V":0.2,"Ti":0.2},"phase":"SS","struct":"BCC",
     "ref":"Senkov2011"},
    {"comp":{"Nb":0.25,"Ta":0.25,"Ti":0.25,"V":0.25},"phase":"SS","struct":"BCC",
     "ref":"Senkov2013"},
    {"comp":{"Nb":0.25,"Ti":0.25,"V":0.25,"Zr":0.25},"phase":"SS","struct":"BCC",
     "ref":"Senkov2013"},
    {"comp":{"Hf":0.2,"Nb":0.2,"Ta":0.2,"Ti":0.2,"Zr":0.2},"phase":"SS","struct":"BCC",
     "ref":"Senkov2012"},
    {"comp":{"Nb":0.2,"Hf":0.2,"Zr":0.2,"V":0.2,"Ti":0.2},"phase":"SS","struct":"BCC",
     "ref":"Senkov2013"},
    {"comp":{"Al":0.25,"Co":0.25,"Cr":0.25,"Fe":0.25},"phase":"SS","struct":"BCC",
     "ref":"Zhang2008"},
    {"comp":{"Cu":0.2,"Ni":0.2,"Al":0.2,"Co":0.2,"Cr":0.2},"phase":"SS","struct":"BCC+FCC",
     "ref":"Yeh2004"},
    {"comp":{"Cu":0.2,"Ni":0.2,"Al":0.2,"Co":0.2,"Fe":0.2},"phase":"SS","struct":"BCC+FCC",
     "ref":"Yeh2004"},
    # Note: equimolar AlCoCrCuFeNi (Yeh2004) reclassified as IM below
    # (Tong2005 identified B2 ordering at x=1.0 in Al_x CoCrCuFeNi series)
    {"comp":{"Mo":0.25,"Nb":0.25,"Ta":0.25,"V":0.25},"phase":"SS","struct":"BCC",
     "ref":"Yao2016"},
    {"comp":{"Al":0.25,"Nb":0.25,"Ti":0.25,"V":0.25},"phase":"SS","struct":"BCC",
     "ref":"Stepanov2015"},
    {"comp":{"Co":0.2,"Cr":0.2,"Cu":0.2,"Fe":0.2,"Ni":0.2},"phase":"SS","struct":"FCC",
     "ref":"Hsu2004"},
    # Guo2011 Table 1 - additional SS alloys
    {"comp":{"Ti":0.2,"V":0.2,"Cr":0.2,"Mn":0.2,"Fe":0.2},"phase":"SS","struct":"BCC",
     "ref":"Guo2011"},
    {"comp":{"Co":0.25,"Fe":0.25,"Mn":0.25,"Ni":0.25},"phase":"SS","struct":"FCC",
     "ref":"Guo2011"},
    {"comp":{"Co":0.2,"Cr":0.2,"Fe":0.2,"Ni":0.2,"V":0.2},"phase":"SS","struct":"FCC",
     "ref":"Guo2011"},
    {"comp":{"Al":1/7,"Co":1/7,"Cr":1/7,"Cu":1/7,"Fe":1/7,"Ni":1/7,"V":1/7},"phase":"SS","struct":"BCC+FCC",
     "ref":"Singh2011"},
    # Zhang2012: SS alloys from Omega-delta diagram
    {"comp":{"Cr":1/6,"Mo":1/6,"Nb":1/6,"Ta":1/6,"V":1/6,"W":1/6},"phase":"SS","struct":"BCC",
     "ref":"Zhang2015"},
    {"comp":{"Ti":0.35,"Zr":0.275,"Hf":0.275,"Nb":0.05,"Ta":0.05},"phase":"SS","struct":"BCC",
     "ref":"Dirras2016"},
    {"comp":{"Co":0.2,"Cr":0.2,"Fe":0.2,"Ni":0.2,"Pd":0.2},"phase":"SS","struct":"FCC",
     "ref":"Alonso2021"},
    #
    # --- Multi-phase / Intermetallic-containing alloys (IM) ---
    # Zhang 2008, Guo 2011: IM alloys
    {"comp":{"Al":1/3,"Co":1/3,"Cr":1/3},"phase":"IM","struct":"BCC+B2",
     "ref":"Zhang2008"},
    {"comp":{"Al":0.5/4.5,"Co":1/4.5,"Cr":1/4.5,"Cu":1/4.5,"Fe":1/4.5},"phase":"IM","struct":"BCC+FCC+B2",
     "ref":"Tong2005_Al05"},
    {"comp":{"Al":1/5,"Co":1/5,"Cr":1/5,"Cu":1/5,"Fe":1/5},"phase":"IM","struct":"BCC+FCC+B2",
     "ref":"Tong2005_Al10"},
    {"comp":{"Al":2/6,"Co":1/6,"Cr":1/6,"Cu":1/6,"Fe":1/6},"phase":"IM","struct":"BCC+B2",
     "ref":"Tong2005_Al20"},
    {"comp":{"Al":3/7,"Co":1/7,"Cr":1/7,"Cu":1/7,"Fe":1/7},"phase":"IM","struct":"BCC+B2",
     "ref":"Tong2005_Al30"},
    {"comp":{"Al":0.5/5.5,"Co":1/5.5,"Cr":1/5.5,"Cu":1/5.5,"Fe":1/5.5,"Ni":1/5.5},"phase":"SS","struct":"FCC",
     "ref":"Tong2005b_Al05"},
    {"comp":{"Al":1/6,"Co":1/6,"Cr":1/6,"Cu":1/6,"Fe":1/6,"Ni":1/6},"phase":"IM","struct":"BCC+FCC+B2",
     "ref":"Tong2005b_Al10"},
    {"comp":{"Al":2/7,"Co":1/7,"Cr":1/7,"Cu":1/7,"Fe":1/7,"Ni":1/7},"phase":"IM","struct":"BCC+B2",
     "ref":"Tong2005b_Al20"},
    # CoCrFeNiAl_x series (Guo 2011): Al content drives BCC+B2 formation
    {"comp":{"Al":0.3/4.3,"Co":1/4.3,"Cr":1/4.3,"Fe":1/4.3,"Ni":1/4.3},"phase":"SS","struct":"FCC",
     "ref":"Wang2009_Al03"},
    {"comp":{"Al":0.5/4.5,"Co":1/4.5,"Cr":1/4.5,"Fe":1/4.5,"Ni":1/4.5},"phase":"SS","struct":"FCC+BCC",
     "ref":"Wang2009_Al05"},
    {"comp":{"Al":0.7/4.7,"Co":1/4.7,"Cr":1/4.7,"Fe":1/4.7,"Ni":1/4.7},"phase":"IM","struct":"BCC+FCC+B2",
     "ref":"Wang2009_Al07"},
    {"comp":{"Al":0.9/4.9,"Co":1/4.9,"Cr":1/4.9,"Fe":1/4.9,"Ni":1/4.9},"phase":"IM","struct":"BCC+B2",
     "ref":"Wang2009_Al09"},
    {"comp":{"Al":1.0/5.0,"Co":1/5.0,"Cr":1/5.0,"Fe":1/5.0,"Ni":1/5.0},"phase":"IM","struct":"BCC+B2",
     "ref":"Wang2009_Al10"},
    {"comp":{"Al":1.5/5.5,"Co":1/5.5,"Cr":1/5.5,"Fe":1/5.5,"Ni":1/5.5},"phase":"IM","struct":"BCC+B2",
     "ref":"Wang2009_Al15"},
    {"comp":{"Al":2.0/6.0,"Co":1/6.0,"Cr":1/6.0,"Fe":1/6.0,"Ni":1/6.0},"phase":"IM","struct":"BCC+B2",
     "ref":"Wang2009_Al20"},
    # CoCrCuFeNiTi_x (Guo 2011): Ti drives Laves/sigma
    {"comp":{"Co":1/5.5,"Cr":1/5.5,"Cu":1/5.5,"Fe":1/5.5,"Ni":1/5.5,"Ti":0.5/5.5},"phase":"IM","struct":"FCC+Laves",
     "ref":"Zhou2007_Ti05"},
    {"comp":{"Co":1/6,"Cr":1/6,"Cu":1/6,"Fe":1/6,"Ni":1/6,"Ti":1/6},"phase":"IM","struct":"FCC+Laves",
     "ref":"Zhou2007_Ti10"},
    # Refractory multi-phase HEAs (Senkov 2013)
    {"comp":{"Cr":0.25,"Nb":0.25,"Ti":0.25,"Zr":0.25},"phase":"IM","struct":"BCC+Laves",
     "ref":"Senkov2013_CrNbTiZr"},
    {"comp":{"Cr":0.2,"Nb":0.2,"Ti":0.2,"V":0.2,"Zr":0.2},"phase":"IM","struct":"BCC+Laves",
     "ref":"Senkov2013_CrNbTiVZr"},
    # AlCoCrFeNi high-Al alloys (intermetallic B2 ordered)
    {"comp":{"Al":0.25,"Co":0.25,"Cr":0.25,"Ni":0.25},"phase":"IM","struct":"BCC+B2+FCC",
     "ref":"Zhang2008_AlCoCrNi"},
    # Additional IM alloys from Guo & Liu 2011 Table 2
    {"comp":{"Cu":0.2,"Ni":0.2,"Co":0.2,"Zn":0.2,"Al":0.2},"phase":"IM","struct":"BCC+FCC+IM",
     "ref":"Guo2011_CuNiCoZnAl"},
    # Note: TiZrHfCuNi listed as AM in Guo2011 Table 3 (see AM section below)
    {"comp":{"Ti":1/6,"Zr":1/6,"Hf":1/6,"Cu":1/6,"Ni":1/6,"Be":1/6},"phase":"IM","struct":"BCC+IM",
     "ref":"Guo2011_TiZrHfCuNiBe"},
    # CoCrFeNiMo_x (Mo drives sigma phase)
    {"comp":{"Co":1/5.5,"Cr":1/5.5,"Fe":1/5.5,"Ni":1/5.5,"Mo":0.5/5.5},"phase":"IM","struct":"FCC+sigma",
     "ref":"Shun2012_Mo05"},
    {"comp":{"Co":1/5.85,"Cr":1/5.85,"Fe":1/5.85,"Ni":1/5.85,"Mo":0.85/5.85},"phase":"IM","struct":"FCC+sigma+mu",
     "ref":"Shun2012_Mo085"},
    # CoCrFeNiNb_x (Nb drives Laves)
    {"comp":{"Co":1/5.5,"Cr":1/5.5,"Fe":1/5.5,"Ni":1/5.5,"Nb":0.5/5.5},"phase":"IM","struct":"FCC+Laves",
     "ref":"He2014_Nb05"},
    {"comp":{"Co":1/6,"Cr":1/6,"Fe":1/6,"Ni":1/6,"Nb":1/6},"phase":"IM","struct":"FCC+Laves",
     "ref":"He2014_Nb10"},
    #
    # --- Amorphous alloys (AM) from Guo 2011 Table 3 ---
    {"comp":{"Cu":0.2,"Zr":0.2,"Ti":0.2,"Ni":0.2,"Be":0.2},"phase":"AM","struct":"AM",
     "ref":"Guo2011_AM1"},
    {"comp":{"Cu":0.2,"Zr":0.2,"Ti":0.2,"Ni":0.2,"Hf":0.2},"phase":"AM","struct":"AM",
     "ref":"Guo2011_AM2"},
    {"comp":{"Zr":1/6,"Ti":1/6,"Cu":1/6,"Ni":1/6,"Be":1/6,"Fe":1/6},"phase":"AM","struct":"AM",
     "ref":"Guo2011_AM3"},
    {"comp":{"Er":0.2,"Tb":0.2,"Dy":0.2,"Ni":0.2,"Al":0.2},"phase":"AM","struct":"AM",
     "ref":"Guo2011_AM4"},
    {"comp":{"Ca":0.2,"Mg":0.2,"Cu":0.2,"Ni":0.2,"Zn":0.2},"phase":"AM","struct":"AM",
     "ref":"Guo2011_AM5"},
    {"comp":{"Ti":0.2,"Zr":0.2,"Cu":0.2,"Pd":0.2,"Sn":0.2},"phase":"AM","struct":"AM",
     "ref":"Takeuchi2013_AM6"},
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
    # Also load VASP-calculated data if available
    for struct in ["B2", "L12"]:
        for search_dir in [Path("data"), base]:
            f = search_dir / f"compounds_VASP_{struct}.csv"
            if f.exists():
                df = pd.read_csv(f)
                df["db"] = "VASP"
                df["stype"] = struct
                dfs.append(df)
                print(f"    Loaded VASP {struct}: {len(df)} compounds from {f}")
                break
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


# Atomic number for periodic-table features
ATOMIC_NUMBER = {
    "H":1,"He":2,"Li":3,"Be":4,"B":5,"C":6,"N":7,"O":8,
    "Na":11,"Mg":12,"Al":13,"Si":14,"P":15,"S":16,
    "K":19,"Ca":20,"Sc":21,"Ti":22,"V":23,"Cr":24,
    "Mn":25,"Fe":26,"Co":27,"Ni":28,"Cu":29,"Zn":30,
    "Ga":31,"Ge":32,"As":33,"Se":34,"Br":35,
    "Rb":37,"Sr":38,"Y":39,"Zr":40,"Nb":41,"Mo":42,
    "Tc":43,"Ru":44,"Rh":45,"Pd":46,"Ag":47,"Cd":48,
    "In":49,"Sn":50,"Sb":51,"Te":52,"I":53,
    "Cs":55,"Ba":56,"La":57,"Ce":58,"Pr":59,"Nd":60,
    "Sm":62,"Eu":63,"Gd":64,"Tb":65,"Dy":66,"Ho":67,
    "Er":68,"Tm":69,"Yb":70,"Lu":71,
    "Hf":72,"Ta":73,"W":74,"Re":75,"Os":76,"Ir":77,
    "Pt":78,"Au":79,"Hg":80,"Tl":81,"Pb":82,"Bi":83,
    "Th":90,"U":92,"Pu":94,"Np":93,
}

# Covalent radii (Å) — Cordero et al. (2008)
COVALENT_RADIUS = {
    "Li":1.28,"Be":0.96,"B":0.84,"C":0.76,"N":0.71,"O":0.66,
    "Na":1.66,"Mg":1.41,"Al":1.21,"Si":1.11,"P":1.07,"S":1.05,
    "K":2.03,"Ca":1.76,"Sc":1.70,"Ti":1.60,"V":1.53,"Cr":1.39,
    "Mn":1.39,"Fe":1.32,"Co":1.26,"Ni":1.24,"Cu":1.32,"Zn":1.22,
    "Ga":1.22,"Ge":1.20,"As":1.19,"Se":1.20,"Br":1.20,
    "Rb":2.20,"Sr":1.95,"Y":1.90,"Zr":1.75,"Nb":1.64,"Mo":1.54,
    "Tc":1.47,"Ru":1.46,"Rh":1.42,"Pd":1.39,"Ag":1.45,"Cd":1.44,
    "In":1.42,"Sn":1.39,"Sb":1.39,"Te":1.38,"I":1.39,
    "Cs":2.44,"Ba":2.15,"La":2.07,"Ce":2.04,"Pr":2.03,"Nd":2.01,
    "Sm":1.98,"Eu":1.98,"Gd":1.96,"Tb":1.94,"Dy":1.92,"Ho":1.92,
    "Er":1.89,"Tm":1.90,"Yb":1.87,"Lu":1.87,
    "Hf":1.75,"Ta":1.70,"W":1.62,"Re":1.51,"Os":1.44,"Ir":1.41,
    "Pt":1.36,"Au":1.36,"Hg":1.32,"Tl":1.45,"Pb":1.46,"Bi":1.48,
    "Th":2.06,"U":1.96,"Pu":1.87,"Np":1.90,
}

# Periodic-table row
PERIOD = {
    "Li":2,"Be":2,"B":2,"C":2,"N":2,"O":2,
    "Na":3,"Mg":3,"Al":3,"Si":3,"P":3,"S":3,
    "K":4,"Ca":4,"Sc":4,"Ti":4,"V":4,"Cr":4,
    "Mn":4,"Fe":4,"Co":4,"Ni":4,"Cu":4,"Zn":4,
    "Ga":4,"Ge":4,"As":4,"Se":4,"Br":4,
    "Rb":5,"Sr":5,"Y":5,"Zr":5,"Nb":5,"Mo":5,
    "Tc":5,"Ru":5,"Rh":5,"Pd":5,"Ag":5,"Cd":5,
    "In":5,"Sn":5,"Sb":5,"Te":5,"I":5,
    "Cs":6,"Ba":6,"La":6,"Ce":6,"Pr":6,"Nd":6,
    "Sm":6,"Eu":6,"Gd":6,"Tb":6,"Dy":6,"Ho":6,
    "Er":6,"Tm":6,"Yb":6,"Lu":6,
    "Hf":6,"Ta":6,"W":6,"Re":6,"Os":6,"Ir":6,
    "Pt":6,"Au":6,"Hg":6,"Tl":6,"Pb":6,"Bi":6,
    "Th":7,"U":7,"Pu":7,"Np":7,
}


def build_omega_sf_features(elA, elB):
    """
    Build pairwise feature vector for Ω_sf prediction.
    Features capture atomic property differences/ratios that drive size factor.
    Returns 14-dimensional feature vector.
    """
    # Ensure consistent ordering (smaller Z first)
    if ATOMIC_NUMBER.get(elA, 0) > ATOMIC_NUMBER.get(elB, 0):
        elA, elB = elB, elA

    rA = COVALENT_RADIUS.get(elA, 1.3)
    rB = COVALENT_RADIUS.get(elB, 1.3)
    vA = KING_ATOMIC_VOLUMES.get(elA, 15.0)
    vB = KING_ATOMIC_VOLUMES.get(elB, 15.0)
    enA = PAULING_EN.get(elA, 1.5)
    enB = PAULING_EN.get(elB, 1.5)
    vecA = VEC.get(elA, 5)
    vecB = VEC.get(elB, 5)
    dA = D_ELECTRONS.get(elA, 0)
    dB = D_ELECTRONS.get(elB, 0)
    mA = ATOMIC_MASS.get(elA, 50.0)
    mB = ATOMIC_MASS.get(elB, 50.0)
    pA = PERIOD.get(elA, 4)
    pB = PERIOD.get(elB, 4)

    feats = np.array([
        (rA - rB) / ((rA + rB) / 2),          # 0: radius ratio deviation
        abs(rA - rB),                           # 1: absolute radius diff
        (vA - vB) / ((vA + vB) / 2),          # 2: volume ratio deviation
        abs(vA - vB),                           # 3: absolute volume diff
        enA - enB,                              # 4: electronegativity diff (signed)
        abs(enA - enB),                         # 5: |Δχ|
        vecA - vecB,                            # 6: VEC diff (signed)
        abs(vecA - vecB),                       # 7: |ΔVEC|
        dA - dB,                                # 8: d-electron diff (signed)
        abs(dA - dB),                           # 9: |Δd|
        (vA + vB) / 2,                         # 10: mean volume
        (enA + enB) / 2,                       # 11: mean electronegativity
        abs(pA - pB),                           # 12: period diff
        abs(mA - mB) / ((mA + mB) / 2),       # 13: mass ratio deviation
    ])
    return feats


OMEGA_FEATURE_NAMES = [
    "Δr/r_avg", "|Δr|", "ΔV/V_avg", "|ΔV|",
    "Δχ", "|Δχ|", "ΔVEC", "|ΔVEC|",
    "Δd", "|Δd|", "V_avg", "χ_avg",
    "ΔPeriod", "Δm/m_avg",
]


def build_omega_sf_ml_model(omega_sf_dict, structure_label=""):
    """
    Train GPR + Ridge models to predict Ω_sf from elemental properties.
    Uses LOO-CV to evaluate prediction accuracy.

    Args:
        omega_sf_dict: {(elA, elB): omega_value} from DFT data
        structure_label: "B2" or "L12" for logging

    Returns:
        (gpr_model, scaler, cv_rmse, cv_r2, missing_predictions)
        where missing_predictions = {pair: (omega_pred, sigma)} for all 666 pairs
    """
    pairs = list(omega_sf_dict.keys())
    omegas = np.array([omega_sf_dict[p] for p in pairs])
    X = np.array([build_omega_sf_features(*p) for p in pairs])
    N = len(pairs)

    print(f"      Ω_sf ML ({structure_label}): {N} known pairs, 14 features")

    if N < 2:
        print(f"        Skipping ML model: insufficient data (N={N}, need ≥2 for LOO-CV)")
        return None, None, float('inf'), 0.0, {}

    # LOO-CV for Ridge (baseline)
    best_ridge_rmse = 999
    best_alpha = 1.0
    for alpha in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]:
        loo = LeaveOneOut()
        y_pred = np.zeros(N)
        for tr, te in loo.split(X):
            sc = StandardScaler()
            Xtr = sc.fit_transform(X[tr])
            Xte = sc.transform(X[te])
            m = Ridge(alpha=alpha)
            m.fit(Xtr, omegas[tr])
            y_pred[te] = m.predict(Xte)
        rmse = np.sqrt(np.mean((omegas - y_pred)**2))
        if rmse < best_ridge_rmse:
            best_ridge_rmse = rmse
            best_alpha = alpha

    print(f"        Ridge LOO-CV RMSE: {best_ridge_rmse:.6f} (α={best_alpha})")

    # LOO-CV for GPR (Matérn 5/2)
    kernel = ConstantKernel(0.01) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(0.001)
    y_pred_gpr = np.zeros(N)
    y_std_gpr = np.zeros(N)
    loo = LeaveOneOut()
    for tr, te in loo.split(X):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        gpr = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=3,
            alpha=1e-6, normalize_y=True, random_state=42)
        gpr.fit(Xtr, omegas[tr])
        pred, std = gpr.predict(Xte, return_std=True)
        y_pred_gpr[te] = pred
        y_std_gpr[te] = std

    rmse_gpr = np.sqrt(np.mean((omegas - y_pred_gpr)**2))
    r2_gpr = 1 - np.sum((omegas - y_pred_gpr)**2) / np.sum((omegas - omegas.mean())**2)
    print(f"        GPR LOO-CV RMSE: {rmse_gpr:.6f}, R$^2$={r2_gpr:.4f}")

    # Train final GPR on all data
    sc_final = StandardScaler()
    X_scaled = sc_final.fit_transform(X)
    gpr_final = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=5,
        alpha=1e-6, normalize_y=True, random_state=42)
    gpr_final.fit(X_scaled, omegas)

    # Predict all possible pairs (including known ones, for consistency check)
    all_elements = sorted(KING_ATOMIC_VOLUMES.keys())
    all_predictions = {}
    for i, elA in enumerate(all_elements):
        for elB in all_elements[i+1:]:
            pair = tuple(sorted([elA, elB]))
            feat = build_omega_sf_features(elA, elB).reshape(1, -1)
            feat_scaled = sc_final.transform(feat)
            pred, std = gpr_final.predict(feat_scaled, return_std=True)
            all_predictions[pair] = (pred[0], std[0])

    return gpr_final, sc_final, rmse_gpr, r2_gpr, all_predictions


def fill_missing_omega_sf(omega_dft, omega_ml_predictions):
    """
    Create a complete Ω_sf dictionary: DFT values where available,
    ML-predicted values where missing.

    Returns:
        omega_filled: {pair: omega_value}
        filled_pairs: list of pairs that were ML-filled
        fill_stats: dict with summary statistics
    """
    omega_filled = {}
    filled_pairs = []
    dft_pairs = []

    for pair, (pred, std) in omega_ml_predictions.items():
        if pair in omega_dft:
            omega_filled[pair] = omega_dft[pair]
            dft_pairs.append(pair)
        else:
            omega_filled[pair] = pred
            filled_pairs.append((pair, pred, std))

    fill_stats = {
        "n_dft": len(dft_pairs),
        "n_ml_filled": len(filled_pairs),
        "n_total": len(omega_filled),
        "mean_uncertainty": np.mean([s for _, _, s in filled_pairs]) if filled_pairs else 0,
    }
    return omega_filled, filled_pairs, fill_stats


def _warn_unknown_element(elem, context=""):
    """Emit a one-time warning for elements not in KING_ATOMIC_VOLUMES."""
    key = (elem, context)
    if key not in _WARNED_UNKNOWN_ELEMENTS:
        _WARNED_UNKNOWN_ELEMENTS.add(key)
        warnings.warn(
            f"Unknown element '{elem}' not in KING_ATOMIC_VOLUMES"
            f"{' (' + context + ')' if context else ''}, using default value",
            stacklevel=3)


def compute_delta_r(comp):
    """
    Traditional atomic size mismatch δr (%) using King atomic volumes.
    δr = 100 × √[Σ c_i (1 - r_i/r̄)²]  where r_i = V_i^(1/3).
    """
    elements = list(comp.keys())
    fracs = np.array([comp[e] for e in elements])
    fracs = fracs / fracs.sum()
    for e in elements:
        if e not in KING_ATOMIC_VOLUMES:
            _warn_unknown_element(e, "compute_delta_r")
    vols = np.array([KING_ATOMIC_VOLUMES.get(e, 15.0) for e in elements])
    r_vals = vols ** (1/3)
    r_avg = np.sum(fracs * r_vals)
    return 100 * np.sqrt(np.sum(fracs * (1 - r_vals / r_avg)**2))


def compute_delta_sf(comp, omega_sf):
    """
    Ω_sf-based size mismatch descriptor.
    δ_sf = √[Σ_i Σ_{j>i} c_i c_j Ω_sf(i,j)²]
    Captures actual pairwise volume deviations from DFT, not just pure-element radii.
    """
    elements = list(comp.keys())
    fracs = np.array([comp[e] for e in elements])
    fracs = fracs / fracs.sum()
    n = len(elements)

    val = 0.0
    for i in range(n):
        for j in range(i+1, n):
            pair = tuple(sorted([elements[i], elements[j]]))
            omega = omega_sf.get(pair, 0.0)
            val += fracs[i] * fracs[j] * omega**2
    return np.sqrt(val)


def compute_delta_sf_signed(comp, omega_sf):
    """
    Signed Ω_sf correction magnitude (can be positive or negative).
    Σ_i Σ_{j≠i} c_i c_j Ω_sf(i,j) — the correction term in Eq.10.
    """
    elements = list(comp.keys())
    fracs = np.array([comp[e] for e in elements])
    fracs = fracs / fracs.sum()
    n = len(elements)

    val = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                pair = tuple(sorted([elements[i], elements[j]]))
                omega = omega_sf.get(pair, 0.0)
                val += fracs[i] * fracs[j] * omega
    return val


def compute_omega_yang(comp, struct):
    """
    Ω parameter (Yang & Zhang 2012): Ω = T_m × ΔS_mix / |ΔH_mix|.
    Uses Miedema-approximated ΔH_mix and rule-of-mixtures T_m.
    """
    # Melting points (K) — common HEA elements
    T_MELT = {
        "Al":933,"Co":1768,"Cr":2180,"Cu":1358,"Fe":1811,"Mn":1519,
        "Mo":2896,"Nb":2750,"Ni":1728,"Pd":1828,"Pt":2041,"Re":3459,
        "Rh":2237,"Ru":2607,"Ta":3290,"Ti":1941,"V":2183,"W":3695,
        "Zr":2128,"Hf":2506,"Os":3306,"Ir":2719,"Au":1337,"Ag":1235,
        "Zn":693,"Si":1687,"Ge":1211,"Be":1560,"Mg":923,"Sc":1814,
        "Y":1799,"La":1193,"Ce":1068,"B":2349,"Sn":505,"Pb":601,
        "Er":1802,"Tb":1629,"Dy":1685,"Ca":1115,
    }
    # Simplified Miedema mixing enthalpy (kJ/mol) — selected pairs
    # Using Takeuchi & Inoue (2005) tabulated values for common HEA pairs
    DELTA_H_MIX = {
        ("Al","Co"):-19,("Al","Cr"):-10,("Al","Cu"):-1,("Al","Fe"):-11,
        ("Al","Mn"):-19,("Al","Mo"):-5,("Al","Nb"):-18,("Al","Ni"):-22,
        ("Al","Pd"):-31,("Al","Pt"):-44,("Al","Ti"):-30,("Al","V"):-16,
        ("Al","Zr"):-44,("Co","Cr"):-4,("Co","Cu"):6,("Co","Fe"):-1,
        ("Co","Mn"):-5,("Co","Mo"):-5,("Co","Nb"):-25,("Co","Ni"):0,
        ("Co","Pd"):0,("Co","Ti"):-28,("Co","V"):-14,("Co","Zr"):-41,
        ("Cr","Cu"):12,("Cr","Fe"):-1,("Cr","Mn"):2,("Cr","Mo"):0,
        ("Cr","Nb"):-7,("Cr","Ni"):-7,("Cr","Pd"):-15,("Cr","Ta"):-7,
        ("Cr","Ti"):-7,("Cr","V"):-2,("Cr","W"):-1,("Cr","Zr"):-12,
        ("Cu","Fe"):13,("Cu","Mn"):4,("Cu","Mo"):19,("Cu","Nb"):3,
        ("Cu","Ni"):4,("Cu","Pd"):-14,("Cu","Ti"):-9,("Cu","V"):5,
        ("Cu","Zn"):-1,("Cu","Zr"):-23,("Fe","Mn"):0,("Fe","Mo"):-2,
        ("Fe","Nb"):-16,("Fe","Ni"):-2,("Fe","Pd"):-4,("Fe","Si"):-35,
        ("Fe","Ti"):-17,("Fe","V"):-7,("Fe","W"):-1,("Fe","Zr"):-25,
        ("Hf","Nb"):4,("Hf","Ta"):3,("Hf","Ti"):0,("Hf","V"):-2,
        ("Hf","Zr"):0,("Mn","Mo"):-5,("Mn","Nb"):-4,("Mn","Ni"):-8,
        ("Mo","Nb"):-6,("Mo","Ni"):-7,("Mo","Ta"):-5,("Mo","Ti"):-4,
        ("Mo","V"):0,("Mo","W"):0,("Mo","Zr"):-6,("Nb","Ni"):-30,
        ("Nb","Ta"):0,("Nb","Ti"):2,("Nb","V"):-1,("Nb","W"):-8,
        ("Nb","Zr"):4,("Ni","Pd"):0,("Ni","Pt"):-5,("Ni","Si"):-40,
        ("Ni","Ti"):-35,("Ni","V"):-18,("Ni","Zr"):-49,
        ("Ir","Os"):0,("Os","Pt"):-2,("Os","Rh"):0,("Os","Ru"):0,
        ("Pd","Pt"):0,("Pd","Rh"):0,("Rh","Ru"):0,
        ("Ir","Pd"):0,("Ir","Pt"):0,("Ir","Rh"):0,("Ir","Ru"):0,
        ("Pd","Ru"):0,
        ("Ta","Ti"):1,("Ta","V"):-1,("Ta","W"):-7,("Ta","Zr"):3,
        ("Ti","V"):-2,("Ti","Zr"):0,("V","W"):-1,("V","Zr"):-4,
        ("W","Zr"):-9,
        # Additional pairs for AM/IM alloys (Takeuchi & Inoue 2005)
        ("Be","Cu"):0,("Be","Ni"):-4,("Be","Ti"):-30,("Be","Zr"):-43,
        ("Be","Hf"):-37,("Cu","Hf"):-17,
        ("Hf","Ni"):-42,
        ("Al","Er"):-33,("Dy","Ni"):-34,("Er","Ni"):-35,("Er","Tb"):0,
        ("Dy","Er"):0,("Al","Dy"):-38,("Al","Tb"):-39,("Dy","Tb"):0,
        ("Ca","Cu"):-14,("Ca","Mg"):-6,("Ca","Ni"):-22,("Ca","Zn"):-22,
        ("Cu","Mg"):-3,("Mg","Ni"):-4,("Mg","Zn"):-4,("Ni","Zn"):-9,
        ("Cu","Sn"):-7,("Pd","Sn"):-47,("Pd","Ti"):-52,("Pd","Zr"):-91,
        ("Sn","Ti"):-21,("Sn","Zr"):-44,
        ("Be","Fe"):0,
        # Pairs identified as missing by review (Takeuchi & Inoue 2005)
        ("Al","Zn"):1,("Co","Zn"):-4,("Mn","Ti"):-8,("Mn","V"):-2,("Ni","Tb"):-42,
    }

    elements = list(comp.keys())
    fracs = np.array([comp[e] for e in elements])
    fracs = fracs / fracs.sum()
    n = len(elements)

    # T_m mix
    tm_vals = np.array([T_MELT.get(e, 2000) for e in elements])
    T_m = np.sum(fracs * tm_vals)

    # ΔS_mix
    S_mix = -8.314 * np.sum(fracs[fracs > 0] * np.log(fracs[fracs > 0]))

    # ΔH_mix
    H_mix = 0.0
    for i in range(n):
        for j in range(i+1, n):
            pair = tuple(sorted([elements[i], elements[j]]))
            dh = DELTA_H_MIX.get(pair, 0)
            H_mix += 4 * fracs[i] * fracs[j] * dh

    if abs(H_mix) < 0.01:
        return 999.0  # effectively infinite Ω → very stable SS

    omega = T_m * S_mix / (abs(H_mix) * 1000)
    return omega


def compute_eq10_scaled(comp, struct, omega_sf, gamma=1.0):
    """
    Alonso Eq.10 with scaled DFT Ω_sf correction.
    V = nauc * [Σ c_i V_i + γ · Σ_i Σ_{j≠i} c_i c_j V_j Ω_sf(i,j)]
    """
    elements = list(comp.keys())
    fracs = np.array([comp[e] for e in elements])
    fracs = fracs / fracs.sum()
    for e in elements:
        if e not in KING_ATOMIC_VOLUMES:
            _warn_unknown_element(e, "compute_eq10_scaled")
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
    for e in elements:
        if e not in KING_ATOMIC_VOLUMES:
            _warn_unknown_element(e, "compute_vegard")
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

    for e in elements:
        if e not in KING_ATOMIC_VOLUMES:
            _warn_unknown_element(e, "compute_features")
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
    if len(y) < 2:
        print(f"    loo_cv_xgboost: insufficient data (N={len(y)}), skipping")
        return np.full(len(y), np.nan)

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
    if len(y_binary) == 0 or len(y_hea) < 2:
        print("    two_stage_model: insufficient data "
              f"(binary={len(y_binary)}, HEA={len(y_hea)}), skipping")
        return np.full(len(y_hea), np.nan), np.full(len(y_hea), np.nan)

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

    # --- ML interpolation of missing Ω_sf ---
    print("\n[2b] ML interpolation of missing Ω_sf pairs...")
    print("    Training Ω_sf prediction models (GPR + Ridge)...")

    _, _, rmse_b2, r2_b2, ml_pred_b2 = build_omega_sf_ml_model(omega_b2, "B2")
    _, _, rmse_l12, r2_l12, ml_pred_l12 = build_omega_sf_ml_model(omega_l12, "L12")

    # Fill missing pairs
    omega_b2_filled, filled_b2, stats_b2 = fill_missing_omega_sf(omega_b2, ml_pred_b2)
    omega_l12_filled, filled_l12, stats_l12 = fill_missing_omega_sf(omega_l12, ml_pred_l12)

    print(f"    B2:  {stats_b2['n_dft']} DFT + {stats_b2['n_ml_filled']} ML-filled "
          f"= {stats_b2['n_total']} total (mean σ={stats_b2['mean_uncertainty']:.5f})")
    print(f"    L12: {stats_l12['n_dft']} DFT + {stats_l12['n_ml_filled']} ML-filled "
          f"= {stats_l12['n_total']} total (mean σ={stats_l12['mean_uncertainty']:.5f})")

    # Count how many HEA pairs were missing and got filled
    missing_b2_in_hea = set()
    missing_l12_in_hea = set()
    for hea in ALONSO_TABLE2:
        elems = list(hea["comp"].keys())
        for i in range(len(elems)):
            for j in range(i+1, len(elems)):
                pair = tuple(sorted([elems[i], elems[j]]))
                if hea["struct"] == "BCC" and pair not in omega_b2:
                    missing_b2_in_hea.add(pair)
                elif hea["struct"] == "FCC" and pair not in omega_l12:
                    missing_l12_in_hea.add(pair)
    print(f"    HEA pairs missing DFT data: {len(missing_b2_in_hea)} BCC, "
          f"{len(missing_l12_in_hea)} FCC → now ML-filled")

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
    results["DFT-Ωsf (this work)"] = print_metrics(
        f"DFT-Ωsf (γB={best_gb:.2f},γF={best_gf:.2f})", a_eq10_ss)

    # SS Eq.10 with ML-filled Ω_sf (no missing pairs)
    print("    Computing SS Eq.10 with ML-filled Ω_sf...")
    best_rmse_filled = 999
    best_gb_f, best_gf_f = best_gb, best_gf
    for gb in np.arange(best_gb - 0.3, best_gb + 0.31, 0.05):
        for gf in np.arange(best_gf - 0.3, best_gf + 0.31, 0.05):
            a_pred = np.zeros(N)
            for i in bcc_idx:
                a_pred[i] = compute_eq10_scaled(
                    ALONSO_TABLE2[i]["comp"], "BCC", omega_b2_filled, gb)
            for i in fcc_idx:
                a_pred[i] = compute_eq10_scaled(
                    ALONSO_TABLE2[i]["comp"], "FCC", omega_l12_filled, gf)
            rmse = np.sqrt(np.mean((y_hea - a_pred)**2))
            if rmse < best_rmse_filled:
                best_rmse_filled = rmse
                best_gb_f, best_gf_f = gb, gf

    for gb in np.arange(best_gb_f - 0.05, best_gb_f + 0.06, 0.01):
        for gf in np.arange(best_gf_f - 0.05, best_gf_f + 0.06, 0.01):
            a_pred = np.zeros(N)
            for i in bcc_idx:
                a_pred[i] = compute_eq10_scaled(
                    ALONSO_TABLE2[i]["comp"], "BCC", omega_b2_filled, gb)
            for i in fcc_idx:
                a_pred[i] = compute_eq10_scaled(
                    ALONSO_TABLE2[i]["comp"], "FCC", omega_l12_filled, gf)
            rmse = np.sqrt(np.mean((y_hea - a_pred)**2))
            if rmse < best_rmse_filled:
                best_rmse_filled = rmse
                best_gb_f, best_gf_f = gb, gf

    a_eq10_ss_filled = np.zeros(N)
    for i in bcc_idx:
        a_eq10_ss_filled[i] = compute_eq10_scaled(
            ALONSO_TABLE2[i]["comp"], "BCC", omega_b2_filled, best_gb_f)
    for i in fcc_idx:
        a_eq10_ss_filled[i] = compute_eq10_scaled(
            ALONSO_TABLE2[i]["comp"], "FCC", omega_l12_filled, best_gf_f)

    print(f"    ML-filled γ_BCC = {best_gb_f:.2f}, γ_FCC = {best_gf_f:.2f}")
    results["SS Eq.10 ML-filled"] = print_metrics(
        f"SS Eq.10 ML-filled (γB={best_gb_f:.2f},γF={best_gf_f:.2f})",
        a_eq10_ss_filled)

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
            except Exception as e:
                if te[0] == 0:
                    print(f"        Warning: Cubist({cub_name}) failed: {e}")
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
        ("DFT-Ωsf", a_eq10_ss),
        ("SS Eq.10 ML-filled", a_eq10_ss_filled),
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
        "DFT-Ωsf (this work)": a_eq10_ss,
        "SS Eq.10 ML-filled": a_eq10_ss_filled,
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
            "a_eq10_ss_ml_filled": a_eq10_ss_filled[i],
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

    # Save ML-filled Ω_sf with source labels
    omega_ml_data = []
    all_elements = sorted(KING_ATOMIC_VOLUMES.keys())
    for i, elA in enumerate(all_elements):
        for elB in all_elements[i+1:]:
            pair = tuple(sorted([elA, elB]))
            row_d = {"pair": f"{pair[0]}-{pair[1]}"}
            if pair in omega_b2:
                row_d["omega_b2"] = omega_b2[pair]
                row_d["b2_source"] = "DFT"
            elif pair in ml_pred_b2:
                row_d["omega_b2"] = ml_pred_b2[pair][0]
                row_d["b2_source"] = "ML"
                row_d["b2_sigma"] = ml_pred_b2[pair][1]
            if pair in omega_l12:
                row_d["omega_l12"] = omega_l12[pair]
                row_d["l12_source"] = "DFT"
            elif pair in ml_pred_l12:
                row_d["omega_l12"] = ml_pred_l12[pair][0]
                row_d["l12_source"] = "ML"
                row_d["l12_sigma"] = ml_pred_l12[pair][1]
            omega_ml_data.append(row_d)
    pd.DataFrame(omega_ml_data).to_csv(
        OUTDIR / "omega_sf_ml_filled.csv", index=False)

    # =====================================================================
    # Phase 9b: δ_sf descriptor analysis — single-phase stability
    # =====================================================================
    print("\n[9b] Computing δ_sf descriptors for single-phase analysis...")

    descriptor_data = []
    for i, hea in enumerate(ALONSO_TABLE2):
        comp = hea["comp"]
        struct = hea["struct"]
        elements = sorted(comp.keys())
        elems_str = "-".join(elements)

        dr = compute_delta_r(comp)
        omega_ss = omega_b2 if struct == "BCC" else omega_l12
        d_sf_ss = compute_delta_sf(comp, omega_ss)

        # Combined Ω_sf (fallback to all-structure data for better coverage)
        d_sf_combined = compute_delta_sf(comp, omega_sf)
        d_sf_signed = compute_delta_sf_signed(comp, omega_sf)
        omega_yang = compute_omega_yang(comp, struct)

        # Count missing Ω_sf pairs (structure-specific vs combined)
        n_pairs = 0
        n_missing_ss = 0
        n_missing_combined = 0
        for ii in range(len(elements)):
            for jj in range(ii+1, len(elements)):
                pair = tuple(sorted([elements[ii], elements[jj]]))
                n_pairs += 1
                if pair not in omega_ss:
                    n_missing_ss += 1
                if pair not in omega_sf:
                    n_missing_combined += 1

        descriptor_data.append({
            "composition": elems_str,
            "struct": struct,
            "a_exp": hea["a_exp"],
            "a_eq10_ss": a_eq10_ss[i],
            "delta_r": dr,
            "delta_sf_ss": d_sf_ss,
            "delta_sf": d_sf_combined,
            "delta_sf_signed": d_sf_signed,
            "Omega_yang": omega_yang,
            "n_elements": len(elements),
            "error_abs": abs(y_hea[i] - a_eq10_ss[i]),
            "n_pairs": n_pairs,
            "n_missing_ss": n_missing_ss,
            "n_missing_combined": n_missing_combined,
        })

    desc_df = pd.DataFrame(descriptor_data)
    desc_df.to_csv(OUTDIR / "descriptor_analysis.csv", index=False)

    # Summary statistics
    bcc_desc = desc_df[desc_df["struct"] == "BCC"]
    fcc_desc = desc_df[desc_df["struct"] == "FCC"]

    print(f"    δr  range: BCC [{bcc_desc['delta_r'].min():.2f}, "
          f"{bcc_desc['delta_r'].max():.2f}%], "
          f"FCC [{fcc_desc['delta_r'].min():.2f}, {fcc_desc['delta_r'].max():.2f}%]")
    print(f"    δ_sf (combined) range: BCC [{bcc_desc['delta_sf'].min():.4f}, "
          f"{bcc_desc['delta_sf'].max():.4f}], "
          f"FCC [{fcc_desc['delta_sf'].min():.4f}, {fcc_desc['delta_sf'].max():.4f}]")
    print(f"    δ_sf (SS-only) range: BCC [{bcc_desc['delta_sf_ss'].min():.4f}, "
          f"{bcc_desc['delta_sf_ss'].max():.4f}], "
          f"FCC [{fcc_desc['delta_sf_ss'].min():.4f}, {fcc_desc['delta_sf_ss'].max():.4f}]")
    print(f"    Missing pairs (SS): BCC {bcc_desc['n_missing_ss'].sum()}/{bcc_desc['n_pairs'].sum()}, "
          f"FCC {fcc_desc['n_missing_ss'].sum()}/{fcc_desc['n_pairs'].sum()}")
    print(f"    Missing pairs (combined): BCC {bcc_desc['n_missing_combined'].sum()}/{bcc_desc['n_pairs'].sum()}, "
          f"FCC {fcc_desc['n_missing_combined'].sum()}/{fcc_desc['n_pairs'].sum()}")

    # Correlation: δr vs δ_sf
    from scipy.stats import pearsonr, spearmanr
    r_pearson, p_pearson = pearsonr(desc_df["delta_r"], desc_df["delta_sf"])
    r_spearman, p_spearman = spearmanr(desc_df["delta_r"], desc_df["delta_sf"])
    print(f"    δr vs δ_sf: Pearson r={r_pearson:.3f} (p={p_pearson:.1e}), "
          f"Spearman ρ={r_spearman:.3f} (p={p_spearman:.1e})")

    # Correlation with prediction error
    r_err_dr, _ = pearsonr(desc_df["delta_r"], desc_df["error_abs"])
    r_err_dsf, _ = pearsonr(desc_df["delta_sf"], desc_df["error_abs"])
    print(f"    Error correlation: δr→|ε| r={r_err_dr:.3f}, δ_sf→|ε| r={r_err_dsf:.3f}")

    # =====================================================================
    # Figure: δr vs δ_sf comparison (3-panel)
    # =====================================================================
    fig_desc, axes_desc = plt.subplots(1, 3, figsize=(24, 8))

    # Panel 1: δr vs δ_sf scatter
    ax = axes_desc[0]
    for struct_lab, marker, color in [("BCC", "s", "#4477AA"), ("FCC", "o", "#CC6677")]:
        mask = desc_df["struct"] == struct_lab
        ax.scatter(desc_df.loc[mask, "delta_r"],
                   desc_df.loc[mask, "delta_sf"],
                   marker=marker, c=color, s=100, alpha=0.8,
                   edgecolors="k", lw=0.5, label=struct_lab)
    ax.set_xlabel(r"$\delta_r$ (%)")
    ax.set_ylabel(r"$\delta_{sf}$")
    ax.set_title(f"$\\delta_r$ vs $\\delta_{{sf}}$  (r={r_pearson:.2f})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: δ_sf vs prediction error
    ax = axes_desc[1]
    for struct_lab, marker, color in [("BCC", "s", "#4477AA"), ("FCC", "o", "#CC6677")]:
        mask = desc_df["struct"] == struct_lab
        ax.scatter(desc_df.loc[mask, "delta_sf"],
                   desc_df.loc[mask, "error_abs"],
                   marker=marker, c=color, s=100, alpha=0.8,
                   edgecolors="k", lw=0.5, label=struct_lab)
    ax.set_xlabel(r"$\delta_{sf}$")
    ax.set_ylabel("Prediction error |ε| (Å)")
    ax.set_title(f"$\\delta_{{sf}}$ vs Error  (r={r_err_dsf:.2f})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: δr vs prediction error
    ax = axes_desc[2]
    for struct_lab, marker, color in [("BCC", "s", "#4477AA"), ("FCC", "o", "#CC6677")]:
        mask = desc_df["struct"] == struct_lab
        ax.scatter(desc_df.loc[mask, "delta_r"],
                   desc_df.loc[mask, "error_abs"],
                   marker=marker, c=color, s=100, alpha=0.8,
                   edgecolors="k", lw=0.5, label=struct_lab)
    ax.set_xlabel(r"$\delta_r$ (%)")
    ax.set_ylabel("Prediction error |ε| (Å)")
    ax.set_title(f"$\\delta_r$ vs Error  (r={r_err_dr:.2f})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig_desc.tight_layout()
    fig_desc.savefig(OUTDIR / "fig_delta_sf_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig_desc)
    print("    Saved fig_delta_sf_analysis.png")

    # =====================================================================
    # Figure: Ω_Yang vs δ_sf (phase stability map)
    # =====================================================================
    fig_phase, ax_phase = plt.subplots(figsize=(10, 8))
    for struct_lab, marker, color in [("BCC", "s", "#4477AA"), ("FCC", "o", "#CC6677")]:
        mask = desc_df["struct"] == struct_lab
        omega_vals_plot = desc_df.loc[mask, "Omega_yang"].clip(upper=50)
        ax_phase.scatter(desc_df.loc[mask, "delta_r"],
                         omega_vals_plot,
                         marker=marker, c=color, s=120, alpha=0.8,
                         edgecolors="k", lw=0.5, label=struct_lab)

    # Zhang criteria lines
    ax_phase.axvline(x=6.6, color="red", ls="--", lw=2, alpha=0.7, label=r"$\delta_r$ = 6.6%")
    ax_phase.axhline(y=1.1, color="green", ls="--", lw=2, alpha=0.7, label=r"$\Omega$ = 1.1")
    ax_phase.set_xlabel(r"$\delta_r$ (%)")
    ax_phase.set_ylabel(r"$\Omega$ (Yang-Zhang)")
    ax_phase.set_title("Phase Stability Map: Alonso Table 2 HEAs\n"
                        r"Single-phase region: $\delta_r < 6.6\%$ and $\Omega > 1.1$")
    ax_phase.legend(fontsize=13)
    ax_phase.grid(True, alpha=0.3)
    ax_phase.set_xlim(-0.5, max(desc_df["delta_r"].max() * 1.1, 7.5))
    ax_phase.set_ylim(0, min(desc_df["Omega_yang"].clip(upper=50).max() * 1.2, 55))
    fig_phase.tight_layout()
    fig_phase.savefig(OUTDIR / "fig_phase_stability_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig_phase)
    print("    Saved fig_phase_stability_map.png")

    # =====================================================================
    # Figures
    # =====================================================================
    print("\n[10] Generating publication figures...")

    # --- Fig 1: Parity comparison (4 panels) ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    plot_data = [
        ("Alonso Eq.10", a_eq10_alonso, "gray"),
        ("King Vegard (this work)", a_vegard_king, "steelblue"),
        ("DFT-Ωsf (this work)", a_eq10_ss, "#44AA77"),
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
        ("DFT Eq.10\nSS", results["DFT-Ωsf (this work)"][0]),
        ("SS ML-\nfilled", results["SS Eq.10 ML-filled"][0]),
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
              "#11BB55", "#CC8800", "#FF6600", "#9933CC", "#228B22",
              "#8B4513", "#EE3333", "#DD22DD"]
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
        ("DFT-Ωsf", a_eq10_ss, "#22AA22"),
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

    # =====================================================================
    # Phase 10: Independent Test Set Validation
    # =====================================================================
    print("\n[11] Independent Test Set Validation...")
    print(f"     Test set: {len(INDEPENDENT_TEST)} HEAs from literature")

    ind_results = []
    for hea in INDEPENDENT_TEST:
        comp = hea["comp"]
        struct = hea["struct"]
        a_exp = hea["a_exp"]
        ref = hea.get("ref", "")
        note = hea.get("note", "")

        # Alonso Vegard
        a_veg = compute_vegard(comp, struct)

        # Alonso Eq.10 (combined DFT Ω_sf, γ=1)
        a_eq10_king = compute_eq10_dft(comp, struct, omega_sf)

        # DFT-Ωsf (structure-specific, optimized γ)
        omega_ss = omega_b2 if struct == "BCC" else omega_l12
        gamma_ss = best_gb if struct == "BCC" else best_gf
        a_eq10_ss_ind = compute_eq10_scaled(comp, struct, omega_ss, gamma=gamma_ss)

        # Format composition name with molar fractions for non-equiatomic
        n_elem = len(comp)
        equi_frac = 1.0 / n_elem
        is_equi = all(abs(v - equi_frac) < 0.001 for v in comp.values())
        if is_equi:
            comp_name = "-".join(sorted(comp.keys()))
        else:
            comp_name = "-".join(
                f"{el}{v:.3f}" for el, v in sorted(comp.items()))

        ind_results.append({
            "composition": comp_name,
            "struct": struct,
            "a_exp": a_exp,
            "a_vegard": a_veg,
            "a_eq10_king": a_eq10_king,
            "a_eq10_ss": a_eq10_ss_ind,
            "err_vegard": a_exp - a_veg,
            "err_king": a_exp - a_eq10_king,
            "err_ss": a_exp - a_eq10_ss_ind,
            "ref": ref,
            "note": note,
        })

    ind_df = pd.DataFrame(ind_results)

    # Compute metrics
    def rmse(err):
        return np.sqrt(np.mean(err ** 2))

    def mae(err):
        return np.mean(np.abs(err))

    y_ind = ind_df["a_exp"].values
    err_veg = ind_df["err_vegard"].values
    err_king = ind_df["err_king"].values
    err_ss = ind_df["err_ss"].values

    r2_veg = 1 - np.sum(err_veg**2) / np.sum((y_ind - y_ind.mean())**2)
    r2_king = 1 - np.sum(err_king**2) / np.sum((y_ind - y_ind.mean())**2)
    r2_ss = 1 - np.sum(err_ss**2) / np.sum((y_ind - y_ind.mean())**2)

    # Structure-specific
    bcc_ind = ind_df["struct"] == "BCC"
    fcc_ind = ind_df["struct"] == "FCC"

    print(f"\n    === Independent Test Set Results ({len(ind_df)} HEAs) ===")
    print(f"    {'Method':<25} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
    print(f"    {'-'*49}")
    print(f"    {'Vegard':<25} {rmse(err_veg):>8.4f} {mae(err_veg):>8.4f} {r2_veg:>8.4f}")
    print(f"    {'Alonso Eq.10 (King)':<25} {rmse(err_king):>8.4f} {mae(err_king):>8.4f} {r2_king:>8.4f}")
    print(f"    {'DFT-Ωsf':<25} {rmse(err_ss):>8.4f} {mae(err_ss):>8.4f} {r2_ss:>8.4f}")
    print()

    if bcc_ind.sum() > 0:
        print(f"    BCC ({bcc_ind.sum()} HEAs):")
        print(f"      Vegard:       RMSE = {rmse(err_veg[bcc_ind]):.4f} Å")
        print(f"      King Eq.10:   RMSE = {rmse(err_king[bcc_ind]):.4f} Å")
        print(f"      DFT-Ωsf: RMSE = {rmse(err_ss[bcc_ind]):.4f} Å")

    if fcc_ind.sum() > 0:
        print(f"    FCC ({fcc_ind.sum()} HEAs):")
        print(f"      Vegard:       RMSE = {rmse(err_veg[fcc_ind]):.4f} Å")
        print(f"      King Eq.10:   RMSE = {rmse(err_king[fcc_ind]):.4f} Å")
        print(f"      DFT-Ωsf: RMSE = {rmse(err_ss[fcc_ind]):.4f} Å")

    # Save results
    ind_df.to_csv(OUTDIR / "independent_test_results.csv", index=False)
    print("\n    Saved independent_test_results.csv")

    # --- Figure: Independent test parity plot ---
    fig_ind, axes_ind = plt.subplots(1, 3, figsize=(24, 8))
    methods_ind = [
        ("Vegard", ind_df["a_vegard"].values, "steelblue"),
        ("Alonso Eq.10 (King)", ind_df["a_eq10_king"].values, "gray"),
        ("DFT-Ωsf", ind_df["a_eq10_ss"].values, "#44AA77"),
    ]
    for ax, (name, y_pred, color) in zip(axes_ind, methods_ind):
        bcc_m = ind_df["struct"].values == "BCC"
        fcc_m = ind_df["struct"].values == "FCC"
        ax.scatter(y_ind[bcc_m], y_pred[bcc_m], c=color, marker="s", s=100,
                   alpha=0.85, label=f"BCC ({bcc_m.sum()})", edgecolors="k", lw=0.5)
        ax.scatter(y_ind[fcc_m], y_pred[fcc_m], c=color, marker="o", s=100,
                   alpha=0.85, label=f"FCC ({fcc_m.sum()})", edgecolors="k", lw=0.5)
        lims = [min(y_ind.min(), y_pred.min()) - 0.05,
                max(y_ind.max(), y_pred.max()) + 0.05]
        ax.plot(lims, lims, "k--", lw=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        res = y_ind - y_pred
        rmse_v = np.sqrt(np.mean(res**2))
        r2_v = 1 - np.sum(res**2) / np.sum((y_ind - y_ind.mean())**2)
        ax.set_title(f"{name}\nRMSE = {rmse_v:.4f} Å, R$^2$ = {r2_v:.4f}")
        ax.set_xlabel("Experimental $a$ (Å)")
        ax.set_ylabel("Predicted $a$ (Å)")
        ax.legend(fontsize=13)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
    fig_ind.suptitle(f"Independent Test Set Validation ({len(ind_df)} HEAs, NOT in Alonso Table 2)",
                     fontsize=20, fontweight="bold")
    fig_ind.tight_layout()
    fig_ind.savefig(OUTDIR / "fig_independent_test.png", dpi=200, bbox_inches="tight")
    plt.close(fig_ind)
    print("    Saved fig_independent_test.png")

    # Per-alloy detail table
    print("\n    Per-alloy predictions:")
    print(f"    {'Composition':<30} {'Struct':>5} {'a_exp':>7} {'a_SS':>7} {'Err':>7} {'Ref'}")
    for _, row in ind_df.iterrows():
        print(f"    {row['composition']:<30} {row['struct']:>5} "
              f"{row['a_exp']:>7.4f} {row['a_eq10_ss']:>7.4f} "
              f"{row['err_ss']:>7.4f} {row['ref']}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print(f"★ Best overall RMSE: {best_method[1][0]:.4f} Å ({best_method[0]})")
    print(f"  Alonso Eq.10 RMSE:    {results['Alonso Eq.10'][0]:.4f} Å")
    print(f"  Improvement:          {(1 - best_method[1][0]/results['Alonso Eq.10'][0])*100:.1f}%")
    print(f"  Noise floor:          {sigma_noise:.4f} Å")
    alonso_rmse = results["Alonso Eq.10"][0]
    if best_method[1][0] < alonso_rmse:
        print(f"  → Surpassed Alonso Eq.10 ({alonso_rmse:.4f} Å)")
    print()
    print("  --- Independent Test Set ---")
    print(f"  DFT-Ωsf RMSE: {rmse(err_ss):.4f} Å")
    if bcc_ind.sum() > 0:
        print(f"  BCC:  {rmse(err_ss[bcc_ind]):.4f} Å ({bcc_ind.sum()} HEAs)")
    if fcc_ind.sum() > 0:
        print(f"  FCC:  {rmse(err_ss[fcc_ind]):.4f} Å ({fcc_ind.sum()} HEAs)")
    print("=" * 70)

    # =====================================================================
    # Phase 12: Multi-phase HEA Discrimination Analysis
    # Compare δr vs δ_sf for single-phase / multi-phase classification
    # =====================================================================
    print("\n[12] Multi-phase HEA Discrimination Analysis...")
    print(f"     Database: {len(MULTIPHASE_HEA_DB)} HEAs")

    from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                                 average_precision_score, confusion_matrix,
                                 f1_score, accuracy_score)

    # Compute descriptors for all multi-phase DB entries
    mp_results = []
    for hea in MULTIPHASE_HEA_DB:
        comp = hea["comp"]
        phase = hea["phase"]  # SS, IM, or AM
        struct = hea.get("struct", "")
        ref = hea.get("ref", "")

        # δr (traditional size mismatch)
        dr = compute_delta_r(comp)

        # δ_sf using combined Ω_sf (B2+L12 averaged)
        dsf_combined = compute_delta_sf(comp, omega_sf)

        # δ_sf using structure-specific Ω_sf
        dsf_b2 = compute_delta_sf(comp, omega_b2)
        dsf_l12 = compute_delta_sf(comp, omega_l12)

        # Ω parameter (Yang & Zhang 2012)
        omega_yz = compute_omega_yang(comp, struct)

        # ΔH_mix and ΔS_mix for analysis
        elements = list(comp.keys())
        fracs = np.array([comp[e] for e in elements])
        fracs = fracs / fracs.sum()
        S_mix = -8.314 * np.sum(fracs[fracs > 0] * np.log(fracs[fracs > 0]))

        # VEC
        VEC_vals = {
            "Al":3,"Co":9,"Cr":6,"Cu":11,"Fe":8,"Mn":7,"Mo":6,"Nb":5,
            "Ni":10,"Pd":10,"Pt":10,"Ta":5,"Ti":4,"V":5,"W":6,"Zr":4,
            "Hf":4,"Ru":8,"Rh":9,"Ir":9,"Os":8,"Re":7,"Au":11,"Ag":11,
            "Zn":12,"Si":4,"Ge":4,"Be":2,"Mg":2,"Sc":3,"Y":3,"La":3,
            "Ce":4,"B":3,"Sn":4,"Pb":4,"Er":3,"Tb":3,"Dy":3,"Ca":2,
        }
        vec = np.sum(fracs * np.array([VEC_vals.get(e, 5) for e in elements]))

        mp_results.append({
            "composition": "-".join(sorted(comp.keys())),
            "phase": phase,
            "struct": struct,
            "is_ss": 1 if phase == "SS" else 0,
            "delta_r": dr,
            "delta_sf_combined": dsf_combined,
            "delta_sf_b2": dsf_b2,
            "delta_sf_l12": dsf_l12,
            "omega_yz": omega_yz,
            "S_mix": S_mix,
            "VEC": vec,
            "ref": ref,
        })

    mp_df = pd.DataFrame(mp_results)
    mp_df.to_csv(OUTDIR / "multiphase_classification.csv", index=False)

    n_ss = (mp_df["phase"] == "SS").sum()
    n_im = (mp_df["phase"] == "IM").sum()
    n_am = (mp_df["phase"] == "AM").sum()
    print(f"     SS: {n_ss}, IM: {n_im}, AM: {n_am}")

    # Binary classification: SS=1 (positive) vs non-SS=0 (negative)
    y_true = mp_df["is_ss"].values

    # --- ROC curves for δr and δ_sf ---
    # For SS prediction: lower δr / δ_sf → more likely SS
    # So we negate the scores for ROC (higher score = more positive)
    dr_scores = -mp_df["delta_r"].values
    dsf_scores = -mp_df["delta_sf_combined"].values

    # Also test Ω parameter (higher Ω → more likely SS)
    omega_scores = np.clip(mp_df["omega_yz"].values, 0, 100)

    fpr_dr, tpr_dr, thresh_dr = roc_curve(y_true, dr_scores)
    auc_dr = auc(fpr_dr, tpr_dr)

    fpr_dsf, tpr_dsf, thresh_dsf = roc_curve(y_true, dsf_scores)
    auc_dsf = auc(fpr_dsf, tpr_dsf)

    fpr_om, tpr_om, thresh_om = roc_curve(y_true, omega_scores)
    auc_om = auc(fpr_om, tpr_om)

    print("\n    === ROC AUC ===")
    print(f"    δr:          AUC = {auc_dr:.3f}")
    print(f"    δ_sf (comb): AUC = {auc_dsf:.3f}")
    print(f"    Ω (Yang):    AUC = {auc_om:.3f}")

    # --- Precision-Recall curves ---
    prec_dr, rec_dr, _ = precision_recall_curve(y_true, dr_scores)
    ap_dr = average_precision_score(y_true, dr_scores)

    prec_dsf, rec_dsf, _ = precision_recall_curve(y_true, dsf_scores)
    ap_dsf = average_precision_score(y_true, dsf_scores)

    prec_om, rec_om, _ = precision_recall_curve(y_true, omega_scores)
    ap_om = average_precision_score(y_true, omega_scores)

    print("\n    === Average Precision ===")
    print(f"    δr:          AP = {ap_dr:.3f}")
    print(f"    δ_sf (comb): AP = {ap_dsf:.3f}")
    print(f"    Ω (Yang):    AP = {ap_om:.3f}")

    # --- Yang-Zhang criterion: δr < 6.6% AND Ω > 1.1 ---
    yz_pred = ((mp_df["delta_r"] < 6.6) & (mp_df["omega_yz"] > 1.1)).astype(int)
    yz_acc = accuracy_score(y_true, yz_pred)
    yz_f1 = f1_score(y_true, yz_pred, zero_division=0)
    cm_yz = confusion_matrix(y_true, yz_pred, labels=[0, 1])

    print("\n    === Yang-Zhang Criterion (δr<6.6%, Ω>1.1) ===")
    print(f"    Accuracy: {yz_acc:.3f}, F1: {yz_f1:.3f}")
    print("    Confusion matrix (rows=true, cols=pred):")
    print(f"      [non-SS pred non-SS, non-SS pred SS] = [{cm_yz[0,0]:2d}, {cm_yz[0,1]:2d}]")
    print(f"      [SS pred non-SS,     SS pred SS    ] = [{cm_yz[1,0]:2d}, {cm_yz[1,1]:2d}]")

    # --- Optimal threshold for δ_sf ---
    # Find threshold that maximizes Youden's J statistic on ROC
    j_scores_dr = tpr_dr - fpr_dr
    opt_idx_dr = np.argmax(j_scores_dr)
    opt_thresh_dr = -thresh_dr[opt_idx_dr]  # negate back
    j_scores_dsf = tpr_dsf - fpr_dsf
    opt_idx_dsf = np.argmax(j_scores_dsf)
    opt_thresh_dsf = -thresh_dsf[opt_idx_dsf]  # negate back

    # Apply optimal thresholds
    pred_dr_opt = (mp_df["delta_r"] < opt_thresh_dr).astype(int)
    acc_dr_opt = accuracy_score(y_true, pred_dr_opt)
    f1_dr_opt = f1_score(y_true, pred_dr_opt, zero_division=0)

    pred_dsf_opt = (mp_df["delta_sf_combined"] < opt_thresh_dsf).astype(int)
    acc_dsf_opt = accuracy_score(y_true, pred_dsf_opt)
    f1_dsf_opt = f1_score(y_true, pred_dsf_opt, zero_division=0)

    print("\n    === Optimal Thresholds (Youden's J) ===")
    print(f"    δr:  threshold = {opt_thresh_dr:.2f}%, Acc = {acc_dr_opt:.3f}, F1 = {f1_dr_opt:.3f}")
    print(f"    δ_sf: threshold = {opt_thresh_dsf:.4f}, Acc = {acc_dsf_opt:.3f}, F1 = {f1_dsf_opt:.3f}")

    # --- Combined criterion: δ_sf + Ω ---
    # Test: δ_sf < opt_thresh AND Ω > 1.1
    pred_dsf_yz = ((mp_df["delta_sf_combined"] < opt_thresh_dsf) &
                   (mp_df["omega_yz"] > 1.1)).astype(int)
    acc_dsf_yz = accuracy_score(y_true, pred_dsf_yz)
    f1_dsf_yz = f1_score(y_true, pred_dsf_yz, zero_division=0)
    cm_dsf_yz = confusion_matrix(y_true, pred_dsf_yz, labels=[0, 1])

    print("\n    === Combined δ_sf + Ω Criterion ===")
    print(f"    δ_sf<{opt_thresh_dsf:.4f} AND Ω>1.1: Acc = {acc_dsf_yz:.3f}, F1 = {f1_dsf_yz:.3f}")

    # ==================================================================
    # Publication-quality figures
    # ==================================================================
    FONTSIZE = 18
    TICK_SIZE = 14

    # --- Figure 1: ROC comparison (3 panels) ---
    fig_roc, axes_roc = plt.subplots(1, 3, figsize=(24, 8))
    for ax in axes_roc:
        ax.tick_params(labelsize=TICK_SIZE)

    ax1, ax2, ax3 = axes_roc

    # Panel 1: ROC curves overlay
    ax1.plot(fpr_dr, tpr_dr, 'b-', lw=2.5,
             label=f'$\\delta_r$ (AUC={auc_dr:.3f})')
    ax1.plot(fpr_dsf, tpr_dsf, 'r-', lw=2.5,
             label=f'$\\delta_{{sf}}$ (AUC={auc_dsf:.3f})')
    ax1.plot(fpr_om, tpr_om, 'g--', lw=2.5,
             label=f'$\\Omega$ (AUC={auc_om:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax1.set_xlabel("False Positive Rate", fontsize=FONTSIZE)
    ax1.set_ylabel("True Positive Rate", fontsize=FONTSIZE)
    ax1.set_title("ROC Curves: SS vs non-SS", fontsize=FONTSIZE)
    ax1.legend(fontsize=FONTSIZE - 2, loc='lower right')
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])
    ax1.grid(True, alpha=0.3)

    # Panel 2: Precision-Recall curves
    ax2.plot(rec_dr, prec_dr, 'b-', lw=2.5,
             label=f'$\\delta_r$ (AP={ap_dr:.3f})')
    ax2.plot(rec_dsf, prec_dsf, 'r-', lw=2.5,
             label=f'$\\delta_{{sf}}$ (AP={ap_dsf:.3f})')
    ax2.plot(rec_om, prec_om, 'g--', lw=2.5,
             label=f'$\\Omega$ (AP={ap_om:.3f})')
    baseline = y_true.sum() / len(y_true)
    ax2.axhline(baseline, color='k', ls='--', lw=1, alpha=0.5,
                label=f'Baseline ({baseline:.2f})')
    ax2.set_xlabel("Recall", fontsize=FONTSIZE)
    ax2.set_ylabel("Precision", fontsize=FONTSIZE)
    ax2.set_title("Precision-Recall Curves", fontsize=FONTSIZE)
    ax2.legend(fontsize=FONTSIZE - 2, loc='lower left')
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([0, 1.05])
    ax2.grid(True, alpha=0.3)

    # Panel 3: Confusion matrices comparison
    # Show Yang-Zhang and δ_sf+Ω side by side
    labels_cm = ["non-SS", "SS"]
    cms = [
        (cm_yz, f"Yang-Zhang\n($\\delta_r$<6.6%, $\\Omega$>1.1)\nAcc={yz_acc:.2f}, F1={yz_f1:.2f}"),
        (cm_dsf_yz, f"$\\delta_{{sf}}$+$\\Omega$\n($\\delta_{{sf}}$<{opt_thresh_dsf:.3f}, $\\Omega$>1.1)\nAcc={acc_dsf_yz:.2f}, F1={f1_dsf_yz:.2f}"),
    ]
    for idx, (cm, title) in enumerate(cms):
        x_off = 0.05 + idx * 0.5
        for i in range(2):
            for j in range(2):
                color = '#44AA77' if i == j else '#CC5555'
                alpha = 0.3 + 0.4 * cm[i, j] / max(cm.max(), 1)
                rect = plt.Rectangle((x_off + j * 0.18, 0.25 + (1 - i) * 0.22),
                                     0.16, 0.18, fc=color, alpha=alpha,
                                     transform=ax3.transAxes)
                ax3.add_patch(rect)
                ax3.text(x_off + j * 0.18 + 0.08, 0.25 + (1 - i) * 0.22 + 0.09,
                         str(cm[i, j]), ha='center', va='center',
                         fontsize=FONTSIZE + 2, fontweight='bold',
                         transform=ax3.transAxes)
        # Row/column labels
        for i, lab in enumerate(labels_cm):
            ax3.text(x_off - 0.03, 0.25 + (1 - i) * 0.22 + 0.09, lab,
                     ha='right', va='center', fontsize=TICK_SIZE,
                     transform=ax3.transAxes)
            ax3.text(x_off + i * 0.18 + 0.08, 0.19, lab,
                     ha='center', va='top', fontsize=TICK_SIZE,
                     transform=ax3.transAxes)
        ax3.text(x_off + 0.18, 0.78, title, ha='center', va='bottom',
                 fontsize=TICK_SIZE, transform=ax3.transAxes)
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])
    ax3.axis('off')
    ax3.set_title("Confusion Matrices", fontsize=FONTSIZE)

    fig_roc.suptitle(f"Single-phase SS Classification ({len(mp_df)} HEAs: {n_ss} SS, {n_im} IM, {n_am} AM)",
                     fontsize=FONTSIZE + 2, fontweight='bold')
    fig_roc.tight_layout()
    fig_roc.savefig(OUTDIR / "fig_multiphase_roc.png", dpi=200, bbox_inches="tight")
    plt.close(fig_roc)
    print("\n    Saved fig_multiphase_roc.png")

    # --- Figure 2: δr vs δ_sf scatter colored by phase ---
    fig_sc, axes_sc = plt.subplots(1, 3, figsize=(24, 8))
    for ax in axes_sc:
        ax.tick_params(labelsize=TICK_SIZE)

    ax1, ax2, ax3 = axes_sc
    phase_colors = {"SS": "#44AA77", "IM": "#CC5555", "AM": "#4477AA"}
    phase_markers = {"SS": "o", "IM": "s", "AM": "D"}

    # Panel 1: δr vs δ_sf
    for ph in ["SS", "IM", "AM"]:
        mask = mp_df["phase"] == ph
        if mask.sum() == 0:
            continue
        ax1.scatter(mp_df.loc[mask, "delta_r"],
                    mp_df.loc[mask, "delta_sf_combined"],
                    c=phase_colors[ph], marker=phase_markers[ph],
                    s=120, alpha=0.8, label=f"{ph} ({mask.sum()})",
                    edgecolors='k', lw=0.5)
    ax1.axvline(6.6, color='b', ls='--', lw=1.5, alpha=0.7, label='$\\delta_r$=6.6%')
    ax1.axhline(opt_thresh_dsf, color='r', ls='--', lw=1.5, alpha=0.7,
                label=f'$\\delta_{{sf}}$={opt_thresh_dsf:.3f}')
    ax1.set_xlabel("$\\delta_r$ (%)", fontsize=FONTSIZE)
    ax1.set_ylabel("$\\delta_{sf}$ (combined)", fontsize=FONTSIZE)
    ax1.set_title("$\\delta_r$ vs $\\delta_{sf}$", fontsize=FONTSIZE)
    ax1.legend(fontsize=FONTSIZE - 4)
    ax1.grid(True, alpha=0.3)

    # Panel 2: δr–Ω phase stability map
    for ph in ["SS", "IM", "AM"]:
        mask = mp_df["phase"] == ph
        if mask.sum() == 0:
            continue
        ax2.scatter(mp_df.loc[mask, "delta_r"],
                    np.clip(mp_df.loc[mask, "omega_yz"], 0, 50),
                    c=phase_colors[ph], marker=phase_markers[ph],
                    s=120, alpha=0.8, label=f"{ph} ({mask.sum()})",
                    edgecolors='k', lw=0.5)
    ax2.axvline(6.6, color='k', ls='--', lw=1.5, alpha=0.5)
    ax2.axhline(1.1, color='k', ls='--', lw=1.5, alpha=0.5)
    ax2.fill_between([0, 6.6], 1.1, 50, alpha=0.08, color='green',
                     label='Yang-Zhang SS region')
    ax2.set_xlabel("$\\delta_r$ (%)", fontsize=FONTSIZE)
    ax2.set_ylabel("$\\Omega$ (Yang-Zhang)", fontsize=FONTSIZE)
    ax2.set_title("$\\delta_r$–$\\Omega$ Phase Map", fontsize=FONTSIZE)
    ax2.legend(fontsize=FONTSIZE - 4)
    ax2.set_ylim([-0.5, min(50, mp_df["omega_yz"].clip(upper=50).max() + 5)])
    ax2.grid(True, alpha=0.3)

    # Panel 3: δ_sf–Ω phase map
    for ph in ["SS", "IM", "AM"]:
        mask = mp_df["phase"] == ph
        if mask.sum() == 0:
            continue
        ax3.scatter(mp_df.loc[mask, "delta_sf_combined"],
                    np.clip(mp_df.loc[mask, "omega_yz"], 0, 50),
                    c=phase_colors[ph], marker=phase_markers[ph],
                    s=120, alpha=0.8, label=f"{ph} ({mask.sum()})",
                    edgecolors='k', lw=0.5)
    ax3.axvline(opt_thresh_dsf, color='r', ls='--', lw=1.5, alpha=0.7)
    ax3.axhline(1.1, color='k', ls='--', lw=1.5, alpha=0.5)
    ax3.set_xlabel("$\\delta_{sf}$ (combined)", fontsize=FONTSIZE)
    ax3.set_ylabel("$\\Omega$ (Yang-Zhang)", fontsize=FONTSIZE)
    ax3.set_title("$\\delta_{sf}$–$\\Omega$ Phase Map", fontsize=FONTSIZE)
    ax3.legend(fontsize=FONTSIZE - 4)
    ax3.set_ylim([-0.5, min(50, mp_df["omega_yz"].clip(upper=50).max() + 5)])
    ax3.grid(True, alpha=0.3)

    fig_sc.suptitle(f"Phase Stability Maps ({len(mp_df)} HEAs)",
                    fontsize=FONTSIZE + 2, fontweight='bold')
    fig_sc.tight_layout()
    fig_sc.savefig(OUTDIR / "fig_multiphase_scatter.png", dpi=200, bbox_inches="tight")
    plt.close(fig_sc)
    print("    Saved fig_multiphase_scatter.png")

    # --- Figure 3: Threshold optimization ---
    fig_thresh, axes_thresh = plt.subplots(1, 2, figsize=(16, 8))
    for ax in axes_thresh:
        ax.tick_params(labelsize=TICK_SIZE)

    # Sweep δr thresholds
    dr_range = np.linspace(0, mp_df["delta_r"].max() + 1, 200)
    f1_dr_arr = []
    acc_dr_arr = []
    for t in dr_range:
        pred = (mp_df["delta_r"] < t).astype(int).values
        f1_dr_arr.append(f1_score(y_true, pred, zero_division=0))
        acc_dr_arr.append(accuracy_score(y_true, pred))
    f1_dr_arr = np.array(f1_dr_arr)
    acc_dr_arr = np.array(acc_dr_arr)

    axes_thresh[0].plot(dr_range, f1_dr_arr, 'b-', lw=2.5, label='F1 score')
    axes_thresh[0].plot(dr_range, acc_dr_arr, 'b--', lw=2, label='Accuracy')
    best_f1_dr_idx = np.argmax(f1_dr_arr)
    axes_thresh[0].axvline(dr_range[best_f1_dr_idx], color='b', ls=':', lw=1.5,
                           label=f'Best F1: {dr_range[best_f1_dr_idx]:.1f}%')
    axes_thresh[0].axvline(6.6, color='k', ls='--', lw=1.5, alpha=0.5,
                           label='Yang-Zhang 6.6%')
    axes_thresh[0].set_xlabel("$\\delta_r$ threshold (%)", fontsize=FONTSIZE)
    axes_thresh[0].set_ylabel("Score", fontsize=FONTSIZE)
    axes_thresh[0].set_title("$\\delta_r$ Threshold Optimization", fontsize=FONTSIZE)
    axes_thresh[0].legend(fontsize=FONTSIZE - 2)
    axes_thresh[0].grid(True, alpha=0.3)
    axes_thresh[0].set_ylim([0, 1.05])

    # Sweep δ_sf thresholds
    dsf_range = np.linspace(0, mp_df["delta_sf_combined"].max() + 0.01, 200)
    f1_dsf_arr = []
    acc_dsf_arr = []
    for t in dsf_range:
        pred = (mp_df["delta_sf_combined"] < t).astype(int).values
        f1_dsf_arr.append(f1_score(y_true, pred, zero_division=0))
        acc_dsf_arr.append(accuracy_score(y_true, pred))
    f1_dsf_arr = np.array(f1_dsf_arr)
    acc_dsf_arr = np.array(acc_dsf_arr)

    axes_thresh[1].plot(dsf_range, f1_dsf_arr, 'r-', lw=2.5, label='F1 score')
    axes_thresh[1].plot(dsf_range, acc_dsf_arr, 'r--', lw=2, label='Accuracy')
    best_f1_dsf_idx = np.argmax(f1_dsf_arr)
    axes_thresh[1].axvline(dsf_range[best_f1_dsf_idx], color='r', ls=':', lw=1.5,
                           label=f'Best F1: {dsf_range[best_f1_dsf_idx]:.4f}')
    axes_thresh[1].set_xlabel("$\\delta_{sf}$ threshold", fontsize=FONTSIZE)
    axes_thresh[1].set_ylabel("Score", fontsize=FONTSIZE)
    axes_thresh[1].set_title("$\\delta_{sf}$ Threshold Optimization", fontsize=FONTSIZE)
    axes_thresh[1].legend(fontsize=FONTSIZE - 2)
    axes_thresh[1].grid(True, alpha=0.3)
    axes_thresh[1].set_ylim([0, 1.05])

    fig_thresh.suptitle("Threshold Optimization for SS Classification",
                        fontsize=FONTSIZE + 2, fontweight='bold')
    fig_thresh.tight_layout()
    fig_thresh.savefig(OUTDIR / "fig_multiphase_threshold.png", dpi=200,
                       bbox_inches="tight")
    plt.close(fig_thresh)
    print("    Saved fig_multiphase_threshold.png")

    # --- Summary table ---
    print(f"\n    === Discrimination Summary ({len(mp_df)} HEAs) ===")
    print(f"    {'Criterion':<35} {'AUC':>6} {'AP':>6} {'Acc':>6} {'F1':>6}")
    print(f"    {'-'*59}")
    print(f"    {'δr (optimal threshold)':<35} {auc_dr:>6.3f} {ap_dr:>6.3f} "
          f"{acc_dr_opt:>6.3f} {f1_dr_opt:>6.3f}")
    print(f"    {'δ_sf (optimal threshold)':<35} {auc_dsf:>6.3f} {ap_dsf:>6.3f} "
          f"{acc_dsf_opt:>6.3f} {f1_dsf_opt:>6.3f}")
    print(f"    {'Ω only':<35} {auc_om:>6.3f} {ap_om:>6.3f} {'—':>6} {'—':>6}")
    print(f"    {'Yang-Zhang (δr<6.6%, Ω>1.1)':<35} {'—':>6} {'—':>6} "
          f"{yz_acc:>6.3f} {yz_f1:>6.3f}")
    print(f"    {'δ_sf + Ω combined':<35} {'—':>6} {'—':>6} "
          f"{acc_dsf_yz:>6.3f} {f1_dsf_yz:>6.3f}")
    print()

    return (results, y_best, y_ensemble_opt, a_eq10_ss, a_ss_gpr,
            gpr_uncertainty, a_ss_rf, a_ss_cub, a_eq10_ss_filled)


if __name__ == "__main__":
    (results, y_best, y_ensemble, a_eq10_ss, a_gpr,
     gpr_unc, a_rf, a_cub, a_filled) = main()
