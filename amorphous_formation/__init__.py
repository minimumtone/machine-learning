"""
Amorphous Formation Model Package
非晶質形成モデルパッケージ

This package provides tools for validating amorphous (glass) formation models
using CALPHAD thermodynamics and Davis-Uhlmann kinetics.

Modules:
    - materials_database: Known material parameters for validation
    - calphad_thermodynamics: Gibbs energy calculations
    - doolittle_viscosity: Temperature-dependent viscosity model
    - davis_uhlmann_model: TTT curve generation and nucleation theory
    - sensitivity_analysis: Parameter sensitivity analysis
    - visualization: Plotting functions for all required diagrams
"""

from .materials_database import MaterialsDatabase, Material
from .calphad_thermodynamics import CALPHADThermodynamics
from .doolittle_viscosity import DoolittleViscosity
from .davis_uhlmann_model import DavisUhlmannModel
from .sensitivity_analysis import SensitivityAnalysis
from .visualization import AmorphousVisualization

__version__ = "1.0.0"
__author__ = "Machine Learning Repository"

__all__ = [
    "MaterialsDatabase",
    "Material",
    "CALPHADThermodynamics",
    "DoolittleViscosity",
    "DavisUhlmannModel",
    "SensitivityAnalysis",
    "AmorphousVisualization",
]
