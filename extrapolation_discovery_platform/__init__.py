"""
Extrapolation Discovery Platform
外挿発見基盤

A systematic framework for evaluating feature sets in materials science,
including OOD detection, feature validity assessment, and candidate
composition proposal for extrapolation regions.

NOTE: HEA (High Entropy Alloys) is used as a concrete example domain;
the platform is designed to be domain-agnostic.

Modules
-------
features       Feature engineering (FS_BASE / FS_THERMO / FS_SIZE / FS_ELECTRON / FS_ALL)
dataset        Synthetic dataset generation with property proxy
splitters      Data splitting strategies (RandomCV / CompositionBlock / ElementExclusion)
workflows      ML workflows (WF-LIN / WF-XGB / WF-ENS)
ood            Out-of-Distribution detection (Mahalanobis / kNN)
evaluation     Feature validity scoring engine
visualization  OOD maps, ranking charts, comparison tables
report         Markdown report generator with literature evidence integration
runner         Experiment orchestrator with MLflow-style tracking

Sub-packages
------------
literature_graph   Literature metadata graph (JSONL + FAISS embedding search)
integrations       External tool adapters (MLflow, Feast, MInt)
"""

# Install pandas C-contiguous compatibility patches BEFORE any other
# imports that might use pandas.  This is the global SIGSEGV fix for
# pandas 3.0's F-contiguous array layout.
from extrapolation_discovery_platform._compat import install as _install_compat
_install_compat()
del _install_compat

__version__ = "0.1.0"
