"""
t2vasp — VASP post-processing and automated structure optimization toolkit.

Modules
-------
parser      : VASP output file parsing (OUTCAR, vasprun.xml, POSCAR, DOSCAR)
calculator  : Physical-quantity extraction and t2 analysis
exporter    : CSV / JSON result export
visualizer  : Matplotlib-based plotting helpers
pipeline    : End-to-end batch automation
structure   : ASE-based structure manipulation and POSCAR generation
optimizer   : Genetic-algorithm / evaluation-function framework
cli         : Command-line interface (argparse)
"""

__version__ = "0.1.0"
