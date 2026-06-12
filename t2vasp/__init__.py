"""
t2vasp — Text-to-VASP: natural language → VASP calculation setup.

Core modules (text-to-VASP)
---------------------------
intent      : Natural language → calculation type classification
entity      : Element / formula / prototype extraction from text
templates   : Calculation-type-specific INCAR template library
generator   : INCAR / POSCAR / KPOINTS / POTCAR script generation
scheduler   : PBS / Slurm / local job script generation

Post-processing modules
-----------------------
parser      : VASP output file parsing (OUTCAR, vasprun.xml, POSCAR, DOSCAR)
calculator  : Physical-quantity extraction and crystal-field analysis
exporter    : CSV / JSON result export
visualizer  : Matplotlib-based plotting helpers
pipeline    : End-to-end batch automation
structure   : ASE-based structure manipulation and POSCAR generation
optimizer   : Genetic-algorithm / evaluation-function framework
cli         : Command-line interface (argparse)
"""

__version__ = "0.1.0"
