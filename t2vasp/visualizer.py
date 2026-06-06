"""
Plotting helpers — produce publication-quality figures from analysis results.

All functions return a ``matplotlib.figure.Figure`` and optionally save to disk.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .calculator import CalculationResult
from .parser import DosData

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _require_mpl() -> None:
    if not HAS_MPL:
        raise ImportError("matplotlib is required for plotting. "
                          "Install with: pip install matplotlib")


# ── Energy bar chart ─────────────────────────────────────────────────
def plot_energy_comparison(
    results: List[CalculationResult],
    save_path: str | Path | None = None,
    figsize: Tuple[float, float] = (10, 5),
) -> "plt.Figure":
    """Bar chart of energy per atom for each calculation.

    Parameters
    ----------
    results : list[CalculationResult]
    save_path : str or Path, optional
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    _require_mpl()

    labels = []
    energies = []
    for r in results:
        if r.energy is not None:
            labels.append(r.label.split("/")[-1])
            energies.append(r.energy.energy_per_atom)

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(labels))
    ax.bar(x, energies, color="steelblue", edgecolor="k", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Energy per atom (eV)")
    ax.set_title("t2vasp — Energy Comparison")
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.5)
    fig.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=150)
        logger.info("Plot saved: %s", save_path)

    return fig


# ── DOS plot ─────────────────────────────────────────────────────────
def plot_dos(
    dos: DosData,
    save_path: str | Path | None = None,
    figsize: Tuple[float, float] = (8, 5),
    energy_range: Tuple[float, float] = (-10.0, 6.0),
) -> "plt.Figure":
    """Total density of states plot (energy relative to Fermi level).

    Parameters
    ----------
    dos : DosData
    save_path : str or Path, optional
    figsize : tuple
    energy_range : tuple
        (min, max) relative to Fermi energy.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _require_mpl()

    e_shifted = dos.energies - dos.fermi_energy
    mask = (e_shifted >= energy_range[0]) & (e_shifted <= energy_range[1])

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(e_shifted[mask], dos.total_dos[mask], color="k", linewidth=0.8)
    ax.axvline(0, color="r", linestyle="--", linewidth=0.6, label="$E_F$")
    ax.fill_between(e_shifted[mask], dos.total_dos[mask],
                    where=(e_shifted[mask] <= 0), alpha=0.15, color="steelblue")
    ax.set_xlabel("$E - E_F$ (eV)")
    ax.set_ylabel("DOS (states/eV)")
    ax.set_title("Total Density of States")
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=150)
        logger.info("DOS plot saved: %s", save_path)

    return fig


# ── ΔE ranking plot ──────────────────────────────────────────────────
def plot_delta_energy(
    delta_e: Dict[str, float],
    save_path: str | Path | None = None,
    figsize: Tuple[float, float] = (10, 5),
) -> "plt.Figure":
    """Horizontal bar chart of ΔE relative to the most stable structure.

    Parameters
    ----------
    delta_e : dict  ``{label: eV/atom}``
    save_path : str or Path, optional
    """
    _require_mpl()

    sorted_items = sorted(delta_e.items(), key=lambda x: x[1])
    labels = [it[0].split("/")[-1] for it in sorted_items]
    values = [it[1] for it in sorted_items]

    fig, ax = plt.subplots(figsize=figsize)
    colors = ["green" if v == 0 else "steelblue" for v in values]
    ax.barh(labels, values, color=colors, edgecolor="k", linewidth=0.5)
    ax.set_xlabel("$\\Delta E$ (eV/atom)")
    ax.set_title("Structure Ranking by Energy")
    fig.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=150)
        logger.info("ΔE plot saved: %s", save_path)

    return fig


# ── Lattice-constant comparison ──────────────────────────────────────
def plot_lattice_comparison(
    results: List[CalculationResult],
    save_path: str | Path | None = None,
    figsize: Tuple[float, float] = (10, 5),
) -> "plt.Figure":
    """Bar chart comparing lattice constants across calculations."""
    _require_mpl()

    labels = []
    a_vals = []
    for r in results:
        if r.structure is not None:
            labels.append(r.label.split("/")[-1])
            a_vals.append(r.structure.lattice_constant)

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(labels))
    ax.bar(x, a_vals, color="coral", edgecolor="k", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Lattice constant (Å)")
    ax.set_title("t2vasp — Lattice Constant Comparison")
    fig.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=150)
        logger.info("Lattice plot saved: %s", save_path)

    return fig
