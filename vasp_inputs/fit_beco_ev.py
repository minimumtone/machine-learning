#!/usr/bin/env python3
"""Fit a Birch--Murnaghan equation of state to Be-Co E--V results.

The input CSV must contain one row per fixed-volume calculation and columns
named ``volume_A3`` and ``energy_eV``.  No result values are embedded here:
the script only performs a fit after VASP results have been extracted.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit


EV_TO_GPA = 160.21766208


def birch_murnaghan(
    volume: np.ndarray, e0: float, v0: float, b0: float, bp: float
) -> np.ndarray:
    eta = (v0 / volume) ** (2.0 / 3.0)
    t = eta - 1.0
    return e0 + (9.0 * v0 * b0 / 16.0) * (
        (t**3) * bp + (t**2) * (6.0 - 4.0 * eta)
    )


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    volumes, energies = [], []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                volumes.append(float(row["volume_A3"]))
                energies.append(float(row["energy_eV"]))
            except (KeyError, TypeError, ValueError):
                continue
    if len(volumes) < 4:
        raise ValueError("At least four valid volume/energy rows are required")
    order = np.argsort(volumes)
    return np.asarray(volumes)[order], np.asarray(energies)[order]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()
    volumes, energies = load_csv(args.csv)
    i_min = int(np.argmin(energies))
    initial = [energies[i_min], volumes[i_min], 150.0 / EV_TO_GPA, 4.0]
    bounds = ([-np.inf, 0.0, 0.0, 1.0], [np.inf, np.inf, np.inf, 10.0])
    params, covariance = curve_fit(
        birch_murnaghan,
        volumes,
        energies,
        p0=initial,
        bounds=bounds,
        maxfev=100000,
    )
    errors = np.sqrt(np.diag(covariance))
    e0, v0, b0, bp = params
    residuals = energies - birch_murnaghan(volumes, *params)
    rms = float(np.sqrt(np.mean(residuals**2)))
    in_range = volumes.min() <= v0 <= volumes.max()
    print(f"n = {len(volumes)}")
    print(f"E0_eV = {e0:.12g} +/- {errors[0]:.3g}")
    print(f"V0_A3 = {v0:.12g} +/- {errors[1]:.3g}")
    print(f"B0_eV_A3 = {b0:.12g} +/- {errors[2]:.3g}")
    print(f"B0_GPa = {b0 * EV_TO_GPA:.12g} +/- {errors[2] * EV_TO_GPA:.3g}")
    print(f"Bprime = {bp:.12g} +/- {errors[3]:.3g}")
    print(f"RMS_residual_eV = {rms:.12g}")
    print(f"V0_in_input_range = {'yes' if in_range else 'no'}")
    if not in_range:
        print(
            f"WARNING: fitted V0={v0:.12g} is outside input range "
            f"[{volumes.min():.12g}, {volumes.max():.12g}]"
        )


if __name__ == "__main__":
    main()
