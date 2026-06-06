"""
Result exporter — CSV, JSON, and plain-text summary.

All export functions accept a list of :class:`CalculationResult` and write
to a specified output path.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .calculator import CalculationResult

logger = logging.getLogger(__name__)


# ── Flatten nested result dict ───────────────────────────────────────
def _flatten(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    items: List = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# ── CSV export ───────────────────────────────────────────────────────
def export_csv(
    results: List[CalculationResult],
    output_path: str | Path,
    delimiter: str = ",",
    precision: int = 6,
) -> Path:
    """Write results to a CSV file.

    Parameters
    ----------
    results : list[CalculationResult]
    output_path : str or Path
    delimiter : str
    precision : int
        Number of decimal places for floats.

    Returns
    -------
    Path
        Resolved path of the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [_flatten(r.as_dict()) for r in results]
    all_keys = list(dict.fromkeys(k for row in rows for k in row))

    with open(output_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=all_keys, delimiter=delimiter,
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            formatted = {}
            for k, v in row.items():
                if isinstance(v, float):
                    formatted[k] = f"{v:.{precision}f}"
                else:
                    formatted[k] = v
            writer.writerow(formatted)

    logger.info("CSV written: %s (%d records)", output_path, len(results))
    return output_path.resolve()


# ── JSON export ──────────────────────────────────────────────────────
def export_json(
    results: List[CalculationResult],
    output_path: str | Path,
    indent: int = 2,
) -> Path:
    """Write results to a JSON file.

    Parameters
    ----------
    results : list[CalculationResult]
    output_path : str or Path
    indent : int

    Returns
    -------
    Path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = [r.as_dict() for r in results]

    def _default(o: Any) -> Any:
        import numpy as np
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    with open(output_path, "w") as fh:
        json.dump(payload, fh, indent=indent, default=_default, ensure_ascii=False)

    logger.info("JSON written: %s (%d records)", output_path, len(results))
    return output_path.resolve()


# ── Plain-text summary ──────────────────────────────────────────────
def export_summary(
    results: List[CalculationResult],
    output_path: str | Path | None = None,
) -> str:
    """Generate a human-readable summary table.

    Returns the summary text and optionally writes it to *output_path*.
    """
    lines: List[str] = []
    header = (
        f"{'Label':<30s} {'E/atom (eV)':>12s} {'a (Å)':>8s} "
        f"{'Vol/at (ų)':>12s} {'Conv':>5s}"
    )
    lines.append("=" * len(header))
    lines.append("t2vasp Analysis Summary")
    lines.append("=" * len(header))
    lines.append(header)
    lines.append("-" * len(header))

    for r in results:
        epa = f"{r.energy.energy_per_atom:.4f}" if r.energy else "N/A"
        a_val = f"{r.structure.lattice_constant:.3f}" if r.structure else "N/A"
        vpa = f"{r.structure.volume_per_atom:.3f}" if r.structure else "N/A"
        conv = "Yes" if r.converged else "No"
        lbl = r.label[-28:] if len(r.label) > 28 else r.label
        lines.append(f"{lbl:<30s} {epa:>12s} {a_val:>8s} {vpa:>12s} {conv:>5s}")

    lines.append("-" * len(header))
    lines.append(f"Total records: {len(results)}")
    converged_count = sum(1 for r in results if r.converged)
    lines.append(f"Converged: {converged_count}/{len(results)}")
    lines.append("=" * len(header))

    text = "\n".join(lines)

    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        logger.info("Summary written: %s", p)

    return text
