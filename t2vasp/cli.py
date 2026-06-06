"""
Command-line interface for t2vasp (Text-to-VASP).

Subcommands
-----------
run        : Generate VASP input files from a natural-language query.
parse      : Parse a single VASP directory and print a summary.
batch      : Batch-process all calculations under a directory.
generate   : Generate candidate structures from a reference POSCAR.
optimise   : Run the iterative optimisation loop (VASP + analyse + generate).

Usage
-----
    python -m t2vasp run "Ni3AlのL12構造を最適化して" -o calc_Ni3Al/
    python -m t2vasp run "BaTiO3の自発分極" -o calc_BTO/ --scheduler slurm
    python -m t2vasp run "Cu酸化物の結晶場分裂" --dry-run
    python -m t2vasp parse /path/to/calc
    python -m t2vasp batch /path/to/all_calcs --output results/
    python -m t2vasp generate CONTCAR --strains -0.02,-0.01,0.01,0.02
    python -m t2vasp optimise /path/to/initial --max-iter 5
"""

import argparse
import sys
from pathlib import Path

from . import __version__
from .config import load_config
from .utils import setup_logging


def _build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="t2vasp",
        description="Text-to-VASP: natural language \u2192 VASP calculation setup & post-processing.",
    )
    root.add_argument("-V", "--version", action="version", version=f"t2vasp {__version__}")
    root.add_argument("-v", "--verbose", action="count", default=1,
                      help="Increase verbosity (-vv for debug).")
    root.add_argument("-q", "--quiet", action="store_true",
                      help="Suppress informational output.")
    root.add_argument("--config", type=str, default=None,
                      help="Path to user YAML config (overrides defaults).")

    sub = root.add_subparsers(dest="command", required=True)

    # -- run -----------------------------------------------------------
    p_run = sub.add_parser("run",
                           help="Generate VASP inputs from natural language.")
    p_run.add_argument("query", type=str,
                       help='Natural-language query, e.g. "Ni3AlのL12構造を最適化して"')
    p_run.add_argument("-o", "--output", type=str, default=None,
                       help="Output directory (auto-generated if omitted).")
    p_run.add_argument("--scheduler", type=str, default="slurm",
                       choices=["slurm", "pbs", "local"],
                       help="Job scheduler (default: slurm).")
    p_run.add_argument("--dry-run", action="store_true",
                       help="Show plan without writing files.")

    # -- parse ---------------------------------------------------------
    p_parse = sub.add_parser("parse", help="Parse a single VASP directory.")
    p_parse.add_argument("calc_dir", type=str, help="Path to calculation directory.")

    # -- batch ---------------------------------------------------------
    p_batch = sub.add_parser("batch", help="Batch-process VASP calculations.")
    p_batch.add_argument("base_dir", type=str, help="Root directory.")
    p_batch.add_argument("-o", "--output", type=str, default=None,
                         help="Output directory.")
    p_batch.add_argument("--format", nargs="+", default=["csv", "json", "summary"],
                         choices=["csv", "json", "summary"],
                         help="Export formats.")

    # -- generate ------------------------------------------------------
    p_gen = sub.add_parser("generate", help="Generate candidate POSCARs.")
    p_gen.add_argument("poscar", type=str, help="Reference POSCAR/CONTCAR.")
    p_gen.add_argument("--strains", type=str, default="-0.02,-0.01,0.01,0.02",
                       help="Comma-separated strain values.")
    p_gen.add_argument("-o", "--output", type=str, default="candidates",
                       help="Output directory.")

    # -- optimise ------------------------------------------------------
    p_opt = sub.add_parser("optimise", help="Iterative optimisation loop.")
    p_opt.add_argument("initial_dir", type=str, help="Initial calculation dir.")
    p_opt.add_argument("--max-iter", type=int, default=5)
    p_opt.add_argument("--vasp-cmd", type=str, default=None)

    # -- plot ----------------------------------------------------------
    p_plot = sub.add_parser("plot", help="Generate plots from a results JSON.")
    p_plot.add_argument("json_file", type=str, help="Path to results.json.")
    p_plot.add_argument("-o", "--output", type=str, default="plots",
                        help="Output directory for figures.")

    return root


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m t2vasp``."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    verbosity = 0 if args.quiet else args.verbose
    setup_logging(verbosity)
    cfg = load_config(args.config)

    if args.command == "run":
        import yaml as _yaml
        from .intent import classify
        from .entity import extract
        from .generator import generate as gen_vasp

        intent = classify(args.query)
        entity = extract(args.query)

        # Auto-generate output dir name if not specified
        if args.output:
            out_dir = Path(args.output)
        else:
            formula = entity.formula_str or "_".join(entity.elements) or "calc"
            out_dir = Path(f"calc_{formula}_{intent.calc_type}")

        if args.dry_run:
            plan = gen_vasp(intent, entity, out_dir, scheduler=args.scheduler,
                            dry_run=True)
            print(_yaml.dump(plan, default_flow_style=False, allow_unicode=True))
        else:
            plan = gen_vasp(intent, entity, out_dir, scheduler=args.scheduler)
            print(f"Generated VASP inputs in: {out_dir}/")
            print(f"  Calculation: {intent.calc_type}")
            print(f"  Formula:     {plan['formula']}")
            print(f"  Prototype:   {plan['prototype']}")
            print(f"  Lattice:     {plan['lattice_constant_angstrom']} Å")
            print(f"  K-points:    {plan['kpoints']}")
            print(f"  Scheduler:   {args.scheduler}")
            print(f"  Files:       {', '.join(plan['files'])}")
            if plan.get("secondary_steps"):
                print(f"  Multi-step:  {plan['secondary_steps']}")

    elif args.command == "parse":
        from .pipeline import process_single
        from .exporter import export_summary
        result = process_single(args.calc_dir, cfg)
        print(export_summary([result]))

    elif args.command == "batch":
        from .pipeline import process_batch
        process_batch(args.base_dir, cfg,
                      output_dir=args.output,
                      export_formats=args.format)

    elif args.command == "generate":
        from .structure import generate_candidates
        from ase.io import read as ase_read
        atoms = ase_read(args.poscar, format="vasp")
        strains = [float(s) for s in args.strains.split(",")]
        paths = generate_candidates(atoms, strain_values=strains,
                                    output_dir=args.output)
        for p in paths:
            print(p)

    elif args.command == "optimise":
        from .pipeline import optimisation_loop
        results = optimisation_loop(
            args.initial_dir, cfg,
            max_iterations=args.max_iter,
            vasp_command=args.vasp_cmd,
        )
        from .exporter import export_summary
        print(export_summary(results))

    elif args.command == "plot":
        import json
        from .calculator import CalculationResult, EnergyResult, StructureResult
        from .visualizer import plot_energy_comparison, plot_lattice_comparison

        data = json.loads(Path(args.json_file).read_text())
        results = []
        for item in data:
            cr = CalculationResult(label=item.get("label", ""))
            cr.converged = item.get("converged", False)
            e = item.get("energy")
            if e:
                cr.energy = EnergyResult(**{k: v for k, v in e.items()
                                            if k in EnergyResult.__dataclass_fields__})
            s = item.get("structure")
            if s:
                cr.structure = StructureResult(**{k: v for k, v in s.items()
                                                  if k in StructureResult.__dataclass_fields__})
            results.append(cr)

        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        plot_energy_comparison(results, save_path=out / "energy_comparison.png")
        plot_lattice_comparison(results, save_path=out / "lattice_comparison.png")
        print(f"Plots saved to {out}")

    return 0
