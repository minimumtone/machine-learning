#!/usr/bin/env python3
"""Validation 1: CFSE textbook table reproduction.

Computes Crystal Field Stabilization Energy (CFSE) and Jahn-Teller
activity for d^0 – d^10 electron configurations in both high-spin and
low-spin octahedral coordination, and compares against authoritative
textbook values (Chemistry LibreTexts / Miessler-Tarr).

Usage:
    python examples/validation/validate_cfse.py
    python examples/validation/validate_cfse.py --output report.md

Reference:
    Miessler, G. L.; Tarr, D. A. "Inorganic Chemistry" (5th ed.)
    Chemistry LibreTexts §8.2.2, §11.02 — Crystal Field Stabilization Energy
    Chemistry LibreTexts §5.08 — Jahn-Teller Effect
"""

import argparse
import sys
from pathlib import Path

# Ensure t2vasp is importable when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from t2vasp.calculator import compute_cfse  # noqa: E402

# ── Textbook reference values ────────────────────────────────────────
# (cfse_over_delta, jt_strength)   jt: "strong" / "weak" / None
_TEXTBOOK_HS = [
    # d0   d1      d2      d3      d4       d5    d6      d7      d8      d9       d10
    (0.0,  None),   # d0
    (-0.4, "weak"), # d1
    (-0.8, "weak"), # d2
    (-1.2, None),   # d3
    (-0.6, "strong"),  # d4  HS — eg uneven (eg^1)
    (0.0,  None),   # d5  HS — half-filled
    (-0.4, "weak"), # d6  HS — t2g^4
    (-0.8, "weak"), # d7  HS — t2g^5
    (-1.2, None),   # d8
    (-0.6, "strong"),  # d9  — eg^3
    (0.0,  None),   # d10
]

_TEXTBOOK_LS = [
    (0.0,  None),     # d0
    (-0.4, "weak"),   # d1
    (-0.8, "weak"),   # d2
    (-1.2, None),     # d3
    (-1.6, "weak"),   # d4  LS — t2g^4 eg^0
    (-2.0, "weak"),   # d5  LS — t2g^5 eg^0
    (-2.4, None),     # d6  LS — t2g^6 eg^0 (filled t2g)
    (-1.8, "strong"), # d7  LS — t2g^6 eg^1 (eg uneven)
    (-1.2, None),     # d8
    (-0.6, "strong"), # d9
    (0.0,  None),     # d10
]


def validate_cfse_table(delta_oct: float = 1.0) -> list[dict]:
    """Run compute_cfse for d0–d10 and compare with textbook.

    Parameters
    ----------
    delta_oct : float
        Crystal field splitting (eV).  Using 1.0 makes CFSE numerically
        equal to the coefficient in units of Δ.

    Returns
    -------
    list of dicts with validation results per (d-count, spin) pair.
    """
    results = []
    for d_count in range(11):
        for spin_label, low_spin, textbook in [
            ("HS", False, _TEXTBOOK_HS),
            ("LS", True,  _TEXTBOOK_LS),
        ]:
            ref_cfse_delta, ref_jt = textbook[d_count]

            cfse_ev, cfse_delta, jt = compute_cfse(
                d_count, delta_oct, low_spin=low_spin, pairing_energy=0.0,
            )

            cfse_match = abs(cfse_delta - ref_cfse_delta) < 1e-10
            jt_match = jt == ref_jt

            results.append({
                "d_count": d_count,
                "spin": spin_label,
                "calc_cfse_delta": cfse_delta,
                "ref_cfse_delta": ref_cfse_delta,
                "cfse_match": cfse_match,
                "calc_jt": jt,
                "ref_jt": ref_jt,
                "jt_match": jt_match,
                "pass": cfse_match and jt_match,
            })
    return results


def format_console(results: list[dict]) -> str:
    """Format results as a plain-text table."""
    lines = []
    lines.append(f"{'d':>3}  {'Spin':>4}  {'CFSE(calc)':>10}  {'CFSE(ref)':>10}  "
                 f"{'JT(calc)':>10}  {'JT(ref)':>10}  {'Result':>6}")
    lines.append("-" * 72)
    for r in results:
        mark = "PASS" if r["pass"] else "FAIL"
        lines.append(
            f"d{r['d_count']:>2}  {r['spin']:>4}  "
            f"{r['calc_cfse_delta']:>10.2f}  {r['ref_cfse_delta']:>10.2f}  "
            f"{str(r['calc_jt']):>10}  {str(r['ref_jt']):>10}  "
            f"{mark:>6}"
        )
    n_pass = sum(1 for r in results if r["pass"])
    lines.append("-" * 72)
    lines.append(f"Total: {n_pass}/{len(results)} passed")
    return "\n".join(lines)


def format_markdown(results: list[dict]) -> str:
    """Format results as a Markdown report."""
    lines = []
    lines.append("# Validation 1: CFSE Textbook Table Reproduction")
    lines.append("")
    lines.append("Crystal Field Stabilization Energy (CFSE) and Jahn-Teller (JT)")
    lines.append("activity for d$^0$ – d$^{10}$ in octahedral coordination,")
    lines.append("compared against standard textbook values.")
    lines.append("")
    lines.append("**Reference**: Miessler & Tarr, *Inorganic Chemistry* (5th ed.);")
    lines.append("Chemistry LibreTexts §8.2.2, §5.08")
    lines.append("")
    lines.append("**Formula**: CFSE = n(t$_{2g}$) $\\times$ (−0.4Δ$_\\mathrm{oct}$)")
    lines.append("+ n(e$_g$) $\\times$ (+0.6Δ$_\\mathrm{oct}$)")
    lines.append("")

    # High-spin table
    lines.append("## High-Spin Configuration")
    lines.append("")
    lines.append("| d$^n$ | CFSE (calc) | CFSE (ref) | JT (calc) | JT (ref) | Result |")
    lines.append("|-------|-------------|------------|-----------|----------|--------|")
    for r in results:
        if r["spin"] != "HS":
            continue
        mark = "PASS" if r["pass"] else "**FAIL**"
        jt_c = r["calc_jt"] or "—"
        jt_r = r["ref_jt"] or "—"
        lines.append(
            f"| d$^{{{r['d_count']}}}$ | {r['calc_cfse_delta']:.1f}Δ "
            f"| {r['ref_cfse_delta']:.1f}Δ | {jt_c} | {jt_r} | {mark} |"
        )

    lines.append("")
    lines.append("## Low-Spin Configuration")
    lines.append("")
    lines.append("| d$^n$ | CFSE (calc) | CFSE (ref) | JT (calc) | JT (ref) | Result |")
    lines.append("|-------|-------------|------------|-----------|----------|--------|")
    for r in results:
        if r["spin"] != "LS":
            continue
        mark = "PASS" if r["pass"] else "**FAIL**"
        jt_c = r["calc_jt"] or "—"
        jt_r = r["ref_jt"] or "—"
        lines.append(
            f"| d$^{{{r['d_count']}}}$ | {r['calc_cfse_delta']:.1f}Δ "
            f"| {r['ref_cfse_delta']:.1f}Δ | {jt_c} | {jt_r} | {mark} |"
        )

    n_pass = sum(1 for r in results if r["pass"])
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"**{n_pass}/{len(results)}** tests passed.")
    if n_pass == len(results):
        lines.append("")
        lines.append("All CFSE values and Jahn-Teller predictions match textbook references.")
    else:
        failed = [r for r in results if not r["pass"]]
        lines.append("")
        lines.append("### Failures:")
        for r in failed:
            lines.append(
                f"- d$^{{{r['d_count']}}}$ {r['spin']}: "
                f"CFSE calc={r['calc_cfse_delta']:.2f}Δ vs ref={r['ref_cfse_delta']:.2f}Δ, "
                f"JT calc={r['calc_jt']} vs ref={r['ref_jt']}"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate CFSE against textbook")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Write Markdown report to file")
    args = parser.parse_args()

    results = validate_cfse_table(delta_oct=1.0)
    print(format_console(results))

    if args.output:
        md = format_markdown(results)
        Path(args.output).write_text(md, encoding="utf-8")
        print(f"\nMarkdown report written to: {args.output}")

    n_fail = sum(1 for r in results if not r["pass"])
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
