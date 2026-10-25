#!/usr/bin/env python3
"""SQS体積サイズファクターの組成依存を測定する。

x=0.25/0.50/0.75の3組成で緩和済みSQSの体積を比較し、
Alonso式が内包する x(1-x) 対称形（正則溶液型）の妥当性と
Redlich--Kister一次項（非対称性）の大きさを定量する。

超過体積 dV(x) = V_alloy(x) - [(1-x) V_A + x V_B]
       dV(x) = x(1-x) [a0 + a1 (1-2x)]
x は組成式でアルファベット順第2元素の原子分率。
a0 は対称項、a1 は非対称項（Alonso式では表現できない成分）。

出力:
  paper/composition_metrics.json
  paper/results_composition_dependence.csv
  paper/fig_omega_composition.png
  paper/fig_quadratic_form.png
  paper/fig_asymmetry.png
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
DATA = ROOT / "data" / "sqs_results.csv"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PAPER))
from generate_all_figures import EXCLUDE_ELEMENTS, load_sqs_data  # noqa: E402
from detect_unrelaxed_volumes import flagged_row  # noqa: E402

plt.rcParams.update({
    "font.size": 20,
    "axes.labelsize": 22,
    "axes.titlesize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 17,
    "figure.dpi": 130,
    "font.family": "DejaVu Sans",
})
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]

DIR_PATTERN = re.compile(r"^([A-Z][a-z]?)(\d+)([A-Z][a-z]?)(\d+)$")

# 構造ごとの (structure_root, natoms, {(n_A, n_B): x_B})
SPECS = {
    "BCC": {
        "root": "BCC_SQS",
        "natoms": 16,
        "lattice": "bcc",
        "compositions": {(12, 4): 0.25, (8, 8): 0.50, (4, 12): 0.75},
        "pure_counts": (8, 8),
    },
    "FCC": {
        "root": "FCC_SQS",
        "natoms": 32,
        "lattice": "fcc",
        "compositions": {(24, 8): 0.25, (16, 16): 0.50, (8, 24): 0.75},
        "pure_counts": (16, 16),
    },
}

XS = (0.25, 0.50, 0.75)


def load_rows() -> list[dict[str, str]]:
    with DATA.open(newline="") as handle:
        return list(csv.DictReader(handle))


def flagged_keys(rows) -> set[tuple[str, str, int]]:
    return {
        (f["dir"], f["structure_root"], int(f["natoms"]))
        for row in rows
        if (f := flagged_row(row)) is not None
    }


def collect(rows, spec, flagged) -> tuple[dict, dict, dict]:
    """緩和済み・非フラグ行から純元素体積と組成別体積を集める。

    Returns (pure_raw, alloy, counts)
      pure_raw: {element: V/atom}
      alloy: {(A, B): {x: V/atom}}  (A, B はアルファベット順)
      counts: 集計情報
    """
    best: dict[tuple[str, int, int], tuple[float, float]] = {}
    counts = defaultdict(int)
    for row in rows:
        if row["structure_root"] != spec["root"]:
            continue
        if row["status"] != "OK" or row.get("relax_converged") != "yes":
            counts["skip_unconverged"] += 1
            continue
        if row["lattice_type"] != spec["lattice"]:
            continue
        try:
            natoms = int(row["natoms"])
            volume = float(row["volume_A3"])
        except (ValueError, KeyError):
            continue
        if natoms != spec["natoms"] or volume <= 0:
            continue
        match = DIR_PATTERN.fullmatch(row["dir"])
        if match is None:
            continue
        if (row["dir"], row["structure_root"], natoms) in flagged:
            counts["skip_unrelaxed"] += 1
            continue
        el_a, n_a, el_b, n_b = match.groups()
        n_a, n_b = int(n_a), int(n_b)
        try:
            energy = float(row["energy_eV"]) / natoms
        except (ValueError, KeyError):
            energy = float("inf")
        key = (row["dir"], n_a, n_b)
        v_per_atom = volume / natoms
        if key not in best or energy < best[key][0]:
            if key in best:
                counts["duplicate_dirs"] += 1
            best[key] = (energy, v_per_atom)

    pure_raw: dict[str, float] = {}
    alloy: dict[tuple[str, str], dict[float, float]] = defaultdict(dict)
    for (dirname, n_a, n_b), (_energy, v_per_atom) in best.items():
        el_a, _, el_b, _ = DIR_PATTERN.fullmatch(dirname).groups()
        if el_a == el_b:
            if (n_a, n_b) == spec["pure_counts"]:
                pure_raw[el_a] = v_per_atom
            continue
        if el_a in EXCLUDE_ELEMENTS or el_b in EXCLUDE_ELEMENTS:
            counts["skip_excluded_element"] += 1
            continue
        x_b = spec["compositions"].get((n_a, n_b))
        if x_b is None:
            continue
        pair = (el_a, el_b) if el_a < el_b else (el_b, el_a)
        # x はアルファベット順第2元素の分率
        x = x_b if el_b == pair[1] else 1.0 - x_b
        prev = alloy[pair].get(x)
        if prev is not None and abs(prev - v_per_atom) > 1e-9:
            counts["conflicting_direction_rows"] += 1
            # 同一組成が両方向のディレクトリ名で存在する場合は平均を取らず
            # 小さい体積側（低エネルギー緩和側）を保持する
            v_per_atom = min(prev, v_per_atom)
        alloy[pair][x] = v_per_atom
        counts["alloy_rows_used"] += 1
    return pure_raw, dict(alloy), dict(counts)


def fit_pair(v_alloy: dict[float, float], v_a: float, v_b: float) -> dict:
    """3組成から対称項a0と非対称項a1を求める。"""
    out: dict[str, float] = {}
    dv = {}
    for x in XS:
        v_veg = (1.0 - x) * v_a + x * v_b
        dv[x] = v_alloy[x] - v_veg
        out[f"omega_{int(x*100)}"] = (v_alloy[x] - v_veg) / v_veg
        out[f"dv_{int(x*100)}"] = dv[x]
    # x(1-x): 0.1875 (x=0.25, 0.75), 0.25 (x=0.50)
    out["a0_mid"] = dv[0.50] / 0.25
    out["a0_ends"] = (dv[0.25] + dv[0.75]) / (2 * 0.1875)
    out["a1"] = (dv[0.25] - dv[0.75]) / 0.1875
    # 最小二乗（3点、2パラメータ）
    design = np.array([[x * (1 - x), x * (1 - x) * (1 - 2 * x)] for x in XS])
    target = np.array([dv[x] for x in XS])
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    out["a0_ls"], out["a1_ls"] = float(coef[0]), float(coef[1])
    resid_rk = target - design @ coef
    coef0 = float(np.linalg.lstsq(design[:, :1], target, rcond=None)[0][0])
    resid_sym = target - design[:, :1] @ np.array([coef0])
    out["a0_sym_only"] = coef0
    out["rss_symmetric"] = float(np.sum(resid_sym ** 2))
    out["rss_rk"] = float(np.sum(resid_rk ** 2))
    out["v_vegard_mid"] = 0.5 * (v_a + v_b)
    # 50:50で校正した対称形 dV_pred(x)=x(1-x)a0_mid を x=0.25/0.75 へ外挿した
    # ときの誤差（Ω_sf単位）。Alonso式の関数形の転用可能性を直接測る。
    for x in (0.25, 0.75):
        v_veg = (1.0 - x) * v_a + x * v_b
        dv_pred = x * (1.0 - x) * out["a0_mid"]
        out[f"form_error_{int(x*100)}"] = (dv[x] - dv_pred) / v_veg
    out["omega_asymmetry"] = out["omega_25"] - out["omega_75"]
    return out


def analyze(
    structure: str, rows, flagged, pure_curated: dict[str, float]
) -> dict:
    spec = SPECS[structure]
    pure_raw, alloy, counts = collect(rows, spec, flagged)
    records = []
    for pair, v_alloy in sorted(alloy.items()):
        if not all(x in v_alloy for x in XS):
            continue
        el_a, el_b = pair
        for label, table in (("curated", pure_curated), ("raw", pure_raw)):
            if el_a not in table or el_b not in table:
                continue
            fit = fit_pair(v_alloy, table[el_a], table[el_b])
            records.append({
                "pair": f"{el_a}-{el_b}",
                "element_A": el_a,
                "element_B": el_b,
                "endpoint": label,
                **{k: v for k, v in fit.items()},
            })
    return {"records": records, "counts": counts,
            "n_pairs_3comp": sum(1 for v in alloy.values()
                                 if all(x in v for x in XS)),
            "n_pairs_any": len(alloy),
            "pure_raw": pure_raw}


def summarize(records: list[dict], endpoint: str) -> dict:
    sel = [r for r in records if r["endpoint"] == endpoint]
    if not sel:
        return {}
    a0 = np.array([r["a0_mid"] for r in sel])
    a0_ends = np.array([r["a0_ends"] for r in sel])
    a1 = np.array([r["a1"] for r in sel])
    om = {x: np.array([r[f"omega_{int(x*100)}"] for r in sel]) for x in XS}
    nonzero = np.abs(a0) > 0.05  # a0 が小さいペアで比が発散するのを避ける
    ratio_form = a0_ends[nonzero] / a0[nonzero]
    ratio_asym = a1[nonzero] / a0[nonzero]
    r_25, _ = pearsonr(om[0.25], om[0.50])
    r_75, _ = pearsonr(om[0.75], om[0.50])
    rho_25, _ = spearmanr(om[0.25], om[0.50])
    sign_ok = sum(
        1 for r in sel
        if (
            np.sign(r["omega_25"])
            == np.sign(r["omega_50"])
            == np.sign(r["omega_75"])
        )
    )
    rss_sym = np.array([r["rss_symmetric"] for r in sel])
    rss_rk = np.array([r["rss_rk"] for r in sel])
    return {
        "n_pairs": len(sel),
        "n_pairs_a0_nonzero": int(nonzero.sum()),
        "median_omega_25_pct": float(np.median(om[0.25]) * 100),
        "median_omega_50_pct": float(np.median(om[0.50]) * 100),
        "median_omega_75_pct": float(np.median(om[0.75]) * 100),
        "median_abs_omega_25_pct": float(np.median(np.abs(om[0.25])) * 100),
        "median_abs_omega_50_pct": float(np.median(np.abs(om[0.50])) * 100),
        "median_abs_omega_75_pct": float(np.median(np.abs(om[0.75])) * 100),
        "pearson_omega25_vs_omega50": float(r_25),
        "pearson_omega75_vs_omega50": float(r_75),
        "spearman_omega25_vs_omega50": float(rho_25),
        "sign_consistent_pairs": int(sign_ok),
        "sign_consistent_frac": float(sign_ok / len(sel)),
        "median_a0_ends_over_a0_mid": float(np.median(ratio_form)),
        "iqr_a0_ends_over_a0_mid": [
            float(np.percentile(ratio_form, 25)),
            float(np.percentile(ratio_form, 75)),
        ],
        "frac_form_within_20pct": float(
            np.mean(np.abs(ratio_form - 1.0) <= 0.20)
        ),
        "median_abs_a1_over_a0": float(np.median(np.abs(ratio_asym))),
        "p90_abs_a1_over_a0": float(np.percentile(np.abs(ratio_asym), 90)),
        "frac_asym_over_20pct": float(np.mean(np.abs(ratio_asym) > 0.20)),
        "median_rss_reduction_rk": float(
            np.median(1.0 - rss_rk[rss_sym > 0] / rss_sym[rss_sym > 0])
        ),
        "a0_threshold_A3": 0.05,
        "median_abs_form_error_25_pct": float(
            np.median(np.abs([r["form_error_25"] for r in sel])) * 100
        ),
        "median_abs_form_error_75_pct": float(
            np.median(np.abs([r["form_error_75"] for r in sel])) * 100
        ),
        "p90_abs_form_error_25_pct": float(
            np.percentile(np.abs([r["form_error_25"] for r in sel]), 90) * 100
        ),
        "p90_abs_form_error_75_pct": float(
            np.percentile(np.abs([r["form_error_75"] for r in sel]), 90) * 100
        ),
        "median_abs_omega_asymmetry_pct": float(
            np.median(np.abs([r["omega_asymmetry"] for r in sel])) * 100
        ),
        "p90_abs_omega_asymmetry_pct": float(
            np.percentile(
                np.abs([r["omega_asymmetry"] for r in sel]), 90
            ) * 100
        ),
        "max_abs_omega_asymmetry_pct": float(
            np.max(np.abs([r["omega_asymmetry"] for r in sel])) * 100
        ),
        "max_asymmetry_pair": max(
            sel, key=lambda r: abs(r["omega_asymmetry"])
        )["pair"],
    }


def bcc_x25_cellsize(rows, flagged) -> dict:
    """x=0.25の非等量組成で 16原子(12:4) と 128原子(96:32) を比較する。

    各セルサイズ自身の純元素端点（16原子: A8A8、128原子: A64A64）を
    DFT Vegard基準に使う。未緩和フラグ行は除外する。
    """
    def gather(natoms, pure_counts, alloy_counts):
        best = {}
        for row in rows:
            if row["structure_root"] != "BCC_SQS":
                continue
            if row["status"] != "OK" or row.get("relax_converged") != "yes":
                continue
            try:
                n = int(row["natoms"])
                volume = float(row["volume_A3"])
                energy = float(row["energy_eV"]) / n
            except (ValueError, KeyError):
                continue
            if n != natoms or volume <= 0:
                continue
            if (row["dir"], row["structure_root"], n) in flagged:
                continue
            match = DIR_PATTERN.fullmatch(row["dir"])
            if match is None:
                continue
            el_a, n_a, el_b, n_b = match.groups()
            key = (row["dir"], int(n_a), int(n_b))
            if key not in best or energy < best[key][0]:
                best[key] = (energy, volume / n)
        pure, alloy = {}, {}
        for (dirname, n_a, n_b), (_e, v) in best.items():
            el_a, _, el_b, _ = DIR_PATTERN.fullmatch(dirname).groups()
            if el_a == el_b and (n_a, n_b) == pure_counts:
                pure[el_a] = v
            elif el_a != el_b and (n_a, n_b) == alloy_counts:
                if el_a in EXCLUDE_ELEMENTS or el_b in EXCLUDE_ELEMENTS:
                    continue
                # majority = el_a (x_B = 0.25)
                alloy[(el_a, el_b)] = v
        return pure, alloy

    pure16, alloy16 = gather(16, (8, 8), (12, 4))
    pure128, alloy128 = gather(128, (64, 64), (96, 32))

    def omega(v_alloy, v_major, v_minor):
        v_veg = 0.75 * v_major + 0.25 * v_minor
        return (v_alloy - v_veg) / v_veg

    common = []
    for key, v128 in sorted(alloy128.items()):
        major, minor = key
        if key not in alloy16:
            continue
        if not all(e in pure16 and e in pure128 for e in (major, minor)):
            continue
        o16 = omega(alloy16[key], pure16[major], pure16[minor])
        o128 = omega(v128, pure128[major], pure128[minor])
        common.append({
            "composition": f"{major}0.75{minor}0.25",
            "omega_16_pct": o16 * 100,
            "omega_128_pct": o128 * 100,
            "delta_pct": (o128 - o16) * 100,
        })
    deltas = (
        np.array([c["delta_pct"] for c in common])
        if common else np.array([])
    )
    return {
        "n_128_x25_rows": len(alloy128),
        "n_common": len(common),
        "median_abs_delta_omega_pct": (
            float(np.median(np.abs(deltas))) if common else None
        ),
        "max_abs_delta_omega_pct": (
            float(np.max(np.abs(deltas))) if common else None
        ),
        "max_composition": (
            max(common, key=lambda c: abs(c["delta_pct"]))["composition"]
            if common else None
        ),
        "sign_reversals": [
            c["composition"] for c in common
            if np.sign(c["omega_16_pct"]) != np.sign(c["omega_128_pct"])
        ],
        "pairs": common,
    }


def fig_omega_composition(bcc: list[dict], fcc: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(17, 7.5))
    for ax, recs, title in (
        (axes[0], bcc, "BCC SQS (16-atom, curated endpoints)"),
        (axes[1], fcc, "FCC SQS (32-atom, curated endpoints)"),
    ):
        sel = [r for r in recs if r["endpoint"] == "curated"]
        sel.sort(key=lambda r: abs(r["omega_50"]), reverse=True)
        for r in sel:
            y = [
                r["omega_25"] * 100,
                r["omega_50"] * 100,
                r["omega_75"] * 100,
            ]
            ax.plot(XS, y, "-", color="0.75", lw=0.7, alpha=0.5, zorder=1)
        for r in sel[:6]:
            y = [
                r["omega_25"] * 100,
                r["omega_50"] * 100,
                r["omega_75"] * 100,
            ]
            ax.plot(XS, y, "o-", lw=2.6, ms=9, zorder=3, label=r["pair"])
        med = [
            np.median([
                r[f"omega_{int(x*100)}"] for r in sel
            ]) * 100 for x in XS
        ]
        ax.plot(XS, med, "k--", lw=3.2, ms=12, marker="s", zorder=4,
                label="Median")
        ax.axhline(0, color="k", lw=1.0)
        ax.set_xlabel(
            "$x$ (atomic fraction of 2nd element, alphabetical)"
        )
        ax.set_ylabel(r"$\Omega_\mathrm{sf}(x)$ (%)")
        ax.set_title(f"{title}  $n$={len(sel)} pairs")
        ax.set_xticks(XS)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=4,
            framealpha=0.9,
        )
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PAPER / "fig_omega_composition.png", bbox_inches="tight")
    plt.close(fig)


def fig_quadratic_form(bcc: list[dict], fcc: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(17, 7.5))
    for ax, recs, title in (
        (axes[0], bcc, "BCC SQS (16-atom, curated endpoints)"),
        (axes[1], fcc, "FCC SQS (32-atom, curated endpoints)"),
    ):
        sel = [r for r in recs if r["endpoint"] == "curated"]
        x = np.array([r["a0_mid"] for r in sel])
        y = np.array([r["a0_ends"] for r in sel])
        ax.scatter(x, y, s=55, alpha=0.65, edgecolor="k", linewidth=0.4)
        lim = [min(x.min(), y.min()), max(x.max(), y.max())]
        pad = 0.05 * (lim[1] - lim[0])
        lim = [lim[0] - pad, lim[1] + pad]
        ax.plot(
            lim, lim, "k--", lw=2.0,
            label="$a_0^\\mathrm{ends}=a_0^\\mathrm{mid}$",
        )
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel(r"$a_0^\mathrm{mid}=\Delta V(0.5)/0.25$ (Å$^3$/atom)")
        ax.set_ylabel(
            r"$a_0^\mathrm{ends}$ (from $x$=0.25 and 0.75) (Å$^3$/atom)"
        )
        r, _ = pearsonr(x, y)
        slope = float(np.polyfit(x, y, 1)[0])
        ax.set_title(f"{title}\n$r$={r:.3f}, slope={slope:.3f}")
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PAPER / "fig_quadratic_form.png", bbox_inches="tight")
    plt.close(fig)


def fig_asymmetry(bcc: list[dict], fcc: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(17, 7.5))
    for ax, recs, title in (
        (axes[0], bcc, "BCC SQS (16-atom, curated endpoints)"),
        (axes[1], fcc, "FCC SQS (32-atom, curated endpoints)"),
    ):
        sel = [r for r in recs if r["endpoint"] == "curated"]
        x = np.array([r["omega_25"] for r in sel]) * 100
        y = np.array([r["omega_75"] for r in sel]) * 100
        ax.scatter(x, y, s=55, alpha=0.65, edgecolor="k", linewidth=0.4)
        lim = [min(x.min(), y.min()), max(x.max(), y.max())]
        pad = 0.05 * (lim[1] - lim[0])
        lim = [lim[0] - pad, lim[1] + pad]
        ax.plot(
            lim, lim, "k--", lw=2.0,
            label="symmetric-form expectation ($y=x$)",
        )
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel(r"$\Omega_\mathrm{sf}(x=0.25)$ (%)")
        ax.set_ylabel(r"$\Omega_\mathrm{sf}(x=0.75)$ (%)")
        r, _ = pearsonr(x, y)
        ax.set_title(f"{title}\n$r$={r:.3f}, $n$={len(sel)}")
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PAPER / "fig_asymmetry.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = load_rows()
    flagged = flagged_keys(rows)
    sqs = load_sqs_data()
    bcc = analyze("BCC", rows, flagged, sqs["pure_vol"])
    fcc = analyze("FCC", rows, flagged, sqs["fcc_pure_vol"])

    # 既存の50:50 Ω_sf との整合チェック
    consistency = {}
    for label, res, ref in (("BCC", bcc, sqs["omega_dft"]),
                            ("FCC", fcc, sqs["fcc_omega_dft"])):
        diffs = []
        for r in res["records"]:
            if r["endpoint"] != "curated":
                continue
            key = (r["element_A"], r["element_B"])
            if key in ref:
                diffs.append(abs(r["omega_50"] - ref[key]))
        consistency[label] = {
            "n_compared": len(diffs),
            "max_abs_diff_omega50": float(max(diffs)) if diffs else None,
        }

    metrics = {
        "BCC": {
            "n_pairs_3compositions": bcc["n_pairs_3comp"],
            "n_pairs_any_composition": bcc["n_pairs_any"],
            "row_counts": bcc["counts"],
            "curated_endpoints": summarize(bcc["records"], "curated"),
            "raw_endpoints": summarize(bcc["records"], "raw"),
        },
        "FCC": {
            "n_pairs_3compositions": fcc["n_pairs_3comp"],
            "n_pairs_any_composition": fcc["n_pairs_any"],
            "row_counts": fcc["counts"],
            "curated_endpoints": summarize(fcc["records"], "curated"),
            "raw_endpoints": summarize(fcc["records"], "raw"),
        },
        "consistency_with_paper_omega50": consistency,
        "bcc_x25_cellsize_16_vs_128": bcc_x25_cellsize(rows, flagged),
    }

    fig_omega_composition(bcc["records"], fcc["records"])
    fig_quadratic_form(bcc["records"], fcc["records"])
    fig_asymmetry(bcc["records"], fcc["records"])

    fields = ["structure", "pair", "element_A", "element_B", "endpoint",
              "omega_25", "omega_50", "omega_75",
              "dv_25", "dv_50", "dv_75",
              "a0_mid", "a0_ends", "a1", "a0_ls", "a1_ls",
              "rss_symmetric", "rss_rk", "v_vegard_mid",
              "form_error_25", "form_error_75", "omega_asymmetry"]
    with (PAPER / "results_composition_dependence.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for structure, res in (("BCC", bcc), ("FCC", fcc)):
            for r in res["records"]:
                writer.writerow({"structure": structure,
                                 **{k: r[k] for k in fields if k in r}})

    with (PAPER / "composition_metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
