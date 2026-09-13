"""化学系単位の交差検証・permutation対照・経験的ベースライン。

査読指摘 I（校正/テストの改善率逆転＝テスト集合の実効独立性）、
II（モデル選択のテストセット上でのリーク）、
IX-1（Omega_sf permutation test）、
IX-2（元素別有効原子体積 V_i^eff の経験的ベースライン、|dr| 代替）に対応する。

評価の考え方
------------
独立テスト31合金は文献系統（Tseng 2019 の Hf-Mo-Nb-Ta-Ti-Zr 系統除去、
Freudenberger 2017 の Au-Cu-Ni-Pd-Pt 系統除去など）が主体で、
合金単位では独立標本になっていない。6元素から作る部分集合は
10ペア中6〜9ペアを共有するため残差が強く相関する。
そこで合金単位ではなく化学系単位で fold を切る。

  - element_set : 元素集合が厳密に一致する合金を1群とする
  - family      : 元素集合の Jaccard 類似度 >= FAMILY_JACCARD の
                  単連結クラスタを1群とする（Tseng の部分集合群は1群になる）
  - holdout     : 特定文献セット（Tseng / Freudenberger / Chen ...）を丸ごと抜く

いずれの fold でも q（gamma）は訓練 fold 内でのみ最適化し、
評価 fold には一切触れない。

出力
----
paper/robust_validation.json  : 全数値の単一ソース
paper/results_robust_cv.csv   : 合金ごとの out-of-fold 予測
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy import stats

OUTDIR = Path(__file__).resolve().parent
REPO = OUTDIR.parent
sys.path.insert(0, str(REPO))

from hea_lattice_xgboost import (  # noqa: E402
    KING_ATOMIC_VOLUMES, ALONSO_TABLE2, INDEPENDENT_TEST,
    compute_vegard, compute_eq10_scaled,
)
sys.path.insert(0, str(OUTDIR))
from generate_all_figures import (  # noqa: E402
    load_compounds, compute_omega_sf_pairwise, load_sqs_data,
)

# 元素集合の Jaccard 類似度がこれ以上なら同一化学系ファミリとみなす
FAMILY_JACCARD = 0.5
# 組成比がこれ未満の元素は化学系の同定には使わない（微量添加元素）
COMP_EPS = 0.02
# permutation test の反復数
N_PERM = 500
RNG_SEED = 20260602


# ---------------------------------------------------------------------------
# データ準備
# ---------------------------------------------------------------------------
def n_auc(struct: str) -> int:
    return 4 if struct == "FCC" else 2


def v_per_atom(a: float, struct: str) -> float:
    return a ** 3 / n_auc(struct)


def a_from_v(v: float, struct: str) -> float:
    return (v * n_auc(struct)) ** (1 / 3)


def major_elements(comp: dict) -> frozenset:
    tot = sum(comp.values())
    return frozenset(e for e, c in comp.items() if c / tot >= COMP_EPS)


def build_pool() -> list[dict]:
    """校正64＋独立テスト31＝95合金を1つのプールにまとめる。"""
    pool = []
    for i, h in enumerate(ALONSO_TABLE2):
        pool.append({
            "comp": h["comp"], "struct": h["struct"], "a_exp": h["a_exp"],
            "ref": "Alonso2005_Table2", "origin": "calibration",
            "idx": i, "eset": major_elements(h["comp"]),
        })
    for i, h in enumerate(INDEPENDENT_TEST):
        pool.append({
            "comp": h["comp"], "struct": h["struct"], "a_exp": h["a_exp"],
            "ref": h.get("ref", "unknown"), "origin": "test",
            "idx": i, "eset": major_elements(h["comp"]),
        })
    return pool


def assign_families(pool: list[dict], thr: float = FAMILY_JACCARD) -> None:
    """元素集合の Jaccard 単連結クラスタリングで family_id を付与する。"""
    sets = [h["eset"] for h in pool]
    n = len(sets)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)

    for i in range(n):
        for j in range(i + 1, n):
            if pool[i]["struct"] != pool[j]["struct"]:
                continue
            a, b = sets[i], sets[j]
            jac = len(a & b) / len(a | b)
            if jac >= thr:
                union(i, j)

    roots = {}
    for i in range(n):
        r = find(i)
        roots.setdefault(r, len(roots))
    for i, h in enumerate(pool):
        h["family_id"] = roots[find(i)]

    # 可読な family 名（構成元素の和集合）
    fam_elems = defaultdict(set)
    for h in pool:
        fam_elems[h["family_id"]] |= set(h["eset"])
    for h in pool:
        h["family_name"] = "-".join(sorted(fam_elems[h["family_id"]]))


# ---------------------------------------------------------------------------
# モデル群
# ---------------------------------------------------------------------------
def fit_q(train: list[dict], omega: dict[str, dict], struct: str) -> float:
    """訓練 fold 内で構造別に q を最適化する。"""
    sub = [h for h in train if h["struct"] == struct]
    if not sub:
        return 1.0
    y = np.array([h["a_exp"] for h in sub])
    om = omega[struct]

    def obj(q):
        p = np.array([compute_eq10_scaled(h["comp"], struct, om, q) for h in sub])
        return float(np.sqrt(np.mean((p - y) ** 2)))

    return float(minimize_scalar(obj, bounds=(-5, 5), method="bounded").x)


def model_vegard(train, omega):
    def predict(h):
        return compute_vegard(h["comp"], h["struct"])
    return predict, {}


def make_eq10_model(omega: dict[str, dict]):
    """Eq.10（q を訓練 fold 内で最適化）。omega は {"BCC":..., "FCC":...}。"""
    def build(train, _ctx=None):
        qs = {s: fit_q(train, omega, s) for s in ("BCC", "FCC")}

        def predict(h):
            return compute_eq10_scaled(h["comp"], h["struct"],
                                       omega[h["struct"]], qs[h["struct"]])
        return predict, {"q_BCC": qs["BCC"], "q_FCC": qs["FCC"]}
    return build


def make_veff_model(lam_grid=(0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0)):
    """元素別有効原子体積 V_i^eff の直接最小二乗（King体積へのリッジ収縮つき）。

    V_alloy(exp) = sum_i c_i V_i^eff を構造別に解く。
    lam=0 は純粋な最小二乗（未出現元素は King 値で補完）、
    lam>0 は King 体積へ収縮させた解。lam は訓練 fold 内の
    内側 family-CV で選ぶ（評価 fold には触れない）。
    """
    def solve(train_sub: list[dict], struct: str, lam: float) -> dict[str, float]:
        elems = sorted({e for h in train_sub for e in h["comp"]})
        if not elems:
            return {}
        idx = {e: k for k, e in enumerate(elems)}
        A = np.zeros((len(train_sub), len(elems)))
        y = np.zeros(len(train_sub))
        for r, h in enumerate(train_sub):
            tot = sum(h["comp"].values())
            for e, c in h["comp"].items():
                A[r, idx[e]] = c / tot
            y[r] = v_per_atom(h["a_exp"], struct)
        v0 = np.array([KING_ATOMIC_VOLUMES.get(e, 15.0) for e in elems])
        # min ||A v - y||^2 + lam * ||v - v0||^2
        AtA = A.T @ A + lam * np.eye(len(elems))
        Aty = A.T @ y + lam * v0
        v = np.linalg.solve(AtA, Aty) if lam > 0 else np.linalg.pinv(A) @ y
        return dict(zip(elems, v))

    def predict_with(vmap: dict[str, float], h: dict) -> float:
        tot = sum(h["comp"].values())
        v = sum((c / tot) * vmap.get(e, KING_ATOMIC_VOLUMES.get(e, 15.0))
                for e, c in h["comp"].items())
        return a_from_v(v, h["struct"])

    def build(train, _ctx=None):
        vmaps, lams = {}, {}
        for struct in ("BCC", "FCC"):
            sub = [h for h in train if h["struct"] == struct]
            if not sub:
                vmaps[struct] = {}
                lams[struct] = 0.0
                continue
            # 内側 family-CV で lam を選ぶ
            fams = sorted({h["family_id"] for h in sub})
            best_lam, best_rmse = lam_grid[0], np.inf
            if len(fams) >= 3:
                for lam in lam_grid:
                    errs = []
                    for f in fams:
                        tr = [h for h in sub if h["family_id"] != f]
                        te = [h for h in sub if h["family_id"] == f]
                        if not tr:
                            continue
                        vm = solve(tr, struct, lam)
                        errs += [predict_with(vm, h) - h["a_exp"] for h in te]
                    if errs:
                        r = float(np.sqrt(np.mean(np.square(errs))))
                        if r < best_rmse:
                            best_rmse, best_lam = r, lam
            vmaps[struct] = solve(sub, struct, best_lam)
            lams[struct] = best_lam

        def predict(h):
            return predict_with(vmaps[h["struct"]], h)
        return predict, {"lam_BCC": lams["BCC"], "lam_FCC": lams["FCC"]}
    return build


def pair_dr(pair: tuple[str, str]) -> float | None:
    """King体積由来の相対半径差 |dr| = 2|r_i-r_j|/(r_i+r_j)。"""
    va, vb = KING_ATOMIC_VOLUMES.get(pair[0]), KING_ATOMIC_VOLUMES.get(pair[1])
    if va is None or vb is None:
        return None
    ra, rb = va ** (1 / 3), vb ** (1 / 3)
    return 2 * abs(ra - rb) / (ra + rb)


def dr_surrogate_omega(omega: dict[str, dict]) -> tuple[dict[str, dict], dict]:
    """Omega_sf を |dr| の単純な線形関数で置き換えた代替 Omega を作る。

    回帰は二元系 Omega_sf DB 上で行う（HEA実験値は使わないので
    fold をまたぐリークは生じない）。
    """
    out, stats_out = {}, {}
    for struct, om in omega.items():
        xs, ys, pairs = [], [], []
        for p, o in om.items():
            d = pair_dr(p)
            if d is None:
                continue
            xs.append(d)
            ys.append(o)
            pairs.append(p)
        xs, ys = np.array(xs), np.array(ys)
        if len(xs) < 3:
            out[struct] = dict(om)
            stats_out[struct] = {"n": int(len(xs))}
            continue
        res = stats.linregress(xs, ys)
        out[struct] = {p: float(res.intercept + res.slope * pair_dr(p))
                       for p in pairs}
        stats_out[struct] = {
            "n": int(len(xs)), "slope": float(res.slope),
            "intercept": float(res.intercept), "r": float(res.rvalue),
            "R2": float(res.rvalue ** 2), "p_value": float(res.pvalue),
        }
    return out, stats_out


def permute_omega(omega: dict[str, dict], rng: np.random.Generator) -> dict[str, dict]:
    """ペアと Omega_sf 値の対応だけを壊す（値の分布は保存）。"""
    out = {}
    for struct, om in omega.items():
        keys = list(om.keys())
        vals = list(om.values())
        rng.shuffle(vals)
        out[struct] = dict(zip(keys, vals))
    return out


# ---------------------------------------------------------------------------
# 交差検証ドライバ
# ---------------------------------------------------------------------------
def group_cv(pool: list[dict], build, group_key: str) -> dict:
    """群単位の leave-one-group-out CV。out-of-fold 予測を返す。"""
    groups = sorted({h[group_key] for h in pool}, key=str)
    preds = np.full(len(pool), np.nan)
    fold_params = []
    for g in groups:
        te_idx = [i for i, h in enumerate(pool) if h[group_key] == g]
        tr = [h for i, h in enumerate(pool) if h[group_key] != g]
        if not tr:
            continue
        predict, params = build(tr)
        for i in te_idx:
            preds[i] = predict(pool[i])
        fold_params.append({"group": str(g), "n_test": len(te_idx), **params})
    return {"pred": preds, "n_groups": len(groups), "folds": fold_params}


def score(pool: list[dict], pred: np.ndarray) -> dict:
    y = np.array([h["a_exp"] for h in pool])
    ok = ~np.isnan(pred)
    out = {}
    for label, mask in (
        ("all", ok),
        ("BCC", ok & np.array([h["struct"] == "BCC" for h in pool])),
        ("FCC", ok & np.array([h["struct"] == "FCC" for h in pool])),
    ):
        if mask.sum() == 0:
            continue
        r = pred[mask] - y[mask]
        out[label] = {
            "n": int(mask.sum()),
            "rmse": float(np.sqrt(np.mean(r ** 2))),
            "mae": float(np.mean(np.abs(r))),
            "median_signed": float(np.median(r)),
        }
    return out


def improvement(base: dict, model: dict, key: str = "rmse") -> dict:
    out = {}
    for k in base:
        if k in model and base[k][key] > 0:
            out[k] = 100.0 * (base[k][key] - model[k][key]) / base[k][key]
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    pool = build_pool()
    assign_families(pool)

    print(f"pool = {len(pool)} HEA "
          f"(calibration {sum(h['origin'] == 'calibration' for h in pool)}, "
          f"test {sum(h['origin'] == 'test' for h in pool)})")

    # --- Omega_sf ソース -------------------------------------------------
    all_df = load_compounds()
    ob2, ol12 = compute_omega_sf_pairwise(all_df)
    omega_cmp = {"BCC": ob2, "FCC": ol12}          # B2/L1_2 + King（要旨代表モデル）
    sqs = load_sqs_data()
    omega_sqs = {"BCC": sqs["omega_dft"], "FCC": sqs["fcc_omega_dft"]}

    print(f"omega B2 {len(ob2)} / L12 {len(ol12)} pairs; "
          f"SQS BCC {len(omega_sqs['BCC'])} / FCC {len(omega_sqs['FCC'])} pairs")

    # --- 化学系の構造 ---------------------------------------------------
    fam_summary = defaultdict(lambda: {"n": 0, "refs": set(), "structs": set()})
    for h in pool:
        s = fam_summary[h["family_name"]]
        s["n"] += 1
        s["refs"].add(h["ref"])
        s["structs"] |= {h["struct"]}
    families = [
        {"family": k, "n_alloys": v["n"],
         "structs": sorted(v["structs"]), "refs": sorted(v["refs"])}
        for k, v in sorted(fam_summary.items(), key=lambda kv: -kv[1]["n"])
    ]
    n_eset = len({h["eset"] for h in pool})
    print(f"element_set groups = {n_eset}, family groups = {len(families)}")

    # --- モデル定義 -----------------------------------------------------
    builders = {
        "vegard": lambda tr: model_vegard(tr, None),
        "eq10_compound_king": make_eq10_model(omega_cmp),
        "eq10_sqs_dft": make_eq10_model(omega_sqs),
        "veff_empirical": make_veff_model(),
    }
    omega_dr, dr_fit = dr_surrogate_omega(omega_cmp)
    builders["eq10_dr_surrogate"] = make_eq10_model(omega_dr)

    results = {}
    per_alloy = {}
    for gkey in ("eset", "family_id"):
        gname = "element_set" if gkey == "eset" else "family"
        results[gname] = {}
        for mname, build in builders.items():
            cv = group_cv(pool, build, gkey)
            sc = score(pool, cv["pred"])
            results[gname][mname] = {
                "n_groups": cv["n_groups"], "scores": sc,
                "folds": cv["folds"] if mname != "vegard" else [],
            }
            per_alloy[(gname, mname)] = cv["pred"]
        base = results[gname]["vegard"]["scores"]
        for mname in builders:
            results[gname][mname]["improvement_vs_vegard_pct"] = improvement(
                base, results[gname][mname]["scores"])
        print(f"\n[{gname} CV] ({results[gname]['vegard']['n_groups']} groups)")
        for mname in builders:
            sc = results[gname][mname]["scores"]
            imp = results[gname][mname]["improvement_vs_vegard_pct"]
            print(f"  {mname:22s} RMSE all={sc['all']['rmse']:.4f} "
                  f"BCC={sc.get('BCC', {}).get('rmse', float('nan')):.4f} "
                  f"FCC={sc.get('FCC', {}).get('rmse', float('nan')):.4f} "
                  f"| imp all={imp.get('all', 0):+.1f}% "
                  f"BCC={imp.get('BCC', 0):+.1f}% FCC={imp.get('FCC', 0):+.1f}%")

    # --- 文献セット丸ごと除外 -------------------------------------------
    holdouts = {}
    ref_counts = defaultdict(int)
    for h in pool:
        ref_counts[h["ref"]] += 1
    for ref, cnt in sorted(ref_counts.items(), key=lambda kv: -kv[1]):
        if ref == "Alonso2005_Table2" or cnt < 2:
            continue
        tr = [h for h in pool if h["ref"] != ref]
        te = [h for h in pool if h["ref"] == ref]
        entry = {"n_test": len(te)}
        for mname, build in builders.items():
            predict, params = build(tr)
            p = np.array([predict(h) for h in te])
            y = np.array([h["a_exp"] for h in te])
            entry[mname] = {
                "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
                "mae": float(np.mean(np.abs(p - y))),
                "params": params,
            }
        for mname in builders:
            if mname == "vegard":
                continue
            b = entry["vegard"]["rmse"]
            entry[mname]["improvement_vs_vegard_pct"] = (
                100.0 * (b - entry[mname]["rmse"]) / b if b > 0 else 0.0)
        holdouts[ref] = entry
        print(f"\n[leave-{ref}-out] n={len(te)}")
        for mname in builders:
            imp = entry[mname].get("improvement_vs_vegard_pct", 0.0)
            print(f"  {mname:22s} RMSE={entry[mname]['rmse']:.4f} "
                  f"imp={imp:+.1f}%")

    # --- permutation test ------------------------------------------------
    print(f"\n[permutation test] N={N_PERM} (family CV, q re-optimised per fold)")
    perm = {}
    for src_name, om in (("eq10_compound_king", omega_cmp),
                         ("eq10_sqs_dft", omega_sqs)):
        obs = results["family"][src_name]["scores"]
        rec = {"observed": {k: obs[k]["rmse"] for k in obs}, "null": {}}
        null = {k: [] for k in obs}
        for _ in range(N_PERM):
            om_p = permute_omega(om, rng)
            cv = group_cv(pool, make_eq10_model(om_p), "family_id")
            sc = score(pool, cv["pred"])
            for k in null:
                if k in sc:
                    null[k].append(sc[k]["rmse"])
        base = results["family"]["vegard"]["scores"]
        for k, vals in null.items():
            v = np.array(vals)
            rec["null"][k] = {
                "n": int(v.size),
                "rmse_mean": float(v.mean()), "rmse_std": float(v.std(ddof=1)),
                "rmse_median": float(np.median(v)),
                "rmse_q025": float(np.quantile(v, 0.025)),
                "rmse_q975": float(np.quantile(v, 0.975)),
                # 帰無分布のもとで実測以上に良くなる確率
                "p_value": float((np.sum(v <= obs[k]["rmse"]) + 1) / (v.size + 1)),
                "median_improvement_vs_vegard_pct": float(
                    100.0 * (base[k]["rmse"] - np.median(v)) / base[k]["rmse"]),
            }
            print(f"  {src_name} {k}: observed {obs[k]['rmse']:.4f} vs "
                  f"null median {np.median(v):.4f} "
                  f"[{np.quantile(v, 0.025):.4f},{np.quantile(v, 0.975):.4f}] "
                  f"p={rec['null'][k]['p_value']:.4f} "
                  f"(null median imp "
                  f"{rec['null'][k]['median_improvement_vs_vegard_pct']:+.1f}%)")
        perm[src_name] = rec

    # --- 従来評価（参照用: 校正 in-sample と テスト out-of-sample） -------
    legacy = {}
    for src_name, om in (("eq10_compound_king", omega_cmp),
                         ("eq10_sqs_dft", omega_sqs)):
        cal = [h for h in pool if h["origin"] == "calibration"]
        tst = [h for h in pool if h["origin"] == "test"]
        predict, params = make_eq10_model(om)(cal)
        d = {"params": params}
        for label, sub in (("calibration", cal), ("test", tst)):
            p = np.array([predict(h) for h in sub])
            v = np.array([compute_vegard(h["comp"], h["struct"]) for h in sub])
            y = np.array([h["a_exp"] for h in sub])
            d[label] = {}
            for k, mask in (("all", np.ones(len(sub), bool)),
                            ("BCC", np.array([h["struct"] == "BCC" for h in sub])),
                            ("FCC", np.array([h["struct"] == "FCC" for h in sub]))):
                if mask.sum() == 0:
                    continue
                rm = float(np.sqrt(np.mean((p[mask] - y[mask]) ** 2)))
                rv = float(np.sqrt(np.mean((v[mask] - y[mask]) ** 2)))
                d[label][k] = {
                    "n": int(mask.sum()), "rmse_vegard": rv, "rmse_model": rm,
                    "improvement_pct": 100.0 * (rv - rm) / rv if rv > 0 else 0.0,
                }
        legacy[src_name] = d
        print(f"\n[legacy split] {src_name}")
        for label in ("calibration", "test"):
            for k, s in d[label].items():
                print(f"  {label:11s} {k:3s} n={s['n']:2d} "
                      f"Vegard={s['rmse_vegard']:.4f} model={s['rmse_model']:.4f} "
                      f"imp={s['improvement_pct']:+.1f}%")

    # --- 入れ子プロトコル（指摘 II: 構成選択を校正64のみで完結させる） ----
    # 校正64の family CV だけでモデル構成を選び、その1つだけを
    # 独立テスト31に1回適用する。テストRMSEでの構成選択は行わない。
    nested = {"selection": {}, "evaluation": {}}
    cal = [h for h in pool if h["origin"] == "calibration"]
    tst = [h for h in pool if h["origin"] == "test"]
    cal_scores = {}
    for mname, build in builders.items():
        cv = group_cv(cal, build, "family_id")
        cal_scores[mname] = score(cal, cv["pred"])
    for struct in ("BCC", "FCC"):
        cand = {m: s[struct]["rmse"] for m, s in cal_scores.items()
                if struct in s}
        chosen = min(cand, key=cand.get)
        nested["selection"][struct] = {
            "candidates_calibration_familyCV_rmse": cand, "chosen": chosen}
        sub = [h for h in tst if h["struct"] == struct]
        predict, params = builders[chosen](cal)
        p = np.array([predict(h) for h in sub])
        v = np.array([compute_vegard(h["comp"], h["struct"]) for h in sub])
        y = np.array([h["a_exp"] for h in sub])
        rm = float(np.sqrt(np.mean((p - y) ** 2)))
        rv = float(np.sqrt(np.mean((v - y) ** 2)))
        nested["evaluation"][struct] = {
            "model": chosen, "n_test": len(sub), "params": params,
            "rmse_vegard": rv, "rmse_model": rm,
            "improvement_pct": 100.0 * (rv - rm) / rv if rv > 0 else 0.0,
        }
    print("\n[nested protocol] 構成選択=校正64のfamily CV / 評価=独立テスト1回")
    for struct, e in nested["evaluation"].items():
        cand = nested["selection"][struct]["candidates_calibration_familyCV_rmse"]
        print(f"  {struct}: chosen={e['model']} (cal familyCV "
              + ", ".join(f"{m}={r:.4f}" for m, r in sorted(cand.items(),
                                                            key=lambda kv: kv[1]))
              + f") -> test n={e['n_test']} Vegard={e['rmse_vegard']:.4f} "
              f"model={e['rmse_model']:.4f} imp={e['improvement_pct']:+.1f}%")

    # --- ノイズフロア sigma の区間（指摘 III） ---------------------------
    def comp_key(comp):
        tot = sum(comp.values())
        return tuple(sorted((e, round(c / tot, 3)) for e, c in comp.items()))

    groups = defaultdict(list)
    for h in pool:
        groups[(h["struct"], comp_key(h["comp"]))].append(h["a_exp"])
    dups = [(k, v) for k, v in groups.items() if len(v) > 1]
    # 群内標本分散をプールする。ペア差を独立標本として数えると、3測定を
    # 含む群の3差が互いに依存するため標準誤差を過小評価する。
    # sigma_p^2 = sum (n_g-1) s_g^2 / sum (n_g-1)、自由度 nu = sum (n_g-1)。
    # 区間は chi^2 分布から取る（群内依存を正しく扱う）。
    noise = {"n_duplicate_groups": len(dups)}
    num = sum((len(v) - 1) * float(np.var(v, ddof=1)) for _, v in dups)
    nu = sum(len(v) - 1 for _, v in dups)
    if nu > 0:
        s2p = num / nu
        sp = float(np.sqrt(s2p))
        lo = float(np.sqrt(nu * s2p / stats.chi2.ppf(0.975, nu)))
        hi = float(np.sqrt(nu * s2p / stats.chi2.ppf(0.025, nu)))
        se = sp / np.sqrt(2 * nu)          # sigma_hat の漸近標準誤差
        # 群間の分散均一性（プールの前提）を Bartlett 検定で確認する
        bart = stats.bartlett(*[v for _, v in dups]) if len(dups) > 1 else None
        noise.update({
            "dof": int(nu),
            "sigma_est_A": sp,
            "sigma_se_A": float(se),
            "sigma_rel_se": float(se / sp),
            "sigma_ci95_A": [lo, hi],
            "ci_method": "pooled within-group variance, chi-square interval",
            "bartlett_stat": float(bart.statistic) if bart else None,
            "bartlett_p": float(bart.pvalue) if bart else None,
            "duplicate_groups": [
                {"struct": k[0], "comp": dict(k[1]), "a_exp": sorted(v),
                 "n": len(v), "spread_A": float(max(v) - min(v)),
                 "sd_A": float(np.std(v, ddof=1))} for k, v in dups],
        })
        print(f"\n[noise floor] {len(dups)} duplicate groups, dof={nu}: "
              f"pooled sigma = {sp:.4f} +/- {se:.4f} A, "
              f"chi2 95%CI [{lo:.4f}, {hi:.4f}]"
              + (f", Bartlett p={bart.pvalue:.3f}" if bart else ""))

    # --- 出力 -----------------------------------------------------------
    out = {
        "config": {
            "family_jaccard_threshold": FAMILY_JACCARD,
            "comp_eps": COMP_EPS, "n_permutations": N_PERM, "rng_seed": RNG_SEED,
        },
        "pool": {
            "n_total": len(pool),
            "n_calibration": sum(h["origin"] == "calibration" for h in pool),
            "n_test": sum(h["origin"] == "test" for h in pool),
            "n_element_set_groups": n_eset,
            "n_family_groups": len(families),
        },
        "families": families,
        "omega_sources": {
            "B2_pairs": len(ob2), "L12_pairs": len(ol12),
            "SQS_BCC_pairs": len(omega_sqs["BCC"]),
            "SQS_FCC_pairs": len(omega_sqs["FCC"]),
        },
        "dr_surrogate_fit": dr_fit,
        "cv": results,
        "reference_holdout": holdouts,
        "permutation_test": perm,
        "legacy_split": legacy,
        "nested_protocol": nested,
        "noise_floor": noise,
    }
    with open(OUTDIR / "robust_validation.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    with open(OUTDIR / "results_robust_cv.csv", "w", newline="") as f:
        w = csv.writer(f)
        cols = list(builders.keys())
        w.writerow(["origin", "ref", "struct", "family", "composition", "a_exp"]
                   + [f"a_pred_family_{c}" for c in cols]
                   + [f"a_pred_eset_{c}" for c in cols])
        for i, h in enumerate(pool):
            comp = "".join(f"{e}{h['comp'][e]:.3f}" for e in sorted(h["comp"]))
            w.writerow([h["origin"], h["ref"], h["struct"], h["family_name"],
                        comp, h["a_exp"]]
                       + [f"{per_alloy[('family', c)][i]:.4f}" for c in cols]
                       + [f"{per_alloy[('element_set', c)][i]:.4f}" for c in cols])

    print("\nwrote paper/robust_validation.json, paper/results_robust_cv.csv")


if __name__ == "__main__":
    main()
