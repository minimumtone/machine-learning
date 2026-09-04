"""リポジトリの tests/ に追加することを想定した回帰テスト.

現状（PR#148, 合成データ n=120）では 8 件中 6 件が FAIL します。
修正後はすべて PASS すべき内容です。
`tests/` にコピーし、既存の conftest / パス解決に合わせて調整してください。

    pytest tests/test_feature_discovery_sanity.py -v

注意:
  - `test_baseline_round_shows_no_spurious_improvement` と
    `test_predictions_stay_within_plausible_range` は合成データでは通ります。
    実データ（HEA_ml_numeric_highconf.csv / generic モード）では両方 FAIL します
    （診断スイートの D2-3 / D3-2 を参照）。実データ版フィクスチャを足すと
    より強い回帰テストになります。
"""
from __future__ import annotations

import gc

import numpy as np
import pandas as pd
import pytest

from extrapolation_discovery_platform.dataset import generate_hea_dataset
from extrapolation_discovery_platform.ood_feature_discovery import (
    augment_dataset,
    identify_boundary_samples,
    run_feature_discovery,
)
from extrapolation_discovery_platform.pipeline import (
    stage1_preprocess,
    stage3_detect_ood,
)


@pytest.fixture(scope="module")
def prepared():
    comps, feat, y = generate_hea_dataset(n_samples=120, seed=1)
    prep = stage1_preprocess(
        features_df=feat, target=y, compositions_df=comps,
        feature_set_names=["FS_ALL"], workflow_names=["WF-LIN"],
        seeds=[42], active_policies=["CompositionBlock"], n_folds=5,
    )
    assert prep.success, prep.error_message
    ood = stage3_detect_ood(feat, prep.effective_cols["FS_ALL"], prep.fold_plan)
    assert ood.success, ood.error_message
    return comps, feat, y, prep, ood


def _candidates(y, n, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "noise_desc": rng.normal(size=n),
        "informative_desc": y.to_numpy(dtype="float64")
        + rng.normal(scale=float(y.std()) * 1.2, size=n),
    })


# ---------------------------------------------------------------------------
# 1. 候補特徴量がモデルに届いているか
# ---------------------------------------------------------------------------

def test_candidate_feature_reaches_the_model(prepared):
    """候補列を変えれば結果も変わるはず（現状は全候補が同値）。"""
    comps, feat, y, _prep, ood = prepared
    extra = _candidates(y, len(feat))
    res = run_feature_discovery(
        workflow_names=["WF-LIN"], feature_set_name="FS_ALL",
        split_policy="CompositionBlock",
        features_df=feat, target=y, compositions_df=comps,
        ood_result=ood.ood_result, ood_test_idx=ood.primary_test_idx,
        candidate_features=list(extra.columns), extra_features_df=extra,
        seed=42, n_folds=5, quick=True,
    )
    assert res.success, res.error_message
    rmses = {r.candidate_feature or "(baseline)": r.ood_rmse for r in res.rounds}
    assert len(set(round(v, 6) for v in rmses.values())) > 1, (
        "全候補で ood_rmse が同一。候補列が Stage1 で除去されています: " + repr(rmses)
    )


# ---------------------------------------------------------------------------
# 2. 評価集合が訓練から分離されているか
# ---------------------------------------------------------------------------

def test_ood_eval_samples_not_in_training(prepared):
    comps, feat, y, _prep, ood = prepared
    binfo = identify_boundary_samples(ood.ood_result, n_ood_samples=0, margin=0.5)
    _X, _y, train_idx, ood_eval_idx = augment_dataset(
        features_df=feat, target=y, boundary_info=binfo,
        ood_test_idx=ood.primary_test_idx,
    )
    if len(ood_eval_idx) == 0:
        pytest.skip("OOD 評価サンプルが 0 件")
    overlap = int(np.isin(ood_eval_idx, train_idx).sum())
    assert overlap == 0, (
        f"OOD 評価 {len(ood_eval_idx)} 件のうち {overlap} 件が訓練に含まれています"
    )


def test_baseline_round_shows_no_spurious_improvement(prepared):
    """候補を追加しないベースライン行の improvement は 0 近傍であるべき。"""
    comps, feat, y, _prep, ood = prepared
    res = run_feature_discovery(
        workflow_names=["WF-LIN"], feature_set_name="FS_ALL",
        split_policy="CompositionBlock",
        features_df=feat, target=y, compositions_df=comps,
        ood_result=ood.ood_result, ood_test_idx=ood.primary_test_idx,
        candidate_features=[], extra_features_df=None,
        seed=42, n_folds=5, quick=True,
    )
    base = [r for r in res.rounds if not r.candidate_feature]
    assert base, "ベースライン行がありません"
    imp = base[0].improvement
    assert abs(imp) < 0.15, (
        f"ベースラインの improvement={imp:+.4f}。"
        "baseline_rmse(CV) と ood_rmse(訓練当てはめ) を比較しています。"
    )


# ---------------------------------------------------------------------------
# 3. ネガティブコントロール
# ---------------------------------------------------------------------------

def test_noise_feature_does_not_outrank_informative_feature(prepared):
    """純ノイズが情報のある特徴量を上回ってはならない。"""
    comps, feat, y, _prep, ood = prepared
    extra = _candidates(y, len(feat))
    res = run_feature_discovery(
        workflow_names=["WF-LIN"], feature_set_name="FS_ALL",
        split_policy="CompositionBlock",
        features_df=feat, target=y, compositions_df=comps,
        ood_result=ood.ood_result, ood_test_idx=ood.primary_test_idx,
        candidate_features=list(extra.columns), extra_features_df=extra,
        seed=42, n_folds=5, quick=True,
    )
    imp = {r.candidate_feature: r.improvement for r in res.rounds if r.candidate_feature}
    assert imp["informative_desc"] > imp["noise_desc"], (
        f"純ノイズ({imp['noise_desc']:+.4f}) が "
        f"情報のある特徴量({imp['informative_desc']:+.4f}) を上回りました"
    )


# ---------------------------------------------------------------------------
# 4. 境界サンプル抽出の契約
# ---------------------------------------------------------------------------

def test_identify_boundary_respects_threshold():
    class _Fake:
        composite_scores = np.linspace(0.0, 1.0, 50)
        ood_threshold = 0.9
        is_ood = composite_scores > 0.9

    b = identify_boundary_samples(_Fake(), n_ood_samples=10, margin=0.5)
    scores = _Fake.composite_scores[b.ood_indices]
    assert (scores >= _Fake.ood_threshold).all(), (
        "n_ood_samples 指定時に threshold 未満のサンプルが OOD 扱いされています"
    )


def test_augment_dataset_validates_extra_row_count(prepared):
    comps, feat, y, _prep, ood = prepared
    binfo = identify_boundary_samples(ood.ood_result, n_ood_samples=0, margin=0.5)
    bad = pd.DataFrame({"cand": np.zeros(len(feat) // 2)})  # 行数が半分
    with pytest.raises((ValueError, AssertionError)):
        augment_dataset(
            features_df=feat, target=y, boundary_info=binfo,
            ood_test_idx=ood.primary_test_idx,
            extra_features_df=bad, candidate_col="cand",
        )


# ---------------------------------------------------------------------------
# 5. 予測の応用領域ガード
# ---------------------------------------------------------------------------

def test_predictions_stay_within_plausible_range(prepared):
    """外挿 fold でも予測が学習値域を大きく超えないこと。"""
    from extrapolation_discovery_platform.pipeline import stage2_train

    comps, feat, y, prep, _ood = prepared
    tr = stage2_train(prep, feat, y, "WF-LIN", "CompositionBlock", "FS_ALL",
                      quick=True, seed=42)
    assert tr.success, tr.error_message
    lo, hi = float(y.min()), float(y.max())
    span = hi - lo
    for run in tr.runs:
        pred = np.asarray(run.y_test_pred, dtype="float64")
        assert pred.max() <= hi + span, (
            f"fold {run.fold}: 予測最大 {pred.max():.0f} が学習値域 "
            f"[{lo:.0f}, {hi:.0f}] を大きく逸脱しています"
        )
        assert pred.min() >= lo - span, (
            f"fold {run.fold}: 予測最小 {pred.min():.0f} が学習値域を大きく下回ります"
        )


# ---------------------------------------------------------------------------
# 6. グローバル副作用
# ---------------------------------------------------------------------------

def test_gc_is_not_globally_disabled():
    import extrapolation_discovery_platform  # noqa: F401
    assert gc.isenabled(), (
        "_compat.install() が循環 GC をプロセス全体で停止しています"
    )


# ---------------------------------------------------------------------------
# 9. 訓練スコープ（OOD 近辺のみで学習）
# ---------------------------------------------------------------------------

def test_neighborhood_plan_excludes_ood_eval_and_respects_scope(prepared):
    from extrapolation_discovery_platform.ood_feature_discovery import (
        compute_neighborhood_plan,
    )
    _comps, feat, _y, _prep, ood = prepared
    b = identify_boundary_samples(ood.ood_result, 0, 0.5)
    ood_idx = np.unique(ood.primary_test_idx[b.ood_indices])
    n = len(feat)

    g = compute_neighborhood_plan(feat, ood_idx, scope="global")
    assert g.is_global and g.n_train_rows == n - len(ood_idx)
    assert (g.copies[ood_idx] == 0).all()

    nb = compute_neighborhood_plan(feat, ood_idx, scope="neighborhood",
                                   neighborhood_quantile=0.3, min_train_rows=20)
    assert (nb.copies[ood_idx] == 0).all()
    assert 20 <= nb.n_train_rows < n - len(ood_idx)
    # 近い行が残り、遠い行が落ちている
    kept = nb.distances[nb.copies >= 1]
    dropped = nb.distances[(nb.copies == 0) & ~np.isin(np.arange(n), ood_idx)]
    if len(dropped):
        assert kept.max() <= dropped.min() + 1e-12

    k = compute_neighborhood_plan(feat, ood_idx, scope="kernel", kernel_max_copies=4)
    assert (k.copies[ood_idx] == 0).all()
    non_eval = ~np.isin(np.arange(n), ood_idx)
    assert k.copies[non_eval].min() >= 1 and k.copies[non_eval].max() <= 4
    # 距離が短いほど複製回数が多い（単調）
    order = np.argsort(k.distances[non_eval])
    c = k.copies[non_eval][order]
    assert (np.diff(c) <= 0).all()


def test_neighborhood_plan_with_no_ood_eval_rows(prepared):
    from extrapolation_discovery_platform.ood_feature_discovery import (
        compute_neighborhood_plan,
    )
    _comps, feat, _y, _prep, _ood = prepared
    for scope in ("neighborhood", "kernel"):
        plan = compute_neighborhood_plan(
            feat, np.array([], dtype=int), scope=scope,
        )
        assert plan.copies.min() == 1
        assert plan.n_train_rows == len(feat)
        assert plan.n_train_aug == len(feat)


def test_augment_with_plan_keeps_ood_rows_out_of_training(prepared):
    from extrapolation_discovery_platform.ood_feature_discovery import (
        compute_neighborhood_plan,
    )
    _comps, feat, y, _prep, ood = prepared
    b = identify_boundary_samples(ood.ood_result, 0, 0.5)
    ood_idx = np.unique(ood.primary_test_idx[b.ood_indices])
    for scope in ("neighborhood", "kernel"):
        plan = compute_neighborhood_plan(feat, ood_idx, scope=scope)
        X_aug, y_aug, tr, ev = augment_dataset(feat, y, b, ood.primary_test_idx,
                                               neighborhood_plan=plan)
        assert len(X_aug) == len(y_aug) == plan.n_train_aug + len(ev) + (
            len(feat) - plan.n_train_rows - len(ev))
        assert len(np.intersect1d(tr, ev)) == 0
        assert len(tr) == plan.n_train_aug
        # 複製行は元行の写し
        for j in tr[tr >= len(feat)]:
            src = X_aug.iloc[j].to_numpy()
            assert np.isfinite(src).all()


def test_discovery_runs_under_each_train_scope(prepared):
    comps, feat, y, _prep, ood = prepared
    cands = _candidates(y, len(feat))
    for scope in ("global", "neighborhood", "kernel"):
        res = run_feature_discovery(
            workflow_names=["WF-LIN"], feature_set_name="FS_ALL",
            split_policy="CompositionBlock",
            features_df=feat, target=y, compositions_df=comps,
            ood_result=ood.ood_result, ood_test_idx=ood.primary_test_idx,
            candidate_features=["informative_desc"], extra_features_df=cands,
            seed=42, n_folds=5, quick=True, train_scope=scope,
            include_negative_control=False,
        )
        assert res.success, res.error_message
        assert res.train_scope == scope
        assert res.n_ood_eval > 0
        for r in res.rounds:
            assert r.success, r.error_message
            assert r.train_scope == scope
            assert r.n_ood_eval == res.n_ood_eval
            assert r.n_train_rows == res.n_train_rows
            assert np.isfinite(r.ood_rmse)
        base = [r for r in res.rounds if not r.candidate_feature][0]
        assert abs(base.improvement) < 1e-9
