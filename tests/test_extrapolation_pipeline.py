"""
PR#148 自動テストスイート
==========================
TEST_CHECKLIST.md の T1〜T9 を自動実行します。

実行方法:
    cd extrapolation_discovery_platform_PR148
    PYTHONPATH=. python3 tests/test_pipeline.py
    # または
    PYTHONPATH=. python3 -m pytest tests/test_pipeline.py -v
"""
import dataclasses
import inspect
import math
import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

# ── テスト用データ生成 ────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def sample_data():
    """80 サンプルの HEA ダミーデータ。FS_ALL の全列を含む。"""
    from extrapolation_discovery_platform.features import FeatureCatalog, FeatureSetName
    rng  = np.random.default_rng(42)
    n    = 80
    cols = FeatureCatalog.columns(FeatureSetName.FS_ALL)
    X    = pd.DataFrame(rng.normal(0, 1, (n, len(cols))), columns=cols)
    y    = pd.Series(rng.normal(500, 100, n))
    comp = pd.DataFrame(
        rng.dirichlet(np.ones(4), n), columns=["Co", "Cr", "Fe", "Ni"]
    )
    return X, y, comp


@pytest.fixture(scope="session")
def preprocess_result(sample_data):
    """Stage1 の結果（セッション全体で共有）。"""
    from extrapolation_discovery_platform.pipeline import stage1_preprocess
    from extrapolation_discovery_platform.features import FeatureSetName
    X, y, comp = sample_data
    prep = stage1_preprocess(
        features_df=X, target=y, compositions_df=comp,
        feature_set_names=[FeatureSetName.FS_ALL.value],
        workflow_names=["WF-LIN"],
        seeds=[42],
        active_policies=["CompositionBlock"],
    )
    assert prep.success, f"Stage1 失敗:\n{prep.error_message}"
    return prep


# ─────────────────────────────────────────────────────────────────────────────
class TestT1_Stage1Reproducibility:
    """T1: Stage1 再現性 — 同一入力で同一 effective_cols。"""

    def test_non_catalog_features_are_retained_and_leaks_excluded(self, sample_data):
        from extrapolation_discovery_platform.features import FeatureSetName
        from extrapolation_discovery_platform.pipeline import stage1_preprocess

        X, y, comp = sample_data
        features = X[["r_avg", "VEC"]].copy()
        rng = np.random.default_rng(123)
        features["experimental_descriptor"] = rng.normal(size=len(features))
        features["target_leak"] = y.to_numpy()

        prep = stage1_preprocess(
            features_df=features,
            target=y,
            compositions_df=comp,
            feature_set_names=[FeatureSetName.FS_BASE.value],
            workflow_names=["WF-LIN"],
            seeds=[42],
            active_policies=["RandomCV"],
            leak_auto_exclude=True,
        )

        assert prep.success, prep.error_message
        effective = prep.effective_cols[FeatureSetName.FS_BASE.value]
        assert "experimental_descriptor" in effective
        assert "delta_r" not in effective
        assert "target_leak" not in effective

    def test_hea_upload_imputes_partial_nan_extra_columns(self, sample_data, tmp_path):
        pytest.importorskip("gradio")
        from extrapolation_discovery_platform.gui.app import _handle_csv_upload
        from extrapolation_discovery_platform.features import FeatureSetName
        from extrapolation_discovery_platform.pipeline import stage1_preprocess

        _, y, comp = sample_data
        raw = comp.copy()
        raw["ys"] = y.to_numpy()
        rng = np.random.default_rng(7)
        grain = rng.uniform(1, 100, len(raw))
        grain[::4] = np.nan
        raw["grain_size_um"] = grain
        raw["all_missing"] = np.nan
        csv_path = tmp_path / "hea_nan.csv"
        raw.to_csv(csv_path, index=False)

        class _File:
            name = str(csv_path)

        session: dict = {}
        _handle_csv_upload(_File(), "ys", session)
        feats = session["features_df"]
        assert "grain_size_um" in feats.columns
        assert "all_missing" not in feats.columns
        assert not feats["grain_size_um"].isna().any()
        assert feats["grain_size_um"].iloc[0] == pytest.approx(np.nanmedian(grain))

        prep = stage1_preprocess(
            features_df=feats, target=session["target"],
            compositions_df=session["compositions_df"],
            feature_set_names=[FeatureSetName.FS_BASE.value],
            workflow_names=["WF-LIN"], seeds=[42],
            active_policies=["RandomCV"],
        )
        assert prep.success, prep.error_message
        assert "grain_size_um" in prep.effective_cols[FeatureSetName.FS_BASE.value]

    def test_effective_cols_identical(self, sample_data):
        from extrapolation_discovery_platform.pipeline import stage1_preprocess
        from extrapolation_discovery_platform.features import FeatureSetName
        X, y, comp = sample_data
        kwargs = dict(
            features_df=X, target=y, compositions_df=comp,
            feature_set_names=[FeatureSetName.FS_ALL.value],
            workflow_names=["WF-LIN"],
            seeds=[42],
            active_policies=["CompositionBlock"],
        )
        prep1 = stage1_preprocess(**kwargs)
        prep2 = stage1_preprocess(**kwargs)
        assert prep1.success and prep2.success
        assert prep1.effective_cols == prep2.effective_cols

    def test_fold_plan_not_empty(self, preprocess_result):
        assert len(preprocess_result.fold_plan) > 0
        assert "CompositionBlock" in preprocess_result.fold_plan
        folds = preprocess_result.fold_plan["CompositionBlock"]
        assert len(folds) > 0
        tr, te = folds[0]
        assert len(tr) > 0 and len(te) > 0

    def test_effective_cols_positive(self, preprocess_result):
        ec = preprocess_result.effective_cols.get("FS_ALL", [])
        assert len(ec) > 0


class TestT2_Stage2Reproducibility:
    """T2: Stage2 再現性 — 同一条件で同一 RMSE。"""

    def test_rmse_identical(self, sample_data, preprocess_result):
        from extrapolation_discovery_platform.pipeline import stage2_train
        X, y, _ = sample_data
        kwargs = dict(
            preprocess_result=preprocess_result,
            features_df=X, target=y,
            workflow_name="WF-LIN",
            split_policy_name="CompositionBlock",
            feature_set_name="FS_ALL",
            quick=True, seed=42,
        )
        tr1 = stage2_train(**kwargs)
        tr2 = stage2_train(**kwargs)
        assert tr1.success and tr2.success
        assert abs(tr1.rmse_test_mean - tr2.rmse_test_mean) < 1e-8

    def test_metrics_finite(self, sample_data, preprocess_result):
        from extrapolation_discovery_platform.pipeline import stage2_train
        X, y, _ = sample_data
        tr = stage2_train(preprocess_result, X, y, "WF-LIN",
                          "CompositionBlock", "FS_ALL", quick=True, seed=42)
        assert tr.success
        assert math.isfinite(tr.rmse_test_mean)
        assert math.isfinite(tr.r2_test_mean)
        assert tr.n_folds_executed > 0

    def test_no_ood_in_train_result(self, sample_data, preprocess_result):
        from extrapolation_discovery_platform.pipeline import stage2_train, TrainResult
        X, y, _ = sample_data
        tr = stage2_train(preprocess_result, X, y, "WF-LIN",
                          "CompositionBlock", "FS_ALL", quick=True, seed=42)
        tr_fields = {f.name for f in dataclasses.fields(tr)}
        assert "ood_result" not in tr_fields, "TrainResult に OOD が混入している"


class TestT3_Stage3Independence:
    """T3: Stage3 独立性 — RunResult に OOD 情報が混入しない。"""

    def test_ood_result_not_none(self, sample_data, preprocess_result):
        from extrapolation_discovery_platform.pipeline import stage3_detect_ood
        X, _, _ = sample_data
        ec = preprocess_result.effective_cols.get("FS_ALL", [])
        ood = stage3_detect_ood(X, ec, preprocess_result.fold_plan)
        assert ood.success, f"Stage3 失敗:\n{ood.error_message}"
        assert ood.ood_result is not None

    def test_run_result_has_no_ood_field(self):
        from extrapolation_discovery_platform.workflows import RunResult
        rr_fields = {f.name for f in dataclasses.fields(RunResult)}
        assert "ood_result" not in rr_fields, "RunResult に ood_result フィールドが混入"

    def test_primary_indices_set(self, sample_data, preprocess_result):
        from extrapolation_discovery_platform.pipeline import stage3_detect_ood
        X, _, _ = sample_data
        ec = preprocess_result.effective_cols.get("FS_ALL", [])
        ood = stage3_detect_ood(X, ec, preprocess_result.fold_plan)
        assert ood.primary_train_idx is not None
        assert ood.primary_test_idx  is not None


class TestT4_RunnerDelegation:
    """T4: runner.py が pipeline.py に委譲している。"""

    def test_pipeline_functions_called(self):
        from extrapolation_discovery_platform.runner import ExperimentRunner
        src = inspect.getsource(ExperimentRunner.run)
        assert "stage1_preprocess" in src
        assert "stage3_detect_ood"  in src

    def test_old_methods_removed(self):
        from extrapolation_discovery_platform.runner import ExperimentRunner
        src = inspect.getsource(ExperimentRunner.run)
        assert "_phase3_precompute_folds" not in src
        assert "_phase6_ood"              not in src

    def test_runner_stores_effective_cols(self, sample_data):
        from extrapolation_discovery_platform.runner import ExperimentRunner
        X, y, comp = sample_data
        runner = ExperimentRunner(seeds=[42], quick=True)
        runner.run(comp, X, y,
                   selected_workflows=["WF-LIN"],
                   selected_feature_sets=["FS_ALL"],
                   selected_split_policies=["CompositionBlock"])
        assert "FS_ALL" in runner._effective_cols
        assert len(runner._effective_cols["FS_ALL"]) > 0


class TestT5_IndividualDelegation:
    """T5: individual_runner.py が pipeline.py に委譲している。"""

    def test_pipeline_functions_called(self):
        from extrapolation_discovery_platform.individual_runner import run_individual
        src = inspect.getsource(run_individual)
        assert "stage1_preprocess" in src
        assert "stage2_train"      in src
        assert "stage3_detect_ood" in src

    def test_ood_result_present(self, sample_data):
        from extrapolation_discovery_platform.individual_runner import run_individual
        X, y, comp = sample_data
        res = run_individual("WF-LIN", "FS_ALL", "CompositionBlock",
                             features_df=X, target=y, compositions_df=comp,
                             seed=42, n_folds=5, quick=True)
        assert res.success
        assert res.runs is not None and len(res.runs) > 0


class TestT6_ResultConsistency:
    """T6: 一括計算と個別計算の結果一致（同一 precomputed_columns 使用）。"""

    def test_bulk_vs_individual_rmse(self, sample_data):
        from extrapolation_discovery_platform.runner import ExperimentRunner
        from extrapolation_discovery_platform.individual_runner import run_individual
        X, y, comp = sample_data

        # 一括計算
        runner = ExperimentRunner(seeds=[42], quick=True)
        runs_bulk, _, _ = runner.run(
            comp, X, y,
            selected_workflows=["WF-LIN"],
            selected_feature_sets=["FS_ALL"],
            selected_split_policies=["CompositionBlock"],
        )
        ec = runner._effective_cols.get("FS_ALL")
        bulk_rmse_list = [
            r.rmse_test for r in runs_bulk
            if r.workflow == "WF-LIN"
            and r.split_policy == "CompositionBlock"
            and r.rmse_test > 0
        ]
        bulk_mean = float(np.mean(bulk_rmse_list)) if bulk_rmse_list else float("nan")

        # 個別計算（precomputed_columns を引き継ぐ）
        res_ind = run_individual(
            "WF-LIN", "FS_ALL", "CompositionBlock",
            features_df=X, target=y, compositions_df=comp,
            seed=42, n_folds=5, quick=True,
            precomputed_columns=ec,
        )
        ind_mean = res_ind.rmse_test_mean

        diff_pct = abs(bulk_mean - ind_mean) / (bulk_mean + 1e-8) * 100
        assert diff_pct < 1.0, (
            f"一括 RMSE={bulk_mean:.4f}, 個別 RMSE={ind_mean:.4f}, "
            f"diff={diff_pct:.2f}% (1% を超えてはいけない)"
        )


class TestT7_RandomCVDefault:
    """T7: RandomCV がデフォルトで無効。"""

    def test_default_excludes_random_cv(self, sample_data):
        from extrapolation_discovery_platform.runner import ExperimentRunner
        X, y, comp = sample_data
        runner = ExperimentRunner(seeds=[42], quick=True)
        runs, _, _ = runner.run(comp, X, y)
        policies = {r.split_policy for r in runs}
        assert "RandomCV" not in policies, f"RandomCV がデフォルトで含まれている: {policies}"

    def test_explicit_random_cv_included(self, sample_data):
        from extrapolation_discovery_platform.runner import ExperimentRunner
        X, y, comp = sample_data
        runner = ExperimentRunner(seeds=[42], quick=True)
        runs, _, _ = runner.run(
            comp, X, y,
            selected_split_policies=["CompositionBlock", "RandomCV"],
        )
        policies = {r.split_policy for r in runs}
        assert "RandomCV" in policies, "明示的に指定しても RandomCV が含まれない"


class TestT8_WorkflowDiversity:
    """T8: WF-ENS の結果が WF-XGB と異なる（base_workflow=ridge 修正確認）。"""

    def test_ens_not_equal_xgb(self):
        from extrapolation_discovery_platform.features import FeatureCatalog, FeatureSetName
        from extrapolation_discovery_platform.workflows import WorkflowENS, WorkflowXGB
        rng  = np.random.default_rng(42)
        cols = FeatureCatalog.columns(FeatureSetName.FS_BASE)
        X    = pd.DataFrame(rng.normal(0, 1, (50, len(cols))), columns=cols)
        y    = pd.Series(rng.normal(500, 100, 50))
        kw   = dict(feature_set="FS_BASE", split_policy="CompositionBlock", fold=0)
        r_xgb = WorkflowXGB(quick=True).run(
            X.iloc[:40], y.iloc[:40], X.iloc[40:], y.iloc[40:], seed=42, **kw
        )
        r_ens = WorkflowENS(quick=True).run(
            X.iloc[:40], y.iloc[:40], X.iloc[40:], y.iloc[40:], seed=42, **kw
        )
        assert abs(r_xgb.rmse_test - r_ens.rmse_test) > 0.01, (
            f"WF-XGB と WF-ENS が同一結果: {r_xgb.rmse_test:.4f}"
        )



class TestT72_NFoldsConfig:
    """T7-2: n_folds の設定が pipeline / runner / GUI に正しく伝達される。"""

    def test_stage1_nfolds_3(self, sample_data):
        from extrapolation_discovery_platform.pipeline import stage1_preprocess
        from extrapolation_discovery_platform.features import FeatureSetName
        X, y, comp = sample_data
        prep = stage1_preprocess(
            X, y, comp, [FeatureSetName.FS_BASE.value], ["WF-LIN"],
            seeds=[42], active_policies=["CompositionBlock"], n_folds=3,
        )
        assert prep.success
        folds = prep.fold_plan.get("CompositionBlock", [])
        assert len(folds) == 3, f"n_folds=3 を指定したが fold 数={len(folds)}"

    def test_stage1_nfolds_default_5(self, sample_data):
        from extrapolation_discovery_platform.pipeline import stage1_preprocess
        from extrapolation_discovery_platform.features import FeatureSetName
        X, y, comp = sample_data
        prep = stage1_preprocess(
            X, y, comp, [FeatureSetName.FS_BASE.value], ["WF-LIN"],
            seeds=[42], active_policies=["CompositionBlock"],
        )
        folds = prep.fold_plan.get("CompositionBlock", [])
        assert len(folds) == 5, f"デフォルト fold 数={len(folds)} (期待: 5)"

    def test_runner_nfolds_stored(self, sample_data):
        from extrapolation_discovery_platform.runner import ExperimentRunner
        runner = ExperimentRunner(seeds=[42], quick=True, n_folds=3)
        assert runner._n_folds == 3, f"_n_folds={runner._n_folds} (期待: 3)"

    def test_runner_nfolds_propagated(self, sample_data):
        from extrapolation_discovery_platform.runner import ExperimentRunner
        X, y, comp = sample_data
        runner = ExperimentRunner(seeds=[42], quick=True, n_folds=3)
        runs, _, _ = runner.run(
            comp, X, y,
            selected_workflows=["WF-LIN"],
            selected_feature_sets=["FS_BASE"],
            selected_split_policies=["CompositionBlock"],
        )
        cb_runs = [r for r in runs if r.split_policy == "CompositionBlock"]
        assert len(cb_runs) > 0
        max_fold = max(r.fold for r in cb_runs)
        assert max_fold == 2, f"n_folds=3 だが最大 fold={max_fold} (期待: 2)"

    def test_gui_slider_defined(self):
        from pathlib import Path
        import extrapolation_discovery_platform
        base = Path(extrapolation_discovery_platform.__file__).parent
        with open(base / 'gui' / 'app.py') as f:
            app_src = f.read()
        assert 'n_folds_slider' in app_src, "GUI に n_folds_slider が定義されていない"
        assert 'n_folds_slider,' in app_src, "n_folds_slider が run_btn.click inputs に配線されていない"

class TestT9_JobFactory:
    """T9: _run_job が _IR_FACTORIES に委譲している（二重実装なし）。"""

    def test_ir_factories_used(self):
        from extrapolation_discovery_platform.runner import _run_job
        src = inspect.getsource(_run_job)
        assert "_IR_FACTORIES" in src, "_run_job が _IR_FACTORIES を使っていない"
        assert "_BUILTIN_FACTORIES" not in src, "旧 _BUILTIN_FACTORIES が残存"


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    # pytest なしで直接実行する場合のシンプルなランナー
    import traceback as tb

    tests = [
        TestT1_Stage1Reproducibility,
        TestT2_Stage2Reproducibility,
        TestT3_Stage3Independence,
        TestT4_RunnerDelegation,
        TestT5_IndividualDelegation,
        TestT6_ResultConsistency,
        TestT7_RandomCVDefault,
        TestT8_WorkflowDiversity,
        TestT9_JobFactory,
    ]

    from extrapolation_discovery_platform.features import FeatureCatalog, FeatureSetName
    rng_  = np.random.default_rng(42)
    n_    = 80
    cols_ = FeatureCatalog.columns(FeatureSetName.FS_ALL)
    X_    = pd.DataFrame(rng_.normal(0, 1, (n_, len(cols_))), columns=cols_)
    y_    = pd.Series(rng_.normal(500, 100, n_))
    comp_ = pd.DataFrame(rng_.dirichlet(np.ones(4), n_), columns=["Co","Cr","Fe","Ni"])
    _sd   = (X_, y_, comp_)

    from extrapolation_discovery_platform.pipeline import stage1_preprocess
    prep_ = stage1_preprocess(X_, y_, comp_, [FeatureSetName.FS_ALL.value], ["WF-LIN"],
                               seeds=[42], active_policies=["CompositionBlock"])
    _pr = prep_

    passed = failed = 0
    for cls in tests:
        obj = cls()
        for name in dir(cls):
            if not name.startswith("test_"):
                continue
            method = getattr(obj, name)
            try:
                # fixture の簡易注入
                import inspect as _ins
                params = list(_ins.signature(method).parameters)
                args = []
                for p in params:
                    if p == "sample_data":       args.append(_sd)
                    elif p == "preprocess_result": args.append(_pr)
                method(*args)
                print(f"  ✅ {cls.__name__}.{name}")
                passed += 1
            except Exception as e:
                print(f"  ❌ {cls.__name__}.{name}: {e}")
                tb.print_exc()
                failed += 1

    print()
    print(f"{'🎉 ALL PASSED' if failed == 0 else '❌ FAILED'}: "
          f"{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
