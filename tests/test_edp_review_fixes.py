import inspect
import math

import numpy as np
import pandas as pd
import pytest


def _generic_frame(n=40):
    rng = np.random.default_rng(11)
    x = pd.DataFrame(
        rng.normal(size=(n, 4)),
        columns=["a", "b", "c", "d"],
    )
    y = pd.Series(np.linspace(100.0, 300.0, n))
    x["leaky"] = y.to_numpy()
    return x, y


def test_train_scope_leakage_keeps_global_diagnostic():
    from extrapolation_discovery_platform.pipeline import stage1_preprocess

    X, y = _generic_frame()
    prep = stage1_preprocess(
        X, y, None, ["generic"], ["WF-LIN"], [42], ["RandomCV"],
        generic_csv_mode=True,
    )
    assert "leaky" in prep.mc_reports["generic"].leak_suspects
    assert "leaky" in prep.effective_cols["generic"]
    assert all(
        "leaky" not in cols
        for cols in prep.fold_selected_cols["generic"]["RandomCV_seed42"]
    )

    prep_no_filter = stage1_preprocess(
        X, y, None, ["generic"], ["WF-LIN"], [42], ["RandomCV"],
        leak_auto_exclude=False,
        generic_csv_mode=True,
    )
    assert all(
        "leaky" in cols
        for cols in prep_no_filter.fold_selected_cols["generic"]["RandomCV_seed42"]
    )


def test_stage2_missing_feature_set_returns_failure():
    from extrapolation_discovery_platform.pipeline import PreprocessResult, stage2_train

    X, y = _generic_frame()
    result = stage2_train(
        PreprocessResult(
            effective_cols={},
            fold_plan={"RandomCV_seed42": [(np.arange(30), np.arange(30, 40))]},
        ),
        X,
        y,
        "WF-LIN",
        "RandomCV",
        "generic",
        generic_csv_mode=True,
    )
    assert result.success is False
    assert "有効列" in result.error_message


def test_shared_trainer_matches_stage2_and_adds_guard_artifact():
    from extrapolation_discovery_platform.pipeline import stage1_preprocess, stage2_train
    from extrapolation_discovery_platform.runner import _Job, _run_job

    rng = np.random.default_rng(5)
    X = pd.DataFrame(rng.normal(size=(24, 4)), columns=list("abcd"))
    y = pd.Series(np.linspace(100.0, 200.0, len(X)))
    prep = stage1_preprocess(
        X, y, None, ["generic"], ["WF-RF"], [42], ["RandomCV"],
        leak_auto_exclude=False,
    )
    staged = stage2_train(
        prep, X, y, "WF-RF", "RandomCV", "generic",
        quick=True, generic_csv_mode=True,
    )
    assert staged.success
    split = prep.fold_plan["RandomCV_seed42"][0]
    cols = prep.fold_selected_cols["generic"]["RandomCV_seed42"][0]
    job = _Job(
        "WF-RF", "generic", "RandomCV", 42, 0,
        split[0], split[1], True, fold_cols=tuple(cols),
    )
    direct = _run_job(job, X.to_numpy(), list(X.columns), y.to_numpy())
    assert math.isclose(
        staged.runs[0].rmse_test, direct.rmse_test, rel_tol=0, abs_tol=1e-9
    )
    assert "n_pred_clipped" in staged.runs[0].artifacts
    assert "n_pred_clipped" in direct.artifacts


def test_ood_constant_range_is_informative():
    from extrapolation_discovery_platform.ood import OODDetector

    out = OODDetector._combine(
        np.array([2.0, 3.0]), np.array([1.0, 1.0]),
        2.0, 2.0, 1.0, 1.0,
    )
    assert out[0] == 0.0
    assert out[1] > 0.0


def test_ood_ensemble_threshold_matches_flag():
    from extrapolation_discovery_platform.pipeline import stage3_detect_ood

    rng = np.random.default_rng(8)
    X = pd.DataFrame(rng.normal(size=(20, 3)), columns=list("abc"))
    train = np.arange(15)
    test = np.arange(15, 20)
    out = stage3_detect_ood(
        X, list(X.columns), {"RandomCV": [(train, test)]}
    )
    assert out.success, out.error_message
    ood = out.ood_result
    assert ood is not None
    assert np.array_equal(
        ood.composite_scores > ood.ood_threshold,
        ood.is_ood,
    )


def test_extrapolation_safety_is_bounded():
    from extrapolation_discovery_platform.evaluation import FeatureValidityEvaluator

    score = FeatureValidityEvaluator._extrapolation_safety({
        "errors": np.array([1.0, 0.1]),
        "uncertainties": np.array([1.0, 2.0]),
        "is_ood": np.array([False, True]),
    })
    assert score <= 1.0
    assert math.isclose(score, 0.6 * 1.0 + 0.4 * 1.0)


def test_composition_zero_fraction_and_negative_validation():
    from extrapolation_discovery_platform.features import compute_features_single

    one = compute_features_single({"Fe": 1.0})
    zero = compute_features_single({"Fe": 1.0, "Ni": 0.0})
    assert one == zero
    with pytest.raises(ValueError):
        compute_features_single({"Fe": 1.2, "Ni": -0.2})


def test_group_aware_inner_cv():
    from extrapolation_discovery_platform.workflows import _inner_cv

    groups = np.repeat(np.arange(4), 5)
    splits = _inner_cv(5, len(groups), groups, 42)
    assert isinstance(splits, list)
    assert len(splits) == 4
    for train, test in splits:
        assert not set(groups[train]) & set(groups[test])
    assert isinstance(_inner_cv(5, len(groups), None, 42), int)


def test_zero_rmse_is_valid_and_default_is_nan():
    from extrapolation_discovery_platform.evaluation import FeatureValidityEvaluator
    from extrapolation_discovery_platform.workflows import RunResult

    run = RunResult("WF-LIN", "generic", "RandomCV", 42, 0, rmse_test=0.0)
    assert FeatureValidityEvaluator._mean_test_rmse([run]) == 0.0
    assert math.isnan(
        RunResult("WF-LIN", "generic", "RandomCV", 42, 0).rmse_test
    )


def test_incomplete_candidate_is_excluded():
    from extrapolation_discovery_platform.model_selection import (
        CandidateResult,
        _summarize_candidate,
    )

    candidate = CandidateResult("RF", outer_scores=[1.0, float("nan")])
    _summarize_candidate(candidate)
    assert candidate.n_failed_folds == 1
    assert candidate.mean_rmse == float("inf")
    assert candidate.std_rmse == float("inf")


def test_workflow_lin_uses_group_aware_ridge_cv():
    from extrapolation_discovery_platform.workflows import WorkflowLIN

    rng = np.random.default_rng(21)
    X = pd.DataFrame(rng.normal(size=(30, 3)), columns=list("abc"))
    y = pd.Series(rng.normal(size=30))
    groups = np.repeat(np.arange(6), 5)
    run = WorkflowLIN(dim_reduction=False).run(
        X,
        y,
        X.iloc[:5],
        y.iloc[:5],
        seed=42,
        groups=groups,
    )
    assert math.isfinite(run.params["alpha"])
    assert "_inner_cv" in inspect.getsource(WorkflowLIN.run)


def test_precomputed_individual_path_reuses_split_preprocessing(monkeypatch):
    import extrapolation_discovery_platform.pipeline as pipeline
    from extrapolation_discovery_platform.individual_runner import run_individual

    rng = np.random.default_rng(31)
    X = pd.DataFrame(rng.normal(size=(40, 4)), columns=list("abcd"))
    y = pd.Series(np.linspace(100.0, 200.0, len(X)))
    compositions = pd.DataFrame(
        rng.dirichlet(np.ones(4), len(X)),
        columns=["Co", "Cr", "Fe", "Ni"],
    )
    compositions.iloc[0, 0] = np.nan
    captured = {}
    original_stage2 = pipeline.stage2_train

    def capture_stage2(*args, **kwargs):
        captured["prep"] = kwargs["preprocess_result"]
        return original_stage2(*args, **kwargs)

    monkeypatch.setattr(pipeline, "stage2_train", capture_stage2)
    result = run_individual(
        "WF-LIN",
        "generic",
        "CompositionBlock",
        features_df=X,
        target=y,
        compositions_df=compositions,
        seed=42,
        n_folds=3,
        quick=True,
        precomputed_columns=list(X.columns),
        generic_csv_mode=True,
    )
    assert result.success, result.error_message
    prep = captured["prep"]
    assert prep.fold_plan
    assert prep.mc_reports["generic"] is not None
