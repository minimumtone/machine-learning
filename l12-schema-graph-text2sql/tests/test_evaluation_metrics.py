"""Tests for evaluation.metrics."""
from evaluation.metrics import (
    syntax_validity,
    execution_validity,
    execution_accuracy,
    hallucinated_table_rate,
    hallucinated_column_rate,
    hallucinated_join_rate,
    multi_hop_success,
)


def test_syntax_validity_valid():
    assert syntax_validity("SELECT * FROM t LIMIT 10;")


def test_execution_validity_true():
    assert execution_validity({"success": True})


def test_execution_validity_false():
    assert not execution_validity({"success": False})


def test_execution_accuracy_exact():
    result = [[1, "a"], [2, "b"]]
    expected = [[1, "a"], [2, "b"]]
    assert execution_accuracy(result, expected) == 1.0


def test_execution_accuracy_partial():
    result = [[1, "a"]]
    expected = [[1, "a"], [2, "b"]]
    assert execution_accuracy(result, expected) == 0.5


def test_hallucinated_table_rate_none():
    rate = hallucinated_table_rate(
        ["material_entry", "structure"],
        ["material_entry", "structure", "composition"],
    )
    assert rate == 0.0


def test_hallucinated_table_rate_one():
    rate = hallucinated_table_rate(
        ["material_entry", "secret"],
        ["material_entry", "structure"],
    )
    assert rate == 0.5


def test_hallucinated_column_rate():
    rate = hallucinated_column_rate(
        ["m.formula", "m.fake"],
        ["m.formula"],
    )
    assert rate == 0.5


def test_hallucinated_join_rate():
    # Fix B5: first arg is now a SQL string, not a list of joins
    sql = "SELECT * FROM a JOIN b ON a.x = b.y"
    rate = hallucinated_join_rate(
        sql,
        ["a.x = b.y", "c.z = d.w"],
    )
    assert rate == 0.0


def test_hallucinated_join_rate_alias_resolution():
    """Test that alias-form joins are resolved to table-form before comparison."""
    # Fix B5: first arg is now a SQL string with aliases
    sql = ("SELECT * FROM material_entry AS m "
           "JOIN structure AS s ON s.entry_id = m.entry_id "
           "JOIN phase_stability AS ps ON ps.entry_id = m.entry_id")
    rate = hallucinated_join_rate(
        sql,
        ["structure.entry_id = material_entry.entry_id",
         "phase_stability.entry_id = material_entry.entry_id"],
    )
    assert rate == 0.0

    # Reversed order should also match (canonical sorting)
    sql2 = "SELECT * FROM material_entry AS m JOIN structure AS s ON m.entry_id = s.entry_id"
    rate2 = hallucinated_join_rate(
        sql2,
        ["structure.entry_id = material_entry.entry_id"],
    )
    assert rate2 == 0.0

    # Truly hallucinated join should still be caught
    sql3 = ("SELECT * FROM material_entry AS m "
            "JOIN structure AS s ON s.entry_id = m.entry_id "
            "JOIN x ON x.foo = y.bar")
    rate3 = hallucinated_join_rate(
        sql3,
        ["structure.entry_id = material_entry.entry_id"],
    )
    assert rate3 == 0.5


def test_hallucinated_join_rate_word_boundary_b16():
    """Fix B16: 'ON' in 'composition'/'calculation' must not be treated as ON clause."""
    sql = ("SELECT m.formula, c.element, c.fraction "
           "FROM material_entry m "
           "JOIN composition c ON c.entry_id = m.entry_id "
           "WHERE c.element = 'Ni'")
    rate = hallucinated_join_rate(
        sql,
        ["composition.entry_id = material_entry.entry_id"],
    )
    # Without \b, the regex matches "compositi*ON*" producing a bogus join → rate > 0
    assert rate == 0.0

    sql2 = ("SELECT * FROM calculation cal "
            "JOIN calculated_property cp ON cp.calc_id = cal.calc_id")
    rate2 = hallucinated_join_rate(
        sql2,
        ["calculated_property.calc_id = calculation.calc_id"],
    )
    assert rate2 == 0.0


def test_hallucinated_join_rate_cte_b17():
    """Fix B17: JOINs to CTE names should not count as hallucinated."""
    sql = ("WITH stable AS ("
           "  SELECT entry_id FROM phase_stability WHERE energy_above_hull <= 0.001"
           ") "
           "SELECT m.formula FROM material_entry m "
           "JOIN stable s ON s.entry_id = m.entry_id")
    rate = hallucinated_join_rate(
        sql,
        ["material_entry.entry_id = phase_stability.entry_id"],
    )
    # The join is to the CTE 'stable', not a hallucinated table
    assert rate == 0.0


def test_multi_hop_success():
    result = multi_hop_success(3, True)
    assert result["is_multi_hop"]
    assert result["correct"]


def test_exact_result_set_match_canonical():
    from evaluation.metrics_strict import exact_result_set_match

    rows = [["Ni3Al", 1.0], ["Cu3Au", 2.0]]
    cols = ["formula", "delta_e"]

    # identical -> exact
    assert exact_result_set_match(rows, rows, cols, cols)
    # missing gold column -> not exact
    assert not exact_result_set_match(
        [["Ni3Al"], ["Cu3Au"]], rows, ["formula"], cols)
    # extra column -> not exact
    assert not exact_result_set_match(
        [r + ["x"] for r in rows], rows, cols + ["extra"], cols)
    # column order mismatch -> not exact
    assert not exact_result_set_match(
        [[r[1], r[0]] for r in rows], rows, ["delta_e", "formula"], cols)
    # duplicate-row multiplicity mismatch -> not exact
    assert not exact_result_set_match(rows + [rows[0]], rows, cols, cols)
    # unordered: permutation is exact
    assert exact_result_set_match(list(reversed(rows)), rows, cols, cols)
    # ordered: permutation is NOT exact
    assert not exact_result_set_match(
        list(reversed(rows)), rows, cols, cols, ordered=True)
    assert exact_result_set_match(rows, rows, cols, cols, ordered=True)


def test_exact_result_set_match_value_strictness():
    from decimal import Decimal

    from evaluation.metrics_strict import exact_result_set_match

    cols = ["v"]
    # type discrimination: number vs string, bool vs string, NULL vs sentinel
    assert not exact_result_set_match([[1]], [["1"]], cols, cols)
    assert not exact_result_set_match([[True]], [["true"]], cols, cols)
    assert not exact_result_set_match([[None]], [["__NULL__"]], cols, cols)
    assert not exact_result_set_match([[True]], [[1]], cols, cols)
    # string case is significant in values
    assert not exact_result_set_match([["Ni"]], [["ni"]], cols, cols)
    assert exact_result_set_match([["Ni"]], [["Ni"]], cols, cols)
    # numeric types unify across driver Decimal / JSON float / int
    assert exact_result_set_match([[Decimal("1.5")]], [[1.5]], cols, cols)
    assert exact_result_set_match([[Decimal("2")]], [[2]], cols, cols)
    # column names stay case-insensitive
    assert exact_result_set_match([[1]], [[1]], ["V"], ["v"])


def test_eval_ablation_load_expected_returns_ordered(tmp_path, monkeypatch):
    import json

    import scripts.eval_ablation as ea

    monkeypatch.setattr(ea, "RESULTS_DIR", tmp_path)
    # Model scoring follows semantic_ordered (the question's own ordering
    # requirement), not the gold-storage ordered flag.
    (tmp_path / "q_x.json").write_text(json.dumps(
        {"rows": [[1], [2]], "columns": ["v"], "ordered": True,
         "semantic_ordered": True}))
    rows, columns, ordered = ea.load_expected("q_x")
    assert (rows, columns, ordered) == ([[1], [2]], ["v"], True)
    (tmp_path / "q_y.json").write_text(json.dumps(
        {"rows": [[1], [2]], "columns": ["v"], "ordered": True,
         "semantic_ordered": False}))
    assert ea.load_expected("q_y") == ([[1], [2]], ["v"], False)
    assert ea.load_expected("q_missing") == ([], [], False)

    # ordered=true: same row set in a different order must not be exact
    from evaluation.metrics_strict import exact_result_set_match
    assert not exact_result_set_match([[2], [1]], rows, columns, columns,
                                      ordered=ordered)


def test_verify_all_provenance_new_keys_tamper_detection(tmp_path, monkeypatch):
    import json

    import pytest

    import scripts.verify_all as va
    from scripts.provenance import build_provenance

    project = tmp_path
    eval_dir = project / "evaluation"
    prompts = project / "llm" / "prompt_templates"
    gold = eval_dir / "gold_sql"
    expected = eval_dir / "expected_results"
    for d in (eval_dir, prompts, gold, expected, eval_dir / "generated_sql"):
        d.mkdir(parents=True)
    (project / "GIT_COMMIT").write_text("deadbeef\n")
    dataset = eval_dir / "evaluation_dataset.jsonl"
    dataset.write_text('{"id": "q1", "question": "x"}\n')
    (gold / "q1.sql").write_text("SELECT 1;\n")
    prompt = prompts / "sql_generation_prompt.md"
    prompt.write_text("template v1\n")
    exp_file = expected / "q1.json"
    exp_file.write_text('{"rows": [[1]], "columns": ["v"], "ordered": false}')

    prov = build_provenance(dataset, gold_dir=gold, prompt_path=prompt,
                            expected_dir=expected)
    prov["git_commit"] = "deadbeef"
    result = eval_dir / "some_eval_results.json"
    result.write_text(json.dumps({
        "model": "m", "provenance": prov,
        "results": [{"qid": "q1", "sql": "SELECT 1;"}],
    }))

    monkeypatch.setattr(va, "PROJECT", project)
    monkeypatch.setattr(va, "EVAL", eval_dir)
    monkeypatch.setattr(va, "PROMPTS", prompts)

    summary, warnings = va.check_provenance()
    assert "1 evaluation provenance blocks" in summary

    # tampering the prompt template must fail static verification
    prompt.write_text("template v2 tampered\n")
    with pytest.raises(va.VerifyError, match="prompt_template_sha256"):
        va.check_provenance()
    prompt.write_text("template v1\n")

    # tampering an expected-result JSON must fail static verification
    exp_file.write_text('{"rows": [[2]], "columns": ["v"], "ordered": false}')
    with pytest.raises(va.VerifyError, match="expected_sha256"):
        va.check_provenance()
    exp_file.write_text('{"rows": [[1]], "columns": ["v"], "ordered": false}')

    # provenance directories must be the ones the dataset itself points
    # at, even when the recorded directories' hashes are self-consistent
    gold2 = eval_dir / "gold_sql_other"
    expected2 = eval_dir / "expected_results_other"
    gold2.mkdir()
    expected2.mkdir()
    (gold2 / "q1.sql").write_text("SELECT 2;\n")
    (expected2 / "q1.json").write_text(
        '{"rows": [[2]], "columns": ["v"], "ordered": false}')
    dataset.write_text(
        '{"id": "q1", "question": "x",'
        ' "gold_sql_path": "gold_sql_other/q1.sql",'
        ' "expected_result_path": "expected_results_other/q1.json"}\n')
    prov2 = build_provenance(dataset, gold_dir=gold, prompt_path=prompt,
                             expected_dir=expected)
    prov2["git_commit"] = "deadbeef"
    result.write_text(json.dumps({
        "model": "m", "provenance": prov2,
        "results": [{"qid": "q1", "sql": "SELECT 1;"}],
    }))
    with pytest.raises(va.VerifyError,
                       match="does not match dataset gold_sql_path"):
        va.check_provenance()


def test_model_comparison_config_hash_staleness():
    import pytest

    from scripts.eval_model_comparison import models_config_sha256
    from scripts.provenance import assert_resumable

    base = {"dataset_sha256": "d", "gold_sha256": "g",
            "prompt_template_sha256": "p", "model": "m", "git_commit": "c"}
    stored = dict(base, models_config_sha256=models_config_sha256(
        [{"name": "gpt-4o", "provider": "openai", "model_id": "gpt-4o"}]))
    current_same = dict(stored)
    assert_resumable(stored, current_same,
                     extra_keys=("models_config_sha256",))

    changed = dict(base, models_config_sha256=models_config_sha256(
        [{"name": "gpt-4o", "provider": "openai", "model_id": "gpt-4o-mini"}]))
    with pytest.raises(RuntimeError):
        assert_resumable(stored, changed,
                         extra_keys=("models_config_sha256",))
