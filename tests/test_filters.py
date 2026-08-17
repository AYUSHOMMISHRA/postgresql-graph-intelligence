"""Unit tests for the typed metadata filter DSL (no database required —
these test SQL compilation, not execution)."""
import pytest

from postgres_graph_rag.filters import compile_metadata_filter


def test_none_filter_compiles_to_unconditional_true():
    sql, params = compile_metadata_filter(None)
    assert sql == "TRUE"
    assert params == {}


def test_empty_list_filter_compiles_to_unconditional_true():
    sql, params = compile_metadata_filter([])
    assert sql == "TRUE"
    assert params == {}


def test_dict_filter_compiles_to_containment():
    sql, params = compile_metadata_filter({"source": "handbook"})
    assert "@>" in sql
    assert list(params.values())[0] == '{"source": "handbook"}'


def test_comparison_ops_compile_with_expected_operators():
    for op, expected_operator in [
        ("eq", "="), ("neq", "<>"), ("gt", ">"), ("gte", ">="), ("lt", "<"), ("lte", "<="),
    ]:
        sql, params = compile_metadata_filter([{"field": "priority", "op": op, "value": 3}])
        assert expected_operator in sql
        assert "::numeric" in sql  # int value -> numeric cast
        assert 3 in params.values()


def test_datetime_looking_string_gets_timestamptz_cast():
    sql, _ = compile_metadata_filter([{"field": "updated_at", "op": "gte", "value": "2024-01-01T00:00:00Z"}])
    assert "::timestamptz" in sql


def test_plain_string_gets_no_cast():
    sql, _ = compile_metadata_filter([{"field": "status", "op": "eq", "value": "active"}])
    assert "::timestamptz" not in sql
    assert "::numeric" not in sql


def test_in_op_requires_nonempty_list():
    with pytest.raises(ValueError):
        compile_metadata_filter([{"field": "status", "op": "in", "value": []}])
    with pytest.raises(ValueError):
        compile_metadata_filter([{"field": "status", "op": "in", "value": "not-a-list"}])


def test_unsupported_op_raises():
    with pytest.raises(ValueError):
        compile_metadata_filter([{"field": "x", "op": "regex", "value": ".*"}])


def test_multiple_clauses_are_anded():
    sql, params = compile_metadata_filter([
        {"field": "priority", "op": "gte", "value": 1},
        {"field": "status", "op": "eq", "value": "archived"},
    ])
    assert " AND " in sql
    assert len(params) == 4  # 2 field params + 2 value params


def test_field_name_is_always_a_bind_parameter_not_interpolated():
    """The core safety property: no matter what a caller passes as
    `field`, it never ends up interpolated into the SQL text — only valid
    parameter placeholders (%(...)s) should appear there."""
    malicious_field = "x'); DROP TABLE graph_nodes; --"
    sql, params = compile_metadata_filter([{"field": malicious_field, "op": "eq", "value": "y"}])
    assert malicious_field not in sql
    assert malicious_field in params.values()


def test_unique_param_prefix_avoids_collisions_across_two_compiled_filters():
    sql1, params1 = compile_metadata_filter([{"field": "a", "op": "eq", "value": 1}], param_prefix="left")
    sql2, params2 = compile_metadata_filter([{"field": "a", "op": "eq", "value": 2}], param_prefix="right")
    merged = {**params1, **params2}
    assert len(merged) == 4  # no key collisions
