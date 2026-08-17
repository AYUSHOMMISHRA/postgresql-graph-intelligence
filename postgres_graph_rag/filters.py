"""Typed metadata filter DSL: equality, comparison, range, and membership
filters over JSONB metadata columns, compiled to safe parameterized SQL —
never raw SQL from a caller.

Two shapes are accepted, kept backward compatible with the original
containment-only filter that `hybrid_search`/`traverse_graph` already
shipped with:

  - a plain dict, e.g. `{"source": "handbook"}` -> JSONB containment
    (`metadata @> filter`), unchanged from the original behavior.
  - a list of clauses, e.g. `[{"field": "updated_at", "op": "gte",
    "value": "2024-01-01"}]` -> ANDed comparisons compiled from Python
    values. Supports `eq`/`neq`/`gt`/`gte`/`lt`/`lte`/`in`. Numeric Python
    values compare numerically; ISO-8601-looking strings compare as
    timestamps (this is what makes "documents updated in the last 90 days"
    possible — plain containment can't express a range at all); everything
    else falls back to text comparison. `in` always compares as text.

Field names are always passed as bind parameters to the `->>'` operator,
never string-interpolated into the query — the same discipline the rest of
this codebase uses for user-supplied values, not a new exception for this
module. There is no way for a caller-supplied `field` or `value` to inject
SQL here, regardless of what's passed.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Tuple, Union

MetadataFilter = Union[None, Dict[str, Any], List[Dict[str, Any]]]

_OPS = {"eq": "=", "neq": "<>", "gt": ">", "gte": ">=", "lt": "<", "lte": "<="}


def _looks_like_datetime(value: str) -> bool:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return True
    except ValueError:
        return False


def compile_metadata_filter(
    filter_: MetadataFilter, column: str = "metadata", param_prefix: str = "mf"
) -> Tuple[str, Dict[str, Any]]:
    """Compiles `filter_` into a single boolean SQL expression plus its
    bind parameters. Returns `("TRUE", {})` for an empty/None filter, so
    callers can always unconditionally AND the result into a WHERE clause.

    `param_prefix` must be unique per call site if compiling more than one
    filter into the same query (e.g. a lexical and a semantic CTE that each
    need their own copy of the same filter) — otherwise parameter names
    collide.
    """
    if not filter_:
        return "TRUE", {}

    if isinstance(filter_, dict):
        key = f"{param_prefix}_containment"
        return f"{column} @> %({key})s::jsonb", {key: json.dumps(filter_)}

    if isinstance(filter_, list):
        if not filter_:
            return "TRUE", {}
        clauses: List[str] = []
        params: Dict[str, Any] = {}
        for i, clause in enumerate(filter_):
            field = clause["field"]
            op = clause["op"]
            value = clause["value"]
            field_param = f"{param_prefix}_field_{i}"
            value_param = f"{param_prefix}_value_{i}"
            params[field_param] = field

            if op == "in":
                if not isinstance(value, (list, tuple)) or not value:
                    raise ValueError("metadata filter op='in' requires a non-empty list value")
                params[value_param] = [str(v) for v in value]
                clauses.append(f"({column}->>%({field_param})s) = ANY(%({value_param})s::text[])")
                continue

            if op not in _OPS:
                raise ValueError(f"Unsupported metadata filter op: {op!r}")

            if isinstance(value, bool):
                cast = "::boolean"
            elif isinstance(value, (int, float)):
                cast = "::numeric"
            elif isinstance(value, str) and _looks_like_datetime(value):
                cast = "::timestamptz"
            else:
                cast = ""

            params[value_param] = value
            clauses.append(f"({column}->>%({field_param})s){cast} {_OPS[op]} %({value_param})s{cast}")

        return " AND ".join(clauses), params

    raise ValueError(f"Unsupported metadata_filter type: {type(filter_)!r}")
