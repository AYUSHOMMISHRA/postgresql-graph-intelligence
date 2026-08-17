"""Tests for the small set of shared leaf utility functions in
`postgres_graph_rag/_db_utils.py` (`normalize_entity`, `content_hash`) that
the secure/tenant path (`tenancy.py`) depends on directly.

This module was originally `database.py`, and tested a `DatabaseManager`
class backing the legacy single-tenant engine. That class was removed
entirely (no customers depended on the single-tenant engine, so it was
deleted rather than kept deprecated) -- these two tests are the only part
of the original file that covered code which still exists. The module was
later renamed from `database.py` to `_db_utils.py`, since after
`DatabaseManager`'s removal it no longer manages a database connection at
all -- just leaf utilities.
"""
from postgres_graph_rag._db_utils import normalize_entity, content_hash


def test_normalize_entity_collapses_whitespace():
    assert normalize_entity("  Apple   Inc.  ") == "Apple Inc."


def test_content_hash_stable_and_sensitive_to_change():
    assert content_hash("abc") == content_hash("abc")
    assert content_hash("abc") != content_hash("abd")
