"""Tests for postgres_graph_rag/studio_cli.py -- the console entry point's
own argument handling (loopback-bind enforcement, missing-extra message),
never actually binding a real port (uvicorn.run is always mocked/never
reached in a passing case here)."""
import sys
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed -- install the 'studio' extra (uv sync --extra studio)")

from postgres_graph_rag import studio_cli  # noqa: E402


def _run_main(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["postgres-graph-rag-studio", *argv])
    studio_cli.main()


def test_refuses_non_loopback_bind_by_default(monkeypatch):
    monkeypatch.setenv("PGR_RUNTIME_URL", "postgresql://pgr_runtime:pw@localhost:5432/graph_rag_test")
    with pytest.raises(SystemExit, match="Refusing to bind"):
        _run_main(["--host", "0.0.0.0"], monkeypatch)


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_allows_loopback_hosts_without_override(monkeypatch, host):
    monkeypatch.setenv("PGR_RUNTIME_URL", "postgresql://pgr_runtime:pw@localhost:5432/graph_rag_test")
    with patch("uvicorn.run", new=MagicMock()) as mocked_run:
        _run_main(["--host", host], monkeypatch)
    mocked_run.assert_called_once()


def test_allows_non_loopback_bind_with_explicit_override(monkeypatch):
    monkeypatch.setenv("PGR_RUNTIME_URL", "postgresql://pgr_runtime:pw@localhost:5432/graph_rag_test")
    with patch("uvicorn.run", new=MagicMock()) as mocked_run:
        _run_main(["--host", "0.0.0.0", "--allow-non-loopback-bind"], monkeypatch)
    mocked_run.assert_called_once()


def test_requires_runtime_url(monkeypatch):
    monkeypatch.delenv("PGR_RUNTIME_URL", raising=False)
    with pytest.raises(SystemExit, match="PGR_RUNTIME_URL is required"):
        _run_main([], monkeypatch)


def test_missing_provider_key_reports_a_clean_startup_error(monkeypatch):
    monkeypatch.setenv("PGR_RUNTIME_URL", "postgresql://pgr_runtime:pw@localhost:5432/graph_rag_test")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="Cannot start the Studio"):
        _run_main(["--provider", "openai"], monkeypatch)


def test_litellm_provider_starts_with_complete_environment(monkeypatch):
    monkeypatch.setenv("PGR_RUNTIME_URL", "postgresql://pgr_runtime:pw@localhost:5432/graph_rag_test")
    monkeypatch.setenv("LITELLM_API_KEY", "sk-test")
    monkeypatch.setenv("LITELLM_BASE_URL", "http://localhost:4000/v1")
    monkeypatch.setenv("LITELLM_CHAT_MODEL", "graph-chat")
    monkeypatch.setenv("LITELLM_EMBEDDING_MODEL", "graph-embed")
    monkeypatch.setenv("LITELLM_EMBEDDING_DIMENSION", "768")
    with patch("uvicorn.run", new=MagicMock()) as mocked_run:
        _run_main(["--provider", "litellm"], monkeypatch)
    mocked_run.assert_called_once()
