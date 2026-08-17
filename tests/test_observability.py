"""Tests for the optional OpenTelemetry sink (postgres_graph_rag/observability.py)."""
from unittest.mock import MagicMock, patch

import pytest

from postgres_graph_rag.observability import Event, make_otel_sink


def test_make_otel_sink_creates_counter_and_histogram():
    pytest.importorskip("opentelemetry")
    mock_meter = MagicMock()
    mock_counter = MagicMock()
    mock_histogram = MagicMock()
    mock_meter.create_counter.return_value = mock_counter
    mock_meter.create_histogram.return_value = mock_histogram

    with patch("opentelemetry.metrics.get_meter", return_value=mock_meter) as mock_get_meter:
        sink = make_otel_sink(meter_name="test_meter")

    mock_get_meter.assert_called_once_with("test_meter")
    mock_meter.create_counter.assert_called_once_with(
        "postgres_graph_rag.events", description="Event counts by kind"
    )
    mock_meter.create_histogram.assert_called_once_with(
        "postgres_graph_rag.duration_ms", description="Event durations"
    )
    assert sink is not None


def test_otel_sink_emit_with_duration_records_histogram():
    pytest.importorskip("opentelemetry")
    mock_meter = MagicMock()
    mock_counter = MagicMock()
    mock_histogram = MagicMock()
    mock_meter.create_counter.return_value = mock_counter
    mock_meter.create_histogram.return_value = mock_histogram

    with patch("opentelemetry.metrics.get_meter", return_value=mock_meter):
        sink = make_otel_sink()

    event = Event(
        kind="ingestion.completed",
        correlation_id="c1",
        tenant_id="t1",
        namespace="ns",
        duration_ms=42.0,
    )
    sink.emit(event)

    mock_counter.add.assert_called_once_with(
        1, {"kind": "ingestion.completed", "tenant_id": "t1", "namespace": "ns"}
    )
    mock_histogram.record.assert_called_once_with(
        42.0, {"kind": "ingestion.completed", "tenant_id": "t1", "namespace": "ns"}
    )


def test_otel_sink_emit_without_duration_skips_histogram():
    pytest.importorskip("opentelemetry")
    mock_meter = MagicMock()
    mock_counter = MagicMock()
    mock_histogram = MagicMock()
    mock_meter.create_counter.return_value = mock_counter
    mock_meter.create_histogram.return_value = mock_histogram

    with patch("opentelemetry.metrics.get_meter", return_value=mock_meter):
        sink = make_otel_sink()

    event = Event(kind="cache.hit", correlation_id="c2")  # duration_ms defaults to None
    sink.emit(event)

    mock_counter.add.assert_called_once()
    mock_histogram.record.assert_not_called()


def test_make_otel_sink_returns_none_when_opentelemetry_missing():
    """Doesn't need the real package installed: forces the `from
    opentelemetry import metrics` import inside make_otel_sink() to raise
    ImportError regardless of what's actually on this machine, by mapping
    the module name to None in sys.modules for the duration of the test."""
    with patch.dict("sys.modules", {"opentelemetry": None, "opentelemetry.metrics": None}):
        assert make_otel_sink() is None
