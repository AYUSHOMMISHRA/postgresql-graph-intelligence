"""Tests for benchmarks/grounding/model_verifier_runner.py -- all against
AlwaysFailingExtractor or a small fake extractor, never a real provider.
No API key or network call anywhere in this file.
"""
import json

import pytest

from benchmarks.grounding.model_verifier_runner import (
    AlwaysFailingExtractor,
    compute_metrics,
    gate_results,
    load_cases,
    required_gates_failed,
    run_all,
    write_markdown_report,
)


@pytest.mark.asyncio
async def test_simulate_failure_marks_every_model_attempted_case_as_failed():
    cases = load_cases("dev")
    outcomes = await run_all(cases, AlwaysFailingExtractor())

    attempted = [o for o in outcomes if o.model_call_attempted]
    assert attempted, "expected at least one case to need the model layer"
    assert all(o.verification_failed for o in attempted)
    assert all(o.verdict is None for o in attempted)

    # Cases the deterministic layer could already decide never attempt a
    # model call at all, even with an always-failing extractor.
    not_attempted = [o for o in outcomes if not o.model_call_attempted]
    assert not_attempted
    assert all(not o.verification_failed for o in not_attempted)


@pytest.mark.asyncio
async def test_verifier_failure_safely_represented_rate_is_100_percent_under_simulated_failure():
    cases = load_cases("dev")
    outcomes = await run_all(cases, AlwaysFailingExtractor())
    metrics = compute_metrics(outcomes, None, None, provider="simulated-failure", model="test-model")
    assert metrics["verifier_failure_safely_represented_rate"] == 1.0
    assert metrics["model_calls_attempted"] > 0
    assert metrics["verification_failure_count"] == metrics["model_calls_attempted"]


@pytest.mark.asyncio
async def test_metrics_match_deterministic_layer_when_model_always_fails():
    """With no model ever succeeding, this run's escape/retention rates
    should match DeterministicVerifier's own recorded result exactly
    (docs/results/grounding-benchmark-deterministic-verifier.md) --
    supported_claim_retention_rate at 66.7%, both escape rates at 0%."""
    cases = load_cases("sealed")
    outcomes = await run_all(cases, AlwaysFailingExtractor())
    metrics = compute_metrics(outcomes, None, None, provider="simulated-failure", model="test-model")

    assert metrics["unsupported_claim_escape_rate"] == 0.0
    assert metrics["contradicted_claim_escape_rate"] == 0.0
    assert round(metrics["supported_claim_retention_rate"], 3) == round(2 / 3, 3)


def test_gate_results_reports_fail_for_known_bad_metrics():
    metrics = {
        "unsupported_claim_escape_rate": 0.0,
        "contradicted_claim_escape_rate": 0.0,
        "supported_claim_retention_rate": 0.667,
        "incorrect_abstention_rate": 0.333,
        "unknown_citation_rejection_rate": 1.0,
        "fabricated_quote_rejection_rate": None,
        "verifier_failure_safely_represented_rate": 1.0,
        "human_verifier_agreement_rate": None,
    }
    gates = gate_results(metrics)
    by_metric = {g["metric"]: g["verdict"] for g in gates}
    assert by_metric["supported_claim_retention_rate"] == "FAIL"
    assert by_metric["incorrect_abstention_rate"] == "FAIL"
    assert by_metric["unsupported_claim_escape_rate"] == "PASS"
    assert by_metric["fabricated_quote_rejection_rate"] == "N/A"
    assert required_gates_failed(gates) is True


def test_required_gates_failed_is_false_when_all_pass_or_na():
    metrics = {
        "unsupported_claim_escape_rate": 0.0,
        "contradicted_claim_escape_rate": 0.0,
        "supported_claim_retention_rate": 1.0,
        "incorrect_abstention_rate": 0.0,
        "unknown_citation_rejection_rate": 1.0,
        "fabricated_quote_rejection_rate": None,
        "verifier_failure_safely_represented_rate": None,
        "human_verifier_agreement_rate": None,
    }
    assert required_gates_failed(gate_results(metrics)) is False


@pytest.mark.asyncio
async def test_cost_estimate_computed_only_when_rates_supplied():
    cases = load_cases("dev")
    outcomes = await run_all(cases, AlwaysFailingExtractor())

    no_rates = compute_metrics(outcomes, None, None, provider="p", model="m")
    assert no_rates["added_cost_per_answer_usd"] is None

    with_rates = compute_metrics(outcomes, 0.001, 0.002, provider="p", model="m")
    # AlwaysFailingExtractor never sets usage, so total tokens are zero and
    # cost is deterministically zero, not None, once rates are supplied.
    assert with_rates["added_cost_per_answer_usd"] == 0.0


def test_write_markdown_report_produces_a_gate_table(tmp_path):
    metrics = {
        "provider": "simulated-failure", "model": "test-model", "case_count": 10,
        "model_calls_attempted": 5, "verification_failure_count": 5,
        "p50_latency_ms": None, "p95_latency_ms": None,
        "total_prompt_tokens": 0, "total_completion_tokens": 0, "total_tokens": 0,
        "added_cost_per_answer_usd": None, "total_cost_usd": None,
    }
    gates = [{"metric": "unsupported_claim_escape_rate", "value": 0.0, "threshold": 0.05,
              "comparison": "<=", "verdict": "PASS", "description": "..."}]
    out_path = tmp_path / "report.md"
    write_markdown_report(out_path, "dev", metrics, gates)

    content = out_path.read_text()
    assert "| unsupported_claim_escape_rate | 0.0% | <= 5% | PASS |" in content
    assert "test-model" in content


def test_load_cases_filters_by_split():
    sealed = load_cases("sealed")
    everything = load_cases("all")
    assert len(sealed) < len(everything)
    assert all(c["split"] == "sealed" for c in sealed)


def test_always_failing_extractor_raises():
    import asyncio

    async def _check():
        with pytest.raises(RuntimeError, match="simulated"):
            await AlwaysFailingExtractor().verify_claims("prompt")

    asyncio.run(_check())


@pytest.mark.asyncio
async def test_metrics_and_gates_are_json_serializable():
    """The runner's --json output path is json.dumps({"metrics": ..., "gates": ...})
    -- confirms compute_metrics()/gate_results()' actual return values
    round-trip cleanly, not just a hand-built stand-in dict."""
    cases = load_cases("dev")
    outcomes = await run_all(cases, AlwaysFailingExtractor())
    metrics = compute_metrics(outcomes, 0.001, 0.002, provider="p", model="m")
    gates = gate_results(metrics)

    payload = json.loads(json.dumps({"metrics": metrics, "gates": gates}))
    assert payload["metrics"]["case_count"] == len(cases)
    assert len(payload["gates"]) == len(gates)
