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
    metrics = compute_metrics(
        outcomes, None, None, provider="simulated-failure", model="test-model",
        is_simulated_failure_run=True,
    )
    assert metrics["verifier_failure_safely_represented_rate"] == 1.0
    assert metrics["model_calls_attempted"] > 0
    assert metrics["verification_failure_count"] == metrics["model_calls_attempted"]


@pytest.mark.asyncio
async def test_verifier_failure_safely_represented_rate_is_none_outside_simulated_failure():
    """The metric must never be computed from a live run's incidental
    failure count -- only from the dedicated --simulate-failure exercise.
    Passing is_simulated_failure_run=False (the live-run default) with the
    exact same always-failing outcomes must still report None, not a
    misleadingly-computed rate."""
    cases = load_cases("dev")
    outcomes = await run_all(cases, AlwaysFailingExtractor())
    metrics = compute_metrics(
        outcomes, None, None, provider="openai", model="test-model",
        is_simulated_failure_run=False,
    )
    assert metrics["verifier_failure_safely_represented_rate"] is None


@pytest.mark.asyncio
async def test_metrics_match_deterministic_layer_when_model_always_fails():
    """With no model call ever succeeding, every case that needed the
    model layer is *excluded* from the escape/retention denominators (an
    incomplete result, not a rejection or an acceptance) -- so
    supported_claim_retention_rate reflects only the cases the
    deterministic layer could resolve on its own, which are all correctly
    retained (100%), not diluted by the excluded incomplete ones down to
    66.7% the way folding them in as "not retained" would produce."""
    cases = load_cases("sealed")
    outcomes = await run_all(cases, AlwaysFailingExtractor())
    metrics = compute_metrics(outcomes, None, None, provider="simulated-failure", model="test-model")

    assert metrics["unsupported_claim_escape_rate"] == 0.0
    assert metrics["contradicted_claim_escape_rate"] == 0.0
    assert metrics["supported_claim_retention_rate"] == 1.0
    assert metrics["verification_failure_count"] > 0


@pytest.mark.asyncio
async def test_outage_cannot_improve_escape_or_rejection_rates():
    """Core regression test for the fix: construct a mix where some
    contradicted-labeled cases resolve to a (bad) 'supported' verdict and
    others are verification_failed, and confirm the reported escape rate
    reflects only the resolved cases -- an outage must never dilute an
    escape rate to look better than the verifier's actual behavior on the
    cases it actually completed.
    """
    def outcome(label, verdict, failed):
        from benchmarks.grounding.model_verifier_runner import CaseOutcome
        return CaseOutcome(
            case={"label": label, "asserted_citation_ids": ["a#0"],
                  "retrieved_evidence": [{"source_id": "a", "ordinal": 0, "content": "x"}]},
            verdict=verdict, verification_failed=failed, model_call_attempted=True,
        )

    # 2 contradicted cases actually resolved, both correctly NOT escaping,
    # plus 8 more contradicted cases that are incomplete (outage) -- if the
    # outage cases were folded into the denominator as "not escaped", the
    # rate would look even better (0/10 = 0%) than the true, honest
    # accounting over what was actually resolved (0/2 = 0%, same in this
    # all-good case) -- so also add one resolved case that DID escape to
    # make the two computations diverge and actually prove the fix.
    outcomes = [
        outcome("contradicted", "supported", False),  # 1 real escape
        outcome("contradicted", "insufficient", False),  # 1 real non-escape
        *[outcome("contradicted", None, True) for _ in range(8)],  # 8 incomplete (outage)
    ]
    metrics = compute_metrics(outcomes, None, None, provider="p", model="m")

    # Honest rate: 1 escape out of 2 *resolved* cases = 50%.
    assert metrics["contradicted_claim_escape_rate"] == 0.5
    # The buggy version would have computed 1/10 = 10%, since the 8
    # incomplete cases would count as "not escaped" in the denominator --
    # diluting the true 50% escape rate down to a falsely reassuring 10%.
    assert metrics["contradicted_claim_escape_rate"] != 0.1
    assert metrics["verification_failure_count"] == 8


@pytest.mark.asyncio
async def test_outage_cannot_improve_unknown_citation_rejection_rate():
    """An outage's verdict is None, which satisfies `!= "supported"` --
    the exact predicate unknown_citation_rejection_rate used to use,
    meaning a provider failure was indistinguishable from a correct
    rejection. Confirms an all-outage batch reports None (no resolved
    cases to measure), not a misleading 100%."""
    def outcome():
        from benchmarks.grounding.model_verifier_runner import CaseOutcome
        return CaseOutcome(
            case={"label": "insufficient", "asserted_citation_ids": ["missing#0"],
                  "retrieved_evidence": [{"source_id": "a", "ordinal": 0, "content": "x"}]},
            verdict=None, verification_failed=True, model_call_attempted=True,
        )

    outcomes = [outcome() for _ in range(5)]
    metrics = compute_metrics(outcomes, None, None, provider="p", model="m")
    assert metrics["unknown_citation_rejection_rate"] is None
    assert metrics["verification_failure_count"] == 5


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


def test_required_gates_failed_strict_mode_fails_on_na():
    """The correction this test locks in: a required gate that's N/A
    (never measured -- e.g. human_verifier_agreement_rate before review,
    or fabricated_quote_rejection_rate if that check didn't run) must
    fail a strict release-gate check, even though every gate that *was*
    measured passes. Permissive mode (the default, for dev/calibration
    iteration) must NOT fail on the same input."""
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
    gates = gate_results(metrics)
    assert required_gates_failed(gates, strict=False) is False
    assert required_gates_failed(gates, strict=True) is True


@pytest.mark.asyncio
async def test_run_fabrication_check_rejects_fabricated_quotes():
    from benchmarks.grounding.model_verifier_runner import run_fabrication_check

    rate = await run_fabrication_check()
    assert rate is not None
    # A stub that claims the fabricated quote supports the claim must be
    # rejected every time -- ModelEntailmentVerifier's server-side
    # validation never trusts an unverified model quote.
    assert rate == 1.0


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
