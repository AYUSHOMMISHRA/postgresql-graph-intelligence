"""Tests for benchmarks/grounding/model_verifier_runner.py -- all against
AlwaysFailingExtractor or a small fake extractor, never a real provider.
No API key or network call anywhere in this file.
"""
import argparse
import json

import pytest

from benchmarks.grounding.model_verifier_runner import (
    AlwaysFailingExtractor,
    compute_human_agreement_rate,
    compute_metrics,
    gate_results,
    load_cases,
    load_reviewed_answer_key,
    main_async,
    required_gates_failed,
    run_all,
    run_failure_safety_check,
    write_markdown_report,
)


def _base_args(**overrides):
    defaults = dict(
        split="dev", provider="openai", simulate_failure=True,
        cost_per_1k_prompt_tokens=None, cost_per_1k_completion_tokens=None,
        json=True, markdown_out=None, fail_on_gate=False, strict_release_gate=False,
        reviewed_answer_key=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


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
async def test_run_failure_safety_check_is_100_percent_against_always_failing_extractor():
    cases = load_cases("dev")
    rate = await run_failure_safety_check(cases)
    assert rate == 1.0


@pytest.mark.asyncio
async def test_failure_safety_rate_flows_into_metrics_regardless_of_primary_run():
    """The correction this locks in: verifier_failure_safely_represented_rate
    must come from run_failure_safety_check()'s own dedicated exercise, not
    from whatever the primary run's extractor happened to do -- so a
    healthy primary run (a fake extractor that never fails) still reports
    this gate as measured, not N/A, once the dedicated rate is passed in."""
    cases = load_cases("dev")
    failure_safety_rate = await run_failure_safety_check(cases)

    class _AlwaysSucceedsExtractor:
        config = {"extraction_model": "test-model"}
        last_usage = None

        async def verify_claims(self, prompt):
            return []

    outcomes = await run_all(cases, _AlwaysSucceedsExtractor())
    metrics = compute_metrics(
        outcomes, None, None, provider="openai", model="test-model",
        verifier_failure_safely_represented_rate=failure_safety_rate,
    )
    assert metrics["verifier_failure_safely_represented_rate"] == 1.0


@pytest.mark.asyncio
async def test_verifier_failure_safely_represented_rate_is_none_when_not_supplied():
    cases = load_cases("dev")
    outcomes = await run_all(cases, AlwaysFailingExtractor())
    metrics = compute_metrics(outcomes, None, None, provider="openai", model="test-model")
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


def test_load_reviewed_answer_key_reads_reconciled_labels(tmp_path):
    path = tmp_path / "sealed_reconciled.json"
    path.write_text(json.dumps({
        "split": "sealed", "reviewers": ["alice", "bob"],
        "reconciled_labels": [{"id": "case-1", "label": "supported"},
                               {"id": "case-2", "label": "contradicted"}],
    }))
    answer_key = load_reviewed_answer_key(path)
    assert answer_key == {"case-1": "supported", "case-2": "contradicted"}


def test_compute_human_agreement_rate_matches_verdict_against_reconciled_label():
    from benchmarks.grounding.model_verifier_runner import CaseOutcome

    def outcome(case_id, verdict, failed=False):
        return CaseOutcome(case={"id": case_id}, verdict=verdict, verification_failed=failed)

    outcomes = [
        outcome("a", "supported"),        # matches key -> agrees
        outcome("b", "supported"),        # key says contradicted -> disagrees
        outcome("c", "contradicted"),     # not in key -> excluded
        outcome("d", None, failed=True),  # verification_failed -> excluded
    ]
    answer_key = {"a": "supported", "b": "contradicted"}
    rate = compute_human_agreement_rate(outcomes, answer_key)
    assert rate == 0.5  # 1 agreement out of 2 cases actually covered by the key


def test_compute_human_agreement_rate_is_none_with_no_matching_cases():
    from benchmarks.grounding.model_verifier_runner import CaseOutcome
    outcomes = [CaseOutcome(case={"id": "x"}, verdict="supported", verification_failed=False)]
    assert compute_human_agreement_rate(outcomes, {"other-case": "supported"}) is None


def test_required_gates_failed_strict_mode_should_be_combined_with_incomplete_run_check():
    """required_gates_failed() itself only ever inspects gate verdicts -- it
    has no notion of verification_failure_count. main_async's strict-mode
    branch is responsible for also failing on any incomplete case; this
    documents that a gate set with every metric passing still must not be
    read as sufficient on its own when verification_failure_count > 0 (see
    the equivalent check in main_async)."""
    metrics = {
        "unsupported_claim_escape_rate": 0.0,
        "contradicted_claim_escape_rate": 0.0,
        "supported_claim_retention_rate": 1.0,
        "incorrect_abstention_rate": 0.0,
        "unknown_citation_rejection_rate": 1.0,
        "fabricated_quote_rejection_rate": 1.0,
        "verifier_failure_safely_represented_rate": 1.0,
        "human_verifier_agreement_rate": 1.0,
        "verification_failure_count": 3,
    }
    gates = gate_results(metrics)
    assert required_gates_failed(gates, strict=True) is False
    incomplete_run = metrics["verification_failure_count"] > 0
    assert incomplete_run is True


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
async def test_strict_release_gate_fails_on_incomplete_run_even_with_no_reviewed_key(capsys):
    """--simulate-failure guarantees verification_failure_count > 0 for the
    dev split (some cases need the model layer). --strict-release-gate must
    exit 1 for that reason alone, on top of the existing N/A-gate check --
    this is the fix for the gap where verification_failure_count was
    disclosed but never actually gated in strict mode."""
    exit_code = await main_async(_base_args(strict_release_gate=True))
    capsys.readouterr()
    assert exit_code == 1


@pytest.mark.asyncio
async def test_reviewed_answer_key_populates_human_agreement_rate(tmp_path, capsys):
    cases = load_cases("dev")
    # Build a trivially-agreeing answer key from the constructed labels so
    # the metric is measurable without needing a real verifier run to
    # happen to match a hand-picked key.
    answer_key_path = tmp_path / "dev_reconciled.json"
    answer_key_path.write_text(json.dumps({
        "split": "dev", "reviewers": ["alice", "bob"],
        "reconciled_labels": [{"id": c["id"], "label": c["label"]} for c in cases],
    }))

    class _AlwaysSucceedsExtractor:
        config = {"extraction_model": "test-model"}
        last_usage = None

        async def verify_claims(self, prompt):
            return []

    args = _base_args(simulate_failure=False, reviewed_answer_key=str(answer_key_path))
    import benchmarks.grounding.model_verifier_runner as runner_module
    original = runner_module.build_real_extractor
    runner_module.build_real_extractor = lambda provider: _AlwaysSucceedsExtractor()
    try:
        await main_async(args)
    finally:
        runner_module.build_real_extractor = original

    printed = json.loads(capsys.readouterr().out)
    assert printed["metrics"]["human_verifier_agreement_rate"] is not None


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
