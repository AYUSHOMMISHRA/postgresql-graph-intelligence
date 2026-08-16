"""Runs postgres_graph_rag.model_verifier.ModelEntailmentVerifier (Release 2
PR 4) against the grounding benchmark -- the "after" result paired with
baseline_runner.py's and deterministic_runner.py's "before" results, and
PR 6's last missing code deliverable.

Each case's retrieved_evidence/asserted_citation_ids are adapted into the
real shapes ModelEntailmentVerifier expects, exactly like
deterministic_runner.py does -- this exercises the actual production
verifier, not a reimplementation of it.

Requires a real provider to produce a live result:

    OPENAI_API_KEY=... python -m benchmarks.grounding.model_verifier_runner \\
        --provider openai --split sealed \\
        --cost-per-1k-prompt-tokens 0.0001 --cost-per-1k-completion-tokens 0.0004

    GOOGLE_API_KEY=... python -m benchmarks.grounding.model_verifier_runner \\
        --provider google --split dev

Three things can be exercised right now, without any real API key or
network call, and are covered by tests/test_model_verifier_runner.py:

- `run_failure_safety_check()` runs every case through a dedicated,
  always-failing extractor and measures
  `verifier_failure_safely_represented_rate` from that exercise alone --
  this runs unconditionally on *every* invocation (live or
  `--simulate-failure`), exactly like `run_fabrication_check()` below,
  precisely so a genuine release run against a real provider still
  populates this gate instead of leaving it permanently N/A (which is
  what happened when this metric was only computed from the primary
  run's own incidental failures under `--simulate-failure`).
- `run_fabrication_check()` measures `fabricated_quote_rejection_rate`
  from the existing `fabrication_fixtures.json` pairs on every run
  (live or simulated) -- no real model needed, since a fixture's
  `fabricated_quote` is guaranteed by construction to never be a literal
  substring of its `chunk_content`; a stub extractor claims it as the
  supporting quote and this checks that `ModelEntailmentVerifier` rejects
  it anyway.
- Passing a custom extractor object programmatically (see `run_all()`'s
  `extractor` parameter) instead of going through `main()`'s
  --provider/API-key path, for any other offline simulation.

`--simulate-failure` still exists to let the *primary* run itself use an
always-failing extractor (useful for exercising the harness's own
reporting/exit-code paths without credentials); since every primary-run
case then fails, every quality gate that depends on a resolved case
(escape/retention/rejection rates) correctly reports N/A rather than a
misleading number -- this is a natural consequence of excluding
`verification_failed` cases from every rate's denominator, not special
cased.

`human_verifier_agreement_rate` is N/A until a `--reviewed-answer-key`
(the reconciled JSON `reconcile_reviews.py` writes) is supplied -- it
needs the sealed split's independent human review, which hasn't happened
yet -- see benchmarks/grounding/LABELING.md. That file is validated, not
merely parsed (see `load_reviewed_answer_key`): it must be for the
`sealed` split, list exactly two distinct reviewers, have no duplicate or
invalid-label entries, and cover exactly the sealed split's case ids --
otherwise a partial or malformed key could silently inflate the reported
agreement rate by shrinking its own denominator instead of failing loudly.

A `verification_failed` outcome (provider error/timeout/malformed
response) is treated as an *incomplete* result for that case, never as
either an acceptance or a rejection -- it is excluded from every escape/
retention/rejection rate's denominator and disclosed only via
`verification_failure_count`. Folding it into those rates would let a
provider outage look identical to (or better than) a correct rejection.

Two gate-checking modes are available: `--fail-on-gate` (permissive --
only an actual FAIL blocks; suitable for iterating against `dev`/
`calibration`) and `--strict-release-gate` (an N/A verdict blocks too;
this is the one an actual release go/no-go decision should use, since a
required gate that was never measured must not silently pass CI).
`--strict-release-gate` additionally requires `--split sealed` and the
finalized `grounding-benchmark-v1` dataset (not `...-v1-candidate`) --
a release decision made against the wrong split or a not-yet-reviewed
candidate dataset isn't a real release decision.
"""
import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from postgres_graph_rag.extractor import LLMExtractor, ModelVerdict
from postgres_graph_rag.grounding import AnswerClaim, VerifierUnavailableError
from postgres_graph_rag.model_verifier import ModelEntailmentVerifier
from postgres_graph_rag.models import GOOGLE_DEFAULT_CONFIG, OPENAI_DEFAULT_CONFIG
from postgres_graph_rag.tenant_engine import TenantRetrievedChunk

from .gates import RELEASE_2_GATES

CASES_PATH = Path(__file__).parent / "cases.json"
FABRICATION_FIXTURES_PATH = Path(__file__).parent / "fabrication_fixtures.json"


class AlwaysFailingExtractor:
    """Injectable stand-in whose verify_claims always raises -- lets
    verifier_failure_safely_represented_rate be measured without a live
    provider needing to fail on its own. Mirrors
    benchmarks/grounding/verifier_fixtures.py's RaisingVerifier in spirit,
    at the LLMExtractor layer this runner actually depends on."""

    config = {"extraction_model": "simulated-failing-model"}
    last_usage = None

    async def verify_claims(self, prompt: str) -> List[Any]:
        raise RuntimeError("simulated: provider unavailable")


def build_real_extractor(provider: str) -> LLMExtractor:
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("--provider openai requires OPENAI_API_KEY to be set")
        return LLMExtractor(config=dict(OPENAI_DEFAULT_CONFIG), openai_api_key=api_key)
    if provider == "google":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise SystemExit("--provider google requires GOOGLE_API_KEY to be set")
        return LLMExtractor(config=dict(GOOGLE_DEFAULT_CONFIG), google_api_key=api_key)
    raise ValueError(f"unknown provider {provider!r}")


def _to_chunk(ev: Dict[str, Any]) -> TenantRetrievedChunk:
    return TenantRetrievedChunk(
        id=f"{ev['source_id']}-{ev['ordinal']}", document_id=ev["source_id"], source_id=ev["source_id"],
        ordinal=ev["ordinal"], content=ev["content"], rrf_score=1.0, lexical_rank=1, semantic_rank=1,
    )


def _to_bracketed(cid: str) -> str:
    return f"[{cid}]"


class _FabricatingExtractor:
    """Stub whose verify_claims always returns a 'supported' verdict
    quoting whatever text it's constructed with -- used only to check that
    ModelEntailmentVerifier's server-side quote validation rejects a quote
    that isn't actually in the cited evidence. Entirely offline: it never
    calls a real model, it just asserts what one *could* claim."""

    config = {"extraction_model": "fabrication-check-stub"}
    last_usage = None

    def __init__(self, claim_id: str, fabricated_quote: str):
        self._claim_id = claim_id
        self._fabricated_quote = fabricated_quote

    async def verify_claims(self, prompt: str) -> List[ModelVerdict]:
        return [ModelVerdict(
            claim_id=self._claim_id, verdict="supported",
            supporting_quote=self._fabricated_quote, confidence=0.95,
        )]


async def run_fabrication_check() -> Optional[float]:
    """Measures fabricated_quote_rejection_rate using the existing
    fabrication_fixtures.json pairs (built for exactly this) -- no live
    model needed. Each fixture's `fabricated_quote` is, by construction
    (see fabrication_fixtures.py's own assertions), never a literal
    substring of `chunk_content`. Using it as both the claim's own text
    (so the deterministic layer can't resolve it via a direct quote match
    and forwards it to the model layer) and as the stub model's proposed
    supporting_quote exercises the real rejection path: if
    ModelEntailmentVerifier ever trusted an unverified model quote, this
    would catch it.
    """
    fixtures = json.loads(FABRICATION_FIXTURES_PATH.read_text())["fixtures"]
    if not fixtures:
        return None

    rejected = 0
    for fixture in fixtures:
        evidence = [_to_chunk({"source_id": "fab", "ordinal": 0, "content": fixture["chunk_content"]})]
        claim = AnswerClaim(
            id=fixture["id"], text=fixture["fabricated_quote"],
            citation_ids=[_to_bracketed("fab#0")],
        )
        verifier = ModelEntailmentVerifier(
            _FabricatingExtractor(fixture["id"], fixture["fabricated_quote"]), evidence,
        )
        [result] = await verifier.verify([claim])
        if result.verdict != "supported":
            rejected += 1

    return rejected / len(fixtures)


async def run_failure_safety_check(cases: List[Dict[str, Any]]) -> Optional[float]:
    """Measures verifier_failure_safely_represented_rate from a dedicated,
    always-run offline exercise -- an AlwaysFailingExtractor run against the
    same case set, exactly parallel to run_fabrication_check() above. This
    runs unconditionally, regardless of --simulate-failure, so the metric is
    populated on every invocation instead of being structurally N/A outside
    a separate manual --simulate-failure command: a live run's own real
    failure count is neither controllable nor a meaningful sample (a
    healthy live run should have ~0 real failures, which would make
    "fraction of attempts that failed" collapse to 0% and wrongly fail a
    gate requiring >=100%), so this dedicated exercise -- not the primary
    run's incidental outcomes -- is the only source for this metric.
    """
    outcomes = await run_all(cases, AlwaysFailingExtractor())
    model_attempted = [o for o in outcomes if o.model_call_attempted]
    if not model_attempted:
        return None
    return sum(1 for o in model_attempted if o.verification_failed) / len(model_attempted)


_VALID_ANSWER_KEY_LABELS = {"supported", "contradicted", "insufficient"}


class AnswerKeyError(ValueError):
    """Raised when a --reviewed-answer-key file fails validation."""


def load_reviewed_answer_key(path: Path) -> Dict[str, str]:
    """Loads and validates a reconciled human answer key in
    reconcile_reviews.py's output shape
    ({"split":..., "reviewers": [...], "reconciled_labels": [{"id":..., "label":...}, ...]})
    into a case-id -> label lookup.

    Validated, not merely parsed, because a malformed or partial key would
    otherwise silently corrupt human_verifier_agreement_rate: a key with
    one matching case out of the sealed split's 28 could report 100%
    agreement while 27 cases were never actually reviewed. Every check
    below closes one such silent-corruption path:
    - wrong split -> a dev/calibration review being mistaken for sealed
    - fewer than two distinct reviewers -> not an independent review
    - a duplicate id -> the second entry silently overwriting the first
    - an invalid label -> a typo silently treated as a valid verdict
    - missing/extra ids -> a partial or mismatched-dataset key inflating
      the rate by shrinking its own denominator instead of failing loudly
    """
    data = json.loads(path.read_text())

    if data.get("split") != "sealed":
        raise AnswerKeyError(
            f"--reviewed-answer-key must be for the sealed split, got {data.get('split')!r}"
        )

    reviewers = data.get("reviewers", [])
    if len(reviewers) != 2 or len(set(reviewers)) != 2:
        raise AnswerKeyError(
            f"--reviewed-answer-key must list exactly two distinct reviewers, got {reviewers!r}"
        )

    answer_key: Dict[str, str] = {}
    for entry in data["reconciled_labels"]:
        case_id, label = entry["id"], entry["label"]
        if case_id in answer_key:
            raise AnswerKeyError(f"--reviewed-answer-key has a duplicate case id: {case_id!r}")
        if label not in _VALID_ANSWER_KEY_LABELS:
            raise AnswerKeyError(
                f"--reviewed-answer-key has an invalid label {label!r} for case {case_id!r} "
                f"(must be one of {sorted(_VALID_ANSWER_KEY_LABELS)})"
            )
        answer_key[case_id] = label

    sealed_ids = {c["id"] for c in load_cases("sealed")}
    key_ids = set(answer_key)
    if key_ids != sealed_ids:
        missing, extra = sorted(sealed_ids - key_ids), sorted(key_ids - sealed_ids)
        raise AnswerKeyError(
            "--reviewed-answer-key must cover exactly the sealed split's case ids -- "
            f"missing {missing}, unexpected {extra}"
        )

    return answer_key


def compute_human_agreement_rate(
    outcomes: List["CaseOutcome"], answer_key: Dict[str, str]
) -> Optional[float]:
    """human_verifier_agreement_rate: fraction of resolved cases with a
    reconciled human label where the verifier's verdict matches it. A
    verification_failed case is excluded from the denominator, same
    reasoning as every other rate in compute_metrics -- but a case simply
    absent from the answer key is not a normal path here: a validated key
    (see load_reviewed_answer_key) always covers every sealed case, so
    this only happens if the primary run's --split wasn't sealed."""
    matched = [o for o in outcomes if not o.verification_failed and o.case["id"] in answer_key]
    if not matched:
        return None
    return sum(1 for o in matched if o.verdict == answer_key[o.case["id"]]) / len(matched)


@dataclass
class CaseOutcome:
    case: Dict[str, Any]
    verdict: Optional[str]  # None iff verification_failed
    verification_failed: bool
    latency_ms: Optional[float] = None
    usage: Optional[Dict[str, int]] = None
    model_call_attempted: bool = False


def load_cases(split: Optional[str]) -> List[Dict[str, Any]]:
    data = json.loads(CASES_PATH.read_text())
    cases = data["cases"]
    if split and split != "all":
        cases = [c for c in cases if c["split"] == split]
    return cases


async def run_case(case: Dict[str, Any], extractor: Any) -> CaseOutcome:
    evidence = [_to_chunk(ev) for ev in case["retrieved_evidence"]]
    claim = AnswerClaim(
        id=case["id"], text=case["claim"],
        citation_ids=[_to_bracketed(cid) for cid in case["asserted_citation_ids"]],
    )
    verifier = ModelEntailmentVerifier(extractor, evidence)
    try:
        [result] = await verifier.verify([claim])
        verdict, failed = result.verdict, False
    except VerifierUnavailableError:
        verdict, failed = None, True

    telemetry = verifier.last_telemetry
    return CaseOutcome(
        case=case, verdict=verdict, verification_failed=failed,
        latency_ms=telemetry.latency_ms, usage=telemetry.usage,
        model_call_attempted=telemetry.model_call_attempted,
    )


async def run_all(cases: List[Dict[str, Any]], extractor: Any) -> List[CaseOutcome]:
    return [await run_case(c, extractor) for c in cases]


def compute_metrics(
    outcomes: List[CaseOutcome],
    cost_per_1k_prompt_tokens: Optional[float],
    cost_per_1k_completion_tokens: Optional[float],
    provider: str,
    model: str,
    fabricated_quote_rejection_rate: Optional[float] = None,
    verifier_failure_safely_represented_rate: Optional[float] = None,
    human_verifier_agreement_rate: Optional[float] = None,
) -> Dict[str, Any]:
    def rate(pairs, predicate):
        return (sum(1 for o in pairs if predicate(o)) / len(pairs)) if pairs else None

    # A verification_failed case is an *incomplete* quality signal, not a
    # rejection and not an acceptance -- it must never appear in any
    # escape/retention/rejection denominator. Folding it in (the original
    # bug here) meant a provider outage could dilute an escape-rate
    # denominator with cases that trivially "didn't escape" (verdict is
    # None, so `== "supported"` is always False), making the rate look
    # better than the verifier's actual behavior on the cases it completed
    # -- and, for unknown_citation_rejection_rate, an outage's `verdict
    # != "supported"` was literally indistinguishable from a correct
    # rejection. `resolved` is the only population these rates are
    # computed over; incomplete cases are tracked solely via
    # verification_failure_count.
    resolved = [o for o in outcomes if not o.verification_failed]

    unsupported = [o for o in resolved if o.case["label"] in ("contradicted", "insufficient")]
    contradicted = [o for o in resolved if o.case["label"] == "contradicted"]
    supported = [o for o in resolved if o.case["label"] == "supported"]
    unknown_citation = [
        o for o in resolved
        if any(
            cid not in {f"{ev['source_id']}#{ev['ordinal']}" for ev in o.case["retrieved_evidence"]}
            for cid in o.case["asserted_citation_ids"]
        )
    ]
    # Cases that actually reached the model layer (attempted a call,
    # whether it then succeeded or failed) -- the only ones relevant to
    # judging failure representation; a case the deterministic layer
    # already decided never touches the model at all.
    model_attempted = [o for o in outcomes if o.model_call_attempted]

    metrics: Dict[str, Any] = {
        "case_count": len(outcomes),
        "provider": provider,
        "model": model,
        "unsupported_claim_escape_rate": rate(unsupported, lambda o: o.verdict == "supported"),
        "contradicted_claim_escape_rate": rate(contradicted, lambda o: o.verdict == "supported"),
        "supported_claim_retention_rate": rate(supported, lambda o: o.verdict == "supported"),
        "incorrect_abstention_rate": rate(supported, lambda o: o.verdict != "supported"),
        "unknown_citation_rejection_rate": rate(unknown_citation, lambda o: o.verdict != "supported"),
        # Preserved as its own count, not folded into an escape/rejection
        # rate or excluded silently -- a verification_failed outcome is
        # neither "supported" nor a normal rejection; it's an incomplete
        # run that a real release decision must see disclosed, not hidden
        # inside a rate that looks clean by construction.
        "verification_failure_count": sum(1 for o in outcomes if o.verification_failed),
        # Measured by run_failure_safety_check()'s own dedicated,
        # always-failing exercise -- run unconditionally on every
        # invocation regardless of --simulate-failure, so this gate is
        # populated on a genuine live run instead of staying permanently
        # N/A outside a separate manual command.
        "verifier_failure_safely_represented_rate": verifier_failure_safely_represented_rate,
        # Measured offline from fabrication_fixtures.json regardless of
        # whether this is a live or simulated run -- see
        # run_fabrication_check(), which needs no real model.
        "fabricated_quote_rejection_rate": fabricated_quote_rejection_rate,
        # None until a --reviewed-answer-key is supplied -- see
        # compute_human_agreement_rate() and load_reviewed_answer_key().
        "human_verifier_agreement_rate": human_verifier_agreement_rate,
        "model_calls_attempted": len(model_attempted),
    }

    latencies = sorted(o.latency_ms for o in model_attempted if o.latency_ms is not None)
    if latencies:
        metrics["p50_latency_ms"] = latencies[len(latencies) // 2]
        metrics["p95_latency_ms"] = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]
    else:
        metrics["p50_latency_ms"] = None
        metrics["p95_latency_ms"] = None

    total_prompt = sum(o.usage.get("prompt_tokens", 0) for o in outcomes if o.usage)
    total_completion = sum(o.usage.get("completion_tokens", 0) for o in outcomes if o.usage)
    total_tokens = sum(o.usage.get("total_tokens", 0) for o in outcomes if o.usage)
    metrics["total_prompt_tokens"] = total_prompt
    metrics["total_completion_tokens"] = total_completion
    metrics["total_tokens"] = total_tokens

    if cost_per_1k_prompt_tokens is not None and cost_per_1k_completion_tokens is not None and outcomes:
        total_cost = (
            total_prompt / 1000 * cost_per_1k_prompt_tokens
            + total_completion / 1000 * cost_per_1k_completion_tokens
        )
        metrics["added_cost_per_answer_usd"] = total_cost / len(outcomes)
        metrics["total_cost_usd"] = total_cost
    else:
        metrics["added_cost_per_answer_usd"] = None
        metrics["total_cost_usd"] = None

    return metrics


def gate_results(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    results = []
    for gate in RELEASE_2_GATES:
        value = metrics.get(gate.metric)
        verdict = gate.passes(value)
        results.append({
            "metric": gate.metric, "value": value, "threshold": gate.threshold,
            "comparison": gate.comparison, "verdict": "N/A" if verdict is None else ("PASS" if verdict else "FAIL"),
            "description": gate.description,
        })
    return results


def print_report(split_label: str, metrics: Dict[str, Any], gates: List[Dict[str, Any]]) -> None:
    print(f"\n=== ModelEntailmentVerifier -- split={split_label} "
          f"(provider={metrics['provider']}, model={metrics['model']}) ===")
    print(f"cases: {metrics['case_count']}, model calls attempted: {metrics['model_calls_attempted']}, "
          f"verification failures: {metrics['verification_failure_count']}")
    print("\nGates:")
    for g in gates:
        value_str = "N/A" if g["value"] is None else f"{g['value']:.1%}"
        print(f"  [{g['verdict']:4}] {g['metric']:42} = {value_str:>6}  "
              f"(gate: {g['comparison']} {g['threshold']:.0%})  -- {g['description']}")
    print("\nTelemetry:")
    print(f"  p50/p95 latency: {metrics['p50_latency_ms']} / {metrics['p95_latency_ms']} ms")
    print(f"  tokens (prompt/completion/total): "
          f"{metrics['total_prompt_tokens']}/{metrics['total_completion_tokens']}/{metrics['total_tokens']}")
    print(f"  cost per answer: {metrics['added_cost_per_answer_usd']} USD "
          f"(total: {metrics['total_cost_usd']} USD)")


def write_markdown_report(path: Path, split_label: str, metrics: Dict[str, Any], gates: List[Dict[str, Any]]) -> None:
    lines = [
        f"# ModelEntailmentVerifier — {split_label} split result",
        "",
        f"Provider: `{metrics['provider']}`. Model: `{metrics['model']}`. "
        f"Cases: {metrics['case_count']}. Model calls attempted: {metrics['model_calls_attempted']}. "
        f"Verification failures: {metrics['verification_failure_count']}.",
        "",
        "| Gate | Result | Threshold | Verdict |",
        "|---|---:|---:|---|",
    ]
    for g in gates:
        value_str = "N/A" if g["value"] is None else f"{g['value']:.1%}"
        lines.append(f"| {g['metric']} | {value_str} | {g['comparison']} {g['threshold']:.0%} | {g['verdict']} |")
    lines += [
        "",
        "## Telemetry",
        "",
        f"- p50/p95 latency: {metrics['p50_latency_ms']} / {metrics['p95_latency_ms']} ms",
        f"- Tokens (prompt/completion/total): {metrics['total_prompt_tokens']}/"
        f"{metrics['total_completion_tokens']}/{metrics['total_tokens']}",
        f"- Cost per answer: {metrics['added_cost_per_answer_usd']} USD "
        f"(total: {metrics['total_cost_usd']} USD)",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")


def required_gates_failed(gates: List[Dict[str, Any]], strict: bool = False) -> bool:
    """strict=False (ordinary/dev runs): only an actual FAIL verdict fails
    the run -- a gate that's legitimately N/A (not yet measurable, e.g.
    human_verifier_agreement_rate before review) doesn't block iteration.

    strict=True (release-gate mode): an N/A verdict fails too. A real
    release decision cannot pass with a required gate silently unmeasured
    -- see the CTO correction this was written to address: N/A gates were
    previously invisible to --fail-on-gate, so CI could exit 0 while the
    human-review and fabricated-quote gates had never actually run.
    """
    if strict:
        return any(g["verdict"] in ("FAIL", "N/A") for g in gates)
    return any(g["verdict"] == "FAIL" for g in gates)


_FINALIZED_DATASET_VERSION = "grounding-benchmark-v1"


async def main_async(args: argparse.Namespace) -> int:
    if args.strict_release_gate:
        if args.split != "sealed":
            print("error: --strict-release-gate requires --split sealed "
                  f"(got --split {args.split!r})", file=sys.stderr)
            return 2
        dataset_version = json.loads(CASES_PATH.read_text())["dataset_version"]
        if dataset_version != _FINALIZED_DATASET_VERSION:
            print(
                f"error: --strict-release-gate requires the finalized dataset "
                f"{_FINALIZED_DATASET_VERSION!r}, got {dataset_version!r} -- the sealed "
                "split hasn't been through independent human review and renamed from "
                "'...-v1-candidate' yet, see benchmarks/grounding/LABELING.md.",
                file=sys.stderr,
            )
            return 2

    cases = load_cases(args.split)

    if args.simulate_failure:
        extractor: Any = AlwaysFailingExtractor()
    else:
        extractor = build_real_extractor(args.provider)

    outcomes = await run_all(cases, extractor)
    fabrication_rate = await run_fabrication_check()
    failure_safety_rate = await run_failure_safety_check(cases)

    answer_key = None
    if args.reviewed_answer_key:
        try:
            answer_key = load_reviewed_answer_key(Path(args.reviewed_answer_key))
        except AnswerKeyError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    human_agreement_rate = (
        compute_human_agreement_rate(outcomes, answer_key) if answer_key is not None else None
    )

    metrics = compute_metrics(
        outcomes, args.cost_per_1k_prompt_tokens, args.cost_per_1k_completion_tokens,
        provider=("simulated-failure" if args.simulate_failure else args.provider),
        model=extractor.config["extraction_model"],
        fabricated_quote_rejection_rate=fabrication_rate,
        verifier_failure_safely_represented_rate=failure_safety_rate,
        human_verifier_agreement_rate=human_agreement_rate,
    )
    gates = gate_results(metrics)

    if args.json:
        print(json.dumps({"metrics": metrics, "gates": gates}, indent=2))
    else:
        print_report(args.split, metrics, gates)

    if args.markdown_out:
        write_markdown_report(Path(args.markdown_out), args.split, metrics, gates)
        print(f"\nWrote {args.markdown_out}")

    if args.strict_release_gate:
        # An incomplete run (any verification_failed case) is invalid for a
        # release decision even if every gate that *could* be computed
        # passed -- distinct from required_gates_failed's N/A check, since
        # an outage on a case that happens to fall outside every gated
        # subpopulation (e.g. a supported-claim case that's neither
        # unsupported nor unknown-citation) wouldn't otherwise surface as
        # any gate's N/A or FAIL.
        incomplete_run = metrics["verification_failure_count"] > 0
        if required_gates_failed(gates, strict=True) or incomplete_run:
            return 1
    elif args.fail_on_gate and required_gates_failed(gates, strict=False):
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="sealed", choices=["dev", "calibration", "sealed", "all"])
    parser.add_argument("--provider", default="openai", choices=["openai", "google"])
    parser.add_argument("--simulate-failure", action="store_true",
                         help="Use an always-failing extractor instead of a real provider -- "
                              "measures verifier_failure_safely_represented_rate without credentials.")
    parser.add_argument("--cost-per-1k-prompt-tokens", type=float, default=None)
    parser.add_argument("--cost-per-1k-completion-tokens", type=float, default=None)
    parser.add_argument("--reviewed-answer-key", default=None,
                         help="Path to a reconcile_reviews.py reconciled-labels JSON file "
                              "(e.g. benchmarks/grounding/packets/sealed_reconciled.json). "
                              "Computes human_verifier_agreement_rate against it; omitted, "
                              "that gate stays N/A. Validated: must be for the sealed split, "
                              "list exactly two distinct reviewers, have no duplicate or "
                              "invalid-label entries, and cover exactly the sealed split's "
                              "case ids -- a malformed or partial key is rejected, not "
                              "silently accepted.")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--markdown-out", default=None, help="Path to write a Markdown report, e.g. "
                         "grounding-benchmark-model-verifier.md")
    parser.add_argument("--fail-on-gate", action="store_true",
                         help="Exit with status 1 if any gate FAILs (N/A gates are permitted -- for iteration).")
    parser.add_argument("--strict-release-gate", action="store_true",
                         help="Exit with status 1 if any gate FAILs OR is N/A, or if any case's "
                              "verification_failed (an incomplete run is not a valid release "
                              "signal) -- for the actual release go/no-go decision, where every "
                              "required gate must be genuinely measured. Requires --split sealed "
                              "and the finalized grounding-benchmark-v1 dataset (exits 2 "
                              "otherwise). Takes precedence over --fail-on-gate if both are given.")
    args = parser.parse_args()

    exit_code = asyncio.run(main_async(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
