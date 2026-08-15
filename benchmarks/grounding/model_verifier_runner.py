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

Two things can be exercised right now, without any real API key or network
call, and are covered by tests/test_model_verifier_runner.py:

- `--simulate-failure` runs every case through an extractor whose
  verify_claims always raises, which is how
  verifier_failure_safely_represented_rate is measured -- confirming every
  case that actually reaches the model layer reports
  grounding "verification_failed" outcomes rather than something silently
  substituted, end-to-end through this runner, not just in
  test_model_verifier.py's unit tests.
- Passing a custom extractor object programmatically (see `run()`'s
  `extractor` parameter) instead of going through `main()`'s
  --provider/API-key path, for any other offline simulation.

`fabricated_quote_rejection_rate` and `human_verifier_agreement_rate`
remain N/A here: the former needs a real model to actually propose a
quote to test rejection against (this runner doesn't fabricate model
output on the model's behalf), and the latter needs the sealed split's
independent human review, which hasn't happened yet -- see
benchmarks/grounding/LABELING.md.
"""
import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from postgres_graph_rag.extractor import LLMExtractor
from postgres_graph_rag.grounding import AnswerClaim, VerifierUnavailableError
from postgres_graph_rag.model_verifier import ModelEntailmentVerifier
from postgres_graph_rag.models import GOOGLE_DEFAULT_CONFIG, OPENAI_DEFAULT_CONFIG
from postgres_graph_rag.tenant_engine import TenantRetrievedChunk

from .gates import RELEASE_2_GATES

CASES_PATH = Path(__file__).parent / "cases.json"


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
) -> Dict[str, Any]:
    def rate(pairs, predicate):
        return (sum(1 for o in pairs if predicate(o)) / len(pairs)) if pairs else None

    unsupported = [o for o in outcomes if o.case["label"] in ("contradicted", "insufficient")]
    contradicted = [o for o in outcomes if o.case["label"] == "contradicted"]
    supported = [o for o in outcomes if o.case["label"] == "supported"]
    unknown_citation = [
        o for o in outcomes
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
        # Preserved as their own count, not folded into an escape/rejection
        # rate -- a verification_failed outcome is neither "supported" nor
        # a normal rejection; hiding it inside another rate would make a
        # provider outage look identical to a correct abstention.
        "verification_failure_count": sum(1 for o in outcomes if o.verification_failed),
        "verifier_failure_safely_represented_rate": (
            rate(model_attempted, lambda o: o.verification_failed) if model_attempted else None
        ),
        "fabricated_quote_rejection_rate": None,  # needs a real model's own proposed quote, not simulated here
        "human_verifier_agreement_rate": None,  # sealed split not yet independently reviewed
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


def required_gates_failed(gates: List[Dict[str, Any]]) -> bool:
    return any(g["verdict"] == "FAIL" for g in gates)


async def main_async(args: argparse.Namespace) -> int:
    cases = load_cases(args.split)

    if args.simulate_failure:
        extractor: Any = AlwaysFailingExtractor()
    else:
        extractor = build_real_extractor(args.provider)

    outcomes = await run_all(cases, extractor)
    metrics = compute_metrics(
        outcomes, args.cost_per_1k_prompt_tokens, args.cost_per_1k_completion_tokens,
        provider=("simulated-failure" if args.simulate_failure else args.provider),
        model=extractor.config["extraction_model"],
    )
    gates = gate_results(metrics)

    if args.json:
        print(json.dumps({"metrics": metrics, "gates": gates}, indent=2))
    else:
        print_report(args.split, metrics, gates)

    if args.markdown_out:
        write_markdown_report(Path(args.markdown_out), args.split, metrics, gates)
        print(f"\nWrote {args.markdown_out}")

    if args.fail_on_gate and required_gates_failed(gates):
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
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--markdown-out", default=None, help="Path to write a Markdown report, e.g. "
                         "docs/results/grounding-benchmark-model-verifier.md")
    parser.add_argument("--fail-on-gate", action="store_true",
                         help="Exit with status 1 if any non-N/A gate fails.")
    args = parser.parse_args()

    exit_code = asyncio.run(main_async(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
