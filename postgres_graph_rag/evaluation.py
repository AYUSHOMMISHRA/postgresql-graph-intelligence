"""Reproducible incident-investigation retrieval benchmark.

The corpus is generated from versioned templates so every run contains the
same 120 labeled questions without committing generated embeddings or model
outputs. It evaluates retrieval evidence, not model fluency.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .core import PostgresGraphRAG
from .extractor import Triplet
from .offline import OfflineExtractor


DATASET_VERSION = "incident-benchmark-v1"
DEFAULT_TENANT = uuid.UUID("22222222-2222-2222-2222-222222222222")


@dataclass(frozen=True)
class BenchmarkDocument:
    source_id: str
    text: str
    triplets: tuple[tuple[str, str, str], ...]


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    category: str
    question: str
    expected_entities: tuple[str, ...]
    required_relations: tuple[str, ...] = ()
    required_sources: tuple[str, ...] = ()
    forbidden_entities: tuple[str, ...] = ()
    abstain: bool = False
    max_hops: int = 2


def build_incident_dataset() -> tuple[List[BenchmarkDocument], List[BenchmarkCase]]:
    documents: List[BenchmarkDocument] = []
    cases: List[BenchmarkCase] = []
    for number in range(1, 31):
        deploy = f"deploy-{number:03d}"
        checkout = f"checkout-service-{number:03d}"
        auth = f"auth-service-{number:03d}"
        team = f"Identity Team {number:03d}"
        runbook = f"auth-rollback-{number:03d}"
        deploy_source = f"incident-{number:03d}-deploy"
        ownership_source = f"incident-{number:03d}-ownership"

        documents.extend([
            BenchmarkDocument(
                deploy_source,
                f"{deploy} deployed {checkout}. {checkout} depends_on {auth}.",
                ((deploy, "deployed", checkout), (checkout, "depends_on", auth)),
            ),
            BenchmarkDocument(
                ownership_source,
                f"{auth} owned_by {team}. {team} has_runbook {runbook}.",
                ((auth, "owned_by", team), (team, "has_runbook", runbook)),
            ),
        ])
        cases.append(BenchmarkCase(
            id=f"semantic-{number:03d}", category="semantic_single_hop",
            question=f"Which service was released by {deploy}?",
            expected_entities=(checkout,), required_relations=("deployed",),
            required_sources=(deploy_source,),
            max_hops=1,
        ))
        cases.append(BenchmarkCase(
            id=f"lexical-{number:03d}", category="identifier_lexical",
            question=f"What runbook identifier is assigned to {team}?",
            expected_entities=(runbook,), required_relations=("has_runbook",),
            required_sources=(ownership_source,),
            max_hops=1,
        ))

        if number <= 20:
            cases.append(BenchmarkCase(
                id=f"multihop-owner-{number:03d}", category="multi_hop",
                question=f"Which team owns the dependency used by {checkout}?",
                expected_entities=(team,),
                required_relations=("depends_on", "owned_by"),
                required_sources=(deploy_source, ownership_source),
                max_hops=2,
            ))
            cases.append(BenchmarkCase(
                id=f"multihop-runbook-{number:03d}", category="multi_hop",
                question=f"Which runbook is connected to {deploy} through its service dependency owner?",
                expected_entities=(runbook,),
                required_relations=("deployed", "depends_on", "owned_by", "has_runbook"),
                required_sources=(deploy_source, ownership_source),
                max_hops=4,
            ))
            cases.append(BenchmarkCase(
                id=f"negative-{number:03d}", category="negative",
                question=f"Which team owns database-{number:03d} used by {checkout}?",
                expected_entities=(), forbidden_entities=(team,), abstain=True,
                max_hops=2,
            ))

    assert len(cases) == 120
    return documents, cases


def offline_extractor(documents: Iterable[BenchmarkDocument]) -> OfflineExtractor:
    return OfflineExtractor({
        document.text: [Triplet(subject=s, predicate=p, object=o) for s, p, o in document.triplets]
        for document in documents
    })


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(len(ordered) * percentile) - 1))
    return ordered[index]


def score_case(case: BenchmarkCase, result: Any) -> Dict[str, Any]:
    chunk_ranked = [chunk.content.lower() for chunk in result.chunks]
    node_ranked = [node.content.lower() for node in result.nodes]
    ranked = chunk_ranked + node_ranked
    expected = [entity.lower() for entity in case.expected_entities]
    entity_ranks = []
    for entity in expected:
        chunk_rank = next((i + 1 for i, text in enumerate(chunk_ranked[:5]) if entity in text), None)
        node_rank = next((i + 1 for i, text in enumerate(node_ranked[:5]) if entity in text), None)
        ranks = [rank for rank in (chunk_rank, node_rank) if rank is not None]
        rank = min(ranks) if ranks else None
        entity_ranks.append(rank)
    hits = sum(rank is not None and rank <= 5 for rank in entity_ranks)
    recall_at_5 = hits / len(expected) if expected else 1.0
    precision_at_5 = hits / min(5, max(1, len(ranked)))
    reciprocal_rank = 1.0 / min(entity_ranks) if entity_ranks and all(entity_ranks) else 0.0
    returned_relations = {edge.relation for edge in result.edges}
    path_match = (
        all(relation in returned_relations for relation in case.required_relations)
        if case.required_relations else True
    )
    returned_sources = {chunk.source_id for chunk in result.chunks}
    source_recall = (
        len(returned_sources.intersection(case.required_sources)) / len(case.required_sources)
        if case.required_sources else 1.0
    )
    false_connection = any(
        forbidden.lower() in text
        for forbidden in case.forbidden_entities
        for text in ranked
    )
    return {
        "id": case.id,
        "category": case.category,
        "recall_at_5": recall_at_5,
        "precision_at_5": precision_at_5,
        "reciprocal_rank": reciprocal_rank,
        "path_match": path_match,
        "source_recall": source_recall,
        "false_connection": false_connection,
        "context_tokens": result.trace.context_tokens if result.trace else 0,
    }


async def evaluate_engine(engine: Any, cases: Sequence[BenchmarkCase], namespace: str) -> Dict[str, Any]:
    modes: Dict[str, Any] = {}
    for mode in ("vector", "hybrid", "hybrid_graph"):
        results = []
        latencies = []
        for case in cases:
            started = time.perf_counter()
            result = await engine.retrieve(
                case.question, namespace, mode=mode, top_k=5, hops=case.max_hops
            )
            latencies.append((time.perf_counter() - started) * 1000)
            score = score_case(case, result)
            if case.abstain:
                answer = await engine.answer(
                    case.question, namespace, mode=mode, top_k=5, hops=case.max_hops
                )
                score["false_connection"] = answer.grounded
            results.append(score)

        categories: Dict[str, Any] = {}
        for category in sorted({item["category"] for item in results}):
            subset = [item for item in results if item["category"] == category]
            categories[category] = _aggregate(subset)
        modes[mode] = {
            **_aggregate(results),
            "p50_ms": statistics.median(latencies),
            "p95_ms": _percentile(latencies, 0.95),
            "p99_ms": _percentile(latencies, 0.99),
            "categories": categories,
        }
    return {"dataset_version": DATASET_VERSION, "question_count": len(cases), "modes": modes}


def _aggregate(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {}
    return {
        "recall_at_5": statistics.mean(item["recall_at_5"] for item in results),
        "precision_at_5": statistics.mean(item["precision_at_5"] for item in results),
        "mrr": statistics.mean(item["reciprocal_rank"] for item in results),
        "path_accuracy": statistics.mean(float(item["path_match"]) for item in results),
        "source_recall": statistics.mean(item["source_recall"] for item in results),
        "false_connection_rate": statistics.mean(float(item["false_connection"]) for item in results),
        "mean_context_tokens": statistics.mean(item["context_tokens"] for item in results),
    }


async def _run(args: argparse.Namespace) -> None:
    documents, cases = build_incident_dataset()
    runtime_url = args.runtime_url or os.getenv("PGR_RUNTIME_URL")
    if not runtime_url:
        raise SystemExit("PGR_RUNTIME_URL is required")
    rag = PostgresGraphRAG(
        runtime_url=runtime_url,
        extractor=offline_extractor(documents),
    )
    engine = rag.for_tenant(uuid.UUID(args.tenant_id))
    try:
        if args.ingest:
            for document in documents:
                await engine.add_document(document.text, args.namespace, document.source_id)
        report = await evaluate_engine(engine, cases, args.namespace)
    finally:
        await rag.close()
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(prog="postgres-graph-rag-eval")
    parser.add_argument("--runtime-url")
    parser.add_argument("--tenant-id", default=str(DEFAULT_TENANT))
    parser.add_argument("--namespace", default="incident-benchmark-v1")
    parser.add_argument("--ingest", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
