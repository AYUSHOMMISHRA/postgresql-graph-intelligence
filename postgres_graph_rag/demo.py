"""Repeatable incident-investigation demo and retrieval evaluation CLI."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import uuid
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse, urlunparse

import psycopg

from .core import PostgresGraphRAG
from .extractor import Triplet
from .offline import OfflineExtractor
from .tenancy import SCHEMA


DEMO_TENANT = uuid.UUID("11111111-1111-1111-1111-111111111111")
DEMO_DOCUMENTS = [
    {
        "source_id": "incident-2026-041",
        "text": "Checkout API returned elevated 5xx responses after release deploy-417. deploy-417 deployed checkout-service. checkout-service depends_on auth-service.",
        "triplets": [
            ("deploy-417", "deployed", "checkout-service"),
            ("checkout-service", "depends_on", "auth-service"),
        ],
    },
    {
        "source_id": "ownership-auth",
        "text": "auth-service is owned_by Identity Team. Identity Team has_runbook auth-rollback-runbook.",
        "triplets": [
            ("auth-service", "owned_by", "Identity Team"),
            ("Identity Team", "has_runbook", "auth-rollback-runbook"),
        ],
    },
    {
        "source_id": "deploy-417-change",
        "text": "deploy-417 uses auth-client-v4. auth-client-v4 caused token validation failures.",
        "triplets": [
            ("deploy-417", "uses", "auth-client-v4"),
            ("auth-client-v4", "caused", "token validation failures"),
        ],
    },
    {
        "source_id": "runbook-auth",
        "text": "auth-rollback-runbook runs_on auth-service and restores the previous authentication release.",
        "triplets": [("auth-rollback-runbook", "runs_on", "auth-service")],
    },
]

DEMO_QUESTIONS = [
    {"question": "Which team owns the dependency of checkout-service?", "expected": ["Identity Team"]},
    {"question": "Which runbook is connected to the checkout deployment?", "expected": ["auth-rollback-runbook"]},
    {"question": "What deployment caused the checkout incident?", "expected": ["deploy-417"]},
    {"question": "Which service does checkout-service depend on?", "expected": ["auth-service"]},
]


def _offline_extractor() -> OfflineExtractor:
    mapping: Dict[str, List[Triplet]] = {}
    for document in DEMO_DOCUMENTS:
        mapping[document["text"]] = [Triplet(subject=s, predicate=p, object=o) for s, p, o in document["triplets"]]
    return OfflineExtractor(mapping)


def _urls(args: argparse.Namespace) -> tuple[str, str]:
    admin_url = args.admin_url or os.getenv("POSTGRES_URL")
    runtime_url = args.runtime_url or os.getenv("PGR_RUNTIME_URL")
    if not admin_url:
        raise SystemExit("POSTGRES_URL or --admin-url is required")
    if not runtime_url:
        parsed = urlparse(admin_url)
        netloc = f"{args.runtime_role}:{args.runtime_password}@{parsed.hostname}:{parsed.port or 5432}"
        runtime_url = urlunparse(parsed._replace(netloc=netloc))
    return admin_url, runtime_url


def _rag(args: argparse.Namespace, runtime_url: str | None = None) -> PostgresGraphRAG:
    extractor = _offline_extractor()
    return PostgresGraphRAG(
        runtime_url=runtime_url or args.runtime_url or os.getenv("PGR_RUNTIME_URL"),
        extractor=extractor,
    )


async def _setup(args: argparse.Namespace) -> None:
    admin_url, runtime_url = _urls(args)
    if args.reset:
        async with await psycopg.AsyncConnection.connect(admin_url) as conn:
            async with conn.cursor() as cur:
                await cur.execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")
            await conn.commit()
        print(f"Reset demo schema {SCHEMA} before setup.")
    rag = PostgresGraphRAG(runtime_url=runtime_url, extractor=_offline_extractor())
    try:
        await rag.setup_secure(
            admin_url=admin_url,
            runtime_role=args.runtime_role,
            runtime_password=args.runtime_password,
            migrate_legacy_data=args.migrate_legacy,
        )
    finally:
        await rag.close()
    print("Secure schema is ready. Set PGR_RUNTIME_URL to the restricted runtime DSN before ingest/query.")


async def _ingest(args: argparse.Namespace) -> None:
    _, runtime_url = _urls(args)
    rag = _rag(args, runtime_url)
    engine = rag.for_tenant(uuid.UUID(args.tenant_id))
    try:
        reports = []
        for document in DEMO_DOCUMENTS:
            reports.append(await engine.add_document_detailed(document["text"], args.namespace, document["source_id"]))
        print(json.dumps(reports, indent=2))
    finally:
        await rag.close()


async def _query(args: argparse.Namespace) -> None:
    _, runtime_url = _urls(args)
    rag = _rag(args, runtime_url)
    engine = rag.for_tenant(uuid.UUID(args.tenant_id))
    try:
        result = await engine.retrieve(args.question, args.namespace, mode=args.mode, top_k=args.top_k, hops=args.hops)
        print(result.to_context_string())
    finally:
        await rag.close()


async def _evaluate(args: argparse.Namespace) -> None:
    _, runtime_url = _urls(args)
    rag = _rag(args, runtime_url)
    engine = rag.for_tenant(uuid.UUID(args.tenant_id))
    report: Dict[str, Any] = {"namespace": args.namespace, "modes": {}}
    try:
        for mode in ("vector", "hybrid", "hybrid_graph"):
            latencies: List[float] = []
            hits = 0
            for item in DEMO_QUESTIONS:
                started = asyncio.get_running_loop().time()
                result = await engine.retrieve(item["question"], args.namespace, mode=mode, top_k=5, hops=3)
                latencies.append((asyncio.get_running_loop().time() - started) * 1000)
                context = result.to_context_string().lower()
                hits += int(all(expected.lower() in context for expected in item["expected"]))
            report["modes"][mode] = {
                "questions": len(DEMO_QUESTIONS),
                "recall": hits / len(DEMO_QUESTIONS),
                "p50_ms": statistics.median(latencies),
                "p95_ms": sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)],
            }
    finally:
        await rag.close()
    output = json.dumps(report, indent=2)
    print(output)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")


async def _reset(args: argparse.Namespace) -> None:
    admin_url, _ = _urls(args)
    async with await psycopg.AsyncConnection.connect(admin_url) as conn:
        async with conn.cursor() as cur:
            await cur.execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")
        await conn.commit()
    print(f"Dropped demo schema {SCHEMA}; run setup again to recreate it.")


def main() -> None:
    parser = argparse.ArgumentParser(prog="postgres-graph-rag-demo")
    parser.add_argument("--admin-url", default=None)
    parser.add_argument("--runtime-url", default=None)
    parser.add_argument("--runtime-role", default=os.getenv("PGR_RUNTIME_ROLE", "pgr_demo_runtime"))
    parser.add_argument("--runtime-password", default=os.getenv("PGR_RUNTIME_PASSWORD", "pgr_demo_runtime_pw"))
    parser.add_argument("--tenant-id", default=os.getenv("PGR_DEMO_TENANT_ID", str(DEMO_TENANT)))
    parser.add_argument("--namespace", default=os.getenv("PGR_DEMO_NAMESPACE", "incident-demo"))
    sub = parser.add_subparsers(dest="command", required=True)
    setup = sub.add_parser("setup")
    setup.add_argument("--migrate-legacy", action="store_true")
    setup.add_argument(
        "--reset",
        action="store_true",
        help="Drop the secure demo schema before setup (destroys its data).",
    )
    sub.add_parser("ingest")
    query = sub.add_parser("query")
    query.add_argument("question")
    query.add_argument("--mode", choices=["vector", "hybrid", "hybrid_graph"], default="hybrid_graph")
    query.add_argument("--top-k", type=int, default=5)
    query.add_argument("--hops", type=int, default=3)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--output")
    sub.add_parser("reset")
    args = parser.parse_args()
    commands = {"setup": _setup, "ingest": _ingest, "query": _query, "evaluate": _evaluate, "reset": _reset}
    asyncio.run(commands[args.command](args))


if __name__ == "__main__":
    main()
