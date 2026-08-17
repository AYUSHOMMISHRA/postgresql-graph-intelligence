"""Repeatable incident-investigation demo and retrieval evaluation CLI."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, TypedDict
from urllib.parse import urlparse, urlunparse

import pydantic
import psycopg

from . import playground_service
from .core import PostgresGraphRAG
from .extractor import LLMExtractor, Triplet
from .models import GOOGLE_DEFAULT_CONFIG, OPENAI_DEFAULT_CONFIG, build_litellm_config
from .offline import OfflineExtractor
from .playground_service import PlaygroundInputError
from .tenancy import SCHEMA

# `setup` only ever needs the *dimension* the chosen provider's config
# declares (setup_secure() reads self.extractor.config["dimension"]) -- an
# LLMExtractor built with no API key makes no paid call and is exactly as
# safe here as OfflineExtractor's own default. Typed as `Any`, matching the
# rest of this codebase's convention for extractor references (e.g.
# PostgresGraphRAG.__init__'s own `extractor: Optional[Any]`) -- there is no
# shared Extractor protocol/base class, both classes are used duck-typed.
_SETUP_EXTRACTORS: Dict[str, Callable[[], Any]] = {
    "offline": lambda: OfflineExtractor(),
    "openai": lambda: LLMExtractor(config=OPENAI_DEFAULT_CONFIG),
    "gemini": lambda: LLMExtractor(config=GOOGLE_DEFAULT_CONFIG),
}


def _setup_extractor(provider: str) -> Any:
    if provider != "litellm":
        return _SETUP_EXTRACTORS[provider]()
    settings = playground_service.litellm_kwargs_from_env()
    dimension = settings["litellm_embedding_dimension"]
    if dimension is None:
        raise PlaygroundInputError(
            "setup --provider litellm requires LITELLM_EMBEDDING_DIMENSION"
        )
    # Setup uses only config["dimension"] and never calls this extractor.
    # Model placeholders keep setup independent of gateway credentials while
    # the actual Playground/Studio startup still validates every setting.
    config = build_litellm_config(
        extraction_model=settings["litellm_chat_model"] or "litellm-setup-only",
        embedding_model=settings["litellm_embedding_model"] or "litellm-setup-only",
        dimension=dimension,
    )
    return LLMExtractor(config=config)


class _DemoDocument(TypedDict):
    source_id: str
    text: str
    triplets: List[Tuple[str, str, str]]


DEMO_TENANT = uuid.UUID("11111111-1111-1111-1111-111111111111")
DEMO_DOCUMENTS: List[_DemoDocument] = [
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
    """Provisions the secure schema at the dimension the chosen
    `--provider` needs (default: offline, 1536-dim, same as OpenAI's
    default). Never makes a paid provider call -- the extractor built here
    only supplies its config's dimension to setup_secure(); no API key is
    used or required."""
    admin_url, runtime_url = _urls(args)
    if args.reset:
        async with await psycopg.AsyncConnection.connect(admin_url) as conn:
            async with conn.cursor() as cur:
                await cur.execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")
            await conn.commit()
        print(f"Reset demo schema {SCHEMA} before setup.")
    try:
        extractor = _setup_extractor(args.provider)
    except PlaygroundInputError as exc:
        raise SystemExit(str(exc)) from exc
    rag = PostgresGraphRAG(runtime_url=runtime_url, extractor=extractor)
    try:
        await rag.setup_secure(
            admin_url=admin_url,
            runtime_role=args.runtime_role,
            runtime_password=args.runtime_password,
            migrate_legacy_data=args.migrate_legacy,
        )
    finally:
        await rag.close()
    print(
        f"Secure schema is ready (provider={args.provider!r}, "
        f"dimension={extractor.config['dimension']}). "
        "Set PGR_RUNTIME_URL to the restricted runtime DSN before ingest/query."
    )


async def _ingest(args: argparse.Namespace) -> None:
    runtime_url = _runtime_only_url(args)
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
    runtime_url = _runtime_only_url(args)
    rag = _rag(args, runtime_url)
    engine = rag.for_tenant(uuid.UUID(args.tenant_id))
    try:
        result = await engine.retrieve(args.question, args.namespace, mode=args.mode, top_k=args.top_k, hops=args.hops)
        print(result.to_context_string())
    finally:
        await rag.close()


async def _evaluate(args: argparse.Namespace) -> None:
    runtime_url = _runtime_only_url(args)
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


def _runtime_only_url(args: argparse.Namespace) -> str:
    """Unlike `_urls()`, this never requires an admin DSN -- for any
    subcommand that only ever touches the restricted runtime role (never
    `setup_secure()` or schema DDL): `ingest`/`query`/`evaluate`/`playground`
    all discarded `_urls()`'s admin_url already, so requiring it upfront was
    a real, reproduced bug (`ingest`/`query` failed with "POSTGRES_URL or
    --admin-url is required" even when --runtime-url was supplied directly
    and no admin access was ever going to be used)."""
    runtime_url = args.runtime_url or os.getenv("PGR_RUNTIME_URL")
    if not runtime_url:
        raise SystemExit("PGR_RUNTIME_URL or --runtime-url is required")
    return runtime_url


def _load_playground_text(args: argparse.Namespace) -> str:
    if args.text and args.text_file:
        raise SystemExit("Pass only one of --text or --text-file, not both")
    if args.text_file:
        try:
            return Path(args.text_file).read_text(encoding="utf-8")
        except OSError as exc:
            raise SystemExit(f"Could not read --text-file {args.text_file!r}: {exc}") from exc
    if args.text is not None:
        return args.text
    raise SystemExit("One of --text or --text-file is required")


def _load_playground_triplets(args: argparse.Namespace) -> List[Triplet] | None:
    if args.triplets_json and args.triplets_file:
        raise SystemExit("Pass only one of --triplets-json or --triplets-file, not both")
    raw, source = None, None
    if args.triplets_file:
        try:
            raw = Path(args.triplets_file).read_text(encoding="utf-8")
        except OSError as exc:
            raise SystemExit(f"Could not read --triplets-file {args.triplets_file!r}: {exc}") from exc
        source = args.triplets_file
    elif args.triplets_json:
        raw, source = args.triplets_json, "--triplets-json"
    if raw is None:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {source}: {exc}") from exc
    try:
        return [Triplet(**item) for item in data["triplets"]]
    except (KeyError, TypeError, pydantic.ValidationError) as exc:
        raise SystemExit(
            f'{source} must look like {{"triplets": [{{"subject": ..., '
            f'"predicate": ..., "object": ...}}]}}: {exc}'
        ) from exc


async def _playground(args: argparse.Namespace) -> None:
    """Interactive/one-shot playground: a user's own text and question
    against the real engine, isolated from the bundled demo's fixed
    tenant/namespace and never touching schema setup or reset -- see
    playground_service.py for the shared logic this reuses."""
    runtime_url = _runtime_only_url(args)
    text = _load_playground_text(args)
    triplets = _load_playground_triplets(args)

    try:
        provider_kwargs = (
            playground_service.litellm_kwargs_from_env()
            if args.provider == "litellm"
            else {}
        )
        result = await playground_service.run_playground_session(
            text=text,
            question=args.question,
            provider=args.provider,
            mode=args.mode,
            runtime_url=runtime_url,
            triplets=triplets,
            tenant_id=uuid.UUID(args.tenant_id) if args.tenant_id else None,
            namespace=args.namespace,
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            google_api_key=os.getenv("GOOGLE_API_KEY") or None,
            **provider_kwargs,
            top_k=args.top_k,
            hops=args.hops,
        )
    except PlaygroundInputError as exc:
        raise SystemExit(str(exc)) from exc

    # Print the (possibly paid, already-obtained) result *before* attempting
    # cleanup: a --cleanup failure must never hide or precede an
    # already-successful result -- it can only ever be reported afterward,
    # as a non-fatal warning, once the result the user paid for is safely
    # printed.
    if args.json:
        print(json.dumps({
            "tenant_id": str(result.scope.tenant_id),
            "namespace": result.scope.namespace,
            "provider": result.scope.provider,
            "mode": result.mode,
            "answer": result.answer,
            "citations": [c.source_id for c in result.citations],
            "grounded": result.grounded,
            "grounding_status": result.grounding_status,
            "graph_path": result.graph_path,
            "usage": result.usage,
            "latency_ms": result.latency_ms,
            "warnings": result.warnings,
        }, indent=2))
    else:
        print(f"Tenant: {result.scope.tenant_id}  Namespace: {result.scope.namespace}")
        print(f"Provider: {result.scope.provider}  Mode: {result.mode}\n")
        print("Answer:")
        print(result.answer)
        if result.grounding_status is not None:
            print(f"\nGrounding: {result.grounding_status}")
        if result.citations:
            print("Citations:", ", ".join(c.source_id for c in result.citations))
        print("\n" + result.retrieval_context)
        if result.graph_path:
            print(f"\nTraversal-selected relationships: {result.graph_path}")
        if result.warnings:
            print("\nWarnings:")
            for warning in result.warnings:
                print(f"- {warning}")

    if args.cleanup:
        try:
            cleanup_extractor = playground_service.build_extractor(
                args.provider,
                openai_api_key=os.getenv("OPENAI_API_KEY") or None,
                google_api_key=os.getenv("GOOGLE_API_KEY") or None,
                **provider_kwargs,
            )
            await playground_service.clear_playground_scope(
                result.scope, runtime_url=runtime_url, extractor=cleanup_extractor,
            )
        except Exception as exc:  # noqa: BLE001 -- must never hide the result already printed above
            print(
                f"\nWarning: --cleanup failed to remove this document ({exc}); it remains at "
                f"tenant={result.scope.tenant_id} namespace={result.scope.namespace!r}.",
                file=sys.stderr,
            )


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
    setup.add_argument(
        "--provider", choices=["offline", "openai", "gemini", "litellm"], default="offline",
        help="Provisions the schema at this provider's embedding dimension "
             "(offline/openai: 1536, gemini: 3072, LiteLLM: configured). Makes no paid API call.",
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
    playground = sub.add_parser(
        "playground",
        help="Try your own document text and question against the real engine "
             "(isolated tenant/namespace, never touches setup/reset).",
    )
    playground.add_argument("question")
    playground.add_argument("--text")
    playground.add_argument("--text-file")
    playground.add_argument("--triplets-json", help='e.g. \'{"triplets": [{"subject":"A","predicate":"depends_on","object":"B"}]}\'')
    playground.add_argument("--triplets-file")
    playground.add_argument(
        "--provider", choices=["offline", "openai", "gemini", "litellm"], default="offline"
    )
    playground.add_argument("--mode", choices=["vector", "hybrid", "hybrid_graph"], default="hybrid_graph")
    # Deliberately default to None (auto-generated), not the top-level
    # --tenant-id/--namespace defaults above -- a playground session must
    # never silently reuse the bundled demo's fixed scope.
    playground.add_argument("--tenant-id", default=None)
    playground.add_argument("--namespace", default=None)
    playground.add_argument("--top-k", type=int, default=5)
    playground.add_argument("--hops", type=int, default=3)
    playground.add_argument("--json", action="store_true")
    playground.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete this invocation's own document after printing the result "
             "(off by default -- each invocation otherwise leaves its isolated, "
             "randomly-scoped tenant/namespace in place indefinitely).",
    )
    args = parser.parse_args()
    commands = {
        "setup": _setup, "ingest": _ingest, "query": _query, "evaluate": _evaluate,
        "reset": _reset, "playground": _playground,
    }
    asyncio.run(commands[args.command](args))


if __name__ == "__main__":
    main()
