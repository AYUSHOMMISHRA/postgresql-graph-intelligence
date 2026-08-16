"""Performance benchmark for the RLS-secured storage layer (SecureGraphStore)
at increasing graph sizes.

This is the secured-path replacement for the deleted `bench_scale.py`,
which measured the legacy single-tenant `DatabaseManager` engine and never
reflected RLS policy-evaluation or transaction-local tenant-context
overhead (see docs/decisions/003-remove-legacy-engine.md's addendum, and
the demo-readiness review's P2-4 finding). Every query here runs through
`SecureGraphStore.tenant_connection()`, the same transaction-scoped
`set_config()` + RLS-policy path a real request takes -- there is no
faster, RLS-bypassing route through this code.

Populates synthetic graphs (no LLM calls -- embeddings and edges are
generated locally) at 1K / 10K node scale and measures:
  - bulk node upsert throughput (non-fuzzy: resolution matching is a
    separate, already-benchmarked concern; this isolates raw write path)
  - vector_search_nodes latency (p50/p95/p99)
  - traverse_graph latency (p50/p95/p99) at 2 hops
  - EXPLAIN ANALYZE for one representative query at each scale, run inside
    a real tenant_connection() so the plan reflects the RLS predicate

Usage:
    POSTGRES_URL=postgresql://... python -m benchmarks.bench_scale_secure \
        --admin-url "$POSTGRES_URL" [--scales 1000,10000]
"""
import argparse
import asyncio
import json
import os
import random
import statistics
import time
import uuid

from dotenv import load_dotenv

from postgres_graph_rag.tenancy import SecureGraphStore, migrate_schema

load_dotenv()

DEFAULT_DIM = 128
DIM = DEFAULT_DIM  # overwritten by main() from --dim; module-level so random_embedding() sees it
NODE_BATCH = 1000
RELATIONS = ["depends_on", "uses", "mentions", "works_at", "part_of"]
RNG = random.Random(1729)
RUNTIME_ROLE = "pgr_bench_runtime"
RUNTIME_PASSWORD = "pgr_bench_runtime_pw"  # noqa: S105 -- benchmark-only, throwaway schema


def random_embedding() -> list:
    v = [RNG.gauss(0, 1) for _ in range(DIM)]
    norm = sum(x * x for x in v) ** 0.5
    return [x / norm for x in v]


def percentile(values, pct):
    values = sorted(values)
    idx = min(len(values) - 1, int(len(values) * pct))
    return values[idx]


def _runtime_url(admin_url: str) -> str:
    import urllib.parse as up

    parsed = up.urlparse(admin_url)
    netloc = f"{RUNTIME_ROLE}:{RUNTIME_PASSWORD}@{parsed.hostname}:{parsed.port or 5432}"
    return up.urlunparse(parsed._replace(netloc=netloc))


async def populate(store: SecureGraphStore, tenant_id: uuid.UUID, namespace: str, n_nodes: int, avg_degree: int = 3):
    node_ids = []
    for start in range(0, n_nodes, NODE_BATCH):
        batch = [
            {"content": f"Entity-{namespace}-{i}", "embedding": random_embedding()}
            for i in range(start, min(start + NODE_BATCH, n_nodes))
        ]
        # fuzzy=False: entity resolution (trigram + embedding confirmation)
        # is a separate, already-tested concern (see
        # tests/test_secure_path_characterization.py's fuzzy-resolution
        # tests); this benchmark isolates raw upsert throughput, matching
        # the deleted bench_scale.py's scope.
        resolved = await store.resolve_and_upsert_nodes(tenant_id, namespace, batch, fuzzy=False)
        node_ids.extend(resolved.values())

    edges = []
    for source in node_ids:
        for _ in range(avg_degree):
            target = node_ids[RNG.randrange(len(node_ids))]
            if target == source:
                continue
            edges.append({
                "source_id": source, "target_id": target,
                "relation": RNG.choice(RELATIONS), "weight": round(RNG.uniform(0.3, 1.0), 2),
            })
    for start in range(0, len(edges), NODE_BATCH):
        await store.upsert_edges(tenant_id, namespace, edges[start:start + NODE_BATCH])

    return node_ids


async def bench_scale(store: SecureGraphStore, tenant_id: uuid.UUID, n_nodes: int, iterations: int = 20):
    namespace = f"bench-seed-1729-{n_nodes}"

    t0 = time.perf_counter()
    await populate(store, tenant_id, namespace, n_nodes)
    populate_s = time.perf_counter() - t0
    print(f"\n=== Scale: {n_nodes} nodes (secure/RLS path) ===")
    print(f"Populate (nodes + ~{3 * n_nodes} edges): {populate_s:.2f}s "
          f"({n_nodes / populate_s:.0f} nodes/s)")

    vector_latencies = []
    traverse_latencies = []
    for _ in range(iterations):
        q = random_embedding()

        t0 = time.perf_counter()
        seeds = await store.vector_search_nodes(tenant_id, namespace, q, top_k=5)
        vector_latencies.append(time.perf_counter() - t0)

        seed_ids = [s["id"] for s in seeds]
        t0 = time.perf_counter()
        await store.traverse_graph(tenant_id, seed_ids, namespace=namespace, max_hops=2)
        traverse_latencies.append(time.perf_counter() - t0)

    def report(name, latencies):
        print(
            f"{name:<20} p50={percentile(latencies, 0.5) * 1000:7.2f}ms  "
            f"p95={percentile(latencies, 0.95) * 1000:7.2f}ms  "
            f"avg={statistics.mean(latencies) * 1000:7.2f}ms"
        )

    report("vector_search_nodes", vector_latencies)
    report("traverse_graph(2hop)", traverse_latencies)
    print(
        "machine_report",
        json.dumps({
            "engine": "SecureGraphStore (RLS-enforced)",
            "seed": 1729,
            "scale": n_nodes,
            "iterations": iterations,
            "populate_nodes_per_s": n_nodes / populate_s,
            "vector_ms": {
                "p50": percentile(vector_latencies, 0.5) * 1000,
                "p95": percentile(vector_latencies, 0.95) * 1000,
                "p99": percentile(vector_latencies, 0.99) * 1000,
            },
            "traversal_ms": {
                "p50": percentile(traverse_latencies, 0.5) * 1000,
                "p95": percentile(traverse_latencies, 0.95) * 1000,
                "p99": percentile(traverse_latencies, 0.99) * 1000,
            },
        }, sort_keys=True),
    )

    # EXPLAIN ANALYZE for one representative query, run inside a real
    # tenant_connection() -- the same transaction-scoped set_config() +
    # FORCE ROW LEVEL SECURITY predicate every production query pays for,
    # unlike the deleted legacy benchmark's plain unscoped connection.
    async with store.tenant_connection(tenant_id) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) SELECT id, content FROM postgres_graph_rag.graph_nodes "
                "WHERE namespace = %s ORDER BY (embedding <=> %s::vector) LIMIT 5",
                (namespace, random_embedding()),
            )
            plan = await cur.fetchall()
            explain = plan[0][list(plan[0].keys())[0]] if isinstance(plan[0], dict) else plan[0][0]
            root = explain[0]["Plan"]
            print(f"\nEXPLAIN summary @ {n_nodes} (RLS-scoped): node={root['Node Type']} "
                  f"actual_ms={root['Actual Total Time']:.3f} rows={root['Actual Rows']}")


async def run(admin_url: str, scales: list) -> None:
    await migrate_schema(
        admin_url=admin_url, runtime_role=RUNTIME_ROLE, runtime_password=RUNTIME_PASSWORD,
        embedding_dimension=DIM, migrate_legacy_data=False,
    )
    store = SecureGraphStore(_runtime_url(admin_url), vector_type="vector")
    tenant_id = uuid.uuid4()
    try:
        for n in scales:
            await bench_scale(store, tenant_id, n)
    finally:
        await store.close()


def main() -> None:
    global DIM

    parser = argparse.ArgumentParser(prog="bench-scale-secure")
    parser.add_argument("--admin-url", default=os.getenv("POSTGRES_URL"))
    parser.add_argument("--scales", default="1000,10000", help="Comma-separated node counts")
    parser.add_argument(
        "--dim", type=int, default=DEFAULT_DIM,
        help="Embedding dimension. Must match the target schema's existing dimension if one is "
             "already migrated (docs/operations.md recommends an isolated database/schema per run).",
    )
    args = parser.parse_args()
    if not args.admin_url:
        parser.error("--admin-url or POSTGRES_URL is required (admin/superuser connection)")
    DIM = args.dim
    scales = [int(s) for s in args.scales.split(",")]
    asyncio.run(run(args.admin_url, scales))


if __name__ == "__main__":
    main()
