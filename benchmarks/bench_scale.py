"""Performance benchmark for the DB layer at increasing graph sizes.

Populates synthetic graphs (no LLM calls — embeddings and edges are
generated locally) at 1K / 10K node scale and measures:
  - bulk node upsert throughput
  - vector_search latency (p50/p95)
  - traverse_graph latency (p50/p95) at 2 hops
  - EXPLAIN ANALYZE for one representative query at each scale

Usage:
    POSTGRES_URL=postgresql://... python benchmarks/bench_scale.py
"""
import asyncio
import argparse
import json
import os
import random
import statistics
import time

from dotenv import load_dotenv

from postgres_graph_rag.database import DatabaseManager

load_dotenv()

POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://postgres:postgres@localhost:5432/graph_rag")
DIM = 128
NODE_BATCH = 1000
RELATIONS = ["depends_on", "uses", "mentions", "works_at", "part_of"]
RNG = random.Random(1729)


def random_embedding() -> list:
    v = [RNG.gauss(0, 1) for _ in range(DIM)]
    norm = sum(x * x for x in v) ** 0.5
    return [x / norm for x in v]


def percentile(values, pct):
    values = sorted(values)
    idx = min(len(values) - 1, int(len(values) * pct))
    return values[idx]


async def populate(db: DatabaseManager, namespace: str, n_nodes: int, avg_degree: int = 3):
    node_ids = []
    for start in range(0, n_nodes, NODE_BATCH):
        batch = [
            {"content": f"Entity-{namespace}-{i}", "embedding": random_embedding()}
            for i in range(start, min(start + NODE_BATCH, n_nodes))
        ]
        ids = await db.upsert_nodes_batch(batch, namespace=namespace)
        node_ids.extend(ids)

    edges = []
    for i, source in enumerate(node_ids):
        for _ in range(avg_degree):
            target = node_ids[RNG.randrange(len(node_ids))]
            if target == source:
                continue
            edges.append(
                {
                    "source_id": source,
                    "target_id": target,
                    "relation": RNG.choice(RELATIONS),
                    "weight": round(RNG.uniform(0.3, 1.0), 2),
                }
            )
    for start in range(0, len(edges), NODE_BATCH):
        await db.upsert_edges_batch(edges[start : start + NODE_BATCH], namespace=namespace)

    return node_ids


async def bench_scale(db: DatabaseManager, n_nodes: int, iterations: int = 20):
    namespace = f"bench-seed-1729-{n_nodes}"

    # Re-running must measure the same graph rather than incrementing edge
    # weights from an earlier run in the same exact benchmark namespace.
    await db._init_pool()
    async with db.pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("DELETE FROM graph_edges WHERE namespace=%s", (namespace,))
            await cur.execute("DELETE FROM graph_nodes WHERE namespace=%s", (namespace,))
        await conn.commit()

    t0 = time.perf_counter()
    await populate(db, namespace, n_nodes)
    populate_s = time.perf_counter() - t0
    print(f"\n=== Scale: {n_nodes} nodes ===")
    print(f"Populate (nodes + ~{3 * n_nodes} edges): {populate_s:.2f}s "
          f"({n_nodes / populate_s:.0f} nodes/s)")

    vector_latencies = []
    traverse_latencies = []
    for _ in range(iterations):
        q = random_embedding()

        t0 = time.perf_counter()
        seeds = await db.vector_search(q, namespace=namespace, top_k=5)
        vector_latencies.append(time.perf_counter() - t0)

        seed_ids = [s["id"] for s in seeds]
        t0 = time.perf_counter()
        await db.traverse_graph(seed_ids, namespace=namespace, max_hops=2)
        traverse_latencies.append(time.perf_counter() - t0)

    def report(name, latencies):
        print(
            f"{name:<20} p50={percentile(latencies,0.5)*1000:7.2f}ms  "
            f"p95={percentile(latencies,0.95)*1000:7.2f}ms  "
            f"avg={statistics.mean(latencies)*1000:7.2f}ms"
        )

    report("vector_search", vector_latencies)
    report("traverse_graph(2hop)", traverse_latencies)
    print(
        "machine_report",
        json.dumps({
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

    # EXPLAIN ANALYZE for one representative query
    await db._init_pool()
    async with db.pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) SELECT id, content FROM graph_nodes "
                "WHERE namespace = %s ORDER BY (embedding <=> %s::vector) LIMIT 5",
                (namespace, random_embedding()),
            )
            plan = await cur.fetchall()
            explain = plan[0][list(plan[0].keys())[0]] if isinstance(plan[0], dict) else plan[0][0]
            root = explain[0]["Plan"]
            print(f"\nEXPLAIN summary @ {n_nodes}: node={root['Node Type']} "
                  f"actual_ms={root['Actual Total Time']:.3f} rows={root['Actual Rows']}")


async def run(scales):
    db = DatabaseManager(POSTGRES_URL)
    await db.setup_database(embedding_dimension=DIM)
    for n in scales:
        await bench_scale(db, n)
    await db.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scales", nargs="+", type=int, default=[1_000, 10_000, 50_000])
    args = parser.parse_args()
    asyncio.run(run(args.scales))


if __name__ == "__main__":
    main()
