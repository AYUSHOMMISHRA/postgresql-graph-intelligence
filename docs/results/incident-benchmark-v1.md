# Incident benchmark v1 — measured result

Run date: 2026-08-15. Environment: Darwin arm64, Python 3.14.4, local
PostgreSQL/pgvector container. Corpus: 60 deterministic synthetic incident
documents and 120 labeled questions. Provider calls are replaced by the
versioned offline extraction fixture, so these numbers evaluate retrieval and
grounding behavior—not real-model extraction quality.

| Mode | Recall@5 | Multi-hop Recall@5 | Multi-hop path accuracy | False connections | p50 | p95 | Mean context tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Vector | 70.8% | 12.5% | 0.0% | 0.0% | 4.6 ms | 9.5 ms | 148 |
| Hybrid | 70.8% | 12.5% | 0.0% | 0.0% | 5.5 ms | 6.7 ms | 148 |
| Hybrid + graph | 100.0% | 100.0% | 100.0% | 0.0% | 50.2 ms | 83.5 ms | 258 |

## Gate result

- Multi-hop improvement: **pass**, +87.5 percentage points versus vector.
- Overall Recall@5 regression: **pass**, +29.2 points.
- Negative false-connection rate ≤10%: **pass**, measured 0%.
- Warm p95 <300 ms on this fixture: **pass**, measured 83.5 ms including
  batch-loading the source chunks that assert retained graph edges.
- Citation validation: **pass** in unit tests; unknown citation IDs abstain.

## What changed because of the benchmark

The first run failed: graph paths were present but Recall@5 stayed at 12.5%.
The trace exposed two causes. Fuzzy resolution merged numbered service IDs, and
graph expansion rooted itself in the highest prose-ranked chunk even when a
second retrieved chunk contained the exact query identifier. The implementation
now treats digit-bearing identifiers as exact identities and uses exact query
anchors to select graph roots.

## Limitations

The templates are intentionally regular, the corpus is small, and the same
fixture defines extraction and labels. This is a regression benchmark, not
proof of customer accuracy or production scale. A 20–30 document anonymized
real-provider canary and the 1K/10K/50K scale benchmark remain required before
making production quality or cost claims.
