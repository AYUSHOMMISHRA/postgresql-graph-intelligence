from types import SimpleNamespace

from postgres_graph_rag.evaluation import build_incident_dataset, score_case


def test_incident_dataset_has_expected_versioned_segments():
    documents, cases = build_incident_dataset()
    assert len(documents) == 60
    assert len(cases) == 120
    counts = {}
    for case in cases:
        counts[case.category] = counts.get(case.category, 0) + 1
    assert counts == {
        "semantic_single_hop": 30,
        "identifier_lexical": 30,
        "multi_hop": 40,
        "negative": 20,
    }


def test_structured_scoring_requires_entities_relations_and_sources():
    _, cases = build_incident_dataset()
    case = next(case for case in cases if case.id == "multihop-owner-001")
    result = SimpleNamespace(
        chunks=[SimpleNamespace(content="auth-service-001 owned by Identity Team 001", source_id="incident-001-ownership")],
        nodes=[SimpleNamespace(content="Identity Team 001")],
        edges=[SimpleNamespace(relation="depends_on"), SimpleNamespace(relation="owned_by")],
        trace=SimpleNamespace(context_tokens=42),
    )
    score = score_case(case, result)
    assert score["recall_at_5"] == 1.0
    assert score["path_match"] is True
    assert score["source_recall"] == 0.5
    assert score["context_tokens"] == 42
