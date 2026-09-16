from pathlib import Path

import numpy as np

from vgraph.bundle import is_stale, load_bundle, trust_tier
from vgraph.graph import build_edges
from vgraph.rag import LoadedIndex, generate, retrieve_index

FIXTURE = Path(__file__).parent / "fixtures" / "mini"


def test_load_bundle_skips_reserved_and_malformed():
    b = load_bundle(FIXTURE)
    assert set(b.concepts) == {"alpha", "beta", "nested/gamma"}
    assert any("broken.md" in w for w in b.warnings)
    assert any("notype.md" in w for w in b.warnings)


def test_verified_mapping_and_trust_tiers():
    b = load_bundle(FIXTURE)
    assert b.concepts["alpha"].trust_tier() == "human-reviewed"
    assert len(b.concepts["alpha"].verified) == 1
    assert b.concepts["beta"].trust_tier() == "unverified"
    assert b.concepts["nested/gamma"].trust_tier() == "machine-confirmed"
    assert trust_tier([]) == "unverified"


def test_stale_and_status_defaults():
    b = load_bundle(FIXTURE)
    assert b.concepts["alpha"].status == "stable"
    assert not b.concepts["alpha"].is_stale()
    assert b.concepts["nested/gamma"].is_stale()
    assert is_stale(None) is False


def test_graph_links_siblings_not_ancestors():
    b = load_bundle(FIXTURE)
    edges = build_edges(b)
    kinds = {(e.source, e.target, e.origin) for e in edges}
    assert ("alpha", "beta", "link") in kinds
    assert ("alpha", "beta", "sibling") in kinds
    assert ("beta", "alpha", "sibling") in kinds
    assert ("nested/gamma", "alpha", "link") in kinds
    assert ("nested/gamma", "beta", "link") in kinds
    assert ("alpha", "nested/gamma", "sibling") not in kinds
    assert ("nested/gamma", "alpha", "sibling") not in kinds
    assert not any(e.target == "nope" for e in edges)


def _index(vectors: dict[str, list[float]]) -> LoadedIndex:
    b = load_bundle(FIXTURE)
    ids = list(vectors)
    adj: dict[str, set[str]] = {cid: set() for cid in b.concepts}
    for e in build_edges(b):
        adj[e.source].add(e.target)
        adj[e.target].add(e.source)
    return LoadedIndex(
        concepts=b.concepts,
        adj=adj,
        ids=ids,
        vectors=np.array([vectors[i] for i in ids], dtype=np.float32),
        mtimes={},
        model="test",
    )


def test_deprecated_is_neighbor_not_seed():
    index = _index({"alpha": [1.0, 0.0], "beta": [0.0, 1.0], "nested/gamma": [0.0, 0.5]})
    hits = retrieve_index(index, np.array([1.0, 0.0], dtype=np.float32), k=1)
    assert [h.id for h in hits if h.role == "seed"] == ["alpha"]
    assert "beta" in [h.id for h in hits if h.role == "neighbor"]


def test_only_deprecated_match_yields_no_hits():
    b = load_bundle(FIXTURE)
    index = LoadedIndex(
        concepts={"beta": b.concepts["beta"]},
        adj={"beta": set()},
        ids=["beta"],
        vectors=np.array([[1.0, 0.0]], dtype=np.float32),
        mtimes={},
        model="test",
    )
    assert retrieve_index(index, np.array([1.0, 0.0], dtype=np.float32), k=5) == []


def test_generate_empty_hits_skips_llm():
    class Boom:
        def __getattr__(self, name):
            raise AssertionError("LLM should not be called")

    assert generate("anything", [], client=Boom()) == "No matching concepts."


def test_include_deprecated_can_seed():
    index = _index({"alpha": [1.0, 0.0], "beta": [0.0, 1.0], "nested/gamma": [0.5, 0.0]})
    hits = retrieve_index(
        index, np.array([0.0, 1.0], dtype=np.float32), k=1, include_deprecated=True
    )
    assert [h.id for h in hits if h.role == "seed"] == ["beta"]
