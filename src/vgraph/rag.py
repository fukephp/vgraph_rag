from __future__ import annotations

import functools
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vgraph.bundle import Concept, load_bundle
from vgraph.graph import Edge, adjacency, build_edges

EMBED_MODEL = "all-MiniLM-L6-v2"
DEFAULT_K = 5
PACK_CAP = 4000
DEFAULT_MODEL = "gpt-4o-mini"


@dataclass
class Hit:
    concept: Concept
    role: str
    score: float | None

    @property
    def id(self) -> str:
        return self.concept.id


@dataclass
class LoadedIndex:
    concepts: dict[str, Concept]
    adj: dict[str, set[str]]
    ids: list[str]
    vectors: np.ndarray
    mtimes: dict
    model: str


def bundle_mtimes(root: Path) -> dict[str, float]:
    return {p.relative_to(root).as_posix(): p.stat().st_mtime for p in root.rglob("*.md")}


@functools.lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL)


def embed_texts(texts: list[str]) -> np.ndarray:
    return np.asarray(_embedder().encode(texts, normalize_embeddings=True), dtype=np.float32)


def ingest(bundle_root: Path, index_dir: Path) -> LoadedIndex:
    bundle = load_bundle(bundle_root)
    edges = build_edges(bundle)
    ids = sorted(bundle.concepts)
    vectors = (
        embed_texts([bundle.concepts[i].embed_text() for i in ids])
        if ids
        else np.zeros((0, 1), dtype=np.float32)
    )
    index = LoadedIndex(
        concepts=bundle.concepts,
        adj=adjacency(edges),
        ids=ids,
        vectors=vectors,
        mtimes=bundle_mtimes(bundle_root),
        model=EMBED_MODEL,
    )
    _save(index_dir, index, edges)
    return index


def index_is_stale(bundle_root: Path, index_dir: Path) -> bool:
    meta_path = index_dir / "meta.json"
    if not meta_path.exists() or not (index_dir / "embeddings.npz").exists():
        return True
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("model") != EMBED_MODEL:
        return True
    return meta.get("mtimes") != bundle_mtimes(bundle_root)


def load_or_ingest(bundle_root: Path, index_dir: Path) -> LoadedIndex:
    if index_is_stale(bundle_root, index_dir):
        return ingest(bundle_root, index_dir)
    return _load(index_dir)


def retrieve_index(
    index: LoadedIndex,
    query_vec: np.ndarray,
    k: int = DEFAULT_K,
    *,
    include_deprecated: bool = False,
) -> list[Hit]:
    if index.vectors.size == 0:
        return []
    q = query_vec.astype(np.float32).ravel()
    qn = np.linalg.norm(q) or 1.0
    vn = np.linalg.norm(index.vectors, axis=1, keepdims=True)
    vn[vn == 0] = 1.0
    scores = (index.vectors / vn) @ (q / qn)
    ranked = sorted(range(len(index.ids)), key=lambda i: float(scores[i]), reverse=True)
    seeds: list[Hit] = []
    for i in ranked:
        c = index.concepts[index.ids[i]]
        if c.status == "deprecated" and not include_deprecated:
            continue
        seeds.append(Hit(c, "seed", float(scores[i])))
        if len(seeds) >= k:
            break
    seed_ids = {h.id for h in seeds}
    neighbors: list[Hit] = []
    seen = set(seed_ids)
    for h in seeds:
        for nid in sorted(index.adj.get(h.id, ())):
            if nid in seen or nid not in index.concepts:
                continue
            seen.add(nid)
            neighbors.append(Hit(index.concepts[nid], "neighbor", None))
    return seeds + neighbors


def retrieve(
    query: str,
    bundle_root: Path,
    index_dir: Path,
    k: int = DEFAULT_K,
    *,
    include_deprecated: bool = False,
) -> list[Hit]:
    index = load_or_ingest(bundle_root, index_dir)
    qvec = embed_texts([query])[0]
    return retrieve_index(index, qvec, k, include_deprecated=include_deprecated)


def pack_hits(hits: list[Hit], cap: int = PACK_CAP) -> str:
    blocks = []
    for h in hits:
        c = h.concept
        header = (
            f"## {c.id} ({h.role}) type={c.type} status={c.status} "
            f"trust={c.trust_tier()} stale={str(c.is_stale()).lower()}"
        )
        body = c.embed_text()
        if len(body) > cap:
            body = body[: cap - 3] + "..."
        blocks.append(f"{header}\n\n{body}")
    return "\n\n".join(blocks)


def generate(query: str, hits: list[Hit], client=None) -> str:
    if not hits:
        return "No matching concepts."
    if client is None:
        client = _client()
    content = pack_hits(hits)
    resp = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL),
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer using only the packed concepts. Cite concept ids. "
                    "Mention stale or unverified when relevant. "
                    "If they do not answer the question, say so."
                ),
            },
            {"role": "user", "content": f"Question: {query}\n\n{content}"},
        ],
    )
    return resp.choices[0].message.content or ""


def ask(
    query: str,
    bundle_root: Path,
    index_dir: Path,
    k: int = DEFAULT_K,
    *,
    include_deprecated: bool = False,
    client=None,
) -> str:
    hits = retrieve(
        query,
        bundle_root,
        index_dir,
        k,
        include_deprecated=include_deprecated,
    )
    return generate(query, hits, client=client)


def _client():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is unset")
    from openai import OpenAI

    kwargs = {}
    if os.environ.get("OPENAI_BASE_URL"):
        kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]
    return OpenAI(**kwargs)


def _save(index_dir: Path, index: LoadedIndex, edges: list[Edge]) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    graph = {
        "concepts": {cid: c.to_dict() for cid, c in index.concepts.items()},
        "edges": [{"source": e.source, "target": e.target, "origin": e.origin} for e in edges],
    }
    (index_dir / "graph.json").write_text(json.dumps(graph), encoding="utf-8")
    np.savez(index_dir / "embeddings.npz", ids=np.array(index.ids, dtype=object), vectors=index.vectors)
    (index_dir / "meta.json").write_text(
        json.dumps({"model": index.model, "mtimes": index.mtimes}),
        encoding="utf-8",
    )


def _load(index_dir: Path) -> LoadedIndex:
    graph = json.loads((index_dir / "graph.json").read_text(encoding="utf-8"))
    concepts = {cid: Concept.from_dict(d) for cid, d in graph["concepts"].items()}
    edges = [Edge(e["source"], e["target"], e["origin"]) for e in graph["edges"]]
    data = np.load(index_dir / "embeddings.npz", allow_pickle=True)
    ids = [str(x) for x in data["ids"].tolist()]
    meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
    return LoadedIndex(
        concepts=concepts,
        adj=adjacency(edges),
        ids=ids,
        vectors=data["vectors"],
        mtimes=meta.get("mtimes", {}),
        model=meta.get("model", EMBED_MODEL),
    )
