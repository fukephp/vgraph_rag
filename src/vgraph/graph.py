from __future__ import annotations

import itertools
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import PurePosixPath

from vgraph.bundle import Bundle, RESERVED

_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    origin: str


def build_edges(bundle: Bundle) -> list[Edge]:
    ids = set(bundle.concepts)
    edges: list[Edge] = []
    seen: set[tuple[str, str, str]] = set()

    def add(src: str, tgt: str, origin: str) -> None:
        key = (src, tgt, origin)
        if src == tgt or tgt not in ids or key in seen:
            return
        seen.add(key)
        edges.append(Edge(src, tgt, origin))

    for cid, concept in bundle.concepts.items():
        for url in _LINK.findall(concept.body):
            tgt = resolve_link(cid, url, ids)
            if tgt:
                add(cid, tgt, "link")

    by_dir: dict[str, list[str]] = defaultdict(list)
    for cid in bundle.concepts:
        by_dir[str(PurePosixPath(cid).parent)].append(cid)
    for group in by_dir.values():
        for a, b in itertools.permutations(group, 2):
            add(a, b, "sibling")

    return edges


def resolve_link(src_id: str, url: str, concept_ids: set[str]) -> str | None:
    raw = url.strip().split()[0].strip("<>")
    if not raw or raw.startswith(("#", "http://", "https://", "mailto:", "ftp:")):
        return None
    path = raw.split("#", 1)[0]
    if not path:
        return None
    if path.startswith("/"):
        rel = PurePosixPath(path[1:])
    else:
        src_dir = PurePosixPath(src_id).parent
        rel = (src_dir / path)
    parts: list[str] = []
    for p in rel.parts:
        if p == "..":
            if parts:
                parts.pop()
        elif p not in (".",):
            parts.append(p)
    if not parts or parts[-1] in RESERVED:
        return None
    name = "/".join(parts)
    cid = name[:-3] if name.endswith(".md") else name
    return cid if cid in concept_ids else None


def adjacency(edges: list[Edge]) -> dict[str, set[str]]:
    adj: dict[str, set[str]] = defaultdict(set)
    for e in edges:
        adj[e.source].add(e.target)
        adj[e.target].add(e.source)
    return adj
