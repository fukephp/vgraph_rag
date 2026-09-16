from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from vgraph.rag import ask, ingest, retrieve


def _load_dotenv(path: Path = Path(".env")) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main(argv: list[str] | None = None) -> None:
    _load_dotenv()
    p = argparse.ArgumentParser(prog="vgraph")
    p.add_argument("--bundle", type=Path, default=Path("knowledge"))
    p.add_argument("--index", type=Path, default=Path(".vgraph"))
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ingest", help="rebuild the derived index")

    pr = sub.add_parser("retrieve", help="print ranked hits as JSON")
    pr.add_argument("query")
    pr.add_argument("-k", type=int, default=5)
    pr.add_argument("--include-deprecated", action="store_true")

    pa = sub.add_parser("ask", help="retrieve and generate an answer")
    pa.add_argument("query")
    pa.add_argument("-k", type=int, default=5)
    pa.add_argument("--include-deprecated", action="store_true")

    args = p.parse_args(argv)
    if args.cmd == "ingest":
        index = ingest(args.bundle, args.index)
        print(f"ingested {len(index.concepts)} concepts -> {args.index}")
        return
    if args.cmd == "retrieve":
        hits = retrieve(
            args.query,
            args.bundle,
            args.index,
            args.k,
            include_deprecated=args.include_deprecated,
        )
        print(json.dumps([_hit_json(h) for h in hits], indent=2))
        return
    print(
        ask(
            args.query,
            args.bundle,
            args.index,
            args.k,
            include_deprecated=args.include_deprecated,
        )
    )


def _hit_json(h) -> dict:
    c = h.concept
    return {
        "id": c.id,
        "role": h.role,
        "score": h.score,
        "type": c.type,
        "title": c.title,
        "status": c.status,
        "trust": c.trust_tier(),
        "stale": c.is_stale(),
    }


if __name__ == "__main__":
    main()
