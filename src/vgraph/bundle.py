from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import yaml

RESERVED = {"index.md", "log.md"}
_FRONTMATTER = re.compile(r"^---\r?\n(.*?)\r?\n---\r?\n?(.*)\Z", re.S)

log = logging.getLogger(__name__)


@dataclass
class Concept:
    id: str
    relpath: str
    type: str
    title: str | None
    description: str | None
    body: str
    status: str
    verified: list
    generated: dict | None
    stale_after: str | None
    frontmatter: dict = field(default_factory=dict)

    def embed_text(self) -> str:
        parts = [self.title or self.id]
        if self.description:
            parts.append(self.description)
        parts.append(self.body)
        return "\n\n".join(parts)

    def trust_tier(self) -> str:
        return trust_tier(self.verified)

    def is_stale(self, now: datetime | None = None) -> bool:
        return is_stale(self.stale_after, now)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "relpath": self.relpath,
            "type": self.type,
            "title": self.title,
            "description": self.description,
            "body": self.body,
            "status": self.status,
            "verified": self.verified,
            "generated": self.generated,
            "stale_after": self.stale_after,
            "frontmatter": self.frontmatter,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Concept:
        return cls(**d)


@dataclass
class Bundle:
    root: Path
    concepts: dict[str, Concept]
    warnings: list[str]


def trust_tier(verified: list) -> str:
    if not verified:
        return "unverified"
    if any(str(e.get("by", "")).startswith("human:") for e in verified if isinstance(e, dict)):
        return "human-reviewed"
    return "machine-confirmed"


def is_stale(stale_after: str | None, now: datetime | None = None) -> bool:
    if not stale_after:
        return False
    now = now or datetime.now(timezone.utc)
    try:
        t = datetime.fromisoformat(str(stale_after).replace("Z", "+00:00"))
    except ValueError:
        return False
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return now >= t


def normalize_verified(value) -> list:
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return value
    return []


def concept_id(relpath: str) -> str:
    rel = relpath.replace("\\", "/")
    return rel[:-3] if rel.endswith(".md") else rel


def load_bundle(root: Path) -> Bundle:
    root = Path(root)
    concepts: dict[str, Concept] = {}
    warnings: list[str] = []
    for path in sorted(root.rglob("*.md")):
        if path.name in RESERVED:
            continue
        relpath = path.relative_to(root).as_posix()
        concept, warn = _parse_file(path, relpath)
        if warn:
            warnings.append(warn)
            log.warning(warn)
        if concept:
            concepts[concept.id] = concept
    return Bundle(root=root, concepts=concepts, warnings=warnings)


def _parse_file(path: Path, relpath: str) -> tuple[Concept | None, str | None]:
    text = path.read_text(encoding="utf-8")
    m = _FRONTMATTER.match(text)
    if not m:
        return None, f"skip {relpath}: missing YAML frontmatter"
    try:
        fm = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError as e:
        return None, f"skip {relpath}: invalid YAML ({e})"
    if not isinstance(fm, dict) or not fm.get("type"):
        return None, f"skip {relpath}: missing non-empty type"
    cid = concept_id(relpath)
    title = fm.get("title") or PurePosixPath(cid).name
    return Concept(
        id=cid,
        relpath=relpath,
        type=str(fm["type"]),
        title=str(title) if title else None,
        description=fm.get("description"),
        body=m.group(2).strip(),
        status=str(fm.get("status") or "stable"),
        verified=normalize_verified(fm.get("verified")),
        generated=fm.get("generated") if isinstance(fm.get("generated"), dict) else None,
        stale_after=str(fm["stale_after"]) if fm.get("stale_after") else None,
        frontmatter=fm,
    ), None
