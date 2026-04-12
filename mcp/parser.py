"""Markdown parser: extracts frontmatter tags, title, and body."""

import re
from pathlib import Path
from datetime import datetime, timezone

import frontmatter



def parse_memory_file(filepath: Path) -> dict:
    """
    Parse a markdown file and return a dict with:
      - title: str (first H1 or filename stem)
      - content: str (full text including frontmatter stripped)
      - tags: list[str]
      - updated_at: str (ISO8601, file mtime)
    """
    post = frontmatter.load(str(filepath))

    # Tags from frontmatter
    raw_tags = post.get("tags", [])
    if isinstance(raw_tags, str):
        raw_tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
    tags = [str(t).lower().strip() for t in raw_tags]

    # Title: first H1 in body, else filename stem
    body = post.content
    title = _extract_title(body) or _slugify_to_title(filepath.stem)

    # updated_at from file mtime
    mtime = filepath.stat().st_mtime
    updated_at = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()

    # Content for embedding: title + tags + body
    embed_text = f"{title}\n\nTags: {', '.join(tags)}\n\n{body}"

    return {
        "title": title,
        "content": body,
        "embed_text": embed_text,
        "tags": tags,
        "updated_at": updated_at,
    }


def _extract_title(body: str) -> str | None:
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return None


def _slugify_to_title(stem: str) -> str:
    return stem.replace("-", " ").replace("_", " ").title()


_EXCLUDED_DIRS = {"entities"}
_EXCLUDED_NAMES = {"README.MD", "REMINDERS.MD"}

def split_sections(content: str) -> list[dict]:
    """Split markdown content into sections by H2 headings.

    Returns list of {heading, body, content_hash} dicts.
    The preamble before the first H2 is heading="".
    """
    import hashlib
    h2_re = re.compile(r"^## .+$", re.MULTILINE)
    boundaries = [m.start() for m in h2_re.finditer(content)] + [len(content)]

    sections = []
    prev = 0
    heading = ""
    for boundary in boundaries:
        body = content[prev:boundary].strip()
        if body or heading == "":
            h = hashlib.md5(body.encode()).hexdigest()[:16]
            sections.append({"heading": heading, "body": body, "content_hash": h})
        # Extract heading text for the next section
        if boundary < len(content):
            line_end = content.find("\n", boundary)
            heading = content[boundary:line_end if line_end != -1 else len(content)].lstrip("#").strip()
        prev = boundary

    return sections


def find_all_markdown_files(memory_dir: Path) -> list[Path]:
    return sorted(
        p for p in memory_dir.rglob("*.md")
        if p.name.upper() not in _EXCLUDED_NAMES
        and not any(part in _EXCLUDED_DIRS for part in p.parts)
    )


def relative_filepath(memory_dir: Path, filepath: Path) -> str:
    """Return path relative to memory_dir as a string key."""
    return str(filepath.relative_to(memory_dir))
