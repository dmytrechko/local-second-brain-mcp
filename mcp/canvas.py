"""Obsidian-compatible output generators: entity hub notes and reminders markdown."""

import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import db


def _slugify(name: str) -> str:
    return re.sub(r"[\s/\\]+", "-", name.strip().lower())


def generate_entity_hubs(db_path: str, memory_dir: str) -> None:
    """Generate one hub note per entity in memory/entities/.

    Each hub links back to every memory that mentions the entity,
    making them appear as graph nodes in Obsidian's native graph view.
    Stale hub files (entities no longer in DB) are removed.
    """
    import db as db_module

    entities_dir = Path(memory_dir) / "entities"
    entities_dir.mkdir(exist_ok=True)

    conn = db_module.get_connection(db_path)
    rows = conn.execute("""
        SELECT e.name, e.type, m.filepath, m.title
        FROM entities e
        JOIN memories m ON m.id = e.memory_id
        ORDER BY e.name, m.filepath
    """).fetchall()
    conn.close()

    # Group by entity: {(name, type): [(filepath, title), ...]}
    entity_map: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for name, etype, filepath, title in rows:
        key = (name, etype)
        entity_map.setdefault(key, []).append((filepath, title))

    written: set[str] = set()

    for (name, etype), memories in entity_map.items():
        slug = _slugify(name)
        filename = f"{slug}.md"
        written.add(filename)

        # Build wikilinks back to source notes (strip .md for cleaner links)
        links = "\n".join(
            f"- [[{fp.replace('.md', '')}|{title}]]"
            for fp, title in memories
        )

        content = f"""---
tags:
  - entity
  - {etype}
generated: true
---

# {name}

*Entity type: {etype}*

## Appears in

{links}
"""
        hub_path = entities_dir / filename
        hub_path.write_text(content, encoding="utf-8")

    # Remove stale hub files for entities no longer in DB
    removed = 0
    for existing in entities_dir.glob("*.md"):
        if existing.name not in written:
            existing.unlink()
            removed += 1

    print(f"Entity hubs written → {entities_dir}  ({len(written)} entities, {removed} removed)")


def generate_tasks_md(db_path: str, output_path: str) -> None:
    """Generate a flat markdown task list for Obsidian visibility."""
    all_tasks = db.get_tasks(db_path, limit=500)

    # Group by status then project
    by_status: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for t in all_tasks:
        by_status[t["status"]][t["project"] or "General"].append(t)

    lines = ["# Tasks\n", "Auto-generated from task notes.\n"]

    status_order = ["active", "waiting", "backlog", "done"]
    checkbox = {"done": "x"}

    total_active = len(by_status.get("active", {}))
    if not any(by_status.values()):
        lines.append("*No tasks found.*")
    else:
        for status in status_order:
            if status not in by_status:
                continue
            projects = by_status[status]
            count = sum(len(v) for v in projects.values())
            lines.append(f"\n## {status.capitalize()} ({count})\n")
            for project, tasks in sorted(projects.items()):
                lines.append(f"\n### {project}\n")
                mark = checkbox.get(status, " ")
                for t in tasks:
                    lines.append(f"- [{mark}] {t['text']}")
            lines.append("")

    lines.append("\n---")
    lines.append(f"\n*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    active_count = sum(len(v) for v in by_status.get("active", {}).values())
    print(f"Tasks written → {output_path}  ({active_count} active)")


def generate_reminders_md(db_path: str, output_path: str) -> None:
    """Generate a markdown file with all pending reminders for Obsidian visibility."""
    reminders = db.get_reminders(db_path, status="pending", limit=100)

    lines = ["# Reminders\n", "Auto-generated from memory temporal commitments.\n"]

    # Group by due status
    today = datetime.now().strftime('%Y-%m-%d')
    overdue = [r for r in reminders if r["due_date"] < today]
    upcoming = [r for r in reminders if r["due_date"] >= today]

    if overdue:
        lines.append(f"## ⚠️ Overdue ({len(overdue)})\n")
        for r in sorted(overdue, key=lambda x: x["due_date"]):
            lines.append(f"- [ ] **{r['due_date']}** — {r['text']}")
            lines.append(f"  → [[{r['filepath'].replace('.md', '')}]]")
            lines.append("")

    if upcoming:
        lines.append(f"## 📅 Upcoming ({len(upcoming)})\n")
        for r in sorted(upcoming, key=lambda x: x["due_date"]):
            lines.append(f"- [ ] **{r['due_date']}** — {r['text']}")
            lines.append(f"  → [[{r['filepath'].replace('.md', '')}]]")
            lines.append("")

    if not reminders:
        lines.append("*No pending reminders found.*")

    lines.append("\n---")
    lines.append(f"\n*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"Reminders written → {output_path}  ({len(reminders)} pending)")


